"""pytest coverage for the /inference stub.

The headline test is `test_predict_runs_end_to_end_on_cpu`: it proves
`InferenceRuntime.predict` runs start to finish with the dummy policy behind it,
on any CPU, without OpenVINO installed and without an exported IR. Run with:

    python -m pytest inference/
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from inference.benchmark import (
    load_benchmark_spec,
    load_instruction,
    measure_combo,
    summarize,
)
from inference.runtime import (
    InferenceRuntime,
    RuntimeConfig,
    extract_camera_image,
    openvino_available_devices,
    resolve_device,
)

INFERENCE_CONFIG = "configs/inference.yaml"
EVAL_CONFIG = "configs/eval.yaml"

# dummy.controls_per_arm is 6 in configs/policy.yaml -> 12 actuated controls.
EXPECTED_ACTION_DIM = 12


def _observation(runtime: InferenceRuntime) -> dict:
    return {"overhead": np.zeros(runtime.image_shape, dtype=np.uint8)}


def test_predict_runs_end_to_end_on_cpu():
    runtime = InferenceRuntime(INFERENCE_CONFIG)
    action = runtime.predict(
        "Open the top drawer and set the plate on the table.",
        _observation(runtime),
        {"joint_positions": [0.0] * EXPECTED_ACTION_DIM},
    )
    assert isinstance(action, np.ndarray)
    assert action.shape == (EXPECTED_ACTION_DIM,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()


def test_predict_accepts_bare_image_and_no_robot_state():
    runtime = InferenceRuntime(INFERENCE_CONFIG)
    action = runtime.predict("pour water", np.zeros((16, 24, 3), dtype=np.uint8))
    assert action.shape == (EXPECTED_ACTION_DIM,)


def test_predict_accepts_eval_style_images_mapping():
    runtime = InferenceRuntime(INFERENCE_CONFIG)
    observation = {
        "images": {
            "overhead": np.zeros(runtime.image_shape, dtype=np.uint8),
            "wrist_left": np.zeros(runtime.image_shape, dtype=np.uint8),
        },
        "robot_state": np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32),
    }
    action = runtime.predict("hand off the plate", observation, observation["robot_state"])
    assert action.shape == (EXPECTED_ACTION_DIM,)


def test_action_chunking_via_override():
    runtime = InferenceRuntime(INFERENCE_CONFIG, overrides={"action_chunk_size": 4})
    action = runtime.predict("open drawer", np.zeros((16, 16, 3), dtype=np.uint8))
    assert action.shape == (4, EXPECTED_ACTION_DIM)


def test_predict_rejects_empty_instruction():
    runtime = InferenceRuntime(INFERENCE_CONFIG)
    with pytest.raises(ValueError):
        runtime.predict("   ", np.zeros((8, 8, 3), dtype=np.uint8))


def test_predict_rejects_missing_observation():
    runtime = InferenceRuntime(INFERENCE_CONFIG)
    with pytest.raises(ValueError):
        runtime.predict("open drawer", None)


def test_runtime_info_reports_cpu_fallback_without_openvino():
    info = InferenceRuntime(INFERENCE_CONFIG).runtime_info()
    assert info["backend"] == "_DummyBackend"
    assert info["performance_hint"] == "LATENCY"
    if not info["openvino_installed"]:
        # No OpenVINO on this box: the requested device is used unverified,
        # which for the default CPU config is still CPU.
        assert info["device_resolved"] == info["device_requested"]
    # runtime_info must be JSON-serializable (it goes into the benchmark report).
    json.dumps(info)


# --- pure device-selection logic (no OpenVINO needed) ----------------------
def test_resolve_device_exact_and_family_match():
    assert resolve_device("CPU", ["CPU", "GPU"]) == "CPU"
    assert resolve_device("GPU", ["CPU", "GPU.0", "GPU.1"]) == "GPU"


def test_resolve_device_falls_back_to_cpu_when_absent():
    assert resolve_device("NPU", ["CPU"]) == "CPU"
    assert resolve_device("GPU", ["CPU"]) == "CPU"


def test_resolve_device_passes_auto_through():
    assert resolve_device("AUTO", ["CPU"]) == "AUTO"
    assert resolve_device("auto", None) == "AUTO"


def test_resolve_device_unverified_when_openvino_absent():
    assert resolve_device("NPU", None) == "NPU"


# --- config validation ---------------------------------------------------
def test_runtime_config_rejects_unknown_device():
    with pytest.raises(ValueError):
        RuntimeConfig.from_config(INFERENCE_CONFIG, overrides={"device": "TPU"})


def test_runtime_config_rejects_non_positive_chunk():
    with pytest.raises(ValueError):
        RuntimeConfig.from_config(INFERENCE_CONFIG, overrides={"action_chunk_size": 0})


def test_runtime_config_rejects_unknown_override_key():
    with pytest.raises(ValueError):
        RuntimeConfig.from_config(INFERENCE_CONFIG, overrides={"nonsense": 1})


def test_extract_camera_image_needs_an_array():
    with pytest.raises(ValueError):
        extract_camera_image({"robot_state": [1, 2, 3]})
    with pytest.raises(TypeError):
        extract_camera_image("not an observation")


# --- benchmark ---------------------------------------------------------
def test_load_benchmark_spec_matches_schema():
    spec = load_benchmark_spec(EVAL_CONFIG)
    assert spec["warmup_iterations"] >= 0
    assert spec["measured_iterations"] >= 1
    assert spec["devices"] and spec["precisions"]
    assert "latency_ms_p50" in spec["metrics"]
    assert load_instruction(EVAL_CONFIG).strip()


def test_measure_combo_returns_one_sample_per_iteration():
    latencies_ms, resolved_device = measure_combo(
        inference_config=INFERENCE_CONFIG,
        device="NPU",
        precision="INT8",
        instruction="open drawer",
        warmup_iterations=1,
        measured_iterations=5,
    )
    assert len(latencies_ms) == 5
    assert all(sample >= 0.0 for sample in latencies_ms)
    available = openvino_available_devices()
    if available is not None and "NPU" not in {d.split(".", 1)[0] for d in available}:
        # OpenVINO present but no NPU on this box -> CPU fallback.
        assert resolved_device == "CPU"
    elif available is None:
        # OpenVINO absent -> request used unverified.
        assert resolved_device == "NPU"


def test_summarize_emits_all_schema_metrics():
    entry = summarize([1.0, 2.0, 3.0, 4.0], "CPU", "FP16")
    assert set(entry) == {
        "latency_ms_p50",
        "latency_ms_p95",
        "throughput_fps",
        "device",
        "precision",
    }
    assert entry["latency_ms_p50"] == 2.5
