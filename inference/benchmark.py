"""Standalone OpenVINO inference benchmark -- challenge deliverable #3.

Measures ONLY the policy inference path. It never imports or steps MuJoCo.

* Benchmark schema comes from ``configs/eval.yaml:benchmark``
  (``warmup_iterations``, ``measured_iterations``, ``devices``, ``precisions``,
  ``report_path``, ``metrics``).
* Runtime settings come from ``configs/inference.yaml`` (model paths, hint,
  cache dir), overridden per (device, precision) pair in the sweep.
* Warmup iterations are discarded. For every measured iteration one latency
  sample is taken; the report gives real p50 and p95 percentiles plus
  throughput -- never a bare mean, which hides the latency tail.
* The report is written to ``benchmark.report_path``.

While ``/policy`` has not exported an OpenVINO IR the runtime delegates to the
dummy policy, so this script still produces a well-formed report today -- the
numbers are the stub's and ``stub_run`` in the report says so.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from inference.runtime import (
    REPO_ROOT,
    InferenceRuntime,
    _resolve_path,
    openvino_available_devices,
)

logger = logging.getLogger("inference.benchmark")

REQUIRED_BENCHMARK_KEYS = (
    "warmup_iterations",
    "measured_iterations",
    "devices",
    "precisions",
    "report_path",
    "metrics",
)


def load_benchmark_spec(eval_config_path: str | Path) -> dict:
    """Read and validate ``configs/eval.yaml:benchmark``."""
    path = _resolve_path(eval_config_path)
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("benchmark"), Mapping):
        raise ValueError(f"{path}: no 'benchmark' mapping")
    spec = dict(raw["benchmark"])

    missing = [key for key in REQUIRED_BENCHMARK_KEYS if key not in spec]
    if missing:
        raise ValueError(f"benchmark config missing keys: {missing}")
    for key in ("warmup_iterations", "measured_iterations"):
        if type(spec[key]) is not int or spec[key] < 0:
            raise ValueError(f"benchmark.{key} must be a non-negative integer")
    if spec["measured_iterations"] < 1:
        raise ValueError("benchmark.measured_iterations must be >= 1")
    for key in ("devices", "precisions", "metrics"):
        if not isinstance(spec[key], list) or not spec[key]:
            raise ValueError(f"benchmark.{key} must be a non-empty list")
    if not isinstance(spec["report_path"], str) or not spec["report_path"].strip():
        raise ValueError("benchmark.report_path must be a non-empty string")
    return spec


def load_instruction(eval_config_path: str | Path) -> str:
    """The natural-language instruction fed to every benchmark iteration."""
    path = _resolve_path(eval_config_path)
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    episode = raw.get("episode") if isinstance(raw, Mapping) else None
    instruction = episode.get("instruction") if isinstance(episode, Mapping) else None
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{path}: episode.instruction is required for the benchmark")
    return instruction


def _synthetic_observation(runtime: InferenceRuntime) -> dict:
    """One synthetic RGB frame at the model's expected size.

    The camera key is a local placeholder, not configuration -- the benchmark
    isolates the inference path, so only the tensor shape matters.
    """
    return {"camera": np.zeros(runtime.image_shape, dtype=np.uint8)}


def measure_combo(
    *,
    inference_config: str | Path,
    device: str,
    precision: str,
    instruction: str,
    warmup_iterations: int,
    measured_iterations: int,
) -> tuple[list[float], str]:
    """Run warmup + measured iterations for one (device, precision) pair.

    Returns ``(latencies_ms, resolved_device)``. ``resolved_device`` is what the
    runtime actually ran on -- it differs from ``device`` when the requested
    accelerator is absent and the runtime falls back to CPU.
    """
    runtime = InferenceRuntime(
        inference_config, overrides={"device": device, "precision": precision}
    )
    resolved_device = runtime.runtime_info()["device_resolved"]
    observation = _synthetic_observation(runtime)
    robot_state = {"joint_positions": [0.0] * runtime.action_dim}

    for _ in range(warmup_iterations):
        runtime.predict(instruction, observation, robot_state)

    latencies_ms: list[float] = []
    for _ in range(measured_iterations):
        start = time.perf_counter()
        runtime.predict(instruction, observation, robot_state)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return latencies_ms, resolved_device


def summarize(latencies_ms: list[float], device: str, precision: str) -> dict:
    """Percentile latency + throughput for one combo. All schema metrics."""
    samples = np.asarray(latencies_ms, dtype=np.float64)
    total_seconds = float(samples.sum()) / 1000.0
    return {
        "latency_ms_p50": round(float(np.percentile(samples, 50)), 4),
        "latency_ms_p95": round(float(np.percentile(samples, 95)), 4),
        "throughput_fps": round(len(samples) / total_seconds, 4) if total_seconds > 0 else None,
        "device": device,
        "precision": precision,
    }


def run(inference_config: str | Path, eval_config: str | Path) -> dict:
    """Execute the full sweep and write the report. Returns the report dict."""
    spec = load_benchmark_spec(eval_config)
    instruction = load_instruction(eval_config)
    available_devices = openvino_available_devices()

    probe_info = InferenceRuntime(inference_config).runtime_info()
    stub_run = probe_info["backend"] != "_IRBackend"

    results: list[dict] = []
    for device in spec["devices"]:
        for precision in spec["precisions"]:
            device_upper = str(device).upper()
            precision_upper = str(precision).upper()
            latencies_ms, resolved_device = measure_combo(
                inference_config=inference_config,
                device=device_upper,
                precision=precision_upper,
                instruction=instruction,
                warmup_iterations=spec["warmup_iterations"],
                measured_iterations=spec["measured_iterations"],
            )
            full = summarize(latencies_ms, resolved_device, precision_upper)
            # Emit exactly the metrics the schema asks for, in that order.
            entry = {name: full[name] for name in spec["metrics"] if name in full}
            if resolved_device != device_upper:
                entry["device_requested"] = device_upper
            results.append(entry)
            logger.info("%s / %s -> %s", device_upper, precision_upper, entry)

    report = {
        "generated_by": "inference/benchmark.py",
        "stub_run": stub_run,
        "note": (
            "No OpenVINO IR exported yet; measurements are the dummy-policy inference "
            "path. Re-run after /policy exports configs/inference.yaml:model.ir_xml."
            if stub_run
            else ""
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "openvino_installed": available_devices is not None,
            "openvino_available_devices": available_devices,
        },
        "warmup_iterations": spec["warmup_iterations"],
        "measured_iterations": spec["measured_iterations"],
        "instruction": instruction,
        "metrics": spec["metrics"],
        "results": results,
    }

    report_path = _resolve_path(spec["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s (%d rows)", report_path, len(results))
    report["report_path"] = str(report_path)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-config",
        required=True,
        help="inference YAML (model paths, hint, cache dir), relative to repo root",
    )
    parser.add_argument(
        "--eval-config",
        required=True,
        help="eval YAML carrying the 'benchmark' block, relative to repo root",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()
    report = run(args.inference_config, args.eval_config)
    print(
        json.dumps(
            {"report_path": report["report_path"], "results": report["results"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
