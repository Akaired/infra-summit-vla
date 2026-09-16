"""Run one deterministic SmolVLA PyTorch/OpenVINO bridge demonstration.

This is deliberately a direct OpenVINO call, rather than the application's
backend selector: demo mode has no PyTorch fallback.  A missing IR, unavailable
CPU plugin, or non-CPU compiled execution device is therefore a hard failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def _error(left: np.ndarray, right: np.ndarray, *, rtol: float, atol: float) -> dict:
    delta = np.abs(left - right)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "allclose": bool(np.allclose(left, right, rtol=rtol, atol=atol)),
    }


def run_demo(export_dir: str | Path, *, rtol: float = 3e-2, atol: float = 3e-2) -> dict:
    """Execute the saved fixed-noise sample through PyTorch and CPU OpenVINO."""
    import openvino as ov
    import torch

    export_dir = Path(export_dir)
    xml_path = export_dir / "velocity_step.xml"
    reference_path = export_dir / "parity_reference.pt"
    manifest_path = export_dir / "manifest.json"
    if not xml_path.is_file() or not reference_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("demo requires velocity_step.xml, manifest.json, and parity_reference.pt")

    core = ov.Core()
    if "CPU" not in core.available_devices:
        raise RuntimeError(f"OpenVINO CPU plugin is unavailable (devices: {core.available_devices})")
    compiled = core.compile_model(core.read_model(xml_path), "CPU")
    execution_devices = list(compiled.get_property("EXECUTION_DEVICES"))
    if not execution_devices or any(not str(name).startswith("CPU") for name in execution_devices):
        raise RuntimeError(f"demo requires CPU OpenVINO execution, got {execution_devices}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    prefix, prefix_pad, prefix_att = tuple(reference["prefix"])
    noise = reference["noise"]

    from policy.openvino_export import _load_policy, make_velocity_step

    policy = _load_policy(
        Path(manifest["checkpoint"]),
        Path(manifest["vlm_model_path"]) if manifest.get("vlm_model_path") else None,
    )
    step = make_velocity_step(policy.model)
    num_steps = int(manifest["num_steps"])

    def run_openvino(x_t, timestep) -> np.ndarray:
        values = (prefix, prefix_pad, prefix_att, x_t, timestep)
        result = compiled({compiled.input(i): value.detach().cpu().numpy() for i, value in enumerate(values)})
        return next(iter(result.values()))

    per_step = []
    x_torch = noise.clone()
    x_openvino = noise.clone()
    pytorch_seconds = 0.0
    openvino_seconds = 0.0
    with torch.inference_mode():
        for index in range(num_steps):
            timestep = torch.full((x_torch.shape[0],), 1.0 - index / num_steps, dtype=torch.float32)
            start = perf_counter()
            torch_velocity = step(prefix, prefix_pad, prefix_att, x_torch, timestep)
            pytorch_seconds += perf_counter() - start
            ov_velocity = run_openvino(x_torch, timestep)
            metrics = _error(ov_velocity, torch_velocity.detach().cpu().numpy(), rtol=rtol, atol=atol)
            per_step.append({"step": index + 1, **metrics})
            x_torch = x_torch - torch_velocity / num_steps

            # Propagate the OpenVINO trajectory separately for final parity.
            start = perf_counter()
            ov_trajectory_velocity = run_openvino(x_openvino, timestep)
            openvino_seconds += perf_counter() - start
            x_openvino = x_openvino - torch.from_numpy(ov_trajectory_velocity) / num_steps

    final_metrics = _error(x_openvino.numpy(), x_torch.numpy(), rtol=rtol, atol=atol)
    report = {
        "openvino_execution_devices": execution_devices,
        "per_step": per_step,
        "steps_passed": sum(item["allclose"] for item in per_step),
        "final": final_metrics,
        "output_shape": list(x_openvino.shape),
        "timing_seconds": {
            "pytorch_10_steps": pytorch_seconds,
            "openvino_10_steps": openvino_seconds,
            "pytorch_per_step": pytorch_seconds / num_steps,
            "openvino_per_step": openvino_seconds / num_steps,
        },
    }
    if report["steps_passed"] != num_steps or not final_metrics["allclose"]:
        raise AssertionError(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--atol", type=float, default=3e-2)
    args = parser.parse_args()
    report = run_demo(**vars(args))
    print("OpenVINO CPU IR: " + ", ".join(report["openvino_execution_devices"]))
    for item in report["per_step"]:
        print(f"step {item['step']:2d}/10: {'PASS' if item['allclose'] else 'FAIL'} "
              f"max_abs={item['max_abs']:.6f} mean_abs={item['mean_abs']:.6f}")
    print(f"per-step parity: {report['steps_passed']}/10 passed")
    print("final parity: " + json.dumps(report["final"]))
    print(f"output shape: {report['output_shape']}")
    print("timing (seconds): " + json.dumps(report["timing_seconds"]))


if __name__ == "__main__":
    main()
