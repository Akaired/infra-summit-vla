"""Check exported OpenVINO SmolVLA actions against PyTorch with fixed noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def check_parity(export_dir: str | Path, *, device: str = "CPU", rtol: float = 3e-2, atol: float = 3e-2) -> dict:
    """Compare cache-free PyTorch, OpenVINO, and native cached SmolVLA output."""
    try:
        import openvino as ov
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("parity checking requires the repository's `.[policy,inference]` extras") from exc
    export_dir = Path(export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    reference = torch.load(export_dir / "parity_reference.pt", map_location="cpu", weights_only=True)
    prefix, prefix_pad, prefix_att = tuple(reference["prefix"])
    noise = reference["noise"]
    torch_reference = reference["output"].detach().cpu().numpy()
    native_inputs = tuple(reference["native_inputs"])

    core = ov.Core()
    compiled = core.compile_model(core.read_model(export_dir / "velocity_step.xml"), device)

    from policy.openvino_export import _load_policy

    policy = _load_policy(
        Path(manifest["checkpoint"]),
        Path(manifest["vlm_model_path"]) if manifest.get("vlm_model_path") else None,
    )
    from policy.openvino_export import make_velocity_step

    step = make_velocity_step(policy.model)

    def invoke_openvino(x_t, timestep):
        values = (prefix, prefix_pad, prefix_att, x_t, timestep)
        return next(iter(compiled({compiled.input(i): value.detach().cpu().numpy() for i, value in enumerate(values)}).values()))

    def error(left, right):
        delta = np.abs(left - right)
        return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()), "allclose": bool(np.allclose(left, right, rtol=rtol, atol=atol))}

    per_step = []
    x_torch = noise.clone()
    x_openvino = noise.clone()
    with torch.inference_mode():
        for index in range(policy.config.num_steps):
            timestep = torch.full((x_torch.shape[0],), 1.0 - index / policy.config.num_steps, dtype=torch.float32)
            torch_velocity = step(prefix, prefix_pad, prefix_att, x_torch, timestep)
            ov_velocity = invoke_openvino(x_torch, timestep)
            per_step.append({"step": index, **error(ov_velocity, torch_velocity.detach().cpu().numpy())})
            x_torch = x_torch - torch_velocity / policy.config.num_steps
            ov_trajectory_velocity = invoke_openvino(x_openvino, timestep)
            x_openvino = x_openvino - torch.from_numpy(ov_trajectory_velocity) / policy.config.num_steps
        native = policy.model.sample_actions(
            list(native_inputs[0].unbind(0)),
            list(native_inputs[1].bool().unbind(0)),
            *native_inputs[2:5],
            noise=native_inputs[5],
        ).detach().cpu().numpy()

    openvino_final = x_openvino.detach().cpu().numpy()
    report = {
        "device": device,
        "rtol": rtol,
        "atol": atol,
        "per_step_openvino_vs_pytorch": per_step,
        "openvino_final_vs_no_cache_pytorch": error(openvino_final, torch_reference),
        # The exported flow tensor always has ``max_action_dim`` columns (32
        # for SmolVLA), whereas native ``sample_actions`` returns only the
        # checkpoint's public action dimensions (12 for bimanual-v0).
        "native_cached_vs_no_cache_pytorch": error(
            native, torch_reference[:, :, : native.shape[-1]]
        ),
        "shape": list(openvino_final.shape),
    }
    if not all(item["allclose"] for item in per_step) or not report["openvino_final_vs_no_cache_pytorch"]["allclose"] or not report[
        "native_cached_vs_no_cache_pytorch"
    ]["allclose"]:
        raise AssertionError(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--atol", type=float, default=3e-2)
    args = parser.parse_args()
    print(json.dumps(check_parity(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
