"""Export a deterministic fixed-shape SmolVLA velocity step to OpenVINO IR.

SmolVLA's normal sampler passes a Transformers ``DynamicCache`` between its
prefix pass and every denoising step. That Python object cannot be an
OpenVINO input. The bridge leaves prefix preparation in PyTorch, exports the
pure tensor velocity step, and runs the ten Euler updates in Python. This
keeps native behavior intact and avoids tracing the large VLM ten times.

This module imports optional ML dependencies only inside its entry points so
the normal project test suite remains usable without the policy extra.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExportManifest:
    """Tensor ABI shared by the exporter and :mod:`inference.smolvla`."""

    format_version: int
    checkpoint: str
    vlm_model_path: str | None
    image_feature_keys: list[str]
    state_feature_key: str
    action_feature_key: str
    chunk_size: int
    max_action_dim: int
    num_steps: int
    inputs: list[str]
    output: str


def _require_dependencies():
    try:
        import openvino as ov
        import torch
        from lerobot.policies.common.vla_utils import make_att_2d_masks
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "SmolVLA export requires the repository's existing policy and "
            "inference extras: install `.[policy,inference]`."
        ) from exc
    return ov, torch, make_att_2d_masks


def make_velocity_step(flow_model):
    """Return the single tensor-only flow velocity step for OpenVINO export."""
    _, torch, make_att_2d_masks = _require_dependencies()

    class SmolVLAVelocityStep(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.num_steps = int(model.config.num_steps)
            self.chunk_size = int(model.config.chunk_size)

        def forward(self, prefix, prefix_pad, prefix_att, x_t, time):
            suffix, suffix_pad, suffix_att = self.model.embed_suffix(x_t, time)
            pad = torch.cat((prefix_pad, suffix_pad), dim=1)
            att = torch.cat((prefix_att, suffix_att), dim=1)
            mask_2d = make_att_2d_masks(pad, att)
            positions = torch.cumsum(pad, dim=1) - 1
            (_, suffix_out), _ = self.model.vlm_with_expert.forward(
                attention_mask=mask_2d,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
            )
            suffix_out = suffix_out[:, -self.chunk_size :].to(dtype=torch.float32)
            return self.model.action_out_proj(suffix_out)

    return SmolVLAVelocityStep(flow_model).eval()


def prepare_prefix(flow_model, images, image_masks, language_tokens, language_mask, state):
    """Run vision/language prefix preparation in PyTorch outside the exported IR."""
    return flow_model.embed_prefix(
        list(images.unbind(0)), list(image_masks.bool().unbind(0)), language_tokens, language_mask.bool(), state
    )


def integrate_velocity(step, prefix, prefix_pad, prefix_att, noise, num_steps):
    """Reference Euler loop; OpenVINO runs the same loop in the parity harness."""
    _, torch, _ = _require_dependencies()
    x_t = noise
    for index in range(num_steps):
        time = torch.full((x_t.shape[0],), 1.0 - index / num_steps, dtype=torch.float32)
        x_t = x_t - step(prefix, prefix_pad, prefix_att, x_t, time) / num_steps
    return x_t


@contextmanager
def _force_eager_transformers_attention():
    """Temporarily select eager attention while LeRobot builds SmolVLM.

    OpenVINO 2025.3 traces the PyTorch module. Transformers 5.5's SDPA mask
    helper receives a scalar symbolic query length under that trace, whereas
    the eager implementation keeps the mask construction traceable.
    """
    from transformers import AutoModelForImageTextToText

    original = AutoModelForImageTextToText.from_pretrained

    def eager_from_pretrained(*args, **kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        return original(*args, **kwargs)

    AutoModelForImageTextToText.from_pretrained = eager_from_pretrained
    try:
        yield
    finally:
        AutoModelForImageTextToText.from_pretrained = original


@contextmanager
def _export_only_smolvlm_vision_mask():
    """Bypass Transformers 5.5's scalar SDPA-mask trace bug for vision only.

    The export examples have no padded image patches, so ``None`` is exactly
    the eager attention representation of full bidirectional attention.  This
    patches the symbol imported by SmolVLM's vision module, not Transformers'
    global mask utilities, and only while OpenVINO traces the wrapper.
    """
    from transformers.models.smolvlm import modeling_smolvlm

    original = modeling_smolvlm.create_bidirectional_mask

    def full_bidirectional_mask(*args, **kwargs):
        return None

    modeling_smolvlm.create_bidirectional_mask = full_bidirectional_mask
    try:
        yield
    finally:
        modeling_smolvlm.create_bidirectional_mask = original


def _load_policy(checkpoint: Path, vlm_model_path: Path | None = None):
    _, _, _ = _require_dependencies()
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = SmolVLAConfig.from_pretrained(str(checkpoint))
    config.device = "cpu"
    if vlm_model_path is not None:
        if not (vlm_model_path / "config.json").is_file():
            raise FileNotFoundError(f"SmolVLM2 config not found: {vlm_model_path}")
        config.vlm_model_name = str(vlm_model_path)
    with _force_eager_transformers_attention():
        return SmolVLAPolicy.from_pretrained(str(checkpoint), config=config).to("cpu").eval()


def _example_inputs(policy, checkpoint: Path, instruction: str, seed: int):
    _, torch, _ = _require_dependencies()
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    preprocess, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    observation = {
        name: torch.zeros(tuple(feature.shape), dtype=torch.float32)
        for name, feature in policy.config.input_features.items()
    }
    observation["task"] = instruction
    batch = preprocess(observation)
    images, image_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (state.shape[0], policy.config.chunk_size, policy.config.max_action_dim), generator=generator
    )
    return (
        torch.stack(images),
        torch.stack(image_masks),
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        state,
        noise,
    )


def export_openvino(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    instruction: str,
    seed: int = 0,
    precision: str = "FP16",
    vlm_model_path: str | Path | None = None,
) -> ExportManifest:
    """Convert one fixed-shape velocity step and write its IR and manifest.

    INT8 is intentionally rejected: the repository does not allow NNCF, which
    is the OpenVINO-supported post-training quantization dependency.  No silent
    fake INT8 export is produced.
    """
    ov, torch, _ = _require_dependencies()
    precision = precision.upper()
    if precision not in {"FP32", "FP16"}:
        raise ValueError("INT8 export requires NNCF, which is not an allowed repository dependency")
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    vlm_path = Path(vlm_model_path).resolve() if vlm_model_path else None
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"SmolVLA checkpoint config not found: {checkpoint}")
    policy = _load_policy(checkpoint, vlm_path)
    if policy.config.rtc_config is not None and policy.config.rtc_config.enabled:
        raise ValueError("RTC is not supported by the OpenVINO tensor bridge")
    example = _example_inputs(policy, checkpoint, instruction, seed)
    prefix, prefix_pad, prefix_att = prepare_prefix(policy.model, *example[:-1])
    step = make_velocity_step(policy.model)
    with torch.inference_mode():
        reference = integrate_velocity(step, prefix, prefix_pad, prefix_att, example[-1], policy.config.num_steps)
    ov_model = ov.convert_model(step, example_input=(prefix, prefix_pad, prefix_att, example[-1], torch.ones(1)))
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / "velocity_step.xml"
    ov.save_model(ov_model, xml_path, compress_to_fp16=precision == "FP16")
    manifest = ExportManifest(
        format_version=1,
        checkpoint=str(checkpoint),
        vlm_model_path=str(vlm_path) if vlm_path else None,
        image_feature_keys=list(policy.config.image_features),
        state_feature_key="observation.state",
        action_feature_key="action",
        chunk_size=int(policy.config.chunk_size),
        max_action_dim=int(policy.config.max_action_dim),
        num_steps=int(policy.config.num_steps),
        inputs=["prefix", "prefix_pad", "prefix_att", "x_t", "timestep"],
        output="velocity",
    )
    (output_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    # Store a deterministic reference for the parity command and CI artifact checks.
    torch.save({"prefix": (prefix, prefix_pad, prefix_att), "noise": example[-1], "output": reference, "native_inputs": example}, output_dir / "parity_reference.pt")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--instruction", default="open drawer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", default="FP16", choices=["FP32", "FP16"])
    parser.add_argument("--vlm-model-path", help="local SmolVLM2 backbone directory; avoids Hub download")
    args = parser.parse_args()
    print(json.dumps(asdict(export_openvino(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
