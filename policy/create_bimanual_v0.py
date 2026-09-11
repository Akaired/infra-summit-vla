"""Create the config-driven bimanual-v0 SmolVLA technical baseline."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import lerobot
import torch
import transformers
import yaml
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_settings(config_path: str) -> dict:
    path = resolve_path(config_path)
    with path.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    if not isinstance(settings, dict):
        raise ValueError("Expected a YAML mapping.")

    return settings


def make_feature(feature_type: FeatureType, shape: list[int]) -> PolicyFeature:
    if not shape or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError(f"Invalid feature shape: {shape}")

    return PolicyFeature(type=feature_type, shape=tuple(shape))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    settings = load_settings(args.config)

    base_settings = settings["base"]
    output_settings = settings["output"]
    runtime_settings = settings["runtime"]
    input_settings = settings["inputs"]
    action_settings = settings["action"]
    normalization = settings["normalization"]

    base_dir = resolve_path(base_settings["checkpoint_dir"])
    output_dir = resolve_path(output_settings["checkpoint_dir"])
    device = runtime_settings["device"]

    if not (base_dir / "config.json").is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_dir}")

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Refusing to overwrite a checkpoint."
        )

    state_settings = input_settings["state"]
    state_key = state_settings["feature_key"]
    state_shape = state_settings["shape"]

    action_key = action_settings["feature_key"]
    action_shape = action_settings["shape"]
    action_names = action_settings["names"]

    if len(action_shape) != 1:
        raise ValueError("action.shape must contain exactly one dimension.")

    if action_shape[0] != len(action_names):
        raise ValueError(
            "action.shape must match the number of action.names entries."
        )

    input_features = {
        state_key: make_feature(FeatureType.STATE, state_shape),
    }

    for camera in input_settings["cameras"]:
        input_features[camera["feature_key"]] = make_feature(
            FeatureType.VISUAL,
            camera["shape"],
        )

    output_features = {
        action_key: make_feature(FeatureType.ACTION, action_shape),
    }

    config = SmolVLAConfig.from_pretrained(str(base_dir))
    config.device = device
    config.input_features = input_features
    config.output_features = output_features

    policy = SmolVLAPolicy.from_pretrained(
        str(base_dir),
        config=config,
        strict=True,
    ).to(device)

    # These layers are tied to the robot state/action interface.
    # The VLM and action-expert backbone remain pretrained.
    policy.model.state_proj.reset_parameters()
    policy.model.action_in_proj.reset_parameters()
    policy.model.action_out_proj.reset_parameters()
    policy.eval()

    output_dir.mkdir(parents=True)
    policy.save_pretrained(output_dir)

    state_dim = state_shape[0]
    action_dim = action_shape[0]

    neutral_stats = {
        state_key: {
            "mean": torch.full(
                (state_dim,),
                float(normalization["state"]["mean"]),
            ),
            "std": torch.full(
                (state_dim,),
                float(normalization["state"]["std"]),
            ),
        },
        action_key: {
            "mean": torch.full(
                (action_dim,),
                float(normalization["action"]["mean"]),
            ),
            "std": torch.full(
                (action_dim,),
                float(normalization["action"]["std"]),
            ),
        },
    }

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=neutral_stats,
    )

    preprocessor.save_pretrained(
        output_dir,
        config_filename=output_settings["preprocessor_file"],
    )
    postprocessor.save_pretrained(
        output_dir,
        config_filename=output_settings["postprocessor_file"],
    )

    manifest = {
        "name": "bimanual-v0",
        "purpose": "technical baseline for inference, export, and benchmark",
        "base_model": "lerobot/smolvla_base",
        "base_revision": base_settings["revision"],
        "lerobot_version": getattr(lerobot, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "input_features": {
            key: list(feature.shape)
            for key, feature in input_features.items()
        },
        "output_features": {
            key: list(feature.shape)
            for key, feature in output_features.items()
        },
        "action_type": action_settings["semantics"],
        "action_units": action_settings["units"],
        "action_order": action_names,
        "normalization": normalization,
    }

    manifest_path = output_dir / output_settings["manifest_file"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    print(f"\nCreated checkpoint: {output_dir}")


if __name__ == "__main__":
    main()