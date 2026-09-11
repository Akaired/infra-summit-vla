"""Smoke-test bimanual-v0 using the config-defined 12D contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from lerobot.policies.factory import make_pre_post_processors
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    settings = load_settings(args.config)

    output_settings = settings["output"]
    runtime_settings = settings["runtime"]
    input_settings = settings["inputs"]
    action_settings = settings["action"]
    smoke_settings = settings["smoke_test"]

    checkpoint_dir = resolve_path(output_settings["checkpoint_dir"])
    device = runtime_settings["device"]

    torch.manual_seed(smoke_settings["seed"])

    policy = SmolVLAPolicy.from_pretrained(
        str(checkpoint_dir),
        strict=True,
    ).to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint_dir),
        preprocessor_config_filename=output_settings["preprocessor_file"],
        postprocessor_config_filename=output_settings["postprocessor_file"],
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )

    state_settings = input_settings["state"]
    observation = {
        state_settings["feature_key"]: torch.zeros(
            tuple(state_settings["shape"]),
            dtype=torch.float32,
        ),
        "task": input_settings["instruction"],
    }

    for camera in input_settings["cameras"]:
        observation[camera["feature_key"]] = torch.zeros(
            tuple(camera["shape"]),
            dtype=torch.float32,
        )

    policy.reset()

    with torch.inference_mode():
        batch = preprocessor(observation)
        action = postprocessor(policy.select_action(batch))

    expected_action_shape = (1, *action_settings["shape"])

    if tuple(action.shape) != expected_action_shape:
        raise RuntimeError(
            f"Expected action shape {expected_action_shape}, "
            f"received {tuple(action.shape)}"
        )

    if not torch.isfinite(action).all():
        raise RuntimeError("Action contains NaN or infinity.")

    print(
        json.dumps(
            {
                "status": "ok",
                "input_state_shape": [1, *state_settings["shape"]],
                "input_cameras": [
                    camera["name"]
                    for camera in input_settings["cameras"]
                ],
                "action_shape": list(action.shape),
                "action": action.cpu().tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()