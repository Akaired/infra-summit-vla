"""Run one SmolVLA prediction with synthetic observations."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    with config_path.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)["experiment"]

    model_path = Path(settings["local_model_dir"])
    if not model_path.is_absolute():
        model_path = repo_root / model_path

    for filename in (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        if not (model_path / filename).is_file():
            parser.error(f"Missing checkpoint file: {model_path / filename}")

    smoke = settings["smoke_test"]
    instruction = smoke["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        parser.error("experiment.smoke_test.instruction must be non-empty")

    torch.manual_seed(smoke["seed"])
    device = settings["device"]

    print(f"Loading SmolVLA on {device}...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(
        str(model_path),
        device=device,
    ).to(device).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )

    # Shapes come from the checkpoint, not hardcoded camera dimensions.
    # Zero-valued float images represent black images in the [0, 1] range.
    observation = {
        name: torch.zeros(tuple(feature.shape), dtype=torch.float32)
        for name, feature in policy.config.input_features.items()
    }
    observation["task"] = instruction

    policy.reset()
    print("Running one prediction...", flush=True)

    with torch.inference_mode():
        batch = preprocess(observation)
        raw_action = policy.select_action(batch)
        action = postprocess(raw_action)

    if not isinstance(action, torch.Tensor):
        raise TypeError(f"Expected a tensor, received {type(action).__name__}")

    expected_shape = (1, *policy.config.output_features["action"].shape)
    if tuple(action.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected action shape: {tuple(action.shape)}; "
            f"expected {expected_shape}"
        )

    if not torch.isfinite(action).all().item():
        raise RuntimeError("Model produced NaN or infinite action values")

    print(json.dumps({
        "status": "ok",
        "instruction": instruction,
        "action_shape": list(action.shape),
        "action": action.detach().cpu().tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()