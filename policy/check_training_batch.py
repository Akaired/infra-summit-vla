"""Verify one real dataset batch can produce a finite SmolVLA loss."""

import argparse
from email import policy
import json

import torch
import yaml

from data.dataloader import PolicyDataset, build_dataloader, resolve_path
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import (
    make_smolvla_pre_post_processors,
)
from typing import Any


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with resolve_path(args.config).open(encoding="utf-8") as file:
        experiment = yaml.safe_load(file)["experiment"]

    dataset_settings = experiment["dataset"]
    device = experiment["device"]
    torch.manual_seed(dataset_settings["loader"]["seed"])

    print("Loading a real dataset batch...", flush=True)
    loader = build_dataloader(args.config)
    batch = next(iter(loader))

    # The wrapper exposes the underlying LeRobot dataset and its statistics.
    dataset = loader.dataset
    if not isinstance(dataset, PolicyDataset):
        raise ValueError(
            f"Expected a PolicyDataset, received {type(dataset)}"
        )
    source = dataset.source 
    camera_map = dataset_settings["camera_map"]

    # Apply the same camera renaming to normalization statistics.
    stats = {
        camera_map.get(key, key): values
        for key, values in source.meta.stats.items()
    }

    for key in ("observation.state", "action"):
        if key not in stats:
            raise ValueError(f"Dataset statistics are missing: {key}")

    model_path = resolve_path(experiment["local_model_dir"])

    print(f"Loading SmolVLA on {device}...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(
        str(model_path),
        device=device,
    ).to(device)

    # Ensure newly constructed processors use the requested device.
    policy.config.device = device

    expected_cameras = set(policy.config.image_features)
    available_cameras = set(camera_map.values())

    unknown_cameras = available_cameras - expected_cameras
    if unknown_cameras:
        raise ValueError(f"Unexpected camera mappings: {unknown_cameras}")

    missing_cameras = expected_cameras - available_cameras

    # Derived from the model/data contracts rather than a fixed camera count.
    # SmolVLA creates masked placeholders for these missing views.
    policy.config.empty_cameras = len(missing_cameras)

    print("Available cameras:", sorted(available_cameras))
    print("Masked missing cameras:", sorted(missing_cameras))

    # Build new processors with this dataset's statistics.
    # Do not reuse the base checkpoint's state/action normalization statistics.
    preprocess, _ = make_smolvla_pre_post_processors(
        config=policy.config,
        dataset_stats=stats,
    )

    print("Tokenizing and normalizing the batch...", flush=True)
    processed = preprocess(batch)

    output_features = policy.config.output_features
    if output_features is None or "action" not in output_features:
        raise ValueError(
            f"Expected 'action' in output_features, received {output_features}"
        )
    expected_action_shape = (
        batch["action"].shape[0],
        policy.config.chunk_size,
        output_features["action"].shape[0],
    )

    if tuple(processed["action"].shape) != expected_action_shape:
        raise ValueError(
            f"Action shape {tuple(processed['action'].shape)} "
            f"does not match {expected_action_shape}"
        )

    for name, value in processed.items():
        if (
            isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and not torch.isfinite(value).all().item()
        ):
            raise ValueError(f"Non-finite values after preprocessing: {name}")

    padding = processed["action_is_pad"]
    if padding.all().item():
        raise ValueError("The batch contains no valid target actions")

    print("Computing one training loss...", flush=True)
    policy.train()

    # This checks the training forward pass without storing gradients
    # or updating any weights, keeping CPU memory use lower.
    with torch.no_grad():
        result: Any = policy.forward(processed)
        
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            f"Expected a tuple of (loss, outputs), received {type(result)}"
        )

    loss, outputs = result
    
    if not isinstance(loss, torch.Tensor):
        raise ValueError(f"Expected a torch.Tensor loss, received {type(loss)}")

    if loss.numel() != 1 or not torch.isfinite(loss).all().item():
        raise RuntimeError(f"Expected a finite scalar loss, received {loss}")

    print(json.dumps({
        "status": "ok",
        "device": device,
        "batch_size": processed["action"].shape[0],
        "action_shape": list(processed["action"].shape),
        "valid_action_steps": int((~padding).sum().item()),
        "masked_cameras": sorted(missing_cameras),
        "loss": float(loss.item()),
        "weights_updated": False,
    }, indent=2))


if __name__ == "__main__":
    main()