"""Load SO101 demonstrations into batches for policy preprocessing."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


class PolicyDataset(Dataset):
    """Expose images, task text, state, action chunks, and padding masks.

    State and actions remain in dataset units. Apply LeRobot's
    dataset-aware normalization before using these batches for training.
    """

    def __init__(self, source, camera_map, model_config):
        self.source = source
        self.camera_map = camera_map
        self.input_features = model_config["input_features"]

        if len(set(camera_map.values())) != len(camera_map):
            raise ValueError("Each camera must map to a unique model input")

        for source_key, target_key in camera_map.items():
            if source_key not in source.meta.camera_keys:
                raise ValueError(f"Dataset camera does not exist: {source_key}")
            if target_key not in self.input_features:
                raise ValueError(f"Model camera does not exist: {target_key}")

        checks = (
            ("observation.state", model_config["input_features"]),
            ("action", model_config["output_features"]),
        )
        for key, features in checks:
            actual = tuple(source.meta.features[key]["shape"])
            expected = tuple(features[key]["shape"])
            if actual != expected:
                raise ValueError(
                    f"{key}: dataset shape {actual}, model shape {expected}"
                )

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        task = sample["task"]
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Missing task instruction at sample {index}")

        result = {
            "task": task,
            "observation.state": sample["observation.state"].float(),
            "action": sample["action"].float(),
            "action_is_pad": sample["action_is_pad"].bool(),
        }

        for source_key, target_key in self.camera_map.items():
            image = sample[source_key]

            # LeRobot returns float images in CHW layout, within [0, 1].
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"{source_key}: expected RGB CHW image, got {image.shape}"
                )

            target_shape = self.input_features[target_key]["shape"]
            result[target_key] = F.interpolate(
                image.unsqueeze(0),
                size=tuple(target_shape[-2:]),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)

        return result


def build_dataloader(config_path):
    with resolve_path(config_path).open(encoding="utf-8") as file:
        experiment = yaml.safe_load(file)["experiment"]

    settings = experiment["dataset"]
    loader_settings = settings["loader"]

    model_config_path = (
        resolve_path(experiment["local_model_dir"]) / "config.json"
    )
    with model_config_path.open(encoding="utf-8") as file:
        model_config = json.load(file)

    chunk_size = model_config["chunk_size"]
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Model chunk_size must be a positive integer")

    metadata = LeRobotDatasetMetadata(
        repo_id=settings["repo_id"],
        revision=settings["revision"],
    )

    episodes = settings["episodes"]
    if episodes is not None:
        if not episodes or any(
            type(ep) is not int or not 0 <= ep < metadata.total_episodes
            for ep in episodes
        ):
            raise ValueError("dataset.episodes contains invalid episode indices")

    source = LeRobotDataset(
        repo_id=settings["repo_id"],
        revision=settings["revision"],
        episodes=episodes,
        delta_timestamps={
            "action": [
                step / metadata.fps
                for step in range(chunk_size)
            ],
        },
        video_backend=loader_settings["video_backend"],
    )

    dataset = PolicyDataset(
        source=source,
        camera_map=settings["camera_map"],
        model_config=model_config,
    )

    generator = torch.Generator()
    generator.manual_seed(loader_settings["seed"])

    return DataLoader(
        dataset,
        batch_size=loader_settings["batch_size"],
        shuffle=loader_settings["shuffle"],
        num_workers=loader_settings["num_workers"],
        generator=generator,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print("Loading dataset and episode videos...", flush=True)
    loader = build_dataloader(args.config)

    print("Decoding first batch...", flush=True)
    batch = next(iter(loader))

    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinite values")
            print(f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{name}: {value}")

    print("Dataset batch loaded successfully.")


if __name__ == "__main__":
    main()