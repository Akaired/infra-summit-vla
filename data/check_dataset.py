"""Inspect public dataset metadata before downloading episode videos."""

import argparse
from pathlib import Path

import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    with config_path.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)["experiment"]["dataset"]

    metadata = LeRobotDatasetMetadata(
        repo_id=settings["repo_id"],
        revision=settings["revision"],
    )

    print("Dataset:", settings["repo_id"])
    print("FPS:", metadata.fps)
    print("Episodes:", metadata.total_episodes)
    print("Frames:", metadata.total_frames)

    print("\nCamera keys:")
    for key in metadata.camera_keys:
        print(f"  {key}: {metadata.features[key]}")

    print("\nRobot state:")
    print(metadata.features["observation.state"])

    print("\nActions:")
    print(metadata.features["action"])

    print("\nTasks:")
    print(metadata.tasks)


if __name__ == "__main__":
    main()