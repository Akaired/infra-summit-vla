"""Download the experimental checkpoint at a reproducible revision."""

import argparse
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resolve-revision", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    settings = config["experiment"]
    checkpoint = settings["pretrained_checkpoint"]
    revision = settings.get("pretrained_revision")

    if not checkpoint:
        parser.error("Set experiment.pretrained_checkpoint in the config.")

    if args.resolve_revision:
        revision = HfApi().model_info(checkpoint).sha
        print("Copy this into experiment.pretrained_revision:")
        print(revision)
        return

    if not revision or len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision.lower()
    ):
        parser.error(
            "Set experiment.pretrained_revision to the full commit SHA. "
            "Run with --resolve-revision first."
        )

    destination = Path(settings["local_model_dir"])
    if not destination.is_absolute():
        destination = repo_root / destination

    path = snapshot_download(
        repo_id=checkpoint,
        revision=revision,
        local_dir=destination,
    )

    print(f"Checkpoint: {checkpoint}")
    print(f"Revision: {revision}")
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()