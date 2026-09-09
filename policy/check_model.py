"""Check that the downloaded SmolVLA checkpoint loads successfully."""

import argparse
from pathlib import Path

import yaml
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

    if not (model_path / "config.json").is_file():
        parser.error(f"Checkpoint config not found in {model_path}")

    device = settings["device"]

    print(f"Loading checkpoint from: {model_path}")
    print(f"Device: {device}")

    policy = SmolVLAPolicy.from_pretrained(
        str(model_path),
        device=device,
    )
    policy.to(device)
    policy.eval()

    print("SmolVLA weights loaded successfully.")
    print("Input features:", policy.config.input_features)
    print("Output features:", policy.config.output_features)


if __name__ == "__main__":
    main()