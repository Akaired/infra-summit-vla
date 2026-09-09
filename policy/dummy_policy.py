"""Reproducible random actions for interface testing; no model or simulator needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import yaml


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class DummyPolicy:
    """Return [left controls..., right controls...] in dimensionless test units.

    Images and instructions are validated, but their meaning is ignored.
    robot_state is reserved for later integration and is currently ignored.
    These actions are not calibrated robot commands.
    """

    def __init__(self, settings: Mapping):
        self.controls_per_arm = _positive_int(
            settings.get("controls_per_arm"), "dummy.controls_per_arm"
        )
        seed = settings.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError("dummy.seed must be a non-negative integer")
        bounds = settings.get("action_bounds")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("dummy.action_bounds must contain [low, high]")
        if any(type(x) not in (int, float) for x in bounds):
            raise ValueError("dummy.action_bounds must be numeric")
        low, high = bounds
        limit = np.finfo(np.float32).max
        if not all(np.isfinite(x) and abs(x) <= limit for x in bounds) or low >= high:
            raise ValueError("dummy.action_bounds must be finite float32 bounds with low < high")
        self.low, self.high = low, high
        self._rng = np.random.default_rng(seed)
        smoke = settings.get("smoke_image")
        if not isinstance(smoke, Mapping):
            raise ValueError("dummy.smoke_image must contain height and width")
        self.smoke_shape = (
            _positive_int(smoke.get("height"), "dummy.smoke_image.height"),
            _positive_int(smoke.get("width"), "dummy.smoke_image.width"),
            3,
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> DummyPolicy:
        """Resolve relative config paths against the repository root."""
        path = Path(config_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, Mapping) or not isinstance(config.get("dummy"), Mapping):
            raise ValueError("Policy config must contain a dummy mapping")
        return cls(config["dummy"])

    def predict(
        self, instruction: str, observation: np.ndarray, robot_state=None
    ) -> np.ndarray:
        """Accept an RGB uint8 image and return a flat float32 bimanual action."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if (
            not isinstance(observation, np.ndarray)
            or observation.ndim != 3
            or observation.shape[2] != 3
            or observation.size == 0
            or observation.dtype != np.uint8
        ):
            raise ValueError("observation must be a non-empty RGB uint8 array of shape (H, W, 3)")
        return self._rng.uniform(
            self.low, self.high, size=2 * self.controls_per_arm
        ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Policy YAML, relative to repository root")
    parser.add_argument("--instruction", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Image path, relative to current directory")
    source.add_argument("--smoke-test", action="store_true", help="Use a synthetic RGB image")
    args = parser.parse_args()
    try:
        policy = DummyPolicy.from_config(args.config)
        if args.smoke_test:
            observation = np.zeros(policy.smoke_shape, dtype=np.uint8)
        else:
            from PIL import Image

            with Image.open(args.image) as image:
                observation = np.array(image.convert("RGB"))
        action = policy.predict(args.instruction, observation)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    split = policy.controls_per_arm
    print(json.dumps({
        "action": action.tolist(),
        "left": action[:split].tolist(),
        "right": action[split:].tolist(),
    }))


if __name__ == "__main__":
    main()
