"""
Policy interface every real policy (SmolVLA, Pi0.5, ACT, ...) must satisfy
to plug into eval/loop.py. DummyPolicy below is the placeholder used to
prove the eval cycle itself works end-to-end before a real model exists.
"""
from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):
    @abstractmethod
    def predict(self, obs: dict, instruction: str) -> np.ndarray:
        """obs: dict from BimanualTableEnv (images, joint_pos, joint_vel,
        drawer_slide). instruction: natural-language task string.
        Returns: (12,) float array, normalized to [-1, 1], in the order
        defined by configs/action_space.yaml `actuators`."""
        raise NotImplementedError


class DummyPolicy(Policy):
    """Ignores obs/instruction entirely -- outputs a smooth, bounded,
    per-joint sine sweep so both arms visibly and continuously move.
    Purpose: exercise the full env/eval loop (camera -> policy -> action ->
    physics) with something deterministic and cheap, standing in for a real
    VLA until one is trained. Not a manipulation strategy of any kind."""

    def __init__(self, amplitude: float = 0.3):
        self.amplitude = amplitude
        self._t = 0

    def predict(self, obs: dict, instruction: str) -> np.ndarray:
        self._t += 1
        phase = self._t * 0.05
        joint_offsets = np.arange(12) * 0.3
        action = self.amplitude * np.sin(phase + joint_offsets)
        return action
