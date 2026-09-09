"""The one interface the harness knows about.

``inference/README.md`` fixes the policy contract as a single call::

    (instruction, observation, robot_state) -> action

The harness depends only on :class:`Policy` below. Today that contract is
satisfied by ``policy/dummy_policy.py`` (no model, random actions); tomorrow it
is an OpenVINO runtime from ``/inference``. Swapping one for the other is a
change to :func:`build_policy` here -- never to the harness, the scene, or the
subtask detectors.

Observation contract handed to :meth:`Policy.predict`:

    observation["images"]      -> dict[camera_name, np.ndarray(H, W, 3) uint8]
                                  one entry per configs/sim.yaml:cameras entry
    observation["robot_state"] -> np.ndarray(float32), actuated-joint qpos then qvel

``robot_state`` is also passed as the third positional argument, so a policy
that wants only the state never has to unpack the dict.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """Anything the harness can drive an episode with."""

    def predict(
        self,
        instruction: str,
        observation: Mapping[str, Any],
        robot_state: np.ndarray,
    ) -> np.ndarray:
        """Map one observation to one flat action vector (float32)."""
        ...


class DummyPolicyAdapter:
    """Adapt ``policy.dummy_policy.DummyPolicy`` to the :class:`Policy` contract.

    ``DummyPolicy.predict`` takes a single RGB frame, not the multi-camera
    observation dict the harness produces, so this narrows the observation to
    the configured primary camera. The real OpenVINO runtime will implement
    :class:`Policy` directly and need no adapter -- or bring its own, still in
    this file.
    """

    def __init__(self, dummy: Any, primary_camera: str) -> None:
        self._dummy = dummy
        self._primary_camera = primary_camera

    def predict(
        self,
        instruction: str,
        observation: Mapping[str, Any],
        robot_state: np.ndarray,
    ) -> np.ndarray:
        images = observation["images"]
        if self._primary_camera not in images:
            raise KeyError(
                f"policy.primary_camera {self._primary_camera!r} is not one of the "
                f"rendered cameras {sorted(images)} -- check configs/eval.yaml against "
                f"configs/sim.yaml:cameras"
            )
        frame = images[self._primary_camera]
        action = self._dummy.predict(instruction, frame, robot_state)
        return np.asarray(action, dtype=np.float32)


def build_policy(name: str, policy_config_path: str, primary_camera: str) -> Policy:
    """Construct the policy named on the command line.

    ``dummy`` is the only wired option today. ``openvino`` is reserved for the
    ``/inference`` runtime and raises until that module exposes a
    :class:`Policy`-compatible loader.
    """
    if name == "dummy":
        from policy.dummy_policy import DummyPolicy

        dummy = DummyPolicy.from_config(policy_config_path)
        return DummyPolicyAdapter(dummy, primary_camera)
    if name == "openvino":
        raise NotImplementedError(
            "the 'openvino' policy is not wired yet -- /inference must first expose a "
            "runtime implementing eval.policy_interface.Policy (see inference/README.md). "
            "Until then run with --policy dummy."
        )
    raise ValueError(f"unknown policy {name!r} (expected 'dummy' or 'openvino')")
