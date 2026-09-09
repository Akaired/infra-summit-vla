"""Thin wrapper around the MuJoCo dinner-table scene for the episode harness.

Everything the harness needs from ``/sim`` goes through this class:
reset-to-seed, render an observation, apply an action, advance physics, and
read back the handful of geometric quantities the subtask detectors check.
The harness never imports ``mujoco`` or ``sim.randomization`` itself.

Model compilation and the ``DomainRandomizer`` are ``/sim`` code, reused here
-- not reimplemented (see eval/README.md, "The harness orchestrates").
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from eval.common import REPO_ROOT, resolve_path

# sim/ is a script directory, not an importable package (no __init__.py), so it
# is added to the path exactly the way sim/test_randomization.py expects to be
# run -- from inside sim/.
sys.path.insert(0, str(REPO_ROOT / "sim"))

import mujoco  # noqa: E402  (after sys.path tweak)

from randomization import DomainRandomizer  # noqa: E402


class EpisodeScene:
    """One compiled scene, reused across every seed in a run."""

    def __init__(self, sim_cfg: dict[str, Any], randomization_cfg_path: str) -> None:
        self.sim_cfg = sim_cfg
        mjcf_path = resolve_path(sim_cfg["scene"]["mjcf"])
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)

        self.randomizer = DomainRandomizer(self.model, str(resolve_path(randomization_cfg_path)))

        self.control_decimation = int(sim_cfg["physics"]["control_decimation"])
        self.robot_prefixes = (
            sim_cfg["robot"]["left_arm_prefix"],
            sim_cfg["robot"]["right_arm_prefix"],
        )

        self._cameras = sim_cfg["cameras"]
        self._renderers = {
            cam["name"]: mujoco.Renderer(
                self.model, height=int(cam["height"]), width=int(cam["width"])
            )
            for cam in self._cameras
        }

        self._actuated_qpos_adr, self._actuated_qvel_adr = self._actuated_addrs()
        self.in_drawer_items: set[str] = set()

    # ------------------------------------------------------------------ #
    def _actuated_addrs(self) -> tuple[list[int], list[int]]:
        """qpos / qvel addresses of the actuated (robot) joints, in actuator
        order. Model-agnostic: 'joint driven by an actuator', no name list."""
        qpos, qvel = [], []
        for act_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[act_id, 0])
            if joint_id >= 0:
                qpos.append(int(self.model.jnt_qposadr[joint_id]))
                qvel.append(int(self.model.jnt_dofadr[joint_id]))
        return qpos, qvel

    # ------------------------------------------------------------------ #
    def reset(self, seed: int) -> None:
        """Reset to the randomized scene for ``seed`` (already settled)."""
        self.randomizer.reset(self.data, seed)
        mujoco.mj_forward(self.model, self.data)
        self.in_drawer_items = set(self.randomizer.last_in_drawer_items)

    def observe(self) -> dict[str, Any]:
        images = {}
        for name, renderer in self._renderers.items():
            renderer.update_scene(self.data, camera=name)
            images[name] = renderer.render()
        return {"images": images, "robot_state": self.robot_state()}

    def robot_state(self) -> np.ndarray:
        qpos = self.data.qpos[self._actuated_qpos_adr]
        qvel = self.data.qvel[self._actuated_qvel_adr]
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def apply_action(self, action: np.ndarray) -> None:
        """Write a flat action into ``data.ctrl``, clipped to actuator ranges.

        With the dummy policy these values are uncalibrated dimensionless test
        numbers (see policy/dummy_policy.py) -- the clip keeps them legal, the
        harness only checks the loop survives, not that the motion is useful.
        """
        flat = np.asarray(action, dtype=np.float64).ravel()
        if flat.shape[0] != self.model.nu:
            raise ValueError(
                f"policy returned {flat.shape[0]} values but the model has "
                f"{self.model.nu} actuators"
            )
        low = self.model.actuator_ctrlrange[:, 0]
        high = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(flat, low, high)

    def step(self) -> None:
        """Advance physics by one policy tick (``control_decimation`` steps)."""
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ #
    # Geometry read-back for subtask detection.
    # ------------------------------------------------------------------ #
    def body_xpos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(name).id].copy()

    def body_linear_speed(self, name: str) -> float:
        dof = int(self.model.body_dofadr[self.model.body(name).id])
        return float(np.linalg.norm(self.data.qvel[dof : dof + 3]))

    def joint_qpos(self, name: str) -> float:
        return float(self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]])

    def joint_range(self, name: str) -> tuple[float, float]:
        low, high = self.model.jnt_range[self.model.joint(name).id]
        return float(low), float(high)

    def table_top_z(self) -> float:
        """World Z of the table's top surface, from the ``table_top`` geom."""
        gid = self.model.geom("table_top").id
        return float(self.data.geom_xpos[gid][2] + self.model.geom_size[gid][2])

    def bodies_touching(self, name: str) -> set[str]:
        """Names of every body currently in contact with body ``name``."""
        target = self.model.body(name).id
        touching: set[str] = set()
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            b1 = int(self.model.geom_bodyid[con.geom1])
            b2 = int(self.model.geom_bodyid[con.geom2])
            if target == b1:
                other = b2
            elif target == b2:
                other = b1
            else:
                continue
            other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other)
            if other_name:
                touching.add(other_name)
        return touching
