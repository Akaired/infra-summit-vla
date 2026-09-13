"""
Scene interface for the dinner-table task: observations in, actions out.

This is the ONLY thing /policy and /eval should ever touch directly instead
of raw mujoco calls -- reset(), step(), and the observation dict are the
whole contract. Swapping the placeholder rig for real SO-101 needs zero
changes here, same as the rest of /sim: everything is resolved by name from
configs/sim.yaml and configs/action_space.yaml, nothing hardcoded.
"""
import sys
import os
import numpy as np
import mujoco
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from randomization import DomainRandomizer  # noqa: E402


class BimanualTableEnv:
    def __init__(self, sim_cfg_path="configs/sim.yaml",
                 randomization_cfg_path="configs/randomization.yaml",
                 action_space_cfg_path="configs/action_space.yaml"):
        with open(sim_cfg_path) as f:
            self.sim_cfg = yaml.safe_load(f)
        with open(action_space_cfg_path) as f:
            self.action_cfg = yaml.safe_load(f)

        self.model = mujoco.MjModel.from_xml_path(self.sim_cfg["scene"]["mjcf"])
        self.data = mujoco.MjData(self.model)
        self.randomizer = DomainRandomizer(self.model, randomization_cfg_path)

        self.control_decimation = self.sim_cfg["physics"]["control_decimation"]

        # Action order is the config's contract; ranges are read from the
        # compiled model, never duplicated as numbers in a yaml file.
        self.actuator_names = self.action_cfg["actuators"]
        self.actuator_ids = np.array(
            [self.model.actuator(n).id for n in self.actuator_names])
        self.ctrl_lo = self.model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[self.actuator_ids, 1].copy()

        # One qpos/qvel/dof address per actuator's joint, same order --
        # this is what makes "joint_pos"/"joint_vel" in the observation line
        # up index-for-index with the action vector.
        self._joint_qpos_adr = []
        self._joint_dof_adr = []
        for act_id in self.actuator_ids:
            joint_id = self.model.actuator_trnid[act_id, 0]
            self._joint_qpos_adr.append(self.model.jnt_qposadr[joint_id])
            self._joint_dof_adr.append(self.model.jnt_dofadr[joint_id])

        self.drawer_qpos_adr = self.model.jnt_qposadr[self.model.joint("drawer_slide").id]

        self._renderers = {}
        for cam in self.sim_cfg["cameras"]:
            self._renderers[cam["name"]] = mujoco.Renderer(
                self.model, height=cam["height"], width=cam["width"])

        self._last_seed = None

    # ------------------------------------------------------------------ #
    def reset(self, seed: int) -> dict:
        self.randomizer.reset(self.data, seed)
        self._last_seed = seed
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, dict]:
        """action: (12,) normalized to [-1, 1], in the order defined by
        configs/action_space.yaml `actuators`. Out-of-range values are
        clipped, not silently trusted -- a policy bug shouldn't be able to
        request something outside what the actuator can even represent."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,):
            raise ValueError(f"action must be shape (12,), got {action.shape}")
        clipped = np.clip(action, -1.0, 1.0)
        real = self.ctrl_lo + (clipped + 1.0) / 2.0 * (self.ctrl_hi - self.ctrl_lo)
        self.data.ctrl[self.actuator_ids] = real

        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = {
            "seed": self._last_seed,
            "drawer_slide": float(self.data.qpos[self.drawer_qpos_adr]),
            "any_nan": bool(not np.all(np.isfinite(self.data.qpos))),
        }
        return obs, info

    # ------------------------------------------------------------------ #
    def render(self) -> dict:
        """RGB images only, keyed by camera name -- for calling outside of
        step() (e.g. a demo-recording loop that renders more often than it
        acts)."""
        images = {}
        for name, renderer in self._renderers.items():
            renderer.update_scene(self.data, camera=name)
            images[name] = renderer.render()
        return images

    def _get_obs(self) -> dict:
        joint_pos = np.array([self.data.qpos[a] for a in self._joint_qpos_adr])
        joint_vel = np.array([self.data.qvel[a] for a in self._joint_dof_adr])
        return {
            "images": self.render(),
            "joint_pos": joint_pos,       # (12,) radians/meters, actuator order
            "joint_vel": joint_vel,       # (12,) rad/s or m/s, same order
            "drawer_slide": float(self.data.qpos[self.drawer_qpos_adr]),
        }

    # ------------------------------------------------------------------ #
    # Testing helpers -- NOT how a policy interacts with the drawer. A
    # policy must grasp the handle and pull; this is for quickly checking
    # "does my observation pipeline correctly reflect an open drawer"
    # without needing a working grasp behavior yet.
    def debug_set_drawer(self, slide_value: float):
        lo, hi = self.model.jnt_range[self.model.joint("drawer_slide").id]
        self.data.qpos[self.drawer_qpos_adr] = float(np.clip(slide_value, lo, hi))
        mujoco.mj_forward(self.model, self.data)
