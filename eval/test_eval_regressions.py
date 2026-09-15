from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from eval.diagnose_plate import longest_successful_hold
from eval.eval_smolvla import (
    TaskTracker,
    load_rollout_config,
    load_yaml,
    select_output_directory,
)
from eval.json_utils import dumps_strict, sanitize_json
from eval.scene import EpisodeScene


class FakeScene:
    def __init__(self) -> None:
        self.drawer = 0.0
        self.plate = np.array([0.2, 0.0, 0.7], dtype=np.float64)
        self.speed = 0.0
        self.contacts: set[str] = set()

    def joint_range(self, name: str) -> tuple[float, float]:
        return 0.0, 1.0

    def joint_qpos(self, name: str) -> float:
        return self.drawer

    def body_xpos(self, name: str) -> np.ndarray:
        return self.plate.copy()

    def body_linear_speed(self, name: str) -> float:
        return self.speed

    def bodies_touching(self, name: str) -> set[str]:
        return set(self.contacts)


def tracker_config() -> dict:
    return {
        "events": {
            "drawer_opened": "drawer_opened",
            "drawer_closed": "drawer_closed",
            "plate_lifted": "plate_lifted",
            "plate_placed": "plate_placed",
        },
        "sequence": [
            "drawer_opened", "drawer_closed", "plate_lifted", "plate_placed"
        ],
        "drawer": {
            "joint_name": "drawer_slide",
            "opened_travel_fraction": 0.6,
            "closed_travel_fraction": 0.15,
        },
        "plate": {
            "body_name": "plate",
            "gripper_action_index": 5,
            "gripper_close_threshold": -0.05,
            "gripper_contact_bodies": ["left_gripper", "left_jaw"],
            "lifted_delta_z_m": 0.05,
            "target_xy_m": [0.0, 0.0],
            "target_radius_m": 0.12,
            "surface_z_tolerance_m": 0.035,
            "resting_linear_speed_mps": 0.08,
        },
    }


def advance_drawer(tracker: TaskTracker, scene: FakeScene) -> None:
    scene.drawer = 0.8
    tracker.update(0, np.zeros(6))
    scene.drawer = 0.0
    tracker.update(1, np.zeros(6))


class EvalRegressionTests(unittest.TestCase):
    def test_rollout_uses_complete_external_grasp_config(self) -> None:
        cfg = load_rollout_config("configs/smolvla_rollout.yaml")
        plate = cfg["grasp_assist"]["plate"]

        self.assertEqual(
            cfg["configs"]["grasp_assist"], "configs/grasp_assist.yaml"
        )
        self.assertEqual(plate["activation_mode"], "contact")
        self.assertGreaterEqual(plate["open_threshold"], 0.6)
        self.assertGreaterEqual(plate["release_open_hold_policy_steps"], 2)

    def test_reserved_full_drawer_seeds_do_not_overlap_training(self) -> None:
        cfg = load_yaml("configs/randomization.yaml")
        training_low, training_high = cfg["training"]["seed_range"]

        self.assertTrue(
            all(
                seed < training_low or seed >= training_high
                for seed in cfg["seeds_full_drawer"]
            )
        )

    def test_scene_reset_restores_compiled_equality_defaults(self) -> None:
        scene = EpisodeScene.__new__(EpisodeScene)
        scene._initial_eq_active0 = np.array([1, 0], dtype=np.uint8)
        scene.model = SimpleNamespace(
            eq_active0=np.array([0, 1], dtype=np.uint8)
        )
        scene.data = SimpleNamespace(
            eq_active=np.array([0, 1], dtype=np.uint8)
        )

        class Randomizer:
            last_in_drawer_items = ["fork"]

            def reset(inner_self, data, seed):
                self.assertTrue(
                    np.array_equal(scene.model.eq_active0, [1, 0])
                )
                data.eq_active[:] = scene.model.eq_active0

        scene.randomizer = Randomizer()
        with mock.patch("eval.scene.mujoco.mj_forward"):
            scene.reset(3)
            scene.model.eq_active0[:] = [0, 1]
            scene.data.eq_active[:] = [0, 1]
            scene.reset(3)

        self.assertTrue(np.array_equal(scene.model.eq_active0, [1, 0]))
        self.assertTrue(np.array_equal(scene.data.eq_active, [1, 0]))
        self.assertEqual(scene.in_drawer_items, {"fork"})

    def test_assisted_lift_and_release_requirements(self) -> None:
        scene = FakeScene()
        grasp = SimpleNamespace(enabled=True, plate_active=True)
        tracker = TaskTracker(scene, tracker_config(), grasp_assist=grasp)
        advance_drawer(tracker, scene)

        scene.plate[2] += 0.06
        open_action = np.zeros(6)
        open_action[5] = 1.2
        tracker.update(2, open_action)
        self.assertIsNone(tracker.events["plate_lifted"])

        closed_action = np.zeros(6)
        closed_action[5] = -0.1
        tracker.update(3, closed_action)
        self.assertEqual(tracker.events["plate_lifted"], 3)

        scene.plate[:] = [0.0, 0.0, 0.7]
        tracker.update(4, open_action)
        self.assertIsNone(tracker.events["plate_placed"])
        self.assertFalse(tracker.currently_successful())

        grasp.plate_active = False
        tracker.update(5, open_action)
        self.assertEqual(tracker.events["plate_placed"], 5)
        self.assertTrue(tracker.currently_successful())

    def test_physical_grasp_without_assist_is_supported(self) -> None:
        scene = FakeScene()
        scene.contacts = {"left_jaw"}
        tracker = TaskTracker(scene, tracker_config())
        advance_drawer(tracker, scene)
        scene.plate[2] += 0.06

        action = np.zeros(6)
        action[5] = -0.1
        tracker.update(2, action)

        self.assertEqual(tracker.events["plate_lifted"], 2)

    def test_strict_json_replaces_nested_non_finite_values(self) -> None:
        value = {
            "nan": float("nan"),
            "nested": [float("inf"), {"negative": float("-inf")}],
            "numpy": np.float32("nan"),
            "finite": np.float64(1.25),
        }
        sanitized = sanitize_json(value)
        encoded = dumps_strict(value, sort_keys=True)

        self.assertIsNone(sanitized["nan"])
        self.assertEqual(sanitized["nested"], [None, {"negative": None}])
        self.assertIsNone(sanitized["numpy"])
        parsed = json.loads(
            encoded,
            parse_constant=lambda token: self.fail(
                f"non-standard JSON constant: {token}"
            ),
        )
        self.assertEqual(parsed["finite"], 1.25)
        self.assertFalse(any(x in encoded for x in ("NaN", "Infinity")))

    def test_diagnostic_hold_requires_released_weld_each_step(self) -> None:
        base = {
            "ev_plate_placed": 1, "ev_plate_lifted": 1,
            "ev_drawer_opened": 1, "ev_drawer_closed": 1,
            "target_dist": 0.01, "lift_vs_init": 0.0, "plate_speed": 0.0,
        }
        rows = [
            {**base, "step": 0, "weld_active": False},
            {**base, "step": 1, "weld_active": True},
            {**base, "step": 2, "weld_active": False},
            {**base, "step": 3, "weld_active": False},
        ]
        result = longest_successful_hold(rows, 0.12, 0.035, 0.08)
        self.assertEqual(result, (2, 2))

    def test_safe_output_directory_preserves_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / "summary.json"
            aggregate.write_text("original", encoding="utf-8")
            cfg = {"directory": str(root), "overwrite": False}
            first = select_output_directory(cfg, [3])
            second = select_output_directory(cfg, [3])

            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, root / "runs")
            self.assertEqual(aggregate.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
