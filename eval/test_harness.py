"""Minimal end-to-end check: the episode loop runs on one seed with the dummy
policy without crashing.

This is infrastructure verification only -- it does NOT assert the task
succeeds or that any subtask completes (the dummy policy emits random actions).
It asserts the harness wiring holds: config resolves, the scene compiles and
resets, the policy is driven through the Protocol, subtask detection returns
the configured set of subtasks, and a well-formed record comes back.

Run:  pytest eval/test_harness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.policy_interface import Policy, build_policy  # noqa: E402
from eval.run_episodes import load_bundle, run_episode  # noqa: E402
from eval.scene import EpisodeScene  # noqa: E402

EVAL_CONFIG = "configs/eval.yaml"
STEPS = 3  # a few policy ticks: enough to exercise observe -> predict -> step -> detect


def test_episode_loop_runs_end_to_end(tmp_path):
    bundle = load_bundle(EVAL_CONFIG)
    scene = EpisodeScene(bundle.sim_cfg, bundle.randomization_config_path)
    policy = build_policy("dummy", bundle.policy_config_path, bundle.primary_camera)
    assert isinstance(policy, Policy)

    seed = bundle.eval_seeds[0]
    record = run_episode(
        scene=scene,
        policy=policy,
        bundle=bundle,
        seed=seed,
        max_policy_steps=STEPS,
        record_video=False,
        output_dir=tmp_path,
        verbose=False,
    )

    assert record["seed"] == seed
    assert record["policy_steps"] == STEPS
    assert record["physics_steps"] == STEPS * scene.control_decimation
    assert set(record["subtasks"]) == set(bundle.eval_cfg["subtasks"])
    assert record["outcome"] in {"success", "failure"}
    # pour has no simulated liquid -- it must be reported honestly, never faked.
    assert record["subtasks"]["pour_completed"] == "not_implemented"
    assert isinstance(record["instruction"], str) and record["instruction"]


def test_observation_contract():
    bundle = load_bundle(EVAL_CONFIG)
    scene = EpisodeScene(bundle.sim_cfg, bundle.randomization_config_path)
    scene.reset(bundle.eval_seeds[0])

    obs = scene.observe()
    expected_cameras = {cam["name"] for cam in bundle.sim_cfg["cameras"]}
    assert set(obs["images"]) == expected_cameras
    for frame in obs["images"].values():
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert str(frame.dtype) == "uint8"
    assert obs["robot_state"].ndim == 1 and obs["robot_state"].shape[0] > 0


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-v"]))
