"""Episode harness -- deliverable #2 (reproducible sim package) and #4 (10-seed
demonstration video / success summary).

One command resets the scene at each of the 10 reserved eval seeds, drives it
with a policy behind :class:`eval.policy_interface.Policy`, detects subtask
completion from physics state, and writes one JSONL record per episode plus an
optional per-seed video, in the layout ``configs/eval.yaml:logging`` specifies.

    python eval/run_episodes.py --config configs/eval.yaml

The standalone Intel benchmark (deliverable #3) is a *separate* entry point and
does not live here -- it must not step physics (see eval/README.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repository root on the path so `policy.*` (and this package) import cleanly
# no matter the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import REPO_ROOT, load_yaml, resolve_path
from eval.policy_interface import Policy, build_policy
from eval.scene import EpisodeScene
from eval.subtasks import STATUS_NOT_IMPLEMENTED, SubtaskTracker


@dataclass
class Bundle:
    """Everything an episode needs, resolved once from configs/eval.yaml."""

    eval_cfg: dict[str, Any]
    sim_cfg: dict[str, Any]
    randomization_cfg: dict[str, Any]
    eval_config_path: str
    randomization_config_path: str
    policy_config_path: str

    @property
    def instruction(self) -> str:
        return " ".join(self.eval_cfg["episode"]["instruction"].split())

    @property
    def eval_seeds(self) -> list[int]:
        return list(self.randomization_cfg["seeds"])

    @property
    def primary_camera(self) -> str:
        return self.eval_cfg["policy"]["primary_camera"]

    def max_policy_steps(self, physics_steps: int) -> int:
        decimation = int(self.sim_cfg["physics"]["control_decimation"])
        return -(-physics_steps // decimation)  # ceil


def load_bundle(eval_config_path: str) -> Bundle:
    eval_cfg = load_yaml(eval_config_path)
    sibling = eval_cfg["configs"]
    return Bundle(
        eval_cfg=eval_cfg,
        sim_cfg=load_yaml(sibling["sim"]),
        randomization_cfg=load_yaml(sibling["randomization"]),
        eval_config_path=eval_config_path,
        randomization_config_path=sibling["randomization"],
        policy_config_path=sibling["policy"],
    )


# --------------------------------------------------------------------------- #
def run_episode(
    *,
    scene: EpisodeScene,
    policy: Policy,
    bundle: Bundle,
    seed: int,
    max_policy_steps: int,
    record_video: bool,
    output_dir: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one seed to success or step budget; return its JSONL record."""
    started = time.perf_counter()
    scene.reset(seed)

    tracker = SubtaskTracker(scene, bundle.eval_cfg, bundle.sim_cfg)
    tracker.bind_spawn_reference()

    primary = bundle.primary_camera
    frames: list[Any] = []
    policy_steps = 0

    for step in range(max_policy_steps):
        obs = scene.observe()
        if record_video:
            frames.append(obs["images"][primary])
        action = policy.predict(bundle.instruction, obs, obs["robot_state"])
        scene.apply_action(action)
        scene.step()
        tracker.update(step)
        policy_steps = step + 1
        if tracker.all_terminal():
            break

    require_all = bool(bundle.eval_cfg["success"]["require_all_subtasks"])
    report = tracker.report(require_all_subtasks=require_all)

    video_path: Path | None = None
    if record_video and frames:
        import mediapy

        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"seed_{seed}.mp4"
        mediapy.write_video(video_path, frames, fps=int(bundle.eval_cfg["logging"]["video_fps"]))

    record = {
        "seed": seed,
        "instruction": bundle.instruction,
        **report,
        "outcome": "success" if report["success"] else "failure",
        "policy_steps": policy_steps,
        "physics_steps": policy_steps * scene.control_decimation,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "in_drawer_items": sorted(scene.in_drawer_items),
        "video_path": (
            str(video_path.relative_to(REPO_ROOT)) if video_path is not None else None
        ),
    }
    if verbose:
        _print_episode_line(record)
    return record


def _print_episode_line(record: dict[str, Any]) -> None:
    done = record["subtasks_completed"]
    total = record["subtasks_total"]
    print(
        f"  seed {record['seed']:>2}: {record['outcome']:<7} "
        f"{done}/{total} subtasks  "
        f"({record['policy_steps']} policy steps, {record['wall_time_s']}s)"
    )


# --------------------------------------------------------------------------- #
def run_evaluation(
    *,
    bundle: Bundle,
    seeds: list[int],
    policy_name: str,
    physics_step_budget: int,
    record_video: bool,
    output_dir: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    scene = EpisodeScene(bundle.sim_cfg, bundle.randomization_config_path)
    policy = build_policy(policy_name, bundle.policy_config_path, bundle.primary_camera)

    non_eval = [s for s in seeds if not scene.randomizer.is_eval_seed(s)]
    if non_eval:
        print(
            f"WARNING: {non_eval} are not in configs/randomization.yaml:seeds -- "
            f"eval results are only reported over the reserved 10.",
            file=sys.stderr,
        )

    max_policy_steps = bundle.max_policy_steps(physics_step_budget)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "episodes.jsonl"

    if verbose:
        print(
            f"Running {len(seeds)} episode(s) | policy={policy_name} | "
            f"budget={physics_step_budget} physics steps "
            f"(<= {max_policy_steps} policy steps) | video={record_video}"
        )
        print(f'instruction: "{bundle.instruction}"')

    records: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for seed in seeds:
            record = run_episode(
                scene=scene,
                policy=policy,
                bundle=bundle,
                seed=seed,
                max_policy_steps=max_policy_steps,
                record_video=record_video,
                output_dir=output_dir,
                verbose=verbose,
            )
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records.append(record)

    summary = _summarise(bundle, seeds, records, policy_name)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if verbose:
        _print_summary(summary, jsonl_path)
    return summary


def _summarise(
    bundle: Bundle,
    seeds: list[int],
    records: list[dict[str, Any]],
    policy_name: str,
) -> dict[str, Any]:
    ordered_subtasks = list(bundle.eval_cfg["subtasks"])
    per_subtask_completed = {
        name: sum(1 for r in records if r["subtasks"].get(name) == "completed")
        for name in ordered_subtasks
    }
    not_implemented = sorted(
        {
            name
            for r in records
            for name, status in r["subtasks"].items()
            if status == STATUS_NOT_IMPLEMENTED
        }
    )
    successes = sum(1 for r in records if r["success"])
    n = len(records)
    return {
        # Deliberately a raw count, not a percentage. With only 10 fixed seeds
        # the confidence interval on a success rate is enormous; a "70.0%"
        # would imply a precision the sample size does not support. Report
        # "successes / N" and let the write-up own the caveat.
        "full_task_success_ratio": f"{successes}/{n}",
        "full_task_successes": successes,
        "n_seeds": n,
        "seeds": seeds,
        "policy": policy_name,
        "not_implemented_subtasks": not_implemented,
        "per_subtask_completed": per_subtask_completed,
        "require_all_subtasks": bool(bundle.eval_cfg["success"]["require_all_subtasks"]),
        "generated_by": "eval/run_episodes.py",
        "configs": {
            "eval": bundle.eval_config_path,
            "sim": bundle.eval_cfg["configs"]["sim"],
            "randomization": bundle.eval_cfg["configs"]["randomization"],
            "policy": bundle.eval_cfg["configs"]["policy"],
        },
    }


def _print_summary(summary: dict[str, Any], jsonl_path: Path) -> None:
    print("\n=== Evaluation summary ===")
    print(f"seeds evaluated  : {summary['n_seeds']}  {summary['seeds']}")
    print(
        f"full-task success: {summary['full_task_success_ratio']}  "
        f"(raw count of {summary['n_seeds']}, not a rate)"
    )
    if summary["not_implemented_subtasks"]:
        joined = ", ".join(summary["not_implemented_subtasks"])
        print(
            f"  note: subtask(s) [{joined}] are not implemented (no physical state "
            f"to check), so\n        full-task success is unreachable by construction "
            f"right now."
        )
    print(f"per-subtask completion (raw counts, N={summary['n_seeds']}):")
    for name, count in summary["per_subtask_completed"].items():
        if name in summary["not_implemented_subtasks"]:
            print(f"  {name:<20}: n/a (not implemented)")
        else:
            print(f"  {name:<20}: {count}/{summary['n_seeds']}")
    print(f"records: {jsonl_path}")


# --------------------------------------------------------------------------- #
def _parse_seed_list(raw: str) -> list[int]:
    return [int(tok) for tok in raw.split(",") if tok.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        default="configs/eval.yaml",
        help="eval config, relative to the repository root (default: configs/eval.yaml)",
    )
    parser.add_argument(
        "--policy",
        default="dummy",
        choices=["dummy", "openvino"],
        help="which policy to drive episodes with (default: dummy)",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seed_list,
        default=None,
        help="comma-separated seed override for debugging; default is "
        "configs/randomization.yaml:seeds (the reserved 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="physics-step budget per episode; default is configs/eval.yaml:episode.max_steps",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory; default is configs/eval.yaml:logging.output_dir",
    )
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--video", dest="video", action="store_true", default=None, help="force per-seed video on")
    video.add_argument("--no-video", dest="video", action="store_false", default=None, help="force per-seed video off")
    parser.add_argument("--quiet", action="store_true", help="suppress per-episode progress output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bundle = load_bundle(args.config)

    seeds = args.seeds if args.seeds is not None else bundle.eval_seeds
    physics_budget = args.max_steps if args.max_steps is not None else int(bundle.eval_cfg["episode"]["max_steps"])
    record_video = args.video if args.video is not None else bool(bundle.eval_cfg["logging"]["record_video"])
    output_dir = resolve_path(args.output_dir or bundle.eval_cfg["logging"]["output_dir"])

    run_evaluation(
        bundle=bundle,
        seeds=seeds,
        policy_name=args.policy,
        physics_step_budget=physics_budget,
        record_video=record_video,
        output_dir=output_dir,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
