"""Rollout-side chunk probe: does the policy hold the gripper closed on its OWN
observations, the way it does on dataset frames?

This is the other half of tools/chunk_probe.py. That one is teacher-forced: it feeds
a frame straight from the demo dataset, and the answer was unambiguous -- all three
checkpoints hold the gripper closed 50/50 through the carry (frames 230, 250) and
release at exactly the right step (frame 280 -> first OPEN at step 30). The gripper
channel is fit.

This script asks whether the same policy behaves the same way on an observation it
generated itself, inside the closed loop. If it holds there too, the rollout failure
is upstream of the gripper -- the arm not reaching the plate. If it flips open, the
eval-time input distribution is the problem, and the first suspect is the image
geometry, which this script reports explicitly at the top:

    training saw   240x320 -> resize_with_pad -> 384x512 content + 128 pad rows on top
    eval feeds     256x256 -> resize_with_pad -> 512x512 content +   0 pad rows

RUN THIS BEFORE touching configs/sim.yaml or the checkpoint's declared image shapes.
The point is to measure the pipeline as it stands; fix the geometry first and this
measurement is gone.

It does NOT modify the eval harness -- it imports EpisodeScene, SmolVLARunner,
GraspAssist and make_render_deterministic and reproduces run_episode's call order,
the same way eval/diagnose_plate.py does.

Two passes per seed, on purpose:
  pass 1 records only the cheap scalar trace and finds the rollout's own close event;
  pass 2 replays the same seed and stashes copies of the observations at the probe
  steps only (mujoco's Renderer may reuse its output buffer, so the images must be
  copied, and copying all 420 steps would cost ~600 MB).
The two traces are compared afterwards: a divergence means the rollout is not
reproducible, which is itself a finding (see EVAL_DIAGNOSIS.md B1 and B2).

    uv run python tools/rollout_probe.py --seeds 6 --deterministic-render
    uv run python tools/rollout_probe.py --seeds 0,6,7 --deterministic-render --json rollout_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.diagnose_plate import make_render_deterministic  # noqa: E402
from eval.eval_smolvla import (  # noqa: E402
    SmolVLARunner,
    load_rollout_config,
    load_yaml,
)
from eval.scene import EpisodeScene  # noqa: E402
from sim.grasp_assist import GraspAssist  # noqa: E402

L_GRIP, R_GRIP = 5, 11
PAD_TARGET = 512  # SmolVLAConfig.resize_imgs_with_padding

# Fallback only, used if the postprocessor cannot unnormalize a chunk.
# action[5] stats of datasets/demo-100 (meta/stats.json), cross-checked against the
# phase layout in scripted_episode_ik.py to 5 decimals.
L_GRIP_MEAN, L_GRIP_STD = 0.4108, 0.6189
DEMO_HOLD_FRAMES = 100  # length of the closed span in every demo episode


def band(v: float, close_thr: float, open_thr: float) -> str:
    if v <= close_thr:
        return "CLOSED"
    if v >= open_thr:
        return "OPEN"
    return "between"


def letterbox(h: int, w: int, target: int = PAD_TARGET) -> tuple[int, int, int, int]:
    """What resize_with_pad(img, target, target) produces: content h/w, then pad h/w.

    Mirrors lerobot/policies/common/vla_utils.py:219-243 -- ratio = max(w/W, h/H),
    then F.pad(..., (pad_w, 0, pad_h, 0)), i.e. padding on the LEFT and TOP.
    """
    ratio = max(w / target, h / target)
    rh, rw = int(h / ratio), int(w / ratio)
    return rh, rw, max(0, target - rh), max(0, target - rw)


def build_batch(runner: SmolVLARunner, instruction: str, observation: dict[str, Any]):
    """What SmolVLARunner.predict builds, minus the select_action call.

    Tensors stay UNBATCHED: eval relies on the preprocessor's add-batch-dimension
    step, so we hand it exactly the shapes it normally receives.
    """
    batch: dict[str, Any] = {
        runner.state_cfg["feature_key"]: runner._state_tensor(observation),
        runner.task_cfg["task_feature_key"]: instruction,
    }
    images = observation[runner.image_cfg["images_source_key"]]
    for camera_cfg in runner.camera_cfgs:
        batch[camera_cfg["feature_key"]] = runner._image_tensor(
            images[camera_cfg["source_name"]], camera_cfg["feature_key"]
        )
    return batch


def unnormalize_gripper(runner: SmolVLARunner, chunk: torch.Tensor) -> tuple[list[float], str]:
    """Raw-radian gripper column of the chunk.

    Primary path: the checkpoint's own postprocessor, applied row by row, because it
    is built for one action vector of shape (B, dim) -- the shape select_action
    returns -- not for a whole chunk.
    """
    try:
        out = []
        for k in range(chunk.shape[0]):
            a = runner.postprocessor(chunk[k : k + 1].clone())
            arr = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
            out.append(float(arr.reshape(-1)[L_GRIP]))
        return out, "postprocessor"
    except Exception as exc:  # noqa: BLE001
        print(f"      NOTE: postprocessor could not unnormalize ({type(exc).__name__}: {exc});"
              f" falling back to mean/std from meta/stats.json")
        return ([float(chunk[k, L_GRIP]) * L_GRIP_STD + L_GRIP_MEAN
                 for k in range(chunk.shape[0])], "mean/std fallback")


def reset_equalities(scene: EpisodeScene) -> None:
    """Clear every equality constraint, in the model AND in data.

    Tool-local workaround for EVAL_DIAGNOSIS.md B1, which is still unfixed in
    eval/scene.py:71-75. activate_grasp_weld writes model.eq_active0 as well as
    data.eq_active (ik_demo_utils.py:524), mj_resetData restores data.eq_active FROM
    eq_active0, and DomainRandomizer.reset never touches it -- so an episode that
    ends with the weld active corrupts the next seed's scene generation.

    Without this, the two passes of this probe cannot be compared at all: pass 1
    ends with the weld active, and pass 2 then starts with the plate already welded
    to the gripper. Run with --no-eq-reset to watch that happen on purpose.
    """
    for eid in range(int(scene.model.neq)):
        scene.model.eq_active0[eid] = 0
        scene.data.eq_active[eid] = 0


def eq_active(scene: EpisodeScene, name: str) -> int | None:
    try:
        import mujoco

        eid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eid < 0:
            return None
        return int(scene.data.eq_active[eid])
    except Exception:  # noqa: BLE001
        return None


def roll(scene, runner, grasp_cfg, cfg, seed, steps, instruction, action_repeat,
         plate_cfg, weld_name, stash_steps: set[int] | None, eq_reset: bool = True):
    """One rollout, reproducing eval/eval_smolvla.py:570-599 exactly."""
    if eq_reset:
        reset_equalities(scene)
    scene.reset(seed)
    runner.reset(seed)
    grasp = GraspAssist(model=scene.model, data=scene.data, cfg=grasp_cfg)
    grasp.reset()

    observation = scene.observe()
    trace: list[dict[str, Any]] = []
    stash: dict[int, dict[str, Any]] = {}

    for step in range(steps):
        if stash_steps and step in stash_steps:
            # mujoco's Renderer may hand back a reused buffer -> copy
            stash[step] = {
                "images": {k: np.array(v, copy=True)
                           for k, v in observation["images"].items()},
                "robot_state": np.array(observation["robot_state"], copy=True),
            }

        action = runner.predict(instruction, observation)
        if bool(cfg["action"]["clip_to_actuator_range"]):
            low = scene.model.actuator_ctrlrange[:, 0]
            high = scene.model.actuator_ctrlrange[:, 1]
            action = np.clip(action, low, high)
        scene.apply_action(action)
        grasp.update(action, step)
        for _ in range(action_repeat):
            scene.step()
        observation = scene.observe()

        trace.append({
            "step": step,
            "l_grip_cmd": float(action[L_GRIP]),
            "r_grip_cmd": float(action[R_GRIP]),
            "weld": eq_active(scene, weld_name),
            "plate_z": float(scene.body_xpos(plate_cfg["object_body"])[2]),
        })

    return trace, stash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smolvla_rollout.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="override model.checkpoint, e.g. a specific .../063250/pretrained_model")
    ap.add_argument("--seeds", default="6",
                    help="comma-separated eval seeds. 0, 6 and 7 are the only ones where the "
                         "plate is touched at all today, so they are the informative ones")
    ap.add_argument("--steps", type=int, default=420)
    ap.add_argument("--offsets", default="0,5,10,20,30,40",
                    help="probe at (first closed-command step + each offset)")
    ap.add_argument("--grid", default="240,280,320,360",
                    help="fallback probe steps when the gripper is never commanded closed")
    ap.add_argument("--noise-seed", type=int, default=12345,
                    help="fixed flow-matching noise, same default as tools/chunk_probe.py")
    ap.add_argument("--show-steps", type=int, default=16)
    ap.add_argument("--deterministic-render", action="store_true")
    ap.add_argument("--keep-shadows", action="store_true")
    ap.add_argument("--no-eq-reset", action="store_true",
                    help="do NOT clear equality constraints between rollouts, i.e. leave "
                         "EVAL_DIAGNOSIS.md B1 in place. Use it to demonstrate the leak")
    ap.add_argument("--release-open-hold", type=int, default=1,
                    help="value injected for grasp_assist.plate.release_open_hold_policy_steps, "
                         "which sim/grasp_assist.py:463 reads unconditionally but which is "
                         "missing from configs/smolvla_rollout.yaml. 1 reproduces the demo "
                         "behaviour (release requested on the first commanded-open step)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cfg = load_rollout_config(args.config)
    if args.checkpoint:
        cfg["model"]["checkpoint"] = args.checkpoint
    cfg["output"]["video"]["enabled"] = False

    # In-memory only -- nothing on disk is modified. sim/grasp_assist.py:456-464 reads
    # this key whenever a lifted plate is commanded open, and it is absent from BOTH
    # configs/smolvla_rollout.yaml and configs/grasp_assist.yaml. Since the
    # 2026-09-15 harness fix the key lives in configs/grasp_assist.yaml, so
    # this injection is normally a no-op and stays only as a safety net.
    missing_key = "release_open_hold_policy_steps"
    if missing_key not in cfg["grasp_assist"]["plate"]:
        cfg["grasp_assist"]["plate"][missing_key] = int(args.release_open_hold)
        print(f"NOTE: grasp_assist.plate.{missing_key} is missing from {args.config}; "
              f"injecting {args.release_open_hold} in memory only. "
              f"sim/grasp_assist.py:463 reads it unconditionally, so a real eval run "
              f"crashes with KeyError as soon as a lifted plate is commanded open. "
              f"This needs fixing in the config, not here.")

    scene = EpisodeScene(load_yaml(cfg["configs"]["sim"]), cfg["configs"]["randomization"])
    runner = SmolVLARunner(cfg)
    runner.validate(scene)
    if args.deterministic_render:
        make_render_deterministic(scene, disable_shadows=not args.keep_shadows)
        print("deterministic render: MSAA off, shadows "
              + ("kept" if args.keep_shadows else "off"))

    plate_cfg = cfg["grasp_assist"]["plate"]
    close_thr = float(plate_cfg["close_threshold"])
    open_thr = float(plate_cfg["open_threshold"])
    weld_name = plate_cfg["equality_name"]
    instruction = " ".join(cfg["task"]["instruction"].split())
    action_repeat = int(cfg["rollout"]["action_repeat"])

    # -------- geometry report: the reason this probe exists --------
    print()
    print("=== image geometry: what the policy actually receives ===")
    scene.reset(int(args.seeds.split(",")[0]))
    obs0 = scene.observe()
    geom: dict[str, Any] = {}
    for camera_cfg in runner.camera_cfgs:
        key = camera_cfg["feature_key"]
        src = np.asarray(obs0["images"][camera_cfg["source_name"]])
        t = runner._image_tensor(src, key)
        h, w = int(t.shape[1]), int(t.shape[2])
        rh, rw, ph, pw = letterbox(h, w)
        geom[camera_cfg["source_name"]] = {"rendered": [int(src.shape[0]), int(src.shape[1])],
                                           "fed": [h, w], "pad_top": ph, "pad_left": pw}
        print(f"  EVAL  {camera_cfg['source_name']:12} rendered {src.shape[0]}x{src.shape[1]}"
              f" -> fed {h}x{w} -> {PAD_TARGET}x{PAD_TARGET}: content {rh}x{rw},"
              f" {ph} pad rows on top, {pw} pad cols on left")
    for nm, (h, w) in (("overhead", (240, 320)), ("wrist_left", (120, 160)),
                       ("wrist_right", (120, 160))):
        rh, rw, ph, pw = letterbox(h, w)
        print(f"  TRAIN {nm:12} stored   {h}x{w}"
              f" -> fed {h}x{w} -> {PAD_TARGET}x{PAD_TARGET}: content {rh}x{rw},"
              f" {ph} pad rows on top, {pw} pad cols on left")
    print("  -> different pad rows means the policy sees the scene at a different")
    print("     vertical offset and scale than it trained on.")

    offsets = [int(v) for v in args.offsets.split(",") if v.strip()]
    grid = [int(v) for v in args.grid.split(",") if v.strip()]
    dump: dict[str, Any] = {"geometry": geom, "seeds": {}}

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        print()
        print("=" * 78)
        print(f"seed {seed}: pass 1 of 2, {args.steps} policy steps (trace only)")
        if not args.no_eq_reset:
            print("  equality constraints cleared before the roll (B1 workaround)")
        trace, _ = roll(scene, runner, cfg["grasp_assist"], cfg, seed, args.steps,
                        instruction, action_repeat, plate_cfg, weld_name, None,
                        eq_reset=not args.no_eq_reset)

        cmds = [t["l_grip_cmd"] for t in trace]
        closed_steps = [i for i, v in enumerate(cmds) if v <= close_thr]
        first_close = closed_steps[0] if closed_steps else None
        weld_steps = [t["step"] for t in trace if t["weld"]]

        print(f"  left-gripper command: {len(closed_steps)}/{len(cmds)} steps read CLOSED")
        print(f"  first CLOSED command at step: {first_close}")
        print("  weld active on steps: "
              + (f"{weld_steps[0]}..{weld_steps[-1]} ({len(weld_steps)} steps)"
                 if weld_steps else "never"))

        runs: list[int] = []
        if first_close is not None:
            cur = 0
            for v in cmds[first_close:]:
                if v <= close_thr:
                    cur += 1
                else:
                    if cur:
                        runs.append(cur)
                    cur = 0
            if cur:
                runs.append(cur)
            lo, hi = max(0, first_close - 3), min(len(cmds), first_close + 25)
            print(f"  executed command trace, steps {lo}..{hi - 1}:")
            print("    " + "  ".join(
                f"{i}:{cmds[i]:+.2f}{band(cmds[i], close_thr, open_thr)[0]}"
                for i in range(lo, hi)))
            print(f"  closed-command run lengths: {runs[:12]}{' …' if len(runs) > 12 else ''}")
            print(f"  -> demos hold closed for {DEMO_HOLD_FRAMES} consecutive frames; "
                  f"longest here is {max(runs) if runs else 0}")
        else:
            print("  the gripper is NEVER commanded closed -- the plate is never grasped, "
                  "so the carry phase does not exist in this rollout")

        probe_steps = ([first_close + o for o in offsets if first_close + o < args.steps]
                       if first_close is not None else [g for g in grid if g < args.steps])

        print(f"  pass 2 of 2: replaying seed {seed}, stashing steps {probe_steps}")
        trace2, stash = roll(scene, runner, cfg["grasp_assist"], cfg, seed, args.steps,
                             instruction, action_repeat, plate_cfg, weld_name, set(probe_steps),
                             eq_reset=not args.no_eq_reset)
        diff = [i for i in range(min(len(trace), len(trace2)))
                if abs(trace[i]["l_grip_cmd"] - trace2[i]["l_grip_cmd"]) > 1e-6]
        if diff:
            print(f"  WARNING: the two passes diverge from step {diff[0]} "
                  f"({len(diff)} differing steps). The rollout is not reproducible -- "
                  f"add --deterministic-render, and see EVAL_DIAGNOSIS.md B1/B2. "
                  f"The probes below still describe real observations, but which ones "
                  f"is not pinned down.")
        else:
            print("  the two passes match exactly -> the rollout is reproducible")

        seed_dump: dict[str, Any] = {"trace": trace, "probe_steps": probe_steps,
                                     "closed_runs": runs, "first_close": first_close,
                                     "reproducible": not diff}

        for ps in probe_steps:
            obs = stash[ps]
            processed = runner.preprocessor(build_batch(runner, instruction, obs))
            torch.manual_seed(args.noise_seed)
            torch.cuda.manual_seed_all(args.noise_seed)
            runner.policy.reset()
            with torch.inference_mode():
                chunk = runner.policy.predict_action_chunk(processed)
            chunk = chunk.detach().float().cpu()[0]
            g, space = unnormalize_gripper(runner, chunk)

            state_g = float(np.asarray(obs["robot_state"], dtype=np.float32)[L_GRIP])
            n_closed = sum(1 for v in g if v <= close_thr)
            first_open = next((k for k, v in enumerate(g) if v >= open_thr), None)

            print()
            print(f"  --- probe at rollout step {ps}"
                  + (f" (first close + {ps - first_close})" if first_close is not None else "")
                  + f" | executed cmd {cmds[ps]:+.4f}, weld={trace[ps]['weld']}"
                  + f", plate_z={trace[ps]['plate_z']:.4f}")
            print(f"      observation.state[{L_GRIP}] = {state_g:+.4f} "
                  f"-> {band(state_g, close_thr, open_thr)}   [{space}]")
            n_show = len(g) if args.show_steps == 0 else min(args.show_steps, len(g))
            print(f"      {'step':>4} {'pred':>9} {'band':>8}")
            for k in range(n_show):
                print(f"      {k:4} {g[k]:+9.4f} {band(g[k], close_thr, open_thr):>8}")
            print(f"      summary: {n_closed}/{len(g)} predicted steps read CLOSED; "
                  f"first step reading OPEN = "
                  f"{first_open if first_open is not None else 'none'}")
            print("      dataset-frame reference (tools/chunk_probe.py, frames 230/250): "
                  "50/50 CLOSED, first OPEN = none")

            seed_dump[str(ps)] = {"state_gripper": state_g, "executed_cmd": cmds[ps],
                                  "weld": trace[ps]["weld"], "plate_z": trace[ps]["plate_z"],
                                  "pred": g, "n_closed": n_closed, "first_open": first_open}

        dump["seeds"][str(seed)] = seed_dump

    if args.json:
        Path(args.json).write_text(json.dumps(dump, indent=2), encoding="utf-8")
        print(f"\njson -> {args.json}")

    print()
    print("How to read this:")
    print("  On dataset frames the policy predicts 50/50 CLOSED through the carry.")
    print("  If these rollout probes also hold closed, the gripper is fine in closed loop")
    print("  too and the failure is upstream: the arm not reaching the plate.")
    print("  If they flip open, the eval-time input is the problem, and the geometry")
    print("  table at the top of this output is the first suspect.")


if __name__ == "__main__":
    main()
