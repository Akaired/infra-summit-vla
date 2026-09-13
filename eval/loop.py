"""
The eval cycle: reset -> (observe -> policy -> act) x N -> log.
main.py at repo root runs this. One episode per seed; seeds should be
configs/randomization.yaml `seeds` (the 10 reserved eval seeds) for a real
submission run, but this accepts any seed list for quick iteration too.
"""
import json
import os
import time
import numpy as np


def run_episode(env, policy, seed: int, instruction: str, max_steps: int,
                 log_dir: str = None) -> dict:
    t0 = time.time()
    obs = env.reset(seed)

    log = {"seed": seed, "instruction": instruction, "steps": []}
    stopped_early = False

    for step_idx in range(max_steps):
        action = policy.predict(obs, instruction)
        obs, info = env.step(action)

        log["steps"].append({
            "step": step_idx,
            "drawer_slide": info["drawer_slide"],
        })

        if info["any_nan"]:
            log["stopped_early"] = f"non-finite state at step {step_idx}"
            stopped_early = True
            break

    log["stopped_early"] = log.get("stopped_early", False)
    log["final_joint_pos"] = obs["joint_pos"].tolist()
    log["final_drawer_slide"] = obs["drawer_slide"]
    log["wall_time_s"] = round(time.time() - t0, 3)
    # Task-outcome detection (did it actually complete the instruction) is
    # not implemented yet -- that needs a real policy and a defined success
    # criterion, both later work. Marked explicitly rather than faked.
    log["outcome"] = "not_yet_implemented"

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"episode_seed{seed}.json")
        with open(path, "w") as f:
            json.dump(log, f, indent=2)

    return log


def run_eval(env, policy, seeds: list, instruction: str, max_steps: int,
             log_dir: str = None) -> list:
    results = []
    for seed in seeds:
        print(f"[eval] seed {seed} ...")
        log = run_episode(env, policy, seed, instruction, max_steps, log_dir)
        status = "OK" if not log["stopped_early"] else f"STOPPED: {log['stopped_early']}"
        print(f"[eval] seed {seed}: {status}  "
              f"(drawer_slide={log['final_drawer_slide']:.3f}, "
              f"{log['wall_time_s']}s)")
        results.append(log)
    return results
