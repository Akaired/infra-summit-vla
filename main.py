"""
Entry point: python main.py [--seeds 0 1 2] [--instruction "..."]

Runs eval/loop.py's cycle with the current policy (DummyPolicy until a real
one is wired in -- see eval/policy_interface.py).
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from sim.env import BimanualTableEnv  # noqa: E402
from eval.policy_interface import DummyPolicy  # noqa: E402
from eval.loop import run_eval  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--instruction", type=str,
                         default="open the drawer and set the table")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--log-dir", type=str, default="eval/logs")
    args = parser.parse_args()

    env = BimanualTableEnv()
    policy = DummyPolicy()

    results = run_eval(env, policy, args.seeds, args.instruction,
                        args.max_steps, args.log_dir)

    n_ok = sum(1 for r in results if not r["stopped_early"])
    print(f"\n[main] {n_ok}/{len(results)} episodes completed without error.")
    print(f"[main] logs written to {args.log_dir}/")


if __name__ == "__main__":
    main()
