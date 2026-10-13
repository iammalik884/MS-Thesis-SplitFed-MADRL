"""
Step 7 - Run the pre-registered DQN debugging experiments in parallel.
================================================================================
ROMAN URDU: Har experiment train_dqn_pilot.py ko ZYADA SE ZYADA EK --set
override ke saath chalata hai (Step7_DQN_Debugging.md ka plan). Har
(experiment, scenario, training seed) ka alag run folder banta hai:
results/runs/<group>/<experiment>/...

ENGLISH: e.g.
  py run_dqn_experiments.py --group phase1 --exp E0= --exp E1=reward_scale=0.1 --seeds 0 1
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = {"default": "configs/environment.yaml",
             "overload": "configs/scenarios/overload.yaml",
             "slow_network": "configs/scenarios/slow_network.yaml"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--exp", action="append", required=True,
                        help="NAME=OVERRIDE (one KEY=VALUE, or empty for the reference)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)

    jobs = []
    for spec in args.exp:
        name, _, override = spec.partition("=")
        if override.count("=") > 1:
            raise SystemExit(f"{name}: only ONE override per experiment, got {override!r}")
        for scenario in args.scenarios:
            for seed in args.seeds:
                cmd = [sys.executable, os.path.join(ROOT, "train_dqn_pilot.py"), "--config", SCENARIOS[scenario],
                       "--train-seed", str(seed), "--name", f"{name}_{scenario}_seed{seed}",
                       "--out-dir", os.path.join(ROOT, "results", "runs", args.group, name)]
                if override:
                    cmd += ["--set", override]
                jobs.append((f"{name}/{scenario}/seed{seed}", cmd))

    def run(job):
        label, cmd = job
        done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        status = "ok" if done.returncode == 0 else f"FAILED ({done.returncode})\n{done.stderr[-2000:]}"
        print(f"{label}: {status}", flush=True)
        return done.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, jobs))
    if any(codes):
        raise SystemExit("some runs failed")


if __name__ == "__main__":
    main()
