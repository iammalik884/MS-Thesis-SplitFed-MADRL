"""
Step 6 - Sanity check: simple fixed policies played through the Gymnasium environment.
================================================================================
ROMAN URDU: Yeh training NAHI hai aur thesis ka result bhi nahi. Sirf yeh
dekhne ke liye ke environment ka reward policy ke faislon par sahi tarah
react karta hai: FCFS, "hamesha Local / Edge / Cloud", random policy aur
proposed ECT heuristic ko same seeds par chala kar average return, latency
aur Critical violation print karta hai (evaluation protocol wahi Step 5 wala).

ENGLISH: plays FCFS, always-Local, always-Edge, always-Cloud, a seeded
random policy and the proposed ECT heuristic (agents/heuristics.py) through
env/gym_env.py on the same seeds and prints the mean episode return,
average latency and Critical deadline-violation rate.

Run:  py check_simple_policies.py [--config configs/scenarios/overload.yaml] [--seeds 0-29]
"""

import argparse
import os
import statistics

import numpy as np

from agents.heuristics import ect_action
from env.gym_env import HospitalSchedulingEnv

ROOT = os.path.dirname(os.path.abspath(__file__))


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def run(config_path: str, seeds):
    env = HospitalSchedulingEnv(config_path)
    policies = {
        "FCFS": lambda e, rng: e.fcfs_action(),
        "always Local": lambda e, rng: 0,
        "always Edge": lambda e, rng: 1,
        "always Cloud": lambda e, rng: 2,
        "random": lambda e, rng: int(rng.integers(0, 3)),
        "ECT heuristic": lambda e, rng: ect_action(e),
    }
    rows = []
    for name, policy in policies.items():
        returns, latencies, critical = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)  # only used by the random policy
            _, info = env.reset(seed=seed)
            terminated = info["no_tasks_in_episode"]
            if terminated:  # nothing to decide: one step ends the episode
                _, _, _, _, info = env.step(0)
            while not terminated:
                _, _, terminated, _, info = env.step(policy(env, rng))
            headline = info["report"]["headline"]
            returns.append(info["episode_return"])
            latencies.append(headline["avg_latency"])
            critical.append(headline["critical_deadline_violation_rate"])
        rows.append((name, _mean(returns), _mean(latencies), _mean(critical)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 6 sanity check with simple fixed policies")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "environment.yaml"))
    parser.add_argument("--seeds", default="0-29", help="range like 0-29")
    args = parser.parse_args(argv)
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = range(lo, hi + 1)
    print(f"{os.path.basename(args.config)}  seeds {lo}-{hi}  (mean over seeds)")
    print(f"{'policy':14s} {'return':>10s} {'avg latency':>12s} {'Critical violation':>19s}")
    for name, ret, lat, crit in run(args.config, seeds):
        fmt = lambda v, f: "None" if v is None else f.format(v)
        print(f"{name:14s} {fmt(ret, '{:10.1f}'):>10s} {fmt(lat, '{:12.2f}'):>12s} {fmt(crit, '{:18.1%}'):>19s}")


if __name__ == "__main__":
    main()
