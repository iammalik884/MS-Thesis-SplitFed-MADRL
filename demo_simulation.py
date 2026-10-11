"""
Live simulation demo - Steps 1-5 (Poisson arrivals, Local / Edge(neighbour) /
Cloud routing, priority deadlines, V2 evaluation protocol).
================================================================================
ROMAN URDU: Yeh script ek run chala kar har task ki poori lifecycle (kahan se
aaya, kahan gaya, kab ready hua, kab shuru, kab khatam) aur phir summary
print karti hai. Koi nayi logic nahi, koi hard-coded number nahi -- sab kuch
env/evaluation.py ke common protocol se aata hai. Kuch save nahi karti.

ENGLISH: Runs one FCFS episode under the common protocol and prints every
task's lifecycle plus the summary. Nothing is saved (use
run_baseline_fcfs.py for saved, provenance-tagged results).

Run with:  py demo_simulation.py [--config PATH] [--seed N]   (from MS_Thesis_Project/)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.evaluation import run_fcfs_evaluation
from env.hospital_env import load_config
from run_baseline_fcfs import CONFIG_PATH, print_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print one FCFS episode task by task")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seed = config.get("seed", 42) if args.seed is None else args.seed

    result = run_fcfs_evaluation(config, seed=seed)
    print("id  | source -> destination (tier)          | prio     | demand | arr ready start done | deadline")
    for t in result.simulator.tasks:
        route = f"{t.source_department} -> {t.destination} ({t.tier})"
        times = f"{t.arrival_time:>3} {t.ready_time:>5} {str(t.start_time):>5} {str(t.completion_time):>4}"
        print(f"{t.task_id:03d} | {route:<38} | {str(t.priority):<8} | {t.compute_demand:6.2f} | {times} | {t.deadline}")
    print()
    print_report(result.report, seed, os.path.basename(args.config))


if __name__ == "__main__":
    main()
