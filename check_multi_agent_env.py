"""
Step 8 - Sanity check of the three-agent environment (not a result).
================================================================================
ROMAN URDU: Har department agent apne tasks ke liye ek fixed rule (FCFS ya
ECT) chalata hai -- agents ek doosre se independent hain (roadmap: "agents
may initially act independently"). Har agent ka apna return aur team return
(= Step 6 ka return) print hota hai, seeds 0-29 ka mean.

Run:  py check_multi_agent_env.py --config configs/scenarios/overload.yaml
"""

import argparse
import os

from agents.heuristics import ect_action
from env.multi_agent_env import MultiAgentHospitalEnv

ROOT = os.path.dirname(os.path.abspath(__file__))
POLICIES = {"FCFS": lambda env: env.inner.fcfs_action(),
            "ECT": lambda env: ect_action(env.inner)}


def run(config, seeds):
    env = MultiAgentHospitalEnv(config)
    table = {}
    for name, policy in POLICIES.items():
        totals = {a: 0.0 for a in env.agents}
        for seed in seeds:
            env.reset(seed=seed)
            while not env.done:
                _, _, info = env.step(policy(env))
            for a in env.agents:
                totals[a] += info["episode_returns"][a] / len(seeds)
        table[name] = totals
    return env.agents, table


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 8 sanity check: per-agent returns of fixed rules")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "environment.yaml"))
    args = parser.parse_args(argv)
    agents, table = run(args.config, range(30))
    print(f"{os.path.basename(args.config)} - mean return over seeds 0-29")
    print(f"{'policy':8s}" + "".join(f"{a:>14s}" for a in agents) + f"{'team':>10s}")
    for name, totals in table.items():
        print(f"{name:8s}" + "".join(f"{totals[a]:14.1f}" for a in agents) + f"{sum(totals.values()):10.1f}")


if __name__ == "__main__":
    main()
