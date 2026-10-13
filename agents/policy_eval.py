"""
Step 7 - One evaluation routine for every policy (DQN and all baselines).
================================================================================
ROMAN URDU: Har policy ko BILKUL ek hi tareeqe se evaluate kiya jata hai:
same config, same workload seeds, Step 6 environment (jo andar Step 5 ka
window + drain protocol chalata hai). Report hota hai: UNDISCOUNTED episode
return, protocol metrics, aur action proportions (overall, priority-wise,
source-department-wise). Discounted return kabhi performance ke taur par
report nahi hota (design note, Section 4).

ENGLISH: evaluate(config, policy, seeds) plays one episode per seed with
policy(env, obs) -> action and returns per-seed rows plus means and action
proportions.
"""

import statistics
from collections import Counter
from typing import Callable, Dict, Iterable

import numpy as np

from agents.heuristics import ect_action
from env.gym_env import ACTIONS, HospitalSchedulingEnv

METRIC_KEYS = ("avg_latency", "throughput_in_window", "deadline_violation_rate",
               "critical_deadline_violation_rate", "unfinished_after_drain")


def baseline_policies(random_seed: int = 0) -> Dict[str, Callable]:
    """Fixed baselines from the design note (no parameters to tune)."""
    rng = np.random.default_rng(random_seed)
    return {
        "FCFS": lambda env, obs: env.fcfs_action(),
        "always Local": lambda env, obs: 0,
        "always Edge": lambda env, obs: 1,
        "always Cloud": lambda env, obs: 2,
        "random": lambda env, obs: int(rng.integers(0, 3)),
        "ECT": lambda env, obs: ect_action(env),
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def evaluate(config, policy: Callable, seeds: Iterable[int]) -> Dict:
    env = HospitalSchedulingEnv(config)
    rows, overall = [], Counter()
    by_priority, by_source = Counter(), Counter()
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        terminated = False
        if info["no_tasks_in_episode"]:          # nothing to decide; one step ends the episode
            obs, _, terminated, _, info = env.step(0)
        while not terminated:
            arrival = env._pending[0]
            action = int(policy(env, obs))
            overall[ACTIONS[action]] += 1
            by_priority[(arrival.priority, ACTIONS[action])] += 1
            by_source[(arrival.department, ACTIONS[action])] += 1
            obs, _, terminated, _, info = env.step(action)
        headline = info["report"]["headline"]
        rows.append({"seed": seed, "return": info["episode_return"], **{k: headline[k] for k in METRIC_KEYS}})
    n = sum(overall.values())

    def proportions(counter, groups):
        out = {}
        for g in groups:
            total = sum(v for (k, _), v in counter.items() if k == g)
            out[g] = {a: (counter[(g, a)] / total if total else None) for a in ACTIONS}
        return out

    return {
        "seeds": [r["seed"] for r in rows],
        "mean_return": _mean([r["return"] for r in rows]),
        "std_return": statistics.stdev([r["return"] for r in rows]) if len(rows) > 1 else 0.0,
        "means": {k: _mean([r[k] for r in rows]) for k in METRIC_KEYS},
        "action_proportions": {a: (overall[a] / n if n else None) for a in ACTIONS},
        "action_proportions_by_priority": proportions(by_priority, sorted({k for k, _ in by_priority})),
        "action_proportions_by_source": proportions(by_source, sorted({k for k, _ in by_source})),
        "per_seed": rows,
    }
