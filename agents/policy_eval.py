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
import torch

from agents.heuristics import ect_action
from env.gym_env import ACTIONS, HospitalSchedulingEnv

METRIC_KEYS = ("avg_latency", "throughput_in_window", "deadline_violation_rate",
               "critical_deadline_violation_rate", "unfinished_after_drain")
COMM_KEYS = ("neighbour_transfers", "cloud_transfers", "synthetic_payload_units_total")


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


def evaluate(config, policy: Callable, seeds: Iterable[int], keep_reports: bool = False) -> Dict:
    """keep_reports=True also stores each seed's full protocol report."""
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
        report = info["report"]
        comm = report["final"]["offloading_communication"]
        row = {"seed": seed, "return": info["episode_return"], **{k: report["headline"][k] for k in METRIC_KEYS},
               "local_tasks_no_transfer": comm["local_tasks_no_transfer"],
               "neighbour_transfers": comm["neighbour_transfers"], "cloud_transfers": comm["cloud_transfers"],
               "synthetic_payload_units_total": comm["synthetic_payload_units"]["total"]}
        if keep_reports:
            row["report"] = report
        rows.append(row)
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
        "means": {k: _mean([r[k] for r in rows]) for k in METRIC_KEYS + COMM_KEYS},
        "action_proportions": {a: (overall[a] / n if n else None) for a in ACTIONS},
        "action_proportions_by_priority": proportions(by_priority, sorted({k for k, _ in by_priority})),
        "action_proportions_by_source": proportions(by_source, sorted({k for k, _ in by_source})),
        "per_seed": rows,
    }


def q_vs_monte_carlo(config, agent, seeds: Iterable[int]) -> Dict:
    """Play the agent's GREEDY policy and compare, at every decision, its
    Q(s, a) with the discounted Monte Carlo return actually obtained from
    that decision on: G_k = sum_j gamma^(j-k) * reward_scale * r_j, with the
    same gamma and reward scale the agent is trained with. Bias = Q - G.
    Values are also reported divided by reward_scale (unscaled latency-steps)."""
    gamma, scale = agent.config.gamma, agent.config.reward_scale
    env = HospitalSchedulingEnv(config)
    q_all, g_all, q_first, g_first = [], [], [], []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        if info["no_tasks_in_episode"]:
            continue
        qs, rewards, terminated = [], [], False
        while not terminated:
            q = agent.q_values(obs)
            action = int(torch.argmax(q).item())
            qs.append(float(q[action]))
            obs, reward, terminated, _, _ = env.step(action)
            rewards.append(scale * reward)
        returns, g = [], 0.0
        for r in reversed(rewards):
            g = r + gamma * g
            returns.append(g)
        returns.reverse()
        q_all += qs
        g_all += returns
        q_first.append(qs[0])
        g_first.append(returns[0])
    q_all, g_all = np.array(q_all), np.array(g_all)
    bias = q_all - g_all
    return {
        "gamma": gamma, "reward_scale": scale, "decisions": int(len(q_all)),
        "mean_q": float(q_all.mean()), "mean_mc_return": float(g_all.mean()),
        "mean_bias": float(bias.mean()), "rmse": float(np.sqrt((bias ** 2).mean())),
        "correlation": float(np.corrcoef(q_all, g_all)[0, 1]) if len(q_all) > 1 else None,
        "unscaled": {"mean_q": float(q_all.mean() / scale), "mean_mc_return": float(g_all.mean() / scale),
                     "mean_bias": float(bias.mean() / scale), "rmse": float(np.sqrt((bias ** 2).mean()) / scale)},
        "first_decision": {"mean_q": float(np.mean(q_first)), "mean_mc_return": float(np.mean(g_first))},
    }
