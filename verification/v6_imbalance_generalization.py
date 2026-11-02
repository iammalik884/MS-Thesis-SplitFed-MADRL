"""V6 (Phase 0) - Frozen policies on department_imbalance (ICU arrival rates x6, neighbours lightly
loaded). No model was trained on this scenario, so every learned result here is ZERO-SHOT
GENERALIZATION. VALIDATION seeds 1000-1029 only, paired across policies; best checkpoints."""
import glob
import json
import os
import statistics
from collections import Counter

import torch

from common import OUT, ROOT, mean_ci, write
from agents.heuristics import estimated_latencies
from agents.policy_eval import baseline_policies, evaluate
from env.gym_env import ACTIONS, HospitalSchedulingEnv
from env.multi_agent_env import MultiAgentHospitalEnv
from evaluate_dqn_test import load_policy
from train_dqn_pilot import TEST_SEEDS, VALIDATION_SEEDS
from train_mappo import destination_load, violation_by_priority

torch.set_num_threads(1)
CONFIG = "configs/scenarios/department_imbalance.yaml"
assert not set(VALIDATION_SEEDS) & set(TEST_SEEDS)
DEPTS = ("Emergency", "ICU", "General_Ward")
GROUPS = {"MAPPO W5 (Step 10)": "results/step10_final/training_runs", "MAPPO W1 (Step 9)": "results/step9_final/training_runs",
          "DQN (Step 7)": "results/step7_final/training_runs"}


def compatible(run):
    m = json.load(open(os.path.join(run, "manifest.json")))
    a, b = MultiAgentHospitalEnv(CONFIG), MultiAgentHospitalEnv(m["config_path"])
    return (a.agents == b.agents and a.state_space.shape == b.state_space.shape
            and all(a.observation_spaces[x].shape == b.observation_spaces[x].shape for x in a.agents)
            and list(a.inner.capacities) == list(b.inner.capacities) and a.inner.cloud_capacity == b.inner.cloud_capacity)


def edge_feasibility(policy):
    """Along the policy's own trajectories, ICU decisions only: how often the ECT estimate (decision-time
    information) says Edge is strictly the fastest option, and what the policy chose then."""
    env, chosen, n, strict = HospitalSchedulingEnv(CONFIG), Counter(), 0, 0
    for seed in VALIDATION_SEEDS:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            est, src = estimated_latencies(env), env._pending[0].department
            a = int(policy(env, obs))
            if src == "ICU":
                n += 1
                if est[1] < min(est[0], est[2]):
                    strict += 1
                    chosen[ACTIONS[a]] += 1
            obs, _, done, _, _ = env.step(a)
    return strict / n, {k: round(v / strict, 2) for k, v in chosen.items()} if strict else {}


def summary(res):
    return {"per_seed": [r["return"] for r in res["per_seed"]], "return": res["mean_return"],
            "critical": violation_by_priority(res)["Critical"]["violation_rate"],
            "icu_shares": res["action_proportions_by_source"]["ICU"], "load": destination_load(res)}


data = {"baselines": {}, "learned": {}}
for name, policy in baseline_policies(random_seed=0).items():
    data["baselines"][name] = summary(evaluate(CONFIG, policy, VALIDATION_SEEDS, keep_reports=True))
    if name in ("ECT", "FCFS"):
        data["baselines"][name]["edge_feasibility"] = edge_feasibility(baseline_policies(random_seed=0)[name])
for group, folder in GROUPS.items():
    for src in ("default", "overload", "slow_network"):
        for run in sorted(glob.glob(os.path.join(folder, f"*_{src}_seed*"))):
            assert compatible(run), run
            _, policy = load_policy(run, "best")
            s = summary(evaluate(CONFIG, policy, VALIDATION_SEEDS, keep_reports=True))
            if group.startswith("MAPPO W5"):
                s["edge_feasibility"] = edge_feasibility(policy)
            data["learned"].setdefault(f"{group} | trained on {src}", []).append(s)
        print("done", group, src, flush=True)
json.dump(data, open(f"{OUT}/v6_imbalance_generalization.json", "w", encoding="utf-8"), indent=1)

B = data["baselines"]
lines = ["# V6 (Phase 0) - department_imbalance, zero-shot generalization (validation seeds 1000-1029)", "",
         "No model was trained on department_imbalance (each model was trained on one of default / overload / "
         "slow network). Imbalance differs from default only in the ICU arrival rates (x6). All 45 checkpoints "
         "passed the compatibility check (observation, state, action sizes, agents, capacities). Learned methods: "
         "mean ± 95% CI over 5 training seeds; paired differences: per workload seed, averaged over training seeds, "
         "t with 29 df (positive = better than the baseline).", "",
         "| Policy | Return | Paired vs FCFS | Paired vs ECT | Critical viol. | ICU: Local / Edge / Cloud | Queue wait E / ICU / GW |",
         "|---|---|---|---|---|---|---|"]


def paired(per_seed_lists, base):
    avg = [statistics.mean(x[i] for x in per_seed_lists) for i in range(len(base))]
    m, _, h = mean_ci([a - b for a, b in zip(avg, base)])
    return f"{m:+.1f} ± {h:.1f}"


for name in ("FCFS", "ECT", "always Cloud", "always Local", "random"):
    s = B[name]
    sh = s["icu_shares"]
    waits = " / ".join(f"{(s['load'].get(d, {}).get('mean_avg_queue_wait') or 0):.2f}" for d in DEPTS)
    lines.append(f"| {name} | {s['return']:.1f} | — | — | {100 * s['critical']:.2f}% | "
                 f"{100 * sh['Local']:.0f} / {100 * sh['Edge']:.0f} / {100 * sh['Cloud']:.0f}% | {waits} |")
for key, runs in data["learned"].items():
    r = mean_ci([x["return"] for x in runs])
    c = statistics.mean(x["critical"] for x in runs)
    sh = {a: mean_ci([x["icu_shares"][a] for x in runs]) for a in ACTIONS}
    waits = " / ".join(f"{statistics.mean((x['load'].get(d, {}).get('mean_avg_queue_wait') or 0) for x in runs):.2f}" for d in DEPTS)
    lines.append(f"| {key.replace(' | ', ' — ')} | {r[0]:.1f} ± {r[2]:.1f} | {paired([x['per_seed'] for x in runs], B['FCFS']['per_seed'])} | "
                 f"{paired([x['per_seed'] for x in runs], B['ECT']['per_seed'])} | {100 * c:.2f}% | "
                 f"{100 * sh['Local'][0]:.0f} / {100 * sh['Edge'][0]:.1f} ± {100 * sh['Edge'][2]:.1f} / {100 * sh['Cloud'][0]:.0f}% | {waits} |")
lines += ["", "Edge feasibility on ICU decisions (ECT estimate, decision-time information; myopic, not ground truth):", ""]
for name in ("ECT", "FCFS"):
    f = B[name]["edge_feasibility"]
    lines.append(f"- {name}: Edge strictly fastest on {100 * f[0]:.1f}% of ICU decisions; chosen then: {f[1]}")
for key, runs in data["learned"].items():
    if key.startswith("MAPPO W5"):
        frac = statistics.mean(x["edge_feasibility"][0] for x in runs)
        lines.append(f"- {key}: Edge strictly fastest on {100 * frac:.1f}% of ICU decisions (mean over 5 seeds)")
write("v6_imbalance_generalization.md", "\n".join(lines) + "\n")
