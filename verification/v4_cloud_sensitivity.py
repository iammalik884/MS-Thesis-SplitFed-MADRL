"""V4 - Limited cloud-capacity and cloud-delay sensitivity of the FROZEN priority-aware MAPPO (W5)
and the baselines. No retraining. VALIDATION seeds 1000-1029 only (test seeds untouched).
One factor at a time; the configs are changed in memory only, nothing in configs/ is modified."""
import glob
import json
import statistics

import torch

from common import OUT, mean_ci, write, SCENARIOS
from agents.policy_eval import baseline_policies, evaluate
from env.hospital_env import load_config
from evaluate_dqn_test import load_policy
from train_dqn_pilot import VALIDATION_SEEDS
from train_mappo import violation_by_priority

torch.set_num_threads(1)
BASELINES = ("FCFS", "ECT", "always Cloud", "always Local")
data = {}
for sc, path in SCENARIOS.items():
    d0 = load_config(path)["offloading"]["cloud_network_delay"]
    settings = [(f"cloud capacity {c}", {"cloud_capacity": float(c)}) for c in (50, 25, 12, 6)] + \
               [(f"cloud delay {d}", {"cloud_network_delay": d}) for d in (d0 + 2, d0 + 4)]
    runs = sorted(glob.glob(f"results/step10_final/training_runs/FINAL_{sc}_seed*"))
    for label, change in settings:
        cfg = load_config(path)
        cfg["offloading"].update(change)
        policies = [(f"W5 trainseed{i}", load_policy(r, "best")[1]) for i, r in enumerate(runs)]
        policies += [(b, p) for b, p in baseline_policies(random_seed=0).items() if b in BASELINES]
        row = {}
        for name, policy in policies:
            res = evaluate(cfg, policy, VALIDATION_SEEDS, keep_reports=True)
            waits = [s["report"]["final"]["by_destination"].get("Cloud", {}).get("avg_queue_wait") for s in res["per_seed"]]
            waits = [x for x in waits if x is not None]
            row[name] = {"return": res["mean_return"], "critical_violation": violation_by_priority(res)["Critical"]["violation_rate"],
                         "cloud_share": res["action_proportions"]["Cloud"], "edge_share": res["action_proportions"]["Edge"],
                         "cloud_queue_wait": statistics.mean(waits) if waits else 0.0}
        data[f"{sc} | {label}"] = row
        print("done", sc, label, flush=True)
json.dump(data, open(f"{OUT}/v4_cloud_sensitivity.json", "w", encoding="utf-8"), indent=1)

lines = ["# V4 - Cloud capacity / delay sensitivity (frozen W5 MAPPO, validation seeds 1000-1029)", "",
         "Current values: cloud capacity 50; cloud delay 2 (default, overload) and 6 (slow network). W5 = mean ± 95% CI "
         "over 5 training seeds; critical violation pooled over seeds. No model was retrained for these settings.", "",
         "| Scenario / setting | W5 return | W5 Critical viol. | W5 Cloud / Edge share | Cloud queue wait | "
         + " | ".join(BASELINES) + " |",
         "|---|---|---|---|---|---|---|---|---|"]
for key, row in data.items():
    w = lambda k: mean_ci([row[f"W5 trainseed{i}"][k] for i in range(5)])
    r, c, cl, e, q = w("return"), w("critical_violation"), w("cloud_share"), w("edge_share"), w("cloud_queue_wait")
    lines.append(f"| {key.replace(' | ', ' — ')} | {r[0]:.1f} ± {r[2]:.1f} | {100 * c[0]:.2f}% | {100 * cl[0]:.1f}% / {100 * e[0]:.1f}% | {q[0]:.2f} | "
                 + " | ".join(f"{row[b]['return']:.1f}" for b in BASELINES) + " |")
write("v4_cloud_sensitivity.md", "\n".join(lines) + "\n")
