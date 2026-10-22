"""V5 - Priority weights and per-priority latency / deadline-violation results on the test seeds.
MAPPO W5, MAPPO W1 and the baselines come from the committed Step 10 evaluation. The DQN has no
committed per-priority numbers, so its frozen best checkpoints are re-evaluated on the same test
seeds (deterministic; the same models and seeds as results/step7_final, no tuning)."""
import glob
import statistics

import torch

from common import load_summary, mean_ci, write, SCENARIOS
from agents.policy_eval import evaluate
from evaluate_dqn_test import load_policy
from train_mappo import violation_by_priority

torch.set_num_threads(1)
P = ("Critical", "High", "Routine")
W5 = load_summary("results/step10_final/summary.json")
W1 = load_summary("results/step10_final/step9_reference/summary.json")
lines = ["# V5 - Priority weights and per-priority results (test seeds 2000-2029)", "",
         "Weights from the deadline offsets (Critical 2, High 5, Routine 10 steps): W1 = 1/1/1, W5 = 5/2/1 "
         "(urgency = 10 / offset), W25 = 25/4/1 (urgency squared). Reward per task = w_p x (-latency - 10 x miss). "
         "W5 was chosen on VALIDATION seeds by a rule committed before the runs (mean Critical violation: "
         "W1 3.380%, W5 0.141%, W25 0.246%; see Step_Documentation/Step10_Priority_Reward.md).", "",
         "Violation = (late + overdue) / decided tasks of that priority, pooled over the 30 test seeds of a run; "
         "latency in steps over completed tasks. Learned methods: mean ± 95% CI over 5 training seeds.", ""]


def mappo(S, sc):
    pb = S["priority_breakdown"][sc]
    labels = [k for k in pb if k.startswith("MAPPO_best")]
    return {p: (mean_ci([pb[l]["violation"][p]["violation_rate"] for l in labels]),
                mean_ci([pb[l]["latency"][p]["avg_latency"] for l in labels])) for p in P}


for sc, path in SCENARIOS.items():
    dqn = {p: ([], []) for p in P}
    for run in sorted(glob.glob(f"results/step7_final/training_runs/*_{sc}_seed*")):
        res = evaluate(path, load_policy(run, "best")[1], range(2000, 2030), keep_reports=True)
        vb = violation_by_priority(res)
        for p in P:
            dqn[p][0].append(vb[p]["violation_rate"])
            dqn[p][1].append(res["latency_by_priority"][p]["avg_latency"])
    rows = {"Priority MAPPO (W5)": mappo(W5, sc), "MAPPO (W1, Step 9)": mappo(W1, sc),
            "DQN (Step 7)": {p: (mean_ci(dqn[p][0]), mean_ci(dqn[p][1])) for p in P}}
    for b in ("ECT", "FCFS", "always Cloud"):
        x = W5["priority_breakdown"][sc][b]
        rows[b] = {p: ((x["violation"][p]["violation_rate"], None, None), (x["latency"][p]["avg_latency"], None, None)) for p in P}
    f = lambda v, pct: (f"{100 * v[0]:.2f}%" if pct else f"{v[0]:.2f}") + \
        ("" if v[2] is None else (f" ± {100 * v[2]:.2f}" if pct else f" ± {v[2]:.2f}"))
    lines += [f"## {sc}", "", "| Method | Critical: violation / latency | High: violation / latency | Routine: violation / latency |",
              "|---|---|---|---|"]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{f(r[p][0], True)} / {f(r[p][1], False)}" for p in P) + " |")
    lines.append("")
write("v5_priority_results.md", "\n".join(lines))
