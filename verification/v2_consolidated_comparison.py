"""V2 - FCFS, ECT, DQN, MAPPO and priority-aware MAPPO on identical test workloads (seeds 2000-2029)
and identical metrics, read from the committed one-time test evaluations."""
from common import load_summary, write

S7 = load_summary("results/step7_final/summary.json")
S9 = load_summary("results/step9_final/summary.json")
S10 = load_summary("results/step10_final/summary.json")
assert S7["evaluation_seeds"] == S9["evaluation_seeds"] == S10["evaluation_seeds"] == [2000, 2029]
BASE = ("FCFS", "ECT", "always Local", "always Edge", "always Cloud", "random")
for sc in S7["results"]:
    assert all(S7["results"][sc][b] == S9["results"][sc][b] == S10["results"][sc][b] for b in BASE), sc

ROWS = [("FCFS", S7, "FCFS"), ("ECT", S7, "ECT"), ("always Cloud", S7, "always Cloud"),
        ("DQN (Step 7, 40k decisions)", S7, "DQN (best)"), ("MAPPO (Step 9, 120k)", S9, "MAPPO (best)"),
        ("Priority MAPPO W5 (Step 10, 120k)", S10, "MAPPO (best)")]


def fmt(v, pct=False):
    m, h = v
    if pct:
        return f"{100 * m:.1f}%" + ("" if h is None else f" ± {100 * h:.1f}")
    d = 1 if abs(m) >= 10 else 2
    return f"{m:.{d}f}" + ("" if h is None else f" ± {h:.{d}f}")


lines = ["# V2 - Consolidated comparison (test seeds 2000-2029, best checkpoints)", "",
         "Baseline results are identical in the Step 7, 9 and 10 evaluations (checked). Learned methods: "
         "mean ± 95% CI over 5 training seeds (t, df 4). Throughput = tasks completed per step in the window.", ""]
for sc in ("default", "overload", "slow_network"):
    lines += [f"## {sc}", "", "| Method | Return | Avg latency | Critical violation | All-task violation | Throughput | "
              "Local / Edge / Cloud | Payload units |", "|---|---|---|---|---|---|---|---|"]
    for name, S, key in ROWS:
        r = S["results"][sc][key]
        sh = r["action_shares"]
        lines.append(f"| {name} | {fmt(r['return'])} | {fmt(r['avg_latency'])} | {fmt(r['critical_deadline_violation_rate'], True)} | "
                     f"{fmt(r['deadline_violation_rate'], True)} | {fmt(r['throughput_in_window'])} | "
                     f"{100 * sh['Local']:.0f} / {100 * sh['Edge']:.0f} / {100 * sh['Cloud']:.0f}% | {fmt(r['synthetic_payload_units_total'])} |")
    lines.append("")
write("v2_consolidated_comparison.md", "\n".join(lines))
