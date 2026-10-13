"""
Step 7 close-out - ONE-TIME evaluation of the frozen DQN on the reserved test seeds.
================================================================================
ROMAN URDU: Frozen setting ke 5 training seeds ke checkpoints (best =
validation par chuna gaya, aur final) ko TEST seeds 2,000-2,029 par ek hi
dafa evaluate karta hai, saath mein har baseline. DQN ke liye 5 training
seeds ka mean +- 95% CI (t-distribution, 4 degrees of freedom). Isi script ke
baad koi tuning nahi (Step7_DQN_Debugging.md, close-out protocol).

ENGLISH: reads the run folders written by run_dqn_experiments.py for the
frozen setting, evaluates them and all baselines on the test seeds, and
writes summary.json, results_table.md, per_seed.csv, full reports (gzip) and
learning_curves.png.

Run:  py evaluate_dqn_test.py --runs results/runs/final/FINAL --out results/step7_final
"""

import argparse
import csv
import glob
import gzip
import json
import math
import os
import statistics

import numpy as np
import torch

from agents.dqn import DQNAgent, DQNConfig
from agents.policy_eval import baseline_policies, evaluate
from env.gym_env import HospitalSchedulingEnv
from env.run_records import collect_provenance
from train_dqn_pilot import TEST_SEEDS, VALIDATION_SEEDS

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = {"default": "configs/environment.yaml",            # same names as run_dqn_experiments.py
             "overload": "configs/scenarios/overload.yaml",
             "slow_network": "configs/scenarios/slow_network.yaml"}
# Student t 97.5% quantiles by degrees of freedom (n training seeds - 1). df 4 is the real case (5 seeds);
# df 1 is only for the --use-validation-seeds script check on 2-seed debugging runs.
T_975 = {1: 12.706204736174698, 4: 2.7764451051977987}
REPORTED = ("return", "avg_latency", "critical_deadline_violation_rate", "deadline_violation_rate",
            "throughput_in_window", "neighbour_transfers", "cloud_transfers", "synthetic_payload_units_total")


def mean_ci(values):
    """Mean and 95% CI half-width over training seeds (t, df = n - 1)."""
    if len(values) - 1 not in T_975:
        raise ValueError(f"unsupported number of training seeds: {len(values)}")
    return statistics.mean(values), T_975[len(values) - 1] * statistics.stdev(values) / math.sqrt(len(values))


def load_agent(run_dir, which):
    manifest = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    env = HospitalSchedulingEnv(os.path.join(ROOT, manifest["config_path"]))
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, DQNConfig(**manifest["dqn_config"]),
                     seed=manifest["train_seed"], unbounded_mask=np.isinf(env.observation_space.high))
    agent.q.load_state_dict(torch.load(os.path.join(run_dir, f"model_{which}.pt")))
    return manifest, agent


def learning_curve_figure(runs_by_scenario, baselines_val, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"     # dataviz reference palette, slots 1-3
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    titles = {"default": "Default", "overload": "Overload", "slow_network": "Slow network"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (scenario, run_dirs) in zip(axes, runs_by_scenario.items()):
        curves = []
        for run_dir in run_dirs:
            with open(os.path.join(run_dir, "validation_curve.csv"), encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            curves.append(([int(r["step"]) for r in rows], [float(r["mean_return"]) for r in rows]))
        n = min(len(c[1]) for c in curves)
        steps = curves[0][0][:n]
        values = np.array([c[1][:n] for c in curves])
        ax.fill_between(steps, values.min(0), values.max(0), color=blue, alpha=0.15, linewidth=0)
        ax.plot(steps, values.mean(0), color=blue, linewidth=2, label=f"DQN (mean, range over {len(curves)} training seeds)")
        ect, fcfs = baselines_val[scenario]["ECT"], baselines_val[scenario]["FCFS"]
        ax.axhline(fcfs, color=aqua, linewidth=2, linestyle=":", label="FCFS")
        ax.axhline(ect, color=orange, linewidth=2, linestyle="--", label="ECT")
        labels = [("ECT = FCFS", ect)] if abs(ect - fcfs) < 0.05 else [("ECT", ect), ("FCFS", fcfs)]
        for text, y in labels:   # direct labels (identity is not colour-only; equal lines would hide each other)
            ax.annotate(text, (1, y), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                        ha="right", va="bottom", color=muted, fontsize=8)
        ax.set_title(titles[scenario], color=ink, fontsize=11, loc="left")
        ax.set_xlabel("training decisions (validation every ~4,000)", color=muted, fontsize=9)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    axes[0].set_ylabel("validation return (undiscounted)", color=muted, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-time test-seed evaluation of the frozen DQN (Step 7)")
    parser.add_argument("--runs", required=True, help="folder with the frozen setting's run folders")
    parser.add_argument("--out", required=True)
    parser.add_argument("--use-validation-seeds", action="store_true",
                        help="ONLY to check this script without touching the reserved test seeds")
    args = parser.parse_args(argv)
    eval_seeds = VALIDATION_SEEDS if args.use_validation_seeds else TEST_SEEDS
    torch.set_num_threads(1)
    os.makedirs(os.path.join(args.out, "reports"), exist_ok=True)

    runs_by_scenario, summary, per_seed_rows, baselines_val = {}, {}, [], {}
    for scenario, config in SCENARIOS.items():
        run_dirs = sorted(d for d in glob.glob(os.path.join(args.runs, f"*_{scenario}_seed*"))
                          if os.path.exists(os.path.join(d, "summary.json")))
        runs_by_scenario[scenario] = run_dirs
        val_summary = json.load(open(os.path.join(run_dirs[0], "summary.json"), encoding="utf-8"))
        baselines_val[scenario] = {k: v["mean_return"] for k, v in val_summary["baselines"].items()}
        results, reports = {}, {}
        for which in ("best", "final"):
            per_run = []
            for run_dir in run_dirs:
                manifest, agent = load_agent(run_dir, which)
                result = evaluate(config, lambda e, obs: agent.greedy_action(obs), eval_seeds, keep_reports=True)
                label = f"DQN_{which}_trainseed{manifest['train_seed']}"
                reports[label] = result
                per_run.append(result)
                for row in result["per_seed"]:
                    per_seed_rows.append({"scenario": scenario, "policy": f"DQN ({which})",
                                          "train_seed": manifest["train_seed"],
                                          **{k: v for k, v in row.items() if k != "report"}})
            results[f"DQN ({which})"] = {
                key: mean_ci([r["means"][key] if key != "return" else r["mean_return"] for r in per_run])
                for key in REPORTED}
            results[f"DQN ({which})"]["action_shares"] = {
                a: statistics.mean(r["action_proportions"][a] for r in per_run) for a in ("Local", "Edge", "Cloud")}
            results[f"DQN ({which})"]["per_training_seed_return"] = [r["mean_return"] for r in per_run]
        for name, policy in baseline_policies(random_seed=0).items():
            result = evaluate(config, policy, eval_seeds, keep_reports=True)
            reports[name] = result
            results[name] = {key: (result["mean_return"] if key == "return" else result["means"][key], None)
                             for key in REPORTED}
            results[name]["action_shares"] = result["action_proportions"]
            per_seed_rows += [{"scenario": scenario, "policy": name, "train_seed": "",
                               **{k: v for k, v in row.items() if k != "report"}} for row in result["per_seed"]]
        with gzip.open(os.path.join(args.out, "reports", f"{scenario}.json.gz"), "wt", encoding="utf-8") as f:
            json.dump(reports, f)
        summary[scenario] = results

    with open(os.path.join(args.out, "per_seed.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"evaluation_seeds": [eval_seeds.start, eval_seeds.stop - 1],
                   "seed_set": "validation (script check only)" if args.use_validation_seeds else "TEST (reserved)",
                   "runs": {s: [os.path.relpath(d, ROOT) for d in ds] for s, ds in runs_by_scenario.items()},
                   "ci": "mean +- 95% CI over training seeds (t, df = n-1); baselines are deterministic (CI n/a)",
                   "provenance": collect_provenance(), "results": summary}, f, indent=2)

    lines = ["| Scenario | Policy | Return | Avg latency | Critical violation | Cloud share |", "|---|---|---|---|---|---|"]
    fmt = lambda m, h, pct=False: (f"{100 * m:.1f}%" if pct else f"{m:.1f}" if abs(m) >= 10 else f"{m:.2f}") + \
        ("" if h is None else (f" ± {100 * h:.1f} pp" if pct else f" ± {h:.1f}" if abs(m) >= 10 else f" ± {h:.2f}"))
    for scenario, results in summary.items():
        for name, r in results.items():
            lines.append(f"| {scenario} | {name} | {fmt(*r['return'])} | {fmt(*r['avg_latency'])} | "
                         f"{fmt(*r['critical_deadline_violation_rate'], pct=True)} | {100 * r['action_shares']['Cloud']:.0f}% |")
    with open(os.path.join(args.out, "results_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    learning_curve_figure(runs_by_scenario, baselines_val, os.path.join(args.out, "learning_curves.png"))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
