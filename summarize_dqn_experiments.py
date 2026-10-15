"""
Step 7 - Summarise DQN debugging runs with the pre-registered criterion.
================================================================================
score = (R - R_random) / (R_ECT - R_random) on validation seeds (0 = random, 1 = ECT),
computed per run with that run's own baseline evaluations. Selection uses the
mean FINAL-checkpoint score over scenarios and training seeds
(Step_Documentation/Step7_DQN_Debugging.md). Step 9: MAPPO run folders work too;
their "Q" columns then show the critic V vs Monte Carlo check.

Run:  py summarize_dqn_experiments.py results/runs/phase1
"""

import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def load(group_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(group_dir, "*", "*", "summary.json"))):
        summary = json.load(open(path, encoding="utf-8"))
        manifest = json.load(open(os.path.join(os.path.dirname(path), "manifest.json"), encoding="utf-8"))
        exp = os.path.basename(os.path.dirname(os.path.dirname(path)))
        scenario = os.path.splitext(os.path.basename(manifest["config_path"]))[0]
        method = "dqn" if "dqn_best" in summary else "mappo"          # Step 7 DQN or Step 9 MAPPO run
        check = summary["q_vs_mc"] if method == "dqn" else summary["v_vs_mc"]   # Q (DQN) or critic V (MAPPO) vs MC
        r_rand = summary["baselines"]["random"]["mean_return"]
        r_ect = summary["baselines"]["ECT"]["mean_return"]
        score = lambda r: (r - r_rand) / (r_ect - r_rand)
        rows.append({
            "exp": exp, "scenario": scenario, "seed": manifest["train_seed"],
            "overrides": manifest[f"{method}_overrides"], "total_steps": manifest["total_steps"], "commit": manifest["provenance"]["git_commit"][:7],
            "dirty": manifest["provenance"]["git_dirty"],
            "best_return": summary[f"{method}_best"]["mean_return"], "final_return": summary[f"{method}_final"]["mean_return"],
            "best_step": summary[f"{method}_best"]["step"],
            "best_score": score(summary[f"{method}_best"]["mean_return"]),
            "final_score": score(summary[f"{method}_final"]["mean_return"]),
            "ect": r_ect, "random": r_rand,
            "q_bias_final_unscaled": check[f"{method}_final"]["unscaled"]["mean_bias"],
            "q_rmse_final_unscaled": check[f"{method}_final"]["unscaled"]["rmse"],
            "q_corr_final": check[f"{method}_final"]["correlation"],
            "cloud_share_final": summary[f"{method}_final"]["action_proportions"]["Cloud"],
        })
    return rows


def main(group_dir):
    rows = load(group_dir)
    print(f"{'exp':4s} {'scenario':13s} {'seed':>4s} {'best R':>9s} {'final R':>9s} {'best sc':>8s} {'final sc':>9s} "
          f"{'Q bias':>8s} {'Q RMSE':>8s} {'corr':>6s} {'cloud':>6s}")
    for r in rows:
        print(f"{r['exp']:4s} {r['scenario']:13s} {r['seed']:4d} {r['best_return']:9.1f} {r['final_return']:9.1f} "
              f"{r['best_score']:8.2f} {r['final_score']:9.2f} {r['q_bias_final_unscaled']:8.1f} "
              f"{r['q_rmse_final_unscaled']:8.1f} {r['q_corr_final']:6.2f} {r['cloud_share_final']:6.2f}")
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[r["exp"]].append(r)
    print("\nPer experiment (mean over scenarios x training seeds):")
    print(f"{'exp':4s} {'override':28s} {'runs':>4s} {'steps':>7s} {'mean final score':>17s} {'mean best score':>16s} {'mean drop':>10s}")
    for exp, rs in sorted(by_exp.items()):
        final = statistics.mean(r["final_score"] for r in rs)
        best = statistics.mean(r["best_score"] for r in rs)
        print(f"{exp:4s} {json.dumps(rs[0]['overrides']):28s} {len(rs):4d} {rs[0]['total_steps']:7d} {final:17.3f} {best:16.3f} {best - final:10.3f}")
    commits = {(r["commit"], r["dirty"]) for r in rows}
    print(f"\nprovenance (commit, dirty): {sorted(commits)}")


if __name__ == "__main__":
    main(sys.argv[1])
