"""
Step 5 - Baseline 1: FCFS (First-Come-First-Served), V2 evaluation protocol.
================================================================================
ROMAN URDU: Yeh "baseline" hai, koi AI nahi. Har node destination-ready FCFS
order mein kaam karta hai (env/edge_node.py) aur TaskRouter ek fixed,
priority-blind rule hai (env/task_router.py). Yeh script usi behaviour ko
common protocol (env/evaluation.py: fixed window + drain) ke saath chalati
hai, metrics print karti hai, aur har run ko apne UNIQUE folder mein
provenance (config, seed, git commit, versions) ke saath save karti hai.

ENGLISH: Runs the FCFS baseline under the common protocol and saves each run
to a unique folder: manifest.json (provenance + config snapshot), metrics.json
(full report) and tasks.csv (per-task lifecycle log). With --seeds it runs
many seeds and saves one summary (mean/std/min/max) instead.

Examples (Windows, from MS_Thesis_Project/):
    py run_baseline_fcfs.py                                   # seed from config (42)
    py run_baseline_fcfs.py --seed 7 --no-save
    py run_baseline_fcfs.py --config configs/scenarios/overload.yaml
    py run_baseline_fcfs.py --seeds 0-29                      # multi-seed check
Default output folder: results/runs/ (git-ignored scratch space).
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.evaluation import protocol_settings, run_fcfs_evaluation
from env.hospital_env import load_config
from env.metrics import TASK_LOG_COLUMNS, task_record
from env.run_records import collect_provenance, unique_run_dir, write_csv, write_json

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "configs", "environment.yaml")
DEFAULT_OUT_DIR = os.path.join(ROOT, "results", "runs")
HEADLINE_KEYS = (
    "throughput_in_window", "avg_latency", "avg_queue_wait",
    "critical_deadline_violation_rate", "deadline_violation_rate", "tasks", "unfinished_after_drain",
)


def run_fcfs_baseline(seed: int = None, config_path: str = CONFIG_PATH) -> dict:
    """Return the FCFS report for one seed (kept for tests / other scripts)."""
    config = load_config(config_path)
    return run_fcfs_evaluation(config, seed=config.get("seed", 42) if seed is None else seed).report


def _fmt(value, pct=False):
    if value is None:
        return "N/A"
    return f"{value:.1%}" if pct else (f"{value:.3f}" if isinstance(value, float) else str(value))


def print_report(report: dict, seed: int, name: str) -> None:
    p, h, d = report["protocol"], report["headline"], report["drain"]
    print(f"FCFS baseline [{name}] seed={seed} | window steps 0-{p['measurement_window'][1]} "
          f"| drain used {p['drain_steps_used']}/{p['drain_limit']}")
    print(f"  Tasks: {h['tasks']} | completed in window: {report['window']['completed']} "
          f"| completed during drain: {d['completions_during_drain']} | unfinished: {h['unfinished_after_drain']}")
    print(f"  Throughput (window): {_fmt(h['throughput_in_window'])} tasks/step")
    print(f"  Avg latency: {_fmt(h['avg_latency'])} steps | avg queue wait: {_fmt(h['avg_queue_wait'])} steps")
    print(f"  Deadline violation rate: overall {_fmt(h['deadline_violation_rate'], True)} "
          f"| Critical {_fmt(h['critical_deadline_violation_rate'], True)}")
    for prio, s in report["final"]["deadlines"]["per_priority"].items():
        print(f"    {prio:<9} on_time={s['on_time']} late={s['completed_late']} "
              f"overdue_unfinished={s['overdue_unfinished']} not_yet_due={s['not_yet_due']}")
    comm = report["final"]["offloading_communication"]
    print(f"  Routing: {dict((k, v['tasks']) for k, v in report['final']['by_tier'].items())} "
          f"| transfers neighbour={comm['neighbour_transfers']} cloud={comm['cloud_transfers']} "
          f"| synthetic payload {comm['synthetic_payload_units']['total']:.1f} SPU")
    if not h["run_complete"]:
        print(f"  WARNING: run INCOMPLETE -- {h['unfinished_after_drain']} task(s) unfinished after drain limit.")


def _parse_seeds(text: str):
    seeds = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    return seeds


def main(argv=None):
    parser = argparse.ArgumentParser(description="FCFS baseline under the V2 evaluation protocol")
    parser.add_argument("--config", default=CONFIG_PATH)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int)
    group.add_argument("--seeds", help="e.g. 0-29 or 0,5,9")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", help="run folder prefix (default: fcfs_<config name>)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    name = args.name or "fcfs_" + os.path.splitext(os.path.basename(args.config))[0]
    provenance = collect_provenance()
    manifest = {
        "method": "FCFS",
        "architecture": "v2 neighbour-edge (Local=own dept, Edge=neighbour dept, Cloud=shared)",
        "config_path": os.path.relpath(os.path.abspath(args.config), ROOT),
        "config_snapshot": config,
        "protocol_settings": protocol_settings(config),
        "provenance": provenance,
        "data_note": "SYNTHETIC placeholder workload (not MIMIC-IV); not clinical validation",
    }

    if args.seeds is None:
        seed = config.get("seed", 42) if args.seed is None else args.seed
        result = run_fcfs_evaluation(config, seed=seed)
        print_report(result.report, seed, name)
        if not args.no_save:
            out = unique_run_dir(args.out_dir, f"{name}_seed{seed}", provenance["git_commit"])
            write_json(os.path.join(out, "manifest.json"), {**manifest, "seed": seed})
            write_json(os.path.join(out, "metrics.json"), result.report)
            factor = manifest["protocol_settings"]["payload_units_per_compute_unit"]
            cutoff = result.report["protocol"]["final_cutoff"]
            write_csv(os.path.join(out, "tasks.csv"),
                      (task_record(t, cutoff, factor) for t in result.simulator.tasks), TASK_LOG_COLUMNS)
            print(f"\nSaved to {out}")
        return

    seeds = _parse_seeds(args.seeds)
    rows = []
    for seed in seeds:
        report = run_fcfs_evaluation(config, seed=seed).report
        row = {"seed": seed, **{k: report["headline"][k] for k in HEADLINE_KEYS},
               "run_complete": report["headline"]["run_complete"],
               "neighbour_transfers": report["final"]["offloading_communication"]["neighbour_transfers"],
               "cloud_transfers": report["final"]["offloading_communication"]["cloud_transfers"]}
        for cat in ("on_time", "completed_late", "overdue_unfinished", "not_yet_due"):
            row[f"critical_{cat}"] = report["final"]["deadlines"]["per_priority"]["Critical"][cat]
        rows.append(row)

    summary = {}
    for key in HEADLINE_KEYS + ("neighbour_transfers", "cloud_transfers"):
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = None if not values else {
            "n": len(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values),
        }
    summary["incomplete_runs"] = sum(1 for r in rows if not r["run_complete"])
    summary["critical_overdue_unfinished_total"] = sum(r["critical_overdue_unfinished"] for r in rows)
    summary["critical_completed_late_total"] = sum(r["critical_completed_late"] for r in rows)

    print(f"FCFS multi-seed [{name}] seeds={seeds[0]}..{seeds[-1]} (n={len(seeds)})")
    for key, s in summary.items():
        if isinstance(s, dict):
            print(f"  {key:<34} mean={s['mean']:.4f} std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f} (n={s['n']})")
        else:
            print(f"  {key:<34} {s}")
    if not args.no_save:
        out = unique_run_dir(args.out_dir, f"{name}_seeds{seeds[0]}-{seeds[-1]}", provenance["git_commit"])
        write_json(os.path.join(out, "manifest.json"), {**manifest, "seeds": seeds})
        write_json(os.path.join(out, "summary.json"), {"summary": summary, "per_seed": rows})
        write_csv(os.path.join(out, "per_seed.csv"), rows, list(rows[0].keys()))
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
