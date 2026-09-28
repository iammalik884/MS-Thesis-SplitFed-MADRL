"""
Step 5 - Baseline 1: FCFS (First-Come-First-Served).
================================================================================
ROMAN URDU: Yeh "baseline" hai, koi nayi scheduling logic NAHI. Steps 1-4 ka
simulator already FCFS hai: har EdgeNode apni queue ko sirf arrival order
mein process karta hai (env/edge_node.py step(), koi reordering nahi), aur
TaskRouter (Step 3) bhi ek fixed, non-learned rule hai. Is script ka kaam
sirf itna hai ke ussi behaviour ko EK REPRODUCIBLE EXPERIMENT ki tarah chalaye
aur uske metrics (env/metrics.py) ko results/ folder mein seed ke saath save
kare -- taake Step 7 (DQN) aur Step 9 (MAPPO) inhi saved numbers se compare
ho sakein.

ENGLISH: This is a BASELINE, not new scheduling logic. The Steps 1-4
simulator is already FCFS: every EdgeNode processes its queue in pure
arrival order (env/edge_node.py's step(), no reordering), and TaskRouter
(Step 3) is a fixed, non-learned rule. This script's only job is to run that
existing behaviour as one reproducible experiment and save its metrics
(env/metrics.py) to results/, tagged with the seed and config used -- so
later Step 7 (DQN) and Step 9 (MAPPO) baselines can be compared against
these exact saved numbers.

Run with:  py run_baseline_fcfs.py   (from MS_Thesis_Project/)
"""

import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.hospital_env import HospitalSimulator
from env.metrics import compute_baseline_metrics
from env.task_generator import MedicalTaskGenerator

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "environment.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

SEED = CONFIG.get("seed", 42)
GENERATION_STEPS = CONFIG["simulation"]["n_steps"]
DRAIN_STEPS = 10  # extra steps with no new arrivals, so already-queued work can finish


def run_fcfs_baseline(seed: int = SEED) -> dict:
    """Run the existing FCFS simulator end-to-end and return its metrics.

    Reuses HospitalSimulator/MedicalTaskGenerator exactly as Steps 1-4 built
    them -- no new simulation logic. "FCFS" here just names the fact that
    nothing in this call reorders or prioritizes the queue.
    """
    sim = HospitalSimulator(CONFIG_PATH)
    generator = MedicalTaskGenerator(CONFIG, seed=seed)

    all_tasks = []
    for _ in range(GENERATION_STEPS):
        arrivals = generator.generate_for_step(sim.current_time)
        created = sim.route_generated_arrivals(arrivals)
        all_tasks.extend(created)
        sim.step()

    for _ in range(DRAIN_STEPS):
        sim.step()

    completed = []
    for node in list(sim.nodes.values()) + list(sim.local_nodes.values()):
        completed.extend(node.completed_tasks)
    if sim.cloud_node is not None:
        completed.extend(sim.cloud_node.completed_tasks)

    total_steps = GENERATION_STEPS + DRAIN_STEPS
    return compute_baseline_metrics(all_tasks, completed, total_steps)


def main():
    metrics = run_fcfs_baseline(seed=SEED)

    print(f"FCFS Baseline (seed={SEED}, {GENERATION_STEPS} generation + {DRAIN_STEPS} drain steps)\n")
    print(f"Total Tasks: {metrics['total_tasks']}")
    print(f"Completed: {metrics['completed_tasks']}")
    print(f"Still Waiting: {metrics['still_waiting']}")
    print(f"Throughput: {metrics['throughput']:.3f} tasks/step")
    if metrics["avg_latency"] is not None:
        print(f"Average Latency: {metrics['avg_latency']:.2f} steps")
    if metrics["deadline_violation_rate"] is not None:
        print(f"Overall Deadline Violation Rate: {metrics['deadline_violation_rate']:.1%}")
    if metrics["critical_deadline_violation_rate"] is not None:
        print(f"Critical Deadline Violation Rate: {metrics['critical_deadline_violation_rate']:.1%}")
    for priority, stats in metrics["per_priority"].items():
        if stats["completed"]:
            print(f"  {priority:<9} {stats['met']}/{stats['completed']} met deadline")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"step5_fcfs_baseline_seed{SEED}.json")
    with open(out_path, "w") as f:
        json.dump({"method": "FCFS", "seed": SEED, "config_path": CONFIG_PATH, "metrics": metrics}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
