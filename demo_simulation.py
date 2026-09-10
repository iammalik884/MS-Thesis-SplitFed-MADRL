"""
Live simulation demo - Steps 1-3 (Poisson arrivals + Local/Edge/Cloud routing).
================================================================================
Reuses HospitalSimulator, MedicalTaskGenerator, and TaskRouter exactly as
implemented and tested in Steps 1-3. No new logic is added here, and nothing
is hard-coded -- every printed number comes from actually running the
simulation with seed=42 (configs/environment.yaml).

Run with:  py demo_simulation.py   (from MS_Thesis_Project/)
"""

import os
import statistics
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.hospital_env import HospitalSimulator
from env.task_generator import MedicalTaskGenerator

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "environment.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

SEED = 42
GENERATION_STEPS = CONFIG["simulation"]["n_steps"]  # 20, from configs/environment.yaml
DRAIN_STEPS = 10  # extra steps with no new arrivals, so already-queued tasks get a chance to finish


def main():
    sim = HospitalSimulator(CONFIG_PATH)
    generator = MedicalTaskGenerator(CONFIG, seed=SEED)

    all_tasks = []

    print(f"Running {GENERATION_STEPS} arrival steps (seed={SEED}), then "
          f"{DRAIN_STEPS} drain steps (no new arrivals) so queued work can finish...\n")

    for _ in range(GENERATION_STEPS):
        arrivals = generator.generate_for_step(sim.current_time)
        created = sim.route_generated_arrivals(arrivals)
        for arrival, task in zip(arrivals, created):
            all_tasks.append(task)
            dept_label = arrival.department.replace("_", " ")
            print(
                f"Task {task.task_id + 1:03d} | {dept_label} | {task.priority} | "
                f"Demand {task.compute_demand:.1f} | {task.tier.upper()}"
            )
        sim.step()

    for _ in range(DRAIN_STEPS):
        sim.step()

    completed = []
    for node in list(sim.nodes.values()) + list(sim.local_nodes.values()):
        completed.extend(node.completed_tasks)
    if sim.cloud_node is not None:
        completed.extend(sim.cloud_node.completed_tasks)

    total = len(all_tasks)
    local_n = sum(1 for t in all_tasks if t.tier == "Local")
    edge_n = sum(1 for t in all_tasks if t.tier == "Edge")
    cloud_n = sum(1 for t in all_tasks if t.tier == "Cloud")
    completed_n = len(completed)
    waiting_n = total - completed_n

    print("\n--- Summary ---")
    print(f"Total Tasks: {total}")
    print(f"Local: {local_n}")
    print(f"Edge: {edge_n}")
    print(f"Cloud: {cloud_n}")
    print(f"Completed: {completed_n}")
    print(f"Still Waiting: {waiting_n}")

    if completed:
        avg_latency = statistics.mean(t.completion_time - t.arrival_time for t in completed)
        print(f"Average Completion Time (steps from arrival to finish): {avg_latency:.2f}")
    else:
        print("Average Completion Time: N/A (no tasks completed)")

    if cloud_n == 0:
        print(
            "\nNOTE: 0 Cloud-routed tasks is expected with the current config, not a bug -- "
            "configs/environment.yaml's max compute_demand (4.0, Routine upper bound) never "
            "exceeds edge_threshold (6.0), so TaskRouter.choose_tier() can never return "
            "'Cloud' with these values. Raise edge_threshold or the Routine demand range in "
            "configs/environment.yaml if you want to see a real Cloud-routed task."
        )


if __name__ == "__main__":
    main()
