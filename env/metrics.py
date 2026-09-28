"""
Step 5 - Baseline evaluation metrics (FCFS baseline, and every baseline after it).
================================================================================
ROMAN URDU: Steps 1-4 ne simulator bana diya (queue, routing, deadlines), lekin
abhi tak humare paas koi aisa number nahi tha jo "yeh baseline kitna acha
perform karta hai" bata sake. Yeh file bilkul wohi 4 metrics nikalti hai jo
roadmap mein approved hain (latency, Critical deadline-violation rate,
throughput, aur completed/waiting count) -- taake Step 5 ka FCFS baseline aur
baad mein Step 7 ka DQN, Step 9 ka MAPPO -- sab EK JAISE numbers report karein
aur unka comparison fair ho.

ENGLISH: Steps 1-4 built the simulator (queueing, routing, deadlines), but
there was no single place that turned a finished run into "how good was this
policy" numbers. This file computes exactly the metrics the roadmap approved
for later comparison across every baseline/agent (latency, Critical
deadline-violation rate, throughput, completed/waiting counts) -- so Step 5's
FCFS baseline and later Step 7's DQN / Step 9's MAPPO all report the SAME
metric definitions, making the comparison fair.

Nothing in here changes simulation behaviour -- it only reads finished Task
objects after a run and summarizes them (read-only, no side effects).
"""

import statistics
from typing import Dict, List

from env.task import PRIORITY_CLASSES, Task


def compute_baseline_metrics(all_tasks: List[Task], completed_tasks: List[Task], n_steps: int) -> Dict:
    """Summarize one finished simulation run into the roadmap's metrics.

    Args:
        all_tasks: every task created during the run (completed or not).
        completed_tasks: the subset of all_tasks that finished (is_complete).
        n_steps: how many time-steps the run covered (throughput denominator).

    Returns a dict with:
        total_tasks, completed_tasks, still_waiting: raw counts.
        throughput: completed tasks per time-step (higher is better).
        avg_latency: mean (completion_time - arrival_time) over completed
            tasks, or None if nothing completed yet.
        deadline_violation_rate: overall fraction of completed, deadline-
            carrying tasks that finished LATE, or None if none had a deadline.
        critical_deadline_violation_rate: same, restricted to Critical-
            priority tasks -- this is the roadmap's headline safety metric,
            reported separately because a missed Critical deadline matters
            far more than a missed Routine one. None if no Critical task with
            a deadline has completed yet.
        per_priority: {priority: {"completed": n, "met": n, "violation_rate": x}}
            for each of PRIORITY_CLASSES, restricted to tasks that carry a
            deadline (None fields where a class has no such completed tasks).
    """
    if n_steps <= 0:
        raise ValueError(f"compute_baseline_metrics: n_steps must be > 0, got {n_steps}")

    total = len(all_tasks)
    completed_n = len(completed_tasks)
    metrics: Dict = {
        "total_tasks": total,
        "completed_tasks": completed_n,
        "still_waiting": total - completed_n,
        "throughput": completed_n / n_steps,
    }

    if completed_tasks:
        metrics["avg_latency"] = statistics.mean(
            t.completion_time - t.arrival_time for t in completed_tasks
        )
    else:
        metrics["avg_latency"] = None

    with_deadline = [t for t in completed_tasks if t.deadline is not None]
    if with_deadline:
        met_n = sum(1 for t in with_deadline if t.met_deadline)
        metrics["deadline_violation_rate"] = 1 - (met_n / len(with_deadline))
    else:
        metrics["deadline_violation_rate"] = None

    per_priority = {}
    for priority in PRIORITY_CLASSES:
        subset = [t for t in with_deadline if t.priority == priority]
        if subset:
            subset_met = sum(1 for t in subset if t.met_deadline)
            per_priority[priority] = {
                "completed": len(subset),
                "met": subset_met,
                "violation_rate": 1 - (subset_met / len(subset)),
            }
        else:
            per_priority[priority] = {"completed": 0, "met": 0, "violation_rate": None}
    metrics["per_priority"] = per_priority
    metrics["critical_deadline_violation_rate"] = per_priority["Critical"]["violation_rate"]

    return metrics
