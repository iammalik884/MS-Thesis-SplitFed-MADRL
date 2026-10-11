"""
Step 5 - Evaluation metrics (FCFS baseline, and every policy after it).
================================================================================
ROMAN URDU: Yeh file ek run ke tasks ko dekh kar roadmap wale metrics nikalti
hai -- latency (aur us ke hisse), deadline results, throughput ke liye
completions, aur offloading communication. Simulation ko kuch nahi badalti
(sirf parhti hai). FCFS, DQN, FedAvg-MAPPO, SplitFed-MAPPO -- sab yahi
definitions istemal karenge taake comparison fair ho.

V2 (AUDIT FIX):
  * Har metric ek "observation cutoff" par nikalta hai: cutoff = aakhri step
    jo chal chuka hai. Jo task cutoff ke baad complete hua woh cutoff par
    "unfinished" hai.
  * Deadline ke 4 alag hisse (pehle sirf completed tasks dekhe jate thay, is
    liye ek overdue unfinished Critical task 0% violation ke peeche chhup
    jata tha):
        on_time            : complete, completion_time <= deadline
        completed_late     : complete, completion_time >  deadline
        overdue_unfinished : unfinished, deadline <= cutoff (waqt guzar gaya)
        not_yet_due        : unfinished, deadline >  cutoff (abhi miss NAHI)
  * HEADLINE deadline_violation_rate = (completed_late + overdue_unfinished)
        / (on_time + completed_late + overdue_unfinished)   ["decided" tasks]
    not_yet_due tasks denominator se bahar hain (abhi faisla nahi hua).
  * Purana completed-only rate alag naam se: completed_only_late_rate =
        completed_late / (on_time + completed_late).
  * Khaali group ka rate None hota hai (0 nahi) -- "0% violations" aur
    "koi task hi nahi tha" do alag baatein hain.

ENGLISH: Read-only summaries of a run, evaluated at an explicit observation
cutoff (the last executed step). See the definitions above; README section
"Evaluation protocol" documents them for every future policy.
"""

import statistics
from typing import Callable, Dict, Iterable, List, Optional

from env.task import PRIORITY_CLASSES, Task, is_finite_number, is_int

DEADLINE_CATEGORIES = ("on_time", "completed_late", "overdue_unfinished", "not_yet_due")


def is_done_by(task: Task, cutoff_time: int) -> bool:
    return task.completion_time is not None and task.completion_time <= cutoff_time


def deadline_status(task: Task, cutoff_time: int) -> Optional[str]:
    """4-way deadline classification at the end of step `cutoff_time`
    (None if the task has no deadline). Exact boundary: completion_time ==
    deadline is on time; an unfinished task becomes overdue once the step
    numbered `deadline` has executed (deadline <= cutoff_time)."""
    if task.deadline is None:
        return None
    if is_done_by(task, cutoff_time):
        return "on_time" if task.completion_time <= task.deadline else "completed_late"
    return "overdue_unfinished" if task.deadline <= cutoff_time else "not_yet_due"


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _mean(values: List[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def deadline_summary(tasks: Iterable[Task], cutoff_time: int) -> Dict:
    """Counts per category + both rates, over tasks that carry a deadline."""
    counts = {c: 0 for c in DEADLINE_CATEGORIES}
    for task in tasks:
        status = deadline_status(task, cutoff_time)
        if status is not None:
            counts[status] += 1
    decided = counts["on_time"] + counts["completed_late"] + counts["overdue_unfinished"]
    return {
        **counts,
        "with_deadline": decided + counts["not_yet_due"],
        "decided": decided,
        "deadline_violation_rate": _rate(counts["completed_late"] + counts["overdue_unfinished"], decided),
        "completed_only_late_rate": _rate(counts["completed_late"], counts["on_time"] + counts["completed_late"]),
    }


def _group_summary(tasks: List[Task], cutoff_time: int, key: Callable[[Task], str]) -> Dict[str, Dict]:
    groups: Dict[str, List[Task]] = {}
    for task in tasks:
        groups.setdefault(key(task), []).append(task)
    out = {}
    for name in sorted(groups):
        members = groups[name]
        done = [t for t in members if is_done_by(t, cutoff_time)]
        out[name] = {
            "tasks": len(members),
            "completed": len(done),
            "avg_latency": _mean([t.latency for t in done]),
            "avg_queue_wait": _mean([t.queue_wait for t in done]),
            "avg_transmission_delay": _mean([t.transmission_delay for t in members]),
        }
    return out


def offloading_communication(tasks: List[Task], payload_units_per_compute_unit: float = 1.0) -> Dict:
    """CURRENT task-offloading communication (one uplink transfer per Edge/
    Cloud task, dispatched at arrival). Payload is SYNTHETIC: no real data
    sizes exist yet, so size = compute_demand x factor, in "synthetic payload
    units" (SPU) -- NOT bytes. Future SplitFed training traffic (activations,
    gradients, model updates) is a separate quantity, not implemented yet."""
    if not is_finite_number(payload_units_per_compute_unit) or payload_units_per_compute_unit < 0:
        raise ValueError(
            f"payload_units_per_compute_unit must be a finite number >= 0, got {payload_units_per_compute_unit!r}"
        )
    neighbour = [t for t in tasks if t.tier == "Edge"]
    cloud = [t for t in tasks if t.tier == "Cloud"]
    spu = lambda group: sum(t.compute_demand for t in group) * payload_units_per_compute_unit
    return {
        "local_tasks_no_transfer": sum(1 for t in tasks if t.tier == "Local"),
        "neighbour_transfers": len(neighbour),
        "cloud_transfers": len(cloud),
        "total_transfers": len(neighbour) + len(cloud),
        "synthetic_payload_units": {"neighbour": spu(neighbour), "cloud": spu(cloud), "total": spu(neighbour + cloud)},
        "payload_model": (
            f"SYNTHETIC: payload = compute_demand x {payload_units_per_compute_unit} SPU "
            "(synthetic payload units, not bytes); result-return messages not modelled"
        ),
    }


def compute_metrics(all_tasks: List[Task], cutoff_time: int, payload_units_per_compute_unit: float = 1.0) -> Dict:
    """Snapshot of a run at the end of step `cutoff_time`.

    Only tasks that had arrived by the cutoff are considered. Latency-type
    averages are over tasks COMPLETED by the cutoff (each is a whole number
    of steps; 0 = finished in its arrival step). Throughput is NOT included
    here on purpose -- it is only meaningful over the fixed measurement
    window (see env/evaluation.py)."""
    if not is_int(cutoff_time) or cutoff_time < 0:
        raise ValueError(f"compute_metrics: cutoff_time must be an int >= 0, got {cutoff_time!r}")
    tasks = [t for t in all_tasks if t.arrival_time <= cutoff_time]
    done = [t for t in tasks if is_done_by(t, cutoff_time)]

    per_priority = {
        p: deadline_summary([t for t in tasks if t.priority == p], cutoff_time) for p in PRIORITY_CLASSES
    }
    return {
        "cutoff_time": cutoff_time,
        "tasks_arrived": len(tasks),
        "completed": len(done),
        "unfinished": len(tasks) - len(done),
        "avg_latency": _mean([t.latency for t in done]),
        "max_latency": max((t.latency for t in done), default=None),
        "avg_transmission_delay": _mean([t.transmission_delay for t in done]),
        "avg_queue_wait": _mean([t.queue_wait for t in done]),
        "avg_execution_span": _mean([t.execution_span for t in done]),
        "deadlines": {"overall": deadline_summary(tasks, cutoff_time), "per_priority": per_priority},
        "critical_deadline_violation_rate": per_priority["Critical"]["deadline_violation_rate"],
        "by_tier": _group_summary(tasks, cutoff_time, lambda t: t.tier or "None"),
        "by_source": _group_summary(tasks, cutoff_time, lambda t: t.source_department or "None"),
        "by_destination": _group_summary(tasks, cutoff_time, lambda t: t.destination or "None"),
        "by_route": _group_summary(tasks, cutoff_time, lambda t: f"{t.source_department}->{t.destination}"),
        "offloading_communication": offloading_communication(tasks, payload_units_per_compute_unit),
    }


# Columns of the per-task lifecycle log (CSV). One row per task.
TASK_LOG_COLUMNS = (
    "task_id", "source_department", "destination", "tier", "priority", "compute_demand",
    "arrival_time", "ready_time", "start_time", "completion_time", "deadline",
    "transmission_delay", "queue_wait", "execution_span", "latency",
    "deadline_status", "remaining_demand", "synthetic_payload_units",
)


def task_record(task: Task, cutoff_time: int, payload_units_per_compute_unit: float = 1.0) -> Dict:
    """One flat lifecycle row for CSV/JSON export (deadline_status at cutoff)."""
    return {
        "task_id": task.task_id,
        "source_department": task.source_department,
        "destination": task.destination,
        "tier": task.tier,
        "priority": task.priority,
        "compute_demand": task.compute_demand,
        "arrival_time": task.arrival_time,
        "ready_time": task.ready_time,
        "start_time": task.start_time,
        "completion_time": task.completion_time,
        "deadline": task.deadline,
        "transmission_delay": task.transmission_delay,
        "queue_wait": task.queue_wait,
        "execution_span": task.execution_span,
        "latency": task.latency,
        "deadline_status": deadline_status(task, cutoff_time),
        "remaining_demand": task.remaining_demand,
        "synthetic_payload_units": (
            task.compute_demand * payload_units_per_compute_unit if task.tier in ("Edge", "Cloud") else 0.0
        ),
    }

