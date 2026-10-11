"""
V2 audit fixes (part b) - deadline categories, fixed measurement window +
drain protocol, offloading communication, and run records.
================================================================================
ROMAN URDU: Har test chhota aur haath se calculate kiya hua hai. Khaas taur
par yeh prove kiya gaya hai ke (1) overdue unfinished Critical task ab
"0% violation" ke peeche nahi chhupta, (2) jin tasks ki deadline abhi nahi
aayi unhein "missed" nahi gina jata, aur (3) drain ki lambai badalne se
primary (window) throughput nahi badalta.

ENGLISH: Small hand-calculated checks for the new metric definitions.

Run with:  py -m pytest tests/test_v2_metrics_protocol.py
"""

import copy
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from env.evaluation import run_fcfs_evaluation, summarize_protocol
from env.hospital_env import HospitalSimulator
from env.metrics import TASK_LOG_COLUMNS, compute_metrics, deadline_status, task_record
from env.run_records import collect_provenance, parse_porcelain, unique_run_dir, write_csv
from env.task import Task

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "environment.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)
EMPTY_BACKLOG = {"tasks": 0, "remaining_demand": 0.0, "per_node": {}}


def _task(task_id, arrival, deadline, completion=None, priority="Critical"):
    t = Task(task_id=task_id, arrival_time=arrival, compute_demand=1.0, priority=priority, deadline=deadline)
    if completion is not None:
        t.completion_time, t.start_time, t.remaining_demand = completion, completion, 0.0
    return t


def test_deadline_status_boundaries():
    assert deadline_status(_task(0, 0, 2, completion=2), cutoff_time=5) == "on_time"          # == deadline
    assert deadline_status(_task(1, 0, 2, completion=3), cutoff_time=5) == "completed_late"
    assert deadline_status(_task(2, 0, 2), cutoff_time=2) == "overdue_unfinished"            # step 2 ran
    assert deadline_status(_task(3, 0, 2), cutoff_time=1) == "not_yet_due"                   # step 2 not yet
    # Completed AFTER the cutoff counts as unfinished at that cutoff.
    assert deadline_status(_task(4, 0, 5, completion=4), cutoff_time=3) == "not_yet_due"
    assert deadline_status(_task(5, 0, None), cutoff_time=3) is None
    print("[PASS] test_deadline_status_boundaries")


def test_overdue_unfinished_critical_is_not_hidden():
    """One Critical on time + one Critical overdue and unfinished.
    Old completed-only rate: 0/1 = 0% (the overdue task was invisible).
    V2 headline: (0 late + 1 overdue) / 2 decided = 50%."""
    tasks = [_task(0, 0, 2, completion=1), _task(1, 0, 2)]
    crit = compute_metrics(tasks, cutoff_time=4)["deadlines"]["per_priority"]["Critical"]
    assert crit["completed_only_late_rate"] == 0.0
    assert crit["deadline_violation_rate"] == 0.5
    assert (crit["on_time"], crit["overdue_unfinished"], crit["decided"]) == (1, 1, 2)
    print("[PASS] test_overdue_unfinished_critical_is_not_hidden")


def test_not_yet_due_is_excluded_from_denominator():
    tasks = [_task(0, 0, 2, completion=1), _task(1, 3, 8)]          # 2nd: deadline 8 > cutoff 4
    crit = compute_metrics(tasks, cutoff_time=4)["deadlines"]["per_priority"]["Critical"]
    assert crit["not_yet_due"] == 1 and crit["decided"] == 1 and crit["with_deadline"] == 2
    assert crit["deadline_violation_rate"] == 0.0
    print("[PASS] test_not_yet_due_is_excluded_from_denominator")


def test_empty_groups_report_none_not_zero():
    m = compute_metrics([], cutoff_time=0)
    assert m["avg_latency"] is None and m["critical_deadline_violation_rate"] is None
    assert m["deadlines"]["overall"]["deadline_violation_rate"] is None
    assert m["deadlines"]["overall"]["decided"] == 0
    print("[PASS] test_empty_groups_report_none_not_zero")


def test_overall_equals_sum_of_priorities_and_latency_parts():
    for seed in range(15):
        report = run_fcfs_evaluation(CONFIG, seed=seed).report
        for snap in (report["window"], report["final"]):
            overall, per = snap["deadlines"]["overall"], snap["deadlines"]["per_priority"]
            for key in ("on_time", "completed_late", "overdue_unfinished", "not_yet_due", "decided", "with_deadline"):
                assert overall[key] == sum(p[key] for p in per.values()), (seed, key)
            assert overall["with_deadline"] == overall["decided"] + overall["not_yet_due"]
            assert snap["completed"] + snap["unfinished"] == snap["tasks_arrived"]
            assert sum(v["tasks"] for v in snap["by_tier"].values()) == snap["tasks_arrived"]
            if snap["completed"]:
                parts = snap["avg_transmission_delay"] + snap["avg_queue_wait"] + snap["avg_execution_span"]
                assert abs(parts - snap["avg_latency"]) < 1e-9
    print("[PASS] test_overall_equals_sum_of_priorities_and_latency_parts")


def test_window_throughput_excludes_drain_completions():
    """Window = 2 steps (0..1). Task A done at t=0; task B (ICU 10.0, capacity
    4) needs t=0,1,2 -> done at t=2, i.e. in the drain. Window throughput =
    1/2, not 2/2 and not 2/3."""
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("Emergency", compute_demand=1.0)
    sim.add_task("ICU", compute_demand=10.0)
    sim.run(3)
    rep = summarize_protocol(sim.tasks, window_steps=2, final_cutoff=2, drain_limit=50,
                             backlog_at_window_end=EMPTY_BACKLOG, backlog_after_drain=EMPTY_BACKLOG)
    assert rep["window"]["completed"] == 1 and rep["window"]["throughput"] == 0.5
    assert rep["drain"]["completions_during_drain"] == 1 and rep["protocol"]["drain_steps_used"] == 1
    print("[PASS] test_window_throughput_excludes_drain_completions")


def test_drain_limit_changes_nothing_in_window_and_flags_incomplete_runs():
    overload = copy.deepcopy(CONFIG)
    for prio in overload["task_generation"]["departments"].values():
        for s in prio.values():
            s["arrival_rate"] *= 6
    short, long_ = copy.deepcopy(overload), copy.deepcopy(overload)
    short["evaluation"]["drain_limit"] = 2
    long_["evaluation"]["drain_limit"] = 200
    a, b = run_fcfs_evaluation(short, seed=3).report, run_fcfs_evaluation(long_, seed=3).report
    assert a["window"] == b["window"], "drain length must not change any window metric"
    assert a["protocol"]["drain_steps_used"] == 2 and a["drain"]["drain_limit_reached"]
    assert not a["headline"]["run_complete"] and a["drain"]["backlog_after_drain"]["tasks"] > 0
    assert b["headline"]["run_complete"] and b["protocol"]["drain_steps_used"] < 200, "drain stops when empty"
    print("[PASS] test_drain_limit_changes_nothing_in_window_and_flags_incomplete_runs")


def test_offloading_communication_and_route_breakdown():
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("ICU", compute_demand=8.0)                    # Local (manual)
    sim.route_task("ICU", compute_demand=2.0)                  # Edge -> Emergency
    sim.route_task("ICU", compute_demand=7.0)                  # Cloud
    sim.run(4)
    m = compute_metrics(sim.tasks, cutoff_time=3, payload_units_per_compute_unit=0.5)
    comm = m["offloading_communication"]
    assert (comm["local_tasks_no_transfer"], comm["neighbour_transfers"], comm["cloud_transfers"]) == (1, 1, 1)
    assert comm["synthetic_payload_units"] == {"neighbour": 1.0, "cloud": 3.5, "total": 4.5}
    assert "SYNTHETIC" in comm["payload_model"]
    assert set(m["by_route"]) == {"ICU->ICU", "ICU->Emergency", "ICU->Cloud"}
    assert m["by_destination"]["Cloud"]["avg_transmission_delay"] == 2
    assert m["by_source"]["ICU"]["tasks"] == 3
    print("[PASS] test_offloading_communication_and_route_breakdown")


def test_task_log_and_unique_run_folders():
    sim = HospitalSimulator(CONFIG_PATH)
    sim.route_task("ICU", compute_demand=7.0, priority="High")
    sim.run(3)
    row = task_record(sim.tasks[0], cutoff_time=2)
    assert list(row) == list(TASK_LOG_COLUMNS)
    assert (row["destination"], row["transmission_delay"], row["latency"], row["deadline_status"]) == ("Cloud", 2, 2, "on_time")
    with tempfile.TemporaryDirectory() as tmp:
        d1, d2 = unique_run_dir(tmp, "x", "abc1234"), unique_run_dir(tmp, "x", "abc1234")
        assert d1 != d2 and os.path.isdir(d1) and os.path.isdir(d2), "rerun must never reuse a folder"
        path = os.path.join(d1, "tasks.csv")
        write_csv(path, [row], TASK_LOG_COLUMNS)
        with open(path, newline="") as f:
            assert list(csv.DictReader(f))[0]["destination"] == "Cloud"
    prov = collect_provenance()
    for key in ("git_commit", "git_dirty", "python_executable", "python_version", "packages"):
        assert key in prov
    # Regression: .strip() used to eat the first line's leading space -> "gitignore".
    assert parse_porcelain(" M .gitignore\nM  env/task.py\n") == [".gitignore", "env/task.py"]
    print("[PASS] test_task_log_and_unique_run_folders")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL V2 METRICS/PROTOCOL TESTS PASSED.")
