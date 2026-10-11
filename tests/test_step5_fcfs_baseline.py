"""
Step 5 - Verification tests for the FCFS baseline metrics.
================================================================================
ROMAN URDU: Yeh tests do cheezein prove karte hain: (1) env/metrics.py haath
se calculate kiye gaye numbers hi deta hai, (2) poora FCFS pipeline
reproducible hai -- same seed hamesha same result.

ENGLISH: (1) metrics match hand-calculated values, (2) the FCFS pipeline is
reproducible (same seed -> identical report).

UPDATED EXPECTATIONS (V2 audit fix) -- why:
  * compute_baseline_metrics(all, completed, n_steps) was replaced by
    compute_metrics(tasks, cutoff_time) + summarize_protocol(): metrics are
    now evaluated at an explicit observation cutoff, and throughput only
    over the fixed measurement window. The SAME hand scenarios are kept.
  * test_missed_deadline_is_counted now ALSO checks the case the old
    function hid: the same Critical task observed at cutoff 2, still
    unfinished, is "overdue_unfinished" (violation rate 1.0), where the old
    completed-only rate reported None/0%.
  * test_invalid_n_steps... -> invalid cutoff/window values (same intent).

Run with:  py -m pytest tests/test_step5_fcfs_baseline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.evaluation import summarize_protocol
from env.hospital_env import HospitalSimulator
from env.metrics import compute_metrics
from run_baseline_fcfs import run_fcfs_baseline

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "environment.yaml")
EMPTY_BACKLOG = {"tasks": 0, "remaining_demand": 0.0, "per_node": {}}


def test_metrics_on_known_scenario():
    """Emergency capacity 5, two Critical tasks (deadline 0+2):
    t=0: t1 (2.0) done, t2 gets 3 (7 left); t=1: t2 gets 5 (2 left);
    t=2: t2 done -> completion 2 == deadline -> ON TIME."""
    sim = HospitalSimulator(CONFIG_PATH)
    t1 = sim.add_task("Emergency", compute_demand=2.0, priority="Critical")
    t2 = sim.add_task("Emergency", compute_demand=10.0, priority="Critical")
    sim.run(3)

    m = compute_metrics(sim.tasks, cutoff_time=2)
    assert (m["tasks_arrived"], m["completed"], m["unfinished"]) == (2, 2, 0)
    assert m["avg_latency"] == 1.0, "(0 + 2) / 2"
    assert m["avg_queue_wait"] == 0.0, "both started at t=0"
    assert m["critical_deadline_violation_rate"] == 0.0
    crit = m["deadlines"]["per_priority"]["Critical"]
    assert (crit["on_time"], crit["completed_late"], crit["decided"]) == (2, 0, 2)
    assert m["deadlines"]["per_priority"]["High"]["deadline_violation_rate"] is None, "empty group -> None"

    report = summarize_protocol(sim.tasks, window_steps=3, final_cutoff=2, drain_limit=0,
                                backlog_at_window_end=EMPTY_BACKLOG, backlog_after_drain=EMPTY_BACKLOG)
    assert report["window"]["throughput"] == 2 / 3
    assert t1.completion_time == 0 and t2.completion_time == 2
    print("[PASS] test_metrics_on_known_scenario")


def test_missed_deadline_is_counted():
    """ICU capacity 4, one Critical 14.0 task (deadline 2): 4+4+4+2 ->
    completion 3 > 2 -> completed_late. At cutoff 2 it is unfinished with its
    deadline passed -> overdue_unfinished (the old metric hid this)."""
    sim = HospitalSimulator(CONFIG_PATH)
    t1 = sim.add_task("ICU", compute_demand=14.0, priority="Critical")
    sim.run(4)
    assert t1.completion_time == 3

    final = compute_metrics(sim.tasks, cutoff_time=3)
    assert final["deadlines"]["per_priority"]["Critical"]["completed_late"] == 1
    assert final["critical_deadline_violation_rate"] == 1.0
    assert final["deadlines"]["overall"]["deadline_violation_rate"] == 1.0

    at_two = compute_metrics(sim.tasks, cutoff_time=2)
    crit = at_two["deadlines"]["per_priority"]["Critical"]
    assert crit["overdue_unfinished"] == 1 and crit["completed_only_late_rate"] is None
    assert at_two["critical_deadline_violation_rate"] == 1.0
    print("[PASS] test_missed_deadline_is_counted")


def test_still_waiting_tasks_excluded_from_latency():
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("General_Ward", compute_demand=1.0)    # finishes at t=0
    sim.add_task("General_Ward", compute_demand=100.0)  # will not finish
    sim.step()
    m = compute_metrics(sim.tasks, cutoff_time=0)
    assert (m["completed"], m["unfinished"]) == (1, 1)
    assert m["avg_latency"] == 0.0, "only the finished task's latency (0) is averaged"
    print("[PASS] test_still_waiting_tasks_excluded_from_latency")


def test_reproducibility_same_seed_same_metrics():
    assert run_fcfs_baseline(seed=42) == run_fcfs_baseline(seed=42)
    print("[PASS] test_reproducibility_same_seed_same_metrics")


def test_different_seeds_can_differ():
    r42, r7 = run_fcfs_baseline(seed=42), run_fcfs_baseline(seed=7)
    assert r42["headline"]["tasks"] > 0 and r7["headline"]["tasks"] > 0
    print("[PASS] test_different_seeds_can_differ")


def test_invalid_n_steps_raises_meaningful_error():
    for bad in (lambda: compute_metrics([], cutoff_time=-1),
                lambda: summarize_protocol([], 0, 0, 0, EMPTY_BACKLOG, EMPTY_BACKLOG)):
        try:
            bad()
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "cutoff_time" in str(e) or "window_steps" in str(e)
    print("[PASS] test_invalid_n_steps_raises_meaningful_error")


if __name__ == "__main__":
    test_metrics_on_known_scenario()
    test_missed_deadline_is_counted()
    test_still_waiting_tasks_excluded_from_latency()
    test_reproducibility_same_seed_same_metrics()
    test_different_seeds_can_differ()
    test_invalid_n_steps_raises_meaningful_error()
    print("\nALL STEP 5 TESTS PASSED.")
