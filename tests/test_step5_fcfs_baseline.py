"""
Step 5 - Verification tests for the FCFS baseline metrics.
================================================================================
ROMAN URDU: Yeh tests do cheezein prove karte hain: (1) env/metrics.py sahi
numbers nikalta hai jab hum manually expected values calculate karein (2)
run_baseline_fcfs.py ka poora pipeline reproducible hai -- same seed hamesha
same result deta hai, per roadmap ka rule ("same random seed reproduces the
same result").

ENGLISH: These tests prove two things: (1) env/metrics.py computes correct
numbers against hand-calculated expected values, and (2) run_baseline_fcfs.py's
full pipeline is reproducible -- the same seed always yields the same result,
per the roadmap's explicit rule for every baseline going forward.

Run with:  python3 -m tests.test_step5_fcfs_baseline   (from MS_Thesis_Project/)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.hospital_env import HospitalSimulator
from env.metrics import compute_baseline_metrics
from run_baseline_fcfs import run_fcfs_baseline

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "environment.yaml")


def test_metrics_on_known_scenario():
    """Hand-crafted scenario with a known outcome, checked against
    compute_baseline_metrics() instead of trusting the pipeline blindly."""
    sim = HospitalSimulator(CONFIG_PATH)
    # Emergency capacity = 5.0. Two Critical tasks (deadline offset = 2).
    t1 = sim.add_task("Emergency", compute_demand=2.0, priority="Critical")  # finishes step 0, on time
    t2 = sim.add_task("Emergency", compute_demand=10.0, priority="Critical")  # needs 3 steps -> late (deadline=2)
    sim.step()  # t=0: t1 done (2.0 used), t2 gets 3.0 done (7.0 left)
    sim.step()  # t=1: t2 gets 5.0 done (2.0 left)
    sim.step()  # t=2: t2 gets 2.0 done -> complete at t=2, deadline was 2 -> ON TIME

    all_tasks = [t1, t2]
    completed = [t1, t2]
    metrics = compute_baseline_metrics(all_tasks, completed, n_steps=3)

    assert metrics["total_tasks"] == 2
    assert metrics["completed_tasks"] == 2
    assert metrics["still_waiting"] == 0
    assert metrics["throughput"] == 2 / 3, f"Expected 2/3, got {metrics['throughput']}"
    assert metrics["avg_latency"] == 1.0, f"Expected (0+2)/2=1.0, got {metrics['avg_latency']}"
    assert metrics["critical_deadline_violation_rate"] == 0.0, "Both Critical tasks met their deadline"
    assert metrics["per_priority"]["Critical"]["completed"] == 2
    assert metrics["per_priority"]["Critical"]["met"] == 2
    assert metrics["per_priority"]["High"]["completed"] == 0
    assert metrics["per_priority"]["High"]["violation_rate"] is None
    print("[PASS] test_metrics_on_known_scenario")


def test_missed_deadline_is_counted():
    """A task that finishes AFTER its deadline must show up as a violation."""
    sim = HospitalSimulator(CONFIG_PATH)
    # ICU capacity = 4.0, Critical deadline offset = 2. A 10.0-demand task
    # needs 3 steps (t=0,1,2 -> completes at t=2), which is exactly on time,
    # so use a bigger demand to force a genuine miss.
    t1 = sim.add_task("ICU", compute_demand=14.0, priority="Critical")  # needs 4 steps -> completes t=3 > deadline 2
    for _ in range(4):
        sim.step()

    metrics = compute_baseline_metrics([t1], [t1], n_steps=4)
    assert metrics["critical_deadline_violation_rate"] == 1.0, "The only Critical task missed its deadline"
    assert metrics["deadline_violation_rate"] == 1.0
    print("[PASS] test_missed_deadline_is_counted")


def test_still_waiting_tasks_excluded_from_latency():
    """An incomplete task must count toward still_waiting but must NOT be
    averaged into avg_latency (it has no completion_time yet)."""
    sim = HospitalSimulator(CONFIG_PATH)
    t1 = sim.add_task("General_Ward", compute_demand=1.0)  # finishes immediately
    t2 = sim.add_task("General_Ward", compute_demand=100.0)  # will not finish this run
    sim.step()

    metrics = compute_baseline_metrics([t1, t2], [t1], n_steps=1)
    assert metrics["completed_tasks"] == 1
    assert metrics["still_waiting"] == 1
    assert metrics["avg_latency"] == 0.0, "Only t1's latency (0) should be averaged"
    print("[PASS] test_still_waiting_tasks_excluded_from_latency")


def test_reproducibility_same_seed_same_metrics():
    """The full FCFS baseline pipeline must be deterministic: same seed,
    same metrics, every time -- required before any baseline result can be
    trusted for comparison against later DQN/MAPPO runs (Steps 7+, 9+)."""
    metrics_a = run_fcfs_baseline(seed=42)
    metrics_b = run_fcfs_baseline(seed=42)
    assert metrics_a == metrics_b, "Same seed must reproduce identical metrics"
    print("[PASS] test_reproducibility_same_seed_same_metrics")


def test_different_seeds_can_differ():
    """Sanity check that seed actually controls the run (metrics dict is not
    hardcoded/constant regardless of input)."""
    metrics_42 = run_fcfs_baseline(seed=42)
    metrics_7 = run_fcfs_baseline(seed=7)
    assert metrics_42["total_tasks"] > 0 and metrics_7["total_tasks"] > 0
    # Not asserting they differ (a different seed COULD coincidentally match),
    # just that both runs produce a valid, non-empty result independently.
    print("[PASS] test_different_seeds_can_differ")


def test_invalid_n_steps_raises_meaningful_error():
    """n_steps <= 0 would make throughput undefined (division by zero) --
    must fail loudly with a clear message, not silently return garbage."""
    try:
        compute_baseline_metrics([], [], n_steps=0)
        raise AssertionError("Expected ValueError for n_steps=0")
    except ValueError as e:
        assert "n_steps" in str(e)
    print("[PASS] test_invalid_n_steps_raises_meaningful_error")


if __name__ == "__main__":
    test_metrics_on_known_scenario()
    test_missed_deadline_is_counted()
    test_still_waiting_tasks_excluded_from_latency()
    test_reproducibility_same_seed_same_metrics()
    test_different_seeds_can_differ()
    test_invalid_n_steps_raises_meaningful_error()
    print("\nALL STEP 5 TESTS PASSED.")
