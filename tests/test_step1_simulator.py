"""
Step 1 - Verification tests for the mini hospital simulator (no AI).
================================================================================
ROMAN URDU: Yeh script simulator ko test karti hai taake hum yaqeen kar
sakein ke queue aur capacity mechanics sahi hain, coding shuru karne se
pehle jaisa project ka rule hai ("test the code after implementation").

ENGLISH: This script tests the simulator so we can confirm the queue and
capacity mechanics behave correctly, per the project rule to always test
code after implementation and report exactly what passed.

Run with:  python3 -m tests.test_step1_simulator   (from MS_Thesis_Project/)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.hospital_env import HospitalSimulator

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "environment.yaml")


def test_single_task_completes_in_one_step():
    """A task smaller than capacity should finish in exactly 1 step."""
    sim = HospitalSimulator(CONFIG_PATH)
    task = sim.add_task("Emergency", compute_demand=2.0)  # capacity = 5.0
    sim.step()
    assert task.is_complete, "Task with demand < capacity should complete in 1 step"
    assert task.completion_time == 0, f"Expected completion_time=0, got {task.completion_time}"
    assert sim.nodes["Emergency"].queue_length == 0
    print("[PASS] test_single_task_completes_in_one_step")


def test_task_spans_multiple_steps_when_larger_than_capacity():
    """A task bigger than capacity must carry over to the next step(s)."""
    sim = HospitalSimulator(CONFIG_PATH)
    task = sim.add_task("ICU", compute_demand=10.0)  # ICU capacity = 4.0 -> needs 3 steps
    sim.step()
    assert not task.is_complete, "Task should NOT finish after 1 step (10.0 demand > 4.0 capacity)"
    assert task.remaining_demand == 6.0, f"Expected 6.0 remaining, got {task.remaining_demand}"
    sim.step()
    assert task.remaining_demand == 2.0, f"Expected 2.0 remaining, got {task.remaining_demand}"
    sim.step()
    assert task.is_complete, "Task should finish by step 3 (4+4+2=10)"
    assert task.completion_time == 2, f"Expected completion_time=2, got {task.completion_time}"
    print("[PASS] test_task_spans_multiple_steps_when_larger_than_capacity")


def test_fifo_ordering_and_capacity_never_exceeded():
    """Node must never do more work in one step than its capacity, and must
    process tasks in FIFO (arrival) order since Step 1 has no scheduling AI."""
    sim = HospitalSimulator(CONFIG_PATH)
    t1 = sim.add_task("General_Ward", compute_demand=2.0)
    t2 = sim.add_task("General_Ward", compute_demand=2.0)
    t3 = sim.add_task("General_Ward", compute_demand=2.0)  # General_Ward capacity = 3.0
    sim.step()
    # Capacity 3.0: t1 (2.0) finishes, t2 gets 1.0 done (1.0 remaining), t3 untouched.
    assert t1.is_complete and t1.completion_time == 0
    assert not t2.is_complete and t2.remaining_demand == 1.0
    assert not t3.is_complete and t3.remaining_demand == 2.0, "t3 should be untouched (FIFO, capacity exhausted)"
    print("[PASS] test_fifo_ordering_and_capacity_never_exceeded")


def test_independent_departments_do_not_interfere():
    """Tasks in one department must not affect another department's capacity."""
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("Emergency", compute_demand=100.0)  # deliberately huge, will never finish soon
    icu_task = sim.add_task("ICU", compute_demand=1.0)
    sim.step()
    assert icu_task.is_complete, "ICU task should complete independently of Emergency's huge task"
    print("[PASS] test_independent_departments_do_not_interfere")


def test_invalid_inputs_raise_meaningful_errors():
    """Input validation: invalid config/task values must raise clear errors."""
    sim = HospitalSimulator(CONFIG_PATH)
    try:
        sim.add_task("Nonexistent_Department", compute_demand=1.0)
        raise AssertionError("Expected ValueError for unknown department")
    except ValueError as e:
        assert "Unknown department" in str(e)
    try:
        sim.add_task("ICU", compute_demand=-5.0)
        raise AssertionError("Expected ValueError for negative compute_demand")
    except ValueError as e:
        assert "compute_demand" in str(e)
    print("[PASS] test_invalid_inputs_raise_meaningful_errors")


if __name__ == "__main__":
    test_single_task_completes_in_one_step()
    test_task_spans_multiple_steps_when_larger_than_capacity()
    test_fifo_ordering_and_capacity_never_exceeded()
    test_independent_departments_do_not_interfere()
    test_invalid_inputs_raise_meaningful_errors()
    print("\nALL STEP 1 TESTS PASSED.")
