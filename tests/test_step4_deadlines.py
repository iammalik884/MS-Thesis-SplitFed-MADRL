"""
Step 4 - Verification tests for priority-based deadlines.
================================================================================
ROMAN URDU: Yeh tests confirm karte hain ke: (1) deadline sahi priority ke
configured offset se calculate hoti hai (current_time + offset), (2) bina
priority ke task ki deadline None rehti hai (backward compatibility), (3)
`met_deadline` sahi teeno states dikhata hai -- None (abhi complete nahi),
True (waqt par), False (der se), boundary (completion_time == deadline)
sahi True count hota hai, (4) deadline arrival_time se pehle nahi ho sakti,
(5) route_task() aur add_task() dono deadline set karte hain, (6) 'deadlines'
config section poori tarah opt-in hai (jaise Step 3 ka 'offloading'), (7)
galat config par saaf error aata hai, aur (8) Poisson-generated arrivals
(Step 2) ke through bhi deadline sahi pass hoti hai.

ENGLISH: These tests confirm: (1) deadline is computed correctly as
current_time + the configured per-priority offset, (2) a task with no
priority keeps deadline=None (backward compatibility), (3) `met_deadline`
correctly reports all three states -- None (not complete yet), True
(on time), False (late), with the boundary case (completion_time exactly
equal to deadline) correctly counting as True, (4) a deadline before
arrival_time is rejected, (5) both add_task() and route_task() set it,
(6) the 'deadlines' config section is fully opt-in (same pattern as Step
3's 'offloading'), (7) a bad config raises a clear error, and (8) Poisson-
generated arrivals (Step 2) get correct deadlines end-to-end.

Run with:  python3 -m tests.test_step4_deadlines   (from MS_Thesis_Project/)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from env.hospital_env import HospitalSimulator
from env.task import Task
from env.task_generator import MedicalTaskGenerator

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "environment.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# From configs/environment.yaml's 'deadlines' section:
#   Critical: 2, High: 5, Routine: 10


def test_deadline_computed_from_priority_and_arrival_time():
    """deadline = current_time (at the moment of creation) + the configured
    offset for that task's priority."""
    sim = HospitalSimulator(CONFIG_PATH)
    critical_task = sim.add_task("Emergency", compute_demand=1.0, priority="Critical")
    assert critical_task.deadline == 0 + 2, f"Expected deadline=2, got {critical_task.deadline}"

    sim.step()
    sim.step()
    sim.step()  # current_time is now 3
    high_task = sim.add_task("ICU", compute_demand=1.5, priority="High")
    assert high_task.deadline == 3 + 5, f"Expected deadline=8, got {high_task.deadline}"
    print("[PASS] test_deadline_computed_from_priority_and_arrival_time")


def test_no_priority_means_no_deadline():
    """Step 1 style add_task() (no priority) must keep deadline=None --
    backward compatibility, nothing before Step 4 breaks."""
    sim = HospitalSimulator(CONFIG_PATH)
    task = sim.add_task("General_Ward", compute_demand=1.0)
    assert task.priority is None
    assert task.deadline is None, "No priority given -> no deadline should be set"
    print("[PASS] test_no_priority_means_no_deadline")


def test_met_deadline_all_three_states():
    """met_deadline must be None before completion, True when finished on
    or before the deadline (boundary case: completion_time == deadline
    counts as met), and False when finished late."""
    not_done = Task(task_id=1, arrival_time=0, compute_demand=1.0, priority="Critical", deadline=2)
    assert not_done.met_deadline is None, "Incomplete task must report met_deadline=None"

    on_time = Task(task_id=2, arrival_time=0, compute_demand=1.0, priority="Critical", deadline=2)
    on_time.completion_time = 2  # exactly on the deadline -> counts as met
    assert on_time.met_deadline is True, "completion_time == deadline must count as met (True)"

    late = Task(task_id=3, arrival_time=0, compute_demand=1.0, priority="Critical", deadline=2)
    late.completion_time = 3
    assert late.met_deadline is False, "completion_time > deadline must count as missed (False)"
    print("[PASS] test_met_deadline_all_three_states")


def test_deadline_before_arrival_rejected():
    """A deadline earlier than arrival_time is nonsensical and must raise."""
    try:
        Task(task_id=4, arrival_time=5, compute_demand=1.0, deadline=3)
        raise AssertionError("Expected ValueError for deadline before arrival_time")
    except ValueError as e:
        assert "deadline" in str(e) and "arrival_time" in str(e)
    print("[PASS] test_deadline_before_arrival_rejected")


def test_route_task_also_sets_deadline():
    """STEP 3+4 integration: route_task() must set deadline exactly like
    add_task() does, in addition to setting tier."""
    sim = HospitalSimulator(CONFIG_PATH)
    task = sim.route_task("General_Ward", compute_demand=7.0, priority="Routine")  # -> Cloud tier
    assert task.tier == "Cloud"
    assert task.deadline == 0 + 10, f"Expected deadline=10, got {task.deadline}"
    print("[PASS] test_route_task_also_sets_deadline")


def test_deadlines_are_opt_in_missing_config_section():
    """Remove 'deadlines' from the config entirely: HospitalSimulator must
    still work exactly as before, deadline just stays None -- no error."""
    no_deadlines_config = {k: v for k, v in CONFIG.items() if k != "deadlines"}
    tmp_path = os.path.join(os.path.dirname(CONFIG_PATH), "_tmp_no_deadlines.yaml")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(no_deadlines_config, f)
    try:
        sim = HospitalSimulator(tmp_path)
        assert sim.deadline_offsets is None
        task = sim.add_task("ICU", compute_demand=1.0, priority="Critical")
        assert task.deadline is None, "No 'deadlines' section -> deadline must stay None, not error"
        assert task.priority == "Critical", "Priority itself must still be set normally"
    finally:
        os.remove(tmp_path)
    print("[PASS] test_deadlines_are_opt_in_missing_config_section")


def test_invalid_deadlines_config_raises_meaningful_errors():
    """A 'deadlines' section missing a priority class, or with a negative
    offset, must fail fast with a clear message."""
    missing_priority_config = dict(CONFIG)
    missing_priority_config["deadlines"] = {"Critical": 2, "High": 5}  # 'Routine' missing on purpose
    tmp_path = os.path.join(os.path.dirname(CONFIG_PATH), "_tmp_bad_deadlines_1.yaml")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(missing_priority_config, f)
    try:
        HospitalSimulator(tmp_path)
        raise AssertionError("Expected ValueError for missing 'Routine' priority class")
    except ValueError as e:
        assert "Routine" in str(e)
    finally:
        os.remove(tmp_path)

    negative_offset_config = dict(CONFIG)
    negative_offset_config["deadlines"] = {"Critical": -1, "High": 5, "Routine": 10}
    tmp_path = os.path.join(os.path.dirname(CONFIG_PATH), "_tmp_bad_deadlines_2.yaml")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(negative_offset_config, f)
    try:
        HospitalSimulator(tmp_path)
        raise AssertionError("Expected ValueError for negative deadline offset")
    except ValueError as e:
        assert "deadlines.Critical" in str(e)
    finally:
        os.remove(tmp_path)

    print("[PASS] test_invalid_deadlines_config_raises_meaningful_errors")


def test_generated_arrivals_get_deadlines_end_to_end():
    """Integration: Poisson-generated arrivals (Step 2), routed through
    Local/Edge/Cloud (Step 3), must each get the correct deadline for their
    priority, and every completed one must resolve met_deadline to True/False
    (never None, since the real config always has 'deadlines' + they always
    have a priority)."""
    sim = HospitalSimulator(CONFIG_PATH)
    gen = MedicalTaskGenerator(CONFIG, seed=42)
    offsets = CONFIG["deadlines"]

    all_tasks = []
    for _ in range(30):
        arrivals = gen.generate_for_step(sim.current_time)
        created = sim.route_generated_arrivals(arrivals)
        for task in created:
            expected_deadline = task.arrival_time + offsets[task.priority]
            assert task.deadline == expected_deadline, (
                f"Task {task.task_id} ({task.priority}): expected deadline "
                f"{expected_deadline}, got {task.deadline}"
            )
            all_tasks.append(task)
        sim.step()

    for _ in range(15):  # drain steps, let queued work finish
        sim.step()

    completed = [t for t in all_tasks if t.is_complete]
    assert completed, "No tasks completed in 45 steps -- test did not actually check met_deadline"
    for task in completed:
        assert task.met_deadline in (True, False), (
            f"Task {task.task_id} completed but met_deadline is {task.met_deadline!r}, "
            f"expected True or False"
        )
    met_count = sum(1 for t in completed if t.met_deadline)
    print(
        f"[PASS] test_generated_arrivals_get_deadlines_end_to_end "
        f"({len(completed)}/{len(all_tasks)} completed, {met_count} met their deadline)"
    )


if __name__ == "__main__":
    test_deadline_computed_from_priority_and_arrival_time()
    test_no_priority_means_no_deadline()
    test_met_deadline_all_three_states()
    test_deadline_before_arrival_rejected()
    test_route_task_also_sets_deadline()
    test_deadlines_are_opt_in_missing_config_section()
    test_invalid_deadlines_config_raises_meaningful_errors()
    test_generated_arrivals_get_deadlines_end_to_end()
    print("\nALL STEP 4 TESTS PASSED.")
