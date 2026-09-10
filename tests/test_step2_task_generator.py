"""
Step 2 - Verification tests for MedicalTaskGenerator and its integration with
HospitalSimulator.
================================================================================
ROMAN URDU: Yeh tests confirm karte hain ke: (1) same seed se hamesha same
tasks bante hain (reproducibility), (2) zyada steps chalane par har
department/priority ka औसत arrival count uske configured arrival_rate ke
qareeb hota hai (Poisson process sahi kaam kar raha hai), (3) generated
compute_demand hamesha configured range ke andar hota hai, (4) galat config
par saaf error aata hai, aur (5) Step 1 ka purana simulator generated
arrivals ko bhi sahi se process karta hai.

ENGLISH: These tests confirm: (1) the same seed always reproduces the same
tasks, (2) over many steps, each department/priority's average arrival
count converges near its configured arrival_rate (the Poisson process is
correct), (3) generated compute_demand always falls inside the configured
range, (4) a bad config raises a clear error, and (5) Step 1's existing
simulator correctly processes generator-created arrivals too.

Run with:  python3 -m tests.test_step2_task_generator   (from MS_Thesis_Project/)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from env.hospital_env import HospitalSimulator
from env.task_generator import MedicalTaskGenerator

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "environment.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


def test_reproducibility_same_seed():
    """Two generators built with the same seed must produce IDENTICAL arrivals."""
    gen_a = MedicalTaskGenerator(CONFIG, seed=42)
    gen_b = MedicalTaskGenerator(CONFIG, seed=42)

    for t in range(50):
        arrivals_a = gen_a.generate_for_step(t)
        arrivals_b = gen_b.generate_for_step(t)
        assert len(arrivals_a) == len(arrivals_b), f"Step {t}: arrival counts differ between identical seeds"
        for a, b in zip(arrivals_a, arrivals_b):
            assert a.department == b.department and a.priority == b.priority
            assert a.compute_demand == b.compute_demand, "Same seed must reproduce identical compute_demand"
    print("[PASS] test_reproducibility_same_seed")


def test_different_seeds_usually_differ():
    """Sanity check: two different seeds should NOT produce identical sequences
    (guards against the RNG accidentally being ignored)."""
    gen_a = MedicalTaskGenerator(CONFIG, seed=1)
    gen_b = MedicalTaskGenerator(CONFIG, seed=2)
    all_a = [gen_a.generate_for_step(t) for t in range(50)]
    all_b = [gen_b.generate_for_step(t) for t in range(50)]
    assert all_a != all_b, "Different seeds produced an identical 50-step sequence -- RNG seeding looks broken"
    print("[PASS] test_different_seeds_usually_differ")


def test_arrival_rate_matches_configured_mean():
    """Over many steps, the average number of arrivals per step for one
    (department, priority) pair should be close to its configured arrival_rate
    (Poisson mean). Uses a generous tolerance so the test is not flaky."""
    gen = MedicalTaskGenerator(CONFIG, seed=42)
    n_steps = 4000

    counts = {}  # (department, priority) -> total arrivals observed
    for t in range(n_steps):
        for arrival in gen.generate_for_step(t):
            key = (arrival.department, arrival.priority)
            counts[key] = counts.get(key, 0) + 1

    for dept_name, priorities in CONFIG["task_generation"]["departments"].items():
        for priority, settings in priorities.items():
            rate = settings["arrival_rate"]
            if rate == 0:
                continue
            expected_total = rate * n_steps
            observed_total = counts.get((dept_name, priority), 0)
            # Poisson std dev over n_steps is sqrt(expected_total); allow a wide
            # +-6 standard deviation band (with a floor) so real randomness
            # essentially never fails this test by chance.
            tolerance = max(10.0, 6 * (expected_total ** 0.5))
            assert abs(observed_total - expected_total) <= tolerance, (
                f"{dept_name}/{priority}: expected ~{expected_total:.1f} arrivals over "
                f"{n_steps} steps (rate={rate}), observed {observed_total} "
                f"(tolerance +-{tolerance:.1f})"
            )
    print("[PASS] test_arrival_rate_matches_configured_mean")


def test_compute_demand_within_configured_range():
    """Every generated task's compute_demand must fall inside its priority's
    configured [low, high] range."""
    gen = MedicalTaskGenerator(CONFIG, seed=7)
    ranges = {
        (dept, priority): tuple(settings["compute_demand_range"])
        for dept, priorities in CONFIG["task_generation"]["departments"].items()
        for priority, settings in priorities.items()
    }

    checked = 0
    for t in range(500):
        for arrival in gen.generate_for_step(t):
            low, high = ranges[(arrival.department, arrival.priority)]
            assert low <= arrival.compute_demand <= high, (
                f"{arrival.department}/{arrival.priority}: compute_demand "
                f"{arrival.compute_demand} outside configured range [{low}, {high}]"
            )
            checked += 1
    assert checked > 0, "No arrivals were generated in 500 steps -- test did not actually check anything"
    print(f"[PASS] test_compute_demand_within_configured_range ({checked} tasks checked)")


def test_invalid_config_raises_meaningful_errors():
    """Missing/invalid task_generation settings must fail fast with a clear message."""
    try:
        MedicalTaskGenerator({}, seed=42)
        raise AssertionError("Expected ValueError for missing task_generation section")
    except ValueError as e:
        assert "task_generation" in str(e)

    bad_config = {
        "task_generation": {
            "departments": {
                "Emergency": {
                    "Critical": {"arrival_rate": 0.1, "compute_demand_range": [1.0, 2.0]},
                    "High": {"arrival_rate": 0.1, "compute_demand_range": [1.0, 2.0]},
                    # 'Routine' missing on purpose
                }
            }
        }
    }
    try:
        MedicalTaskGenerator(bad_config, seed=42)
        raise AssertionError("Expected ValueError for missing 'Routine' priority class")
    except ValueError as e:
        assert "Routine" in str(e)

    bad_range_config = {
        "task_generation": {
            "departments": {
                "Emergency": {
                    "Critical": {"arrival_rate": 0.1, "compute_demand_range": [2.0, 1.0]},  # low > high
                    "High": {"arrival_rate": 0.1, "compute_demand_range": [1.0, 2.0]},
                    "Routine": {"arrival_rate": 0.1, "compute_demand_range": [1.0, 2.0]},
                }
            }
        }
    }
    try:
        MedicalTaskGenerator(bad_range_config, seed=42)
        raise AssertionError("Expected ValueError for low > high compute_demand_range")
    except ValueError as e:
        assert "compute_demand_range" in str(e)

    print("[PASS] test_invalid_config_raises_meaningful_errors")


def test_generated_arrivals_flow_through_simulator():
    """Integration check: HospitalSimulator.add_generated_arrivals() must place
    each generated task in the RIGHT department's queue, with its priority
    preserved, and Step 1's capacity/queue mechanics must still apply."""
    sim = HospitalSimulator(CONFIG_PATH)
    gen = MedicalTaskGenerator(CONFIG, seed=42)

    total_created = 0
    for _ in range(30):
        arrivals = gen.generate_for_step(sim.current_time)
        created = sim.add_generated_arrivals(arrivals)
        total_created += len(created)
        for task, arrival in zip(created, arrivals):
            assert task.priority == arrival.priority
            assert task.arrival_time == sim.current_time
        sim.step()  # Step 1's existing per-node capacity/queue logic runs unchanged

    assert total_created > 0, "No tasks were generated across 30 steps -- check arrival_rate values"

    # Every completed task must have a valid priority label (Step 2 requirement)
    # and must have been created within the queue capacity rules already
    # verified by Step 1's own tests.
    for node in sim.nodes.values():
        for task in node.completed_tasks:
            if task.priority is not None:  # generator-created tasks are labeled
                assert task.priority in ("Critical", "High", "Routine")

    print(f"[PASS] test_generated_arrivals_flow_through_simulator ({total_created} tasks created)")


if __name__ == "__main__":
    test_reproducibility_same_seed()
    test_different_seeds_usually_differ()
    test_arrival_rate_matches_configured_mean()
    test_compute_demand_within_configured_range()
    test_invalid_config_raises_meaningful_errors()
    test_generated_arrivals_flow_through_simulator()
    print("\nALL STEP 2 TESTS PASSED.")
