"""
Step 3 - Verification tests for Local/Edge/Cloud task routing.
================================================================================
ROMAN URDU: Yeh script confirm karti hai ke TaskRouter sahi tier chunta hai
(compute_demand ke thresholds ke mutabiq), har tier (Local/Edge/Cloud) apni
sahi node par task bhejta hai, Cloud ka network_delay sahi kaam karta hai
(task turant process nahi hota, pehle "transit" mein wait karta hai), aur
Step 1/2 ka purana add_task()/add_generated_arrivals() bilkul waisa hi kaam
karta hai jaisa pehle karta tha (koi routing nahi, hamesha Edge).

ENGLISH: This script confirms TaskRouter picks the right tier (based on
compute_demand thresholds), each tier (Local/Edge/Cloud) dispatches the task
to the correct node, the Cloud tier's network_delay behaves correctly (a
task waits in transit before any processing starts), and Step 1/2's
existing add_task()/add_generated_arrivals() still behave exactly as before
(no routing, always Edge) -- proving Step 3 did not break earlier steps.

Run with:  python3 -m tests.test_step3_routing   (from MS_Thesis_Project/)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.hospital_env import HospitalSimulator
from env.task_generator import MedicalTaskGenerator
from env.task_router import TaskRouter
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "environment.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# From configs/environment.yaml's 'offloading' section:
#   local_threshold=1.5, edge_threshold=6.0, local_capacity=1.0,
#   cloud_capacity=50.0, cloud_network_delay=2


def test_small_task_routes_to_local():
    """demand <= local_threshold (1.5) -> Local tier, on that department's
    own tiny local node, NOT the shared Edge/Cloud node."""
    sim = HospitalSimulator(CONFIG_PATH)
    task = sim.route_task("Emergency", compute_demand=1.0)
    assert task.tier == "Local", f"Expected Local, got {task.tier}"
    assert sim.local_nodes["Emergency"].queue_length == 1
    assert sim.nodes["Emergency"].queue_length == 0
    assert sim.cloud_node.queue_length == 0
    print("[PASS] test_small_task_routes_to_local")


def test_medium_task_routes_to_edge():
    """local_threshold < demand <= edge_threshold -> Edge tier, same node
    Step 1/2 always used (sim.nodes[department])."""
    sim = HospitalSimulator(CONFIG_PATH)
    task = sim.route_task("ICU", compute_demand=3.0)
    assert task.tier == "Edge", f"Expected Edge, got {task.tier}"
    assert sim.nodes["ICU"].queue_length == 1
    assert sim.local_nodes["ICU"].queue_length == 0
    assert sim.cloud_node.queue_length == 0
    print("[PASS] test_medium_task_routes_to_edge")


def test_boundary_values_at_thresholds():
    """Values exactly AT a threshold must fall on the cheaper side
    (<=, not <), per TaskRouter.choose_tier()."""
    sim = HospitalSimulator(CONFIG_PATH)
    at_local = sim.route_task("Emergency", compute_demand=1.5)   # == local_threshold
    at_edge = sim.route_task("Emergency", compute_demand=6.0)    # == edge_threshold
    just_above_edge = sim.route_task("Emergency", compute_demand=6.01)
    assert at_local.tier == "Local", f"1.5 should be Local, got {at_local.tier}"
    assert at_edge.tier == "Edge", f"6.0 should be Edge, got {at_edge.tier}"
    assert just_above_edge.tier == "Cloud", f"6.01 should be Cloud, got {just_above_edge.tier}"
    print("[PASS] test_boundary_values_at_thresholds")


def test_large_task_routes_to_cloud_and_waits_for_network_delay():
    """demand > edge_threshold -> Cloud tier, and the task must NOT be
    touched until cloud_network_delay (2) steps have passed."""
    sim = HospitalSimulator(CONFIG_PATH)
    task = sim.route_task("General_Ward", compute_demand=20.0)
    assert task.tier == "Cloud", f"Expected Cloud, got {task.tier}"
    assert sim.cloud_node.queue_length == 1

    sim.step()  # current_time 0 -> 1: 0 steps elapsed since arrival, still in transit
    assert task.remaining_demand == 20.0, "Task should not be touched during network transit"
    sim.step()  # current_time 1 -> 2: 1 step elapsed, still in transit (< 2)
    assert task.remaining_demand == 20.0, "Task should still be in transit after 1 step"
    sim.step()  # current_time 2 -> 3: 2 steps elapsed, transit over -> cloud processes it
    assert task.is_complete, "Cloud (capacity 50) should finish a 20.0-demand task in 1 step once transit ends"
    assert task.completion_time == 2, f"Expected completion_time=2, got {task.completion_time}"
    print("[PASS] test_large_task_routes_to_cloud_and_waits_for_network_delay")


def test_route_generated_arrivals_integration():
    """Poisson-generated arrivals (Step 2) must flow correctly through
    Step 3's routing (each task's tier set, every task landing in exactly
    one of Local/Edge/Cloud)."""
    sim = HospitalSimulator(CONFIG_PATH)
    gen = MedicalTaskGenerator(CONFIG, seed=42)

    total_created = 0
    for _ in range(30):
        arrivals = gen.generate_for_step(sim.current_time)
        created = sim.route_generated_arrivals(arrivals)
        total_created += len(created)
        for task in created:
            assert task.tier in ("Local", "Edge", "Cloud")
        sim.step()

    assert total_created > 0, "No tasks were generated across 30 steps -- check arrival_rate values"

    counted = (
        sum(node.queue_length + len(node.completed_tasks) for node in sim.nodes.values())
        + sum(node.queue_length + len(node.completed_tasks) for node in sim.local_nodes.values())
        + sim.cloud_node.queue_length + len(sim.cloud_node.completed_tasks)
    )
    assert counted == total_created, (
        f"Every routed task must be findable on exactly one node: expected {total_created}, found {counted}"
    )
    print(f"[PASS] test_route_generated_arrivals_integration ({total_created} tasks routed)")


def test_step1_add_task_still_bypasses_routing():
    """Step 1/2's add_task() must still ALWAYS use the Edge node directly --
    Step 3 routing is opt-in via route_task(), not a change to add_task()."""
    sim = HospitalSimulator(CONFIG_PATH)
    task = sim.add_task("Emergency", compute_demand=1.0)  # would be "Local" if routed
    assert task.tier is None, "add_task() must not set a tier (unchanged Step 1/2 behaviour)"
    assert sim.nodes["Emergency"].queue_length == 1
    assert sim.local_nodes["Emergency"].queue_length == 0
    print("[PASS] test_step1_add_task_still_bypasses_routing")


def test_invalid_inputs_raise_meaningful_errors():
    """Bad department names, non-positive compute_demand, and a missing
    'offloading' config section must all fail with clear messages."""
    sim = HospitalSimulator(CONFIG_PATH)
    try:
        sim.route_task("Nonexistent_Department", compute_demand=1.0)
        raise AssertionError("Expected ValueError for unknown department")
    except ValueError as e:
        assert "Unknown department" in str(e)

    try:
        sim.route_task("ICU", compute_demand=-1.0)
        raise AssertionError("Expected ValueError for non-positive compute_demand")
    except ValueError as e:
        assert "compute_demand" in str(e)

    try:
        TaskRouter(local_threshold=5.0, edge_threshold=5.0)  # must be strictly >
        raise AssertionError("Expected ValueError for edge_threshold <= local_threshold")
    except ValueError as e:
        assert "edge_threshold" in str(e)

    no_offloading_config = {k: v for k, v in CONFIG.items() if k != "offloading"}
    tmp_path = os.path.join(os.path.dirname(CONFIG_PATH), "_tmp_no_offloading.yaml")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(no_offloading_config, f)
    try:
        sim_no_routing = HospitalSimulator(tmp_path)
        assert sim_no_routing.router is None
        try:
            sim_no_routing.route_task("ICU", compute_demand=1.0)
            raise AssertionError("Expected ValueError when config has no 'offloading' section")
        except ValueError as e:
            assert "offloading" in str(e)
    finally:
        os.remove(tmp_path)

    print("[PASS] test_invalid_inputs_raise_meaningful_errors")


if __name__ == "__main__":
    test_small_task_routes_to_local()
    test_medium_task_routes_to_edge()
    test_boundary_values_at_thresholds()
    test_large_task_routes_to_cloud_and_waits_for_network_delay()
    test_route_generated_arrivals_integration()
    test_step1_add_task_still_bypasses_routing()
    test_invalid_inputs_raise_meaningful_errors()
    print("\nALL STEP 3 TESTS PASSED.")
