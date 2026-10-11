"""
Step 3 - Verification tests for Local / Edge(neighbour) / Cloud task routing.
================================================================================
ROMAN URDU: Yeh script confirm karti hai ke TaskRouter (V2, roadmap wala
action space) sahi jagah chunta hai: Local = apna department, Edge = sab se
kam backlog wala PADOSI department, Cloud = shared server; Cloud ka network
delay sahi kaam karta hai; aur Step 1/2 ka add_task() abhi bhi bina routing
ke apne hi node par task daalta hai.

ENGLISH: Confirms the V2 routing rule (Local = own department node, Edge =
least-loaded NEIGHBOUR department node, Cloud = shared server), the Cloud
network delay, and that Step 1/2's add_task() still bypasses routing.

UPDATED EXPECTATIONS (V2 audit fix) -- why each old test changed:
  * test_small_task_routes_to_local / test_medium_task_routes_to_edge: the
    old tests asserted a separate tiny `local_nodes[...]` bedside tier and
    "Edge = own department node". That tier was outside thesis scope and has
    been removed; Local now means the department's OWN node and Edge a
    NEIGHBOUR node, so these tests now check that meaning instead.
  * test_boundary_values_at_thresholds: thresholds are now cloud_threshold
    (compute demand) and offload_threshold (source backlog); both boundaries
    are still checked on the "cheaper" side (<=, not <).
  * test_step1_add_task_still_bypasses_routing: add_task() still never
    routes, but the task is now labelled tier="Local" (it executes on its
    own department node, which IS the Local tier) instead of tier=None.
  * test_invalid_inputs_raise_meaningful_errors: TaskRouter's constructor
    arguments changed; old pre-v2 config keys must now fail loudly.
No test was deleted; every behaviour the old tests covered is still covered.

Run with:  py -m pytest tests/test_step3_routing.py   (from MS_Thesis_Project/)
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from env.hospital_env import HospitalSimulator
from env.task_generator import MedicalTaskGenerator
from env.task_router import TaskRouter

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "environment.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# From configs/environment.yaml: capacities Emergency 5, ICU 4, General_Ward 3;
# offloading: cloud_threshold 6.0, offload_threshold 3.0, cloud_capacity 50,
# cloud_network_delay 2, neighbour_network_delay 1,
# tie-break order Emergency -> ICU -> General_Ward.


def test_small_task_routes_to_local():
    """Idle source node (backlog 0 <= 3.0) and demand <= 6.0 -> Local, i.e. the
    source department's OWN edge node (no separate bedside node any more)."""
    sim = HospitalSimulator(CONFIG_PATH)
    task = sim.route_task("Emergency", compute_demand=1.0)
    assert task.tier == "Local" and task.destination == "Emergency"
    assert task.source_department == "Emergency"
    assert sim.nodes["Emergency"].queue_length == 1
    assert sim.cloud_node.queue_length == 0
    assert not hasattr(sim, "local_nodes"), "the pre-v2 bedside Local tier must be gone"
    print("[PASS] test_small_task_routes_to_local")


def test_medium_task_routes_to_edge():
    """Busy source (backlog > offload_threshold) -> Edge = a NEIGHBOUR's node,
    never the source itself."""
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("ICU", compute_demand=5.0)            # ICU backlog 5.0 > 3.0
    task = sim.route_task("ICU", compute_demand=3.0)
    assert task.tier == "Edge", f"Expected Edge, got {task.tier}"
    assert task.destination == "Emergency", "both neighbours idle -> tie -> Emergency first"
    assert task in sim.nodes["Emergency"].queue
    assert task not in sim.nodes["ICU"].queue
    print("[PASS] test_medium_task_routes_to_edge")


def test_boundary_values_at_thresholds():
    """Exactly AT a threshold stays on the cheaper side (<=, not <)."""
    sim = HospitalSimulator(CONFIG_PATH)
    at_cloud = sim.route_task("Emergency", compute_demand=6.0)       # == cloud_threshold
    just_above = sim.route_task("Emergency", compute_demand=6.01)
    assert at_cloud.tier != "Cloud", f"6.0 must not go to Cloud, got {at_cloud.tier}"
    assert just_above.tier == "Cloud", f"6.01 should be Cloud, got {just_above.tier}"

    sim2 = HospitalSimulator(CONFIG_PATH)
    sim2.add_task("ICU", compute_demand=3.0)                         # backlog == offload_threshold
    assert sim2.route_task("ICU", compute_demand=1.0).tier == "Local"
    sim3 = HospitalSimulator(CONFIG_PATH)
    sim3.add_task("ICU", compute_demand=3.01)                        # just above
    assert sim3.route_task("ICU", compute_demand=1.0).tier == "Edge"
    print("[PASS] test_boundary_values_at_thresholds")


def test_large_task_routes_to_cloud_and_waits_for_network_delay():
    """demand > cloud_threshold -> Cloud, untouched until cloud_network_delay (2)
    steps have passed (ready_time = 0 + 2). Unchanged from the original test."""
    sim = HospitalSimulator(CONFIG_PATH)
    task = sim.route_task("General_Ward", compute_demand=20.0)
    assert task.tier == "Cloud" and task.destination == "Cloud"
    assert sim.cloud_node.queue_length == 1
    assert task.ready_time == 2

    sim.step()  # t=0: in transit
    assert task.remaining_demand == 20.0, "Task should not be touched during network transit"
    sim.step()  # t=1: in transit
    assert task.remaining_demand == 20.0, "Task should still be in transit after 1 step"
    sim.step()  # t=2: ready -> processed
    assert task.is_complete
    assert task.completion_time == 2, f"Expected completion_time=2, got {task.completion_time}"
    print("[PASS] test_large_task_routes_to_cloud_and_waits_for_network_delay")


def test_route_generated_arrivals_integration():
    """Poisson arrivals flow through routing: every task gets a tier/destination
    and is findable on exactly one node (task conservation)."""
    sim = HospitalSimulator(CONFIG_PATH)
    gen = MedicalTaskGenerator(CONFIG, seed=42)

    total_created = 0
    for _ in range(30):
        created = sim.route_generated_arrivals(gen.generate_for_step(sim.current_time))
        total_created += len(created)
        for task in created:
            assert task.tier in ("Local", "Edge", "Cloud")
            assert (task.tier == "Local") == (task.destination == task.source_department)
        sim.step()

    assert total_created > 0
    counted = sum(node.queue_length + len(node.completed_tasks) for node in sim.all_nodes)
    assert counted == total_created, f"expected {total_created} tasks on nodes, found {counted}"
    print(f"[PASS] test_route_generated_arrivals_integration ({total_created} tasks routed)")


def test_step1_add_task_still_bypasses_routing():
    """add_task() never consults the router: even a busy source keeps the task
    on its own node. V2: labelled tier='Local' (that is where it executes)."""
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("Emergency", compute_demand=10.0)                   # busy, router would offload
    task = sim.add_task("Emergency", compute_demand=1.0)
    assert task.tier == "Local" and task.destination == "Emergency"
    assert sim.nodes["Emergency"].queue_length == 2
    assert sim.nodes["ICU"].queue_length == 0 and sim.cloud_node.queue_length == 0
    print("[PASS] test_step1_add_task_still_bypasses_routing")


def test_invalid_inputs_raise_meaningful_errors():
    """Bad department, non-positive demand, bad router arguments, stale pre-v2
    config keys and a missing 'offloading' section all fail clearly."""
    sim = HospitalSimulator(CONFIG_PATH)
    for bad_call, expected in [
        (lambda: sim.route_task("Nonexistent_Department", compute_demand=1.0), "Unknown department"),
        (lambda: sim.route_task("ICU", compute_demand=-1.0), "compute_demand"),
        (lambda: TaskRouter(cloud_threshold=0.0, offload_threshold=1.0, neighbour_order=["A", "B"]), "cloud_threshold"),
        (lambda: TaskRouter(cloud_threshold=6.0, offload_threshold=-1.0, neighbour_order=["A", "B"]), "offload_threshold"),
        (lambda: TaskRouter(cloud_threshold=6.0, offload_threshold=1.0, neighbour_order=["A", "A"]), "neighbour_order"),
    ]:
        try:
            bad_call()
            raise AssertionError(f"Expected ValueError mentioning {expected!r}")
        except ValueError as e:
            assert expected in str(e), str(e)

    stale = copy.deepcopy(CONFIG)
    stale["offloading"]["local_capacity"] = 1.0
    try:
        HospitalSimulator(stale)
        raise AssertionError("Expected ValueError for removed pre-v2 key local_capacity")
    except ValueError as e:
        assert "local_capacity" in str(e)

    no_offloading = {k: v for k, v in CONFIG.items() if k != "offloading"}
    sim_no_routing = HospitalSimulator(no_offloading)
    assert sim_no_routing.router is None
    try:
        sim_no_routing.route_task("ICU", compute_demand=1.0)
        raise AssertionError("Expected ValueError when config has no 'offloading' section")
    except ValueError as e:
        assert "offloading" in str(e)
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
