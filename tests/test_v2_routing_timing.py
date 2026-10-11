"""
V2 audit fixes (part a) - neighbour routing, per-task lifecycle timing,
destination-ready FCFS, utilization, and same-step arrival order.
================================================================================
ROMAN URDU: Har test ek chhoti, haath se calculate ki hui example hai. Har
assert ke saath comment mein likha hai ke number kahan se aaya, taake aap
isay kaghaz par khud dobara calculate kar sakein.

ENGLISH: Every test is a small hand-calculated example; the comments next
to each assert show where each number comes from.

Default config numbers used below: capacities Emergency 5, ICU 4,
General_Ward 3; offload_threshold 3.0; cloud_threshold 6.0; neighbour delay 1;
cloud capacity 50, cloud delay 2; deadlines Critical 2 / High 5 / Routine 10.

Run with:  py -m pytest tests/test_v2_routing_timing.py
"""

import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from env.hospital_env import HospitalSimulator
from env.task_generator import MedicalTaskGenerator

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "environment.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


def make_config(**offloading_overrides):
    cfg = copy.deepcopy(CONFIG)
    cfg["offloading"].update(offloading_overrides)
    return cfg


# ---------------------------------------------------------------------------
# 1. Edge = a neighbour's REAL node (shared queue + capacity)
# ---------------------------------------------------------------------------
def test_icu_offload_uses_emergency_real_queue_and_capacity():
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("ICU", compute_demand=8.0)                       # ICU backlog 8 > 3 -> ICU is "busy"
    off = sim.route_task("ICU", compute_demand=2.0, priority="Critical")
    assert (off.tier, off.destination, off.source_department) == ("Edge", "Emergency", "ICU")
    emergency = sim.nodes["Emergency"]
    assert off in emergency.queue                                  # Emergency's OWN queue
    assert emergency.backlog_demand == 2.0                         # reservation visible immediately
    assert emergency.transit_demand() == 2.0 and emergency.ready_demand() == 0.0
    assert emergency.projected_utilization() == 0.0                # in transit -> no CPU at t=0

    sim.step()                                                     # t=0: nothing ready at Emergency
    assert emergency.last_step_utilization == 0.0

    own = sim.add_task("Emergency", compute_demand=5.0)            # t=1, ready 1, enqueued AFTER off
    sim.step()                                                     # t=1: capacity 5 = off 2.0 + own 3.0
    assert off.completion_time == 1
    assert own.remaining_demand == 2.0, "ICU's task took 2 of Emergency's 5 units"
    assert emergency.last_step_utilization == 1.0
    sim.step()                                                     # t=2: own finishes
    assert own.completion_time == 2
    # Lifecycle of the offloaded task: arrival 0, ready 1, start 1, completion 1.
    assert (off.transmission_delay, off.queue_wait, off.execution_span, off.latency) == (1, 0, 0, 1)
    print("[PASS] test_icu_offload_uses_emergency_real_queue_and_capacity")


def test_shared_receiving_node_contention():
    """Two departments offload to the same neighbour: they share its 5 units."""
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("ICU", compute_demand=8.0)
    sim.add_task("General_Ward", compute_demand=8.0)
    a = sim.route_task("ICU", compute_demand=4.0)           # neighbours E 0, GW 8 -> Emergency
    b = sim.route_task("General_Ward", compute_demand=4.0)  # neighbours E 4 (reserved), ICU 8 -> Emergency
    assert a.destination == b.destination == "Emergency"
    sim.step()                                              # t=0: both in transit
    sim.step()                                              # t=1: a gets 4, b gets 1 (FCFS by enqueue)
    assert a.completion_time == 1 and b.remaining_demand == 3.0
    sim.step()                                              # t=2: b gets 3 -> done
    assert b.completion_time == 2 and b.queue_wait == 0 and b.execution_span == 1
    print("[PASS] test_shared_receiving_node_contention")


def test_in_transit_reservation_counts_in_routing_backlog():
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("ICU", compute_demand=8.0)
    first = sim.route_task("ICU", compute_demand=5.0)       # -> Emergency (tie at 0, Emergency first)
    second = sim.route_task("ICU", compute_demand=1.0)      # Emergency backlog 5 (in transit) vs GW 0
    assert first.destination == "Emergency" and second.destination == "General_Ward"
    print("[PASS] test_in_transit_reservation_counts_in_routing_backlog")


def test_tie_break_order_and_source_exclusion():
    for source, expected in [("Emergency", "ICU"), ("ICU", "Emergency"), ("General_Ward", "Emergency")]:
        sim = HospitalSimulator(CONFIG_PATH)
        sim.add_task(source, compute_demand=10.0)
        task = sim.route_task(source, compute_demand=1.0)
        assert task.destination == expected and task.destination != source
    # Configurable order is honoured.
    sim = HospitalSimulator(make_config(neighbour_tie_break_order=["General_Ward", "ICU", "Emergency"]))
    sim.add_task("ICU", compute_demand=10.0)
    assert sim.route_task("ICU", compute_demand=1.0).destination == "General_Ward"
    print("[PASS] test_tie_break_order_and_source_exclusion")


def test_routing_is_priority_blind():
    results = set()
    for priority in ("Critical", "High", "Routine", None):
        sim = HospitalSimulator(CONFIG_PATH)
        sim.add_task("ICU", compute_demand=8.0)
        t = sim.route_task("ICU", compute_demand=2.0, priority=priority)
        results.add((t.tier, t.destination))
    assert results == {("Edge", "Emergency")}
    print("[PASS] test_routing_is_priority_blind")


def test_critical_task_cloud_eligibility_is_demand_only():
    """Critical tasks are NOT excluded from Cloud by priority -- only by demand.
    With the default generator ranges (Critical <= 2.5 < 6.0) they never
    qualify; a crafted 7.0-demand Critical task does."""
    sim = HospitalSimulator(CONFIG_PATH)
    t = sim.route_task("ICU", compute_demand=7.0, priority="Critical")
    assert t.tier == "Cloud" and t.ready_time == 2
    sim.run(3)
    assert t.completion_time == 2 and t.met_deadline is True        # 2 <= deadline 0+2
    max_critical = max(p["Critical"]["compute_demand_range"][1] for p in CONFIG["task_generation"]["departments"].values())
    assert max_critical <= CONFIG["offloading"]["cloud_threshold"]
    print("[PASS] test_critical_task_cloud_eligibility_is_demand_only")


# ---------------------------------------------------------------------------
# 2. Network readiness + destination-ready FCFS
# ---------------------------------------------------------------------------
def test_zero_neighbour_delay_processes_same_step():
    sim = HospitalSimulator(make_config(neighbour_network_delay=0))
    sim.add_task("ICU", compute_demand=8.0)
    t = sim.route_task("ICU", compute_demand=2.0)
    sim.step()
    assert t.destination == "Emergency" and t.ready_time == 0
    assert t.completion_time == 0 and t.latency == 0
    print("[PASS] test_zero_neighbour_delay_processes_same_step")


def test_slow_neighbour_delay_larger_than_cloud_is_allowed():
    sim = HospitalSimulator(make_config(neighbour_network_delay=5))
    sim.add_task("ICU", compute_demand=8.0)
    t = sim.route_task("ICU", compute_demand=2.0)
    sim.run(6)
    assert t.ready_time == 5 and t.start_time == 5 and t.completion_time == 5 and t.latency == 5
    print("[PASS] test_slow_neighbour_delay_larger_than_cloud_is_allowed")


def test_transit_task_does_not_block_ready_task():
    """An offloaded task enqueued FIRST but still in transit must not stop a
    ready local task from using the CPU (destination-ready FCFS)."""
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("ICU", compute_demand=8.0)
    transit = sim.route_task("ICU", compute_demand=2.0)        # -> Emergency, ready 1
    ready = sim.add_task("Emergency", compute_demand=3.0)      # ready 0, enqueued later
    assert sim.nodes["Emergency"].queue == [ready, transit]    # service order = ready_time order
    sim.step()                                                 # t=0
    assert ready.completion_time == 0 and transit.start_time is None
    assert sim.nodes["Emergency"].last_step_utilization == 0.6  # 3/5, transit uses no CPU
    print("[PASS] test_transit_task_does_not_block_ready_task")


def test_equal_ready_time_served_in_enqueue_order():
    sim = HospitalSimulator(CONFIG_PATH)
    tasks = [sim.add_task("General_Ward", compute_demand=2.0) for _ in range(3)]
    sim.run(2)  # capacity 3: t=0 -> t0 done, t1 1/2; t=1 -> t1 done, t2 2/2 done
    assert [t.completion_time for t in tasks] == [0, 1, 1]
    assert [t.start_time for t in tasks] == [0, 0, 1]
    assert [t.queue_wait for t in tasks] == [0, 0, 1]
    print("[PASS] test_equal_ready_time_served_in_enqueue_order")


def test_cloud_and_neighbour_ready_time_rule_is_identical():
    sim = HospitalSimulator(make_config(neighbour_network_delay=2))
    sim.add_task("ICU", compute_demand=8.0)
    edge = sim.route_task("ICU", compute_demand=2.0)
    cloud = sim.route_task("ICU", compute_demand=7.0)
    assert edge.ready_time == cloud.ready_time == 2
    sim.run(3)
    assert edge.start_time == cloud.start_time == 2 and edge.completion_time == cloud.completion_time == 2
    print("[PASS] test_cloud_and_neighbour_ready_time_rule_is_identical")


# ---------------------------------------------------------------------------
# 3. Lifecycle decomposition (no double counting)
# ---------------------------------------------------------------------------
def test_latency_decomposition_hand_calculated():
    """Cloud task, demand 60, capacity 50, delay 2, arrival 1:
    ready 3, start 3, 50 done at t=3, 10 at t=4 -> completion 4.
    transmission 2 + wait 0 + span 1 = latency 3 = 4 - 1."""
    sim = HospitalSimulator(CONFIG_PATH)
    sim.step()                                       # t=0 (empty)
    t = sim.route_task("ICU", compute_demand=60.0)   # arrival 1
    sim.run(5)
    assert (t.arrival_time, t.ready_time, t.start_time, t.completion_time) == (1, 3, 3, 4)
    assert (t.transmission_delay, t.queue_wait, t.execution_span, t.latency) == (2, 0, 1, 3)
    assert t.transmission_delay + t.queue_wait + t.execution_span == t.latency
    print("[PASS] test_latency_decomposition_hand_calculated")


def test_queue_wait_excludes_transit_time():
    """Two 50-demand Cloud tasks at t=0: both ready at 2; the second waits 1
    step AFTER becoming ready -> queue_wait 1 (not 3)."""
    sim = HospitalSimulator(CONFIG_PATH)
    a = sim.route_task("ICU", compute_demand=50.0)
    b = sim.route_task("ICU", compute_demand=50.0)
    sim.run(4)
    assert (a.completion_time, b.start_time, b.completion_time) == (2, 3, 3)
    assert b.transmission_delay == 2 and b.queue_wait == 1 and b.latency == 3
    print("[PASS] test_queue_wait_excludes_transit_time")


# ---------------------------------------------------------------------------
# 4. Utilization (audit regression)
# ---------------------------------------------------------------------------
def test_cloud_utilization_ignores_transit_regression():
    """Audit case: Cloud capacity 50, delay 2, arrival 0, demand 20.
    OLD: utilization at time 0 = 0.4 although nothing could be processed."""
    sim = HospitalSimulator(CONFIG_PATH)
    sim.route_task("ICU", compute_demand=20.0)
    cloud = sim.cloud_node
    assert cloud.utilization == 0.0 and cloud.projected_utilization(0) == 0.0
    assert cloud.transit_demand(0) == 20.0 and cloud.in_transit_count(0) == 1
    sim.step(); sim.step()                           # t=0, t=1: still transit
    assert cloud.last_step_utilization == 0.0
    assert cloud.projected_utilization() == 0.4      # clock 2: ready -> 20/50
    sim.step()                                       # t=2: processed
    assert cloud.last_step_utilization == 0.4 and cloud.utilization == 0.0
    print("[PASS] test_cloud_utilization_ignores_transit_regression")


def test_capacity_never_exceeded_and_tasks_conserved_random_run():
    for seed in range(10):
        sim = HospitalSimulator(CONFIG_PATH)
        gen = MedicalTaskGenerator(CONFIG, seed=seed)
        for _ in range(60):
            sim.route_generated_arrivals(gen.generate_for_step(sim.current_time))
            sim.step()
            for node in sim.all_nodes:
                assert node.last_step_work_done <= node.capacity + 1e-9
        placed = [t for n in sim.all_nodes for t in n.queue + n.completed_tasks]
        assert sorted(t.task_id for t in placed) == list(range(len(sim.tasks)))
        for t in sim.completed_tasks:
            assert t.ready_time <= t.start_time <= t.completion_time
            assert t.transmission_delay + t.queue_wait + t.execution_span == t.latency
    print("[PASS] test_capacity_never_exceeded_and_tasks_conserved_random_run")


# ---------------------------------------------------------------------------
# 5. Same-step arrival order: seeded shuffle, no systematic advantage
# ---------------------------------------------------------------------------
def _first_position_stats(shuffle: bool, n_seeds: int = 300, n_steps: int = 40):
    """Over many seeds, for every step with >= 2 arrivals: how often is the
    FIRST arrival Critical / from Emergency, versus the share expected if the
    order were random (sum of per-step fractions)."""
    obs_crit = exp_crit = var_crit = 0.0
    obs_em = exp_em = var_em = 0.0
    for seed in range(n_seeds):
        gen = MedicalTaskGenerator(CONFIG, seed=seed, shuffle_arrivals=shuffle)
        for t in range(n_steps):
            arr = gen.generate_for_step(t)
            if len(arr) < 2:
                continue
            p_c = sum(a.priority == "Critical" for a in arr) / len(arr)
            p_e = sum(a.department == "Emergency" for a in arr) / len(arr)
            obs_crit += arr[0].priority == "Critical"; exp_crit += p_c; var_crit += p_c * (1 - p_c)
            obs_em += arr[0].department == "Emergency"; exp_em += p_e; var_em += p_e * (1 - p_e)
    return (obs_crit - exp_crit) / math.sqrt(var_crit), (obs_em - exp_em) / math.sqrt(var_em)


def test_same_step_order_gives_no_systematic_priority_or_department_advantage():
    z_crit, z_em = _first_position_stats(shuffle=True)
    assert abs(z_crit) < 4 and abs(z_em) < 4, f"shuffled order looks biased: z={z_crit:.2f}, {z_em:.2f}"
    # Control: the OLD loop order is detectably biased -> the test has power.
    z_crit_old, z_em_old = _first_position_stats(shuffle=False)
    assert z_crit_old > 10 and z_em_old > 10, (z_crit_old, z_em_old)
    print(f"[PASS] test_same_step_order... (z shuffled={z_crit:.2f}/{z_em:.2f}, old={z_crit_old:.1f}/{z_em_old:.1f})")


def test_shuffle_is_reproducible_and_keeps_workload():
    a = MedicalTaskGenerator(CONFIG, seed=5)
    b = MedicalTaskGenerator(CONFIG, seed=5)
    raw = MedicalTaskGenerator(CONFIG, seed=5, shuffle_arrivals=False)
    key = lambda x: (x.department, x.priority, x.compute_demand)
    for t in range(200):
        sa, sb, sr = a.generate_for_step(t), b.generate_for_step(t), raw.generate_for_step(t)
        assert [key(x) for x in sa] == [key(x) for x in sb]          # same seed -> same order
        assert sorted(map(key, sa)) == sorted(map(key, sr))          # same workload as legacy draws
    print("[PASS] test_shuffle_is_reproducible_and_keeps_workload")


def test_department_list_order_gives_no_advantage():
    """Reversing the 'departments' list (node iteration order) must not change
    any task's outcome -- nodes share no capacity."""
    def outcome(cfg):
        sim = HospitalSimulator(cfg)
        script = [("ICU", 8.0), ("Emergency", 2.0), ("ICU", 2.0), ("General_Ward", 7.0), ("General_Ward", 4.0)]
        for dept, d in script:
            sim.route_task(dept, d)
        sim.run(5)
        return [(t.task_id, t.destination, t.completion_time) for t in sim.tasks]
    reversed_cfg = copy.deepcopy(CONFIG)
    reversed_cfg["departments"] = list(reversed(reversed_cfg["departments"]))
    assert outcome(CONFIG) == outcome(reversed_cfg)
    print("[PASS] test_department_list_order_gives_no_advantage")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL V2 ROUTING/TIMING TESTS PASSED.")
