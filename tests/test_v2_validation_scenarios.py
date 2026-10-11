"""
V2 audit fixes (part c) - input validation (no partial mutation) and the
synthetic verification scenarios in configs/scenarios/.
================================================================================
ROMAN URDU: (1) Galat input par saaf ValueError aaye -- KeyError na leak ho,
aur simulator ki state (queues, task IDs, task list) bilkul na badle.
(2) Har scenario reproducible ho aur wahi dikhaye jis ke liye bana hai.
(3) Roadmap Step 3 ke checkpoints chhoti haath-calculated examples se.

ENGLISH: (1) bad input -> clear ValueError, zero state mutation; (2) each
scenario is reproducible and shows what it was built for; (3) the roadmap's
Step 3 checkpoints as small hand-calculated examples.
NOTE: scenario results are synthetic verification only, not clinical validation.

Run with:  py -m pytest tests/test_v2_validation_scenarios.py
"""

import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.evaluation import run_fcfs_evaluation
from env.hospital_env import HospitalSimulator, load_config
from env.task_generator import GeneratedArrival, MedicalTaskGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "environment.yaml")
SCENARIOS = os.path.join(ROOT, "configs", "scenarios")
CONFIG = load_config(CONFIG_PATH)


def scenario(name):
    return load_config(os.path.join(SCENARIOS, f"{name}.yaml"))


def _state(sim):
    return (sim._next_task_id, len(sim.tasks), [n.queue_length for n in sim.all_nodes])


def expect_error(fn, *fragments):
    try:
        fn()
    except ValueError as e:
        for fragment in fragments:
            assert fragment in str(e), f"{fragment!r} not in {e}"
        return
    raise AssertionError(f"expected ValueError mentioning {fragments}")


# ---------------------------------------------------------------------------
# 1. Runtime input validation, no partial mutation
# ---------------------------------------------------------------------------
def test_invalid_priority_is_valueerror_not_keyerror_and_mutates_nothing():
    sim = HospitalSimulator(CONFIG_PATH)
    sim.add_task("ICU", compute_demand=1.0)
    before = _state(sim)
    for call in (lambda: sim.add_task("ICU", 1.0, priority="critical"),
                 lambda: sim.route_task("ICU", 1.0, priority="CRITICAL")):
        expect_error(call, "priority")
    assert _state(sim) == before
    print("[PASS] test_invalid_priority_is_valueerror_not_keyerror_and_mutates_nothing")


def test_nan_inf_bool_demand_rejected_without_mutation():
    sim = HospitalSimulator(CONFIG_PATH)
    before = _state(sim)
    for bad in (float("nan"), float("inf"), -float("inf"), 0.0, -1.0, True, "3"):
        expect_error(lambda: sim.add_task("ICU", bad), "compute_demand")
        expect_error(lambda: sim.route_task("ICU", bad), "compute_demand")
    assert _state(sim) == before
    print("[PASS] test_nan_inf_bool_demand_rejected_without_mutation")


def test_batch_with_one_bad_arrival_places_nothing():
    sim = HospitalSimulator(CONFIG_PATH)
    batch = [GeneratedArrival("ICU", "High", 2.0), GeneratedArrival("ICU", "High", float("nan"))]
    before = _state(sim)
    expect_error(lambda: sim.route_generated_arrivals(batch), "compute_demand")
    expect_error(lambda: sim.add_generated_arrivals([GeneratedArrival("Pharmacy", "High", 1.0)]), "Unknown department")
    assert _state(sim) == before
    expect_error(lambda: sim.run(2.5), "n_steps")
    print("[PASS] test_batch_with_one_bad_arrival_places_nothing")


# ---------------------------------------------------------------------------
# 2. Config validation (each case built from the valid default)
# ---------------------------------------------------------------------------
def _mutated(mutate):
    cfg = copy.deepcopy(CONFIG)
    mutate(cfg)
    return cfg


CONFIG_CASES = [
    (lambda c: c["departments"][0].update(capacity=float("nan")), "departments[0].capacity"),
    (lambda c: c["departments"][0].update(capacity=float("inf")), "departments[0].capacity"),
    (lambda c: c["departments"][1].update(capacity=-4.0), "departments[1].capacity"),
    (lambda c: c["departments"][1].update(name="Emergency"), "unique"),
    (lambda c: c["departments"][1].update(name="Cloud"), "unique"),
    (lambda c: c["simulation"].update(n_steps=20.5), "simulation.n_steps"),
    (lambda c: c["simulation"].update(n_steps=True), "simulation.n_steps"),
    (lambda c: c["simulation"].update(n_steps=0), "simulation.n_steps"),
    (lambda c: c.update(seed=-1), "seed"),
    (lambda c: c.update(deadline={"Critical": 2}), "unknown top-level"),
    (lambda c: c["task_generation"]["departments"].update(Pharmacy=c["task_generation"]["departments"]["ICU"]), "must match"),
    (lambda c: c["task_generation"]["departments"].pop("ICU"), "must match"),
    (lambda c: c["task_generation"]["departments"]["ICU"].update(critical={"arrival_rate": 0.1, "compute_demand_range": [1, 2]}), "unknown key"),
    (lambda c: c["task_generation"]["departments"]["ICU"]["High"].update(arrival_rate=float("nan")), "ICU.High.arrival_rate"),
    (lambda c: c["task_generation"]["departments"]["ICU"]["High"].update(arrival_rate=-0.1), "ICU.High.arrival_rate"),
    (lambda c: c["task_generation"]["departments"]["ICU"]["High"].update(compute_demand_range=[1.0, float("inf")]), "compute_demand_range"),
    (lambda c: c["task_generation"]["departments"]["ICU"]["High"].update(compute_demand_range=[0.0, 2.0]), "compute_demand_range"),
    (lambda c: c["offloading"].update(neighbour_network_delay=1.5), "neighbour_network_delay"),
    (lambda c: c["offloading"].update(cloud_network_delay=-1), "cloud_network_delay"),
    (lambda c: c["offloading"].update(cloud_threshold=float("nan")), "cloud_threshold"),
    (lambda c: c["offloading"].update(offload_threshold=-1.0), "offload_threshold"),
    (lambda c: c["offloading"].update(cloud_capacity=0), "cloud_capacity"),
    (lambda c: c["offloading"].update(neighbour_tie_break_order=["ICU", "Emergency"]), "neighbour_tie_break_order"),
    (lambda c: c["offloading"].update(edge_threshold=6.0), "removed pre-v2"),
    (lambda c: c["offloading"].update(neighbor_delay=1), "unknown key"),
    (lambda c: c["offloading"].pop("cloud_threshold"), "missing"),
    (lambda c: c["deadlines"].update(Critical=2.5), "deadlines.Critical"),
    (lambda c: c["deadlines"].update(High=float("inf")), "deadlines.High"),
    (lambda c: c["evaluation"].update(drain_limit=-5), "evaluation.drain_limit"),
    (lambda c: c["evaluation"].update(synthetic_payload_units_per_compute_unit=float("nan")), "synthetic_payload"),
]


def test_invalid_configs_rejected_with_clear_messages():
    for mutate, fragment in CONFIG_CASES:
        expect_error(lambda: HospitalSimulator(_mutated(mutate)), fragment)
    single = _mutated(lambda c: c.update(departments=[c["departments"][0]]))
    single["task_generation"]["departments"] = {"Emergency": CONFIG["task_generation"]["departments"]["Emergency"]}
    expect_error(lambda: HospitalSimulator(single), ">= 2 departments")
    expect_error(lambda: MedicalTaskGenerator(CONFIG, seed=-3), "seed")
    expect_error(lambda: MedicalTaskGenerator(CONFIG, seed=1.5), "seed")
    assert HospitalSimulator(copy.deepcopy(CONFIG)) is not None, "the default config itself stays valid"
    print(f"[PASS] test_invalid_configs_rejected_with_clear_messages ({len(CONFIG_CASES) + 3} cases)")


# ---------------------------------------------------------------------------
# 3. Roadmap Step 3 checkpoints, hand-calculated
# ---------------------------------------------------------------------------
def test_checkpoint_idle_neighbour_makes_offloading_faster():
    """ICU backlog 12 (3 steps of ICU work). A 2.0 task kept Local waits until
    t=3 -> latency 3. Offloaded to idle Emergency: ready t=1, done t=1 -> 1."""
    local = HospitalSimulator(CONFIG_PATH)
    local.add_task("ICU", 12.0)
    kept = local.add_task("ICU", 2.0)
    routed = HospitalSimulator(CONFIG_PATH)
    routed.add_task("ICU", 12.0)
    off = routed.route_task("ICU", 2.0)
    local.run(5); routed.run(5)
    assert (kept.queue_wait, kept.latency) == (3, 3)
    assert (off.destination, off.queue_wait, off.latency) == ("Emergency", 0, 1)
    print("[PASS] test_checkpoint_idle_neighbour_makes_offloading_faster")


def test_checkpoint_busy_local_cpu_increases_waiting():
    waits = []
    for backlog in (0.0, 4.0, 8.0):
        sim = HospitalSimulator(CONFIG_PATH)
        if backlog:
            sim.add_task("ICU", backlog)
        t = sim.add_task("ICU", 1.0)
        sim.run(4)
        waits.append(t.queue_wait)
    assert waits == [0, 1, 2], waits      # ICU capacity 4: each 4 units ahead = 1 step
    print("[PASS] test_checkpoint_busy_local_cpu_increases_waiting")


def test_checkpoint_slow_cloud_link_increases_cloud_latency():
    fast = run_fcfs_evaluation(CONFIG, seed=11).report["final"]["by_destination"]["Cloud"]
    slow = run_fcfs_evaluation(scenario("slow_network"), seed=11).report["final"]["by_destination"]["Cloud"]
    assert fast["avg_transmission_delay"] == 2 and slow["avg_transmission_delay"] == 6
    assert slow["avg_latency"] >= 6 > fast["avg_latency"]
    print("[PASS] test_checkpoint_slow_cloud_link_increases_cloud_latency")


# ---------------------------------------------------------------------------
# 4. Scenario configs
# ---------------------------------------------------------------------------
SCENARIO_NAMES = ("overload", "department_imbalance", "slow_network", "overdue_critical", "empty_workload")


def test_every_scenario_is_valid_and_reproducible():
    for name in SCENARIO_NAMES:
        cfg = scenario(name)
        assert cfg["scenario"]["name"] == name and "extends" not in cfg
        assert run_fcfs_evaluation(cfg, seed=4).report == run_fcfs_evaluation(cfg, seed=4).report, name
    print("[PASS] test_every_scenario_is_valid_and_reproducible")


def test_default_workload_unchanged_by_scenarios():
    """Scenarios inherit from the default file; they must not alter it."""
    assert CONFIG["task_generation"]["departments"]["ICU"]["Critical"]["arrival_rate"] == 0.15
    assert CONFIG["evaluation"]["drain_limit"] == 50 and "scenario" not in CONFIG
    print("[PASS] test_default_workload_unchanged_by_scenarios")


def test_empty_workload_reports_none_not_zero():
    rep = run_fcfs_evaluation(scenario("empty_workload"), seed=0).report
    h = rep["headline"]
    assert h["tasks"] == 0 and h["throughput_in_window"] == 0.0 and h["run_complete"]
    assert h["avg_latency"] is None and h["critical_deadline_violation_rate"] is None
    assert h["deadline_violation_rate"] is None and rep["protocol"]["drain_steps_used"] == 0
    print("[PASS] test_empty_workload_reports_none_not_zero")


def test_overload_builds_backlog_and_offloads():
    rep = run_fcfs_evaluation(scenario("overload"), seed=1).report
    assert rep["drain"]["backlog_at_window_end"]["tasks"] > 0
    assert rep["drain"]["completions_during_drain"] > 0
    assert rep["final"]["offloading_communication"]["neighbour_transfers"] > 0
    print("[PASS] test_overload_builds_backlog_and_offloads")


def test_department_imbalance_moves_icu_work_to_neighbours():
    rep = run_fcfs_evaluation(scenario("department_imbalance"), seed=2).report
    routes = rep["final"]["by_route"]
    icu_out = sum(v["tasks"] for k, v in routes.items() if k in ("ICU->Emergency", "ICU->General_Ward"))
    assert icu_out > 0, routes
    print(f"[PASS] test_department_imbalance_moves_icu_work_to_neighbours ({icu_out} ICU tasks offloaded)")


def test_overdue_critical_scenario_exposes_hidden_violation():
    hits = 0
    for seed in range(10):
        crit = run_fcfs_evaluation(scenario("overdue_critical"), seed=seed).report["final"]["deadlines"]["per_priority"]["Critical"]
        if crit["overdue_unfinished"]:
            hits += 1
            # Headline counts MORE violating tasks than the completed-only view,
            # and its rate is never lower (both can be 1.0 if every completion was late).
            assert crit["completed_late"] + crit["overdue_unfinished"] > crit["completed_late"]
            assert crit["deadline_violation_rate"] >= (crit["completed_only_late_rate"] or 0.0)
    assert hits >= 5, hits
    print(f"[PASS] test_overdue_critical_scenario_exposes_hidden_violation ({hits}/10 seeds)")


def test_multiseed_invariants_default_and_overload():
    """Seeds 0-29: conservation, window <= final completions, valid rates."""
    for cfg in (CONFIG, scenario("overload")):
        for seed in range(30):
            rep = run_fcfs_evaluation(cfg, seed=seed).report
            w, f = rep["window"], rep["final"]
            assert w["completed"] <= f["completed"] <= f["tasks_arrived"]
            assert f["completed"] + rep["drain"]["backlog_after_drain"]["tasks"] == f["tasks_arrived"]
            for snap in (w, f):
                rate = snap["deadlines"]["overall"]["deadline_violation_rate"]
                assert rate is None or 0.0 <= rate <= 1.0
                assert snap["avg_latency"] is None or not math.isnan(snap["avg_latency"])
    print("[PASS] test_multiseed_invariants_default_and_overload")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL V2 VALIDATION/SCENARIO TESTS PASSED.")
