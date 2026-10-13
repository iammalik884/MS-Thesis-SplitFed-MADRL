"""
Step 7 (proposed baseline) - tests for the ECT heuristic (agents/heuristics.py).
================================================================================
ROMAN URDU: Haath se calculate kiye hue andaze, khaali nodes par andaza =
asal latency, tie ka order, aur yeh ke heuristic aage aane wale tasks
(future information) ko nahi dekhta.

Run with:  py -m pytest tests/test_ect_heuristic.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.heuristics import ect_action, estimated_latencies
from env.gym_env import HospitalSchedulingEnv
from env.task_generator import GeneratedArrival

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "environment.yaml")
LOCAL, EDGE, CLOUD = 0, 1, 2


def test_estimates_hand_calculated():
    """Seed 42, first task: ICU, demand 6.7164, all nodes idle.
    Local: 0 + ceil(6.7164/4) - 1 = 1; Edge (Emergency, cap 5): 1 + ceil(6.7164/5) - 1 = 2;
    Cloud: 2 + ceil(6.7164/50) - 1 = 2  ->  ECT picks Local."""
    env = HospitalSchedulingEnv(CONFIG_PATH)
    env.reset(seed=42)
    assert estimated_latencies(env) == {LOCAL: 1, EDGE: 2, CLOUD: 2}
    assert ect_action(env) == LOCAL
    print("[PASS] test_estimates_hand_calculated")


def test_estimate_equals_real_latency_on_idle_nodes():
    """With idle nodes and no later interference, each estimate is exact."""
    for action in (LOCAL, EDGE, CLOUD):
        env = HospitalSchedulingEnv(CONFIG_PATH)
        env.reset(seed=42)
        estimate = estimated_latencies(env)[action]
        env.step(action)
        task = env.sim.tasks[0]
        while task.completion_time is None:
            env.sim.step()
        assert task.latency == estimate, (action, task.latency, estimate)
    print("[PASS] test_estimate_equals_real_latency_on_idle_nodes")


def test_ties_prefer_local_then_edge_then_cloud():
    """Emergency task, demand 5.0. Edge = ICU (both neighbours idle, tie order), cap 4:
    idle:            Local 0 + ceil(5/5) - 1 = 0, Edge 1 + ceil(5/4) - 1 = 2, Cloud 2 + ceil(5/50) - 1 = 2
    Emergency 5.0:   Local ceil(10/5) - 1 = 1, Edge 2, Cloud 2
    Emergency 10.0:  Local ceil(15/5) - 1 = 2 = Edge = Cloud -> tie -> Local."""
    env = HospitalSchedulingEnv(CONFIG_PATH)
    env.reset(seed=42)  # nothing placed yet: all nodes idle
    env._pending = [GeneratedArrival(department="Emergency", priority="High", compute_demand=5.0)]
    assert estimated_latencies(env) == {LOCAL: 0, EDGE: 2, CLOUD: 2}
    env.sim.add_task("Emergency", 5.0)
    assert estimated_latencies(env) == {LOCAL: 1, EDGE: 2, CLOUD: 2}
    env.sim.add_task("Emergency", 5.0)
    assert estimated_latencies(env) == {LOCAL: 2, EDGE: 2, CLOUD: 2}
    assert ect_action(env) == LOCAL, "full tie -> Local"
    print("[PASS] test_ties_prefer_local_then_edge_then_cloud")


def test_uses_no_future_information():
    """Adding further (not yet shown) arrivals behind the pending task must not
    change the estimates: only the current task and node backlogs are used."""
    env = HospitalSchedulingEnv(CONFIG_PATH)
    env.reset(seed=42)
    before = estimated_latencies(env)
    env._pending.append(GeneratedArrival(department="ICU", priority="Critical", compute_demand=2.0))
    assert estimated_latencies(env) == before
    print("[PASS] test_uses_no_future_information")


def test_full_episode_runs_and_is_deterministic():
    reports = []
    for _ in range(2):
        env = HospitalSchedulingEnv(os.path.join(ROOT, "configs", "scenarios", "overload.yaml"))
        env.reset(seed=3)
        done = False
        while not done:
            _, _, done, _, info = env.step(ect_action(env))
        reports.append(info["report"])
    assert reports[0] == reports[1]
    print("[PASS] test_full_episode_runs_and_is_deterministic")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL ECT HEURISTIC TESTS PASSED.")
