"""
Step 6 - Verification tests for the Gymnasium environment (env/gym_env.py).
================================================================================
ROMAN URDU: Roadmap ke Step 6 checkpoints: (1) reset() valid observation
deta hai, (2) step(action) system ko badalta hai aur (obs, reward,
terminated, truncated, info) deta hai. Saath mein sab se zaroori check:
agar agent har task par wahi action chune jo FCFS router chunta, to har
config aur seeds 0-29 par report BILKUL Step 5 ke FCFS jaisi aati hai --
yani wrapper ne simulator ka koi rule nahi badla.

ENGLISH: roadmap Step 6 checkpoints, Gymnasium's own env checker, exact
FCFS equivalence on every config x seeds 0-29, reward bookkeeping, action
meanings, determinism, the empty-episode edge case and invalid usage.

Run with:  py -m pytest tests/test_step6_gym_env.py
"""

import glob
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.evaluation import run_fcfs_evaluation
from env.gym_env import ACTIONS, HospitalSchedulingEnv
from env.hospital_env import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "environment.yaml")
ALL_CONFIGS = [CONFIG_PATH] + sorted(glob.glob(os.path.join(ROOT, "configs", "scenarios", "*.yaml")))
LOCAL, EDGE, CLOUD = 0, 1, 2


def _run_episode(env, seed, policy):
    """Play one episode; returns (actions, observations, rewards, last info)."""
    obs, info = env.reset(seed=seed)
    actions, observations, rewards = [], [obs], []
    while True:
        action = policy(env)
        obs, reward, terminated, truncated, info = env.step(action)
        assert truncated is False
        actions.append(action)
        observations.append(obs)
        rewards.append(reward)
        if terminated:
            return actions, observations, rewards, info


def _random_policy(rng):
    return lambda env: int(rng.integers(0, 3))


def test_gymnasium_env_checker_passes():
    """Gymnasium's own API checker. The only accepted warning is the infinite
    upper bound of the backlog/demand features (they have no true maximum)."""
    env = HospitalSchedulingEnv(CONFIG_PATH)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, skip_render_check=True)
    other = [str(w.message) for w in caught if "maximum value is infinity" not in str(w.message)]
    assert not other, f"unexpected env_checker warnings: {other}"
    print("[PASS] test_gymnasium_env_checker_passes")


def test_reset_returns_valid_observation():
    """Roadmap checkpoint 1. Seed 42's first task (from the Step 2 example) is
    ICU / Routine / demand 6.7164 at t=0; ICU capacity 4 -> 1.679 steps."""
    env = HospitalSchedulingEnv(CONFIG_PATH)
    obs, info = env.reset(seed=42)
    assert obs.dtype == np.float32 and obs.shape == (13,) and env.observation_space.contains(obs)
    named = dict(zip(env.observation_names, obs.tolist()))
    assert all(named[f"backlog_steps[{d}]"] == 0.0 for d in ("Emergency", "ICU", "General_Ward", "Cloud"))
    assert abs(named["task_demand_steps_on_own_node"] - 6.716385831661723 / 4.0) < 1e-6
    assert named["source[ICU]"] == 1.0 and named["source[Emergency]"] == 0.0
    assert named["priority[Routine]"] == 1.0 and named["priority[Critical]"] == 0.0
    assert named["deadline_offset_fraction"] == 1.0, "Routine offset 10 / max offset 10"
    assert named["time_fraction"] == 0.0
    assert info == {"workload_seed": 42, "time": 0, "no_tasks_in_episode": False}
    obs2, _ = HospitalSchedulingEnv(CONFIG_PATH).reset(seed=42)
    assert np.array_equal(obs, obs2), "same seed -> same first observation"
    print("[PASS] test_reset_returns_valid_observation")


def test_step_changes_system_and_returns_five_values():
    """Roadmap checkpoint 2: step() changes the system and returns
    (observation, reward, terminated, truncated, info)."""
    env = HospitalSchedulingEnv(CONFIG_PATH)
    env.reset(seed=42)
    result = env.step(LOCAL)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs) and isinstance(reward, float)
    assert terminated is False and truncated is False
    assert info["placed_task_id"] == 0 and info["tier"] == "Local" and info["destination"] == "ICU"
    assert len(env.sim.tasks) == 1, "the decided task now exists in the simulator"
    print("[PASS] test_step_changes_system_and_returns_five_values")


def test_each_action_places_the_task_as_documented():
    """Same first task (ICU, demand 6.7164, t=0, all nodes idle):
    Local -> ICU, no delay; Edge -> least-loaded neighbour, tie -> Emergency,
    neighbour delay 1; Cloud -> Cloud, cloud delay 2."""
    expected = {LOCAL: ("ICU", 0), EDGE: ("Emergency", 1), CLOUD: ("Cloud", 2)}
    for action, (destination, ready_time) in expected.items():
        env = HospitalSchedulingEnv(CONFIG_PATH)
        env.reset(seed=42)
        env.step(action)
        task = env.sim.tasks[0]
        assert (task.tier, task.destination, task.ready_time) == (ACTIONS[action], destination, ready_time)
    print("[PASS] test_each_action_places_the_task_as_documented")


def test_fcfs_policy_reproduces_step5_baseline_exactly():
    """If the agent always picks the FCFS router's action, every config x
    seeds 0-29 gives the IDENTICAL protocol report and task records as
    run_fcfs_evaluation() -- the wrapper changes nothing in the simulator."""
    runs = 0
    for path in ALL_CONFIGS:
        env = HospitalSchedulingEnv(path)
        for seed in range(30):
            _, _, _, info = _run_episode(env, seed, lambda e: e.fcfs_action() if e._pending else LOCAL)
            baseline = run_fcfs_evaluation(path, seed)
            assert info["report"] == baseline.report, f"{os.path.basename(path)} seed {seed}: report differs"
            fields = lambda t: (t.task_id, t.source_department, t.tier, t.destination, t.priority, t.compute_demand,
                                t.arrival_time, t.ready_time, t.start_time, t.completion_time, t.deadline)
            assert [fields(t) for t in env.sim.tasks] == [fields(t) for t in baseline.simulator.tasks]
            runs += 1
    print(f"[PASS] test_fcfs_policy_reproduces_step5_baseline_exactly ({runs} runs)")


def test_episode_return_matches_latency_and_deadline_metrics():
    """Each task is rewarded exactly once, so the episode return equals
    -(sum of latencies) - penalty * (late + overdue), with an unfinished task's
    latency counted as cutoff - arrival + 1. overdue_critical has drain 0, so
    unfinished tasks really occur there."""
    rng = np.random.default_rng(0)
    unfinished_seen = 0
    for path in (CONFIG_PATH, os.path.join(ROOT, "configs", "scenarios", "overload.yaml"),
                 os.path.join(ROOT, "configs", "scenarios", "overdue_critical.yaml")):
        env = HospitalSchedulingEnv(path)
        penalty = load_config(path)["rl_environment"]["deadline_miss_penalty"]
        for seed in range(10):
            _, _, rewards, info = _run_episode(env, seed, _random_policy(rng))
            cutoff = info["report"]["protocol"]["final_cutoff"]
            expected, misses = 0.0, 0
            for t in env.sim.tasks:
                if t.completion_time is not None:
                    missed = t.completion_time > t.deadline
                    expected -= t.latency
                else:
                    missed = t.deadline <= cutoff
                    expected -= cutoff - t.arrival_time + 1
                    unfinished_seen += 1
                misses += missed
                expected -= penalty * missed
            overall = info["report"]["final"]["deadlines"]["overall"]
            assert misses == overall["completed_late"] + overall["overdue_unfinished"]
            assert abs(sum(rewards) - expected) < 1e-9 and abs(info["episode_return"] - expected) < 1e-9
    assert unfinished_seen > 0, "the unfinished-task reward path must actually be exercised"
    print(f"[PASS] test_episode_return_matches_latency_and_deadline_metrics ({unfinished_seen} unfinished tasks)")


def test_random_policy_is_deterministic_and_loses_no_task():
    """Same seed + same actions -> identical episode. One decision per task;
    every task ends up on exactly one node or completed."""
    for path in (CONFIG_PATH, os.path.join(ROOT, "configs", "scenarios", "overload.yaml")):
        env = HospitalSchedulingEnv(path)
        for seed in range(5):
            a1, o1, r1, info1 = _run_episode(env, seed, _random_policy(np.random.default_rng(seed)))
            a2, o2, r2, info2 = _run_episode(env, seed, _random_policy(np.random.default_rng(seed)))
            assert a1 == a2 and r1 == r2 and all(np.array_equal(x, y) for x, y in zip(o1, o2))
            assert info1["report"] == info2["report"]
            assert len(a1) == len(env.sim.tasks), "exactly one decision per task"
            on_nodes = sum(n.queue_length + len(n.completed_tasks) for n in env.sim.all_nodes)
            assert on_nodes == len(env.sim.tasks)
            assert all(env.observation_space.contains(o) for o in o1)
    print("[PASS] test_random_policy_is_deterministic_and_loses_no_task")


def test_empty_workload_episode():
    """No task ever arrives: reset says so, the first step ends the episode
    with reward 0 and the usual (empty) report."""
    env = HospitalSchedulingEnv(os.path.join(ROOT, "configs", "scenarios", "empty_workload.yaml"))
    obs, info = env.reset(seed=0)
    assert info["no_tasks_in_episode"] is True and env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(LOCAL)
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["report"]["headline"]["tasks"] == 0 and info["episode_return"] == 0.0
    print("[PASS] test_empty_workload_episode")


def test_invalid_usage_raises_meaningful_errors():
    env = HospitalSchedulingEnv(CONFIG_PATH)
    env.reset(seed=1)
    for bad in (3, -1, True, 1.0, "Local"):
        try:
            env.step(bad)
            raise AssertionError(f"Expected ValueError for action {bad!r}")
        except ValueError as e:
            assert "action" in str(e)
    assert env.sim.tasks == [], "a rejected action must not place a task"
    try:
        env.sim.place_task("ICU", 1.0, "High", "Neighbour")
        raise AssertionError("Expected ValueError for an unknown tier")
    except ValueError as e:
        assert "tier" in str(e)
    assert env.sim.tasks == [] and env.sim._next_task_id == 0, "a rejected placement must change nothing"
    for bad_seed in (-1, 2.5):
        try:
            env.reset(seed=bad_seed)
            raise AssertionError(f"Expected ValueError for seed {bad_seed!r}")
        except ValueError as e:
            assert "seed" in str(e)
    _run_episode(env, 1, lambda e: LOCAL)
    try:
        env.step(LOCAL)
        raise AssertionError("Expected RuntimeError after the episode ended")
    except RuntimeError as e:
        assert "reset" in str(e)
    config = load_config(CONFIG_PATH)
    del config["rl_environment"]
    try:
        HospitalSchedulingEnv(config)
        raise AssertionError("Expected ValueError without an 'rl_environment' section")
    except ValueError as e:
        assert "rl_environment" in str(e)
    config = load_config(CONFIG_PATH)
    config["rl_environment"]["deadline_miss_penalty"] = float("nan")
    try:
        HospitalSchedulingEnv(config)
        raise AssertionError("Expected ValueError for a NaN penalty")
    except ValueError as e:
        assert "deadline_miss_penalty" in str(e)
    print("[PASS] test_invalid_usage_raises_meaningful_errors")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL STEP 6 TESTS PASSED.")
