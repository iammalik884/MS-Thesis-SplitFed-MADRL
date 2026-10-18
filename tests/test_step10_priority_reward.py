"""
Step 10 - Tests for the priority-aware reward (rl_environment.priority_weights).
================================================================================
ROMAN URDU: Weighted reward ka hisaab haath se (adhoore tasks samet), config
ki ghalat values par saaf error, weights se simulation/report nahi badalti
(sirf reward badalta hai), multi-agent env mein har agent ko us ke tasks ka
weighted reward, aur evaluate() ka per-priority latency sahi.

Run with:  py -m pytest tests/test_step10_priority_reward.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agents.policy_eval import baseline_policies, evaluate
from env.gym_env import HospitalSchedulingEnv
from env.hospital_env import load_config
from env.multi_agent_env import MultiAgentHospitalEnv
from train_mappo import violation_by_priority

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "environment.yaml")
OVERLOAD = os.path.join(ROOT, "configs", "scenarios", "overload.yaml")
OVERDUE = os.path.join(ROOT, "configs", "scenarios", "overdue_critical.yaml")
W = {"Critical": 5.0, "High": 2.0, "Routine": 1.0}


def _weighted(path, weights=W):
    config = load_config(path)
    config["rl_environment"]["priority_weights"] = dict(weights)
    return config


def _play(env, seed, actions_rng):
    env.reset(seed=seed)
    actions, rewards, done, info = [], [], False, {}
    while not done:
        a = int(actions_rng.integers(0, 3))
        _, r, done, _, info = env.step(a)
        actions.append(a)
        rewards.append(r)
    return actions, rewards, info


def test_default_weights_are_one_so_steps_6_to_9_are_unchanged():
    env = HospitalSchedulingEnv(CONFIG_PATH)
    assert env.priority_weights == {"Critical": 1.0, "High": 1.0, "Routine": 1.0}
    assert "priority_weights" not in load_config(CONFIG_PATH)["rl_environment"]
    print("[PASS] test_default_weights_are_one_so_steps_6_to_9_are_unchanged")


def test_weighted_return_by_hand_including_unfinished_tasks():
    """return = -sum_tasks w_p * (latency + penalty * missed); unfinished tasks
    use cutoff - arrival + 1 (overdue_critical has drain 0, so they occur)."""
    unfinished = 0
    for path in (OVERLOAD, OVERDUE):
        env = HospitalSchedulingEnv(_weighted(path))
        for seed in range(5):
            _, rewards, info = _play(env, seed, np.random.default_rng(seed))
            cutoff = info["report"]["protocol"]["final_cutoff"]
            expected = 0.0
            for t in env.sim.tasks:
                if t.completion_time is not None:
                    latency, missed = t.latency, t.completion_time > t.deadline
                else:
                    latency, missed = cutoff - t.arrival_time + 1, t.deadline <= cutoff
                    unfinished += 1
                expected -= W[t.priority] * (latency + env.deadline_miss_penalty * missed)
            assert abs(sum(rewards) - expected) < 1e-9 and abs(info["episode_return"] - expected) < 1e-9
    assert unfinished > 0
    print(f"[PASS] test_weighted_return_by_hand_including_unfinished_tasks ({unfinished} unfinished tasks)")


def test_weights_change_only_the_reward_not_the_simulation():
    """Same seed + same actions with and without weights: identical protocol
    report and task records; only the rewards differ."""
    plain, weighted = HospitalSchedulingEnv(OVERLOAD), HospitalSchedulingEnv(_weighted(OVERLOAD))
    for seed in range(3):
        a1, r1, i1 = _play(plain, seed, np.random.default_rng(seed))
        a2, r2, i2 = _play(weighted, seed, np.random.default_rng(seed))
        assert a1 == a2 and i1["report"] == i2["report"]
        assert sum(r1) != sum(r2), "weights 5/2/1 must change the return in overload"
    print("[PASS] test_weights_change_only_the_reward_not_the_simulation")


def test_multi_agent_rewards_are_weighted_per_source_task():
    env = MultiAgentHospitalEnv(_weighted(OVERDUE))
    rng = np.random.default_rng(0)
    for seed in range(3):
        env.reset(seed=seed)
        while not env.done:
            _, _, info = env.step(int(rng.integers(0, 3)))     # step() checks sum(per agent) == team reward
        cutoff = info["report"]["protocol"]["final_cutoff"]
        expected = {a: 0.0 for a in env.agents}
        for t in env.inner.sim.tasks:
            if t.completion_time is not None:
                latency, missed = t.latency, t.completion_time > t.deadline
            else:
                latency, missed = cutoff - t.arrival_time + 1, t.deadline <= cutoff
            expected[t.source_department] -= W[t.priority] * (latency + env.inner.deadline_miss_penalty * missed)
        assert all(abs(info["episode_returns"][a] - expected[a]) < 1e-9 for a in env.agents)
    print("[PASS] test_multi_agent_rewards_are_weighted_per_source_task")


def test_invalid_priority_weights_raise_meaningful_errors():
    bad_cases = [{"Critical": 5.0, "High": 2.0},                                 # missing class
                 {"Critical": 5.0, "High": 2.0, "Routine": 1.0, "Urgent": 9.0},   # unknown class
                 {"Critical": 0.0, "High": 2.0, "Routine": 1.0},                  # not > 0
                 {"Critical": -1.0, "High": 2.0, "Routine": 1.0},
                 {"Critical": float("nan"), "High": 2.0, "Routine": 1.0},
                 {"Critical": True, "High": 2.0, "Routine": 1.0}]
    for weights in bad_cases:
        try:
            HospitalSchedulingEnv(_weighted(CONFIG_PATH, weights))
            raise AssertionError(f"Expected ValueError for {weights}")
        except ValueError as e:
            assert "priority_weights" in str(e), str(e)
    print("[PASS] test_invalid_priority_weights_raise_meaningful_errors")


def test_latency_and_violation_by_priority_match_the_tasks_and_reports():
    seeds = range(4)
    result = evaluate(OVERLOAD, baseline_policies()["FCFS"], seeds, keep_reports=True)
    env = HospitalSchedulingEnv(OVERLOAD)
    sums, counts = {}, {}
    for seed in seeds:
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _, _ = env.step(env.fcfs_action())
        for t in env.sim.tasks:
            if t.completion_time is not None:
                sums[t.priority] = sums.get(t.priority, 0) + t.latency
                counts[t.priority] = counts.get(t.priority, 0) + 1
    for p, row in result["latency_by_priority"].items():
        assert row["completed"] == counts[p] and abs(row["avg_latency"] - sums[p] / counts[p]) < 1e-12
    by_p = violation_by_priority(result)
    critical = [r["report"]["final"]["deadlines"]["per_priority"]["Critical"] for r in result["per_seed"]]
    assert abs(by_p["Critical"]["violation_rate"]
               - sum(c["completed_late"] + c["overdue_unfinished"] for c in critical) / sum(c["decided"] for c in critical)) < 1e-12
    print("[PASS] test_latency_and_violation_by_priority_match_the_tasks_and_reports")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL STEP 10 TESTS PASSED.")
