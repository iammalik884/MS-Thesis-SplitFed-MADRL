"""
Step 8 - Verification tests for the three-agent environment (env/multi_agent_env.py).
================================================================================
ROMAN URDU: Roadmap ke Step 8 checkpoints: (1) kya teeno agents ek hi
simulation step mein action le sakte hain? (2) kya doosre department ko
offload karne se us department ki queue/load sahi badalti hai? Saath mein:
agar har agent FCFS rule chune to har config aur seeds 0-29 par nateeja
BILKUL Step 5 ke FCFS jaisa; har task ka reward us ke source department ke
agent ko, aur sab agents ka jor = Step 6 ka reward.

Run with:  py -m pytest tests/test_step8_multi_agent_env.py
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.evaluation import run_fcfs_evaluation
from env.gym_env import HospitalSchedulingEnv
from env.multi_agent_env import MultiAgentHospitalEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "environment.yaml")
ALL_CONFIGS = [CONFIG_PATH] + sorted(glob.glob(os.path.join(ROOT, "configs", "scenarios", "*.yaml")))
LOCAL, EDGE, CLOUD = 0, 1, 2


def _run_episode(env, seed, policy):
    """policy(env, agent, obs) -> action. Returns [(time, agent, action, rewards)], last info."""
    info = env.reset(seed=seed)
    trace = []
    while not env.done:
        agent = env.agent_selection
        time = env.inner.sim.current_time
        action = policy(env, agent, env.observe(agent))
        rewards, terminated, info = env.step(action)
        trace.append((time, agent, action, rewards))
    return trace, info


def test_reset_spaces_and_observations():
    """Seed 42: the first task is ICU / Routine / demand 6.7164 at t=0, so ICU
    acts first; only ICU's observation carries task features."""
    env = MultiAgentHospitalEnv(CONFIG_PATH)
    info = env.reset(seed=42)
    assert env.agents == ("Emergency", "ICU", "General_Ward")
    assert info["agent_selection"] == env.agent_selection == "ICU" and not env.done
    for a in env.agents:
        obs = env.observe(a)
        assert obs.dtype == np.float32 and obs.shape == (10,) and env.observation_spaces[a].contains(obs)
    icu = dict(zip(env.observation_names["ICU"], env.observe("ICU").tolist()))
    assert abs(icu["task_demand_steps_on_own_node"] - 6.716385831661723 / 4.0) < 1e-6
    assert icu["priority[Routine]"] == 1.0 and icu["deadline_offset_fraction"] == 1.0
    other = dict(zip(env.observation_names["Emergency"], env.observe("Emergency").tolist()))
    assert other["task_demand_steps_on_own_node"] == 0.0 and other["priority[Routine]"] == 0.0
    assert np.array_equal(env.state(), env.inner._observation()) and env.state_space.contains(env.state())
    print("[PASS] test_reset_spaces_and_observations")


def test_checkpoint1_all_three_agents_act_in_the_same_simulation_step():
    """Roadmap checkpoint 1. Every decision is taken in the step its task
    arrives, by the task's own department; some steps have all three agents."""
    env = MultiAgentHospitalEnv(CONFIG_PATH)
    steps_with_all_three = 0
    for seed in range(30):
        trace, _ = _run_episode(env, seed, lambda e, a, o: LOCAL)
        assert len(trace) == len(env.inner.sim.tasks), "one decision per task"
        for (time, agent, _, _), task in zip(trace, env.inner.sim.tasks):
            assert task.arrival_time == time and task.source_department == agent
        agents_by_time = {}
        for time, agent, _, _ in trace:
            agents_by_time.setdefault(time, set()).add(agent)
        steps_with_all_three += sum(len(s) == 3 for s in agents_by_time.values())
    assert steps_with_all_three > 0
    print(f"[PASS] test_checkpoint1_all_three_agents_act_in_the_same_simulation_step "
          f"({steps_with_all_three} steps with all three agents, seeds 0-29)")


def test_checkpoint2_offloading_changes_the_other_departments_load():
    """Roadmap checkpoint 2. ICU's first task (demand 6.7164) offloaded with
    Edge goes to Emergency (tie -> Emergency): Emergency's queue and the load
    every agent observes for Emergency grow by exactly that demand, ICU's own
    load stays 0, and Emergency's node later processes the task."""
    env = MultiAgentHospitalEnv(CONFIG_PATH)
    env.reset(seed=42)
    demand = 6.716385831661723
    _, _, info = env.step(EDGE)
    assert info["agent"] == "ICU" and info["destination"] == "Emergency"
    emergency = env.inner.sim.nodes["Emergency"]
    assert abs(emergency.backlog_demand - demand) < 1e-9 and emergency.queue_length == 1
    assert env.inner.sim.nodes["ICU"].backlog_demand == 0.0
    own = dict(zip(env.observation_names["Emergency"], env.observe("Emergency").tolist()))
    seen_by_icu = dict(zip(env.observation_names["ICU"], env.observe("ICU").tolist()))
    assert abs(own["backlog_steps[own]"] - demand / 5.0) < 1e-6, "Emergency sees its own new load"
    assert abs(seen_by_icu["backlog_steps[Emergency]"] - demand / 5.0) < 1e-6
    assert seen_by_icu["backlog_steps[own]"] == 0.0
    while not env.done:
        env.step(LOCAL)
    task = env.inner.sim.tasks[0]
    assert task in emergency.completed_tasks and task.ready_time == 1, "neighbour delay 1, run by Emergency"
    print("[PASS] test_checkpoint2_offloading_changes_the_other_departments_load")


def test_fcfs_agents_reproduce_step5_baseline_exactly():
    """If every agent picks the FCFS router's action, every config x seeds
    0-29 gives the IDENTICAL report and task records as run_fcfs_evaluation()."""
    runs = 0
    fields = lambda t: (t.task_id, t.source_department, t.tier, t.destination, t.priority, t.compute_demand,
                        t.arrival_time, t.ready_time, t.start_time, t.completion_time, t.deadline)
    for path in ALL_CONFIGS:
        env = MultiAgentHospitalEnv(path)
        for seed in range(30):
            _, info = _run_episode(env, seed, lambda e, a, o: e.inner.fcfs_action())
            baseline = run_fcfs_evaluation(path, seed)
            assert info["report"] == baseline.report, f"{os.path.basename(path)} seed {seed}: report differs"
            assert [fields(t) for t in env.inner.sim.tasks] == [fields(t) for t in baseline.simulator.tasks]
            runs += 1
    print(f"[PASS] test_fcfs_agents_reproduce_step5_baseline_exactly ({runs} runs)")


def test_rewards_go_to_the_source_agent_and_add_up_to_step6():
    """Same actions in the Step 6 env and here: identical team reward per
    decision; each agent's return = reward of its own department's tasks
    (unfinished tasks included: overdue_critical has drain 0)."""
    unfinished = 0
    for path in (CONFIG_PATH, os.path.join(ROOT, "configs", "scenarios", "overload.yaml"),
                 os.path.join(ROOT, "configs", "scenarios", "overdue_critical.yaml")):
        env, single = MultiAgentHospitalEnv(path), HospitalSchedulingEnv(path)
        for seed in range(10):
            rng = np.random.default_rng(seed)
            trace, info = _run_episode(env, seed, lambda e, a, o: int(rng.integers(0, 3)))
            single.reset(seed=seed)
            for _, _, action, rewards in trace:
                _, r, _, _, single_info = single.step(action)
                assert sum(rewards.values()) == r
            assert single_info["episode_return"] == sum(info["episode_returns"].values()) == \
                sum(info["episode_returns"][a] for a in env.agents)
            cutoff = info["report"]["protocol"]["final_cutoff"]
            expected = {a: 0.0 for a in env.agents}
            for t in env.inner.sim.tasks:
                if t.completion_time is not None:
                    latency, missed = t.latency, t.completion_time > t.deadline
                else:
                    latency, missed = cutoff - t.arrival_time + 1, t.deadline <= cutoff
                    unfinished += 1
                expected[t.source_department] -= latency + env.inner.deadline_miss_penalty * missed
            assert all(abs(info["episode_returns"][a] - expected[a]) < 1e-9 for a in env.agents)
    assert unfinished > 0, "the unfinished-task reward path must actually be exercised"
    print(f"[PASS] test_rewards_go_to_the_source_agent_and_add_up_to_step6 ({unfinished} unfinished tasks)")


def test_local_observation_matches_the_global_state():
    """The acting agent's observation holds the same numbers as the Step 6
    observation (own load first); other agents see loads but no task."""
    env = MultiAgentHospitalEnv(os.path.join(ROOT, "configs", "scenarios", "overload.yaml"))
    rng = np.random.default_rng(1)
    for seed in range(3):
        env.reset(seed=seed)
        while not env.done:
            g = dict(zip(env.inner.observation_names, env.state().tolist()))
            for a in env.agents:
                o = dict(zip(env.observation_names[a], env.observe(a).tolist()))
                assert o["backlog_steps[own]"] == g[f"backlog_steps[{a}]"]
                assert all(o[f"backlog_steps[{d}]"] == g[f"backlog_steps[{d}]"] for d in env.agents if d != a)
                assert o["backlog_steps[Cloud]"] == g["backlog_steps[Cloud]"] and o["time_fraction"] == g["time_fraction"]
                if a == env.agent_selection:
                    assert g[f"source[{a}]"] == 1.0
                    same = ("task_demand_steps_on_own_node", "priority[Critical]", "priority[High]",
                            "priority[Routine]", "deadline_offset_fraction")
                    assert all(o[k] == g[k] for k in same)
                else:
                    assert o["task_demand_steps_on_own_node"] == 0.0 and o["deadline_offset_fraction"] == 0.0
                assert env.observation_spaces[a].contains(env.observe(a))
            env.step(int(rng.integers(0, 3)))
    print("[PASS] test_local_observation_matches_the_global_state")


def test_deterministic_with_the_same_seed_and_actions():
    env = MultiAgentHospitalEnv(os.path.join(ROOT, "configs", "scenarios", "overload.yaml"))
    for seed in range(3):
        runs = []
        for _ in range(2):
            rng = np.random.default_rng(seed)
            runs.append(_run_episode(env, seed, lambda e, a, o: int(rng.integers(0, 3))))
        (t1, i1), (t2, i2) = runs
        assert t1 == t2 and i1["report"] == i2["report"] and i1["episode_returns"] == i2["episode_returns"]
    print("[PASS] test_deterministic_with_the_same_seed_and_actions")


def test_empty_workload_and_invalid_usage():
    env = MultiAgentHospitalEnv(os.path.join(ROOT, "configs", "scenarios", "empty_workload.yaml"))
    info = env.reset(seed=0)
    assert env.done and env.agent_selection is None and info["report"]["headline"]["tasks"] == 0
    assert all(env.observation_spaces[a].contains(env.observe(a)) for a in env.agents)
    env = MultiAgentHospitalEnv(CONFIG_PATH)
    env.reset(seed=1)
    for bad in (3, -1, True, 1.0, "Local"):
        try:
            env.step(bad)
            raise AssertionError(f"Expected ValueError for action {bad!r}")
        except ValueError as e:
            assert "action" in str(e)
    assert env.inner.sim.tasks == [], "a rejected action must not place a task"
    try:
        env.observe("Pharmacy")
        raise AssertionError("Expected ValueError for an unknown agent")
    except ValueError as e:
        assert "agent" in str(e)
    while not env.done:
        env.step(LOCAL)
    try:
        env.step(LOCAL)
        raise AssertionError("Expected RuntimeError after the episode ended")
    except RuntimeError as e:
        assert "reset" in str(e)
    print("[PASS] test_empty_workload_and_invalid_usage")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL STEP 8 TESTS PASSED.")
