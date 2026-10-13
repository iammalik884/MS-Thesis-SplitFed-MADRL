"""
Step 7 (pilot) - tests for the DQN components and the shared policy evaluation.
================================================================================
ROMAN URDU: Chhote, tez checks: replay buffer, TD target (terminal par
bootstrap nahi), epsilon schedule, greedy action, same seed par training
bilkul same, aur yeh ke evaluate() ka FCFS result Step 5 ke FCFS jaisa hai.

Run with:  py -m pytest tests/test_dqn_pilot.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agents.dqn import DQNAgent, DQNConfig, ReplayBuffer
from agents.policy_eval import baseline_policies, evaluate
from env.evaluation import run_fcfs_evaluation
from train_dqn_pilot import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "environment.yaml")


def test_replay_buffer_stores_and_wraps():
    buf = ReplayBuffer(capacity=3, obs_dim=2, rng=np.random.default_rng(0))
    for i in range(5):
        buf.add(np.array([i, i], np.float32), i % 3, float(i), np.array([i + 1, i + 1], np.float32), i == 4)
    assert buf.size == 3 and buf.pos == 2
    assert sorted(buf.rewards.tolist()) == [2.0, 3.0, 4.0], "oldest entries overwritten"
    assert buf.terminated[(buf.pos - 1) % 3] == 1.0
    obs, actions, rewards, next_obs, term = buf.sample(8)
    assert obs.shape == (8, 2) and torch.all(next_obs == obs + 1)
    print("[PASS] test_replay_buffer_stores_and_wraps")


def test_td_target_does_not_bootstrap_at_episode_end():
    agent = DQNAgent(obs_dim=2, n_actions=3, config=DQNConfig(gamma=0.9), seed=0)
    next_obs = torch.zeros((2, 2))
    with torch.no_grad():
        next_max = agent.q_target(next_obs).max(dim=1).values
    targets = agent.td_targets(torch.tensor([-1.0, -1.0]), next_obs, torch.tensor([0.0, 1.0]))
    assert torch.isclose(targets[0], -1.0 + 0.9 * next_max[0])
    assert targets[1] == -1.0, "terminated -> target is the reward only"
    print("[PASS] test_td_target_does_not_bootstrap_at_episode_end")


def test_epsilon_schedule_and_greedy_action():
    agent = DQNAgent(obs_dim=2, n_actions=3, config=DQNConfig(epsilon_decay_fraction=0.5), seed=0)
    assert agent.epsilon(0, 100) == 1.0 and abs(agent.epsilon(25, 100) - 0.525) < 1e-12
    assert agent.epsilon(50, 100) == 0.05 and agent.epsilon(99, 100) == 0.05
    obs = np.array([0.3, -0.2], np.float32)
    with torch.no_grad():
        expected = int(torch.argmax(agent.q(torch.from_numpy(obs).unsqueeze(0))).item())
    assert agent.greedy_action(obs) == expected and agent.act(obs, epsilon=0.0) == expected
    print("[PASS] test_epsilon_schedule_and_greedy_action")


def test_training_is_reproducible_with_the_same_seed():
    torch.set_num_threads(1)
    cfg = DQNConfig(learning_starts=100, buffer_size=2_000, target_update_every=50)
    runs = [train(CONFIG_PATH, train_seed=3, total_steps=600, eval_every=300, dqn_config=cfg, log=lambda *_: None)
            for _ in range(2)]
    (a1, lc1, vc1, b1), (a2, lc2, vc2, b2) = runs
    assert lc1 == lc2 and vc1 == vc2
    assert all(torch.equal(a1.q.state_dict()[k], a2.q.state_dict()[k]) for k in a1.q.state_dict())
    assert lc1[-1]["step"] >= 600 and len(vc1) >= 2
    seeds = {row["workload_seed"] for row in lc1}
    assert all(10_000 <= s <= 19_999 for s in seeds), "training uses only training workload seeds"
    print("[PASS] test_training_is_reproducible_with_the_same_seed")


def test_evaluate_matches_step5_fcfs():
    """The shared evaluation routine with the FCFS policy gives the Step 5 FCFS headline."""
    result = evaluate(CONFIG_PATH, baseline_policies()["FCFS"], range(3))
    for row in result["per_seed"]:
        headline = run_fcfs_evaluation(CONFIG_PATH, row["seed"]).report["headline"]
        assert row["avg_latency"] == headline["avg_latency"]
        assert row["critical_deadline_violation_rate"] == headline["critical_deadline_violation_rate"]
    assert abs(sum(result["action_proportions"].values()) - 1.0) < 1e-12
    print("[PASS] test_evaluate_matches_step5_fcfs")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL DQN PILOT TESTS PASSED.")
