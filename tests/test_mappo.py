"""
Step 9 - Tests for MAPPO (agents/mappo.py, train_mappo.py).
================================================================================
ROMAN URDU: GAE haath ke hisaab se (episode ke end par bootstrap nahi), PPO
update sahi simt mein (achhe action ki probability badhti hai), greedy
action, same seed par bilkul same training (sirf training workload seeds),
aur yeh ke evaluate() ke andar actor wahi local observation dekhta hai jo
Step 8 ka multi-agent env deta hai.

Run with:  py -m pytest tests/test_mappo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agents.mappo import MAPPOAgent, MAPPOConfig
from agents.policy_eval import evaluate
from env.multi_agent_env import MultiAgentHospitalEnv
from train_mappo import greedy_policy, train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "environment.yaml")
OVERLOAD = os.path.join(ROOT, "configs", "scenarios", "overload.yaml")
AGENTS = ("Emergency", "ICU", "General_Ward")


def test_gae_by_hand_and_no_bootstrap_at_episode_end():
    """Two episodes in one batch: [r0, r1 | done] and [r2 | done].
    A_k = delta_k + gamma*lambda*A_{k+1} within an episode, delta uses V(next)
    only if the episode continues."""
    agent = MAPPOAgent(AGENTS, 10, 13, 3, MAPPOConfig(gamma=0.9, gae_lambda=0.8), seed=0)
    rewards, values, dones = [1.0, -2.0, 0.5], [0.3, -0.4, 0.2, 99.0], [False, True, True]
    adv, ret = agent.advantages(rewards, values, dones)
    d1 = -2.0 - (-0.4)                       # episode ends: no bootstrap
    d0 = 1.0 + 0.9 * (-0.4) - 0.3
    d2 = 0.5 - 0.2                           # values[3] = 99 must be ignored
    expected = [d0 + 0.9 * 0.8 * d1, d1, d2]
    assert np.allclose(adv, expected) and np.allclose(ret, np.array(expected) + values[:3])
    print("[PASS] test_gae_by_hand_and_no_bootstrap_at_episode_end")


def test_ppo_update_moves_the_policy_toward_the_rewarded_action():
    """One-step episodes (a bandit): only ICU's action 2 is rewarded. After a
    few updates ICU's probability of action 2 must have increased."""
    torch.set_num_threads(1)
    cfg = MAPPOConfig(rollout_decisions=200, minibatch_size=50, ppo_epochs=4, entropy_coef=0.0)
    agent = MAPPOAgent(AGENTS, 10, 13, 3, cfg, seed=0)
    obs, state = np.zeros(10, np.float32), np.zeros(13, np.float32)
    before = agent._dist("ICU", obs).probs[2].item()
    for _ in range(5):
        batch = {k: [] for k in ("state", "agent", "obs", "action", "log_prob", "reward", "done")}
        for _ in range(200):
            action, logp = agent.act("ICU", obs)
            for key, value in zip(batch, (state, "ICU", obs, action, logp, 1.0 if action == 2 else 0.0, True)):
                batch[key].append(value)
        agent.update(batch)
    after = agent._dist("ICU", obs).probs[2].item()
    assert after > before + 0.2, (before, after)
    assert agent._dist("Emergency", obs).probs[2].item() < 0.5, "other actors are not trained on ICU's data"
    print(f"[PASS] test_ppo_update_moves_the_policy_toward_the_rewarded_action ({before:.2f} -> {after:.2f})")


def test_greedy_action_and_config_validation():
    agent = MAPPOAgent(AGENTS, 10, 13, 3, MAPPOConfig(), seed=0)
    obs = np.linspace(0, 1, 10).astype(np.float32)
    probs = agent._dist("General_Ward", obs).probs
    assert agent.act("General_Ward", obs, greedy=True)[0] == int(torch.argmax(probs).item())
    for bad in ({"gamma": 1.5}, {"reward_scale": 0.0}, {"actor_lr": -1e-3}, {"minibatch_size": 0}, {"clip_eps": 0.0}):
        try:
            MAPPOConfig(**bad)
            raise AssertionError(f"Expected ValueError for {bad}")
        except ValueError:
            pass
    print("[PASS] test_greedy_action_and_config_validation")


def test_training_is_reproducible_and_uses_training_seeds_only():
    torch.set_num_threads(1)
    cfg = MAPPOConfig(rollout_decisions=300, minibatch_size=100, ppo_epochs=2)
    runs = [train(OVERLOAD, train_seed=3, total_steps=900, eval_every=450, cfg=cfg, log=lambda *_: None)
            for _ in range(2)]
    (a1, lc1, vc1, b1), (a2, lc2, vc2, b2) = runs
    assert lc1 == lc2 and vc1 == vc2 and b1["step"] == b2["step"]
    s1, s2 = a1.state_dict(), a2.state_dict()
    assert all(torch.equal(s1[g][k], s2[g][k]) for g in s1 for k in s1[g])
    assert all(10_000 <= row["workload_seed"] <= 19_999 for row in lc1)
    assert sum(row["critic_loss"] is not None for row in lc1) >= 3, "updates happened, incl. the last rollout"
    print("[PASS] test_training_is_reproducible_and_uses_training_seeds_only")


def test_evaluation_uses_the_same_local_observations_as_the_multi_agent_env():
    """greedy_policy() inside the shared evaluate() and a direct episode in
    MultiAgentHospitalEnv with the same actors give identical returns."""
    agent = MAPPOAgent(AGENTS, 10, 13, 3, MAPPOConfig(), seed=5)
    result = evaluate(OVERLOAD, greedy_policy(agent), range(3))
    env = MultiAgentHospitalEnv(OVERLOAD)
    for row in result["per_seed"]:
        env.reset(seed=row["seed"])
        while not env.done:
            a = env.agent_selection
            _, _, info = env.step(agent.act(a, env.observe(a), greedy=True)[0])
        assert sum(info["episode_returns"].values()) == row["return"]
    print("[PASS] test_evaluation_uses_the_same_local_observations_as_the_multi_agent_env")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL MAPPO TESTS PASSED.")
