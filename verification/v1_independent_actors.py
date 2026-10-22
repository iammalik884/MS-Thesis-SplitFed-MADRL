"""V1 - Each department has an independent actor; the centralized critic is used only in training.
Checks one saved Step 10 model (overload, training seed 0, best checkpoint)."""
import glob
import json
import os

import numpy as np
import torch

from common import ROOT, write
from agents.mappo import MAPPOAgent, MAPPOConfig
from agents.policy_eval import evaluate
from env.multi_agent_env import MultiAgentHospitalEnv
from train_mappo import greedy_policy

torch.set_num_threads(1)
run = sorted(glob.glob("results/step10_final/training_runs/FINAL_overload_seed0_*"))[0]
m = json.load(open(os.path.join(run, "manifest.json")))
env = MultiAgentHospitalEnv(m["config_path"])
agent = MAPPOAgent(env.agents, env.observation_spaces[env.agents[0]].shape[0], env.state_space.shape[0],
                   env.action_spaces[env.agents[0]].n, MAPPOConfig(**m["mappo_config"]), seed=m["train_seed"])
agent.load_state_dict(torch.load(os.path.join(run, "model_best.pt")))
lines = [f"# V1 - Independent actors, critic only in training", "", f"Model: `{os.path.relpath(run, ROOT).replace(chr(92), chr(47))}/model_best.pt`", ""]

ptrs = {a: {p.data_ptr() for p in agent.actors[a].parameters()} for a in env.agents}
n_params = {a: sum(p.numel() for p in agent.actors[a].parameters()) for a in env.agents}
shared = set.intersection(*ptrs.values())
w = {a: agent.actors[a][0].weight.detach() for a in env.agents}
lines += [f"- Actors: {list(agent.actors.keys())}, parameters each: {n_params}",
          f"- Parameter tensors shared between actors: {len(shared)}",
          f"- First-layer weights identical? Emergency-ICU: {torch.equal(w['Emergency'], w['ICU'])}, "
          f"ICU-General_Ward: {torch.equal(w['ICU'], w['General_Ward'])}",
          f"- Actor input size: {agent.actors['ICU'][0].in_features} (local observation); "
          f"critic input size: {agent.critic[0].in_features} (global state)"]

rng = np.random.default_rng(0)
differ = sum(len({agent.act(a, o, greedy=True)[0] for a in env.agents}) > 1
             for o in (rng.random(10).astype(np.float32) for _ in range(1000)))
lines.append(f"- Random local observations (1000, seed 0) on which the three actors choose different actions: {differ}")

test_seeds = range(2000, 2030)
with_critic = evaluate(m["config_path"], greedy_policy(agent), test_seeds)["mean_return"]
calls = {"n": 0}
original_state = MultiAgentHospitalEnv.state
MultiAgentHospitalEnv.state = lambda self: (calls.__setitem__("n", calls["n"] + 1), original_state(self))[1]
agent.critic = None                                   # execution must not need the critic
without_critic = evaluate(m["config_path"], greedy_policy(agent), test_seeds)["mean_return"]
MultiAgentHospitalEnv.state = original_state
committed = json.load(open("results/step10_final/summary.json", encoding="utf-8"))[
    "results"]["overload"]["MAPPO (best)"]["per_training_seed_return"][m["train_seed"]]
lines += [f"- Test return (seeds 2000-2029) with critic: {with_critic}; critic deleted: {without_critic}; "
          f"committed result: {committed}; identical: {with_critic == without_critic == committed}",
          f"- Calls of env.state() (global state) during critic-free evaluation: {calls['n']}"]
assert not shared and with_critic == without_critic == committed and calls["n"] == 0
write("v1_independent_actors.md", "\n".join(lines) + "\n")
