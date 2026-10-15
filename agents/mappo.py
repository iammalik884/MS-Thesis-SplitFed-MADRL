"""
Step 9 - MAPPO with CTDE (centralized training, decentralized execution).
================================================================================
ROMAN URDU: Har department (Emergency, ICU, General_Ward) ka APNA actor
network hai jo sirf apni local observation (Step 8, 10 values) se Local /
Edge / Cloud chunta hai. Training ke dauraan ek CENTRALIZED critic global
state (Step 6 ki 13 values) dekh kar team ki value V(s) seekhta hai. Faisle
ke waqt (execution) sirf actor chalta hai -- critic ki zaroorat nahi.

Reward: TEAM reward (teeno agents ka jor = Step 6 ka reward) -- cooperative
objective (roadmap Step 9: "Does joint reward improve?").

Faisle ek tarteeb mein hote hain (Step 8: turn-based), is liye ek episode ek
lambi sequence hai: (state_k, agent_k, local_obs_k, action_k, team_reward_k).
Advantage GAE (Schulman et al. 2016) se is poori sequence par critic ke
saath nikalta hai; har agent ka actor sirf apne faislon par PPO clipped loss
(Schulman et al. 2017) se update hota hai. MAPPO: Yu et al. 2022.

ENGLISH: separate actor per agent (no parameter sharing), one centralized
critic V(global state), team reward scaled by a positive constant for
training only, GAE over the joint decision sequence, PPO clipped updates.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    rollout_decisions: int = 1_000   # decisions collected (whole episodes) before each update
    ppo_epochs: int = 10
    minibatch_size: int = 250
    entropy_coef: float = 0.01
    max_grad_norm: float = 10.0
    hidden_size: int = 64
    reward_scale: float = 0.01      # POSITIVE constant on the team reward used for training (no clipping)

    def __post_init__(self):
        for name in ("actor_lr", "critic_lr", "reward_scale", "clip_eps"):
            value = getattr(self, name)
            if not (isinstance(value, float) and np.isfinite(value) and value > 0):
                raise ValueError(f"MAPPOConfig.{name} must be a finite float > 0, got {value!r}")
        for name in ("gamma", "gae_lambda"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"MAPPOConfig.{name} must be in [0, 1], got {getattr(self, name)!r}")
        for name in ("rollout_decisions", "ppo_epochs", "minibatch_size", "hidden_size"):
            if not (isinstance(getattr(self, name), int) and getattr(self, name) > 0):
                raise ValueError(f"MAPPOConfig.{name} must be an int > 0, got {getattr(self, name)!r}")

    def to_dict(self):
        return asdict(self)


def _mlp(n_in: int, n_out: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


class MAPPOAgent:
    def __init__(self, agents: Sequence[str], obs_dim: int, state_dim: int, n_actions: int,
                 config: MAPPOConfig, seed: int):
        self.agents, self.config, self.n_actions = tuple(agents), config, n_actions
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.actors = nn.ModuleDict({a: _mlp(obs_dim, n_actions, config.hidden_size) for a in self.agents})
        self.critic = _mlp(state_dim, 1, config.hidden_size)
        self.actor_opt = {a: torch.optim.Adam(self.actors[a].parameters(), lr=config.actor_lr) for a in self.agents}
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    # ---- acting (decentralized: local observation only) -----------------------
    def _dist(self, agent: str, obs) -> torch.distributions.Categorical:
        logits = self.actors[agent](torch.as_tensor(np.asarray(obs), dtype=torch.float32))
        return torch.distributions.Categorical(logits=logits)

    def act(self, agent: str, obs, greedy: bool = False):
        """Returns (action, log-probability). greedy=True -> most likely action
        (ties -> lowest index: Local, Edge, Cloud), used for evaluation."""
        with torch.no_grad():
            dist = self._dist(agent, obs)
            if greedy:
                action = int(torch.argmax(dist.probs).item())
            else:
                cdf = np.cumsum(dist.probs.numpy().astype(np.float64))
                action = min(int(np.searchsorted(cdf, self.rng.random() * cdf[-1], side="right")), self.n_actions - 1)
            return action, float(dist.log_prob(torch.tensor(action)).item())

    def value(self, states) -> np.ndarray:
        with torch.no_grad():
            return self.critic(torch.as_tensor(np.asarray(states), dtype=torch.float32)).squeeze(-1).numpy()

    # ---- learning (centralized critic) ----------------------------------------
    def advantages(self, rewards, values, dones):
        """GAE over the joint decision sequence. rewards are already scaled;
        dones[k] = True if decision k ended its episode (no bootstrap)."""
        c = self.config
        adv = np.zeros(len(rewards), dtype=np.float64)
        last = 0.0
        for k in reversed(range(len(rewards))):
            next_value = 0.0 if dones[k] else values[k + 1]
            delta = rewards[k] + c.gamma * next_value - values[k]
            last = delta + c.gamma * c.gae_lambda * (0.0 if dones[k] else last)
            adv[k] = last
        return adv, adv + np.asarray(values[: len(rewards)], dtype=np.float64)

    def update(self, batch: Dict[str, List]) -> Dict[str, float]:
        """batch: lists over decisions -- state, agent, obs, action, log_prob,
        reward (scaled), done. The rollout always ends with a finished episode."""
        c = self.config
        values = self.value(batch["state"])
        adv, returns = self.advantages(batch["reward"], np.append(values, 0.0), batch["done"])
        states = torch.as_tensor(np.asarray(batch["state"]), dtype=torch.float32)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        stats = {"critic_loss": [], "actor_loss": [], "entropy": []}
        n = len(adv)
        for _ in range(c.ppo_epochs):
            for idx in np.array_split(self.rng.permutation(n), max(1, n // c.minibatch_size)):
                loss = nn.functional.mse_loss(self.critic(states[idx]).squeeze(-1), returns_t[idx])
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                self.critic_opt.step()
                stats["critic_loss"].append(float(loss.item()))
        agents = np.asarray(batch["agent"])
        for a in self.agents:
            mine = np.flatnonzero(agents == a)
            if len(mine) == 0:
                continue
            obs = torch.as_tensor(np.asarray(batch["obs"])[mine], dtype=torch.float32)
            actions = torch.as_tensor(np.asarray(batch["action"])[mine])
            old_logp = torch.as_tensor(np.asarray(batch["log_prob"])[mine], dtype=torch.float32)
            a_adv = adv[mine]
            a_adv = torch.as_tensor((a_adv - a_adv.mean()) / (a_adv.std() + 1e-8), dtype=torch.float32)
            m = len(mine)
            for _ in range(c.ppo_epochs):
                for idx in np.array_split(self.rng.permutation(m), max(1, m // c.minibatch_size)):
                    dist = torch.distributions.Categorical(logits=self.actors[a](obs[idx]))
                    ratio = torch.exp(dist.log_prob(actions[idx]) - old_logp[idx])
                    surrogate = torch.min(ratio * a_adv[idx],
                                          torch.clamp(ratio, 1 - c.clip_eps, 1 + c.clip_eps) * a_adv[idx])
                    entropy = dist.entropy().mean()
                    loss = -surrogate.mean() - c.entropy_coef * entropy
                    self.actor_opt[a].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[a].parameters(), c.max_grad_norm)
                    self.actor_opt[a].step()
                    stats["actor_loss"].append(float(loss.item()))
                    stats["entropy"].append(float(entropy.item()))
        return {k: float(np.mean(v)) if v else None for k, v in stats.items()}

    def state_dict(self):
        return {"actors": self.actors.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, state):
        self.actors.load_state_dict(state["actors"])
        self.critic.load_state_dict(state["critic"])
