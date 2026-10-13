"""
Step 7 (pilot) - Single-agent DQN (Mnih et al. 2015): experience replay + target network.
================================================================================
ROMAN URDU: Ek centralized controller jo har task par Local / Edge / Cloud
mein se action chunta hai. Q-network har action ki "aage milne wala
discounted reward" estimate karta hai; training ke dauraan purane
transitions (replay buffer) se seekhta hai, aur target network ko har
`target_update_every` gradient steps baad copy kiya jata hai (stability).
Exploration: epsilon-greedy, epsilon linear tareeqe se kam hota hai.

Yeh PILOT hai: hyperparameters aam istemal hone wale default values hain,
TUNE NAHI kiye gaye (design note, Section 5/6).

ENGLISH: standard DQN with a replay buffer, a hard-updated target network,
Huber loss, Adam and gradient-norm clipping. Terminal transitions are not
bootstrapped (target = r). Everything is seeded for reproducibility.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000        # env steps collected before the first update
    target_update_every: int = 1_000    # gradient steps between target-network copies
    hidden_size: int = 128
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.5  # of total training steps
    max_grad_norm: float = 10.0
    # Debugging knobs (Step 7, one at a time; 1.0 = pilot behaviour):
    reward_scale: float = 1.0   # POSITIVE constant applied to rewards used for training (no clipping)
    obs_scale: float = 1.0      # factor on the unbounded observation features before the network

    def __post_init__(self):
        for name in ("reward_scale", "obs_scale", "learning_rate"):
            value = getattr(self, name)
            if not (isinstance(value, float) and np.isfinite(value) and value > 0):
                raise ValueError(f"DQNConfig.{name} must be a finite float > 0, got {value!r}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"DQNConfig.gamma must be in [0, 1], got {self.gamma!r}")

    def to_dict(self):
        return asdict(self)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular buffer of (obs, action, reward, next_obs, terminated)."""

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.size, self.pos = 0, 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.actions[idx]),
                torch.from_numpy(self.rewards[idx]), torch.from_numpy(self.next_obs[idx]),
                torch.from_numpy(self.terminated[idx]))


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig, seed: int, unbounded_mask=None):
        self.config, self.n_actions = config, n_actions
        # Which observation features obs_scale applies to (the env's unbounded ones).
        mask = np.zeros(obs_dim, dtype=bool) if unbounded_mask is None else np.asarray(unbounded_mask, dtype=bool)
        self._obs_factor = torch.ones(obs_dim)
        self._obs_factor[torch.from_numpy(mask)] = config.obs_scale
        self._scale_obs = config.obs_scale != 1.0
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q = QNetwork(obs_dim, n_actions, config.hidden_size)
        self.q_target = QNetwork(obs_dim, n_actions, config.hidden_size)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, self.rng)
        self.gradient_steps = 0

    def epsilon(self, step: int, total_steps: int) -> float:
        c = self.config
        decay_steps = max(1, int(c.epsilon_decay_fraction * total_steps))
        if step >= decay_steps:
            return c.epsilon_end
        return c.epsilon_start + (step / decay_steps) * (c.epsilon_end - c.epsilon_start)

    def net_input(self, obs: torch.Tensor) -> torch.Tensor:
        """Observation as fed to the network (obs_scale on unbounded features)."""
        return obs * self._obs_factor if self._scale_obs else obs

    def q_values(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return self.q(self.net_input(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)))[0]

    def greedy_action(self, obs) -> int:
        with torch.no_grad():
            q = self.q(self.net_input(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)))
        return int(torch.argmax(q, dim=1).item())  # ties -> lowest index (Local, Edge, Cloud)

    def act(self, obs, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return self.greedy_action(obs)

    def td_targets(self, rewards, next_obs, terminated):
        """r + gamma * (1 - terminated) * max_a' Q_target(s', a'); no bootstrap at the end."""
        with torch.no_grad():
            next_max = self.q_target(self.net_input(next_obs)).max(dim=1).values
        return rewards + self.config.gamma * (1.0 - terminated) * next_max

    def update(self) -> float:
        obs, actions, rewards, next_obs, terminated = self.buffer.sample(self.config.batch_size)
        targets = self.td_targets(rewards, next_obs, terminated)
        q_taken = self.q(self.net_input(obs)).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_taken, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())
