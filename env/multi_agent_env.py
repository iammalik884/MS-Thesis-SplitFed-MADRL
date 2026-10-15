"""
Step 8 - Three-agent environment: each department is its own agent.
================================================================================
ROMAN URDU: Emergency, ICU aur General_Ward -- har department ek alag agent
hai. Har agent SIRF apne department mein aane wale tasks ka faisla karta hai
(Local / Edge / Cloud), apni local observation se. Sab agents ek hi hospital
simulation mein act karte hain, is liye ek agent ka offloading doosre
department ki queue aur load badal deta hai.

Yeh Step 6 ke HospitalSchedulingEnv par ek WRAPPER hai: simulator, reward ki
definition, cloud capacity aur evaluation protocol bilkul wahi hain.

TURNS (faisle ki tarteeb): ek step ke tasks usi seeded order mein aate hain
jis mein FCFS router aur Step 6/7 ka env unhein dikhata hai; har task par us
ke SOURCE department ka agent faisla karta hai (`agent_selection`). Ek hi
simulation step mein teeno agents act karte hain (agar teeno ke tasks aaye).
Har faisla pichle faislon ka asar (backlog, raste wale tasks samet) dekhta
hai -- bilkul FCFS ki tarah. Is liye FCFS, ECT, DQN aur aage MAPPO sab ka
faisle ka process ek jaisa hai aur comparison fair rehta hai.

LOCAL OBSERVATION (har agent ki, float32, 3 departments par 10 values):
    apne node ka backlog / apni capacity
    baqi do departments ka backlog / un ki capacity   (department order mein)
    cloud ka backlog / cloud capacity
    task ki demand / apni capacity
    priority one-hot, deadline offset / sab se badi offset, time / n_steps
  Doosre departments ka sirf LOAD dikhta hai (hospital network par shared
  load signal), un ke tasks ya priorities nahi. Jis agent ki baari nahi, us
  ki observation mein task wale features 0 hote hain.
GLOBAL STATE (state()): Step 6 ki 13 values -- Step 9 ke centralized critic
  ke liye (CTDE: sirf training mein).

REWARD: har task ka reward (Step 6 jaisa: -(latency) - penalty x miss) us
task ke SOURCE department ke agent ko milta hai. Team reward = teeno ka jor
= Step 6 ka reward, bilkul barabar (step() har baar check karta hai).

ENGLISH: turn-based (AEC-style) multi-agent wrapper around the Step 6 env.
reset(seed) -> info; then repeatedly: agent = env.agent_selection,
obs = env.observe(agent), rewards, terminated, info = env.step(action).
"""

from typing import Dict, Optional

import numpy as np
from gymnasium import spaces

from env.gym_env import ACTIONS, HospitalSchedulingEnv
from env.task import PRIORITY_CLASSES


class MultiAgentHospitalEnv:
    """Three department agents sharing one hospital simulation."""

    def __init__(self, config):
        self.inner = HospitalSchedulingEnv(config)
        self.agents = tuple(self.inner.departments)
        n = len(self.agents)
        self.observation_names: Dict[str, tuple] = {
            a: ("backlog_steps[own]",
                *(f"backlog_steps[{d}]" for d in self.agents if d != a),
                "backlog_steps[Cloud]",
                "task_demand_steps_on_own_node",
                *(f"priority[{p}]" for p in PRIORITY_CLASSES),
                "deadline_offset_fraction",
                "time_fraction")
            for a in self.agents}
        size = len(self.observation_names[self.agents[0]])
        high = np.ones(size, dtype=np.float32)
        high[: n + 2] = np.inf  # backlogs (all departments + cloud) and task demand are unbounded
        box = spaces.Box(low=np.zeros(size, dtype=np.float32), high=high, dtype=np.float32)
        self.observation_spaces = {a: box for a in self.agents}
        self.action_spaces = {a: spaces.Discrete(len(ACTIONS)) for a in self.agents}
        self.state_space = self.inner.observation_space
        self.agent_selection: Optional[str] = None
        self.done = True

    def reset(self, seed: Optional[int] = None) -> Dict:
        _, info = self.inner.reset(seed=seed)
        self._seen = set()
        self.episode_returns = {a: 0.0 for a in self.agents}
        self.done = False
        if info["no_tasks_in_episode"]:      # no decision at all: finish the episode right away
            _, _, _, _, end_info = self.inner.step(0)
            self.done = True
            info["report"] = end_info["report"]
        self.agent_selection = self._next_agent()
        info["agent_selection"] = self.agent_selection
        return info

    def step(self, action):
        """The agent in `agent_selection` decides its department's pending task.
        Returns (rewards per agent, terminated, info)."""
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        agent = self.agent_selection
        _, team_reward, terminated, _, info = self.inner.step(action)   # validates the action
        rewards = self._rewards_by_source(info)
        if abs(sum(rewards.values()) - team_reward) > 1e-9 * max(1.0, abs(team_reward)):
            raise AssertionError(f"per-agent rewards {rewards} do not add up to the team reward {team_reward}")
        for a in self.agents:
            self.episode_returns[a] += rewards[a]
        self.done = terminated
        self.agent_selection = self._next_agent()
        info.update(agent=agent, team_reward=team_reward, agent_selection=self.agent_selection)
        if terminated:
            info["episode_returns"] = dict(self.episode_returns)
        return rewards, terminated, info

    def observe(self, agent: str) -> np.ndarray:
        """Local observation of `agent` (task features only on its own turn)."""
        if agent not in self.agents:
            raise ValueError(f"observe: unknown agent {agent!r}; agents are {self.agents}")
        sim, n = self.inner.sim, len(self.agents)
        order = [agent] + [d for d in self.agents if d != agent]
        obs = np.zeros(len(self.observation_names[agent]), dtype=np.float32)
        obs[:n] = [sim.nodes[d].backlog_demand / sim.nodes[d].capacity for d in order]
        obs[n] = sim.cloud_node.backlog_demand / self.inner.cloud_capacity
        if agent == self.agent_selection and not self.done:
            arrival = self.inner._pending[0]
            obs[n + 1] = arrival.compute_demand / sim.nodes[agent].capacity
            obs[n + 2 + PRIORITY_CLASSES.index(arrival.priority)] = 1.0
            if self.inner.max_deadline_offset > 0:
                obs[n + 5] = self.inner.deadline_offsets[arrival.priority] / self.inner.max_deadline_offset
            obs[n + 6] = sim.current_time / self.inner.settings["window_steps"]
        else:
            obs[n + 6] = (1.0 if self.done else sim.current_time / self.inner.settings["window_steps"])
        return obs

    def state(self) -> np.ndarray:
        """Global state (the Step 6 observation) for a centralized critic."""
        return self.inner._observation()

    def _next_agent(self) -> Optional[str]:
        return self.inner._pending[0].department if self.inner._pending and not self.inner._done else None

    def _rewards_by_source(self, info) -> Dict[str, float]:
        """Rewards the inner env paid in this step, split by the task's source
        department (same formula as HospitalSchedulingEnv._task_reward)."""
        rewards = {a: 0.0 for a in self.agents}
        cutoff = info["report"]["protocol"]["final_cutoff"] if "report" in info else None
        for task in self.inner.sim.tasks:
            if task.task_id in self.inner._rewarded and task.task_id not in self._seen:
                self._seen.add(task.task_id)
                if task.completion_time is not None:
                    r = self.inner._task_reward(task.latency, task.completion_time > task.deadline)
                else:                                   # unfinished after the drain (episode end only)
                    r = self.inner._task_reward(cutoff - task.arrival_time + 1, task.deadline <= cutoff)
                rewards[task.source_department] += r
        return rewards
