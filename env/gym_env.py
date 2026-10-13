"""
Step 6 - Gymnasium environment: the hospital simulator behind reset()/step().
================================================================================
ROMAN URDU: Is file mein simulator ko standard RL interface (Gymnasium) mein
lapeta gaya hai, taake Step 7 ka DQN agent observe -> act -> reward ka loop
chala sake. Simulator ka koi rule nahi badla; sirf "task kahan jaye" ka
faisla ab FCFS router ke bajaye agent ka action karta hai.

DECISION POINT: har naye task par EK faisla. Ek step ke saare arrivals usi
seeded order mein ek ek kar ke agent ko dikhaye jate hain jis order mein FCFS
unhein route karta hai. Us step ke aakhri task ke baad simulator ek step
chalta hai; jin steps mein koi task nahi aata woh khud-ba-khud chal jate hain.

ACTION  (Discrete(3), roadmap ke mutabiq):
    0 = Local  -> task ke apne department ka node
    1 = Edge   -> padosi department ka node; KAUNSA padosi, yeh wahi rule
                  chunta hai jo FCFS istemal karta hai (sab se kam backlog,
                  tie par fixed order) -- TaskRouter.least_loaded_neighbour()
    2 = Cloud  -> shared cloud
  Koi action mana nahi: agent chhota task Cloud bhi bhej sakta hai.

OBSERVATION (Box, float32) -- sirf woh jo faisle ke waqt maloom hai:
    har department: backlog / capacity   (kitne steps ka kaam pehle se hai)
    cloud: backlog / cloud capacity
    task: compute_demand / apne department ki capacity
    source department one-hot, priority one-hot
    deadline offset / sab se badi offset   (0..1)
    current_time / n_steps                 (0..1)
  "backlog" mein raste wale (transit) tasks bhi shamil hain, bilkul FCFS
  router ki tarah. Yeh poori hidden state nahi (har task ki detail nahi),
  is liye environment partially observable hai.

REWARD: har task ka reward us ke KHATAM hone par milta hai:
    -(latency) - deadline_miss_penalty * (1 agar deadline miss)
  Jo task drain ke baad bhi adhoora reh jaye: latency ki kam se kam qeemat
  (final_cutoff - arrival_time + 1) aur, agar deadline guzar chuki ho,
  penalty -- episode ke aakhri step par. Har task ka reward EXACTLY EK baar.
  Episode ka total reward = -(sab latency) - penalty * (deadline misses),
  evaluation metrics ki wohi definitions (Step 5).

TERMINATION: window (n_steps) ke baad koi naya task nahi aata, is liye
aakhri faisle ke baad simulator drain chalata hai aur episode terminated=True
hota hai. Yeh episode ka asal (MDP ka) khatma hai -- observation mein time
shamil hai -- is liye truncated hamesha False hai.

ENGLISH: one decision per arriving task; Discrete(3) actions Local / Edge /
Cloud (Edge's neighbour picked by the same rule as FCFS); a float32 vector
observation of node backlogs and the pending task; per-task reward
-(latency) - penalty * [deadline missed], paid when the task finishes;
episodes end after the measurement window + drain of the Step 5 protocol.
The final step's info["report"] is the same report summarize_protocol()
produces for FCFS, so learned policies are evaluated identically.
"""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.evaluation import backlog_snapshot, drain_and_report, protocol_settings
from env.hospital_env import HospitalSimulator, load_config
from env.task import PRIORITY_CLASSES, is_int
from env.task_generator import MedicalTaskGenerator

ACTIONS = ("Local", "Edge", "Cloud")


class HospitalSchedulingEnv(gym.Env):
    """Gymnasium wrapper around HospitalSimulator for the Local/Edge/Cloud decision."""

    metadata = {"render_modes": []}

    def __init__(self, config):
        super().__init__()
        self.config = load_config(config)
        probe = HospitalSimulator(self.config)  # validates the whole config
        for section in ("offloading", "deadlines", "task_generation", "rl_environment"):
            if not self.config.get(section):
                raise ValueError(f"HospitalSchedulingEnv needs a non-empty '{section}' section in the config")
        self.departments: List[str] = list(probe.nodes)
        self.capacities = np.array([probe.nodes[d].capacity for d in self.departments], dtype=np.float64)
        self.cloud_capacity = probe.cloud_node.capacity
        self.deadline_offsets: Dict[str, int] = dict(self.config["deadlines"])
        self.max_deadline_offset = max(self.deadline_offsets.values())
        self.deadline_miss_penalty = float(self.config["rl_environment"]["deadline_miss_penalty"])
        self.settings = protocol_settings(self.config)

        n = len(self.departments)
        self.observation_names: Tuple[str, ...] = (
            *(f"backlog_steps[{d}]" for d in self.departments),
            "backlog_steps[Cloud]",
            "task_demand_steps_on_own_node",
            *(f"source[{d}]" for d in self.departments),
            *(f"priority[{p}]" for p in PRIORITY_CLASSES),
            "deadline_offset_fraction",
            "time_fraction",
        )
        low = np.zeros(len(self.observation_names), dtype=np.float32)
        high = np.ones(len(self.observation_names), dtype=np.float32)
        high[: n + 2] = np.inf  # backlogs (departments + cloud) and task demand are unbounded
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))

        self.sim: Optional[HospitalSimulator] = None
        self._done = True

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """seed (int >= 0) is used AS the task-generator seed, so reset(seed=s)
        reproduces exactly the workload of run_fcfs_evaluation(config, s).
        Without a seed, a generator seed is drawn from self.np_random."""
        if seed is not None and (not is_int(seed) or seed < 0):
            raise ValueError(f"reset: seed must be an int >= 0 or None, got {seed!r}")
        super().reset(seed=seed)
        self.workload_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.sim = HospitalSimulator(self.config)
        self.generator = MedicalTaskGenerator(self.sim.config, seed=self.workload_seed)
        self._pending: List = []
        self._rewarded = set()
        self._episode_return = 0.0
        self._report: Optional[Dict] = None
        self._done = False
        self._advance_to_next_arrival()  # may already end the window (no tasks at all)
        info = {"workload_seed": self.workload_seed, "time": self.sim.current_time,
                "no_tasks_in_episode": not self._pending}
        return self._observation(), info

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"step: action must be an int in {{0, 1, 2}} ({ACTIONS}), got {action!r}")
        info: Dict = {}
        if self._pending:
            arrival = self._pending.pop(0)
            task = self.sim.place_task(arrival.department, arrival.compute_demand, arrival.priority,
                                       ACTIONS[int(action)])
            info.update(placed_task_id=task.task_id, tier=task.tier, destination=task.destination)
            if not self._pending:          # last task of this step placed -> run the step
                self.sim.step()
                self._advance_to_next_arrival()
        # else: an episode with no tasks at all -- the action has nothing to act on.

        reward = self._collect_completion_rewards()
        terminated = not self._pending
        if terminated:
            reward += self._finish_episode()
            info["report"] = self._report
        self._episode_return += reward
        if terminated:
            info["episode_return"] = self._episode_return
        info["time"] = self.sim.current_time
        return self._observation(), float(reward), terminated, False, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def fcfs_action(self) -> int:
        """The action the Step 5 FCFS router would take for the pending task
        (used to check that this environment reproduces the FCFS baseline)."""
        if not self._pending:
            raise RuntimeError("fcfs_action: no pending task")
        arrival = self._pending[0]
        backlogs = {name: node.backlog_demand for name, node in self.sim.nodes.items()}
        tier, _ = self.sim.router.choose_destination(arrival.department, arrival.compute_demand, backlogs)
        return ACTIONS.index(tier)

    def _advance_to_next_arrival(self) -> None:
        """Run window steps until a step brings arrivals (they become the
        pending decisions) or the measurement window ends. The generator is
        called once for every window step, exactly as in run_fcfs_evaluation."""
        while self.sim.current_time < self.settings["window_steps"]:
            arrivals = self.generator.generate_for_step(self.sim.current_time)
            if arrivals:
                self.sim._check_batch(arrivals)  # all-or-nothing, as route_generated_arrivals()
                self._pending = list(arrivals)
                return
            self.sim.step()

    def _task_reward(self, latency: int, missed: bool) -> float:
        return -float(latency) - (self.deadline_miss_penalty if missed else 0.0)

    def _collect_completion_rewards(self) -> float:
        total = 0.0
        for task in self.sim.tasks:
            if task.completion_time is not None and task.task_id not in self._rewarded:
                self._rewarded.add(task.task_id)
                total += self._task_reward(task.latency, task.completion_time > task.deadline)
        return total

    def _finish_episode(self) -> float:
        """Drain + protocol report; unfinished tasks get their reward here."""
        self._report = drain_and_report(self.sim, self.settings, backlog_snapshot(self.sim))
        total = self._collect_completion_rewards()       # tasks finished during the drain
        cutoff = self._report["protocol"]["final_cutoff"]
        for task in self.sim.tasks:
            if task.task_id not in self._rewarded:       # still unfinished after the drain
                self._rewarded.add(task.task_id)
                total += self._task_reward(cutoff - task.arrival_time + 1, task.deadline <= cutoff)
        self._done = True
        return total

    def _observation(self) -> np.ndarray:
        n = len(self.departments)
        obs = np.zeros(len(self.observation_names), dtype=np.float32)
        obs[:n] = [self.sim.nodes[d].backlog_demand for d in self.departments] / self.capacities
        obs[n] = self.sim.cloud_node.backlog_demand / self.cloud_capacity
        if self._pending and not self._done:
            arrival = self._pending[0]
            src = self.departments.index(arrival.department)
            obs[n + 1] = arrival.compute_demand / self.capacities[src]
            obs[n + 2 + src] = 1.0
            obs[2 * n + 2 + PRIORITY_CLASSES.index(arrival.priority)] = 1.0
            if self.max_deadline_offset > 0:
                obs[2 * n + 5] = self.deadline_offsets[arrival.priority] / self.max_deadline_offset
            obs[2 * n + 6] = self.sim.current_time / self.settings["window_steps"]
        else:
            obs[2 * n + 6] = 1.0  # no decision left: the window is over
        return obs
