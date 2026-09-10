"""
Step 2 - MedicalTaskGenerator: turns the static Step-1 simulator into a
dynamic one, by creating new tasks automatically using a Poisson arrival
process, separately for each department and each priority class.
================================================================================
ROMAN URDU: Step 1 mein tasks hum apne haath se `add_task()` karte thay.
Real hospital mein tasks khud aate rehte hain -- kabhi zyada, kabhi kam,
random lekin ek "average rate" ke around. Yeh statistical pattern "Poisson
process" kehlata hai. Yeh file har department (Emergency/ICU/General_Ward)
aur har priority class (Critical/High/Routine) ke liye ALAG Poisson process
chalati hai, kyunke ICU mein Critical task ka aana General Ward se zyada
common hai -- dono ki "rate" alag honi chahiye.

ENGLISH: In Step 1 we injected tasks by hand. In a real hospital, tasks
arrive on their own, unpredictably but around some average rate -- this
statistical pattern is a "Poisson process". This file runs one independent
Poisson process per (department, priority) pair, because a Critical task in
the ICU is far more likely than a Critical task in the General Ward -- each
pair needs its own rate.

WHY A SEPARATE FILE (not inside HospitalSimulator): keeps "how tasks arrive"
cleanly separate from "how the hospital processes them" -- one new idea at a
time, and each is independently testable (project roadmap rule).

DATA SOURCE WARNING: the arrival_rate and compute_demand_range numbers this
class reads (from configs/environment.yaml, section `task_generation`) are
PLACEHOLDER values -- see the warning comment in that config file. They are
clinically plausible but not yet derived from MIMIC-IV. This is a Step 2
LIMITATION, stated here on purpose so it is never silently forgotten.

Formula used (Poisson distribution, discrete time-step form):
    count ~ Poisson(arrival_rate)
Meaning: "count" is how many tasks of this priority arrive at this
department THIS time-step. On average, over many steps, the mean of
"count" converges to arrival_rate -- see tests/test_step2_task_generator.py
for the statistical check that proves this.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from env.task import PRIORITY_CLASSES


@dataclass
class GeneratedArrival:
    """One task the generator decided should arrive this time-step, before
    it becomes an actual `Task` object (that conversion happens in
    HospitalSimulator.add_generated_arrivals, Step 2 update)."""

    department: str
    priority: str
    compute_demand: float


class MedicalTaskGenerator:
    """Generates Critical/High/Routine tasks per department using
    independent Poisson arrival processes.

    Usage:
        gen = MedicalTaskGenerator(config, seed=config["seed"])
        arrivals = gen.generate_for_step(current_time)   # call once per step
        sim.add_generated_arrivals(arrivals)
    """

    def __init__(self, config: dict, seed: int):
        """
        Args:
            config: the full loaded environment.yaml dict (must contain a
                'task_generation' section -- see configs/environment.yaml).
            seed: random seed for NumPy's Generator, so the same seed always
                reproduces the same sequence of generated tasks (project rule:
                use reproducible random seeds).
        """
        task_generation_config = config.get("task_generation")
        if not task_generation_config or "departments" not in task_generation_config:
            raise ValueError(
                "Config is missing a 'task_generation.departments' section. "
                "Step 2 requires arrival_rate and compute_demand_range for "
                "every department and priority class -- see configs/environment.yaml."
            )

        self.department_settings: Dict[str, Dict[str, dict]] = task_generation_config["departments"]
        self._validate_settings()

        # np.random.default_rng is NumPy's modern, reproducible RNG (project
        # rule: reproducible seeds). Two generators built with the same seed
        # will always produce the exact same sequence of arrivals.
        self.rng = np.random.default_rng(seed)

    def _validate_settings(self) -> None:
        """Input validation with meaningful error messages (project coding rule).
        Checked once at construction time so a bad config fails immediately,
        not silently halfway through a long training run."""
        if not self.department_settings:
            raise ValueError("task_generation.departments must not be empty")

        for dept_name, priorities in self.department_settings.items():
            for priority in PRIORITY_CLASSES:
                if priority not in priorities:
                    raise ValueError(
                        f"task_generation.departments.{dept_name} is missing "
                        f"priority class '{priority}' (need all of {PRIORITY_CLASSES})"
                    )
                settings = priorities[priority]
                rate = settings.get("arrival_rate")
                if rate is None or rate < 0:
                    raise ValueError(
                        f"task_generation.departments.{dept_name}.{priority}.arrival_rate "
                        f"must be a number >= 0, got {rate!r}"
                    )
                demand_range = settings.get("compute_demand_range")
                if not demand_range or len(demand_range) != 2:
                    raise ValueError(
                        f"task_generation.departments.{dept_name}.{priority}.compute_demand_range "
                        f"must be a [low, high] pair, got {demand_range!r}"
                    )
                low, high = demand_range
                if low <= 0 or high < low:
                    raise ValueError(
                        f"task_generation.departments.{dept_name}.{priority}.compute_demand_range "
                        f"must satisfy 0 < low <= high, got [{low}, {high}]"
                    )

    def generate_for_step(self, current_time: int) -> List[GeneratedArrival]:
        """Draw this step's new arrivals for every department and priority class.

        For each (department, priority) pair independently:
          1. count ~ Poisson(arrival_rate)  -- how many tasks arrive this step
          2. for each of those `count` tasks, compute_demand ~ Uniform(low, high)

        Args:
            current_time: the simulator's current time-step (kept as an
                argument, not read from self, so the generator stays a pure
                function of (state, time) -- easier to unit-test and to
                later swap for a non-stationary/MIMIC-IV-calibrated rate that
                depends on the time of day).

        Returns:
            A list of GeneratedArrival (possibly empty, if nothing arrived
            this step -- perfectly normal for low arrival_rate values).
        """
        arrivals: List[GeneratedArrival] = []
        for dept_name, priorities in self.department_settings.items():
            for priority in PRIORITY_CLASSES:
                settings = priorities[priority]
                rate = settings["arrival_rate"]
                if rate == 0:
                    continue  # no point drawing Poisson(0); always 0 arrivals
                count = int(self.rng.poisson(rate))
                if count == 0:
                    continue
                low, high = settings["compute_demand_range"]
                for _ in range(count):
                    demand = float(self.rng.uniform(low, high))
                    arrivals.append(
                        GeneratedArrival(department=dept_name, priority=priority, compute_demand=demand)
                    )
        return arrivals
