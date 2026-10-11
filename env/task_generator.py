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

from env.config_validation import validate_task_generation
from env.task import PRIORITY_CLASSES, is_int


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

    def __init__(self, config: dict, seed: int, shuffle_arrivals: bool = True):
        """
        Args:
            config: the full loaded environment.yaml dict (must contain a
                'task_generation' section -- see configs/environment.yaml).
            seed: random seed for NumPy's Generator, so the same seed always
                reproduces the same sequence of generated tasks (project rule:
                use reproducible random seeds).
            shuffle_arrivals: V2 -- shuffle each step's arrivals with a seeded
                RNG (default True). False keeps the raw loop order
                (department -> Critical/High/Routine); only for tests.
        """
        # V2: shared validator (env/config_validation.py) -- NaN/inf rates,
        # unknown/missing departments and unknown priority keys are rejected.
        department_names = [d["name"] for d in config["departments"]] if config.get("departments") else None
        validate_task_generation(config.get("task_generation"), department_names)
        if not is_int(seed) or seed < 0:
            raise ValueError(f"MedicalTaskGenerator: seed must be an int >= 0, got {seed!r}")
        self.department_settings: Dict[str, Dict[str, dict]] = config["task_generation"]["departments"]

        # np.random.default_rng is NumPy's modern, reproducible RNG (project
        # rule: reproducible seeds). Two generators built with the same seed
        # will always produce the exact same sequence of arrivals.
        self.rng = np.random.default_rng(seed)
        # V2 (AUDIT FIX): ALAG seeded RNG sirf same-step arrivals ka order
        # shuffle karne ke liye. Alag stream is liye ke workload (counts +
        # demands) same seed par purane version jaisa hi rahe -- sirf order
        # badle. SeedSequence.spawn() independent child stream deta hai.
        # ENGLISH: separate seeded stream used ONLY to shuffle the order of
        # same-step arrivals, so the generated workload for a seed is unchanged.
        self.order_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        self.shuffle_arrivals = shuffle_arrivals

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
            this step -- perfectly normal for low arrival_rate values), in a
            seeded random order (V2) -- this order becomes the enqueue order.
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
        # V2 (AUDIT FIX): loop order (department -> Critical/High/Routine) se
        # enqueue order tay hota tha, yani same-step Critical tasks aur config
        # mein pehle likhe department hamesha queue mein aage -- FCFS ko
        # "chhupa hua" priority advantage. Ab seeded shuffle: same seed = same order.
        if self.shuffle_arrivals and len(arrivals) > 1:
            order = self.order_rng.permutation(len(arrivals))
            arrivals = [arrivals[i] for i in order]
        return arrivals
