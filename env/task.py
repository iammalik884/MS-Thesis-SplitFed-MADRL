"""
Step 1/2/4 - Task representation (priority since Step 2, deadline since Step 4).
================================================================================
ROMAN URDU: Yeh sabse chota building-block hai. Ek "Task" matlab ek medical
computation job -- jaise ek ECG reading process karna. Step 1 mein hum sirf
itna jaante thay: task kab aaya (arrival_time) aur usay process karne ke
liye kitna "compute demand" chahiye. STEP 2 update: ab har task ka apna
"priority" (Critical / High / Routine) bhi hota hai, kyunke Poisson
generator (env/task_generator.py) teeno priority classes ko alag-alag
arrival-rate aur compute-demand se generate karta hai -- is liye task ko
yeh label yaad rakhna zaroori hai. STEP 4 update: ab task apni "deadline"
bhi yaad rakhta hai -- kitne time-step tak usay complete ho jana chahiye
(priority ke hisaab se, configs/environment.yaml ke 'deadlines' section se
tay hoti hai -- Critical sab se tight, Routine sab se loose). Yeh deadline
sirf TRACK hoti hai (`met_deadline` property se) -- abhi koi scheduling
decision isay istemal nahi karti, queue order abhi bhi Step 1 wala FIFO hai.

ENGLISH: This is the smallest building block of the simulation. A "Task"
represents one medical computation job (e.g. processing an ECG reading).
Step 1 tracked only when it arrived and how much compute it needs. STEP 2
adds a `priority` label (Critical / High / Runtime -> Routine) because the
Poisson generator now creates tasks with different arrival rates and
compute-demand ranges per priority class, so each task must remember which
class it belongs to. STEP 4 update: a task now also remembers its
"deadline" -- how many time-steps it should finish within, set from
configs/environment.yaml's 'deadlines' section (per priority, Critical
tightest, Routine loosest). This is tracked only (`met_deadline` property)
-- no scheduling decision uses it yet; queue order is still Step 1's FIFO.
"""

from dataclasses import dataclass, field
from typing import Optional

# The three approved priority classes (synopsis-locked). Kept as a module-level
# constant so every file that needs to validate/iterate priorities imports the
# same list instead of retyping the strings (avoids silent typos like "critical").
PRIORITY_CLASSES = ("Critical", "High", "Routine")

# STEP 3: the three places a task can be executed. Same reasoning as
# PRIORITY_CLASSES above -- one shared constant so "Local"/"Edge"/"Cloud"
# is never retyped (and possibly mistyped) in more than one file.
TIER_CLASSES = ("Local", "Edge", "Cloud")


@dataclass
class Task:
    """A single unit of work arriving at a hospital department's edge node.

    Attributes:
        task_id: unique identifier for this task (for logging/debugging).
        arrival_time: the simulation time-step at which the task arrived.
        compute_demand: total amount of "compute capacity" needed to finish
            this task. If a node cannot finish it in one step, the leftover
            work carries over to the next step (see EdgeNode.step()).
        remaining_demand: compute still required to finish the task. Starts
            equal to compute_demand and decreases as the node works on it.
        completion_time: the time-step at which the task finished, or None
            if it has not finished yet.
        priority: one of PRIORITY_CLASSES ("Critical"/"High"/"Routine"), or
            None for tasks created the Step-1 way (manual add_task without a
            priority) -- kept optional so Step 1's existing tests and API do
            not break (project rule: preserve working code).
        tier: one of TIER_CLASSES ("Local"/"Edge"/"Cloud"), set by
            HospitalSimulator.route_task() (Step 3) to record WHERE this task
            ended up executing. None for tasks created the Step-1/2 way
            (add_task/add_generated_arrivals always use the Edge tier but do
            not bother labeling it) -- again optional so nothing before
            Step 3 breaks.
        deadline: the time-step by which this task is expected to finish
            (arrival_time + a priority-based offset from
            configs/environment.yaml's 'deadlines' section), set by
            HospitalSimulator (Step 4). None for tasks with no priority, or
            when the config has no 'deadlines' section -- deadline tracking
            is opt-in, same pattern as tier/offloading in Step 3.
    """

    task_id: int
    arrival_time: int
    compute_demand: float
    remaining_demand: float = field(init=False)
    completion_time: Optional[int] = None
    priority: Optional[str] = None
    tier: Optional[str] = None
    deadline: Optional[int] = None

    def __post_init__(self):
        # Input validation with meaningful error messages (project coding rule).
        if self.compute_demand <= 0:
            raise ValueError(
                f"Task {self.task_id}: compute_demand must be > 0, got {self.compute_demand}"
            )
        if self.arrival_time < 0:
            raise ValueError(
                f"Task {self.task_id}: arrival_time cannot be negative, got {self.arrival_time}"
            )
        if self.priority is not None and self.priority not in PRIORITY_CLASSES:
            raise ValueError(
                f"Task {self.task_id}: priority must be one of {PRIORITY_CLASSES} or None, "
                f"got {self.priority!r}"
            )
        if self.tier is not None and self.tier not in TIER_CLASSES:
            raise ValueError(
                f"Task {self.task_id}: tier must be one of {TIER_CLASSES} or None, "
                f"got {self.tier!r}"
            )
        if self.deadline is not None and self.deadline < self.arrival_time:
            raise ValueError(
                f"Task {self.task_id}: deadline ({self.deadline}) cannot be before "
                f"arrival_time ({self.arrival_time})"
            )
        self.remaining_demand = self.compute_demand

    @property
    def is_complete(self) -> bool:
        """True once the node has done enough work on this task."""
        return self.remaining_demand <= 0

    @property
    def met_deadline(self) -> Optional[bool]:
        """STEP 4 - whether this task finished on or before its deadline.

        Returns None (not yet decidable) if the task has no deadline, or has
        not completed yet. Once completed, True if completion_time <=
        deadline, False otherwise. Deliberately read-only/derived -- nothing
        else sets it, so it can never drift out of sync with completion_time.
        """
        if self.deadline is None or self.completion_time is None:
            return None
        return self.completion_time <= self.deadline
