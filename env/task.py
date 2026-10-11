"""
Step 1/2/4 - Task representation (priority since Step 2, deadline since Step 4,
full lifecycle timing since the v2 audit fixes).
================================================================================
ROMAN URDU: Yeh sabse chota building-block hai. Ek "Task" matlab ek medical
computation job -- jaise ek ECG reading process karna. Step 1 mein hum sirf
itna jaante thay: task kab aaya (arrival_time) aur usay process karne ke
liye kitna "compute demand" chahiye. STEP 2 update: ab har task ka apna
"priority" (Critical / High / Routine) bhi hota hai. STEP 4 update: task
apni "deadline" bhi yaad rakhta hai (sirf TRACK hoti hai, scheduling isay
istemal nahi karti).

V2 (AUDIT FIX) update: ab har task apni poori "zindagi" (lifecycle) record
karta hai -- kis department se aaya (source_department), kahan process hua
(destination + tier), network se kab pohancha (ready_time), pehli baar CPU
kab mila (start_time), aur kab khatam hua (completion_time). Is se latency
ko teen hisson mein toda ja sakta hai:
    latency = transmission_delay + queue_wait + execution_span

ENGLISH: The smallest building block. STEP 2 added `priority`, STEP 4 added
a tracked-only `deadline`. V2 (audit fix): every task now records its full
lifecycle -- source department, destination node + tier, transmission-ready
time, first processing (start) time and completion time -- so latency can
be decomposed exactly (see the properties at the bottom of the class).

TIME CONVENTION (discrete steps, unchanged from Step 1):
  - A task arriving at step t is created at the START of step t and can be
    processed in that same step t (if it is ready and capacity is left).
  - completion_time = the index of the step in which the last unit of work
    was done. A task that arrives and finishes in step t has latency 0
    ("finished within its arrival step"), NOT latency 1.
  - ready_time = arrival_time + transfer delay (0 for Local). The task can
    first be processed in step ready_time.
"""

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Optional

# The three approved priority classes (synopsis-locked). Kept as a module-level
# constant so every file that needs to validate/iterate priorities imports the
# same list instead of retyping the strings (avoids silent typos like "critical").
PRIORITY_CLASSES = ("Critical", "High", "Routine")

# The three scheduling actions (roadmap/synopsis): Local = source department's
# OWN edge node, Edge = a NEIGHBOUR department's edge node, Cloud = the one
# shared remote server. Same reasoning as PRIORITY_CLASSES: one shared constant.
TIER_CLASSES = ("Local", "Edge", "Cloud")

# Name used for the shared cloud node (destination field of Cloud tasks).
CLOUD_NODE_NAME = "Cloud"


def is_int(value) -> bool:
    """True for real integers only (bool is excluded on purpose -- YAML
    'true' must not silently become a step count of 1)."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def is_finite_number(value) -> bool:
    """True for finite int/float values (rejects NaN, +/-inf, bool, str)."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass
class Task:
    """A single unit of work created in a hospital department.

    Attributes:
        task_id: unique identifier (assigned in enqueue order by the simulator).
        arrival_time: step at which the task was created at its source.
        compute_demand: total compute units needed (finite, > 0).
        remaining_demand: compute still required; starts = compute_demand.
        completion_time: step in which the task finished, or None.
        priority: one of PRIORITY_CLASSES, or None (Step-1 style tasks).
        tier: "Local" / "Edge" / "Cloud" -- WHERE the task executes relative
            to its source (None only for hand-built tasks never placed by
            the simulator).
        deadline: arrival_time + priority offset (Step 4), or None.
        source_department: department that created the task.
        destination: name of the node that executes it (a department name,
            or "Cloud").
        ready_time: first step at which the destination may process it
            (arrival_time + transfer delay). Defaults to arrival_time.
        start_time: step in which the task first received any compute.
    """

    task_id: int
    arrival_time: int
    compute_demand: float
    remaining_demand: float = field(init=False)
    completion_time: Optional[int] = None
    priority: Optional[str] = None
    tier: Optional[str] = None
    deadline: Optional[int] = None
    source_department: Optional[str] = None
    destination: Optional[str] = None
    ready_time: Optional[int] = None
    start_time: Optional[int] = None

    def __post_init__(self):
        # Input validation with meaningful error messages (project coding rule).
        # V2: NaN/inf/bool/non-integer times are rejected too -- `nan <= 0` is
        # False, so the old check silently accepted NaN demand.
        if not is_int(self.task_id):
            raise ValueError(f"Task: task_id must be an int, got {self.task_id!r}")
        if not is_finite_number(self.compute_demand) or self.compute_demand <= 0:
            raise ValueError(
                f"Task {self.task_id}: compute_demand must be a finite number > 0, "
                f"got {self.compute_demand!r}"
            )
        if not is_int(self.arrival_time) or self.arrival_time < 0:
            raise ValueError(
                f"Task {self.task_id}: arrival_time must be an int >= 0, got {self.arrival_time!r}"
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
        if self.deadline is not None:
            if not is_int(self.deadline):
                raise ValueError(f"Task {self.task_id}: deadline must be an int, got {self.deadline!r}")
            if self.deadline < self.arrival_time:
                raise ValueError(
                    f"Task {self.task_id}: deadline ({self.deadline}) cannot be before "
                    f"arrival_time ({self.arrival_time})"
                )
        if self.ready_time is None:
            self.ready_time = self.arrival_time  # no transfer -> ready immediately
        elif not is_int(self.ready_time) or self.ready_time < self.arrival_time:
            raise ValueError(
                f"Task {self.task_id}: ready_time must be an int >= arrival_time "
                f"({self.arrival_time}), got {self.ready_time!r}"
            )
        self.compute_demand = float(self.compute_demand)
        self.remaining_demand = self.compute_demand

    @property
    def is_complete(self) -> bool:
        """True once the node has done enough work on this task."""
        return self.remaining_demand <= 0

    @property
    def met_deadline(self) -> Optional[bool]:
        """STEP 4 - whether this task finished on or before its deadline.

        None if there is no deadline or the task has not completed yet;
        otherwise completion_time <= deadline (exact boundary = on time).
        NOTE (v2): None does NOT mean "fine" -- an unfinished task can already
        be overdue. Use env.metrics.deadline_status(task, cutoff) for the full
        4-way classification used by the evaluation metrics.
        """
        if self.deadline is None or self.completion_time is None:
            return None
        return self.completion_time <= self.deadline

    # ------------------------------------------------------------------
    # V2 lifecycle timing (all in steps; None until the event happened).
    # ROMAN URDU: latency = transmission_delay + queue_wait + execution_span
    # -- yeh teeno hisse aapas mein overlap NAHI karte (transit ka waqt
    # queue wait mein dobara count nahi hota).
    # ------------------------------------------------------------------
    @property
    def transmission_delay(self) -> int:
        """Steps spent in network transit before the destination may process it."""
        return self.ready_time - self.arrival_time

    @property
    def queue_wait(self) -> Optional[int]:
        """Steps the task was READY at its destination but received no compute
        (start_time - ready_time). Transit time is NOT included."""
        return None if self.start_time is None else self.start_time - self.ready_time

    @property
    def execution_span(self) -> Optional[int]:
        """completion_time - start_time: 0 means it finished in the same step
        it started; k means it needed k further steps (carry-over)."""
        if self.start_time is None or self.completion_time is None:
            return None
        return self.completion_time - self.start_time

    @property
    def latency(self) -> Optional[int]:
        """End-to-end latency = completion_time - arrival_time (0 = finished in
        its arrival step)."""
        return None if self.completion_time is None else self.completion_time - self.arrival_time
