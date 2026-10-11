"""
Step 1 - EdgeNode: a single hospital department's limited-capacity computer.
================================================================================
ROMAN URDU: Har hospital department (Emergency, ICU, General Ward) ke paas
ek chhota "edge computer" hota hai. Uski processing power LIMITED hoti hai
-- ek time-step mein woh sirf utna kaam kar sakta hai jitni "capacity" hai.
Zyada kaam ho to baaqi tasks agle step tak wait karte hain. Koi AI nahi --
node sirf FCFS (pehle aao, pehle pao) order mein kaam karta hai.

V2 (AUDIT FIX) -- "DESTINATION-READY FCFS":
  Ab ek node par teen qisam ke tasks aa sakte hain: apne department ke
  (Local, delay 0), padosi department se offload hue (Edge, neighbour delay),
  aur Cloud par (cloud delay). Har task ka apna `ready_time` hota hai --
  network se pohanchne ka waqt. Rule:
    1. Jo task abhi network mein hai (ready_time > current_time) woh CPU
       istemal NAHI karta aur ready tasks ko block bhi NAHI karta.
    2. Ready tasks ko `ready_time` ke order mein serve kiya jata hai;
       barabar ready_time par jo pehle enqueue hua woh pehle (stable order).
  Queue list hamesha isi service order mein rakhi jati hai (insert ke waqt).

ENGLISH: Each department owns one edge node with a fixed compute capacity
per step. V2 rule ("destination-ready FCFS"): a task still in network
transit (ready_time > current_time) neither consumes capacity nor blocks
ready tasks; ready tasks are served in ready_time order, ties broken by
enqueue order (stable insert). Once a task has started it is never
pre-empted: every later insertion has ready_time >= the current step,
which is strictly greater than a started task's ready_time.

Formula (per time-step t):
    available = capacity
    for task in queue (service order):
        if task.ready_time > t: skip (in transit, keeps its place)
        work = min(available, task.remaining_demand)
        task.remaining_demand -= work; available -= work
        if task.remaining_demand == 0: completion_time = t
"""

import bisect
from typing import List, Optional

from env.task import Task, is_finite_number, is_int

# Floating-point guard: capacity left over smaller than this is treated as
# zero, so a task never "starts" on a 1e-16 crumb of capacity.
CAPACITY_EPSILON = 1e-9


class EdgeNode:
    """A hospital department's edge computer with limited per-step capacity."""

    def __init__(self, name: str, capacity: float):
        if not is_finite_number(capacity) or capacity <= 0:
            raise ValueError(f"EdgeNode '{name}': capacity must be a finite number > 0, got {capacity!r}")
        self.name = name
        self.capacity = float(capacity)
        self.queue: List[Task] = []            # unfinished tasks, in FCFS service order
        self.completed_tasks: List[Task] = []  # finished tasks, kept for verification/testing
        # V2: the node's own clock = index of the NEXT step it will run.
        self.clock = 0
        self.last_step_work_done = 0.0         # compute actually spent in the last executed step

    def add_task(self, task: Task, transfer_delay: int = 0) -> None:
        """Enqueue a task. ready_time = arrival_time + transfer_delay
        (0 for the department's own tasks, neighbour/cloud delay otherwise).
        Inserted after every task with ready_time <= its own (stable FCFS)."""
        if not is_int(transfer_delay) or transfer_delay < 0:
            raise ValueError(
                f"EdgeNode '{self.name}': transfer_delay must be an int >= 0, got {transfer_delay!r}"
            )
        task.ready_time = task.arrival_time + transfer_delay
        position = bisect.bisect_right(self.queue, task.ready_time, key=lambda t: t.ready_time)
        self.queue.insert(position, task)

    def step(self, current_time: int) -> None:
        """Advance this node by one time-step (see module docstring)."""
        if current_time < self.clock:
            raise ValueError(
                f"EdgeNode '{self.name}': step({current_time}) would move time backwards "
                f"(node already ran up to step {self.clock - 1})"
            )
        available = self.capacity
        still_in_queue = []

        for task in self.queue:
            # Transit wale task ko skip karo (jagah wahi rahegi), ya capacity khatam.
            if task.ready_time > current_time or available <= CAPACITY_EPSILON:
                still_in_queue.append(task)
                continue

            work_done = min(available, task.remaining_demand)
            if task.start_time is None:
                task.start_time = current_time  # pehli baar CPU mila
            task.remaining_demand -= work_done
            available -= work_done

            if task.is_complete:
                task.completion_time = current_time
                self.completed_tasks.append(task)
            else:
                still_in_queue.append(task)

        self.queue = still_in_queue
        self.last_step_work_done = self.capacity - available
        self.clock = current_time + 1

    # ------------------------------------------------------------------
    # Read-only views. `current_time` defaults to the node's clock (the step
    # about to run), which always equals HospitalSimulator.current_time.
    # ------------------------------------------------------------------
    def _now(self, current_time: Optional[int]) -> int:
        return self.clock if current_time is None else current_time

    @property
    def queue_length(self) -> int:
        """All unfinished tasks assigned here, INCLUDING those still in transit."""
        return len(self.queue)

    @property
    def backlog_demand(self) -> float:
        """Total remaining demand assigned to this node, including in-transit
        reservations. This is what the FCFS router compares (v2)."""
        return sum(t.remaining_demand for t in self.queue)

    def ready_demand(self, current_time: Optional[int] = None) -> float:
        """Remaining demand of tasks that are processing-eligible at that step."""
        now = self._now(current_time)
        return sum(t.remaining_demand for t in self.queue if t.ready_time <= now)

    def transit_demand(self, current_time: Optional[int] = None) -> float:
        """Remaining demand of tasks still in network transit to this node."""
        now = self._now(current_time)
        return sum(t.remaining_demand for t in self.queue if t.ready_time > now)

    def in_transit_count(self, current_time: Optional[int] = None) -> int:
        now = self._now(current_time)
        return sum(1 for t in self.queue if t.ready_time > now)

    def projected_utilization(self, current_time: Optional[int] = None) -> float:
        """QUEUE-DEMAND ESTIMATE: fraction of capacity step(current_time) would
        spend, given the current queue. Only READY tasks count -- a task still
        in transit uses no CPU (v2 fix: CloudNode used to report 0.4 here for a
        task that could not be processed yet)."""
        return min(self.ready_demand(current_time), self.capacity) / self.capacity

    @property
    def utilization(self) -> float:
        """Backward-compatible name (Step 1) for projected_utilization() at the
        node's clock, i.e. the queue-demand estimate for the next step."""
        return self.projected_utilization()

    @property
    def last_step_utilization(self) -> float:
        """ACTUAL utilization: compute spent in the most recently executed step
        / capacity (0.0 before any step has run)."""
        return self.last_step_work_done / self.capacity
