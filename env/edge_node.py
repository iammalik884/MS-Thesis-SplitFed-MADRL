"""
Step 1 - EdgeNode: a single hospital department's limited-capacity computer.
================================================================================
ROMAN URDU: Har hospital department (Emergency, ICU, General Ward) ke paas
ek chhota "edge computer" hota hai. Uski processing power LIMITED hoti hai
-- yani ek time-step mein woh sirf itna kaam kar sakta hai jitni uski
"capacity" ijazat deti hai. Agar queue mein zyada tasks hon, baaqi tasks
wait karte hain ya agla step carry-over hote hain. Abhi koi "intelligence"
(AI) nahi hai -- node sirf queue ke front wale task par pehle kaam karta
hai (FIFO order), jo sirf yeh test karne ke liye hai ke queue aur capacity
mechanics sahi kaam kar rahe hain.

ENGLISH: Each hospital department has one edge node with a fixed compute
capacity per time-step. If the queue holds more work than the node can do
in one step, the remaining work waits or carries into the next step. There
is no learning/AI here yet -- the node simply works through its queue in
arrival order (FIFO), purely to verify the queueing + capacity mechanics
are correct before any scheduling intelligence is added (later steps).

Formula (per time-step):
    available_capacity = capacity
    for each task in queue (front to back):
        work_done = min(available_capacity, task.remaining_demand)
        task.remaining_demand -= work_done
        available_capacity -= work_done
        if task.remaining_demand == 0: task is complete, remove from queue
        if available_capacity == 0: stop (no more capacity this step)
"""

from typing import List
from env.task import Task


class EdgeNode:
    """A hospital department's edge computer with limited per-step capacity."""

    def __init__(self, name: str, capacity: float):
        if capacity <= 0:
            raise ValueError(f"EdgeNode '{name}': capacity must be > 0, got {capacity}")
        self.name = name
        self.capacity = capacity
        self.queue: List[Task] = []          # tasks waiting / in-progress, FIFO order
        self.completed_tasks: List[Task] = []  # finished tasks, kept for verification/testing

    def add_task(self, task: Task) -> None:
        """Enqueue a new task at this department's edge node."""
        self.queue.append(task)

    def step(self, current_time: int) -> None:
        """Advance this node's simulation by one time-step.

        Works through the queue in FIFO order, spending up to `self.capacity`
        units of compute. Tasks that finish are moved to completed_tasks;
        unfinished tasks stay at the front of the queue for the next step.
        """
        available_capacity = self.capacity
        still_in_queue = []

        for task in self.queue:
            if available_capacity <= 0:
                # No compute left this step -- remaining tasks simply wait.
                still_in_queue.append(task)
                continue

            work_done = min(available_capacity, task.remaining_demand)
            task.remaining_demand -= work_done
            available_capacity -= work_done

            if task.is_complete:
                task.completion_time = current_time
                self.completed_tasks.append(task)
            else:
                still_in_queue.append(task)

        self.queue = still_in_queue

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    @property
    def utilization(self) -> float:
        """Fraction of this step's capacity that step() would actually spend,
        given the current queue (1.0 = fully busy this step, 0.0 = idle).
        Useful for logging/plots and, later, as an RL observation feature.

        Mirrors step()'s FIFO capacity-consumption loop (without mutating any
        task) so it accounts for EVERY task capacity reaches this step, not
        just the one at the front of the queue -- a node with three small
        tasks that together fill its capacity is 100% utilized, even though
        no single task uses the full capacity by itself.
        """
        if not self.queue:
            return 0.0
        available_capacity = self.capacity
        for task in self.queue:
            if available_capacity <= 0:
                break
            available_capacity -= min(available_capacity, task.remaining_demand)
        return (self.capacity - available_capacity) / self.capacity
