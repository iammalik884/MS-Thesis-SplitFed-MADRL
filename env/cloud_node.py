"""
Step 3 - CloudNode: the one shared, powerful, but "far away" compute resource.
================================================================================
ROMAN URDU: Ab tak har department ka apna "Edge" computer tha (Step 1/2).
Step 3 mein hum do naye "tiers" add kar rahe hain: "Local" (bilkul chhota,
device ke andar hi, jaise ek bedside monitor) aur "Cloud" (bahut powerful,
lekin poore hospital ke liye SIRF EK shared server, aur wahan tak data
bhejne mein waqt lagta hai -- "network_delay"). Local device banane ke liye
hum EdgeNode class hi dobara istemal kar lete hain (chhoti capacity, delay
zero). Lekin Cloud ke liye ek nayi class chahiye kyunke usme EXTRA behaviour
hai: task pohanchne ke baad turant kaam shuru nahi hota, pehle
`network_delay` steps "safar" (transit) mein guzarte hain.

ENGLISH: Until now every department had its own "Edge" computer (Step 1/2).
Step 3 adds two new tiers: "Local" (tiny, on-device, e.g. a bedside monitor)
and "Cloud" (very powerful, but only ONE shared server for the whole
hospital, and reaching it costs time -- "network_delay"). The Local tier
reuses the existing EdgeNode class as-is (small capacity, zero delay). The
Cloud tier needs its own class because of one EXTRA rule: a task does not
start being processed the instant it arrives -- it first spends
`network_delay` steps "in transit" over the network.

CloudNode is an EdgeNode with one added rule inserted at the front of the
per-step loop: a task is only eligible for compute once
    current_time - task.arrival_time >= network_delay
Before that, it just waits (same as waiting for capacity in Step 1) -- so
all of Step 1's proven queueing/capacity mechanics still apply underneath,
this class only adds the network-delay gate on top.
"""

from env.edge_node import EdgeNode


class CloudNode(EdgeNode):
    """The hospital-wide shared cloud server: big capacity, but every task
    must wait `network_delay` steps after arriving before it can be worked
    on (simulates the time to transmit the task over the network)."""

    def __init__(self, name: str, capacity: float, network_delay: int):
        super().__init__(name, capacity)
        if network_delay < 0:
            raise ValueError(
                f"CloudNode '{name}': network_delay must be >= 0, got {network_delay}"
            )
        self.network_delay = network_delay

    def step(self, current_time: int) -> None:
        """Same FIFO + capacity loop as EdgeNode.step(), except a task is
        skipped (kept waiting) until it has spent `network_delay` steps in
        the queue, simulating network transmission time to the cloud."""
        available_capacity = self.capacity
        still_in_queue = []

        for task in self.queue:
            still_in_transit = (current_time - task.arrival_time) < self.network_delay
            if still_in_transit or available_capacity <= 0:
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
