"""
Step 3 - CloudNode: the one shared, powerful, but "far away" compute resource.
================================================================================
ROMAN URDU: Poore hospital ke liye SIRF EK shared Cloud server hai -- bahut
powerful, lekin wahan tak data bhejne mein waqt lagta hai ("network_delay").
Task Cloud par pohanchne (ready_time) se pehle process nahi ho sakta.

V2 (AUDIT FIX): pehle Cloud ka apna alag step() tha jo "current_time -
arrival_time >= network_delay" check karta tha, jabke `utilization` transit
wale tasks ko bhi count kar leta tha (capacity 50, demand 20, delay 2 ->
time 0 par 0.4 dikhata tha jabke CPU bilkul khaali tha). Ab network readiness
har task ke `ready_time` mein hai aur EdgeNode hi sab rules apply karta hai
-- Cloud aur padosi (neighbour) transfer ke ready-time rules ek jaise hain.

ENGLISH: One shared cloud server for the whole hospital. V2: network
readiness is stored per task (ready_time = arrival_time + network_delay) and
enforced by EdgeNode's destination-ready FCFS loop, so Cloud and neighbour
transfers follow ONE consistent rule and utilization never counts transit
work. CloudNode only adds a default transfer delay.
"""

from env.edge_node import EdgeNode
from env.task import Task, is_int


class CloudNode(EdgeNode):
    """Hospital-wide shared cloud server: big capacity, every task arrives
    `network_delay` steps after it was sent."""

    def __init__(self, name: str, capacity: float, network_delay: int):
        super().__init__(name, capacity)
        if not is_int(network_delay) or network_delay < 0:
            raise ValueError(
                f"CloudNode '{name}': network_delay must be an int >= 0, got {network_delay!r}"
            )
        self.network_delay = network_delay

    def add_task(self, task: Task, transfer_delay: int = None) -> None:
        """Enqueue with the cloud link delay unless a delay is given explicitly."""
        super().add_task(task, self.network_delay if transfer_delay is None else transfer_delay)
