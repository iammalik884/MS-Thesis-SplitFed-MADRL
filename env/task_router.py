"""
Step 3 / Step 5 - TaskRouter: the fixed FCFS-baseline routing rule
(Local / Edge(neighbour) / Cloud).
================================================================================
ROMAN URDU: Yeh class faisla karti hai ke naya task kahan process hoga.
**YEH AI NAHI HAI** -- ek fixed, priority-blind (FCFS) rule hai:

    1. compute_demand > cloud_threshold                     -> Cloud
    2. warna, source node ka backlog <= offload_threshold    -> Local
       (apne department ka edge node)
    3. warna                                                 -> Edge
       (padosi department jis ka backlog sab se kam ho; tie ho to fixed
        order Emergency -> ICU -> General_Ward)

"Backlog" = node ko assign kiye gaye tamam unfinished tasks ki remaining
demand, transit wale (raste mein) tasks samet -- warna ek hi step mein do
tasks ek hi "khaali" padosi par chale jate. Priority ko yeh rule bilkul
nahi dekhta (FCFS baseline). Steps 7+ mein yahin RL agent aayega.

ENGLISH: Fixed, non-learned, priority-blind routing rule for the FCFS
baseline (see the 3 lines above). "Backlog" = total remaining demand
assigned to a node, including in-transit reservations. The tie-break order
is an intentional deterministic preference, NOT a claim of fairness.

V2 (AUDIT FIX): the old rule (demand <= local_threshold -> a tiny bedside
"Local" node; <= edge_threshold -> the department's own node; else Cloud)
did not match the thesis action space (Local = own department, Edge =
neighbour department). The old `choose_tier(compute_demand)` is replaced by
`choose_destination(source, compute_demand, backlogs)`.
"""

from typing import Dict, Mapping, Sequence, Tuple

from env.task import CLOUD_NODE_NAME, is_finite_number


class TaskRouter:
    """FCFS-baseline routing rule based on compute_demand and node backlogs."""

    def __init__(self, cloud_threshold: float, offload_threshold: float, neighbour_order: Sequence[str]):
        if not is_finite_number(cloud_threshold) or cloud_threshold <= 0:
            raise ValueError(f"TaskRouter: cloud_threshold must be a finite number > 0, got {cloud_threshold!r}")
        if not is_finite_number(offload_threshold) or offload_threshold < 0:
            raise ValueError(
                f"TaskRouter: offload_threshold must be a finite number >= 0, got {offload_threshold!r}"
            )
        neighbour_order = list(neighbour_order)
        if len(neighbour_order) < 2 or len(set(neighbour_order)) != len(neighbour_order):
            raise ValueError(
                f"TaskRouter: neighbour_order must list >= 2 distinct departments, got {neighbour_order!r}"
            )
        self.cloud_threshold = float(cloud_threshold)
        self.offload_threshold = float(offload_threshold)
        self.neighbour_order = neighbour_order
        self._rank: Dict[str, int] = {name: i for i, name in enumerate(neighbour_order)}

    def choose_destination(
        self, source: str, compute_demand: float, backlogs: Mapping[str, float]
    ) -> Tuple[str, str]:
        """Return (tier, destination_node_name) for a new task.

        Args:
            source: the department that created the task.
            compute_demand: the task's total demand (finite, > 0).
            backlogs: {department: backlog_demand} for EVERY department, measured
                just before this task is placed.
        """
        if source not in self._rank:
            raise ValueError(f"TaskRouter: unknown source department {source!r}")
        if not is_finite_number(compute_demand) or compute_demand <= 0:
            raise ValueError(
                f"TaskRouter.choose_destination: compute_demand must be a finite number > 0, "
                f"got {compute_demand!r}"
            )
        if compute_demand > self.cloud_threshold:
            return "Cloud", CLOUD_NODE_NAME
        if backlogs[source] <= self.offload_threshold:
            return "Local", source
        return "Edge", self.least_loaded_neighbour(source, backlogs)

    def least_loaded_neighbour(self, source: str, backlogs: Mapping[str, float]) -> str:
        """The neighbour an Edge offload goes to: smallest backlog, ties broken
        by neighbour_order. Shared by the FCFS rule and the Step 6 RL action
        "Edge", so both pick the neighbour in exactly the same way."""
        if source not in self._rank:
            raise ValueError(f"TaskRouter: unknown source department {source!r}")
        # Source khud candidate nahi; (backlog, fixed rank) se sab se halka padosi.
        neighbours = [d for d in self.neighbour_order if d != source]
        return min(neighbours, key=lambda d: (backlogs[d], self._rank[d]))
