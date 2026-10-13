"""
Step 7 (proposed baseline) - Estimated-completion-time (ECT) heuristic.
================================================================================
ROMAN URDU: Ek seedhi, adaptive policy: har action (Local / Edge / Cloud) ke
liye andaza lagao ke task kitne steps mein khatam hoga, aur sab se jaldi
wala action chuno. Sirf woh maloomat istemal hoti hai jo DQN ki observation
mein bhi hai (har node ka backlog, capacities, network delays, task ki
demand) -- na future arrivals, na queue ke andar har task ki detail. Is
tarah DQN aur yeh heuristic barabar information par compare hote hain.

Andaza (latency ki definition jaisa, 0 = arrival wale step mein khatam):
    est(a) = delay(a) + ceil((backlog(a) + demand) / capacity(a)) - 1
  delay: Local 0, Edge = neighbour_network_delay, Cloud = cloud_network_delay
  backlog: us node ka poora baqi kaam (raste wale tasks samet)
  Edge ka node: TaskRouter.least_loaded_neighbour() -- wahi jo env ka Edge
  action istemal karta hai.
  Barabar andaze par order: Local, Edge, Cloud (kam transfer pehle).

Yeh andaza jaan-bujh kar SIMPLE hai: maan leta hai ke node ka saara backlog
hamare task se pehle aur poori capacity se process hoga, aur aage aane wale
tasks ka asar nahi lagata (future information istemal nahi hoti). Is liye
yeh exact nahi, sirf decision-time estimate hai. Is ka koi tunable parameter
nahi.

ENGLISH: picks the action with the smallest estimated latency
delay + ceil((backlog + demand) / capacity) - 1, using only quantities that
are in the Step 6 observation (unscaled) plus fixed config constants.
"""

import math

from env.gym_env import ACTIONS, HospitalSchedulingEnv

# Float tolerance so that e.g. (3.0 + 2.0) / 5.0 is treated as exactly 1 step.
_EPS = 1e-9


def estimated_latencies(env: HospitalSchedulingEnv) -> dict:
    """{action: estimated latency in steps} for the env's pending task."""
    sim = env.sim
    if not env._pending:
        raise RuntimeError("estimated_latencies: no pending task")
    arrival = env._pending[0]
    src, demand = arrival.department, arrival.compute_demand
    backlogs = {name: node.backlog_demand for name, node in sim.nodes.items()}
    neighbour = sim.router.least_loaded_neighbour(src, backlogs)
    candidates = {
        "Local": (0, backlogs[src], sim.nodes[src].capacity),
        "Edge": (sim.neighbour_delay, backlogs[neighbour], sim.nodes[neighbour].capacity),
        "Cloud": (sim.cloud_node.network_delay, sim.cloud_node.backlog_demand, sim.cloud_node.capacity),
    }
    return {
        ACTIONS.index(tier): delay + math.ceil((backlog + demand) / capacity - _EPS) - 1
        for tier, (delay, backlog, capacity) in candidates.items()
    }


def ect_action(env: HospitalSchedulingEnv) -> int:
    """Action with the smallest estimated latency; ties -> Local, Edge, Cloud."""
    estimates = estimated_latencies(env)
    return min(estimates, key=lambda a: (estimates[a], a))
