"""
Step 1/2 - HospitalSimulator: ties the 3 department EdgeNodes into one simulator.
================================================================================
ROMAN URDU: Yeh class poore "mini hospital" ko represent karti hai. Isme
teen departments (Emergency, ICU, General Ward) hain, har ek apna EdgeNode
rakhta hai. Simulator ek simple "clock" chalata hai: har time-step par woh
har department ke edge node ko ek step aage badhata hai. Step 1 mein tasks
sirf manually `add_task()` se dalay jate thay. STEP 2 update: ab
`add_generated_arrivals()` bhi mojood hai, jo MedicalTaskGenerator
(env/task_generator.py) ke banaye hue Poisson-arrivals ko simulator mein
daal deta hai -- lekin `add_task()` ka purana behaviour (Step 1 tests ke
liye) bilkul waisa hi rakha gaya hai, sirf ek naya optional `priority`
parameter add hua hai.

ENGLISH: This class represents the whole "mini hospital". It holds three
departments (Emergency, ICU, General Ward), each with its own EdgeNode.
The simulator runs a simple clock: each time-step, every department's edge
node advances by one step. Step 1 only supported manual task injection via
`add_task()`. STEP 2 adds `add_generated_arrivals()`, a thin wrapper that
takes the output of MedicalTaskGenerator (Poisson arrivals per department
and priority class) and injects it the same way `add_task()` always did --
so Step 1's tests and behaviour are completely unchanged.

Must Remember:
- Priority labels exist on tasks now (Step 2), but deadlines and the
  "Critical must be faster" policy are still Step 4.

STEP 3 update: add_task()/add_generated_arrivals() are UNCHANGED on purpose
(Step 1/2 tests still call them and still expect every task to land on the
department's Edge node) -- Local/Edge/Cloud ROUTING is opt-in through two
new methods, route_task() and route_generated_arrivals(), which use a
TaskRouter (env/task_router.py, a fixed rule, not AI yet) to decide the
tier, then dispatch to whichever node actually represents that tier:
  - "Local" -> self.local_nodes[department]  (small, per-department, EdgeNode)
  - "Edge"  -> self.nodes[department]        (same node Step 1/2 always used)
  - "Cloud" -> self.cloud_node                (ONE node shared by the whole
                                                hospital, has network_delay)
"""

from typing import Dict, List, Optional
import yaml

from env.cloud_node import CloudNode
from env.edge_node import EdgeNode
from env.task import Task
from env.task_router import TaskRouter


class HospitalSimulator:
    """Multi-department hospital simulator. No learning/scheduling AI yet --
    Step 3 adds a fixed, non-learned rule for choosing Local/Edge/Cloud."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if "departments" not in config or not config["departments"]:
            raise ValueError(f"Config '{config_path}' must define a non-empty 'departments' list")

        self.seed = config.get("seed", 42)
        self.n_steps = config["simulation"]["n_steps"]

        # Build one EdgeNode per configured department (Step 1, unchanged).
        self.nodes: Dict[str, EdgeNode] = {}
        for dept in config["departments"]:
            self.nodes[dept["name"]] = EdgeNode(name=dept["name"], capacity=dept["capacity"])

        # STEP 3: Local tier (one tiny EdgeNode per department) + Cloud tier
        # (one CloudNode shared by the whole hospital) + the routing rule
        # that decides which tier a task goes to. All optional -- if the
        # config has no 'offloading' section, routing simply is not
        # available and route_task()/route_generated_arrivals() explain why.
        offloading_config = config.get("offloading")
        if offloading_config:
            self.local_nodes: Dict[str, EdgeNode] = {
                dept_name: EdgeNode(name=f"{dept_name}_Local", capacity=offloading_config["local_capacity"])
                for dept_name in self.nodes
            }
            self.cloud_node: Optional[CloudNode] = CloudNode(
                name="Cloud",
                capacity=offloading_config["cloud_capacity"],
                network_delay=offloading_config["cloud_network_delay"],
            )
            self.router: Optional[TaskRouter] = TaskRouter(
                local_threshold=offloading_config["local_threshold"],
                edge_threshold=offloading_config["edge_threshold"],
            )
        else:
            self.local_nodes = {}
            self.cloud_node = None
            self.router = None

        self.current_time = 0
        self._next_task_id = 0

    def add_task(self, department_name: str, compute_demand: float, priority: Optional[str] = None) -> Task:
        """Manually inject a single task into a department's queue.

        Still used directly by Step 1 tests (without `priority`) and now also
        called internally, once per generated arrival, by
        `add_generated_arrivals()` (Step 2, with `priority` set).
        """
        if department_name not in self.nodes:
            raise ValueError(
                f"Unknown department '{department_name}'. Valid options: {list(self.nodes.keys())}"
            )
        task = Task(
            task_id=self._next_task_id,
            arrival_time=self.current_time,
            compute_demand=compute_demand,
            priority=priority,
        )
        self._next_task_id += 1
        self.nodes[department_name].add_task(task)
        return task

    def add_generated_arrivals(self, arrivals: List) -> List[Task]:
        """Step 2 - inject a batch of MedicalTaskGenerator arrivals at once.

        `arrivals` is the list of `GeneratedArrival` objects returned by
        `MedicalTaskGenerator.generate_for_step()` for the CURRENT time-step.
        Each one is added through the same `add_task()` used by Step 1, so
        the queueing/capacity mechanics tested in Step 1 apply unchanged to
        generated tasks too.
        """
        created_tasks = []
        for arrival in arrivals:
            task = self.add_task(
                department_name=arrival.department,
                compute_demand=arrival.compute_demand,
                priority=arrival.priority,
            )
            created_tasks.append(task)
        return created_tasks

    def route_task(self, department_name: str, compute_demand: float, priority: Optional[str] = None) -> Task:
        """STEP 3 - like add_task(), but instead of always using the
        department's Edge node, asks self.router which tier (Local/Edge/
        Cloud) this task's compute_demand belongs on, and places it there.
        The returned Task's `.tier` field records the decision.
        """
        if self.router is None:
            raise ValueError(
                "HospitalSimulator: config has no 'offloading' section, so routing "
                "is not available. Add local_capacity/cloud_capacity/"
                "cloud_network_delay/local_threshold/edge_threshold under "
                "'offloading' in configs/environment.yaml (see the Step 3 section)."
            )
        if department_name not in self.nodes:
            raise ValueError(
                f"Unknown department '{department_name}'. Valid options: {list(self.nodes.keys())}"
            )

        tier = self.router.choose_tier(compute_demand)
        task = Task(
            task_id=self._next_task_id,
            arrival_time=self.current_time,
            compute_demand=compute_demand,
            priority=priority,
            tier=tier,
        )
        self._next_task_id += 1

        if tier == "Local":
            self.local_nodes[department_name].add_task(task)
        elif tier == "Edge":
            self.nodes[department_name].add_task(task)
        else:  # "Cloud"
            self.cloud_node.add_task(task)

        return task

    def route_generated_arrivals(self, arrivals: List) -> List[Task]:
        """STEP 3 - like add_generated_arrivals(), but routes each arrival
        through route_task() (Local/Edge/Cloud) instead of always the
        department's Edge node."""
        created_tasks = []
        for arrival in arrivals:
            task = self.route_task(
                department_name=arrival.department,
                compute_demand=arrival.compute_demand,
                priority=arrival.priority,
            )
            created_tasks.append(task)
        return created_tasks

    def step(self) -> Dict[str, dict]:
        """Advance the whole simulator by one time-step. Returns a small
        status snapshot per node (useful for logging/plots later)."""
        status = {}
        for name, node in self.nodes.items():
            node.step(self.current_time)
            status[name] = {
                "queue_length": node.queue_length,
                "completed_total": len(node.completed_tasks),
            }
        # STEP 3: Local and Cloud tiers advance the same way every step too,
        # so tasks routed there make progress even between route_task() calls.
        for name, node in self.local_nodes.items():
            node.step(self.current_time)
            status[f"{name}_Local"] = {
                "queue_length": node.queue_length,
                "completed_total": len(node.completed_tasks),
            }
        if self.cloud_node is not None:
            self.cloud_node.step(self.current_time)
            status["Cloud"] = {
                "queue_length": self.cloud_node.queue_length,
                "completed_total": len(self.cloud_node.completed_tasks),
            }
        self.current_time += 1
        return status

    def run(self, n_steps: int = None) -> None:
        """Run the simulator for a fixed number of time-steps (no tasks are
        generated automatically -- inject tasks with add_task() beforehand
        or between steps)."""
        steps = n_steps if n_steps is not None else self.n_steps
        for _ in range(steps):
            self.step()
