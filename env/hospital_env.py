"""
Step 1-5 - HospitalSimulator: ties the department edge nodes + shared Cloud
into one simulator.
================================================================================
ROMAN URDU: Yeh class poore "mini hospital" ko represent karti hai. Teen
departments (Emergency, ICU, General Ward), har ek ka apna EdgeNode, aur
poore hospital ka EK shared Cloud. Simulator ek "clock" chalata hai: har
time-step par har node ek step aage badhta hai.

  - add_task()               (Step 1): task ko source department ke APNE node
                              par daalta hai (koi routing nahi) -> tier "Local".
  - add_generated_arrivals() (Step 2): Poisson arrivals, har ek add_task() se.
  - route_task()             (Step 3/5): TaskRouter (fixed FCFS rule, AI nahi)
                              se faisla: Local / Edge (padosi) / Cloud.
  - route_generated_arrivals(): Poisson arrivals, har ek route_task() se.
  - place_task()             (Step 6): tier ka faisla bahar se (RL policy)
                              aata hai; baqi placement route_task() jaisa.

V2 (AUDIT FIX) -- roadmap/synopsis ke mutabiq action space:
  Local = task source department ke APNE edge node par process ho.
  Edge  = task kisi DOOSRE (padosi) department ke ASLI edge node par jaye --
          wahan ki real queue aur capacity share hoti hai (koi duplicate
          resource nahi). Raste mein `neighbour_network_delay` steps lagte hain.
  Cloud = ek shared remote server, `cloud_network_delay` ke saath.
Purana chhota bedside "Local" tier (local_nodes / local_capacity) hata diya
gaya hai -- woh thesis scope mein nahi tha.

ENGLISH: V2 action space follows the roadmap/synopsis: Local = the source
department's own node; Edge = a NEIGHBOUR department's real node (shared
queue + capacity, after neighbour_network_delay steps of transit); Cloud =
the one shared remote server (after cloud_network_delay). The old tiny
bedside Local tier was removed. Every task records source_department,
destination, tier and its lifecycle times (see env/task.py).

Deadlines (Step 4) are still tracking-only: nothing here uses them to decide.
"""

import copy
import os
from typing import Dict, List, Optional

import yaml

from env.cloud_node import CloudNode
from env.config_validation import validate_config
from env.edge_node import EdgeNode
from env.task import CLOUD_NODE_NAME, PRIORITY_CLASSES, Task, is_finite_number, is_int
from env.task_router import TaskRouter


def _deep_merge(base: dict, override: dict) -> dict:
    """override wins; nested dicts merge key by key, lists are replaced."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config) -> dict:
    """Accept a YAML path or an already-loaded dict (deep-copied, so the
    simulator can never mutate the caller's dict). V2: a YAML file may say
    `extends: <other.yaml>` (path relative to itself) to start from another
    config and override only some values -- used by configs/scenarios/."""
    if isinstance(config, dict):
        return copy.deepcopy(config)
    if not isinstance(config, (str, os.PathLike)):
        raise ValueError(f"config must be a YAML path or a dict, got {type(config).__name__}")
    with open(config, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file '{config}' must contain a YAML mapping")
    parent = loaded.pop("extends", None)
    if parent is None:
        return loaded
    parent_path = os.path.join(os.path.dirname(os.path.abspath(config)), parent)
    return _deep_merge(load_config(parent_path), loaded)


class HospitalSimulator:
    """Multi-department hospital simulator. No learning/scheduling AI --
    routing (if enabled) is the fixed FCFS-baseline rule in TaskRouter."""

    def __init__(self, config):
        config = load_config(config)
        validate_config(config)  # V2: whole config checked before ANY state is built
        self.config = config
        self.seed = config.get("seed", 42)
        self.n_steps = config["simulation"]["n_steps"]

        # One EdgeNode per configured department (Step 1).
        self.nodes: Dict[str, EdgeNode] = {
            dept["name"]: EdgeNode(name=dept["name"], capacity=dept["capacity"]) for dept in config["departments"]
        }

        # Cloud + routing rule -- optional, only if 'offloading' is configured.
        offloading_config = config.get("offloading")
        if offloading_config:
            self.cloud_node: Optional[CloudNode] = CloudNode(
                name=CLOUD_NODE_NAME,
                capacity=offloading_config["cloud_capacity"],
                network_delay=offloading_config["cloud_network_delay"],
            )
            self.neighbour_delay: Optional[int] = offloading_config["neighbour_network_delay"]
            self.router: Optional[TaskRouter] = TaskRouter(
                cloud_threshold=offloading_config["cloud_threshold"],
                offload_threshold=offloading_config["offload_threshold"],
                neighbour_order=offloading_config.get("neighbour_tie_break_order", list(self.nodes)),
            )
        else:
            self.cloud_node = None
            self.neighbour_delay = None
            self.router = None

        # STEP 4: deadline offsets per priority (optional; validated as ints >= 0).
        deadlines_config = config.get("deadlines")
        self.deadline_offsets: Optional[Dict[str, int]] = dict(deadlines_config) if deadlines_config else None

        self.current_time = 0
        self._next_task_id = 0
        self.tasks: List[Task] = []  # V2: every task ever created, in creation order

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def all_nodes(self) -> List[EdgeNode]:
        """Department nodes followed by the Cloud node (if any)."""
        return list(self.nodes.values()) + ([self.cloud_node] if self.cloud_node else [])

    @property
    def completed_tasks(self) -> List[Task]:
        return [t for node in self.all_nodes for t in node.completed_tasks]

    def _compute_deadline(self, priority: Optional[str]) -> Optional[int]:
        """STEP 4 - current_time + this priority's offset, or None."""
        if priority is None or self.deadline_offsets is None:
            return None
        return self.current_time + self.deadline_offsets[priority]

    def _check_new_task(self, department_name: str, compute_demand: float, priority: Optional[str]) -> None:
        """Validate BEFORE anything is mutated (no task id consumed, no queue
        touched). V2 fix: an invalid priority like "critical" used to reach
        the deadline lookup first and leak a raw KeyError."""
        if department_name not in self.nodes:
            raise ValueError(
                f"Unknown department '{department_name}'. Valid options: {list(self.nodes.keys())}"
            )
        if priority is not None and priority not in PRIORITY_CLASSES:
            raise ValueError(f"priority must be one of {PRIORITY_CLASSES} or None, got {priority!r}")
        if not is_finite_number(compute_demand) or compute_demand <= 0:
            raise ValueError(f"compute_demand must be a finite number > 0, got {compute_demand!r}")

    def _check_batch(self, arrivals: List) -> None:
        """All-or-nothing: validate every arrival before placing any of them."""
        for arrival in arrivals:
            self._check_new_task(arrival.department, arrival.compute_demand, arrival.priority)

    def _require_router(self) -> None:
        if self.router is None:
            raise ValueError(
                "HospitalSimulator: config has no 'offloading' section, so routing is not "
                "available. Add cloud_capacity/cloud_network_delay/neighbour_network_delay/"
                "cloud_threshold/offload_threshold under 'offloading' (see README)."
            )

    def _place(self, department_name, compute_demand, priority, tier, destination) -> Task:
        """Create the Task (validates it), then -- only if that succeeded --
        consume a task id and enqueue it on its destination node."""
        task = Task(
            task_id=self._next_task_id,
            arrival_time=self.current_time,
            compute_demand=compute_demand,
            priority=priority,
            tier=tier,
            deadline=self._compute_deadline(priority),
            source_department=department_name,
            destination=destination,
        )
        if tier == "Cloud":
            self.cloud_node.add_task(task)  # cloud_network_delay
        elif tier == "Edge":
            self.nodes[destination].add_task(task, transfer_delay=self.neighbour_delay)
        else:  # "Local"
            self.nodes[destination].add_task(task, transfer_delay=0)
        self._next_task_id += 1
        self.tasks.append(task)
        return task

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_task(self, department_name: str, compute_demand: float, priority: Optional[str] = None) -> Task:
        """Step 1 - place a task on its own department's node (no routing).
        V2: the task is now labelled tier="Local", destination=department,
        because that is literally where it executes."""
        self._check_new_task(department_name, compute_demand, priority)
        return self._place(department_name, compute_demand, priority, "Local", department_name)

    def add_generated_arrivals(self, arrivals: List) -> List[Task]:
        """Step 2 - inject a batch of generator arrivals via add_task()."""
        self._check_batch(arrivals)
        return [self.add_task(a.department, a.compute_demand, a.priority) for a in arrivals]

    def route_task(self, department_name: str, compute_demand: float, priority: Optional[str] = None) -> Task:
        """STEP 3/5 - ask the FCFS router where this task goes (Local / Edge
        neighbour / Cloud) using the CURRENT backlogs, then place it there.
        The task's tier/destination/source_department record the decision."""
        self._require_router()
        self._check_new_task(department_name, compute_demand, priority)
        backlogs = {name: node.backlog_demand for name, node in self.nodes.items()}
        tier, destination = self.router.choose_destination(department_name, compute_demand, backlogs)
        return self._place(department_name, compute_demand, priority, tier, destination)

    def route_generated_arrivals(self, arrivals: List) -> List[Task]:
        """STEP 3/5 - route a batch of generator arrivals, in the given (seeded,
        shuffled) order. Each routing decision sees the backlog left by the
        previous ones. The batch is validated first (all-or-nothing)."""
        self._require_router()
        self._check_batch(arrivals)
        return [self.route_task(a.department, a.compute_demand, a.priority) for a in arrivals]

    def place_task(self, department_name: str, compute_demand: float, priority: Optional[str], tier: str) -> Task:
        """STEP 6 - place a task on a tier chosen by a POLICY (e.g. an RL agent)
        instead of the FCFS router. Local = own node, Cloud = shared cloud,
        Edge = the neighbour chosen by TaskRouter.least_loaded_neighbour()
        (the same neighbour rule the FCFS router uses). Placement, delays and
        bookkeeping are the same _place() used by route_task()."""
        self._require_router()
        self._check_new_task(department_name, compute_demand, priority)
        if tier == "Local":
            destination = department_name
        elif tier == "Cloud":
            destination = CLOUD_NODE_NAME
        elif tier == "Edge":
            backlogs = {name: node.backlog_demand for name, node in self.nodes.items()}
            destination = self.router.least_loaded_neighbour(department_name, backlogs)
        else:
            raise ValueError(f"place_task: tier must be 'Local', 'Edge' or 'Cloud', got {tier!r}")
        return self._place(department_name, compute_demand, priority, tier, destination)

    def step(self) -> Dict[str, dict]:
        """Advance every node by one time-step. Department nodes do not share
        capacity with each other (and all arrivals were placed before the
        step), so the iteration order here gives no department an advantage."""
        status = {}
        for node in self.all_nodes:
            node.step(self.current_time)
            status[node.name] = {
                "queue_length": node.queue_length,
                "in_transit": node.in_transit_count(self.current_time + 1),
                "completed_total": len(node.completed_tasks),
                "last_step_utilization": node.last_step_utilization,
            }
        self.current_time += 1
        return status

    def run(self, n_steps: int = None) -> None:
        """Run for a fixed number of steps (no automatic task generation)."""
        steps = n_steps if n_steps is not None else self.n_steps
        if not is_int(steps) or steps < 0:
            raise ValueError(f"run(): n_steps must be an int >= 0, got {steps!r}")
        for _ in range(steps):
            self.step()
