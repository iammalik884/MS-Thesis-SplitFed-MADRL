"""
V2 (audit fix) - One place that validates the whole configuration.
================================================================================
ROMAN URDU: Pehle har class apni thori si validation khud karti thi, aur kai
galat values chupke se guzar jati thin: NaN (`nan <= 0` False hota hai),
infinity, `true` jaisi bool values, 2.5 jaise fractional steps, duplicate ya
unknown department, ya typo wali key (jaise `deadline:` bajaye `deadlines:`).
Ab simulator ya generator kuch bhi banane se PEHLE poori config yahan check
hoti hai -- galat config par saaf ValueError, aur koi aadhi-adhuri state
nahi banti.

ENGLISH: Validates the full config BEFORE any simulator/generator state is
built: finite numbers only (no NaN/inf/bool), integer-only step fields,
unique and consistent department names, no unknown or removed keys.
Every error message names the exact config path.
"""

from typing import Iterable, Optional

from env.task import CLOUD_NODE_NAME, PRIORITY_CLASSES, is_finite_number, is_int

TOP_LEVEL_KEYS = {"seed", "simulation", "departments", "task_generation", "offloading",
                  "deadlines", "evaluation", "scenario", "rl_environment"}
RL_ENVIRONMENT_KEYS = {"deadline_miss_penalty", "priority_weights"}   # priority_weights: Step 10, optional
OFFLOADING_KEYS = {"cloud_capacity", "cloud_network_delay", "neighbour_network_delay",
                   "cloud_threshold", "offload_threshold", "neighbour_tie_break_order"}
REMOVED_OFFLOADING_KEYS = ("local_capacity", "local_threshold", "edge_threshold")
EVALUATION_KEYS = {"drain_limit", "synthetic_payload_units_per_compute_unit"}


def _fail(path: str, message: str, value) -> None:
    raise ValueError(f"Config '{path}' {message}, got {value!r}")


def _check_keys(section: dict, allowed: set, path: str) -> None:
    if not isinstance(section, dict):
        _fail(path, "must be a mapping", section)
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"Config '{path}' has unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _int(value, path: str, minimum: int = 0) -> None:
    if not is_int(value) or value < minimum:
        _fail(path, f"must be an int >= {minimum}", value)


def _number(value, path: str, positive: bool) -> None:
    if not is_finite_number(value) or (value <= 0 if positive else value < 0):
        _fail(path, "must be a finite number " + ("> 0" if positive else ">= 0"), value)


def validate_task_generation(section, department_names: Optional[Iterable[str]] = None) -> None:
    """task_generation.departments: every configured department, each with
    exactly Critical/High/Routine, finite rates >= 0 and 0 < low <= high."""
    if not section or not isinstance(section, dict) or not section.get("departments"):
        raise ValueError(
            "Config is missing a 'task_generation.departments' section. Step 2 requires "
            "arrival_rate and compute_demand_range for every department and priority class."
        )
    _check_keys(section, {"departments"}, "task_generation")
    gen_departments = section["departments"]
    if not isinstance(gen_departments, dict):
        _fail("task_generation.departments", "must be a mapping", gen_departments)
    if department_names is not None and set(gen_departments) != set(department_names):
        raise ValueError(
            f"Config 'task_generation.departments' {sorted(gen_departments)} must match the "
            f"'departments' list {sorted(department_names)} exactly (no unknown or missing department)"
        )
    for dept, priorities in gen_departments.items():
        base = f"task_generation.departments.{dept}"
        if not isinstance(priorities, dict):
            _fail(base, "must be a mapping", priorities)
        for priority in PRIORITY_CLASSES:
            if priority not in priorities:
                raise ValueError(f"{base} is missing priority class '{priority}' (need all of {PRIORITY_CLASSES})")
        _check_keys(priorities, set(PRIORITY_CLASSES), base)
        for priority, settings in priorities.items():
            path = f"{base}.{priority}"
            _check_keys(settings, {"arrival_rate", "compute_demand_range"}, path)
            _number(settings.get("arrival_rate"), f"{path}.arrival_rate", positive=False)
            demand_range = settings.get("compute_demand_range")
            if (not isinstance(demand_range, (list, tuple)) or len(demand_range) != 2
                    or not all(is_finite_number(v) for v in demand_range)):
                _fail(f"{path}.compute_demand_range", "must be a [low, high] pair of finite numbers", demand_range)
            low, high = demand_range
            if low <= 0 or high < low:
                _fail(f"{path}.compute_demand_range", "must satisfy 0 < low <= high", demand_range)


def validate_config(config) -> None:
    """Raise ValueError on the first problem; return None if valid."""
    if not isinstance(config, dict):
        _fail("<root>", "must be a mapping", type(config).__name__)
    unknown = sorted(set(config) - TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"Config has unknown top-level key(s) {unknown}; allowed: {sorted(TOP_LEVEL_KEYS)}")

    _int(config.get("seed", 42), "seed")
    simulation = config.get("simulation")
    _check_keys(simulation, {"n_steps"}, "simulation")
    _int(simulation.get("n_steps"), "simulation.n_steps", minimum=1)

    departments = config.get("departments")
    if not isinstance(departments, list) or not departments:
        _fail("departments", "must be a non-empty list", departments)
    names = []
    for i, dept in enumerate(departments):
        _check_keys(dept, {"name", "capacity"}, f"departments[{i}]")
        name = dept.get("name")
        if not isinstance(name, str) or not name.strip():
            _fail(f"departments[{i}].name", "must be a non-empty string", name)
        if name in names or name == CLOUD_NODE_NAME:
            _fail(f"departments[{i}].name", f"must be unique and not '{CLOUD_NODE_NAME}'", name)
        names.append(name)
        _number(dept.get("capacity"), f"departments[{i}].capacity", positive=True)

    if "task_generation" in config:
        validate_task_generation(config["task_generation"], names)

    offloading = config.get("offloading")
    if offloading:
        stale = [k for k in REMOVED_OFFLOADING_KEYS if k in offloading]
        if stale:
            raise ValueError(
                f"Config 'offloading' uses removed pre-v2 keys {stale}: the bedside Local tier no "
                "longer exists (Local = own department node, Edge = neighbour). Use cloud_threshold / "
                "offload_threshold / neighbour_network_delay -- see README."
            )
        _check_keys(offloading, OFFLOADING_KEYS, "offloading")
        missing = sorted(OFFLOADING_KEYS - {"neighbour_tie_break_order"} - set(offloading))
        if missing:
            raise ValueError(f"Config 'offloading' is missing key(s) {missing}")
        if len(names) < 2:
            _fail("departments", "needs >= 2 departments when 'offloading' is enabled (Edge = a neighbour)", names)
        _number(offloading["cloud_capacity"], "offloading.cloud_capacity", positive=True)
        _int(offloading["cloud_network_delay"], "offloading.cloud_network_delay")
        _int(offloading["neighbour_network_delay"], "offloading.neighbour_network_delay")
        _number(offloading["cloud_threshold"], "offloading.cloud_threshold", positive=True)
        _number(offloading["offload_threshold"], "offloading.offload_threshold", positive=False)
        order = offloading.get("neighbour_tie_break_order", names)
        if not isinstance(order, list) or sorted(order) != sorted(names):
            _fail("offloading.neighbour_tie_break_order", f"must list every department exactly once {names}", order)

    deadlines = config.get("deadlines")
    if deadlines:
        if not isinstance(deadlines, dict):
            _fail("deadlines", "must be a mapping", deadlines)
        for priority in PRIORITY_CLASSES:
            if priority not in deadlines:
                raise ValueError(
                    f"Config 'deadlines' section is missing priority class '{priority}' (need all of {PRIORITY_CLASSES})"
                )
        _check_keys(deadlines, set(PRIORITY_CLASSES), "deadlines")
        for priority, offset in deadlines.items():
            _int(offset, f"deadlines.{priority}")

    evaluation = config.get("evaluation")
    if evaluation is not None:
        _check_keys(evaluation, EVALUATION_KEYS, "evaluation")
        if "drain_limit" in evaluation:
            _int(evaluation["drain_limit"], "evaluation.drain_limit")
        if "synthetic_payload_units_per_compute_unit" in evaluation:
            _number(evaluation["synthetic_payload_units_per_compute_unit"],
                    "evaluation.synthetic_payload_units_per_compute_unit", positive=False)

    rl_environment = config.get("rl_environment")
    if rl_environment is not None:
        _check_keys(rl_environment, RL_ENVIRONMENT_KEYS, "rl_environment")
        if "deadline_miss_penalty" not in rl_environment:
            raise ValueError("Config 'rl_environment' is missing key 'deadline_miss_penalty'")
        _number(rl_environment["deadline_miss_penalty"], "rl_environment.deadline_miss_penalty", positive=False)
        weights = rl_environment.get("priority_weights")
        if weights is not None:                      # Step 10: optional, absent = 1.0 for every priority
            _check_keys(weights, set(PRIORITY_CLASSES), "rl_environment.priority_weights")
            for priority in PRIORITY_CLASSES:
                if priority not in weights:
                    raise ValueError(f"Config 'rl_environment.priority_weights' is missing priority class '{priority}'")
                _number(weights[priority], f"rl_environment.priority_weights.{priority}", positive=True)
