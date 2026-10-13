"""
Step 5 (V2) - Common evaluation protocol: fixed measurement window + drain.
================================================================================
ROMAN URDU: Pehle FCFS run "20 generation + 10 drain steps" chalta tha aur
throughput = completed / 30 tha. Masla formula nahi tha -- masla yeh tha ke
drain ki lambai (10) arbitrary thi: drain badlo to throughput badal jata,
aur drain mein complete hone wale tasks bhi gin liye jate. Ab protocol:

  1. MEASUREMENT WINDOW = steps 0 .. G-1 (G = simulation.n_steps). Naye
     tasks sirf isi window mein aate hain. PRIMARY throughput =
     (window ke andar complete hue tasks) / G. Window ke baad complete hone
     wale tasks is mein NAHI gine jate (inflated throughput nahi).
  2. DRAIN PHASE = window ke baad, bina naye arrivals ke, zyada se zyada
     `evaluation.drain_limit` steps -- ya jab tak sab tasks khatam na ho
     jayen. Drain completions aur bacha hua backlog ALAG report hote hain.
     Drain limit tak bhi kaam baqi ho to run "incomplete" flag hota hai
     (fixed drain ko guaranteed completion nahi samjha jata).
  3. FINAL snapshot (drain ke baad) se headline latency aur deadline
     violation rates aate hain -- har window-task ka deadline outcome.

Yahi protocol FCFS, DQN, FedAvg-MAPPO aur SplitFed-MAPPO sab ke liye hai:
same config, same seeds, same G aur drain_limit.

ENGLISH: see the three numbered rules above. `run_fcfs_evaluation()` runs
the FCFS baseline under this protocol; later policies must reuse
`summarize_protocol()` with the same window/drain settings.
"""

from dataclasses import dataclass
from typing import Dict, List

from env.hospital_env import HospitalSimulator
from env.metrics import compute_metrics
from env.task import Task, is_int
from env.task_generator import MedicalTaskGenerator

PROTOCOL_VERSION = "v2-window-drain"


@dataclass
class EvaluationResult:
    simulator: HospitalSimulator
    report: Dict


def backlog_snapshot(sim: HospitalSimulator) -> Dict:
    """Unfinished work on every node right now (incl. in-transit tasks)."""
    return {
        "tasks": sum(node.queue_length for node in sim.all_nodes),
        "remaining_demand": sum(node.backlog_demand for node in sim.all_nodes),
        "per_node": {node.name: node.queue_length for node in sim.all_nodes},
    }


def summarize_protocol(
    tasks: List[Task],
    window_steps: int,
    final_cutoff: int,
    drain_limit: int,
    backlog_at_window_end: Dict,
    backlog_after_drain: Dict,
    payload_units_per_compute_unit: float = 1.0,
) -> Dict:
    """Build the report for one run (policy-independent)."""
    if not is_int(window_steps) or window_steps < 1:
        raise ValueError(f"summarize_protocol: window_steps must be an int >= 1, got {window_steps!r}")
    window_cutoff = window_steps - 1
    if not is_int(final_cutoff) or final_cutoff < window_cutoff:
        raise ValueError(f"summarize_protocol: final_cutoff must be an int >= {window_cutoff}, got {final_cutoff!r}")
    window = compute_metrics(tasks, window_cutoff, payload_units_per_compute_unit)
    window["throughput"] = window["completed"] / window_steps   # tasks per step, window only
    final = compute_metrics(tasks, final_cutoff, payload_units_per_compute_unit)
    drain_steps_used = final_cutoff - window_cutoff
    run_complete = final["unfinished"] == 0
    return {
        "protocol": {
            "version": PROTOCOL_VERSION,
            "measurement_window": [0, window_cutoff],
            "window_steps": window_steps,
            "drain_limit": drain_limit,
            "drain_steps_used": drain_steps_used,
            "final_cutoff": final_cutoff,
            "time_unit": "simulation step; latency 0 = finished in its arrival step",
        },
        "headline": {
            "throughput_in_window": window["throughput"],
            "avg_latency": final["avg_latency"],
            "avg_queue_wait": final["avg_queue_wait"],
            "critical_deadline_violation_rate": final["critical_deadline_violation_rate"],
            "deadline_violation_rate": final["deadlines"]["overall"]["deadline_violation_rate"],
            "tasks": final["tasks_arrived"],
            "unfinished_after_drain": final["unfinished"],
            "run_complete": run_complete,
        },
        "window": window,
        "final": final,
        "drain": {
            "completions_during_drain": final["completed"] - window["completed"],
            "backlog_at_window_end": backlog_at_window_end,
            "backlog_after_drain": backlog_after_drain,
            "drain_limit_reached": (not run_complete) and drain_steps_used >= drain_limit,
            "run_complete": run_complete,
        },
        "training_communication": {
            "status": "not implemented in this phase",
            "note": "future FedAvg/SplitFed steps: model updates, smashed activations, gradients -- "
                    "kept separate from offloading_communication",
        },
    }


def protocol_settings(config: dict) -> Dict:
    evaluation = config.get("evaluation", {})
    return {
        "window_steps": config["simulation"]["n_steps"],
        "drain_limit": evaluation.get("drain_limit", 50),
        "payload_units_per_compute_unit": evaluation.get("synthetic_payload_units_per_compute_unit", 1.0),
    }


def run_fcfs_evaluation(config, seed: int) -> EvaluationResult:
    """FCFS baseline under the common protocol. `config` = path or dict."""
    sim = HospitalSimulator(config)
    settings = protocol_settings(sim.config)
    generator = MedicalTaskGenerator(sim.config, seed=seed)

    for _ in range(settings["window_steps"]):          # 1. measurement window
        sim.route_generated_arrivals(generator.generate_for_step(sim.current_time))
        sim.step()
    report = drain_and_report(sim, settings, backlog_snapshot(sim))
    return EvaluationResult(simulator=sim, report=report)


def drain_and_report(sim: HospitalSimulator, settings: Dict, backlog_at_window_end: Dict) -> Dict:
    """Run the drain phase on a simulator whose measurement window has just
    ended, then build the protocol report. Shared by every policy (FCFS here,
    the Step 6 Gymnasium environment for learned policies)."""
    if sim.current_time != settings["window_steps"]:
        raise ValueError(
            f"drain_and_report: the window must have just ended (current_time "
            f"{sim.current_time} != window_steps {settings['window_steps']})"
        )
    drain_steps = 0                                      # 2. drain (no arrivals)
    while drain_steps < settings["drain_limit"] and any(node.queue for node in sim.all_nodes):
        sim.step()
        drain_steps += 1

    return summarize_protocol(
        sim.tasks,
        window_steps=settings["window_steps"],
        final_cutoff=sim.current_time - 1,
        drain_limit=settings["drain_limit"],
        backlog_at_window_end=backlog_at_window_end,
        backlog_after_drain=backlog_snapshot(sim),
        payload_units_per_compute_unit=settings["payload_units_per_compute_unit"],
    )
