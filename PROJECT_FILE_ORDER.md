# File Order Reference (VSCode shows files A-Z, not by step)

**ROMAN URDU:** VSCode ka file explorer hamesha alphabetical order mein files
dikhata hai (`cloud_node.py` se pehle `edge_node.py`, waghera) — yeh order
*step order NAHI hai*. Yeh file sirf isliye banayi gayi hai ke jab bhi confuse
ho "kaunsi file pehle bani thi", isko khol kar dekh lein.

**ENGLISH:** VSCode's file explorer always sorts alphabetically — that is
NOT the order things were built in. This file exists purely as a lookup so
you (or Mam) can see, at a glance, which step each file belongs to.

**Important — why files aren't literally renamed `step1_...`, `step2_...`:**
Most files were **created** in one step and then **edited again** in later
steps (that's normal — e.g. `task.py` got a new field almost every step).
Naming a file `step1_task.py` would be **wrong information**, because most
of its current code was actually written across four different steps, not
just Step 1. The table below shows the truth: created-in-step AND every
step that later touched it.

---

## `env/` folder — core simulator code

| File | Created | Also edited in | What it does |
|---|---|---|---|
| [task.py](env/task.py) | **Step 1** | Step 2 (+priority), Step 3 (+tier), Step 4 (+deadline, met_deadline) | `Task` — one medical job (id, arrival time, compute demand, priority, tier, deadline) |
| [edge_node.py](env/edge_node.py) | **Step 1** | Post-Step-3 fix (utilization bug), V2 audit (destination-ready FCFS) | `EdgeNode` — one department's limited-capacity queue (ready-time FCFS) |
| [hospital_env.py](env/hospital_env.py) | **Step 1** | Step 2 (+add_generated_arrivals), Step 3 (+route_task), Step 4 (+deadline calc), V2 audit (neighbour Edge, config validation), Step 6 (+place_task) | `HospitalSimulator` — ties all departments + cloud together, runs the clock |
| [task_generator.py](env/task_generator.py) | **Step 2** | V2 audit (seeded same-step shuffle, shared validation) | `MedicalTaskGenerator` — Poisson-process arrivals per dept/priority |
| [task_router.py](env/task_router.py) | **Step 3** | V2 audit (neighbour Edge), Step 6 (+least_loaded_neighbour, shared with the RL Edge action) | `TaskRouter` — fixed (non-AI) rule: demand → Cloud, else source backlog → Local / least-loaded neighbour Edge |
| [cloud_node.py](env/cloud_node.py) | **Step 3** | V2 audit (delay stored per task) | `CloudNode` — shared hospital-wide node with network delay |
| [metrics.py](env/metrics.py) | **Step 5** | V2 audit (rewritten) | `compute_metrics()` — latency, throughput, 4 deadline categories, tier/route breakdowns |
| [evaluation.py](env/evaluation.py) | **Step 5 (V2 audit)** | Step 6 (+drain_and_report, shared with the Gym env) | Common window + drain evaluation protocol for every policy |
| [run_records.py](env/run_records.py) | **Step 5 (V2 audit)** | Step 6 (+gymnasium version in provenance) | Unique run folders, manifest with config snapshot and git/package provenance |
| [config_validation.py](env/config_validation.py) | **Step 5 (V2 audit)** | Step 6 (+rl_environment section) | Validates every config before a simulation is built |
| [gym_env.py](env/gym_env.py) | **Step 6** | — | `HospitalSchedulingEnv` — Gymnasium reset()/step() around the simulator (Local / Edge / Cloud action) |

*(Step 4 added no new file — only edited `task.py` and `hospital_env.py` above.)*

## Root folder — runnable scripts

| File | Step | What it does |
|---|---|---|
| [demo_simulation.py](demo_simulation.py) | Steps 1-4 demo | Runs the full pipeline once with seed=42 and prints a readable summary |
| [check_simple_policies.py](check_simple_policies.py) | **Step 6** | Sanity check: FCFS / always Local / Edge / Cloud / random played through the Gym env |
| [run_baseline_fcfs.py](run_baseline_fcfs.py) | **Step 5** | Runs the FCFS baseline (`--seed`, `--seeds 0-29`, `--config`), saves a run folder under `results/runs/` |

## `configs/environment.yaml` — one file, grew every step

| Section | Added in |
|---|---|
| `seed`, `simulation`, `departments` | Step 0/1 |
| `task_generation` | Step 2 |
| `offloading` | Step 3 |
| `deadlines` | Step 4 |
| `evaluation` | Step 5 (V2 audit) |
| `rl_environment` | Step 6 |

Stress scenarios live in `configs/scenarios/*.yaml`; each one `extends:` this file and overrides only what it tests (V2 audit).

## `tests/` folder — one test file per step (this pattern IS chronological)

| File | Step |
|---|---|
| [test_step1_simulator.py](tests/test_step1_simulator.py) | Step 1 |
| [test_step2_task_generator.py](tests/test_step2_task_generator.py) | Step 2 |
| [test_step3_routing.py](tests/test_step3_routing.py) | Step 3 |
| [test_step4_deadlines.py](tests/test_step4_deadlines.py) | Step 4 |
| [test_step5_fcfs_baseline.py](tests/test_step5_fcfs_baseline.py) | Step 5 |
| [test_v2_routing_timing.py](tests/test_v2_routing_timing.py) | V2 audit (a) |
| [test_v2_metrics_protocol.py](tests/test_v2_metrics_protocol.py) | V2 audit (b) |
| [test_v2_validation_scenarios.py](tests/test_v2_validation_scenarios.py) | V2 audit (c) |
| [test_step6_gym_env.py](tests/test_step6_gym_env.py) | Step 6 |

---

## Fastest way to check a file's step without this table

Open the file — every file in this project starts with a docstring whose
first line names its step, e.g.:

```python
"""
Step 5 - Baseline evaluation metrics (FCFS baseline, and every baseline after it).
================================================================================
```

That header is always accurate and updates faster than this table will, so
if the two ever disagree, trust the file's own docstring.
