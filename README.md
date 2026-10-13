# MS Thesis Project: Privacy-Preserving SplitFed-MADRL

Practical implementation for: *Privacy-Preserving SplitFed-MADRL for Priority-Aware
Task Scheduling in Healthcare Edge Computing* (Usama Manzoor, MSIT71S25S008,
University of Sargodha).

The code follows the 18-step practical roadmap (`My Research Practical ROADMAP.docx`).
Each step adds one new idea and is verified before the next step starts.

**Current status: Steps 1–5 complete** (hospital simulator + FCFS baseline).
No AI or learning yet. Steps 6–18 (Gymnasium wrapper, DQN, MAPPO, SplitFed,
privacy, robustness) have not been started.

> **Data note:** all workload numbers (arrival rates, compute demands,
> deadlines) are **synthetic placeholders**, not MIMIC-IV statistics.
> PhysioNet credentialed access is still pending. Only the config file needs
> to change once real statistics are available.

## Quick start

Requires Python ≥ 3.10 (tested with 3.14.6 on Windows 11).

```bash
pip install -r requirements.txt
python -m pytest -q                      # 74 tests, all should pass
python demo_simulation.py                # one readable run (seed 42)
python run_baseline_fcfs.py --seeds 0-29 # FCFS baseline, 30 seeds
python run_baseline_fcfs.py --config configs/scenarios/overload.yaml --seeds 0-29
```

Each test file can also be run on its own, e.g. `python -m tests.test_step3_routing`.

## System model (current, "v2")

Three hospital departments (Emergency, ICU, General Ward), each with its own
edge computer, plus one shared cloud server. Every new task can be processed
in one of three places:

| Action | Where the task runs | Network delay (default) |
|---|---|---|
| **Local** | the source department's own edge node | 0 steps |
| **Edge** | a *neighbour* department's edge node (its real queue and capacity are shared) | 1 step |
| **Cloud** | one shared cloud server (capacity 50) | 2 steps |

Each node processes its queue in **destination-ready FCFS** order: a task in
network transit does not use CPU or block other tasks; ready tasks are served
in order of arrival at the node, up to the node's capacity per step. Large tasks
carry over to later steps.

**FCFS routing rule** (fixed, hand-written, priority-blind; this is the decision
the RL agents will learn from Step 7 onward):

1. `compute_demand > cloud_threshold` (6.0) → **Cloud**
2. otherwise, if the source node's backlog ≤ `offload_threshold` (3.0) → **Local**
3. otherwise → **Edge**: the neighbour with the smallest backlog
   (ties broken by `neighbour_tie_break_order`)

All parameters are in [configs/environment.yaml](configs/environment.yaml) and
are validated before a simulation starts (bad types, NaN/inf, unknown keys,
and removed pre-v2 keys raise clear errors).

## Progress

- [x] **Step 1 — Mini hospital simulator (no AI).** `Task`, `EdgeNode`,
      `HospitalSimulator`: fixed per-step capacity, FCFS queues, carry-over of
      large tasks, departments independent of each other.
- [x] **Step 2 — Medical task generation.** `MedicalTaskGenerator`: independent
      Poisson arrivals per (department, priority) with uniform compute demand
      per priority class; seeded and reproducible. Same-step arrivals are put
      in a seeded random order (so no department is always served first).
- [x] **Step 3 — Local / Edge / Cloud processing.** `TaskRouter` + `CloudNode`,
      with the neighbour-edge model and routing rule described above.
- [x] **Step 4 — Priority and deadlines.** Deadline = arrival time + priority
      offset (Critical 2, High 5, Routine 10 steps; placeholders). Deadlines are
      tracked only; the FCFS baseline does not use them to make decisions.
- [x] **Step 5 — Baseline 1: FCFS.** A common evaluation protocol (below),
      metrics, multi-seed runs, verification scenarios, and saved run records
      with full provenance.
- [ ] Steps 6–18 — see the roadmap document. Next: Step 6, Gymnasium wrapper.

The design was revised after an audit (commits `470a079`, `acdb3d4`,
`83e7f3f`). The pre-audit version is preserved under the git tag
`v0.5-pre-audit-fixes`; its result is kept in [results/legacy/](results/legacy/)
and is **not directly comparable** with current results.

## Evaluation protocol

Used for FCFS now, and for every later policy (DQN, MAPPO, FedAvg-MAPPO,
SplitFed-MAPPO) so the comparison is fair. Implemented in
[env/evaluation.py](env/evaluation.py) and [env/metrics.py](env/metrics.py).

- **Measurement window:** steps `0 … n_steps-1` (default 20). Tasks arrive only here.
- **Throughput** (primary) = tasks completed inside the window / `n_steps`.
- **Drain:** after the window, the simulation keeps running with no new
  arrivals for up to `drain_limit` (50) steps or until all queues are empty.
  Work still unfinished after that is reported (`unfinished_after_drain`,
  `run_complete = false`), never silently dropped.
- **Latency** = completion time − arrival time (in steps; 0 = finished in the
  arrival step) = transmission delay + queue wait + execution span.
- **Deadline status** of every task at the observation cutoff:
  `on_time`, `completed_late`, `overdue_unfinished` (deadline passed, not
  finished), or `not_yet_due` (never counted as a miss).
- **Deadline violation rate** = (completed_late + overdue_unfinished) / decided
  tasks, reported overall and for **Critical** tasks separately. An empty group
  is reported as `None`, not 0.
- Also reported: breakdowns by tier, source and destination, and offloading
  communication (transfer counts and *synthetic* payload units, not bytes).
  Training communication (FedAvg/SplitFed) is not implemented yet.

## FCFS baseline results (seeds 0–29, mean)

From [results/baselines/](results/baselines/), produced at commit `83e7f3f`
from a clean working tree.

| Scenario | Tasks / run | Throughput (tasks/step) | Avg latency (steps) | Deadline violation | Critical violation |
|---|---|---|---|---|---|
| Default | 26.1 | 1.26 | 0.60 | 0.0% | 0.0% |
| Department imbalance (ICU ×6) | 65.4 | 3.02 | 1.06 | 1.1% | 3.2% |
| Overload (all rates ×5) | 124.2 | 4.86 | 3.44 | 22.2% | 59.6% |
| Slow network (rates ×3, delays 6/3) | 74.1 | 2.86 | 3.71 | 20.0% | 83.4% |
| Overdue Critical (artificial stress test) | 26.1 | 1.15 | 1.66 | 19.6% | 87.3% |
| Empty workload | 0 | 0.00 | None | None | None |

**Note:** the default workload is light, so FCFS already meets every deadline.
Learned policies should therefore also be compared on the stressed scenarios
(overload, slow network, department imbalance), where FCFS clearly struggles.

Each run folder contains `manifest.json` (full config snapshot, seeds, git
commit and dirty flag, Python and package versions), plus `summary.json` and
`per_seed.csv` for multi-seed runs, or `metrics.json` and `tasks.csv` (one row per task)
for a single seed. Re-running the same command reproduces these files exactly.
New runs go to `results/runs/` (git-ignored); a run folder is never overwritten.

## Verification

74 automated tests (`python -m pytest -q`):

| File | What it checks |
|---|---|
| `test_step1_simulator.py` | capacity per step, carry-over, department independence, utilization, invalid input |
| `test_step2_task_generator.py` | Poisson rates, demand ranges, seed reproducibility |
| `test_step3_routing.py` | Local / Edge / Cloud routing, network delay |
| `test_step4_deadlines.py` | deadline calculation and on-time / late boundary |
| `test_step5_fcfs_baseline.py` | metrics against hand-calculated values, determinism |
| `test_v2_routing_timing.py` | neighbour routing, ready-time FCFS, latency decomposition, utilization |
| `test_v2_metrics_protocol.py` | deadline categories, window/drain protocol, run records |
| `test_v2_validation_scenarios.py` | config validation (30+ invalid cases), scenarios, roadmap Step 3 checkpoints, seeds 0–29 invariants |

## Folder structure

```
MS_Thesis_Project/
├── configs/
│   ├── environment.yaml          # all parameters (default workload)
│   └── scenarios/                # stress scenarios; each extends environment.yaml
├── env/
│   ├── task.py                   # Task: one medical job and its timing fields
│   ├── edge_node.py              # EdgeNode: capacity-limited destination-ready FCFS queue
│   ├── cloud_node.py             # CloudNode: shared server with network delay
│   ├── task_generator.py         # MedicalTaskGenerator: Poisson arrivals
│   ├── task_router.py            # TaskRouter: fixed FCFS Local/Edge/Cloud rule
│   ├── hospital_env.py           # HospitalSimulator: ties everything together, runs the clock
│   ├── config_validation.py      # config checks run before anything is built
│   ├── metrics.py                # latency, throughput, deadline categories, breakdowns
│   ├── evaluation.py             # common window + drain evaluation protocol
│   └── run_records.py            # run folders, manifests, provenance
├── tests/                        # 74 tests (see Verification)
├── results/
│   ├── baselines/                # committed FCFS v2 results
│   └── legacy/                   # pre-audit result, not comparable
├── Step_Documentation/           # per-step Word write-ups (Roman Urdu), see note below
├── demo_simulation.py            # one readable run with seed 42
├── run_baseline_fcfs.py          # FCFS baseline runner (--seed / --seeds / --config)
└── requirements.txt
```

**Note on `Step_Documentation/`:** the Step 3 and Step 5 write-ups were
rewritten for the current (v2) design. Steps 1, 2 and 4 start with a
"V2 UPDATE" box listing what changed; their code listings are simplified
versions of the code in `env/`.

## Known limitations

- Workload and deadline values are placeholders, not MIMIC-IV derived.
- Time is discrete (whole steps); network delays are fixed integers.
- Offloading payload is a synthetic unit proportional to compute demand, not bytes.
- No learning yet. FCFS routing ignores priority and deadlines by design.
