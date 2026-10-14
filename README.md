# MS Thesis Project: Privacy-Preserving SplitFed-MADRL

Practical implementation for: *Privacy-Preserving SplitFed-MADRL for Priority-Aware
Task Scheduling in Healthcare Edge Computing* (Usama Manzoor, MSIT71S25S008,
University of Sargodha).

The code follows the 18-step practical roadmap (`My Research Practical ROADMAP.docx`).
Each step adds one new idea and is verified before the next step starts.

**Current status: Steps 1–7 complete** (hospital simulator, FCFS baseline,
Gymnasium environment, single-agent DQN baseline). The DQN was evaluated once
on reserved test seeds; it beats FCFS under overload but does not beat the
ECT heuristic in any scenario, and its training is not stable (see Step 7
below). Steps 8–18 (MAPPO, SplitFed, privacy, robustness) have not been
started.

> **Data note:** all workload numbers (arrival rates, compute demands,
> deadlines) are **synthetic placeholders**, not MIMIC-IV statistics.
> PhysioNet credentialed access is still pending. Only the config file needs
> to change once real statistics are available.

## Quick start

Requires Python ≥ 3.10 (tested with 3.14.6 on Windows 11).

```bash
pip install -r requirements.txt
python -m pytest -q                      # 96 tests, all should pass
python demo_simulation.py                # one readable run (seed 42)
python run_baseline_fcfs.py --seeds 0-29 # FCFS baseline, 30 seeds
python run_baseline_fcfs.py --config configs/scenarios/overload.yaml --seeds 0-29
python check_simple_policies.py --config configs/scenarios/overload.yaml  # Step 6 sanity check
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
- [x] **Step 6 — Gymnasium environment.** `HospitalSchedulingEnv`
      ([env/gym_env.py](env/gym_env.py)): `reset()` / `step(action)` around the
      simulator, with the observation, action, reward and termination defined
      below. With the FCFS router's actions it reproduces the Step 5 baseline
      exactly.
- [x] **Step 7 — Baseline 2: single-agent DQN.** Design note (cloud
      finding, ECT heuristic, γ, seed/evaluation plan), pilot, two
      pre-registered debugging rounds on validation seeds, and a one-time
      evaluation of the frozen setting (Double DQN) on test seeds; see below.
      Still open with the supervisor: the modelling options in the design
      note (none applied).
- [ ] Steps 8–18 — see the roadmap document.

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

## Step 6: Gymnasium environment

`HospitalSchedulingEnv(config)` in [env/gym_env.py](env/gym_env.py) (Gymnasium 1.4.0).
Nothing in the simulator changes; only the Local / Edge / Cloud decision comes
from the agent instead of the FCFS router.

- **Decision point:** one decision per arriving task. The tasks of a step are
  shown one by one, in the same seeded order FCFS routes them; after the last
  one the simulator runs the step. Steps with no arrivals run automatically.
- **Action:** `Discrete(3)` — 0 = Local (own node), 1 = Edge, 2 = Cloud. For
  Edge, the neighbour is chosen by the same rule FCFS uses (smallest backlog,
  fixed tie-break order; `TaskRouter.least_loaded_neighbour`). No action is
  masked.
- **Observation:** `Box` float32, 13 values for 3 departments: backlog/capacity
  of each department node and of the cloud, the task's demand / its own node's
  capacity, source department one-hot, priority one-hot, deadline offset /
  largest offset, and current time / `n_steps`. Backlogs include tasks in
  transit, as in the FCFS router. Backlog and demand features have no upper
  bound (`inf`), which Gymnasium's checker reports as a warning. The
  observation does not contain every queued task, so the problem is partially
  observable.
- **Reward:** each task gives `-(latency) - deadline_miss_penalty × [deadline missed]`
  once, when it finishes. A task still unfinished after the drain gives
  `-(final_cutoff - arrival + 1)` and the penalty if its deadline has passed.
  The episode return therefore equals `-(total latency) - penalty × (number of
  deadline misses)`. `deadline_miss_penalty` = 10.0 in
  [configs/environment.yaml](configs/environment.yaml) (`rl_environment`) is a
  design choice (equal to the largest deadline offset), not derived from data;
  priority weighting is roadmap Step 10.
- **Termination:** after the measurement window there are no more decisions;
  the drain runs and the episode ends with `terminated=True` (`truncated` is
  always `False`). The last `info["report"]` is the same Step 5 protocol report.
- **Seeds:** `reset(seed=s)` uses `s` as the workload seed, so it produces the same
  tasks as `run_fcfs_evaluation(config, s)`.

**Sanity check, not a result** (`python check_simple_policies.py`, seeds 0–29,
mean):

| Scenario | FCFS | always Local | always Edge | always Cloud | random | ECT (Step 7 note) |
|---|---|---|---|---|---|---|
| Default — return | −15.9 | −23.9 | −39.0 | −52.3 | −34.3 | −15.9 |
| Department imbalance — return | −78.7 | −701.9 | −219.4 | −130.8 | −109.9 | −80.2 |
| Overload — return | −720.0 | −1809.2 | −1916.6 | −248.5 | −504.9 | **−210.1** |
| Overload — Critical violation | 59.6% | 75.6% | 87.9% | **0.0%** | 30.6% | 0.7% |
| Slow network — return | −428.7 | −385.6 | −666.3 | −881.6 | −507.8 | **−253.3** |
| Slow network — Critical violation | 83.4% | 24.2% | 100.0% | 100.0% | 68.4% | 31.8% |

The reward responds to the decisions as expected. Which policy is best depends
on the scenario. Under overload, always-Cloud beats FCFS by a wide margin: the
2-step network delay is in the reward, but there is no separate usage or
communication cost, and the placeholder cloud (capacity 50) almost never fills
up (with always-Cloud under overload, 1 of 3,727 tasks over seeds 0–29 had to
wait). Always-Cloud is still not optimal: the adaptive ECT heuristic beats it
there, and it is the worst policy in the default and slow-network scenarios.
FCFS is beaten by simple policies in some scenarios, so it is a weak reference
on its own. Possible modelling changes, the γ choice and the Step 7 protocol
are in the [Step 7 design note](Step_Documentation/Step7_Design_Note.md). No
change to the cloud or the reward has been made.

## Step 7: single-agent DQN baseline

[Step_Documentation/Step7_Design_Note.md](Step_Documentation/Step7_Design_Note.md) covers:

- the scenario-specific cloud finding;
- the baselines, including the proposed **ECT** heuristic
  ([agents/heuristics.py](agents/heuristics.py)): smallest estimated completion
  time, using only observation-level information;
- possible modelling changes with units and sensitivity plans (**none applied**);
- the choice of γ for per-task, completion-time rewards;
- the Step 7 protocol: training seeds 10,000–19,999, validation 1,000–1,029,
  test 2,000–2,029 (reserved), 5 training seeds, learning curves, action
  proportions, and the same evaluation protocol for all policies.

**Pilot, preliminary** (`python train_dqn_pilot.py --config <scenario>`, 1
training seed, 40,000 decisions, validation return, runs in
[results/pilot/](results/pilot/)). The best checkpoints scored −20.4 (default),
−340.7 (overload) and −478.3 (slow network). ECT scored −14.6, −200.2 and
−240.8 on the same seeds. Training was unstable in overload and slow network.
Diagnosis and next debugging steps are in the note, Section 6. No thesis claim
is made from the pilot.

**Debugging** ([Step7_DQN_Debugging.md](Step_Documentation/Step7_DQN_Debugging.md),
each plan committed before its runs). Settings were changed one at a time,
with 2 training seeds, on validation seeds only (runs in
[results/debug/](results/debug/)). Score: 0 = random, 1 = ECT, mean over the
final checkpoints of 3 scenarios × 2 seeds.

| Round | Best setting of the round | Mean final score |
|---|---|---|
| 1 (reward scale, obs. scale, learning rate, γ) | reward × 0.01, γ = 0.95 | −0.565 |
| 2 (Double DQN; 120,000 decisions) | + Double DQN, 40,000 decisions | **0.486** |

In Round 1 the overload Q estimates were far above the matching Monte Carlo
returns (overestimation); Double DQN reduced this most. No setting met the
pre-registered stability rule. The frozen setting is the one with the highest
mean final score: `reward_scale = 0.01`, `gamma = 0.95`, `double_dqn = true`,
40,000 decisions.

**Close-out** (`python run_dqn_experiments.py --group final --base reward_scale=0.01 --base gamma=0.95 --base double_dqn=1 --exp FINAL= --seeds 0 1 2 3 4 --workers 1`,
copy the run folders without their large `evaluation/` subfolders to
`results/step7_final/training_runs/`, then
`python evaluate_dqn_test.py --runs results/step7_final/training_runs --out results/step7_final`).
5 training seeds per scenario, checkpoint chosen on validation seeds, then
**one** evaluation on test seeds 2,000–2,029 ([results/step7_final/](results/step7_final/)).
DQN = mean ± 95% CI over the 5 training seeds.

| Test seeds, return | Default | Overload | Slow network |
|---|---|---|---|
| DQN (best checkpoint) | −20.1 ± 0.7 | −278.6 ± 71.5 | −411.9 ± 49.3 |
| DQN (final checkpoint) | −20.8 ± 1.1 | −584.2 ± 512.1 | −463.8 ± 72.7 |
| **ECT** | **−15.5** | **−215.2** | **−246.0** |
| FCFS | −15.7 | −772.7 | −409.6 |
| always Local | −19.9 | −1,932.2 | −402.9 |
| always Cloud | −52.7 | −253.5 | −868.7 |
| random | −34.7 | −555.2 | −492.6 |

ECT is the best policy in every scenario. The DQN beats FCFS and random under
overload, is about equal to FCFS and always Local in the slow network, and is
worse than FCFS in the default scenario. Its final checkpoints are much worse
than the validation-selected ones under overload. All metrics (latency,
Critical violations, offloading communication, action shares) are in
`results/step7_final/summary.json` (short table: `results_table.md`); the analysis is in
[Step7_DQN_Debugging.md](Step_Documentation/Step7_DQN_Debugging.md).

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

96 automated tests (`python -m pytest -q`):

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
| `test_ect_heuristic.py` | ECT estimates by hand, exact on idle nodes, tie order, no future information |
| `test_dqn_pilot.py` | replay buffer, no bootstrapping at episode end, Double DQN target, ε schedule, reproducible training, shared evaluation = Step 5 FCFS, scaling knobs, Q vs Monte Carlo |
| `test_step6_gym_env.py` | Gymnasium `check_env`, roadmap Step 6 checkpoints, exact FCFS reproduction on 6 configs × seeds 0–29, reward bookkeeping, action meanings, determinism, empty episode, invalid usage |

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
│   ├── run_records.py            # run folders, manifests, provenance
│   └── gym_env.py                # Step 6: Gymnasium environment (HospitalSchedulingEnv)
├── agents/
│   ├── heuristics.py             # Step 7: ECT heuristic baseline
│   ├── policy_eval.py            # Step 7: one evaluation routine for all policies
│   └── dqn.py                    # Step 7: DQN (replay buffer, target network, optional Double DQN)
├── tests/                        # 96 tests (see Verification)
├── results/
│   ├── baselines/                # committed FCFS v2 results
│   ├── pilot/                    # Step 7 DQN pilot runs (preliminary)
│   ├── debug/                    # Step 7 DQN debugging runs, rounds 1-2 (compact artefacts)
│   ├── step7_final/              # Step 7 close-out: frozen DQN, test-seed results
│   └── legacy/                   # pre-audit result, not comparable
├── Step_Documentation/           # per-step Word write-ups (Roman Urdu), see note below
├── demo_simulation.py            # one readable run with seed 42
├── run_baseline_fcfs.py          # FCFS baseline runner (--seed / --seeds / --config)
├── check_simple_policies.py      # Step 6 sanity check with simple fixed policies (+ ECT)
├── train_dqn_pilot.py            # Step 7: DQN training with seed splits (--set for one override)
├── run_dqn_experiments.py        # Step 7: runs the pre-registered experiments (one override each)
├── summarize_dqn_experiments.py  # Step 7: pre-registered score per experiment
├── evaluate_dqn_test.py          # Step 7: one-time test-seed evaluation of the frozen DQN
└── requirements.txt
```

**Note on `Step_Documentation/`:** the Step 3 and Step 5 write-ups were
rewritten for the current (v2) design; Steps 6 and 7 have their own write-ups
(Step 7: `Step7_DQN_Baseline.docx`, with the design note and the debugging/close-out
record as Markdown next to it). Steps 1, 2 and 4 start with a
"V2 UPDATE" box listing what changed; their code listings are simplified
versions of the code in `env/`.

## Known limitations

- Workload and deadline values are placeholders, not MIMIC-IV derived.
- Time is discrete (whole steps); network delays are fixed integers.
- Offloading payload is a synthetic unit proportional to compute demand, not bytes.
- The DQN baseline does not beat the ECT heuristic, and its training is not
  stable across training seeds (checkpoint selection on validation seeds is
  needed). FCFS routing ignores priority and deadlines by design.
- In the stressed scenarios the cloud almost never fills up and is free to use
  (see the Step 6 sanity check). Whether to add a cloud usage cost or change
  the capacity is still to be decided with the supervisor; any such change
  requires re-running all baselines and the DQN.
