# Step 7 Design Note — before the DQN baseline

Status: draft for discussion with the supervisor. Nothing in this note has
been applied to the simulator, the reward or the saved Step 5 results.
The Step 6 environment (commit `24d87f6`) is kept as the current verified
implementation.

All numbers below are means over workload seeds 0–29 from
`python check_simple_policies.py --config <scenario>` (reproducible). They are
sanity checks of fixed policies, not thesis results.

## 1. Cloud dominance is scenario-specific

| Scenario | FCFS | always Local | always Edge | always Cloud | random | ECT (proposed) |
|---|---|---|---|---|---|---|
| Default | −15.9 | −23.9 | −39.0 | −52.3 | −34.3 | −15.9 |
| Department imbalance | −78.7 | −701.9 | −219.4 | −130.8 | −109.9 | −80.2 |
| Overload | −720.0 | −1809.2 | −1916.6 | −248.5 | −504.9 | −210.1 |
| Slow network | −428.7 | −385.6 | −666.3 | −881.6 | −507.8 | −253.3 |
| Overdue Critical (artificial) | −93.3 | −103.0 | −115.2 | −50.9 | −78.4 | −26.6 |

Values are the undiscounted episode return = −(total latency) − 10 × (deadline misses).

| Critical deadline violation | FCFS | always Cloud | ECT |
|---|---|---|---|
| Default | 0.0% | 0.0% | 0.0% |
| Department imbalance | 3.2% | 0.0% | 0.0% |
| Overload | 59.6% | 0.0% | 0.7% |
| Slow network | 83.4% | 100.0% | 31.8% |

What these numbers show:

- **The network delay is already in the reward.** Latency = transmission delay
  + queue wait + execution span, so every Cloud task pays its 2-step delay
  (6 in slow network). What the reward does **not** contain is a separate
  usage or communication penalty for offloading.
- **Always-Cloud is the best of the five fixed policies only in overload and in
  the artificial overdue-Critical case.** It is the worst in the default
  workload and in slow network, and FCFS beats it under department imbalance.
- **Always-Cloud is not established as optimal anywhere.** In overload the
  adaptive ECT heuristic (Section 2) beats it (−210.1 vs −248.5) while sending
  only 28.5% of tasks to the cloud (Local 65.3%, Edge 6.3%). Of its Local
  choices, 906 of 2,433 had a strictly smaller estimate than Cloud; the rest
  were ties, resolved towards Local. Only five fixed policies and one heuristic
  were compared, so no policy is shown to be optimal.
- Under overload with always-Cloud, 1 of 3,727 tasks (seeds 0–29) waited at the
  cloud; all others finished after exactly the network delay. With the current
  placeholder capacity (50 compute units/step) the cloud almost never saturates.

## 2. Baselines for Step 7

All baselines are evaluated with the same Step 5 protocol (window + drain,
`drain_and_report`) and the same seeds as the DQN.

| Baseline | Rule | Parameters |
|---|---|---|
| FCFS | Step 3/5 router (demand > 6 → Cloud, else own backlog ≤ 3 → Local, else Edge) | fixed |
| always Local / Edge / Cloud | one action for every task | none |
| random | uniform over the 3 actions, seeded | none |
| **ECT (proposed)** | smallest estimated completion time | none |

**ECT heuristic** ([agents/heuristics.py](../agents/heuristics.py)). For each action *a*:

```
est(a) = delay(a) + ceil((backlog(a) + demand) / capacity(a)) - 1     [steps]
delay: Local 0, Edge = neighbour_network_delay, Cloud = cloud_network_delay
backlog(a): remaining work already assigned to that node, incl. tasks in transit
Edge node: least_loaded_neighbour() — the same node the environment's Edge action uses
ties: Local, then Edge, then Cloud
```

It uses only information available at decision time that is also in the DQN
observation (node backlogs, the task's demand and source) plus fixed config
constants (capacities, delays). It uses no future arrivals and no per-task
queue contents, so the DQN and ECT see the same information. The estimate is
deliberately simple. It assumes the whole backlog is served before the new
task at full capacity, and it ignores later arrivals that may be served first
because they become ready earlier. It has no tunable parameters.

## 3. Possible modelling changes (not applied; for the supervisor's decision)

No change is recommended without a modelling justification. If the cloud's
low cost is judged unrealistic, these are the options, with units and a
sensitivity plan:

| Option | Form | Units | Justification needed | Sensitivity plan |
|---|---|---|---|---|
| A. Offloading communication cost | add `−λ × payload` per Edge/Cloud task; payload = synthetic payload units (SPU) already logged by `metrics.py` | λ in latency-steps per SPU, so the reward stays in "steps" | a source linking data size to delay/energy/money in hospital networks; current SPU is synthetic (∝ compute demand), not bytes | λ ∈ {0, 0.1, 0.25, 0.5}; report where the ranking of policies changes |
| B. Cloud capacity | change `offloading.cloud_capacity` | compute units per step (same unit as department capacities; now 50 vs 12 total at the edge) | a source for realistic cloud vs edge compute ratios | cloud capacity ∈ {12, 25, 50} with all baselines |
| C. Deadline penalty | existing `deadline_miss_penalty` | latency-steps per missed deadline | already a documented design choice (= largest deadline offset) | penalty ∈ {0, 5, 10, 20} |

Network delay sensitivity already exists as the `slow_network` scenario.
Priority weighting (Critical > High > Routine) stays in roadmap Step 10.

## 4. Discount factor γ for per-task decisions and completion rewards

- One environment step = one task decision. Discounting therefore applies **per
  decision**, not per simulator time step. The number of decisions per simulator
  step grows with load, so the same γ discounts simulated time more strongly
  under heavy load.
- A task's reward arrives when it **finishes**, several decisions after its own
  decision. Measured under the ECT policy (seeds 0–29):

  | Scenario | decisions / episode (mean, max) | reward lag in decisions (median, 95th pct.) | γ^p95 for γ = 0.99 | for γ = 0.9 |
  |---|---|---|---|---|
  | Default | 26.1, 42 | 1, 4 | 0.96 | 0.66 |
  | Department imbalance | 65.4, 80 | 5, 11 | 0.90 | 0.31 |
  | Overload | 124.2, 147 | 13, 22 | 0.80 | 0.10 |
  | Slow network | 74.1, 110 | 8, 24 | 0.79 | 0.08 |

  The rewards for a decision often land after earlier rewards caused by other
  tasks. Because nodes are shared, a decision also changes the latency of other
  tasks. A small γ (e.g. 0.9) would discount a typical consequence in overload to
  about 10%, so the agent would mostly ignore it.
- **Pilot choice: γ = 0.99.** Typical consequences keep ≥ 0.79 of their weight,
  while bootstrapping still has some variance reduction. Episodes are finite
  and end with `terminated=True`, so γ = 1 is also valid. The sensitivity set
  for Step 7 is γ ∈ {0.95, 0.99, 1.0}, chosen on validation seeds only.
- **Training objective ≠ evaluation measure.** DQN maximises the expected
  *discounted* return Σ γ^k r_k (a training surrogate). All reported results use
  the *undiscounted* episode return (= −total latency − penalty × misses) and
  the Step 5 protocol metrics (latency, throughput, overall and Critical
  deadline violation). Discounted returns are never reported as performance.
- The observation summarises queues (backlogs only), so the problem is
  partially observable. This is a known limitation of the Step 6 interface.

## 5. Step 7 experimental plan

**Seeds (all disjoint):**

| Use | Workload seeds | Purpose |
|---|---|---|
| Training | drawn per episode from 10,000–19,999 | learning only |
| Validation | 1,000–1,029 | checkpoint selection, γ and hyperparameter choice |
| Test | 2,000–2,029 | final numbers only, used once after all choices are fixed |
| Reference | 0–29 | continuity with the saved Step 5 FCFS results; already inspected, so not used as the unbiased test |

**Training seeds** (network initialisation + exploration): 5 independent seeds
(0–4) per configuration for final results; report mean ± 95% CI across them.
The pilot uses 1 seed.

**Per scenario:** train and evaluate a separate agent per scenario config
(default, department imbalance, overload, slow network); `overdue_critical` and
`empty_workload` are metric checks, not training targets.

**Always reported:**

- learning curves: training episode return (undiscounted) and periodic validation
  return, with the baseline returns as reference lines;
- action proportions (Local / Edge / Cloud) overall, by priority and by source
  department, for the DQN and for every baseline;
- the full Step 5 protocol metrics on the same seeds for the DQN and all baselines.

**How results are judged.** A DQN that mostly chooses one action is **not**
automatically a training failure. If always-X is close to the best available
policy in a scenario (e.g. Cloud in overload), learning mostly X is a valid
outcome. Training success is judged by validation and test return and metrics
relative to all baselines, including always-X and ECT, not by the action
distribution alone.

## 6. Preliminary DQN pilot

Next, a pilot DQN is trained on the current environment, without changing the
model. It exists to debug the training pipeline and see the learning dynamics.
It uses one training seed and a small budget, and reports validation seeds
only. Pilot numbers are preliminary. Thesis claims will be made only after the
modelling decisions in Section 3 and the full protocol in Section 5.
