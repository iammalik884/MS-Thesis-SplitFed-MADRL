# Step 9 — MAPPO with CTDE

Status: **Step 9 is complete** (see "Close-out — results" at the end). The
plan below was written and committed **before** any MAPPO experiment was run. The simulator, the reward definition, the cloud capacity
and the Step 6/8 environments are unchanged. Privacy layers are not part of
this step (roadmap: "first prove that cooperation works").

## Method

- **Actors (decentralized execution):** one actor network per department
  (Emergency, ICU, General Ward), no parameter sharing. Each actor sees only
  its Step 8 local observation (10 values) and picks Local / Edge / Cloud
  for its own department's tasks.
- **Centralized critic (training only):** V(global state), with the global
  state = the 13-value Step 6 observation (all loads, the pending task, its
  source department, time).
- **Cooperative objective:** the team reward (sum over agents = the Step 6
  reward) for every decision. During training it is multiplied by the
  positive constant `reward_scale = 0.01`, the same as the frozen DQN.
  This is not clipping, and all reported returns are in original units.
- **Learning:** decisions are taken in turns (Step 8), so an episode is one
  sequence of decisions. GAE (λ) over this sequence with the centralized
  critic. Each actor gets a PPO clipped update on its own decisions only, with
  advantages normalised per actor and an entropy bonus. The critic uses MSE
  on the GAE returns. No bootstrap at episode end.
- **Evaluation:** greedy (most likely) action of the acting department's
  actor, through the same `agents/policy_eval.evaluate()` as the DQN and all
  baselines, i.e. the same Step 5 protocol metrics and undiscounted return.
- Code: [agents/mappo.py](../agents/mappo.py), [train_mappo.py](../train_mappo.py);
  tests in [tests/test_mappo.py](../tests/test_mappo.py).

**Reference setting M0** (`MAPPOConfig` defaults, commonly used PPO values,
not tuned):

| Parameter | Value |
|---|---|
| γ / GAE λ | 0.99 / 0.95 |
| PPO clip ε | 0.2 |
| actor / critic learning rate (Adam) | 5e-4 / 5e-4 |
| rollout per update | ≥ 1,000 decisions (whole episodes; the last, shorter rollout is used too) |
| epochs per update / minibatch | 10 / 250 |
| entropy coefficient | 0.01 |
| hidden layers | 2 × 64 (tanh), actors and critic |
| gradient-norm clip | 10 |
| reward scale | 0.01 |
| training budget | 40,000 decisions (same as the frozen DQN) |

## Plan (same rules as Step 7)

**Fixed for every run:**

- scenarios: default, overload, slow network;
- training workload seeds 10,000–19,999, from the same seed stream as the DQN;
- validation every 4,000 decisions on validation seeds 1,000–1,029;
- **test seeds 2,000–2,029 are not used before the close-out.**

**Round 1** (training seeds 0 and 1):

| Exp. | Change from M0 |
|---|---|
| M0 | — |
| M1 | `gamma = 0.95` (the γ chosen for the DQN in Step 7) |

**Round 2** (training seeds 0 and 1):

| Exp. | Change |
|---|---|
| LONG | 120,000 decisions instead of 40,000, applied to the better Round 1 setting |

The DQN was given the same option in Step 7, so the comparison stays fair.

**Score, selection and stability** (as in Step 7):

- Score: `score = (R − R_random) / (R_ECT − R_random)` on the validation seeds.
- Selection: the setting with the highest mean **final**-checkpoint score
  over 3 scenarios × 2 training seeds becomes the frozen setting.
- Stable: final score ≥ 0 in all 6 runs **and** a mean best → final drop
  ≤ 0.25.

**Close-out** (runs once, stable or not):

- Train the frozen setting with training seeds 0–4 on each of the 3 scenarios.
- Choose the best checkpoint on the validation seeds; report the final one
  too.
- Evaluate once on test seeds 2,000–2,029, against the Step 7 DQN results and
  every baseline on the same test seeds.
- Report the mean ± 95% CI over the training seeds (t, 4 degrees of freedom).
- No tuning after that.

**Roadmap checkpoints, and how they are measured:**

1. *Does joint reward improve?* The team return on the validation seeds over
   training (validation curve), against the untrained start and the
   baselines.
2. *Do agents use spare capacity in other departments rather than overload
   one node?* For each policy, the tasks and the mean queue wait per
   destination node (`destination_load` in each run's `summary.json`), and
   the Edge share. This is compared with FCFS and always Local.

**Also saved per run:**

- best and final checkpoints;
- learning and validation curves;
- per-seed results and full reports, including offloading communication;
- critic V vs the discounted Monte Carlo team return.

## Round 1 and Round 2 — results (validation seeds only)

All runs come from the clean commit `608debe`. **Test seeds were not used.**
Compact artefacts are in [results/debug/mappo1/](../results/debug/mappo1/) and
[results/debug/mappo2/](../results/debug/mappo2/). Reproduce the table with
`python summarize_dqn_experiments.py results/debug/mappo1` (or `mappo2`).

| Setting | Decisions | Mean final score | Mean best score | Best → final drop |
|---|---|---|---|---|
| M0 (defaults, γ = 0.99) | 40,000 | 1.036 | 1.096 | 0.060 |
| M1 (γ = 0.95) | 40,000 | 1.032 | 1.081 | 0.049 |
| **LONG** (M0, longer run) | 120,000 | **1.207** | 1.224 | **0.017** |

Score: 0 = random, 1 = ECT. Per run, final-checkpoint score (seed 0 / seed 1):

| Setting | Default | Overload | Slow network |
|---|---|---|---|
| M0 | 1.01 / 1.04 | 0.97 / 1.12 | 0.96 / 1.11 |
| M1 | 1.02 / 1.04 | 1.20 / 1.17 | 0.70 / 1.06 |
| LONG | 1.03 / 1.03 | 1.26 / 1.24 | 1.30 / 1.38 |

- All three settings meet the pre-registered stability rule: every final
  score is ≥ 0, and every drop is ≤ 0.25.
- M0 was ahead of M1 by 0.004, so Round 2 used M0.
- The critic is close to the Monte Carlo returns, e.g. LONG final overload:
  V −57.5 vs MC −56.2 (unscaled means). There is no sign of the DQN's
  overestimation.

**Frozen Step 9 configuration** (highest mean final score): **LONG** =
`MAPPOConfig` defaults (γ = 0.99, reward × 0.01), 120,000 decisions.

Note for the comparison: the frozen DQN uses 40,000 decisions. The DQN was
given the same 120,000-decision option in Step 7, but scored worse with it
(mean final 0.034), so its own rule kept 40,000. Both budgets are reported
next to the results.

**First observation (validation, overload):** MAPPO sends most large
Routine tasks to the cloud and keeps Critical and High tasks on the local
node. This keeps every department's queue short, and the deadline violations
were 0%. MAPPO hardly uses the Edge action (0–2% of final-checkpoint decisions in overload, all 6 runs);
it balances load through the cloud, not through the neighbour departments.
This fits the Step 6/7 finding that the cloud has no usage cost and almost
never fills. Whether a cloud cost should be added is still the open
modelling question for the supervisor.

## Close-out — results (test seeds, evaluated once)

**Training.** The frozen setting (LONG) was trained with training seeds 0–4
on each of the 3 scenarios (15 runs) at the clean commit `58c8914`. Seven
runs finished in the first attempt. The other eight stopped when the PC
session ended (Windows process-termination codes, not a Python error) and
were re-run with `--only-missing` from the same commit and a clean tree.
Seeds 0 and 1 are byte-identical to the Round 2 LONG runs.

On the validation seeds the 15 runs have mean final score 1.211, mean best
score 1.226 and drop 0.015. Every final score is ≥ 1.0; the lowest, default seed 2, equals ECT's validation return exactly. Compact artefacts:
[results/step9_final/training_runs/](../results/step9_final/training_runs/).

**Test evaluation.** `python evaluate_dqn_test.py --runs results/step9_final/training_runs --out results/step9_final`
was run once, at commit `3447356`, on test seeds 2,000–2,029. Results are in
[results/step9_final/](../results/step9_final/). The baseline results there
are identical to the Step 7 test evaluation. MAPPO and DQN values are the
mean ± 95% CI over 5 training seeds; the DQN values are from
[results/step7_final/](../results/step7_final/).

Undiscounted return on the test seeds (higher is better):

| Policy | Default | Overload | Slow network |
|---|---|---|---|
| **MAPPO (best checkpoint)**, 120k decisions | **−14.8 ± 0.4** | **−141.6 ± 4.8** | **−153.6 ± 19.8** |
| MAPPO (final checkpoint) | −14.8 ± 0.4 | −143.6 ± 4.9 | −156.5 ± 17.7 |
| DQN (best checkpoint), 40k decisions | −20.1 ± 0.7 | −278.6 ± 71.5 | −411.9 ± 49.3 |
| ECT | −15.5 | −215.2 | −246.0 |
| FCFS | −15.7 | −772.7 | −409.6 |
| always Local | −19.9 | −1,932.2 | −402.9 |
| always Cloud | −52.7 | −253.5 | −868.7 |
| random | −34.7 | −555.2 | −492.6 |

Critical-task deadline violation rate (test seeds):

| Policy | Default | Overload | Slow network |
|---|---|---|---|
| MAPPO (best) | 0.0% | 0.0% | 5.9% ± 5.3 pp |
| DQN (best) | 0.1% ± 0.2 pp | 5.7% ± 6.3 pp | 55.0% ± 15.0 pp |
| ECT | 0.0% | 1.2% | 30.8% |
| FCFS | 0.0% | 64.8% | 84.0% |

**Roadmap checkpoint 1 — does the joint reward improve?** Yes. The team
return on the validation seeds rises during training in all three scenarios
and passes ECT (`learning_curves.png`). On the test seeds MAPPO has the best
return in every scenario. In the default scenario the margin over ECT is
small (−14.8 ± 0.4 vs −15.5); under overload and in the slow network it is
large. The final checkpoints are almost as good as the validation-selected
ones, so training is stable.

**Roadmap checkpoint 2 — spare capacity instead of overloading one node?**
Partly. MAPPO overloads no node: under overload, each department's mean queue
wait is 0.09–0.11 steps, against 0.64–0.91 for ECT and up to 3.54 for FCFS
(General Ward). But it does this through the **cloud**, not through the
neighbour departments:

- under overload its Edge share is 0.0% (43% Cloud);
- in the default scenario Edge is 3.3%, and in the slow network 1.7%.

As a consequence, its offloading communication under overload is higher than
ECT's (synthetic payload 250.7 vs 201.0 units per episode). In this
environment the cloud has no usage cost and almost never fills (Step 6/7
finding). Whether MAPPO would use the neighbours if cloud use had a cost is
the open modelling question for the supervisor.

**Conclusion of Step 9.** Cooperative MAPPO with CTDE was trained and frozen
by the pre-registered rules, and evaluated once on the test seeds.

- It beats every baseline and the DQN in all three scenarios.
- Its Critical deadline violation rate is 0.0% in the default and overload
  scenarios and 5.9% in the slow network (ECT: 30.8%).
- It is stable across training seeds.

Two caveats are reported next to the result:

- MAPPO trained for 120,000 decisions, the DQN for 40,000. Both had the same
  options, and each budget was chosen by its own rule.
- The load balancing relies on the free cloud.

Priority weighting is roadmap Step 10, and privacy (SplitFed) follows later.
