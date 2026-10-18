# Step 10 — Priority-aware reward

Status: the plan below was written and committed **before** any Step 10
experiment was run. The simulator, the cloud capacity, the deadlines and the
evaluation protocol are unchanged. Only the training reward gets an
optional priority weight.

## Reward

Roadmap: *"missing a Critical deadline receives a much larger penalty than
missing a Routine deadline; successful fast processing of urgent tasks
receives stronger positive value; the exact weights are tuned carefully and
reported."*

Each task's Step 6 reward is multiplied by the weight of its priority:

```
r_task = w_priority × ( −latency − 10 × [deadline missed] )
```

- A Critical miss costs `w_C × 10` instead of 10.
- Every step of latency costs an urgent task more, so finishing it fast is
  worth more. All rewards stay ≤ 0, so no positive bonus is added.
- The weights are set in `rl_environment.priority_weights` (optional;
  absent = 1.0 for every priority).
- With the default weights the reward is **exactly** the Steps 6–9 reward. A
  re-run of a Step 9 LONG run with the new code gave byte-identical models,
  curves and per-seed results.
- Code: [env/gym_env.py](../env/gym_env.py) (`_task_reward`), validation in
  [env/config_validation.py](../env/config_validation.py), training option
  `--set priority_weights=C/H/R` in [train_mappo.py](../train_mappo.py).
- Tests: [tests/test_step10_priority_reward.py](../tests/test_step10_priority_reward.py).

**Candidate weights** are derived from the deadline offsets (Critical 2, High
5, Routine 10 steps), not guessed:

| Name | Critical / High / Routine | Rule |
|---|---|---|
| W1 | 1 / 1 / 1 | no priority weighting (= Step 9) |
| W5 | 5 / 2 / 1 | urgency = largest offset / own offset (10/2, 10/5, 10/10) |
| W25 | 25 / 4 / 1 | urgency squared (a much stronger preference) |

## Plan

**Base.** The frozen Step 9 MAPPO setting (`MAPPOConfig` defaults, 120,000
decisions). Only the reward weights change.

**Round 1.**

- Settings: W1, W5 and W25.
- 3 scenarios × training seeds 0 and 1.
- Validation every 4,000 decisions on validation seeds 1,000–1,029.
- **Test seeds are not used.**
- Each run picks its best checkpoint on the validation return of the reward
  it was trained on.
- All reported returns and scores use the original, unweighted reward, so
  they compare directly with Steps 7–9.

**Selection rule (fixed now).**

- **Primary metric:** the Critical deadline violation rate of the final
  checkpoint on the validation seeds. It is pooled over the seeds of a run
  (late + overdue Critical tasks / decided Critical tasks), then averaged
  over the 6 runs of a setting.
- A weighted setting (W5 or W25) is chosen only if its rate is at least
  **0.5 percentage points lower than W1's**.
- If both qualify, the lower rate wins. If they are within 0.5 points of each
  other, the smaller weights (W5) win.
- If neither qualifies, W1 stays. The result is then reported as
  "priority weighting did not reduce Critical violations in this environment".

**Trade-off, reported but not used for selection:**

- unweighted return and score (0 = random, 1 = ECT);
- overall, High and Routine violation rates;
- average latency per priority.

**Close-out** (runs once):

- Train the chosen setting with training seeds 0–4 on the 3 scenarios.
- Evaluate it once on test seeds 2,000–2,029.
- Re-evaluate the frozen Step 9 MAPPO on the same test seeds, to get its
  per-priority numbers. This uses the same saved models, deterministically,
  and is not a new tuning step.
- Report all baselines as before.
- If W1 is chosen, the Step 9 close-out is the Step 10 result and no new
  training is done.

**Roadmap checkpoint:** *when resources are limited, does the policy reduce
Critical deadline violations even if some Routine tasks wait longer?*

It is answered with the Critical violation rate and the per-priority average
latency of the chosen setting against Step 9 (W1), in the overload and
slow-network scenarios.

## Round 1 — results (validation seeds only)

All 18 runs come from the clean commit `20537fd`. **Test seeds were not
used.** Compact artefacts are in [results/debug/prio1/](../results/debug/prio1/).
The six W1 runs are byte-identical to the Step 9 LONG runs (models, curves,
per-seed results), so W1 is exactly Step 9.

Final checkpoint, deadline violation rate per priority, pooled per run and
averaged over 3 scenarios × 2 training seeds:

| Setting | Critical | High | Routine | Mean final score (unweighted) |
|---|---|---|---|---|
| W1 (1/1/1) | 3.380% | 0.203% | 0.000% | 1.207 |
| **W5 (5/2/1)** | **0.141%** | 0.061% | 0.000% | 1.204 |
| W25 (25/4/1) | 0.246% | 1.419% | 0.000% | 1.089 |

Slow network, per run (seed 0 / seed 1). The violation rates and average
latencies are in steps, final checkpoint:

| Setting | Critical violation | Critical latency | Routine latency |
|---|---|---|---|
| W1 | 13.29% / 6.33% | 1.14 / 0.88 | 2.81 / 2.87 |
| W5 | 0.21% / 0.63% | 0.35 / 0.29 | 4.38 / 4.87 |
| W25 | 1.27% / 0.21% | 0.49 / 0.24 | 3.87 / 5.92 |

**Selection (rule fixed before the runs).**

- W5 and W25 both lower the Critical rate by more than 0.5 percentage points
  against W1 (by 3.24 and 3.13 points).
- They differ from each other by 0.105 points, which is less than 0.5, so the
  smaller weights win.
- **Frozen Step 10 setting: W5 (Critical 5, High 2, Routine 1)**, Step 9 MAPPO
  otherwise unchanged (120,000 decisions).

**Observations.**

- **W5 trades Routine latency for Critical deadlines, as the roadmap asks.**
  In the slow network, Critical violations fall from 13.29% / 6.33% to
  0.21% / 0.63%. Routine latency rises from about 2.8 to 4.4–4.9 steps, but
  Routine still has no violations. The unweighted score stays almost the same
  (1.204 vs 1.207).
- **W25 goes too far.** High-priority violations rise (1.419% vs 0.203%), the
  unweighted score drops (1.089), and the default-scenario score falls below
  ECT (0.83–0.89).
- In one W25 run (slow network, seed 0) the critic is far off the Monte Carlo
  return (mean bias −362.5, in units of the weighted reward it is trained on).
  W25 is not used further.

## Close-out — results (test seeds, evaluated once)

**Training.** W5 was trained with training seeds 0–4 on the 3 scenarios
(15 runs) at the clean commit `672f628`. Seeds 0 and 1 are byte-identical to
the Round 1 W5 runs. On the validation seeds the mean final score is 1.212
(drop 0.010).

**Test evaluation.** This command was run once, at commit `da022d9`, on test
seeds 2,000–2,029:

```
python evaluate_dqn_test.py --runs results/step10_final/training_runs --out results/step10_final
```

As planned, the frozen Step 9 MAPPO was re-evaluated with the same command
(`--runs results/step9_final/training_runs --out results/step10_final/step9_reference`)
to get its per-priority numbers. Its `per_seed.csv`, table, figure and
results are identical to [results/step9_final/](../results/step9_final/).

All values below are means ± 95% CI over the 5 training seeds, best
checkpoint, test seeds. "Violation" is pooled over the 30 test seeds of a
run: late + overdue tasks / decided tasks of that priority. This is
slightly different from the headline `critical_deadline_violation_rate`
(the mean of per-seed rates, e.g. 5.9% for Step 9 in the slow network).
Latency is the mean over completed tasks, in steps. No task was left
unfinished.

**Slow network** (where Critical deadlines are at risk):

| Priority | Step 9 MAPPO (W1): violation | latency | **Step 10 MAPPO (W5): violation** | **latency** | ECT: violation / latency |
|---|---|---|---|---|---|
| Critical | 6.78% ± 5.96 | 0.87 ± 0.21 | **0.22% ± 0.19** | **0.35 ± 0.06** | 31.53% / 1.80 |
| High | 0.49% ± 0.71 | 1.23 ± 0.27 | 0.15% ± 0.20 | 0.58 ± 0.07 | 3.31% / 2.23 |
| Routine | 0.00% | 3.05 ± 0.06 | 0.00% | 4.15 ± 0.46 | 0.00% / 3.21 |

**Overload:** Critical violations are 0.00% for both. Critical latency falls
from 0.57 ± 0.07 to 0.34 ± 0.02, and Routine latency rises from 1.78 ± 0.07
to 1.98 ± 0.04.

**Default:** no violations for either. Critical latency 0.15 → 0.12, and
Routine latency 1.05 → 1.12.

**Unweighted return** (comparable with Steps 7–9):

| Policy | Default | Overload | Slow network |
|---|---|---|---|
| **Step 10 MAPPO (W5)** | −15.1 ± 0.4 | −142.9 ± 2.7 | **−151.1 ± 11.3** |
| Step 9 MAPPO (W1) | **−14.8 ± 0.4** | **−141.6 ± 4.8** | −153.6 ± 19.8 |
| ECT | −15.5 | −215.2 | −246.0 |
| FCFS | −15.7 | −772.7 | −409.6 |

**Roadmap checkpoint — when resources are limited, does the policy reduce
Critical deadline violations even if some Routine tasks wait longer? Yes.**

- In the slow network, W5 cuts the Critical violation rate from 6.78% to
  0.22% and the Critical latency from 0.87 to 0.35 steps.
- In exchange, Routine tasks wait longer: 3.05 → 4.15 steps, with still no
  Routine violation.
- Under overload, W5 has the same pattern on latency (Critical faster,
  Routine slower), and both settings have no violations.
- The unweighted return of W5 and Step 9 is about the same: their 95%
  confidence intervals overlap in all three scenarios.
- W5 has a better mean return than ECT and FCFS in every scenario. In the
  default scenario the margin over ECT is very small: −15.1 ± 0.4 vs −15.5,
  and one of the 5 training seeds equals ECT exactly (−15.47).

**Costs, reported honestly.**

- W5 uses the cloud more: Cloud share in the slow network 27.5% vs 14.1%,
  and under overload 48.5% vs 42.7%.
- Its offloading communication is higher: synthetic payload 112.7 vs 59.5
  units in the slow network, and 284.1 vs 250.7 under overload.
- Edge is still almost unused (0.0–1.9%).
- The priority-aware policy therefore relies even more on the cost-free
  cloud. The open modelling question about a cloud usage cost (supervisor)
  matters even more here.

**Conclusion of Step 10.** A priority-weighted team reward with weights from
the deadline offsets (W5 = 5/2/1) was chosen by the pre-registered rule. On
the test seeds it almost removes Critical deadline violations where they
occurred, accepts longer Routine waits, and keeps the overall return. This
is the healthcare-specific contribution the roadmap asks for. Privacy
(SplitFed) is the next part of the roadmap.
