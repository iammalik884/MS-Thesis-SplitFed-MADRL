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
