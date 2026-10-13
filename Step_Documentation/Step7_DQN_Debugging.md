# Step 7 — DQN debugging experiments

Status: Step 7 is **not** complete; Step 8 has not started. The simulator, the
reward definition, the cloud capacity and the Step 6 environment are
unchanged. The pilot in [results/pilot/](../results/pilot/) is kept as the
reference.

This plan was written and committed **before** any experiment was run.

## Plan

**Question.** Why does the pilot DQN get worse during training in overload and
slow network, with Q-values on the wrong scale (design note, Section 6)?

**Reference (E0)** = the pilot settings in `agents/dqn.py`: no scaling,
learning rate 5e-4, γ = 0.99, 40,000 decisions. New code with these settings
must reproduce the pilot files byte-for-byte.

**Phase 1 — one change at a time from E0** (everything else identical):

| Exp. | Single change | What it is |
|---|---|---|
| E1 | `reward_scale = 0.1` | rewards stored for training are multiplied by a **positive constant**. This does not change which policy is optimal, because every return is scaled by the same factor. It is **not** clipping: clipping (e.g. to [−1, 1]) changes the relative size of rewards and can change the optimal policy. No clipping is used. |
| E2 | `reward_scale = 0.01` | as E1, a stronger constant |
| E3 | `obs_scale = 0.1` | the 5 unbounded observation features (backlogs, task demand) are multiplied by 0.1 before the network; bounded features unchanged |
| E4 | `learning_rate = 1e-4` | lower Adam learning rate |

Evaluation always uses the **original, unscaled** rewards and the Step 5
protocol metrics. Scaling affects only what the network is trained on.

**Phase 2 — γ (design note, Section 4).** Starting from the configuration
chosen in Phase 1, only γ changes: γ ∈ {0.95, 0.99, 1.0}.

**Fixed for every run:**

- scenarios: default, overload, slow network;
- training seeds 0 and 1 (2 per setting, to see seed variation);
- 40,000 decisions;
- validation every 4,000 decisions on validation seeds 1,000–1,029;
- training workload seeds 10,000–19,999;
- **test seeds 2,000–2,029 are not used.**

**Saved per run:**

- best and final checkpoints;
- learning and validation curves;
- per-seed results and the full protocol report, incl. offloading
  communication metrics, for the best and final DQN and for every baseline;
- Q estimates vs matching discounted Monte Carlo returns for the best and
  final checkpoints.

**Selection criterion (fixed in advance).** For each run and scenario, the
validation return is normalised as

```
score = (R − R_random) / (R_ECT − R_random)      0 = random baseline, 1 = ECT
```

The chosen setting has the highest mean normalised score of the **final**
checkpoint, averaged over the 3 scenarios and 2 training seeds. The final
checkpoint is used, rather than the best one, so that instability is
penalised. The best-checkpoint score and the best-to-final drop are reported
next to it. If no setting beats E0, E0 stays. Only after a stable setting is
found are multiple training seeds verified, and only with frozen settings is
the reserved test set used.

## Results

(to be filled in after the runs)
