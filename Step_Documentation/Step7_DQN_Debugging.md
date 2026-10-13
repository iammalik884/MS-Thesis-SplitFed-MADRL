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

All runs are on validation seeds 1,000–1,029. **Test seeds were not used.**
The compact artefacts of every run (manifest, summary, per-seed results incl.
offloading metrics, learning/validation curves, best and final checkpoints)
are in [results/debug/](../results/debug/). Reproduce the tables with
`python summarize_dqn_experiments.py results/debug/phase1` (or `phase2`).
The full per-seed protocol reports (~155 MB) are kept locally in
`results/runs/` (git-ignored). The same commands regenerate them, because
runs are deterministic.

**Provenance.**

- Phase 1 runs come from commits `78e8474` and `d7abd02`; Phase 2 from `cbe84ed`.
  All are clean trees, and the agent, training and environment code is identical
  across them (only the experiment runner changed).
- 10 of the first 30 Phase 1 runs failed with `MemoryError`: 10 parallel runs ×
  ~300 MB did not fit in this PC's free RAM. They were re-run one at a time.
  Nothing else changed.
- E0 seed 0 reproduced the pilot (−20.4 / −340.7 / −478.3).

### Phase 1 — one change at a time from E0

Score: 0 = random baseline, 1 = ECT. Each score is a mean over 3 scenarios × 2
training seeds.

| Exp. | Change | Mean final score | Mean best score | Best → final drop |
|---|---|---|---|---|
| E0 | — (pilot settings) | −1.879 | 0.457 | 2.337 |
| E1 | reward × 0.1 | −1.831 | −0.120 | 1.711 |
| **E2** | **reward × 0.01** | **−1.291** | 0.130 | 1.421 |
| E3 | unbounded obs × 0.1 | −2.685 | 0.273 | 2.958 |
| E4 | learning rate 1e-4 | −3.792 | 0.254 | 4.046 |

E2 is selected by the pre-registered rule. It only reduces the instability; it
does not remove it.

### Phase 2 — γ, starting from E2 (reward × 0.01)

| γ | Mean final score | Mean best score | Best → final drop |
|---|---|---|---|
| **0.95** | **−0.565** | 0.306 | 0.871 |
| 0.99 (= E2) | −1.291 | 0.130 | 1.421 |
| 1.0 | −1.640 | 0.015 | 1.655 |

Per scenario, with γ = 0.95 (final checkpoint; seed 0 / seed 1):

- default: score 0.84 / 0.80 (return −17.5 / −18.1; ECT −14.6);
- overload: −4.16 / −0.15;
- slow network: −0.79 / 0.08.

### Q estimates vs discounted Monte Carlo returns

Means over both training seeds, greedy policy, in unscaled latency-steps:

| Setting | Overload: Q vs MC (best ckpt) | Overload: Q vs MC (final ckpt) | Default: Q vs MC (final) |
|---|---|---|---|
| E0 | 0.0 vs −178.6 | −6.5 vs −1014.4 | −12.0 vs −12.4 |
| E2 (reward × 0.01) | +34.3 vs −310.6 | +83.1 vs −921.7 | −10.1 vs −11.7 |
| γ = 0.95 (+ E2) | −5.3 vs −78.5 | −15.3 vs −162.1 | −7.2 vs −7.8 |
| γ = 1.0 (+ E2) | +56.0 vs −580.3 | +58.8 vs −1752.5 | −10.9 vs −12.4 |

- In the **default** scenario the DQN learns the value scale (Q ≈ MC).
- In **overload** (and slow network) Q is far too high. In several settings it is
  even **positive**, which is impossible because every reward is ≤ 0. This is the
  pattern of DQN's max-operator **overestimation**.
- γ = 0.95 gives the smallest bias and the highest Q–MC correlation (0.68 best /
  0.85 final in overload), but the bias is still large.

### Reward timing (measurement, environment unchanged)

On validation seeds under overload, the share of the episode return paid at the
**last** step was:

| Policy | Share of return at the last step |
|---|---|
| always Local | 73% |
| FCFS | 46% |
| ECT | 16% |
| always Cloud | 15% |

The worse the policy, the more of its cost arrives in one rare terminal
transition (drain completions and unfinished tasks), about once per 124
decisions. This is how the Step 6 reward is defined. It is not changed here, but
it helps explain why the value of congested states is learned so poorly.

### Conclusion of this round

- **No stable configuration was found.** The best setting by the
  pre-registered rule is reward × 0.01 with γ = 0.95, and its mean final score is
  still below the random baseline (−0.565), with overload seed 0 at −4.16.
- Following the plan, there was **no multi-seed verification and no test-set
  evaluation**. Step 7 stays open, and Step 8 has not started.
- γ = 0.95 contradicts the design note's preference for γ = 0.99, which was
  argued from reward delays. The data favours 0.95 for stability with the
  current training setup. Which γ to use is still an open choice; nothing is
  frozen yet.
- **No DQN checkpoint beats ECT in any scenario.** Some best checkpoints beat
  FCFS: in overload, e.g. E0 seed 0 scored −340.7 against FCFS −691.2; in slow
  network, E0 seed 1 scored −367.3 against −416.6. All 14 overload final
  checkpoints (7 settings × 2 seeds) score below random (score < 0).

### Proposed next round (one change at a time, from reward × 0.01 and γ = 0.95)

1. **Double DQN** targets: Q-values above the possible maximum (0) point to
   overestimation.
2. A longer training budget (e.g. 120,000 decisions), to see whether the drop
   comes from training length.
3. n-step returns (e.g. n = 5), for the delayed completion rewards.

The same pre-registered score and rules would apply. The environment, the
reward definition and the cloud capacity stay unchanged.

## Round 2 and Step 7 close-out — plan (committed before these runs)

Base for Round 2 = the Round 1 choice: `reward_scale = 0.01`, `γ = 0.95`.
Everything else is fixed as in Round 1: 3 scenarios, training seeds 0 and 1,
validation seeds only, environment / reward / cloud unchanged.

| Exp. | Single change from the base | Compared with |
|---|---|---|
| R2a `DDQN` | `double_dqn = 1` (Double DQN target, van Hasselt et al. 2016) | G095 (Round 1) |
| R2b `LONG` | 120,000 instead of 40,000 decisions, applied to the better of G095 / DDQN by the score | that setting at 40,000 |

**Stable** (defined now, before the runs) means both of these hold for a setting:

- its final-checkpoint score is ≥ 0 (not worse than random) in all 6 runs;
- its mean best → final drop is ≤ 0.25.

**Frozen Step 7 configuration.** The tested setting with the highest mean
final-checkpoint score, over all settings of Round 1 and Round 2.

**Close-out protocol.** It runs once, whether or not the frozen setting is
stable, so that Step 7 ends with an honest DQN baseline instead of a tuned
"winner":

- Train the frozen setting with training seeds 0–4 on each of the 3 scenarios.
- The checkpoint is chosen on the validation seeds (best validation return); the
  final checkpoint is reported too.
- Evaluate once on the **test seeds 2,000–2,029**: the DQN (5 training seeds)
  and every baseline (FCFS, always Local / Edge / Cloud, random, ECT).
- Report the mean ± 95% CI over the 5 training seeds (t-distribution, 4 degrees
  of freedom); per-seed results; full reports incl. offloading communication;
  action shares; learning curves.
- If the setting is not stable, Step 7 says so explicitly. The result is then
  "DQN as the single-agent baseline in this environment", not a claim that DQN
  works well. No further tuning is done after the test seeds have been used.
