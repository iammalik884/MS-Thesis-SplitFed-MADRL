# Verification (supervisor's questions after Step 10)

These scripts answer the supervisor's verification questions after Step 10.
They **do not change** the environment, the rewards, the training code or any
committed result. They only read the saved models and results, and re-run
them. Outputs are in [output/](output/).

Run every script from the repository root, for example
`python verification/v1_independent_actors.py`.

| Script | Question | Seeds | Runtime |
|---|---|---|---|
| [v1_independent_actors.py](v1_independent_actors.py) | Each department has an independent actor; the centralized critic is used only in training | test 2000–2029 (one saved model, re-evaluated) | < 1 min |
| [v2_consolidated_comparison.py](v2_consolidated_comparison.py) | FCFS, ECT, DQN, MAPPO and priority-aware MAPPO on identical test workloads and metrics | test (read from the committed evaluations) | seconds |
| [v3_reproduce.py](v3_reproduce.py) | Reproduction: automated tests, the three test evaluations re-run, one model retrained from scratch | test (re-run of the same frozen models) | ~5–10 min |
| [v4_cloud_sensitivity.py](v4_cloud_sensitivity.py) | Cloud capacity (50 → 25 / 12 / 6) and cloud delay (+2 / +4) sensitivity of the frozen priority-aware MAPPO and the baselines | **validation** 1000–1029 | ~5 min |
| [v5_priority_results.py](v5_priority_results.py) | How the priority weights were chosen; per-priority latency and deadline violations | test (committed; the DQN is re-evaluated) | ~1 min |
| [v6_imbalance_generalization.py](v6_imbalance_generalization.py) | Phase 0: frozen policies on `department_imbalance` (zero-shot); Edge use when the neighbours have spare capacity | **validation** 1000–1029 | ~1–2 min |

**Notes**

- **Seeds and tuning:**
  - Nothing here is tuned. No result is used to choose any setting.
  - V4 and V6 use only validation seeds.
  - The scripts that touch the test seeds re-evaluate the already frozen
    models deterministically, and reproduce the committed numbers exactly
    (V1 and V3 check this).
- **V4 and V6 evaluate frozen models in situations they were not trained
  for.** They show how much the current decisions depend on the large,
  free cloud, and how the policies generalize. They do **not** show how
  models trained under those conditions would perform.
- **Shared code:** `common.py` holds the shared helpers, the 95% t-intervals
  and the output folder.
- **Provenance:** the outputs were produced from the clean commit named in the
  commit message that added them.
