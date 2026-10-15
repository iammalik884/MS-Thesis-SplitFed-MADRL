# Step 9 — MAPPO with CTDE

Status: the plan below was written and committed **before** any MAPPO
experiment was run. The simulator, the reward definition, the cloud capacity
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
