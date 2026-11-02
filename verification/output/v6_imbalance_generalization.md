# V6 (Phase 0) - department_imbalance, zero-shot generalization (validation seeds 1000-1029)

No model was trained on department_imbalance (each model was trained on one of default / overload / slow network). Imbalance differs from default only in the ICU arrival rates (x6). All 45 checkpoints passed the compatibility check (observation, state, action sizes, agents, capacities). Learned methods: mean ± 95% CI over 5 training seeds; paired differences: per workload seed, averaged over training seeds, t with 29 df (positive = better than the baseline).

| Policy | Return | Paired vs FCFS | Paired vs ECT | Critical viol. | ICU: Local / Edge / Cloud | Queue wait E / ICU / GW |
|---|---|---|---|---|---|---|
| FCFS | -64.3 | — | — | 1.61% | 48 / 42 / 10% | 0.01 / 0.00 / 0.10 |
| ECT | -74.2 | — | — | 0.00% | 67 / 25 / 8% | 0.13 / 0.59 / 0.06 |
| always Cloud | -125.9 | — | — | 0.00% | 0 / 0 / 100% | 0.00 / 0.00 / 0.00 |
| always Local | -626.6 | — | — | 68.86% | 100 / 0 / 0% | 0.06 / 6.59 / 0.45 |
| random | -101.5 | — | — | 3.05% | 31 / 34 / 35% | 0.03 / 0.37 / 0.19 |
| MAPPO W5 (Step 10) — trained on default | -76.4 ± 13.7 | -12.1 ± 8.8 | -2.2 ± 8.3 | 3.69% | 62 / 28.5 ± 30.2 / 10% | 0.15 / 0.18 / 0.23 |
| MAPPO W5 (Step 10) — trained on overload | -65.2 ± 3.4 | -0.9 ± 5.3 | +9.0 ± 4.1 | 0.00% | 55 / 0.0 ± 0.0 / 45% | 0.00 / 0.04 / 0.00 |
| MAPPO W5 (Step 10) — trained on slow_network | -74.0 ± 13.7 | -9.7 ± 6.0 | +0.2 ± 4.7 | 2.95% | 73 / 1.4 ± 2.3 / 26% | 0.02 / 0.48 / 0.03 |
| MAPPO W1 (Step 9) — trained on default | -97.6 ± 26.5 | -33.3 ± 19.9 | -23.4 ± 19.0 | 7.06% | 52 / 45.9 ± 13.4 / 2% | 0.20 / 0.29 / 0.36 |
| MAPPO W1 (Step 9) — trained on overload | -60.3 ± 3.2 | +4.1 ± 4.9 | +13.9 ± 3.4 | 0.00% | 63 / 0.0 ± 0.0 / 37% | 0.00 / 0.07 / 0.01 |
| MAPPO W1 (Step 9) — trained on slow_network | -104.0 ± 41.7 | -39.7 ± 9.7 | -29.8 ± 8.6 | 9.95% | 75 / 10.2 ± 12.1 / 15% | 0.09 / 0.84 / 0.15 |
| DQN (Step 7) — trained on default | -267.4 ± 169.2 | -203.1 ± 56.1 | -193.2 ± 55.7 | 29.25% | 69 / 19.5 ± 9.9 / 12% | 0.13 / 2.77 / 0.43 |
| DQN (Step 7) — trained on overload | -161.2 ± 57.0 | -96.9 ± 21.1 | -87.0 ± 20.0 | 12.23% | 30 / 23.0 ± 33.1 / 47% | 0.12 / 1.25 / 0.26 |
| DQN (Step 7) — trained on slow_network | -221.4 ± 84.2 | -157.1 ± 25.3 | -147.2 ± 24.3 | 28.06% | 56 / 19.9 ± 14.8 / 24% | 0.09 / 2.28 / 0.21 |

Edge feasibility on ICU decisions (ECT estimate, decision-time information; myopic, not ground truth):

- ECT: Edge strictly fastest on 16.4% of ICU decisions; chosen then: {'Edge': 1.0}
- FCFS: Edge strictly fastest on 2.0% of ICU decisions; chosen then: {'Edge': 1.0}
- MAPPO W5 (Step 10) | trained on default: Edge strictly fastest on 10.7% of ICU decisions (mean over 5 seeds)
- MAPPO W5 (Step 10) | trained on overload: Edge strictly fastest on 1.7% of ICU decisions (mean over 5 seeds)
- MAPPO W5 (Step 10) | trained on slow_network: Edge strictly fastest on 21.2% of ICU decisions (mean over 5 seeds)
