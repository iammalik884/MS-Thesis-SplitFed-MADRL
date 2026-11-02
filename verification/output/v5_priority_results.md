# V5 - Priority weights and per-priority results (test seeds 2000-2029)

Weights from the deadline offsets (Critical 2, High 5, Routine 10 steps): W1 = 1/1/1, W5 = 5/2/1 (urgency = 10 / offset), W25 = 25/4/1 (urgency squared). Reward per task = w_p x (-latency - 10 x miss). W5 was chosen on VALIDATION seeds by a rule committed before the runs (mean Critical violation: W1 3.380%, W5 0.141%, W25 0.246%; see Step_Documentation/Step10_Priority_Reward.md).

Violation = (late + overdue) / decided tasks of that priority, pooled over the 30 test seeds of a run; latency in steps over completed tasks. Learned methods: mean ± 95% CI over 5 training seeds.

## default

| Method | Critical: violation / latency | High: violation / latency | Routine: violation / latency |
|---|---|---|---|
| Priority MAPPO (W5) | 0.00% ± 0.00 / 0.12 ± 0.01 | 0.00% ± 0.00 / 0.24 ± 0.01 | 0.00% ± 0.00 / 1.12 ± 0.05 |
| MAPPO (W1, Step 9) | 0.00% ± 0.00 / 0.15 ± 0.02 | 0.00% ± 0.00 / 0.26 ± 0.02 | 0.00% ± 0.00 / 1.05 ± 0.01 |
| DQN (Step 7) | 0.11% ± 0.30 / 0.33 ± 0.03 | 0.00% ± 0.00 / 0.44 ± 0.05 | 0.00% ± 0.00 / 1.28 ± 0.05 |
| ECT | 0.00% / 0.18 | 0.00% / 0.31 | 0.00% / 1.06 |
| FCFS | 0.00% / 0.12 | 0.00% / 0.26 | 0.00% / 1.15 |
| always Cloud | 0.00% / 2.00 | 0.00% / 2.00 | 0.00% / 2.00 |

## overload

| Method | Critical: violation / latency | High: violation / latency | Routine: violation / latency |
|---|---|---|---|
| Priority MAPPO (W5) | 0.00% ± 0.00 / 0.34 ± 0.02 | 0.00% ± 0.00 / 0.67 ± 0.04 | 0.00% ± 0.00 / 1.98 ± 0.04 |
| MAPPO (W1, Step 9) | 0.00% ± 0.00 / 0.57 ± 0.07 | 0.00% ± 0.00 / 0.72 ± 0.07 | 0.00% ± 0.00 / 1.78 ± 0.07 |
| DQN (Step 7) | 5.90% ± 6.43 / 1.67 ± 0.46 | 1.56% ± 2.21 / 1.91 ± 0.41 | 0.71% ± 0.83 / 2.22 ± 0.36 |
| ECT | 1.25% / 1.47 | 0.00% / 1.60 | 0.00% / 1.86 |
| FCFS | 65.53% / 3.95 | 23.05% / 3.92 | 1.09% / 3.43 |
| always Cloud | 0.23% / 2.00 | 0.00% / 2.00 | 0.00% / 2.00 |

## slow_network

| Method | Critical: violation / latency | High: violation / latency | Routine: violation / latency |
|---|---|---|---|
| Priority MAPPO (W5) | 0.22% ± 0.19 / 0.35 ± 0.06 | 0.15% ± 0.20 / 0.58 ± 0.07 | 0.00% ± 0.00 / 4.15 ± 0.46 |
| MAPPO (W1, Step 9) | 6.78% ± 5.96 / 0.87 ± 0.21 | 0.49% ± 0.71 / 1.23 ± 0.27 | 0.00% ± 0.00 / 3.05 ± 0.06 |
| DQN (Step 7) | 54.95% ± 14.25 / 2.91 ± 0.55 | 23.63% ± 2.70 / 3.07 ± 0.39 | 2.14% ± 2.17 / 4.11 ± 0.22 |
| ECT | 31.53% / 1.80 | 3.31% / 2.23 | 0.00% / 3.21 |
| FCFS | 85.31% / 3.01 | 2.70% / 3.23 | 0.00% / 4.38 |
| always Cloud | 100.00% / 6.00 | 100.00% / 6.00 | 0.00% / 6.00 |
