# V2 - Consolidated comparison (test seeds 2000-2029, best checkpoints)

Baseline results are identical in the Step 7, 9 and 10 evaluations (checked). Learned methods: mean ± 95% CI over 5 training seeds (t, df 4). Throughput = tasks completed per step in the window.

## default

| Method | Return | Avg latency | Critical violation | All-task violation | Throughput | Local / Edge / Cloud | Payload units |
|---|---|---|---|---|---|---|---|
| FCFS | -15.7 | 0.58 | 0.0% | 0.0% | 1.28 | 78 / 9 / 13% | 30.6 |
| ECT | -15.5 | 0.57 | 0.0% | 0.0% | 1.28 | 93 / 7 / 1% | 8.33 |
| always Cloud | -52.7 | 2.00 | 0.0% | 0.0% | 1.19 | 0 / 0 / 100% | 88.6 |
| DQN (Step 7, 40k decisions) | -20.1 ± 0.7 | 0.74 ± 0.02 | 0.1% ± 0.2 | 0.0% ± 0.1 | 1.27 ± 0.01 | 81 / 18 / 1% | 23.9 ± 3.7 |
| MAPPO (Step 9, 120k) | -14.8 ± 0.4 | 0.55 ± 0.02 | 0.0% ± 0.0 | 0.0% ± 0.0 | 1.28 ± 0.00 | 90 / 3 / 7% | 14.4 ± 4.4 |
| Priority MAPPO W5 (Step 10, 120k) | -15.1 ± 0.4 | 0.56 ± 0.02 | 0.0% ± 0.0 | 0.0% ± 0.0 | 1.28 ± 0.00 | 86 / 2 / 12% | 21.9 ± 3.2 |

## overload

| Method | Return | Avg latency | Critical violation | All-task violation | Throughput | Local / Edge / Cloud | Payload units |
|---|---|---|---|---|---|---|---|
| FCFS | -772.7 | 3.67 | 64.8% | 23.3% | 4.83 | 5 / 81 / 14% | 405.6 |
| ECT | -215.2 | 1.67 | 1.2% | 0.3% | 5.84 | 65 / 6 / 29% | 201.0 |
| always Cloud | -253.5 | 2.00 | 0.2% | 0.0% | 5.77 | 0 / 0 / 100% | 422.1 |
| DQN (Step 7, 40k decisions) | -278.6 ± 71.5 | 1.98 ± 0.34 | 5.7% ± 6.3 | 2.2% ± 2.5 | 5.76 ± 0.10 | 21 / 13 / 65% | 337.5 ± 47.1 |
| MAPPO (Step 9, 120k) | -141.6 ± 4.8 | 1.12 ± 0.04 | 0.0% ± 0.0 | 0.0% ± 0.0 | 6.01 ± 0.02 | 57 / 0 / 43% | 250.7 ± 10.4 |
| Priority MAPPO W5 (Step 10, 120k) | -142.9 ± 2.7 | 1.13 ± 0.02 | 0.0% ± 0.0 | 0.0% ± 0.0 | 6.00 ± 0.01 | 51 / 0 / 49% | 284.1 ± 6.3 |

## slow_network

| Method | Return | Avg latency | Critical violation | All-task violation | Throughput | Local / Edge / Cloud | Payload units |
|---|---|---|---|---|---|---|---|
| FCFS | -409.6 | 3.64 | 84.0% | 18.4% | 2.93 | 12 / 73 / 15% | 227.1 |
| ECT | -246.0 | 2.52 | 30.8% | 7.5% | 3.13 | 90 / 8 / 2% | 32.9 |
| always Cloud | -868.7 | 6.00 | 100.0% | 57.6% | 2.56 | 0 / 0 / 100% | 253.1 |
| DQN (Step 7, 40k decisions) | -411.9 ± 49.3 | 3.45 ± 0.28 | 55.0% ± 15.0 | 20.8% ± 4.0 | 2.97 ± 0.05 | 57 / 23 / 20% | 107.7 ± 29.0 |
| MAPPO (Step 9, 120k) | -153.6 ± 19.8 | 1.90 ± 0.11 | 5.9% ± 5.3 | 1.5% ± 1.3 | 3.31 ± 0.03 | 84 / 2 / 14% | 59.5 ± 11.5 |
| Priority MAPPO W5 (Step 10, 120k) | -151.1 ± 11.3 | 2.04 ± 0.16 | 0.2% ± 0.1 | 0.1% ± 0.1 | 3.33 ± 0.03 | 72 / 0 / 27% | 112.7 ± 11.9 |
