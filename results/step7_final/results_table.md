| Scenario | Policy | Return | Avg latency | Critical violation | Cloud share |
|---|---|---|---|---|---|
| default | DQN (best) | -20.1 ± 0.7 | 0.74 ± 0.02 | 0.1% ± 0.2 pp | 1% |
| default | DQN (final) | -20.8 ± 1.1 | 0.76 ± 0.04 | 0.1% ± 0.3 pp | 1% |
| default | FCFS | -15.7 | 0.58 | 0.0% | 13% |
| default | always Local | -19.9 | 0.69 | 0.7% | 0% |
| default | always Edge | -38.2 | 1.43 | 0.0% | 0% |
| default | always Cloud | -52.7 | 2.00 | 0.0% | 100% |
| default | random | -34.7 | 1.32 | 0.0% | 36% |
| default | ECT | -15.5 | 0.57 | 0.0% | 1% |
| overload | DQN (best) | -278.6 ± 71.5 | 1.98 ± 0.34 | 5.7% ± 6.3 pp | 65% |
| overload | DQN (final) | -584.2 ± 512.1 | 3.30 ± 2.15 | 26.6% ± 30.6 pp | 47% |
| overload | FCFS | -772.7 | 3.67 | 64.8% | 14% |
| overload | always Local | -1932.2 | 9.72 | 70.2% | 0% |
| overload | always Edge | -2007.6 | 9.51 | 89.9% | 0% |
| overload | always Cloud | -253.5 | 2.00 | 0.2% | 100% |
| overload | random | -555.2 | 3.07 | 32.1% | 34% |
| overload | ECT | -215.2 | 1.67 | 1.2% | 29% |
| slow_network | DQN (best) | -411.9 ± 49.3 | 3.45 ± 0.28 | 55.0% ± 15.0 pp | 20% |
| slow_network | DQN (final) | -463.8 ± 72.7 | 3.67 ± 0.41 | 62.9% ± 14.6 pp | 30% |
| slow_network | FCFS | -409.6 | 3.64 | 84.0% | 15% |
| slow_network | always Local | -402.9 | 3.60 | 26.3% | 0% |
| slow_network | always Edge | -667.0 | 5.32 | 100.0% | 0% |
| slow_network | always Cloud | -868.7 | 6.00 | 100.0% | 100% |
| slow_network | random | -492.6 | 3.83 | 72.3% | 34% |
| slow_network | ECT | -246.0 | 2.52 | 30.8% | 2% |
