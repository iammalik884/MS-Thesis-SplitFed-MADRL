| Scenario | Policy | Return | Avg latency | Critical violation | Cloud share |
|---|---|---|---|---|---|
| default | MAPPO (best) | -15.1 ± 0.4 | 0.56 ± 0.02 | 0.0% ± 0.0 pp | 12% |
| default | MAPPO (final) | -14.8 ± 0.3 | 0.55 ± 0.01 | 0.0% ± 0.0 pp | 10% |
| default | FCFS | -15.7 | 0.58 | 0.0% | 13% |
| default | always Local | -19.9 | 0.69 | 0.7% | 0% |
| default | always Edge | -38.2 | 1.43 | 0.0% | 0% |
| default | always Cloud | -52.7 | 2.00 | 0.0% | 100% |
| default | random | -34.7 | 1.32 | 0.0% | 36% |
| default | ECT | -15.5 | 0.57 | 0.0% | 1% |
| overload | MAPPO (best) | -142.9 ± 2.7 | 1.13 ± 0.02 | 0.0% ± 0.0 pp | 49% |
| overload | MAPPO (final) | -144.4 ± 3.4 | 1.14 ± 0.03 | 0.0% ± 0.0 pp | 50% |
| overload | FCFS | -772.7 | 3.67 | 64.8% | 14% |
| overload | always Local | -1932.2 | 9.72 | 70.2% | 0% |
| overload | always Edge | -2007.6 | 9.51 | 89.9% | 0% |
| overload | always Cloud | -253.5 | 2.00 | 0.2% | 100% |
| overload | random | -555.2 | 3.07 | 32.1% | 34% |
| overload | ECT | -215.2 | 1.67 | 1.2% | 29% |
| slow_network | MAPPO (best) | -151.1 ± 11.3 | 2.04 ± 0.16 | 0.2% ± 0.1 pp | 27% |
| slow_network | MAPPO (final) | -157.8 ± 14.7 | 2.13 ± 0.21 | 0.2% ± 0.1 pp | 30% |
| slow_network | FCFS | -409.6 | 3.64 | 84.0% | 15% |
| slow_network | always Local | -402.9 | 3.60 | 26.3% | 0% |
| slow_network | always Edge | -667.0 | 5.32 | 100.0% | 0% |
| slow_network | always Cloud | -868.7 | 6.00 | 100.0% | 100% |
| slow_network | random | -492.6 | 3.83 | 72.3% | 34% |
| slow_network | ECT | -246.0 | 2.52 | 30.8% | 2% |
