# V4 - Cloud capacity / delay sensitivity (frozen W5 MAPPO, validation seeds 1000-1029)

Current values: cloud capacity 50; cloud delay 2 (default, overload) and 6 (slow network). W5 = mean ± 95% CI over 5 training seeds; critical violation pooled over seeds. No model was retrained for these settings.

| Scenario / setting | W5 return | W5 Critical viol. | W5 Cloud / Edge share | Cloud queue wait | FCFS | ECT | always Cloud | always Local |
|---|---|---|---|---|---|---|---|---|
| default — cloud capacity 50 | -14.0 ± 0.4 | 0.00% | 12.7% / 2.4% | 0.00 | -14.6 | -14.6 | -48.5 | -22.0 |
| default — cloud capacity 25 | -14.0 ± 0.4 | 0.00% | 12.6% / 2.5% | 0.00 | -14.6 | -14.6 | -48.5 | -22.0 |
| default — cloud capacity 12 | -14.1 ± 0.4 | 0.00% | 12.4% / 2.5% | 0.01 | -14.7 | -14.6 | -54.0 | -22.0 |
| default — cloud capacity 6 | -16.1 ± 0.4 | 0.00% | 11.8% / 2.6% | 0.03 | -17.9 | -14.7 | -87.7 | -22.0 |
| default — cloud delay 4 | -20.1 ± 1.2 | 0.00% | 12.6% / 2.5% | 0.00 | -20.9 | -14.8 | -147.6 | -22.0 |
| default — cloud delay 6 | -26.3 ± 2.5 | 0.00% | 12.4% / 2.4% | 0.00 | -27.1 | -14.8 | -290.7 | -22.0 |
| overload — cloud capacity 50 | -133.3 ± 2.6 | 0.00% | 47.5% / 0.0% | 0.00 | -691.2 | -200.2 | -243.5 | -1616.2 |
| overload — cloud capacity 25 | -136.2 ± 3.0 | 0.02% | 44.2% / 0.0% | 0.03 | -691.4 | -247.4 | -319.8 | -1616.2 |
| overload — cloud capacity 12 | -197.4 ± 14.9 | 2.01% | 37.7% / 0.1% | 0.65 | -697.8 | -393.7 | -1817.5 | -1616.2 |
| overload — cloud capacity 6 | -530.9 ± 114.3 | 23.87% | 26.6% / 1.3% | 3.65 | -741.4 | -681.6 | -4164.9 | -1616.2 |
| overload — cloud delay 4 | -243.2 ± 6.9 | 0.45% | 45.1% / 0.0% | 0.00 | -721.8 | -602.1 | -765.1 | -1616.2 |
| overload — cloud delay 6 | -416.0 ± 15.1 | 0.45% | 43.5% / 0.0% | 0.00 | -752.5 | -979.6 | -1469.7 | -1616.2 |
| slow_network — cloud capacity 50 | -146.0 ± 10.3 | 0.42% | 26.6% / 0.1% | 0.00 | -416.6 | -240.8 | -872.8 | -397.2 |
| slow_network — cloud capacity 25 | -141.0 ± 6.6 | 0.93% | 23.4% / 0.0% | 0.00 | -416.7 | -240.8 | -874.7 | -397.2 |
| slow_network — cloud capacity 12 | -143.2 ± 5.3 | 3.12% | 19.2% / 0.0% | 0.02 | -419.1 | -245.0 | -999.0 | -397.2 |
| slow_network — cloud capacity 6 | -173.4 ± 18.7 | 9.07% | 15.3% / 0.1% | 0.15 | -432.4 | -251.5 | -1943.7 | -397.2 |
| slow_network — cloud delay 8 | -181.6 ± 14.0 | 0.42% | 25.7% / 0.1% | 0.00 | -437.0 | -260.4 | -1019.7 | -397.2 |
| slow_network — cloud delay 10 | -216.3 ± 15.8 | 0.59% | 25.0% / 0.0% | 0.00 | -457.3 | -260.7 | -1166.7 | -397.2 |
