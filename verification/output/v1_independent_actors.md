# V1 - Independent actors, critic only in training

Model: `results/step10_final/training_runs/FINAL_overload_seed0_20261018T183923Z_672f628/model_best.pt`

- Actors: ['Emergency', 'ICU', 'General_Ward'], parameters each: {'Emergency': 5059, 'ICU': 5059, 'General_Ward': 5059}
- Parameter tensors shared between actors: 0
- First-layer weights identical? Emergency-ICU: False, ICU-General_Ward: False
- Actor input size: 10 (local observation); critic input size: 13 (global state)
- Random local observations (1000, seed 0) on which the three actors choose different actions: 297
- Test return (seeds 2000-2029) with critic: -144.76666666666668; critic deleted: -144.76666666666668; committed result: -144.76666666666668; identical: True
- Calls of env.state() (global state) during critic-free evaluation: 0
