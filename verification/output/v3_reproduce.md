# V3 - Reproduction of the main results

- Automated tests: `115 passed in 38.04s`
- Test evaluation `results/step7_final` re-run: files identical {'per_seed.csv': True, 'results_table.md': True, 'learning_curves.png': True}; results identical: True
- Test evaluation `results/step9_final` re-run: files identical {'per_seed.csv': True, 'results_table.md': True, 'learning_curves.png': True}; results identical: True
- Test evaluation `results/step10_final` re-run: files identical {'per_seed.csv': True, 'results_table.md': True, 'learning_curves.png': True}; results identical: True
- Priority MAPPO (W5), default scenario, training seed 0, retrained from scratch (120,000 decisions): identical to `results/step10_final/training_runs/FINAL_default_seed0_20261018T182037Z_672f628`: {'model_best.pt': True, 'model_final.pt': True, 'learning_curve.csv': True, 'validation_curve.csv': True, 'per_seed.csv': True}

All checks passed: True
