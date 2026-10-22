"""V3 - Reproduce the main results: run the automated tests, re-run the three one-time test
evaluations into a temporary folder and compare them with the committed files, and retrain one
priority-MAPPO model from scratch and compare it byte-for-byte (about 5-10 minutes in total)."""
import filecmp
import glob
import json
import os
import subprocess
import sys
import tempfile

from common import ROOT, write

py = sys.executable
lines = ["# V3 - Reproduction of the main results", ""]
tests = subprocess.run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
lines.append(f"- Automated tests: `{tests.stdout.strip().splitlines()[-1]}`")
ok = tests.returncode == 0
with tempfile.TemporaryDirectory() as tmp:
    for step in ("step7", "step9", "step10"):
        out = os.path.join(tmp, step)
        subprocess.run([py, "evaluate_dqn_test.py", "--runs", f"results/{step}_final/training_runs", "--out", out],
                       cwd=ROOT, capture_output=True, check=True)
        same = {f: filecmp.cmp(os.path.join(out, f), f"results/{step}_final/{f}", shallow=False)
                for f in ("per_seed.csv", "results_table.md", "learning_curves.png")}
        results_same = (json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))["results"]
                        == json.load(open(f"results/{step}_final/summary.json", encoding="utf-8"))["results"])
        ok &= all(same.values()) and results_same
        lines.append(f"- Test evaluation `results/{step}_final` re-run: files identical {same}; results identical: {results_same}")
    out = os.path.join(tmp, "retrain")
    subprocess.run([py, "train_mappo.py", "--config", "configs/environment.yaml", "--train-seed", "0", "--total-steps",
                    "120000", "--set", "priority_weights=5/2/1", "--name", "FINAL_default_seed0", "--out-dir", out],
                   cwd=ROOT, capture_output=True, check=True)
    new = glob.glob(os.path.join(out, "*"))[0]
    old = glob.glob("results/step10_final/training_runs/FINAL_default_seed0_*")[0]
    same = {f: filecmp.cmp(os.path.join(new, f), os.path.join(old, f), shallow=False)
            for f in ("model_best.pt", "model_final.pt", "learning_curve.csv", "validation_curve.csv", "per_seed.csv")}
    ok &= all(same.values())
    lines.append(f"- Priority MAPPO (W5), default scenario, training seed 0, retrained from scratch (120,000 decisions): "
                 f"identical to `{os.path.relpath(old, ROOT).replace(chr(92), chr(47))}`: {same}")
lines.append(f"\nAll checks passed: {ok}")
write("v3_reproduce.md", "\n".join(lines) + "\n")
assert ok
