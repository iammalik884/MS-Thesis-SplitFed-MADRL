"""Shared helpers for the verification scripts (supervisor's questions, after Step 10).
Run every script from the repository root, e.g.  python verification/v1_independent_actors.py"""
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = os.environ.get("VERIFICATION_OUT") or os.path.join(ROOT, "verification", "output")   # dashboard re-runs write to a temp folder
os.makedirs(OUT, exist_ok=True)

T_975 = {4: 2.7764451051977987, 29: 2.045229642132703}   # Student t, 95% two-sided, by degrees of freedom
SCENARIOS = {"default": "configs/environment.yaml", "overload": "configs/scenarios/overload.yaml",
             "slow_network": "configs/scenarios/slow_network.yaml"}


def mean_ci(values):
    """(mean, sd, 95% CI half-width) with t for n-1 degrees of freedom."""
    n = len(values)
    sd = statistics.stdev(values)
    return statistics.mean(values), sd, T_975[n - 1] * sd / math.sqrt(n)


def load_summary(path):
    return json.load(open(path, encoding="utf-8"))


def write(name, text):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    try:
        shown = os.path.relpath(path, ROOT)
    except ValueError:                     # output folder on another drive (e.g. a temp folder)
        shown = path
    print(f"[saved {shown}]")
