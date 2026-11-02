"""
Project dashboard - every step of the thesis project in ONE file (backend + frontend).
================================================================================
ROMAN URDU: `py dashboard.py` chalayein; browser mein dashboard khul jata
hai. Overview aur Concepts pages, phir har step ke teen tabs: Samjhein
(detailed explanation), Nateeje (committed results ke charts/tables) aur
Run (backend par ASAL script chalti hai -- tests, evaluation, sanity
checks -- aur naya nateeja committed nateeje se compare hota hai).
Saare numbers result/config files se parhe jate hain -- koi number haath se
nahi likha.

ENGLISH: a local web app on 127.0.0.1 (Python standard library only). The
backend reads the committed result files and, on "Run", executes the same
commands as documented in the README, then compares the new outputs with the
committed ones. It never edits configs, code or committed results; temporary
outputs go to the system temp folder or to results/runs/dashboard/ (git-ignored).

Run:  py dashboard.py                (opens http://127.0.0.1:8765; "python dashboard.py" too, if python is on PATH)
      py dashboard.py --port 8800 --no-browser
"""

import argparse
import csv
import filecmp
import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
PY = sys.executable
CACHE = os.path.join(ROOT, "results", "runs", "dashboard")      # git-ignored
SCEN = {"default": "configs/environment.yaml", "overload": "configs/scenarios/overload.yaml",
        "slow_network": "configs/scenarios/slow_network.yaml"}
SCEN_NAME = {"default": "Default", "overload": "Overload", "slow_network": "Slow network",
             "department_imbalance": "Department imbalance", "overdue_critical": "Overdue Critical",
             "empty_workload": "Empty workload"}
T4 = 2.7764451051977987   # t(0.975, df 4): 95% CI over 5 training seeds

# One fixed colour per policy across the whole dashboard (reference palette, slots in documented order).
COLOR = {"Priority MAPPO (W5)": "s1", "MAPPO": "s2", "ECT": "s3", "FCFS": "s4", "DQN": "s5",
         "always Cloud": "s6", "always Local": "s7", "random": "s8"}


# ============================================================================
# Helpers
# ============================================================================
def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ci(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return (values[0] if values else None), None
    return statistics.mean(values), T4 * statistics.stdev(values) / math.sqrt(len(values))


def f1(x):
    return "—" if x is None else f"{x:,.1f}"


def f2(x):
    return "—" if x is None else f"{x:.2f}"


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def pm(v, fmt=f1):
    m, h = v
    return fmt(m) + ("" if h is None else f" ± {fmt(h) if fmt is not pct else f'{100 * h:.1f}'}")


def config_of(path):
    from env.hospital_env import load_config
    return load_config(path)


def tests_block(files):
    return {"type": "note", "text": "Is step ke automated tests: " + ", ".join(f"<code>{f}</code>" for f in files)
            + ". Inhein <b>▶ Chala kar check karein</b> tab se chalayein."}


# ============================================================================
# Data builders: read committed files and return display blocks
# ============================================================================
def data_step1():
    cfg = config_of(SCEN["default"])
    demo = subprocess.run([PY, "demo_simulation.py"], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout
    caps = [{"label": d["name"], "value": f"{d['capacity']:g}", "note": "compute units / step"} for d in cfg["departments"]]
    caps.append({"label": "Cloud", "value": f"{cfg['offloading']['cloud_capacity']:g}", "note": "compute units / step"})
    return [{"type": "kpis", "title": "Nodes aur un ki capacity (configs/environment.yaml)", "items": caps},
            {"type": "pre", "title": "Demo run (python demo_simulation.py, seed 42) — har task ka safar", "text": demo},
            tests_block(["tests/test_step1_simulator.py"])]


def data_step2():
    cfg = config_of(SCEN["default"])
    rows = []
    for dep, prs in cfg["task_generation"]["departments"].items():
        for p, v in prs.items():
            rows.append([dep, p, f"{v['arrival_rate']:g}", f"{v['compute_demand_range'][0]:g} – {v['compute_demand_range'][1]:g}"])
    panels = []
    for key in ("default", "overload", "slow_network", "department_imbalance"):
        c = config_of(SCEN.get(key, f"configs/scenarios/{key}.yaml"))
        caps = {d["name"]: d["capacity"] for d in c["departments"]}
        vals = []
        for dep, prs in c["task_generation"]["departments"].items():
            load = sum(v["arrival_rate"] * sum(v["compute_demand_range"]) / 2 for v in prs.values())
            vals.append(load / caps[dep])
        panels.append({"title": SCEN_NAME[key], "categories": list(caps),
                       "series": [{"name": "load / capacity (ρ)", "color": "s1", "values": vals}]})
    return [{"type": "table", "title": "Default workload: arrival rate (tasks / step) aur compute demand (uniform)",
             "columns": ["Department", "Priority", "Arrival rate", "Demand range"], "rows": rows},
            {"type": "bars", "title": "Har department par load ÷ capacity (ρ) — 1 se upar = department akela nahi sambhal sakta",
             "panels": panels, "fmt": "n2", "ref": [{"value": 1.0, "label": "ρ = 1"}]},
            tests_block(["tests/test_step2_task_generator.py"])]


def data_step3():
    off = config_of(SCEN["default"])["offloading"]
    rows = [["Local", "apna department node", "0"],
            ["Edge (neighbour)", "sab se kam backlog wala padosi department", str(off["neighbour_network_delay"])],
            ["Cloud", f"shared cloud (capacity {off['cloud_capacity']:g})", str(off["cloud_network_delay"])]]
    rule = (f"1) demand > {off['cloud_threshold']:g} → Cloud &nbsp; 2) apna backlog ≤ {off['offload_threshold']:g} → Local "
            f"&nbsp; 3) warna → Edge (sab se halka padosi)")
    return [{"type": "table", "title": "Teen jagah jahan task process ho sakta hai", "columns": ["Action", "Kahan", "Network delay (steps)"], "rows": rows},
            {"type": "note", "text": "<b>FCFS routing rule</b> (fixed, priority nahi dekhta): " + rule},
            tests_block(["tests/test_step3_routing.py", "tests/test_v2_routing_timing.py"])]


def data_step4():
    d = config_of(SCEN["default"])["deadlines"]
    return [{"type": "table", "title": "Deadline = arrival time + priority offset", "columns": ["Priority", "Deadline offset (steps)"],
             "rows": [[p, str(v)] for p, v in d.items()]},
            {"type": "note", "text": "Har task ka deadline status 4 mein se ek: <b>on_time</b>, <b>completed_late</b>, "
             "<b>overdue_unfinished</b>, <b>not_yet_due</b> (miss nahi gina jata)."},
            tests_block(["tests/test_step4_deadlines.py", "tests/test_v2_metrics_protocol.py"])]


def _baseline_runs():
    runs = {"default": glob.glob("results/baselines/step5_fcfs_v2_seeds0-29_*")[0]}
    for d in sorted(glob.glob("results/baselines/scenarios/*")):
        runs[re.match(r"fcfs_v2_(.+)_seeds", os.path.basename(d)).group(1)] = d
    return runs


def data_step5():
    rows, keys, lat, crit = [], [], [], []
    for key, d in _baseline_runs().items():
        s = jload(os.path.join(d, "summary.json"))["summary"]
        g = lambda k: s[k]["mean"] if s.get(k) else None     # empty workload: no tasks -> metric is None, not 0
        rows.append([SCEN_NAME[key], f1(g("tasks")), f2(g("throughput_in_window")), f2(g("avg_latency")),
                     pct(g("deadline_violation_rate")), pct(g("critical_deadline_violation_rate"))])
        if key != "empty_workload":
            keys.append(SCEN_NAME[key]); lat.append(g("avg_latency")); crit.append(g("critical_deadline_violation_rate"))
    return [{"type": "table", "title": "FCFS baseline, seeds 0–29 ka mean (results/baselines/)",
             "columns": ["Scenario", "Tasks / run", "Throughput", "Avg latency", "Deadline violation", "Critical violation"], "rows": rows},
            {"type": "bars", "title": "FCFS: average latency (steps)", "fmt": "n2",
             "panels": [{"title": "", "categories": keys, "series": [{"name": "FCFS", "color": COLOR["FCFS"], "values": lat}]}]},
            {"type": "bars", "title": "FCFS: Critical tasks ki deadline violation", "fmt": "pct1",
             "panels": [{"title": "", "categories": keys, "series": [{"name": "FCFS", "color": COLOR["FCFS"], "values": crit}]}]}]


def _cached(name):
    path = os.path.join(CACHE, name)
    return jload(path) if os.path.exists(path) else None


def data_step6():
    blocks = [{"type": "note", "text": "Environment: observation 13 values, action Discrete(3) = Local / Edge / Cloud, reward = "
               "−latency − 10 × deadline miss (task khatam hone par). FCFS ke actions ke saath yeh Step 5 ko bilkul dohrata hai (tested)."}]
    c = _cached("step6_sanity.json")
    if not c:
        return blocks + [{"type": "empty", "text": "Sanity-check results abhi nahi bane. <b>Run</b> dabayein (≈1 minute)."}]
    names = ["FCFS", "ECT", "always Cloud", "always Local", "random"]
    panels = [{"title": SCEN_NAME[k], "categories": [""], "series": [
        {"name": n, "color": COLOR[n], "values": [c[k][n]["cost"]]} for n in names]} for k in SCEN]
    rows = [[SCEN_NAME[k], n, f1(c[k][n]["cost"]), f2(c[k][n]["latency"]), pct(c[k][n]["critical"])]
            for k in SCEN for n in names + ["always Edge"]]
    return blocks + [{"type": "bars", "title": "Simple fixed policies (sanity check, seeds 0–29): cost = −return (kam = behtar)",
                      "panels": panels, "fmt": "n1", "independent": True},
                     {"type": "table", "title": "Sanity check ki table", "columns": ["Scenario", "Policy", "Cost", "Avg latency", "Critical violation"], "rows": rows,
                      "group": True, "dot": 1}]


def _curves(folder, scenario):
    runs = sorted(glob.glob(os.path.join(folder, f"*_{scenario}_seed*")))
    cur = []
    for r in runs:
        with open(os.path.join(r, "validation_curve.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cur.append(([int(x["step"]) for x in rows], [-float(x["mean_return"]) for x in rows]))
    n = min(len(c[1]) for c in cur)
    x = cur[0][0][:n]
    cols = list(zip(*[c[1][:n] for c in cur]))
    base = jload(os.path.join(runs[0], "summary.json"))["baselines"]
    return {"title": SCEN_NAME[scenario], "x": x, "series": [{"mean": [statistics.mean(v) for v in cols],
            "lo": [min(v) for v in cols], "hi": [max(v) for v in cols]}],
            "refs": [{"name": "ECT", "color": COLOR["ECT"], "value": -base["ECT"]["mean_return"]},
                     {"name": "FCFS", "color": COLOR["FCFS"], "value": -base["FCFS"]["mean_return"]}]}


def _best_rows(costs):
    """Row indices with the lowest cost inside each scenario group; costs = [(scenario, cost), ...]."""
    low = {}
    for sc, c in costs:
        low[sc] = min(c, low.get(sc, c))
    return [i for i, (sc, c) in enumerate(costs) if c == low[sc]]


def _test_table(summary, key, label):
    rows, costs = [], []
    for sc in ("default", "overload", "slow_network"):
        R = summary["results"][sc]
        for name, k in ((label, key), ("ECT", "ECT"), ("FCFS", "FCFS"), ("always Cloud", "always Cloud")):
            r = R[k]
            cost = (-r["return"][0], r["return"][1])
            costs.append((sc, round(cost[0], 1)))
            rows.append([SCEN_NAME[sc], name, pm(cost), pm(r["avg_latency"], f2), pm(r["critical_deadline_violation_rate"], pct),
                         f"{100 * r['action_shares']['Local']:.0f} / {100 * r['action_shares']['Edge']:.0f} / {100 * r['action_shares']['Cloud']:.0f}%"])
    return {"type": "table", "title": f"Test seeds 2000–2029 (sirf ek dafa) — {label} best checkpoint, mean ± 95% CI (5 training seeds)",
            "columns": ["Scenario", "Policy", "Cost (−return)", "Avg latency", "Critical violation", "Local / Edge / Cloud"], "rows": rows,
            "group": True, "dot": 1, "best": _best_rows(costs), "best_note": "har scenario mein sab se kam cost"}


def _round_scores(groups):
    import summarize_dqn_experiments as S
    rows = []
    for g in groups:
        rows += S.load(g)
    order, out = [], {}
    for r in rows:
        if r["exp"] not in out:
            order.append(r["exp"]); out[r["exp"]] = []
        out[r["exp"]].append(r)
    return order, {e: (statistics.mean(x["final_score"] for x in v), statistics.mean(x["best_score"] for x in v),
                       v[0]["total_steps"]) for e, v in out.items()}


def data_step7():
    order, sc = _round_scores(["results/debug/phase1", "results/debug/phase2", "results/debug/round2"])
    s7 = jload("results/step7_final/summary.json")
    m = jload(glob.glob("results/step7_final/training_runs/*/manifest.json")[0])
    c, n_runs = m["dqn_config"], len(glob.glob("results/step7_final/training_runs/*"))
    return [{"type": "kpis", "title": "Frozen DQN (pre-registered rule se)", "items": [
                {"label": "Setting", "value": "Double DQN" if c["double_dqn"] else "DQN", "note": f"reward × {c['reward_scale']:g}, γ {c['gamma']:g}"},
                {"label": "Training", "value": f"{m['total_steps'] // 1000}k", "note": f"decisions; {n_runs} runs (5 seeds × 3 scenarios)"},
                {"label": "Overload cost (test)", "value": f1(-s7["results"]["overload"]["DQN (best)"]["return"][0]), "note": f"ECT {f1(-s7['results']['overload']['ECT']['return'][0])}"}]},
            {"type": "bars", "title": "Debugging (validation seeds): mean FINAL-checkpoint score — 0 = random, 1 = ECT", "fmt": "n2",
             "ref": [{"value": 0, "label": "random"}, {"value": 1, "label": "ECT"}],
             "panels": [{"title": "", "categories": [f"{e} ({sc[e][2] // 1000}k)" for e in order],
                         "series": [{"name": "DQN setting", "color": COLOR["DQN"], "values": [sc[e][0] for e in order]}]}]},
            {"type": "lines", "title": "Learning curves (validation seeds): cost vs training decisions — DQN mean, band = 5 seeds ka range",
             "series_name": "DQN", "color": COLOR["DQN"], "panels": [_curves("results/step7_final/training_runs", k) for k in SCEN]},
            _test_table(s7, "DQN (best)", "DQN")]


def data_step8():
    blocks = [{"type": "note", "text": "Har department ek agent: Emergency, ICU, General Ward. Local observation 10 values; "
               "task ka reward us ke source department ko; teeno ka jor = Step 6 ka reward (har step check hota hai)."}]
    c = _cached("step8_agents.json")
    if not c:
        return blocks + [{"type": "empty", "text": "Per-agent results abhi nahi bane. <b>Run</b> dabayein (≈1 minute)."}]
    agents = ["Emergency", "ICU", "General_Ward"]
    panels = [{"title": SCEN_NAME[k], "categories": agents,
               "series": [{"name": rule, "color": COLOR[rule], "values": [-c[k][rule][a] for a in agents]} for rule in ("FCFS", "ECT")]}
              for k in SCEN]
    rows = [[SCEN_NAME[k], rule] + [f1(-c[k][rule][a]) for a in agents] + [f1(-sum(c[k][rule].values()))]
            for k in SCEN for rule in ("FCFS", "ECT")]
    return blocks + [{"type": "bars", "title": "Har agent ka cost (seeds 0–29), har agent same fixed rule independently", "panels": panels,
                      "fmt": "n1", "independent": True},
                     {"type": "table", "title": "Per-agent aur team cost", "columns": ["Scenario", "Rule"] + agents + ["Team"], "rows": rows,
                      "group": True, "dot": 1}]


def data_step9():
    order, sc = _round_scores(["results/debug/mappo1", "results/debug/mappo2"])
    s9 = jload("results/step9_final/summary.json")
    return [{"type": "kpis", "title": "Frozen MAPPO (CTDE)", "items": [
                {"label": "Actors", "value": "3", "note": "har department ka alag, local observation"},
                {"label": "Critic", "value": "1", "note": "global state, sirf training mein"},
                {"label": "Overload cost (test)", "value": f1(-s9["results"]["overload"]["MAPPO (best)"]["return"][0]),
                 "note": f"ECT {f1(-s9['results']['overload']['ECT']['return'][0])}"}]},
            {"type": "bars", "title": "Experiments (validation seeds): mean FINAL score — 1 = ECT", "fmt": "n2",
             "ref": [{"value": 1, "label": "ECT"}],
             "panels": [{"title": "", "categories": [f"{e} ({sc[e][2] // 1000}k)" for e in order],
                         "series": [{"name": "MAPPO setting", "color": COLOR["MAPPO"], "values": [sc[e][0] for e in order]}]}]},
            {"type": "lines", "title": "Learning curves (validation seeds): cost vs training decisions", "series_name": "MAPPO",
             "color": COLOR["MAPPO"], "panels": [_curves("results/step9_final/training_runs", k) for k in SCEN]},
            _test_table(s9, "MAPPO (best)", "MAPPO"),
            {"type": "note", "text": "Dhyan dein: MAPPO load ko zyada tar <b>Cloud</b> se balance karta hai (overload mein Edge 0%), kyunke "
             "is model mein Cloud ki koi cost nahi aur woh kabhi nahi bharta. Checks page (V4) par cloud sensitivity dekhein."}]


def _prio(summary, sc, label_prefix):
    pb = summary["priority_breakdown"][sc]
    labels = [k for k in pb if k.startswith(label_prefix)]
    return {p: (ci([pb[l]["violation"][p]["violation_rate"] for l in labels]), ci([pb[l]["latency"][p]["avg_latency"] for l in labels]))
            for p in ("Critical", "High", "Routine")}


def data_step10():
    crit = {}
    for e in ("W1", "W5", "W25"):
        crit[e] = statistics.mean(jload(p)["violation_by_priority"]["mappo_final"]["Critical"]["violation_rate"]
                                  for p in glob.glob(f"results/debug/prio1/{e}/*/summary.json"))
    A = jload("results/step10_final/summary.json")
    R = jload("results/step10_final/step9_reference/summary.json")
    blocks = [{"type": "kpis", "title": "Priority weights (deadline offsets 2 / 5 / 10 se)", "items": [
                  {"label": "W1", "value": "1 / 1 / 1", "note": "koi priority nahi (= Step 9)"},
                  {"label": "W5 (chosen)", "value": "5 / 2 / 1", "note": "urgency = 10 / offset"},
                  {"label": "W25", "value": "25 / 4 / 1", "note": "urgency squared"}]},
              {"type": "bars", "title": "Selection (validation seeds): mean Critical violation, final checkpoint", "fmt": "pct2",
               "panels": [{"title": "", "categories": ["W1", "W5", "W25"],
                           "series": [{"name": "Priority MAPPO", "color": COLOR["Priority MAPPO (W5)"], "values": [crit[e] for e in ("W1", "W5", "W25")]}]}]}]
    for sc in ("slow_network", "overload"):
        w5, w1 = _prio(A, sc, "MAPPO_best"), _prio(R, sc, "MAPPO_best")
        base = {b: A["priority_breakdown"][sc][b] for b in ("ECT", "FCFS")}
        pr = ("Critical", "High", "Routine")
        series_v = [{"name": "Priority MAPPO (W5)", "color": COLOR["Priority MAPPO (W5)"], "values": [w5[p][0][0] for p in pr], "err": [w5[p][0][1] for p in pr]},
                    {"name": "MAPPO", "color": COLOR["MAPPO"], "values": [w1[p][0][0] for p in pr], "err": [w1[p][0][1] for p in pr]},
                    {"name": "ECT", "color": COLOR["ECT"], "values": [base["ECT"]["violation"][p]["violation_rate"] for p in pr]},
                    {"name": "FCFS", "color": COLOR["FCFS"], "values": [base["FCFS"]["violation"][p]["violation_rate"] for p in pr]}]
        series_l = [{"name": "Priority MAPPO (W5)", "color": COLOR["Priority MAPPO (W5)"], "values": [w5[p][1][0] for p in pr], "err": [w5[p][1][1] for p in pr]},
                    {"name": "MAPPO", "color": COLOR["MAPPO"], "values": [w1[p][1][0] for p in pr], "err": [w1[p][1][1] for p in pr]},
                    {"name": "ECT", "color": COLOR["ECT"], "values": [base["ECT"]["latency"][p]["avg_latency"] for p in pr]},
                    {"name": "FCFS", "color": COLOR["FCFS"], "values": [base["FCFS"]["latency"][p]["avg_latency"] for p in pr]}]
        blocks.append({"type": "bars", "title": f"{SCEN_NAME[sc]} (test seeds): har priority ki deadline violation aur latency",
                       "fmt": "auto", "panels": [{"title": "Deadline violation (pooled)", "categories": list(pr), "series": series_v, "fmt": "pct1"},
                                                  {"title": "Average latency (steps)", "categories": list(pr), "series": series_l, "fmt": "n2"}],
                       "independent": True})
    blocks.append(_test_table(A, "MAPPO (best)", "Priority MAPPO (W5)"))
    F = facts()
    blocks.append({"type": "note", "text": "Roadmap checkpoint: resources kam hon to Critical violations kam, chahe Routine thora zyada wait kare — "
                   f"<b>haan</b> (slow network: Critical {F['s10_crit_w1']} → {F['s10_crit_w5']}, Routine latency {F['s10_rout_w1']} → "
                   f"{F['s10_rout_w5']}, Routine violation {F['s10_routviol_w5']}). Keemat: Cloud ka istemal zyada ({F['s10_cloud_w5']} vs {F['s10_cloud_w1']})."})
    return blocks


def data_results():
    S7, S9, S10 = (jload(f"results/{s}_final/summary.json") for s in ("step7", "step9", "step10"))
    methods = [("Priority MAPPO (W5)", S10, "MAPPO (best)"), ("MAPPO", S9, "MAPPO (best)"), ("ECT", S7, "ECT"),
               ("FCFS", S7, "FCFS"), ("DQN", S7, "DQN (best)"), ("always Cloud", S7, "always Cloud")]
    cost, crit, shares, rows, costs = [], [], [], [], []
    for sc in ("default", "overload", "slow_network"):
        R = lambda S, k: S["results"][sc][k]
        cost.append({"title": SCEN_NAME[sc], "categories": [""], "series": [
            {"name": n, "color": COLOR[n], "values": [-R(S, k)["return"][0]], "err": [R(S, k)["return"][1]]} for n, S, k in methods]})
        crit.append({"title": SCEN_NAME[sc], "categories": [""], "series": [
            {"name": n, "color": COLOR[n], "values": [R(S, k)["critical_deadline_violation_rate"][0]],
             "err": [R(S, k)["critical_deadline_violation_rate"][1]]} for n, S, k in methods]})
        shares.append({"title": SCEN_NAME[sc], "categories": [n for n, _, _ in methods], "parts": [
            {"name": a, "color": c, "values": [R(S, k)["action_shares"][a] for _, S, k in methods]}
            for a, c in (("Local", "q1"), ("Edge", "q2"), ("Cloud", "q3"))]})
        for n, S, k in methods:
            r = R(S, k)
            costs.append((sc, round(-r["return"][0], 1)))
            rows.append([SCEN_NAME[sc], n, pm((-r["return"][0], r["return"][1])), pm(r["avg_latency"], f2),
                         pm(r["critical_deadline_violation_rate"], pct), pm(r["deadline_violation_rate"], pct), pm(r["throughput_in_window"], f2),
                         f1(r["synthetic_payload_units_total"][0])])
    return [{"type": "note", "text": "Sab methods <b>same test workloads</b> (seeds 2000–2029) aur <b>same metrics</b> par. Learned methods: "
             "best checkpoint, 5 training seeds ka mean ± 95% CI (error bar). Baselines deterministic hain. DQN 40k, MAPPO 120k training decisions "
             "(dono ko same options mile; har budget apne pre-registered rule se)."},
            {"type": "bars", "title": "Cost = −return (kam = behtar), har scenario ka apna scale", "panels": cost, "fmt": "n1", "independent": True},
            {"type": "bars", "title": "Critical tasks ki deadline violation", "panels": crit, "fmt": "pct1", "independent": True},
            {"type": "stacked", "title": "Faisle kahan gaye: Local / Edge (padosi) / Cloud", "panels": shares},
            {"type": "table", "title": "Poori table (test seeds)", "columns": ["Scenario", "Method", "Cost", "Avg latency", "Critical viol.",
             "All-task viol.", "Throughput", "Payload"], "rows": rows, "group": True, "dot": 1, "best": _best_rows(costs),
             "best_note": "har scenario mein sab se kam cost"}]


def data_verification():
    blocks = [{"type": "note", "text": "Supervisor ke sawalon ke checks (folder <code>verification/</code>). Yeh scripts koi model ya "
               "environment nahi badalte. V4 aur V6 sirf validation seeds use karte hain. Neeche committed outputs hain; Run dabane par "
               "script dobara chalti hai aur naya output committed output se compare hota hai."}]
    titles = {"v1_independent_actors": "V1 — Independent actors, critic sirf training mein",
              "v2_consolidated_comparison": "V2 — Consolidated comparison",
              "v3_reproduce": "V3 — Reproduction (tests, evaluations, retraining)",
              "v4_cloud_sensitivity": "V4 — Cloud capacity / delay sensitivity",
              "v5_priority_results": "V5 — Priority weights aur per-priority results",
              "v6_imbalance_generalization": "V6 — Department imbalance (Phase 0)"}
    v4 = jload("verification/output/v4_cloud_sensitivity.json")
    panels = []
    for sc in ("overload", "slow_network"):
        keys = [k for k in v4 if k.startswith(sc + " |")]
        w5 = [[-v4[k][f"W5 trainseed{i}"]["return"] for i in range(5)] for k in keys]
        panels.append({"title": SCEN_NAME[sc], "categories": [k.split("| ")[1].replace("cloud ", "") for k in keys], "series": [
            {"name": "Priority MAPPO (W5)", "color": COLOR["Priority MAPPO (W5)"], "values": [ci(x)[0] for x in w5], "err": [ci(x)[1] for x in w5]},
            {"name": "ECT", "color": COLOR["ECT"], "values": [-v4[k]["ECT"]["return"] for k in keys]},
            {"name": "FCFS", "color": COLOR["FCFS"], "values": [-v4[k]["FCFS"]["return"] for k in keys]}]})
    blocks.append({"type": "bars", "title": "V4 — Cloud chhota ya slow ho to cost (validation seeds, frozen models; pehla group = maujooda setting)",
                   "panels": panels, "fmt": "n1", "independent": True})
    v6 = jload("verification/output/v6_imbalance_generalization.json")
    cats = ["FCFS", "ECT"] + [f"W5 · {SCEN_NAME[s]}" for s in SCEN]
    vals = [v6["baselines"][b]["icu_shares"]["Edge"] for b in ("FCFS", "ECT")]
    errs = [None, None]
    for s in SCEN:
        m, h = ci([x["icu_shares"]["Edge"] for x in v6["learned"][f"MAPPO W5 (Step 10) | trained on {s}"]])
        vals.append(m); errs.append(h)
    blocks.append({"type": "bars", "title": "V6 — Department imbalance: ICU ke kitne tasks padosi departments (Edge) ko gaye (validation seeds; W5 = jis scenario par train hua)",
                   "fmt": "pct1", "panels": [{"title": "", "categories": cats, "series": [{"name": "ICU Edge share", "color": "q2", "values": vals, "err": errs}]}]})
    for k, t in titles.items():
        p = f"verification/output/{k}.md"
        if os.path.exists(p):
            blocks.append({"type": "md", "title": t, "text": open(p, encoding="utf-8").read(), "collapsed": True})
    return blocks


# ============================================================================
# Run actions (backend). Each returns a list of verdicts {"check", "ok", "detail"}.
# ============================================================================
class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.step = self.option = None
        self.log, self.verdicts, self.started, self.finished = [], [], None, None

    def write(self, line):
        self.log.append(line.rstrip("\n"))
        del self.log[:-400]


JOB = Job()


def sh(args, timeout=3600, env_extra=None):
    """Run a command, stream its output into the job log, return (exit code, full output)."""
    JOB.write("$ python " + " ".join(args))
    p = subprocess.Popen([PY] + args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env_extra or {})})
    out = []
    for line in p.stdout:
        out.append(line)
        JOB.write(line)
    p.wait(timeout=timeout)
    return p.returncode, "".join(out)


def run_tests(files):
    code, out = sh(["-m", "pytest", "-q", "-p", "no:cacheprovider"] + files)
    last = [l for l in out.strip().splitlines() if l.strip()][-1] if out.strip() else ""
    return [{"check": "Automated tests (" + ", ".join(os.path.basename(f) for f in files) + ")", "ok": code == 0, "detail": last}]


def run_step5():
    v = run_tests(["tests/test_step5_fcfs_baseline.py", "tests/test_v2_validation_scenarios.py"])
    with tempfile.TemporaryDirectory() as tmp:
        sh(["run_baseline_fcfs.py", "--seed", "42", "--out-dir", tmp, "--name", "step5_fcfs_v2"])
        sh(["run_baseline_fcfs.py", "--seeds", "0-29", "--out-dir", tmp, "--name", "step5_fcfs_v2"])
        for key in ("department_imbalance", "empty_workload", "overdue_critical", "overload", "slow_network"):
            sh(["run_baseline_fcfs.py", "--config", f"configs/scenarios/{key}.yaml", "--seeds", "0-29", "--out-dir", tmp, "--name", f"fcfs_v2_{key}"])
        n = bad = 0
        for d in glob.glob("results/baselines/step5_*") + glob.glob("results/baselines/scenarios/*"):
            prefix = re.sub(r"_2026.*", "", os.path.basename(d))
            new = glob.glob(os.path.join(tmp, prefix + "_2026*"))[0]
            for f in os.listdir(d):
                if f == "manifest.json":
                    continue
                n += 1
                bad += not filecmp.cmp(os.path.join(d, f), os.path.join(new, f), shallow=False)
    return v + [{"check": "FCFS baselines dobara chala kar committed results/baselines se compare", "ok": bad == 0,
                 "detail": f"{n - bad} / {n} files byte-for-byte identical (manifest mein sirf time/commit hota hai, us ke ilawa)"}]


def run_step6():
    v = run_tests(["tests/test_step6_gym_env.py"])
    import check_simple_policies
    out = {}
    for k, path in SCEN.items():
        JOB.write(f"sanity check: {k} (seeds 0-29)")
        rows = check_simple_policies.run(path, range(30))
        out[k] = {(n if n != "ECT heuristic" else "ECT"): {"cost": -r, "latency": lat, "critical": c} for n, r, lat, c in rows}
    os.makedirs(CACHE, exist_ok=True)
    json.dump(out, open(os.path.join(CACHE, "step6_sanity.json"), "w", encoding="utf-8"), indent=1)
    return v + [{"check": "Sanity check (6 fixed policies × 3 scenarios × 30 seeds) chala", "ok": True, "detail": "charts upar update ho gaye"}]


def run_step8():
    v = run_tests(["tests/test_step8_multi_agent_env.py"])
    import check_multi_agent_env
    out = {}
    for k, path in SCEN.items():
        JOB.write(f"per-agent sanity check: {k} (seeds 0-29)")
        _, table = check_multi_agent_env.run(path, range(30))
        out[k] = table
    os.makedirs(CACHE, exist_ok=True)
    json.dump(out, open(os.path.join(CACHE, "step8_agents.json"), "w", encoding="utf-8"), indent=1)
    return v + [{"check": "Per-agent sanity check chala", "ok": True, "detail": "charts upar update ho gaye"}]


def run_reeval(step):
    """Re-run the one-time test evaluation of the frozen models into a temp folder; compare with the committed files."""
    with tempfile.TemporaryDirectory() as tmp:
        code, _ = sh(["evaluate_dqn_test.py", "--runs", f"results/{step}_final/training_runs", "--out", tmp])
        same = {f: code == 0 and filecmp.cmp(os.path.join(tmp, f), f"results/{step}_final/{f}", shallow=False)
                for f in ("per_seed.csv", "results_table.md", "learning_curves.png")}
        new = jload(os.path.join(tmp, "summary.json")) if code == 0 else {}
    old = jload(f"results/{step}_final/summary.json")
    keys = [k for k in ("results", "destination_load", "priority_breakdown") if k in old]
    res_same = code == 0 and all(new.get(k) == old[k] for k in keys)
    return [{"check": f"Frozen models dobara evaluate (test seeds) vs committed results/{step}_final", "ok": all(same.values()) and res_same,
             "detail": ", ".join(f"{f}: {'identical' if s else 'DIFFERENT'}" for f, s in same.items()) + f"; {', '.join(keys)}: "
             + ("identical" if res_same else "DIFFERENT")}]


def run_retrain_w5():
    with tempfile.TemporaryDirectory() as tmp:
        sh(["train_mappo.py", "--config", "configs/environment.yaml", "--train-seed", "0", "--total-steps", "120000",
            "--set", "priority_weights=5/2/1", "--name", "FINAL_default_seed0", "--out-dir", tmp])
        new = glob.glob(os.path.join(tmp, "*"))[0]
        old = glob.glob("results/step10_final/training_runs/FINAL_default_seed0_*")[0]
        same = {f: filecmp.cmp(os.path.join(new, f), os.path.join(old, f), shallow=False)
                for f in ("model_best.pt", "model_final.pt", "learning_curve.csv", "validation_curve.csv", "per_seed.csv")}
    return [{"check": "Priority MAPPO (default, seed 0) shuru se dobara train (120k decisions) vs committed model", "ok": all(same.values()),
             "detail": ", ".join(f"{f}: {'identical' if s else 'DIFFERENT'}" for f, s in same.items())}]


def run_verification(name):
    with tempfile.TemporaryDirectory() as tmp:
        code, _ = sh([f"verification/{name}.py"], env_extra={"VERIFICATION_OUT": tmp})
        new_p, old_p = os.path.join(tmp, f"{name}.md"), f"verification/output/{name}.md"
        if code != 0 or not os.path.exists(new_p):
            return [{"check": name, "ok": False, "detail": f"script exit code {code}"}]
        strip = lambda t: re.sub(r"passed in [0-9.]+s", "passed in Xs", t)     # V3 prints the test run time
        same = strip(open(new_p, encoding="utf-8").read()) == strip(open(old_p, encoding="utf-8").read())
    return [{"check": f"{name}: script chala (exit 0) aur output committed verification/output/{name}.md se", "ok": same,
             "detail": "identical" if same else "DIFFERENT (output dekhein)"}]


def _tests(files):
    """A tests-only run option: (description, callable)."""
    names = ", ".join(f"<code>{os.path.basename(f)}</code>" for f in files)
    return f"Is step ke automated tests (pytest) chalte hain: {names}. PASS = code wahi karta hai jo documentation kehti hai.", (lambda: run_tests(files))


def _reeval_desc(step, method):
    return (f"Frozen {method} models test seeds 2000–2029 par temp folder mein dobara evaluate hote hain; per_seed.csv, results_table.md, "
            f"learning_curves.png aur summary ke nateeje committed <code>results/{step}_final/</code> se byte-for-byte compare.")


# (option id, button label, description, function)
RUNS = {
    "step1": [("tests", "Simulator tests (≈5 s)", *_tests(["tests/test_step1_simulator.py"]))],
    "step2": [("tests", "Task generator tests (≈5 s)", *_tests(["tests/test_step2_task_generator.py"]))],
    "step3": [("tests", "Routing tests (≈10 s)", *_tests(["tests/test_step3_routing.py", "tests/test_v2_routing_timing.py"]))],
    "step4": [("tests", "Deadline tests (≈10 s)", *_tests(["tests/test_step4_deadlines.py", "tests/test_v2_metrics_protocol.py"]))],
    "step5": [("baselines", "Tests + FCFS baselines dobara (≈30 s)", "Tests, phir saare FCFS baselines (6 scenarios × 30 seeds, aur seed 42 "
               "demo) temp folder mein dobara bante hain aur committed <code>results/baselines/</code> ki har file se byte-for-byte compare "
               "hote hain (manifest ke ilawa, us mein sirf waqt aur commit hota hai).", run_step5)],
    "step6": [("sanity", "Tests + sanity check (≈1 min)", "Tests, phir simple fixed policies (FCFS, ECT, always Cloud / Local / Edge, "
               "random) × 3 scenarios × 30 seeds. Nateeje tab ke charts isi run se bante hain.", run_step6)],
    "step7": [("reeval", "Test evaluation dobara (≈30 s)", _reeval_desc("step7", "DQN"), lambda: run_reeval("step7")),
              ("tests", "DQN aur ECT tests (≈20 s)", *_tests(["tests/test_dqn_pilot.py", "tests/test_ect_heuristic.py"]))],
    "step8": [("sanity", "Tests + per-agent check (≈1 min)", "Tests, phir har agent FCFS aur ECT rule ke saath (3 scenarios × 30 seeds); "
               "har agent aur team ka cost Nateeje tab mein.", run_step8)],
    "step9": [("reeval", "Test evaluation dobara (≈30 s)", _reeval_desc("step9", "MAPPO"), lambda: run_reeval("step9")),
              ("tests", "MAPPO tests (≈20 s)", *_tests(["tests/test_mappo.py"]))],
    "step10": [("reeval", "Test evaluation dobara (≈30 s)", _reeval_desc("step10", "Priority MAPPO"), lambda: run_reeval("step10")),
               ("retrain", "Ek model shuru se train (≈3–5 min)", "Priority MAPPO (default scenario, training seed 0) bilkul shuru se 120k "
                "decisions train hota hai; model files aur curves committed model se byte-for-byte compare. Reproducibility ka sab se "
                "sakht test.", run_retrain_w5),
               ("tests", "Step 10 tests (≈10 s)", *_tests(["tests/test_step10_priority_reward.py"]))],
    "results": [("all_tests", "Saare automated tests (≈1 min)", "Poore project ke saare automated tests (<code>tests/</code>) ek saath.",
                 lambda: run_tests(["tests"]))],
    "verification": [(n, lbl, f"<code>verification/{n}.py</code> dobara chalta hai (output temp folder mein) aur committed "
                      f"<code>verification/output/{n}.md</code> se compare hota hai. {d}", (lambda n=n: run_verification(n))) for n, lbl, d in (
        ("v1_independent_actors", "V1 · Independent actors (≈30 s)", "Har actor alag, critic ke baghair evaluation same."),
        ("v2_consolidated_comparison", "V2 · Consolidated comparison (≈5 s)", "Saare methods ki ek table."),
        ("v5_priority_results", "V5 · Priority results (≈1 min)", "Weights ka chunao aur per-priority nateeje."),
        ("v6_imbalance_generalization", "V6 · Department imbalance (≈1–2 min)", "Validation seeds, zero-shot."),
        ("v4_cloud_sensitivity", "V4 · Cloud sensitivity (≈5 min)", "Validation seeds, Cloud capacity / delay badal kar."),
        ("v3_reproduce", "V3 · Reproduction (≈5–10 min)", "Tests, teeno evaluations aur ek model ki retraining."))],
}

def start_job(step, option):
    with JOB.lock:
        if JOB.running:
            return False
        JOB.running, JOB.step, JOB.option = True, step, option
        JOB.log, JOB.verdicts, JOB.started, JOB.finished = [], [], time.time(), None
    fn = {o: f for o, _, _, f in RUNS[step]}[option]

    def work():
        try:
            JOB.verdicts = fn()
        except Exception as e:                                   # shown to the user, never hidden
            JOB.write(f"ERROR: {type(e).__name__}: {e}")
            JOB.verdicts = [{"check": "Run", "ok": False, "detail": f"{type(e).__name__}: {e}"}]
        finally:
            JOB.finished, JOB.running = time.time(), False
    threading.Thread(target=work, daemon=True).start()
    return True


def facts():
    """Numbers quoted in the explanation texts, computed from the committed files (never typed by hand)."""
    b = _baseline_runs()
    crit = lambda k: jload(os.path.join(b[k], "summary.json"))["summary"]["critical_deadline_violation_rate"]["mean"]
    s9 = jload("results/step9_final/summary.json")["results"]["overload"]
    A, R = jload("results/step10_final/summary.json"), jload("results/step10_final/step9_reference/summary.json")
    w5, w1 = _prio(A, "slow_network", "MAPPO_best"), _prio(R, "slow_network", "MAPPO_best")
    cloud = lambda S: S["results"]["slow_network"]["MAPPO (best)"]["action_shares"]["Cloud"]
    v4 = jload("verification/output/v4_cloud_sensitivity.json")
    v4m = lambda key, k: statistics.mean(v4[key][f"W5 trainseed{i}"][k] for i in range(5))
    v6 = jload("verification/output/v6_imbalance_generalization.json")["learned"]
    v6e = lambda src: statistics.mean(x["icu_shares"]["Edge"] for x in v6[f"MAPPO W5 (Step 10) | trained on {src}"])
    cfg, slow = config_of(SCEN["default"]), config_of(SCEN["slow_network"])
    caps = {d["name"]: d["capacity"] for d in cfg["departments"]}
    off, dl = cfg["offloading"], cfg["deadlines"]
    m7 = jload(glob.glob("results/step7_final/training_runs/*/manifest.json")[0])
    m9 = jload(glob.glob("results/step9_final/training_runs/*/manifest.json")[0])
    c7, c9 = m7["dqn_config"], m9["mappo_config"]
    s10 = jload("results/step10_final/summary.json")["results"]
    best = lambda S, sc: S[sc]["MAPPO (best)"]
    return {"cap_e": f"{caps['Emergency']:g}", "cap_i": f"{caps['ICU']:g}", "cap_g": f"{caps['General_Ward']:g}",
            "cloud_cap": f"{off['cloud_capacity']:g}", "cloud_delay": off["cloud_network_delay"], "nb_delay": off["neighbour_network_delay"],
            "cloud_thr": f"{off['cloud_threshold']:g}", "off_thr": f"{off['offload_threshold']:g}",
            "slow_cloud": slow["offloading"]["cloud_network_delay"], "slow_nb": slow["offloading"]["neighbour_network_delay"],
            "dl_c": dl["Critical"], "dl_h": dl["High"], "dl_r": dl["Routine"], "penalty": f"{cfg['rl_environment']['deadline_miss_penalty']:g}",
            "window": cfg["simulation"]["n_steps"], "drain": cfg["evaluation"]["drain_limit"],
            "dqn_hidden": c7["hidden_size"], "dqn_buffer": f"{c7['buffer_size']:,}", "dqn_gamma": f"{c7['gamma']:g}",
            "dqn_scale": f"{c7['reward_scale']:g}", "dqn_steps": f"{m7['total_steps'] // 1000}k", "dqn_double": "Double DQN" if c7["double_dqn"] else "DQN",
            "eps0": f"{c7['epsilon_start']:g}", "eps1": f"{c7['epsilon_end']:g}",
            "m_hidden": c9["hidden_size"], "m_gamma": f"{c9['gamma']:g}", "m_lambda": f"{c9['gae_lambda']:g}", "m_clip": f"{c9['clip_eps']:g}",
            "m_rollout": f"{c9['rollout_decisions']:,}", "m_epochs": c9["ppo_epochs"], "m_scale": f"{c9['reward_scale']:g}",
            "m_steps": f"{m9['total_steps'] // 1000}k", "train_lo": f"{m9['training_workload_seed_range'][0]:,}",
            "train_hi": f"{m9['training_workload_seed_range'][1]:,}",
            "n_tests": sum(len(re.findall(r"^\s*def test_", open(f, encoding="utf-8").read(), re.M)) for f in glob.glob("tests/*.py")),
            "w5_over": f1(-best(s10, "overload")["return"][0]), "w5_slow": f1(-best(s10, "slow_network")["return"][0]),
            "ect_slow": f1(-s10["slow_network"]["ECT"]["return"][0]), "fcfs_slow": f1(-s10["slow_network"]["FCFS"]["return"][0]),
            "s5_crit_overload": pct(crit("overload")), "s5_crit_slow": pct(crit("slow_network")),
            "s9_mappo": f1(-s9["MAPPO (best)"]["return"][0]), "s9_ect": f1(-s9["ECT"]["return"][0]), "s9_fcfs": f1(-s9["FCFS"]["return"][0]),
            "s10_crit_w1": pct(w1["Critical"][0][0], 2), "s10_crit_w5": pct(w5["Critical"][0][0], 2),
            "s10_clat_w1": f2(w1["Critical"][1][0]), "s10_clat_w5": f2(w5["Critical"][1][0]),
            "s10_rout_w1": f2(w1["Routine"][1][0]), "s10_rout_w5": f2(w5["Routine"][1][0]), "s10_routviol_w5": pct(w5["Routine"][0][0]),
            "s10_cloud_w5": pct(cloud(A)), "s10_cloud_w1": pct(cloud(R)),
            "v4_cost50": f1(-v4m("overload | cloud capacity 50", "return")), "v4_cost6": f1(-v4m("overload | cloud capacity 6", "return")),
            "v4_crit50": pct(v4m("overload | cloud capacity 50", "critical_violation")), "v4_crit6": pct(v4m("overload | cloud capacity 6", "critical_violation")),
            "v6_edge_default": pct(v6e("default")), "v6_edge_overload": pct(v6e("overload"))}


# ============================================================================
# Overview and concepts pages (numbers from facts(), never typed by hand)
# ============================================================================
def data_overview():
    F = facts()
    S7, S9, S10 = (jload(f"results/{s}_final/summary.json") for s in ("step7", "step9", "step10"))
    o = lambda S, k: f1(-S["results"]["overload"][k]["return"][0])
    n_steps = sum(s["id"].startswith("step") for s in STEPS)
    cost = data_results()[1]
    cost["title"] = "Final comparison: cost = −return (kam = behtar), test seeds, error bar = 95% CI"
    return [
        {"type": "kpis", "big": True, "title": "Ek nazar mein (test seeds 2000–2029, har method sirf ek dafa)", "items": [
            {"label": "Overload cost · Priority MAPPO", "value": o(S10, "MAPPO (best)"),
             "note": f"ECT {o(S7, 'ECT')} · FCFS {o(S7, 'FCFS')} · DQN {o(S7, 'DQN (best)')} (kam = behtar)"},
            {"label": "Slow network · Critical violation", "value": f"{F['s10_crit_w1']} → {F['s10_crit_w5']}",
             "note": "MAPPO → Priority MAPPO (W5)"},
            {"label": "Automated tests", "value": str(F["n_tests"]), "note": "Run button se khud chala kar dekhein"},
            {"label": "Steps complete", "value": f"{n_steps} / {n_steps}", "note": "Step 11 (federated / privacy) approval ka intezar"}]},
        {"type": "arch", "title": "System ka naqsha — task kahan ja sakta hai",
         "cloud": f"capacity {F['cloud_cap']} · delay {F['cloud_delay']} steps",
         "nodes": [["Emergency", F["cap_e"]], ["ICU", F["cap_i"]], ["General Ward", F["cap_g"]]],
         "edge": f"Edge (padosi) · delay {F['nb_delay']}", "cloud_link": f"Cloud · delay {F['cloud_delay']}"},
        {"type": "pipeline", "title": "Project ka safar — kisi bhi step par click karein"},
        cost,
        {"type": "points", "title": "Ahem nateeje", "items": [
            ["🏆", "MAPPO sab se behtar", f"Stress scenarios mein MAPPO / Priority MAPPO ka cost ECT aur FCFS se kaafi kam "
             f"(overload: Priority MAPPO {F['w5_over']}, ECT {F['s9_ect']}, FCFS {F['s9_fcfs']})."],
            ["🚑", "Critical patients mehfooz", f"Priority-aware reward (W5) se slow network mein Critical violations "
             f"{F['s10_crit_w1']} → {F['s10_crit_w5']}. Keemat: Routine tasks thora zyada intezar ({F['s10_rout_w1']} → {F['s10_rout_w5']} steps)."],
            ["🤖", "Single-agent DQN kaafi nahi", "Centralized DQN FCFS se behtar magar simple ECT heuristic ko kisi scenario mein beat nahi "
             "karta — is liye multi-agent (har department ka agent) design."],
            ["☁️", "Sab se bada caveat", f"Learned policies muft aur bare Cloud par tiki hain: Cloud capacity {F['cloud_cap']} → 6 par "
             f"overload cost {F['v4_cost50']} → {F['v4_cost6']} (V4, validation seeds, frozen models)."],
            ["🔁", "Reproducible", "Frozen models ki re-evaluation aur ek model ki poori retraining byte-for-byte same nikli (V3). "
             "Har step ka Run button yahi dobara check karta hai."]]},
        {"type": "points", "title": "Aage kya — supervisor ke faisle ka intezar (abhi kuch implement nahi hua)", "tone": "pending", "items": [
            ["❓", "Cloud ka model", "Cloud ki cost ya limited capacity model mein shamil ki jaye ya nahi."],
            ["❓", "Network aur energy", "Network delay task ke size (payload) par depend kare; energy metric shamil ho."],
            ["❓", "Department imbalance", "Imbalance scenario (sirf ICU bhara) training mein shamil ho, jahan padosi madad kar sakte hain."],
            ["🔒", "Step 11: privacy", "Federated (FedAvg) / SplitFed training — departments data share kiye baghair mil kar seekhein."]]}]


def data_concepts():
    F = facts()
    g = [("🏥 System", [
            ("Edge node", f"Har department ka apna chhota server. Har time-step mein fixed kaam (capacity) kar sakta hai: Emergency {F['cap_e']}, "
                          f"ICU {F['cap_i']}, General Ward {F['cap_g']} compute units."),
            ("Task", "Ek medical computing job (maslan lab result analysis). Har task ka department, priority, compute demand aur arrival time hota hai."),
            ("Offloading", "Task ko apne node ke bajaye kahin aur process karwana — padosi department (Edge) ya Cloud par."),
            ("Local / Edge / Cloud", f"Teen faisle: Local = apna node (delay 0); Edge = padosi department (delay {F['nb_delay']}); "
                                     f"Cloud = shared server (capacity {F['cloud_cap']}, delay {F['cloud_delay']})."),
            ("Backlog", "Kisi node par baqi kaam ka jor (raste mein aane wale tasks samet). Zyada backlog = lamba intezar."),
            ("ρ (load ÷ capacity)", "Department par aane wala average kaam ÷ us ki capacity. ρ > 1 = department akela kaam waqt par khatam nahi kar sakta."),
            ("Latency", "Task aane se khatam hone tak ke steps. Kam latency = jaldi nateeja."),
            ("Priority aur deadline", f"Critical, High, Routine. Deadline = arrival + {F['dl_c']} / {F['dl_h']} / {F['dl_r']} steps."),
            ("Deadline violation", "Task deadline ke baad khatam hua ya khatam hi nahi hua. Critical violation sab se khatarnak."),
            ("Scenario", f"Workload ki halat: default (halka), overload (arrivals ×5), slow network (arrivals ×3, delays {F['slow_cloud']} / {F['slow_nb']}), "
                         "department imbalance (sirf ICU ×6).")]),
         ("🧭 Policies (faisla karne ke tareeqe)", [
            ("FCFS", "Fixed rule, priority nahi dekhta: bara task → Cloud; apna node khali → Local; warna sab se halka padosi. Baseline 1."),
            ("ECT", "Earliest Completion Time heuristic: har raaste ka andaza lagao ke task kab khatam hoga, sab se jaldi wala chuno. Bina training, mazboot baseline."),
            ("always Cloud / Local / random", "Sanity check ke liye simple fixed policies."),
            ("DQN", "Ek centralized RL agent jo saare departments ke faisle karta hai (Step 7)."),
            ("MAPPO", "Har department ka apna RL agent; training mein ek central critic madad karta hai (Step 9)."),
            ("Priority MAPPO (W5)", "MAPPO jis ke reward mein Critical task ka wazan 5, High 2, Routine 1 (Step 10).")]),
         ("🧠 Reinforcement learning", [
            ("Agent", "Woh program jo dekhta hai, faisla karta hai aur nateeje se seekhta hai."),
            ("Observation", "Faisle ke waqt agent jo dekhta hai: nodes ka backlog, task ka size, priority, deadline, waqt. Step 6: 13 values; har department agent: 10 values."),
            ("Action", "Faisla: Local, Edge ya Cloud."),
            ("Reward", f"Har task khatam hone par: −latency − {F['penalty']} × (deadline miss). Step 10 mein priority ka wazan bhi."),
            ("Return aur cost", "Return = ek workload (episode) ke saare rewards ka jor. Cost = −return: kam = behtar. Charts mein cost dikhaya gaya hai."),
            ("Episode", f"Ek poora workload: {F['window']} steps tak naye tasks, phir baqi kaam khatam (drain)."),
            ("Actor", "Policy network: observation se faisla. MAPPO mein har department ka alag actor."),
            ("Critic", "Value network: andaza ke halat kitni achi hai. MAPPO mein ek central critic, sirf training mein."),
            ("CTDE", "Centralized Training, Decentralized Execution: training mein poori maloomat, deployment mein har agent sirf apni local maloomat se."),
            ("PPO", f"Policy ko chhote, mehfooz qadmon mein update karna (clip {F['m_clip']}) taake training stable rahe."),
            ("GAE aur γ", f"γ = future rewards ka wazan ({F['m_gamma']}). GAE (λ {F['m_lambda']}) advantage ka kam-shor andaza."),
            ("Checkpoint (best / final)", "Training ke dauran save kiya gaya model. Best = jis ka validation result sab se acha; final = training ke aakhir wala.")]),
         ("📏 Evaluation", [
            ("Seeds", f"Random workload ka number. Same seed = bilkul same tasks. Training {F['train_lo']}–{F['train_hi']}, validation 1000–1029, test 2000–2029."),
            ("Validation vs test", "Validation seeds par settings chuni jati hain; test seeds par har method sirf ek dafa — taake nateeja inflate na ho."),
            ("Training seeds (5)", "Har method 5 alag random shuruaat se train hota hai; nateeja in ka mean."),
            ("95% CI (error bar)", "Mean ke gird range: 5 training seeds mein kitna farq. Chhota bar = zyada bharosa."),
            ("Paired comparison", "Har method same workloads par — farq sirf policy ka hota hai, qismat ka nahi."),
            ("Pre-registered rule", "Experiment se pehle likha gaya faisla ka qaida (maslan 'sab se kam Critical violation wala weight'), taake baad mein pasand ka nateeja na chuna jaye."),
            ("Reproducibility", "Same code + same seed = byte-for-byte same nateeja. Har run folder mein config, seeds aur git commit.")]),
         ("🔒 Aage (Step 11+, pending)", [
            ("Federated learning (FedAvg)", "Departments apna data share nahi karte; sirf model ke weights average hote hain."),
            ("SplitFed", "Model ka ek hissa department par, doosra server par — raw data department se bahar nahi jata."),
            ("Privacy", "Thesis ka aakhri maqsad: patient data department mein rahe, phir bhi agents mil kar seekhein.")])]
    return [{"type": "glossary", "groups": [{"title": t, "items": [list(i) for i in items]} for t, items in g]}]


# ============================================================================
# Step texts (explanations, Roman Urdu with English terms). {name} = a number from facts().
# ============================================================================
PHASES = {"intro": "Shuruaat", "base": "Simulator aur baseline", "single": "Single-agent RL", "multi": "Multi-agent RL",
          "final": "Nateeje aur jaanch"}
TEXT_KEYS = ("short", "takeaway", "goal", "did", "how", "result", "limits")
STEPS = [
    dict(id="overview", label="Overview", num="★", phase="intro", kind="page", title="Project ek nazar mein",
         short="Hospital ke departments ke liye privacy-preserving, priority-aware task scheduling — multi-agent reinforcement learning se."),
    dict(id="concepts", label="Concepts", num="?", phase="intro", kind="page", title="Asaan alfaaz mein concepts",
         short="Dashboard mein aane wali har term ki seedhi Roman Urdu explanation."),
    dict(id="step1", label="Step 1", num="1", phase="base", title="Mini hospital simulator",
         short="Departments ke edge nodes, fixed capacity aur FCFS queues — bina AI ke.",
         takeaway="Poori thesis ka 'lab': ek chhota, reproducible hospital jahan har policy bilkul same halaat mein test hoti hai.",
         goal="Ek simple magar sahi hospital model banana jis par baad mein har scheduling method (FCFS, ECT, DQN, MAPPO) <b>same halaat</b> "
              "mein chal kar compare ho sake. Is step mein koi AI nahi — sirf buniyad.",
         did="<ul><li>3 departments: <b>Emergency</b>, <b>ICU</b>, <b>General Ward</b>.</li><li>Har department ka apna <b>edge node</b> "
             "(chhota local server) jis ki har time-step ki fixed capacity hai: {cap_e} / {cap_i} / {cap_g} compute units.</li>"
             "<li>Har node ki apni queue, <b>FCFS</b> tarteeb (jo pehle aaya, pehle chala).</li><li>Bara task ek step mein khatam na ho "
             "to baqi kaam agle step mein jaari rehta hai (<i>carry-over</i>).</li></ul>",
         how="<p>Simulator ek <b>discrete clock</b> chalata hai. Har step par: naye tasks queue mein aate hain → har node apni capacity "
             "jitna kaam karta hai → jo task khatam ho us ka completion time likha jata hai.</p><p><b>Latency</b> = completion time − "
             "arrival time (0 = usi step mein khatam).</p><p>Misaal: ICU ki capacity {cap_i} hai; queue mein 6 units ka kaam ho to "
             "khatam hone mein 2 steps lagte hain.</p>",
         result="Nateeje tab mein demo run (seed 42): har task kab aaya, kahan chala aur kab khatam hua. Automated tests capacity, "
                "queue order aur carry-over check karte hain.",
         limits="Capacities aur task sizes <b>placeholder</b> hain (asal hospital data, MIMIC-IV, ka access abhi pending hai). "
                "Step 1 mein departments ek doosre ki madad nahi karte — yeh Step 3 mein aata hai.",
         files=["env/task.py", "env/edge_node.py", "env/hospital_env.py", "demo_simulation.py"]),
    dict(id="step2", label="Step 2", num="2", phase="base", title="Medical task generation",
         short="Har department aur priority ke liye Poisson arrivals, seeded aur reproducible.",
         takeaway="Tasks random magar <b>reproducible</b> aate hain: same seed = bilkul same tasks, is liye har method ka muqabla insaaf se hota hai.",
         goal="Hospital ke medical tasks (lab analysis, monitoring alerts, reports) ka ek realistic aur dohraya ja sakne wala stream banana.",
         did="<ul><li>Har (department, priority) jori ke tasks <b>Poisson process</b> se: har step mein average <i>arrival rate</i> tasks.</li>"
             "<li>Har task ki <b>compute demand</b> us ki priority ki range mein uniform random.</li><li>Teen priorities: <b>Critical</b>, "
             "<b>High</b>, <b>Routine</b>.</li><li>Ek step ke tasks seeded random tarteeb mein, taake koi department hamesha pehle na aaye.</li>"
             "<li>Stress scenarios: <b>overload</b> (arrivals ×5), <b>slow network</b> (arrivals ×3, Cloud delay {slow_cloud}, padosi delay "
             "{slow_nb}), <b>department imbalance</b> (sirf ICU ×6), aur edge cases (empty workload, overdue Critical).</li></ul>",
         how="<p>Arrival rate 0.10 = average har 10 steps mein ek task.</p><p><b>ρ (load ÷ capacity)</b> = har priority ka (arrival rate × "
             "average demand) jor ÷ node ki capacity. ρ &gt; 1 ka matlab: department akela apna kaam waqt par khatam nahi kar sakta, "
             "queue lambi hoti jati hai.</p>",
         result="Chart: default mein har department ρ &lt; 1 (aaram); <b>overload</b> mein har department ρ &gt; 1 (padosiyon ke paas bhi "
                "jagah nahi); <b>department imbalance</b> mein sirf ICU bhara hai.",
         limits="Arrival rates aur demand ranges <b>synthetic placeholders</b> hain. MIMIC-IV access milne par sirf config badlegi, code nahi.",
         files=["env/task_generator.py", "configs/environment.yaml", "configs/scenarios/"]),
    dict(id="step3", label="Step 3", num="3", phase="base", title="Local / Edge / Cloud routing",
         short="Task apne node, padosi department ya shared Cloud par ja sakta hai.",
         takeaway="Har task ke teen raaste: Local, Edge (padosi) ya Cloud. Yahi faisla baad mein AI agents seekhte hain.",
         goal="Department bhara ho to task kahin aur bhejne ka intezam (<b>offloading</b>). Yahi thesis ka asal scheduling faisla hai.",
         did="<ul><li><b>Local</b>: apna node, network delay 0.</li><li><b>Edge</b>: padosi department ka node, delay {nb_delay}; padosi "
             "ki <b>asal queue</b> share hoti hai.</li><li><b>Cloud</b>: ek shared server, capacity {cloud_cap}, delay {cloud_delay}.</li>"
             "<li><b>Destination-ready FCFS</b>: raste (transit) mein task CPU nahi leta, pohanch kar queue mein lagta hai.</li>"
             "<li>Ek fixed <b>FCFS routing rule</b> = thesis ka Baseline 1.</li></ul>",
         how="<p>FCFS rule (priority nahi dekhta):</p><ol><li>demand &gt; {cloud_thr} → <b>Cloud</b></li><li>warna apna backlog ≤ "
             "{off_thr} → <b>Local</b></li><li>warna → <b>Edge</b>: sab se kam backlog wala padosi</li></ol><p><b>Backlog</b> = node ko "
             "diye gaye saare adhoore kaam ka jor (raste wale tasks samet).</p>",
         result="Automated tests routing rule, network delay ki timing aur padosi ki queue sharing check karte hain.",
         limits="Padosi ka chunao <b>raw backlog</b> se hota hai, capacity se nahi, is liye kabhi kabhi slow padosi chuna jata hai. Delays "
                "fixed hain (task ke size par depend nahi karte) aur Cloud ki koi cost nahi. Yeh modelling sawal supervisor ke saath pending hain.",
         files=["env/task_router.py", "env/cloud_node.py"]),
    dict(id="step4", label="Step 4", num="4", phase="base", title="Priority aur deadlines",
         short="Critical, High, Routine ki deadlines; har task ka deadline status track hota hai.",
         takeaway="Har task ki deadline us ki priority se: Critical {dl_c}, High {dl_h}, Routine {dl_r} steps.",
         goal="Hospital mein sab tasks barabar nahi: Critical task ki der patient ke liye khatarnak ho sakti hai. Is liye har task ki "
              "deadline track karna.",
         did="<ul><li>Deadline = arrival time + priority offset.</li><li>Har task ka status: <b>on_time</b>, <b>completed_late</b>, "
             "<b>overdue_unfinished</b>, <b>not_yet_due</b>.</li><li>Violation rate = (late + overdue) ÷ decided tasks; Critical ke liye alag.</li></ul>",
         how="<p>Misaal: Critical task step 7 par aaya → deadline = 7 + {dl_c}. Us waqt tak (ya pehle) khatam = on time; baad mein = "
             "violation.</p><p><b>not_yet_due</b> tasks miss nahi gine jate — un ka faisla abhi nahi hua.</p>",
         result="Deadlines is step mein sirf <b>track</b> hoti hain, FCFS in par faisla nahi karta. Step 10 mein reward inhi deadlines se weight hota hai.",
         limits="Deadline offsets placeholder hain, clinical data se nahi.",
         files=["env/task.py", "env/metrics.py"]),
    dict(id="step5", label="Step 5", num="5", phase="base", title="FCFS baseline + evaluation protocol",
         short="Ek fixed 'imtihaan' jis par har policy naapi jati hai; 30 seeds, 6 scenarios.",
         takeaway="Har method isi protocol, inhi seeds aur inhi metrics se naapa jata hai — koi apne liye alag tareeqa nahi chunta.",
         goal="Insaaf ka muqabla: har method ka nateeja same workloads aur same definitions par.",
         did="<ul><li>Measurement window: {window} steps (naye tasks sirf yahan aate hain).</li><li>Phir <b>drain</b>: naye tasks band, zyada "
             "se zyada {drain} steps tak baqi kaam khatam hone diya jata hai.</li><li>Metrics: throughput, average latency, deadline "
             "violation (Critical alag), offloading communication (synthetic payload).</li><li>FCFS baseline: 6 scenarios × 30 seeds (0–29).</li>"
             "<li>Har run folder mein config, seeds, git commit aur package versions.</li></ul>",
         how="<p>Har method ko same seed par <b>same tasks</b> milte hain (paired comparison). Step 7 se seeds ke teen alag set: training "
             "workloads ({train_lo}–{train_hi}), validation 1000–1029 (settings chunne ke liye), test 2000–2029 (aakhir mein sirf ek dafa).</p>",
         result="Default load mein FCFS koi deadline miss nahi karta. Stress mein bohat kamzor: Critical violations overload mein "
                "{s5_crit_overload}, slow network mein {s5_crit_slow}.",
         limits="FCFS jaan-bujh kar simple (priority-blind) hai — yeh neeche ki line hai. Mazboot baseline ECT Step 7 mein aata hai.",
         files=["env/evaluation.py", "run_baseline_fcfs.py", "results/baselines/"]),
    dict(id="step6", label="Step 6", num="6", phase="single", title="Gymnasium environment",
         short="Simulator ko RL interface (reset / step) mein wrap kiya.",
         takeaway="Simulator ab ek RL 'game' hai: agent dekhta hai (observation), faisla karta hai (action), aur score paata hai (reward).",
         goal="Simulator ko standard <b>Gymnasium</b> interface mein wrap karna taake koi bhi RL algorithm is par seekh sake.",
         did="<ul><li>Har naye task par ek faisla: <b>Local / Edge / Cloud</b> (Discrete(3)).</li><li><b>Observation, 13 values</b>: teen "
             "departments ka backlog ÷ capacity (3), Cloud backlog ÷ capacity (1), task demand ÷ apni capacity (1), source department "
             "one-hot (3), priority one-hot (3), deadline offset (1), waqt (1).</li><li><b>Reward</b> = −latency − {penalty} × deadline miss, "
             "task khatam hone par.</li></ul>",
         how="<p>Episode = ek workload ({window} steps window + drain). Episode ka total reward (<b>return</b>) = −(saari latency ka jor) − "
             "{penalty} × (misses). Is liye <b>cost = −return</b>: kam = behtar.</p><p>Agent ko sirf woh pata hai jo faisle ke waqt "
             "maloom hai (future tasks nahi) — environment partially observable hai.</p>",
         result="Agar agent FCFS ke faisle kare to 6 configs × 30 seeds par Step 5 ka nateeja <b>bilkul</b> dohrata hai (tested). Sanity check: "
                "overload mein 'always Cloud' FCFS se kaafi behtar, kyunke Cloud bara (capacity {cloud_cap}) aur muft hai.",
         limits="Penalty {penalty} ek design choice hai (= sab se lambi deadline), data se nahi. Energy aur Cloud ki cost reward mein nahi.",
         files=["env/gym_env.py", "check_simple_policies.py"]),
    dict(id="step7", label="Step 7", num="7", phase="single", title="DQN baseline (single agent)",
         short="Ek centralized DQN — debugging ke baad bhi ECT se kamzor.",
         takeaway="Pehla AI agent (DQN) FCFS se behtar magar simple ECT heuristic se kamzor — is liye multi-agent approach.",
         goal="Single-agent RL baseline: ek centralized agent jo saare departments ke faisle kare.",
         did="<ul><li><b>ECT heuristic</b>: har raaste ka andaza delay + ⌈(backlog + demand) ÷ capacity⌉ − 1; sab se jaldi wala chuno. "
             "Bina training, mazboot baseline.</li><li><b>DQN</b>: Q-network (13 → {dqn_hidden} → {dqn_hidden} → 3), replay buffer "
             "{dqn_buffer}, target network, ε-greedy exploration ({eps0} → {eps1}).</li><li>Debugging: har dafa <b>ek</b> tabdeeli "
             "(reward scaling, γ, Double DQN, lambi training), plan pehle commit, sirf validation seeds.</li><li>Frozen: {dqn_double}, "
             "reward × {dqn_scale}, γ {dqn_gamma}, {dqn_steps} decisions — pre-registered rule se.</li></ul>",
         how="<p>DQN har action ki <b>Q-value</b> (aage milne wala total reward) seekhta hai aur sab se bari Q wala action chunta hai.</p>"
             "<p>Training: 5 seeds × 3 scenarios. Har run ka <b>best checkpoint</b> validation par chuna gaya, phir test seeds 2000–2029 "
             "par <b>sirf ek dafa</b> evaluation.</p>",
         result="DQN overload mein FCFS se behtar, lekin <b>ECT ko kisi scenario mein beat nahi karta</b>, aur training unstable hai "
                "(Nateeje tab ke learning curves).",
         limits="Single agent ko poore hospital ki global observation chahiye — privacy aur scalability ke liye theek nahi. Yahi Step 8–9 ki wajah hai.",
         files=["agents/heuristics.py", "agents/dqn.py", "train_dqn_pilot.py", "Step_Documentation/Step7_DQN_Debugging.md"]),
    dict(id="step8", label="Step 8", num="8", phase="multi", title="Three-agent environment",
         short="Har department ek alag agent, apni local observation ke saath.",
         takeaway="Har department ab apna agent hai jo sirf apni (local) maloomat dekh kar faisla karta hai.",
         goal="Multi-agent setup: har department khud faisla kare. Yahi baad mein federated / privacy-preserving training ki buniyad hai.",
         did="<ul><li><code>MultiAgentHospitalEnv</code>: 3 agents (Emergency, ICU, General Ward).</li><li><b>Local observation, 10 values</b>: "
             "apna backlog pehle, phir do padosi (3), Cloud (1), task demand (1), priority one-hot (3), deadline (1), waqt (1).</li>"
             "<li>Har task ka reward us ke <b>source department</b> ko.</li><li>Faisle usi seeded tarteeb mein jo FCFS / ECT / DQN ki thi.</li></ul>",
         how="<p>Jis department mein task aata hai, wahi agent us ka faisla karta hai (turn-based). Teeno agents ke rewards ka jor = Step 6 "
             "ka reward (har step par tested).</p><p>Global state (Step 6 wali 13 values) alag se milti hai — sirf training ke critic ke liye.</p>",
         result="Tested: Edge offload padosi ki queue exactly utni badhata hai; FCFS agents Step 5 ko bilkul dohrate hain. Nateeje tab mein "
                "har agent ka cost (FCFS aur ECT rules).",
         limits="Is step mein learning nahi — sirf environment. Agents ek doosre ka patient data nahi dekhte, magar padosiyon ka <b>load</b> "
                "(backlog) observation mein hai.",
         files=["env/multi_agent_env.py", "check_multi_agent_env.py"]),
    dict(id="step9", label="Step 9", num="9", phase="multi", title="MAPPO with CTDE",
         short="Har department ka actor, training mein central critic — sab se kam cost.",
         takeaway="MAPPO teeno scenarios mein sab se kam cost wala method — magar load zyada tar Cloud par bhejta hai.",
         goal="Har department ka apna learning agent, jo mil kar (team reward) hospital ka total cost kam kare.",
         did="<ul><li><b>3 actors</b>, har department ka alag (parameters share nahi): 10 → {m_hidden} → {m_hidden} → 3.</li><li><b>1 "
             "centralized critic</b>: global state (13 values), <b>sirf training mein</b>.</li><li>Team reward × {m_scale}, GAE (γ "
             "{m_gamma}, λ {m_lambda}), PPO clip {m_clip}, rollout {m_rollout} decisions, {m_epochs} epochs.</li><li>Experiments sirf "
             "validation par; frozen setting: {m_steps} training decisions.</li></ul>",
         how="<p><b>CTDE</b> = Centralized Training, Decentralized Execution. Training mein critic poora hospital dekh kar batata hai ke "
             "faisla kitna acha tha; deployment mein har actor <b>sirf apni local observation</b> se faisla karta hai (V1 mein checked).</p>"
             "<p><b>PPO</b> policy ko chhote, mehfooz qadmon mein badalta hai (clip) taake training stable rahe.</p>",
         result="Test seeds: overload cost {s9_mappo} vs ECT {s9_ect} vs FCFS {s9_fcfs}. Default, overload aur slow network — teeno mein "
                "sab se kam cost.",
         limits="Load balancing zyada tar <b>Cloud</b> se hoti hai, padosiyon se bohat kam — overload mein padosi bhi bhare hain aur Cloud "
                "muft aur bara hai. V4: Cloud chhota ho to yeh policy kaafi kharab hoti hai.",
         files=["agents/mappo.py", "train_mappo.py", "Step_Documentation/Step9_MAPPO.md"]),
    dict(id="step10", label="Step 10", num="10", phase="multi", title="Priority-aware reward",
         short="Critical tasks ko zyada wazan — Critical misses taqreeban khatam.",
         takeaway="Critical tasks ko zyada wazan diya: slow network mein Critical violations {s10_crit_w1} → {s10_crit_w5}.",
         goal="Healthcare ki asal zaroorat: resources kam hon to <b>Critical</b> patients pehle. Reward ko priority-aware banana.",
         did="<ul><li>Reward = w<sub>priority</sub> × (−latency − {penalty} × miss).</li><li>Weights deadlines se nikale (andaze se nahi): "
             "W1 = 1/1/1, <b>W5 = 5/2/1</b> ({dl_r} ÷ offset), W25 = 25/4/1 (square).</li><li>Pre-registered rule: validation par sab se "
             "kam Critical violation → <b>W5</b>.</li><li>Phir 5 seeds × 3 scenarios training, test par ek dafa evaluation.</li></ul>",
         how="<p>Critical task ki der ab 5 guna mehngi hai, is liye agent seekhta hai ke Critical task tez raaste par jaye aur Routine thora "
             "intezar kare.</p><p>Evaluation mein return <b>bina weight</b> ke report hota hai, taake Step 9 se seedha muqabla ho.</p>",
         result="Slow network: Critical violation {s10_crit_w1} → {s10_crit_w5}, Critical latency {s10_clat_w1} → {s10_clat_w5}; Routine "
                "latency {s10_rout_w1} → {s10_rout_w5}, Routine violation {s10_routviol_w5}. Overall cost: W5 {w5_slow}, ECT {ect_slow}, FCFS {fcfs_slow}.",
         limits="Cloud ka istemal zyada ({s10_cloud_w5} vs {s10_cloud_w1}, slow network). Weights placeholder deadlines se hain, clinical data se nahi.",
         files=["env/gym_env.py", "train_mappo.py", "Step_Documentation/Step10_Priority_Reward.md"]),
    dict(id="results", label="Results", num="🏆", phase="final", title="Final comparison",
         short="FCFS, ECT, DQN, MAPPO aur Priority MAPPO — same test workloads, same metrics.",
         takeaway="MAPPO aur Priority MAPPO har stress scenario mein sab se behtar; Priority MAPPO Critical patients ke liye sab se mehfooz.",
         goal="Saare methods ka ek jagah, insaaf se muqabla — test seeds 2000–2029 par, har method sirf ek dafa.",
         did="<ul><li><b>FCFS</b>: fixed rule (Step 5).</li><li><b>ECT</b>: heuristic (Step 7).</li><li><b>DQN</b>: single agent, {dqn_steps} "
             "decisions.</li><li><b>MAPPO</b>: multi-agent, {m_steps} decisions.</li><li><b>Priority MAPPO (W5)</b>: Step 10.</li><li><b>always "
             "Cloud</b>: reference.</li></ul>",
         how="<p><b>Cost = −return</b> = har task ki latency + {penalty} × deadline miss ka jor; <b>kam = behtar</b>. Har scenario ka scale "
             "alag hai, is liye har scenario ka alag panel.</p><p>Learned methods: best checkpoint, 5 training seeds ka mean; error bar = 95% "
             "CI. Table mein ★ = us scenario mein sab se kam cost.</p>",
         result="Overload: Priority MAPPO {w5_over}, MAPPO {s9_mappo}, ECT {s9_ect}, FCFS {s9_fcfs}. Slow network: Priority MAPPO {w5_slow}, "
                "ECT {ect_slow}, FCFS {fcfs_slow}.",
         limits="Learned policies <b>muft aur bare Cloud</b> par tiki hain (Checks page, V4). Data synthetic hai. Privacy (federated / SplitFed) "
                "Step 11 se aage — abhi pending.",
         files=["results/step7_final/", "results/step9_final/", "results/step10_final/", "verification/output/v2_consolidated_comparison.md"]),
    dict(id="verification", label="Checks", num="✓", phase="final", title="Verification (supervisor ke sawal)",
         short="Independent actors, reproducibility, cloud sensitivity, department imbalance.",
         takeaway="Har dawa ka saboot: scripts jo koi model ya environment nahi badalte, sirf saved models aur results dobara chalate hain.",
         goal="Supervisor ke sawalon ke jawab, aise checks se jo koi bhi dobara chala sake.",
         did="<ul><li><b>V1</b>: har department ka alag actor; critic hata kar bhi evaluation bilkul same.</li><li><b>V2</b>: saare methods "
             "same test workloads aur metrics par.</li><li><b>V3</b>: saare automated tests, teeno test evaluations aur ek model ki retraining "
             "byte-for-byte same.</li><li><b>V4</b>: Cloud capacity ya delay badal kar frozen policies.</li><li><b>V5</b>: priority weights "
             "ka chunao aur per-priority nateeje.</li><li><b>V6</b>: department imbalance par zero-shot test.</li></ul>",
         how="<p>V4 aur V6 sirf <b>validation</b> seeds use karte hain; koi setting in nateejon se nahi chuni gayi. Run button script dobara "
             "chala kar naya output committed output se compare karta hai.</p>",
         result="<b>V4</b>: overload mein Cloud capacity {cloud_cap} → 6 par cost {v4_cost50} → {v4_cost6}, Critical violation {v4_crit50} → "
                "{v4_crit6} — faisle Cloud par bohat munhasir. <b>V6</b>: sirf ICU bhara ho to default-trained MAPPO ICU ke {v6_edge_default} "
                "tasks padosiyon ko bhejta hai; overload-trained {v6_edge_overload}.",
         limits="V4 aur V6 <b>frozen</b> models ki generalization dikhate hain, un halaat mein retrain kiye gaye models ke nateeje nahi.",
         files=["verification/README.md", "verification/output/"]),
]


# ============================================================================
# Frontend (single page; no external libraries, works offline)
# ============================================================================
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Thesis Dashboard</title><script>try{var t=localStorage.getItem('theme');if(t)document.documentElement.dataset.theme=t}catch(e){}</script><style>
:root{color-scheme:light;--bg:#f3f5fa;--surface:#ffffff;--surface2:#f7f8fc;--ink:#0f172a;--ink2:#475569;--muted:#64748b;--grid:#e6e9f0;--axis:#c5cbd6;
--border:#e2e6ee;--brand:#4f46e5;--brand2:#0d9488;--soft:#eef2ff;--soft2:#e7f7f5;--good:#0b8a3e;--crit:#d03b3b;--warn:#b26b00;--shadow:0 1px 2px rgba(15,23,42,.05),0 6px 20px rgba(15,23,42,.06);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;--q1:#b7d3f6;--q2:#5598e7;--q3:#184f95}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#0b1020;--surface:#131a2c;--surface2:#0f1626;--ink:#e7ebf3;--ink2:#a9b4c8;--muted:#7d89a1;
--grid:#222c42;--axis:#36425c;--border:#243049;--brand:#8b93ff;--brand2:#2dd4bf;--soft:#1d2142;--soft2:#11302f;--good:#3fcf74;--crit:#f07272;--warn:#e0a94a;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--q1:#86b6ef;--q2:#3987e5;--q3:#1c5cab}}
:root[data-theme=dark]{color-scheme:dark;--bg:#0b1020;--surface:#131a2c;--surface2:#0f1626;--ink:#e7ebf3;--ink2:#a9b4c8;--muted:#7d89a1;
--grid:#222c42;--axis:#36425c;--border:#243049;--brand:#8b93ff;--brand2:#2dd4bf;--soft:#1d2142;--soft2:#11302f;--good:#3fcf74;--crit:#f07272;--warn:#e0a94a;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--q1:#86b6ef;--q2:#3987e5;--q3:#1c5cab}
*{box-sizing:border-box}html,body{margin:0}body{background:var(--bg);color:var(--ink);font:14.5px/1.6 "Segoe UI",system-ui,-apple-system,sans-serif}
code{font:12.5px ui-monospace,Consolas,monospace;background:var(--surface2);border:1px solid var(--border);padding:1px 5px;border-radius:5px}
.app{display:flex;min-height:100vh}
aside{width:276px;flex:none;background:linear-gradient(180deg,#0f172a 0%,#1e1b4b 60%,#312e81 100%);color:#e2e8f0;position:sticky;top:0;height:100vh;overflow-y:auto;padding:18px 14px;display:flex;flex-direction:column}
.brand{display:flex;gap:10px;align-items:center;padding:4px 8px 16px;border-bottom:1px solid rgba(255,255,255,.1);margin-bottom:10px}
.logo{width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#2dd4bf,#6366f1);display:grid;place-items:center;font-size:20px;flex:none}
.brand b{display:block;font-size:14px;color:#fff;line-height:1.3}.brand span{font-size:11.5px;color:#a5b4fc}
.ng{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:#818cf8;margin:14px 10px 6px;font-weight:600}
.ni{display:flex;align-items:center;gap:10px;width:100%;text-align:left;border:0;background:none;color:#cbd5e1;padding:7px 10px;border-radius:10px;cursor:pointer;font:inherit;font-size:13.5px}
.ni:hover{background:rgba(255,255,255,.07);color:#fff}.ni.on{background:rgba(255,255,255,.13);color:#fff;box-shadow:inset 3px 0 0 #2dd4bf}
.nn{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:12px;font-weight:700;flex:none;background:rgba(255,255,255,.08);color:#e0e7ff}
.ni.on .nn{background:linear-gradient(135deg,#2dd4bf,#6366f1);color:#fff}.ni .ok{margin-left:auto;color:#5eead4;font-size:12px}
.side-foot{margin-top:auto;padding-top:14px;border-top:1px solid rgba(255,255,255,.1);display:flex;flex-direction:column;gap:8px}
.tbtn{border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);color:#e2e8f0;border-radius:10px;padding:8px;cursor:pointer;font:inherit;font-size:13px}
.tbtn:hover{background:rgba(255,255,255,.12)}#jobInd{font-size:12px;color:#fde68a;display:none;padding:6px 8px;border-radius:8px;background:rgba(253,230,138,.1)}
main{flex:1;min-width:0;padding:26px 34px 60px}.inner{max-width:1180px;margin:0 auto}
.hero{border-radius:20px;padding:26px 28px;color:#fff;display:flex;gap:20px;align-items:center;box-shadow:var(--shadow);position:relative;overflow:hidden}
.hero:after{content:"";position:absolute;right:-60px;top:-60px;width:240px;height:240px;border-radius:50%;background:rgba(255,255,255,.08)}
.ph-intro{background:linear-gradient(120deg,#0e7490,#4f46e5)}.ph-base{background:linear-gradient(120deg,#0f766e,#0e7490)}
.ph-single{background:linear-gradient(120deg,#3730a3,#4f46e5)}.ph-multi{background:linear-gradient(120deg,#5b21b6,#7c3aed)}.ph-final{background:linear-gradient(120deg,#9a3412,#c2410c)}
.hbadge{width:74px;height:74px;border-radius:20px;background:rgba(255,255,255,.16);display:grid;place-items:center;font-size:30px;font-weight:800;flex:none;border:1px solid rgba(255,255,255,.25)}
.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.1em;opacity:.85}.hero h1{margin:2px 0 4px;font-size:27px;line-height:1.25}.hero p{margin:0;opacity:.92;max-width:760px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.chip{font-size:12px;padding:3px 10px;border-radius:999px;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.25)}
.stepper{display:flex;align-items:center;margin:20px 4px 4px;overflow-x:auto;padding:4px 0}
.sdot{flex:none;width:32px;height:32px;border-radius:50%;display:grid;place-items:center;font-size:12.5px;font-weight:700;cursor:pointer;border:2px solid var(--border);background:var(--surface);color:var(--muted)}
.sdot.done{border-color:var(--brand2);color:var(--brand2)}.sdot.on{background:linear-gradient(135deg,#0d9488,#4f46e5);color:#fff;border-color:transparent;transform:scale(1.12)}
.sline{flex:1;min-width:14px;height:2px;background:var(--brand2);opacity:.45}
.tabs{display:flex;gap:6px;margin:20px 0 14px;background:var(--surface);padding:5px;border-radius:14px;border:1px solid var(--border);width:fit-content;max-width:100%;overflow-x:auto;box-shadow:var(--shadow)}
.tab{border:0;background:none;color:var(--ink2);padding:8px 16px;border-radius:10px;cursor:pointer;font:inherit;font-weight:600;white-space:nowrap}
.tab:hover{color:var(--ink)}.tab.on{background:var(--brand);color:#fff}
.takeaway{border-radius:16px;padding:16px 20px;margin:0 0 16px;background:linear-gradient(120deg,var(--soft),var(--soft2));border:1px solid var(--border)}
.takeaway span{font-size:12px;font-weight:700;color:var(--brand);text-transform:uppercase;letter-spacing:.06em}.takeaway p{margin:4px 0 0;font-size:16px;color:var(--ink)}
.secgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.sec{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:16px 18px;box-shadow:var(--shadow)}
.sec h4{margin:0 0 8px;font-size:14.5px;display:flex;align-items:center;gap:8px}.sec h4 i{font-style:normal;width:30px;height:30px;border-radius:9px;display:grid;place-items:center;background:var(--soft)}
.sec p,.sec li{color:var(--ink2)}.sec p{margin:4px 0}.sec ul,.sec ol{margin:4px 0;padding-left:20px}.sec li{margin:3px 0}.sec.wide{grid-column:1/-1}
.sec.warn{border-left:4px solid var(--warn)}.sec.res{border-left:4px solid var(--brand2)}
.files{display:flex;gap:6px;flex-wrap:wrap}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:16px 18px;margin:0 0 16px;box-shadow:var(--shadow)}
.card>h3{margin:0 0 12px;font-size:15px;display:flex;align-items:center;gap:10px}.card>h3 .tog{margin-left:auto}
.note{color:var(--ink2);background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--brand);padding:12px 16px;border-radius:12px;margin:0 0 16px}
.empty{color:var(--ink2);padding:26px;text-align:center;border:2px dashed var(--axis);border-radius:16px;margin:0 0 16px;background:var(--surface)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.kpi{border:1px solid var(--border);border-radius:14px;padding:14px 16px;background:var(--surface2);position:relative;overflow:hidden}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:linear-gradient(180deg,#0d9488,#4f46e5)}
.kpi span{color:var(--muted);font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}.kpi b{display:block;font-size:24px;margin:2px 0;color:var(--ink)}
.kpis.big .kpi b{font-size:30px;background:linear-gradient(120deg,#0d9488,#4f46e5);-webkit-background-clip:text;background-clip:text;color:transparent}
.kpi i{font-style:normal;color:var(--ink2);font-size:12.5px}
.tbl{overflow-x:auto;border:1px solid var(--border);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th{background:var(--surface2);color:var(--ink2);font-weight:600;text-align:left;padding:9px 12px;border-bottom:1px solid var(--border);white-space:nowrap;position:sticky;top:0}
td{padding:8px 12px;border-bottom:1px solid var(--grid);color:var(--ink)}tr:last-child td{border-bottom:0}tbody tr:hover td{background:var(--surface2)}
td.num,th.num{text-align:right;white-space:nowrap}tr.grp td{background:var(--soft);color:var(--brand);font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.05em;padding:6px 12px}
tr.best td{background:color-mix(in srgb,var(--brand2) 10%,transparent);font-weight:600}.star{font-size:11px;color:var(--brand2);margin-left:6px;font-weight:700}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px;vertical-align:-1px}
.tnote{font-size:12px;color:var(--muted);margin-top:6px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.panel{min-width:0}.panel svg{display:block;max-width:100%;height:auto}
.panel h5{margin:0 0 4px;font-size:13px;color:var(--ink2);font-weight:600}
.legend{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 10px;font-size:12px;color:var(--ink2)}.legend span{display:inline-flex;align-items:center;gap:6px;padding:2px 9px;border-radius:999px;background:var(--surface2);border:1px solid var(--border)}
.sw{width:10px;height:10px;border-radius:3px;display:inline-block}.swl{width:16px;height:0;border-top:2px dashed;display:inline-block}
svg text{fill:var(--muted);font-size:11px}svg .lab{fill:var(--ink2)}
.tog{font-size:12px;color:var(--brand);cursor:pointer;border:1px solid var(--border);background:var(--surface2);border-radius:8px;padding:3px 10px;font-weight:600}
pre{white-space:pre-wrap;font:12px/1.5 ui-monospace,Consolas,monospace;color:var(--ink2);max-height:420px;overflow:auto;margin:0;background:var(--surface2);border-radius:10px;padding:12px;border:1px solid var(--border)}
.runintro{display:flex;gap:12px;align-items:flex-start}
.runcard{display:flex;gap:16px;align-items:center;justify-content:space-between;background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:14px 18px;margin:0 0 10px;box-shadow:var(--shadow)}
.runcard h4{margin:0 0 2px;font-size:14.5px}.runcard p{margin:0;color:var(--ink2);font-size:13px}
.btn{border:0;background:linear-gradient(120deg,#0d9488,#4f46e5);color:#fff;border-radius:10px;padding:9px 16px;cursor:pointer;font:inherit;font-weight:600;white-space:nowrap}
.btn:hover{filter:brightness(1.08)}.btn:disabled{opacity:.4;cursor:default;filter:none}.btn.ghost{background:var(--surface);color:var(--brand);border:1px solid var(--border)}
.verdict{display:flex;gap:12px;align-items:flex-start;border-radius:12px;padding:12px 16px;margin:8px 0;border:1px solid var(--border);background:var(--surface)}
.verdict b{font-size:13px;white-space:nowrap;padding:2px 10px;border-radius:999px;color:#fff}.verdict.ok b{background:var(--good)}.verdict.bad b{background:var(--crit)}
.verdict.ok{border-left:4px solid var(--good)}.verdict.bad{border-left:4px solid var(--crit)}.verdict small{display:block;color:var(--ink2)}
.running{display:flex;align-items:center;gap:10px;padding:12px 16px;border-radius:12px;background:var(--soft);color:var(--ink);margin:8px 0}
.spin{width:16px;height:16px;border-radius:50%;border:2px solid var(--brand);border-top-color:transparent;animation:sp 1s linear infinite}@keyframes sp{to{transform:rotate(360deg)}}
details.log{margin-top:8px}details.log summary,details.vout summary{cursor:pointer;color:var(--brand);font-weight:600;font-size:13px}
.pager{display:flex;justify-content:space-between;gap:12px;margin-top:26px}.pager button{flex:1;max-width:48%;text-align:left;border:1px solid var(--border);background:var(--surface);border-radius:14px;padding:12px 16px;cursor:pointer;font:inherit;color:var(--ink);box-shadow:var(--shadow)}
.pager button:hover{border-color:var(--brand)}.pager small{display:block;color:var(--muted);font-size:12px}.pager .nx{text-align:right;margin-left:auto}
.pipe{display:grid;grid-template-columns:repeat(auto-fill,minmax(205px,1fr));gap:12px}
.pcard{text-align:left;border:1px solid var(--border);background:var(--surface2);border-radius:14px;padding:12px 14px;cursor:pointer;font:inherit;color:var(--ink);display:flex;flex-direction:column;gap:4px;transition:transform .12s}
.pcard:hover{transform:translateY(-2px);border-color:var(--brand)}.pcard .pn{display:flex;align-items:center;gap:8px;font-weight:700;font-size:13.5px}
.pcard .pn i{font-style:normal;width:26px;height:26px;border-radius:8px;display:grid;place-items:center;color:#fff;font-size:12px;flex:none}.pcard small{color:var(--ink2);font-size:12px;line-height:1.45}
.pts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.pt{display:flex;gap:12px;border:1px solid var(--border);border-radius:14px;padding:14px;background:var(--surface2)}
.pt .ic{font-size:22px;line-height:1}.pt b{display:block;margin-bottom:2px}.pt p{margin:0;color:var(--ink2);font-size:13px}.pending .pt{border-style:dashed}
.arch{width:100%;height:auto;display:block}.arch .bx{fill:var(--surface2);stroke:var(--border);stroke-width:1.5}.arch .cl{fill:var(--soft);stroke:var(--brand);stroke-width:1.5}
.arch .cr{fill:none;stroke:var(--brand2);stroke-width:1.5;stroke-dasharray:5 4}.arch .ln{stroke:var(--axis);stroke-width:2;fill:none}.arch .lc{stroke:var(--brand);stroke-width:2;fill:none;stroke-dasharray:6 5}
.arch .le{stroke:var(--brand2);stroke-width:2.5;fill:none}.arch .t1{fill:var(--ink);font-size:14px;font-weight:700}.arch .t2{fill:var(--ink2);font-size:12px}.arch .t3{fill:var(--brand2);font-size:12px;font-weight:600}
.arch .t4{fill:var(--brand);font-size:12px;font-weight:600}.arch .pill{fill:var(--brand);opacity:.14}.arch .tp{fill:var(--brand);font-size:11.5px;font-weight:700}
.gsearch{width:100%;max-width:420px;padding:10px 14px;border-radius:12px;border:1px solid var(--border);background:var(--surface);color:var(--ink);font:inherit;margin:0 0 16px}
.ggrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-bottom:22px}.gtitle{font-size:16px;margin:4px 0 10px}
.gcard{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:13px 15px;box-shadow:var(--shadow)}.gcard b{color:var(--brand)}.gcard p{margin:4px 0 0;color:var(--ink2);font-size:13px}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:10px;padding:7px 10px;font-size:12px;box-shadow:var(--shadow);display:none;z-index:9;max-width:320px}
.md h2{font-size:14px;margin:12px 0 6px}.md p{color:var(--ink2);margin:4px 0}.md .tbl{margin:8px 0}
.menu{display:none}
@media(max-width:900px){.app{flex-direction:column}aside{position:static;width:auto;height:auto;padding:12px}.navlist{display:flex;overflow-x:auto;gap:4px}.ng{display:none}
.ni{flex:none;width:auto}.ni small,.ni .ok{display:none}.side-foot{flex-direction:row}main{padding:16px}.secgrid{grid-template-columns:1fr}.hero{flex-direction:column;align-items:flex-start;padding:20px}
.runcard{flex-direction:column;align-items:flex-start}}
</style></head><body><div class="app"><aside><div class="brand"><div class="logo">🏥</div><div><b>SplitFed-MADRL</b><span>Thesis dashboard · Usama Manzoor</span></div></div>
<div class="navlist" id="nav"></div><div class="side-foot"><div id="jobInd"></div><button class="tbtn" id="themeBtn"></button></div></aside>
<main><div class="inner" id="page"></div></main></div><div id="tip"></div>
<script>
const $=s=>document.querySelector(s),tip=$('#tip'),COLORS=__COLORS__,PHASES=__PHASES__;
let STEPS=[],CUR=null,TAB='explain',DATA={},POLL=null,LASTFIN=null,JOB={};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue('--'+v).trim();
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');
const fmt=(v,f)=>v==null||isNaN(v)?'—':f==='pct1'?(100*v).toFixed(1)+'%':f==='pct2'?(100*v).toFixed(2)+'%':f==='n2'?v.toFixed(2):
  Math.abs(v)>=100?v.toLocaleString(undefined,{maximumFractionDigits:1}):v.toFixed(1);
function showTip(e,html){tip.innerHTML=html;tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-330)+'px';tip.style.top=(e.clientY+14)+'px'}
function hideTip(){tip.style.display='none'}
const el=(t,a={},h='')=>{const n=document.createElement(t);for(const k in a)n.setAttribute(k,a[k]);if(h)n.innerHTML=h;return n};
const S='http://www.w3.org/2000/svg',sv=(t,a)=>{const n=document.createElementNS(S,t);for(const k in a)n.setAttribute(k,a[k]);return n};
const svt=(x,y,txt,cls,anchor='middle')=>{const t=sv('text',{x,y,'text-anchor':anchor,class:cls});t.textContent=txt;return t};
function nice(lo,hi){if(lo===hi){hi=lo+1}const span=hi-lo,step=Math.pow(10,Math.floor(Math.log10(span/4)));const m=[1,2,2.5,5,10].find(m=>span/(m*step)<=5)*step;
  return {lo:Math.floor(lo/m)*m,hi:Math.ceil(hi/m)*m,step:m}}
function legend(items){const d=el('div',{class:'legend'});items.forEach(i=>{const s=el('span');s.append(el('i',{class:i.line?'swl':'sw',style:(i.line?'border-color:':'background:')+css(i.color)}),document.createTextNode(i.name));d.append(s)});return d}
/* ---------- tables ---------- */
const isNum=c=>/^[−\-+]?[\d.,]+(%| ±|$| \/| –|–)/.test(String(c).trim());
function tableOf(cols,rows,o={}){const off=o.group?1:0,best=new Set(o.best||[]),cs=cols.slice(off);
  const numc=cs.map((_,j)=>rows.length&&rows.every(r=>isNum(r[j+off])||r[j+off]==='—'));
  let h='<div class="tbl"><table><thead><tr>'+cs.map((c,j)=>`<th class="${numc[j]?'num':''}">${c}</th>`).join('')+'</tr></thead><tbody>';let g=null;
  rows.forEach((r,i)=>{if(o.group&&r[0]!==g){g=r[0];h+=`<tr class="grp"><td colspan="${cs.length}">${g}</td></tr>`}
    h+=`<tr class="${best.has(i)?'best':''}">`+r.slice(off).map((c,j)=>{let v=c;if(o.dot!=null&&j===o.dot-off&&COLORS[c])v=`<span class="dot" style="background:${css(COLORS[c])}"></span>${c}`;
      if(best.has(i)&&o.dot!=null&&j===o.dot-off)v+='<span class="star">★</span>';return `<td class="${numc[j]?'num':''}">${v}</td>`}).join('')+'</tr>'});
  const w=el('div',{},h+'</tbody></table></div>');if(o.best&&o.best.length)w.append(el('div',{class:'tnote'},'★ = '+(o.best_note||'best')));return w}
function addTable(box,cols,rows){const t=el('button',{class:'tog'},'Table dikhao');const w=tableOf(cols,rows);w.style.display='none';
  t.onclick=()=>{const o=w.style.display==='none';w.style.display=o?'block':'none';t.textContent=o?'Table chhupao':'Table dikhao'};box.querySelector('h3').append(t);box.append(w)}
/* ---------- charts ---------- */
function barPanel(p,f,ref,W){W=Math.max(260,Math.round(W||560));const H=250,L=58,R=(ref&&ref.length)?60:12,T=12,B=34,svg=sv('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:'img'});
  let vals=[];p.series.forEach(s=>s.values.forEach((v,i)=>{if(v!=null){vals.push(v);if(s.err&&s.err[i]!=null){vals.push(v+s.err[i],v-s.err[i])}}}));(ref||[]).forEach(r=>vals.push(r.value));
  const nonneg=p.series.every(s=>s.values.every(v=>v==null||v>=0));if(nonneg)vals=vals.map(v=>Math.max(0,v));
  const sc=nice(Math.min(0,...vals),Math.max(0,...vals)),y=v=>T+(H-T-B)*(sc.hi-v)/(sc.hi-sc.lo);
  for(let v=sc.lo;v<=sc.hi+1e-9;v+=sc.step){svg.append(sv('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:css('grid')}));svg.append(svt(L-6,y(v)+4,fmt(v,f),'','end'))}
  const nC=p.categories.length,nS=p.series.length,gw=(W-L-R)/nC,bw=Math.max(4,Math.min(34,(gw-14)/nS-2));
  p.categories.forEach((c,ci)=>{const g0=L+ci*gw+(gw-nS*(bw+2))/2;
    p.series.forEach((s,si)=>{const v=s.values[ci];if(v==null)return;const x=g0+si*(bw+2),y0=y(0),y1=y(v),top=Math.min(y0,y1),h=Math.max(1,Math.abs(y1-y0));
      svg.append(sv('rect',{x,y:top,width:bw,height:h,rx:Math.min(4,bw/3),fill:css(s.color)}));
      const e=s.err&&s.err[ci];if(e){const cx=x+bw/2,lo=nonneg?Math.max(0,v-e):v-e;svg.append(sv('line',{x1:cx,x2:cx,y1:y(lo),y2:y(v+e),stroke:css('ink2'),'stroke-width':1.5}));
        [lo,v+e].forEach(z=>svg.append(sv('line',{x1:cx-4,x2:cx+4,y1:y(z),y2:y(z),stroke:css('ink2'),'stroke-width':1.5})))}
      const hit=sv('rect',{x:x-1,y:T,width:bw+2,height:H-T-B,fill:'transparent'});
      hit.addEventListener('mousemove',ev=>showTip(ev,`<b>${s.name}</b>${c?' · '+c:''}<br>${fmt(v,f)}${e?' ± '+fmt(e,f):''}`));hit.addEventListener('mouseleave',hideTip);svg.append(hit)});
    if(c)svg.append(svt(L+ci*gw+gw/2,H-B+16,c,'lab'))});
  svg.append(sv('line',{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:css('axis')}));
  (ref||[]).forEach(r=>{svg.append(sv('line',{x1:L,x2:W-R,y1:y(r.value),y2:y(r.value),stroke:css('ink2'),'stroke-dasharray':'4 4'}));svg.append(svt(W-R+6,y(r.value)+4,r.label,'','start'))});return svg}
function panelsOf(b,box){const g=el('div',{class:'panels'});box.append(g);return b.panels.map(p=>{const d=el('div',{class:'panel'});if(p.title)d.append(el('h5',{},p.title));g.append(d);return d})}
function blockBars(b,box){const names=[];b.panels.forEach(p=>p.series.forEach(s=>{if(!names.find(n=>n.name===s.name))names.push({name:s.name,color:s.color})}));
  if(names.length>1)box.append(legend(names));const ds=panelsOf(b,box);
  b.panels.forEach((p,i)=>ds[i].append(barPanel(p,p.fmt||b.fmt,b.ref,ds[i].clientWidth)));
  const rows=[];b.panels.forEach(p=>p.categories.forEach((c,ci)=>p.series.forEach(s=>rows.push([p.title||'',c,s.name,fmt(s.values[ci],p.fmt||b.fmt)+(s.err&&s.err[ci]!=null?' ± '+fmt(s.err[ci],p.fmt||b.fmt):'')]))));
  addTable(box,['Panel','Category','Series','Value'],rows)}
function linePanel(p,color,name,W){W=Math.max(260,Math.round(W||560));const H=240,L=58,R=12,T=12,B=34,svg=sv('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:'img'});const s=p.series[0];
  const vals=[...s.lo,...s.hi,...p.refs.map(r=>r.value)],sc=nice(Math.min(...vals),Math.max(...vals)),y=v=>T+(H-T-B)*(sc.hi-v)/(sc.hi-sc.lo);
  const xm=Math.max(...p.x),x=v=>L+(W-L-R)*v/xm;
  for(let v=sc.lo;v<=sc.hi+1e-9;v+=sc.step){svg.append(sv('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:css('grid')}));svg.append(svt(L-6,y(v)+4,fmt(v,'n1'),'','end'))}
  [0,.25,.5,.75,1].forEach(q=>svg.append(svt(x(q*xm),H-B+16,Math.round(q*xm/1000)+'k','')));
  let band='M'+p.x.map((v,i)=>`${x(v)},${y(s.hi[i])}`).join('L')+'L'+p.x.map((v,i)=>`${x(v)},${y(s.lo[i])}`).reverse().join('L')+'Z';
  svg.append(sv('path',{d:band,fill:css(color),opacity:.16}));
  p.refs.forEach(r=>svg.append(sv('line',{x1:L,x2:W-R,y1:y(r.value),y2:y(r.value),stroke:css(r.color),'stroke-width':2,'stroke-dasharray':'6 4'})));
  svg.append(sv('path',{d:'M'+p.x.map((v,i)=>`${x(v)},${y(s.mean[i])}`).join('L'),fill:'none',stroke:css(color),'stroke-width':2}));
  const cross=sv('line',{y1:T,y2:H-B,stroke:css('axis'),visibility:'hidden'});svg.append(cross);
  const hit=sv('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent'});
  hit.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect(),vx=(ev.clientX-r.left)*W/r.width;let i=0,best=1e9;
    p.x.forEach((v,k)=>{const d=Math.abs(x(v)-vx);if(d<best){best=d;i=k}});cross.setAttribute('x1',x(p.x[i]));cross.setAttribute('x2',x(p.x[i]));cross.setAttribute('visibility','visible');
    showTip(ev,`<b>${p.x[i].toLocaleString()} decisions</b><br>${name}: ${fmt(s.mean[i])} (range ${fmt(s.lo[i])} – ${fmt(s.hi[i])})`+p.refs.map(r=>`<br>${r.name}: ${fmt(r.value)}`).join(''))});
  hit.addEventListener('mouseleave',()=>{hideTip();cross.setAttribute('visibility','hidden')});svg.append(hit);return svg}
function blockLines(b,box){box.append(legend([{name:b.series_name+' (mean; band = 5 seeds ka range)',color:b.color},{name:'ECT',color:'s3',line:1},{name:'FCFS',color:'s4',line:1}]));
  const ds=panelsOf(b,box);b.panels.forEach((p,i)=>ds[i].append(linePanel(p,b.color,b.series_name,ds[i].clientWidth)));
  const rows=[];b.panels.forEach(p=>p.x.forEach((v,i)=>rows.push([p.title,v.toLocaleString(),fmt(p.series[0].mean[i]),fmt(p.series[0].lo[i])+' – '+fmt(p.series[0].hi[i])])));
  addTable(box,['Scenario','Decisions','Mean cost','Range'],rows)}
function blockStacked(b,box){box.append(legend(b.panels[0].parts.map(p=>({name:p.name,color:p.color}))));const ds=panelsOf(b,box);
  b.panels.forEach((p,pi)=>{const d=ds[pi];const W=Math.max(260,d.clientWidth||560),rowH=28,L=138,H=p.categories.length*rowH+10;
    const svg=sv('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H});p.categories.forEach((c,i)=>{let x0=L;svg.append(svt(L-8,i*rowH+21,c,'lab','end'));
      p.parts.forEach(part=>{const v=part.values[i]||0,w=(W-L-10)*v;if(w>0){const r=sv('rect',{x:x0,y:i*rowH+6,width:Math.max(0,w-2),height:rowH-10,rx:3,fill:css(part.color)});
        r.addEventListener('mousemove',ev=>showTip(ev,`<b>${c}</b> · ${part.name}: ${(100*v).toFixed(1)}%`));r.addEventListener('mouseleave',hideTip);svg.append(r)}x0+=w})});
    d.append(svg)});
  const rows=[];b.panels.forEach(p=>p.categories.forEach((c,i)=>rows.push([p.title,c].concat(p.parts.map(x=>(100*x.values[i]).toFixed(1)+'%')))));
  addTable(box,['Scenario','Method'].concat(b.panels[0].parts.map(p=>p.name)),rows)}
/* ---------- overview / concepts blocks ---------- */
function blockArch(b,box){const W=860,H=370,svg=sv('svg',{viewBox:`0 0 ${W} ${H}`,class:'arch',role:'img'}),xs=[150,430,710],by=230,bw=210,bh=92;
  xs.forEach(cx=>svg.append(sv('path',{d:`M${cx},${by} L${430+(cx-430)*0.25},96`,class:'lc'})));
  svg.append(sv('rect',{x:300,y:20,width:260,height:76,rx:18,class:'cl'}));svg.append(svt(430,52,'☁️  Cloud (shared)','t1'));svg.append(svt(430,76,b.cloud,'t2'));
  svg.append(svt(578,132,b.cloud_link,'t4','start'));
  svg.append(sv('rect',{x:20,y:24,width:236,height:68,rx:14,class:'cr'}));svg.append(svt(138,52,'🧠 Central critic','t1'));svg.append(svt(138,74,'sirf training mein (global state)','t2'));
  [[0,1],[1,2]].forEach(([a,c])=>svg.append(sv('path',{d:`M${xs[a]+bw/2},${by+bh/2} L${xs[c]-bw/2},${by+bh/2}`,class:'le'})));
  svg.append(sv('path',{d:`M${xs[0]},${by+bh} Q430,${by+bh+50} ${xs[2]},${by+bh}`,class:'le'}));svg.append(svt(430,H-6,b.edge,'t3'));
  b.nodes.forEach(([n,cap],i)=>{const cx=xs[i];svg.append(sv('rect',{x:cx-bw/2,y:by,width:bw,height:bh,rx:16,class:'bx'}));svg.append(svt(cx,by+26,n,'t1'));
    svg.append(svt(cx,by+46,`edge node · capacity ${cap}`,'t2'));svg.append(sv('rect',{x:cx-78,y:by+56,width:156,height:24,rx:12,class:'pill'}));svg.append(svt(cx,by+72,'🤖 Actor · Local: delay 0','tp'))});
  box.append(svg);box.append(el('div',{class:'tnote'},'Har department ka actor sirf apni local maloomat se faisla karta hai: task <b>Local</b> (apna node), <b>Edge</b> (padosi department) ya <b>Cloud</b>. Critic sirf training mein madad karta hai.'))}
function blockPipeline(b,box){const g=el('div',{class:'pipe'});STEPS.filter(s=>s.kind!=='page').forEach(s=>{const c=el('button',{class:'pcard'},
  `<span class="pn"><i class="ph-${s.phase}">${s.num}</i>${s.title}</span><small>${s.short}</small>`);c.onclick=()=>go(s.id,'explain');g.append(c)});box.append(g)}
function blockPoints(b,box){if(b.tone)box.classList.add(b.tone);const g=el('div',{class:'pts'});b.items.forEach(([ic,t,x])=>g.append(el('div',{class:'pt'},`<span class="ic">${ic}</span><div><b>${t}</b><p>${x}</p></div>`)));box.append(g)}
function blockGlossary(b,out){const inp=el('input',{class:'gsearch',placeholder:'🔎 Koi term dhoondein (maslan MAPPO, deadline, seed)…'});out.append(inp);const secs=[];
  b.groups.forEach(gr=>{const w=el('div');w.append(el('h3',{class:'gtitle'},gr.title));const g=el('div',{class:'ggrid'});
    gr.items.forEach(([t,x])=>{const c=el('div',{class:'gcard'},`<b>${t}</b><p>${x}</p>`);c.dataset.k=(t+' '+x).toLowerCase();g.append(c)});w.append(g);out.append(w);secs.push(w)});
  inp.oninput=()=>{const q=inp.value.toLowerCase().trim();secs.forEach(w=>{let any=0;w.querySelectorAll('.gcard').forEach(c=>{const s=!q||c.dataset.k.includes(q);c.style.display=s?'':'none';any|=s});w.style.display=any?'':'none'})}}
function mdToHtml(t){const lines=t.split('\n');let h='',tbl=null;const flush=()=>{if(tbl){h+='<div class="tbl"><table><thead><tr>'+tbl[0].map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>'+tbl.slice(1).map(r=>'<tr>'+r.map(c=>`<td class="${isNum(c.replace(/<[^>]+>/g,''))?'num':''}">${c}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>';tbl=null}};
  const inl=s=>esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>').replace(/`(.+?)`/g,'<code>$1</code>');
  lines.forEach(l=>{if(/^\|/.test(l)){if(/^\|[-| ]+\|$/.test(l))return;(tbl=tbl||[]).push(l.slice(1,-1).split('|').map(c=>inl(c.trim())));return}flush();
    if(/^#+ /.test(l))h+='<h2>'+inl(l.replace(/^#+ /,''))+'</h2>';else if(/^- /.test(l))h+='<p>• '+inl(l.slice(2))+'</p>';else if(l.trim())h+='<p>'+inl(l)+'</p>'});flush();return h}
function render(blocks,out){out.innerHTML='';blocks.forEach(b=>{
  if(b.type==='note'){out.append(el('div',{class:'note'},'ℹ️ '+b.text));return}if(b.type==='empty'){out.append(el('div',{class:'empty'},b.text));return}
  if(b.type==='glossary'){blockGlossary(b,out);return}
  const box=el('div',{class:'card'});box.append(el('h3',{},b.title||''));out.append(box);
  if(b.type==='kpis'){const k=el('div',{class:'kpis'+(b.big?' big':'')});b.items.forEach(i=>k.append(el('div',{class:'kpi'},`<span>${i.label}</span><b>${i.value}</b><i>${i.note||''}</i>`)));box.append(k)}
  else if(b.type==='table')box.append(tableOf(b.columns,b.rows,b));else if(b.type==='bars')blockBars(b,box);else if(b.type==='lines')blockLines(b,box);
  else if(b.type==='stacked')blockStacked(b,box);else if(b.type==='pre')box.append(el('pre',{},esc(b.text)));
  else if(b.type==='arch')blockArch(b,box);else if(b.type==='pipeline')blockPipeline(b,box);else if(b.type==='points')blockPoints(b,box);
  else if(b.type==='md'){const d=el('details',{class:'vout'});d.append(el('summary',{},'Poora output dikhao'));d.append(el('div',{class:'md'},mdToHtml(b.text)));box.append(d)}})}
/* ---------- pages ---------- */
const SEC=[['goal','🎯','Maqsad',''],['did','🛠️','Kya kiya',''],['how','⚙️','Kaise kaam karta hai',''],['result','📈','Nateeja','res'],['limits','⚠️','Hadood / dhyan dein','warn']];
function hero(s){const steps=STEPS.filter(x=>x.kind!=='page');return `<section class="hero ph-${s.phase}"><div class="hbadge">${s.num}</div><div><div class="eyebrow">${s.label} · ${PHASES[s.phase]}</div>
  <h1>${s.title}</h1><p>${s.short}</p><div class="chips">${s.id==='overview'?`<span class="chip">${steps.length} hisse · sab complete</span>`:s.kind==='page'?'':'<span class="chip">✓ Complete</span>'}
  ${s.runs&&s.runs.length?`<span class="chip">▶ ${s.runs.length} run option${s.runs.length>1?'s':''}</span>`:''}${s.files?`<span class="chip">📁 ${s.files.length} files</span>`:''}</div></div></section>`}
function stepper(id){const st=STEPS.filter(x=>x.id.startsWith('step'));const i=st.findIndex(x=>x.id===id);if(i<0)return '';
  return '<div class="stepper">'+st.map((x,k)=>(k?'<span class="sline"></span>':'')+`<span class="sdot ${k===i?'on':'done'}" title="${x.label}: ${x.title}" onclick="go('${x.id}')">${x.num}</span>`).join('')+'</div>'}
function pager(id){const i=STEPS.findIndex(x=>x.id===id),p=STEPS[i-1],n=STEPS[i+1];
  return `<div class="pager">${p?`<button onclick="go('${p.id}')"><small>← Pichla</small>${p.label}: ${p.title}</button>`:'<span></span>'}${n?`<button class="nx" onclick="go('${n.id}')"><small>Agla →</small>${n.label}: ${n.title}</button>`:''}</div>`}
function go(id,tab){const s=STEPS.find(x=>x.id===id);if(!s)return;const prev=STEPS.find(x=>x.id===CUR);CUR=id;if(s.kind==='page')TAB='results';else if(tab)TAB=tab;else if(!prev||prev.kind==='page')TAB='explain';
  history.replaceState(null,'','#'+id+(s.kind==='page'?'':'/'+TAB));document.querySelectorAll('.ni').forEach(b=>b.classList.toggle('on',b.dataset.id===id));
  const tabs=s.kind==='page'?'':`<div class="tabs">${[['explain','📖 Samjhein'],['results','📊 Nateeje'],['run','▶ Chala kar check karein']].map(([t,l])=>`<button class="tab ${t===TAB?'on':''}" onclick="go('${id}','${t}')">${l}</button>`).join('')}</div>`;
  $('#page').innerHTML=hero(s)+stepper(id)+tabs+'<div id="tabbody" style="margin-top:'+(s.kind==='page'?'20px':'0')+'"></div>'+pager(id);scrollTo(0,0);renderTab()}
async function renderTab(){const s=STEPS.find(x=>x.id===CUR),body=$('#tabbody');if(!body)return;
  if(TAB==='explain'){body.innerHTML=`<div class="takeaway"><span>💡 Ek line mein</span><p>${s.takeaway}</p></div><div class="secgrid">`+
      SEC.filter(([k])=>s[k]).map(([k,ic,t,c])=>`<div class="sec ${c}"><h4><i>${ic}</i>${t}</h4>${/^\s*<(ul|ol|p)/.test(s[k])?s[k]:'<p>'+s[k]+'</p>'}</div>`).join('')+
      `<div class="sec"><h4><i>📁</i>Files</h4><div class="files">${s.files.map(f=>`<code>${f}</code>`).join('')}</div></div></div>
      <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap"><button class="btn" onclick="go('${s.id}','results')">📊 Nateeje dekhein →</button><button class="btn ghost" onclick="go('${s.id}','run')">▶ Khud chala kar check karein</button></div>`;return}
  if(TAB==='run'){body.innerHTML=`<div class="note">ℹ️ Run button backend par <b>asal script</b> chalata hai (wahi commands jo README mein hain) aur naya nateeja committed file se compare karta hai. Kuch bhi overwrite nahi hota — naye outputs temp folder ya git-ignored <code>results/runs/dashboard/</code> mein jate hain.</div>`+
      s.runs.map(r=>`<div class="runcard"><div><h4>${r.label}</h4><p>${r.desc}</p></div><button class="btn runb" onclick="run('${s.id}','${r.id}')">▶ Chalao</button></div>`).join('')+'<div id="status"></div>';showJob();return}
  if(!DATA[CUR]){body.innerHTML='<div class="empty">⏳ Load ho raha hai…</div>';const id=CUR,r=await fetch('/api/data/'+id).then(r=>r.json());DATA[id]=r;if(id!==CUR||TAB!=='results')return}
  const r=DATA[CUR];if(r.error)body.innerHTML=`<div class="empty">Error: ${esc(r.error)}</div>`;else render(r.blocks,body)}
async function run(id,opt){const r=await fetch(`/api/run/${id}/${opt}`,{method:'POST'});if(r.status===409){alert('Ek run pehle se chal raha hai — us ke khatam hone ka intezar karein.');return}poll()}
function showJob(){const j=JOB;document.querySelectorAll('.runb').forEach(b=>b.disabled=!!j.running);const ind=$('#jobInd'),s=STEPS.find(x=>x.id===j.step);
  ind.style.display=j.running?'block':'none';if(j.running&&s)ind.innerHTML=`⏳ ${s.label} chal raha hai… ${Math.round(Date.now()/1000-j.started)} s`;
  const st=$('#status');if(!st)return;if(j.step!==CUR||!(j.running||(j.verdicts||[]).length)){st.innerHTML='';return}
  const secs=Math.round((j.finished||Date.now()/1000)-j.started),lbl=((s&&s.runs.find(r=>r.id===j.option))||{}).label||'';
  st.innerHTML=`<div class="card"><h3>Nateeja: ${lbl}</h3>`+(j.running?`<div class="running"><div class="spin"></div>Backend par chal raha hai… ${secs} s</div>`:`<div class="tnote">Waqt: ${secs} s</div>`)+
    (j.verdicts||[]).map(v=>`<div class="verdict ${v.ok?'ok':'bad'}"><b>${v.ok?'✓ PASS':'✗ FAIL'}</b><div>${esc(v.check)}<small>${esc(v.detail)}</small></div></div>`).join('')+
    `<details class="log" ${j.running?'open':''}><summary>Run log (backend output)</summary><pre>${esc((j.log||[]).join('\n'))}</pre></details></div>`}
async function poll(){clearTimeout(POLL);JOB=await fetch('/api/job').then(r=>r.json());
  if(!JOB.running&&JOB.finished&&JOB.finished!==LASTFIN){delete DATA[JOB.step];LASTFIN=JOB.finished}
  showJob();if(JOB.running)POLL=setTimeout(poll,1000)}
function setTheme(t){document.documentElement.dataset.theme=t;try{localStorage.setItem('theme',t)}catch(e){}$('#themeBtn').textContent=t==='dark'?'☀️ Light mode':'🌙 Dark mode';if(CUR&&TAB==='results')renderTab()}
$('#themeBtn').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
let RT=null,LW=innerWidth;addEventListener('resize',()=>{clearTimeout(RT);RT=setTimeout(()=>{if(innerWidth!==LW&&TAB==='results'){LW=innerWidth;renderTab()}},300)});
fetch('/api/steps').then(r=>r.json()).then(s=>{STEPS=s;let ph=null;const nav=$('#nav');
  s.forEach(x=>{if(x.phase!==ph){ph=x.phase;nav.append(el('div',{class:'ng'},PHASES[ph]))}const b=el('button',{class:'ni','data-id':x.id},`<span class="nn">${x.num}</span><span>${x.title}</span>${x.kind==='page'?'':'<span class="ok">✓</span>'}`);
    b.onclick=()=>go(x.id);nav.append(b)});
  setTheme(document.documentElement.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'));
  const [h,t]=location.hash.slice(1).split('/');go(s.find(x=>x.id===h)?h:s[0].id,t||'explain');poll()});
</script></body></html>""".replace("__COLORS__", json.dumps(COLOR)).replace("__PHASES__", json.dumps(PHASES))


DATA = {"overview": data_overview, "concepts": data_concepts, "step1": data_step1, "step2": data_step2, "step3": data_step3,
        "step4": data_step4, "step5": data_step5, "step6": data_step6, "step7": data_step7, "step8": data_step8, "step9": data_step9,
        "step10": data_step10, "results": data_results, "verification": data_verification}


# ============================================================================
# HTTP server (127.0.0.1 only)
# ============================================================================
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html")
        if self.path == "/api/steps":
            F, out = facts(), []
            for s in STEPS:
                d = {k: (v.format(**F) if k in TEXT_KEYS else v) for k, v in s.items()}
                d["runs"] = [{"id": o, "label": lab, "desc": desc} for o, lab, desc, _ in RUNS.get(s["id"], [])]
                out.append(d)
            return self._send(200, json.dumps(out))
        if self.path.startswith("/api/data/"):
            key = self.path.rsplit("/", 1)[1]
            if key not in DATA:
                return self._send(404, json.dumps({"error": "unknown step"}))
            try:
                return self._send(200, json.dumps({"blocks": DATA[key]()}))
            except Exception as e:                       # shown in the page, never hidden
                return self._send(200, json.dumps({"error": f"{type(e).__name__}: {e}"}))
        if self.path == "/api/job":
            return self._send(200, json.dumps({"running": JOB.running, "step": JOB.step, "option": JOB.option, "log": JOB.log,
                                               "verdicts": JOB.verdicts, "started": JOB.started, "finished": JOB.finished,
                                               "just_finished": JOB.finished}))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        m = re.fullmatch(r"/api/run/(\w+)/(\w+)", self.path)
        if not m or m.group(1) not in RUNS or m.group(2) not in [o for o, _, _, _ in RUNS[m.group(1)]]:
            return self._send(404, json.dumps({"error": "unknown run"}))
        return self._send(200 if start_job(m.group(1), m.group(2)) else 409, json.dumps({"ok": True}))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Thesis project dashboard (local, 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Dashboard: {url}   (band karne ke liye Ctrl+C)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
