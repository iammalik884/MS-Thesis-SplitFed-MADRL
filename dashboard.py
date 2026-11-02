"""
Project dashboard - every step of the thesis project in ONE file (backend + frontend).
================================================================================
ROMAN URDU: `python dashboard.py` chalayein; browser mein dashboard khul jata
hai. Har step ka card: explanation (click karke khulti hai), committed
results ke charts/tables, aur "Run" button jo backend par ASAL script chalata
hai (tests, evaluation, sanity checks) aur naya nateeja committed nateeje se
compare karta hai (identical / different). Saare numbers result files se
parhe jate hain -- koi number haath se nahi likha.

ENGLISH: a local web app on 127.0.0.1 (Python standard library only). The
backend reads the committed result files and, on "Run", executes the same
commands as documented in the README, then compares the new outputs with the
committed ones. It never edits configs, code or committed results; temporary
outputs go to the system temp folder or to results/runs/dashboard/ (git-ignored).

Run:  python dashboard.py            (opens http://127.0.0.1:8765)
      python dashboard.py --port 8800 --no-browser
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
            + ". <b>Run</b> dabane par yeh tests chalte hain aur nateeja upar badge mein aata hai."}


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
                     {"type": "table", "title": "Sanity check ki table", "columns": ["Scenario", "Policy", "Cost", "Avg latency", "Critical violation"], "rows": rows}]


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


def _test_table(summary, key, label):
    rows = []
    for sc in ("default", "overload", "slow_network"):
        R = summary["results"][sc]
        for name, k in ((label, key), ("ECT", "ECT"), ("FCFS", "FCFS"), ("always Cloud", "always Cloud")):
            r = R[k]
            cost = (-r["return"][0], r["return"][1])
            rows.append([SCEN_NAME[sc], name, pm(cost), pm(r["avg_latency"], f2), pm(r["critical_deadline_violation_rate"], pct),
                         f"{100 * r['action_shares']['Local']:.0f} / {100 * r['action_shares']['Edge']:.0f} / {100 * r['action_shares']['Cloud']:.0f}%"])
    return {"type": "table", "title": f"Test seeds 2000–2029 (sirf ek dafa) — {label} best checkpoint, mean ± 95% CI (5 training seeds)",
            "columns": ["Scenario", "Policy", "Cost (−return)", "Avg latency", "Critical violation", "Local / Edge / Cloud"], "rows": rows}


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
                     {"type": "table", "title": "Per-agent aur team cost", "columns": ["Scenario", "Rule"] + agents + ["Team"], "rows": rows}]


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
             "is model mein Cloud ki koi cost nahi aur woh kabhi nahi bharta. Verification tab ka cloud sensitivity dekhein."}]


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
    cost, crit, shares, rows = [], [], [], []
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
             "All-task viol.", "Throughput", "Payload"], "rows": rows}]


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


RUNS = {
    "step1": [("tests", "Tests + demo chalao (≈5 s)", lambda: run_tests(["tests/test_step1_simulator.py"]))],
    "step2": [("tests", "Tests chalao (≈5 s)", lambda: run_tests(["tests/test_step2_task_generator.py"]))],
    "step3": [("tests", "Tests chalao (≈10 s)", lambda: run_tests(["tests/test_step3_routing.py", "tests/test_v2_routing_timing.py"]))],
    "step4": [("tests", "Tests chalao (≈10 s)", lambda: run_tests(["tests/test_step4_deadlines.py", "tests/test_v2_metrics_protocol.py"]))],
    "step5": [("baselines", "Tests + FCFS baselines dobara (≈30 s)", run_step5)],
    "step6": [("sanity", "Tests + sanity check (≈1 min)", run_step6)],
    "step7": [("reeval", "Test evaluation dobara (≈30 s)", lambda: run_reeval("step7")),
              ("tests", "DQN tests (≈20 s)", lambda: run_tests(["tests/test_dqn_pilot.py", "tests/test_ect_heuristic.py"]))],
    "step8": [("sanity", "Tests + per-agent check (≈1 min)", run_step8)],
    "step9": [("reeval", "Test evaluation dobara (≈30 s)", lambda: run_reeval("step9")),
              ("tests", "MAPPO tests (≈20 s)", lambda: run_tests(["tests/test_mappo.py"]))],
    "step10": [("reeval", "Test evaluation dobara (≈30 s)", lambda: run_reeval("step10")),
               ("retrain", "Ek model shuru se train (≈3–5 min)", run_retrain_w5),
               ("tests", "Step 10 tests (≈10 s)", lambda: run_tests(["tests/test_step10_priority_reward.py"]))],
    "results": [("all_tests", "Saare automated tests (≈1 min)", lambda: run_tests(["tests"]))],
    "verification": [(n, lbl, (lambda n=n: run_verification(n))) for n, lbl in (
        ("v1_independent_actors", "V1 (≈30 s)"), ("v2_consolidated_comparison", "V2 (≈5 s)"), ("v5_priority_results", "V5 (≈1 min)"),
        ("v6_imbalance_generalization", "V6 (≈1–2 min)"), ("v4_cloud_sensitivity", "V4 (≈5 min)"), ("v3_reproduce", "V3 (≈5–10 min)"))],
}
DATA = {"step1": data_step1, "step2": data_step2, "step3": data_step3, "step4": data_step4, "step5": data_step5,
        "step6": data_step6, "step7": data_step7, "step8": data_step8, "step9": data_step9, "step10": data_step10,
        "results": data_results, "verification": data_verification}


def start_job(step, option):
    with JOB.lock:
        if JOB.running:
            return False
        JOB.running, JOB.step, JOB.option = True, step, option
        JOB.log, JOB.verdicts, JOB.started, JOB.finished = [], [], time.time(), None
    fn = dict((o, f) for o, _, f in RUNS[step])[option]

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
    return {"s5_crit_overload": pct(crit("overload")), "s5_crit_slow": pct(crit("slow_network")),
            "s9_mappo": f1(-s9["MAPPO (best)"]["return"][0]), "s9_ect": f1(-s9["ECT"]["return"][0]), "s9_fcfs": f1(-s9["FCFS"]["return"][0]),
            "s10_crit_w1": pct(w1["Critical"][0][0], 2), "s10_crit_w5": pct(w5["Critical"][0][0], 2),
            "s10_clat_w1": f2(w1["Critical"][1][0]), "s10_clat_w5": f2(w5["Critical"][1][0]),
            "s10_rout_w1": f2(w1["Routine"][1][0]), "s10_rout_w5": f2(w5["Routine"][1][0]), "s10_routviol_w5": pct(w5["Routine"][0][0]),
            "s10_cloud_w5": pct(cloud(A)), "s10_cloud_w1": pct(cloud(R)),
            "v4_cost50": f1(-v4m("overload | cloud capacity 50", "return")), "v4_cost6": f1(-v4m("overload | cloud capacity 6", "return")),
            "v4_crit50": pct(v4m("overload | cloud capacity 50", "critical_violation")), "v4_crit6": pct(v4m("overload | cloud capacity 6", "critical_violation")),
            "v6_edge_default": pct(v6e("default")), "v6_edge_overload": pct(v6e("overload"))}


# ============================================================================
# Step texts (explanations, Roman Urdu with English terms)
# ============================================================================
STEPS = [
    ("step1", "Step 1", "Mini hospital simulator", "Departments ke edge nodes, fixed capacity aur FCFS queues — bina AI ke.",
     """<h4>Kya kiya</h4><p>Hospital ka ek chhota simulator banaya: 3 departments (Emergency, ICU, General Ward), har ek ka apna
     <b>edge node</b> jis ki har time-step mein fixed <b>capacity</b> hai (compute units). Tasks queue mein FCFS tarteeb se chalte hain;
     bada task agle steps tak jaari rehta hai (carry-over).</p><h4>Kaise kaam karta hai</h4><p><code>Task</code> ek medical job hai,
     <code>EdgeNode</code> us ki queue chalata hai, aur <code>HospitalSimulator</code> clock chalata hai. Departments ek doosre se
     independent hain.</p><h4>Files</h4><p><code>env/task.py</code>, <code>env/edge_node.py</code>, <code>env/hospital_env.py</code>,
     <code>demo_simulation.py</code></p>"""),
    ("step2", "Step 2", "Medical task generation", "Har department aur priority ke liye Poisson arrivals, seeded aur reproducible.",
     """<h4>Kya kiya</h4><p><code>MedicalTaskGenerator</code>: har (department, priority) ke tasks <b>Poisson</b> process se aate hain;
     compute demand har priority ki range mein uniform. Same seed = same tasks. Ek step ke tasks seeded random tarteeb mein,
     taake koi department hamesha pehle na aaye.</p><h4>Ahem baat</h4><p>Yeh numbers <b>synthetic placeholders</b> hain (MIMIC-IV access
     pending). Chart dikhata hai ke <b>overload</b> mein har department ρ &gt; 1 hai (padosiyon ke paas bhi jagah nahi), jabke
     <b>department imbalance</b> mein sirf ICU bhara hai.</p><h4>Files</h4><p><code>env/task_generator.py</code>, <code>configs/</code></p>"""),
    ("step3", "Step 3", "Local / Edge / Cloud routing", "Task apne node, padosi department ya shared Cloud par ja sakta hai.",
     """<h4>Kya kiya</h4><p>Teen jagah: <b>Local</b> (apna node, delay 0), <b>Edge</b> (padosi department, delay 1, us ki asal queue share
     hoti hai), <b>Cloud</b> (capacity 50, delay 2). Har node <b>destination-ready FCFS</b>: raste mein task CPU nahi leta.</p>
     <h4>FCFS rule</h4><p>Fixed, hand-written, priority nahi dekhta — yahi thesis ka <b>Baseline 1</b> hai. Step 6 ke baad isi faisle ko
     RL agent seekhta hai.</p><h4>Files</h4><p><code>env/task_router.py</code>, <code>env/cloud_node.py</code></p>"""),
    ("step4", "Step 4", "Priority aur deadlines", "Critical 2, High 5, Routine 10 steps; deadline status track hota hai.",
     """<h4>Kya kiya</h4><p>Deadline = arrival time + priority offset (Critical 2, High 5, Routine 10 steps — placeholders). Deadlines sirf
     <b>track</b> hoti hain; FCFS in par faisla nahi karta.</p><h4>Metrics</h4><p>Har task 4 categories mein se ek: on_time,
     completed_late, overdue_unfinished, not_yet_due. Violation rate = (late + overdue) / decided tasks, Critical ke liye alag.</p>
     <h4>Files</h4><p><code>env/task.py</code>, <code>env/metrics.py</code></p>"""),
    ("step5", "Step 5", "FCFS baseline + evaluation protocol", "Ek fixed protocol jis par har policy naapi jati hai; 30 seeds, 6 scenarios.",
     """<h4>Kya kiya</h4><p>Common evaluation protocol: 20 steps ka <b>window</b> (tasks sirf yahan aate hain), phir <b>drain</b> (≤ 50 steps)
     taake adhoore tasks gine jayen. Metrics: throughput, latency, deadline violations (Critical alag), offloading communication.</p>
     <h4>Nateeja</h4><p>Default load halka hai — FCFS koi deadline miss nahi karta. Stress scenarios (overload, slow network) mein FCFS
     bohat kamzor: Critical violations {s5_crit_overload} aur {s5_crit_slow}.</p><h4>Reproducibility</h4><p>Har run folder mein config, seeds, git commit aur
     package versions. <b>Run</b> se saare baselines dobara bante hain aur byte-for-byte compare hote hain.</p>
     <h4>Files</h4><p><code>env/evaluation.py</code>, <code>run_baseline_fcfs.py</code>, <code>results/baselines/</code></p>"""),
    ("step6", "Step 6", "Gymnasium environment", "Simulator ko RL interface (reset / step) mein wrap kiya.",
     """<h4>Kya kiya</h4><p><code>HospitalSchedulingEnv</code>: har naye task par ek faisla (Local / Edge / Cloud). Observation 13 values
     (har node ka load, task ki demand, source, priority, deadline, time). Reward = −latency − 10 × deadline miss, task khatam hone par.</p>
     <h4>Verification</h4><p>Agar agent FCFS ke actions chune to 6 configs × 30 seeds par Step 5 ka nateeja <b>bilkul</b> dohrata hai.</p>
     <h4>Sanity check</h4><p>Simple fixed policies se pata chala: overload mein "always Cloud" FCFS se kaafi behtar — Cloud free aur
     taqreeban kabhi nahi bharta. Yeh finding aage har step mein ahem hai.</p><h4>Files</h4><p><code>env/gym_env.py</code>,
     <code>check_simple_policies.py</code></p>"""),
    ("step7", "Step 7", "DQN baseline (single agent)", "Ek centralized DQN — debugging ke baad bhi ECT se kamzor.",
     """<h4>Kya kiya</h4><p>Pehle <b>ECT heuristic</b> banaya (sab se jaldi khatam hone ka andaza) — ek mazboot, bina training baseline.
     Phir DQN: pilot unstable tha; debugging (har dafa ek tabdeeli, plan pehle commit, sirf validation seeds): reward scaling, γ,
     Double DQN, lambi training.</p><h4>Frozen setting</h4><p>Double DQN, reward × 0.01, γ 0.95, 40k decisions — pre-registered rule se
     (sab se zyada mean final score). Phir 5 seeds × 3 scenarios training, aur test seeds 2000–2029 par <b>sirf ek dafa</b> evaluation.</p>
     <h4>Nateeja</h4><p>DQN overload mein FCFS se behtar, lekin <b>ECT ko kisi scenario mein beat nahi karta</b>, aur training stable nahi.</p>
     <h4>Files</h4><p><code>agents/dqn.py</code>, <code>train_dqn_pilot.py</code>, <code>Step_Documentation/Step7_DQN_Debugging.md</code></p>"""),
    ("step8", "Step 8", "Three-agent environment", "Har department ek alag agent, apni local observation ke saath.",
     """<h4>Kya kiya</h4><p><code>MultiAgentHospitalEnv</code>: Emergency, ICU, General Ward alag agents; har agent apne department ke tasks
     ka faisla karta hai. Local observation (10 values): apna load, padosiyon ka load, cloud load, apna task. Faisle usi seeded tarteeb mein
     jo FCFS/ECT/DQN ki thi — fair comparison.</p><h4>Checkpoints (tested)</h4><p>Teeno agents ek hi step mein act karte hain; Edge offload
     padosi ki queue exactly utni badhata hai; FCFS agents Step 5 ko bilkul dohrate hain.</p><h4>Files</h4><p><code>env/multi_agent_env.py</code>,
     <code>check_multi_agent_env.py</code></p>"""),
    ("step9", "Step 9", "MAPPO with CTDE", "Har department ka actor, training mein central critic — sab se behtar method.",
     """<h4>Kya kiya</h4><p>Har department ka <b>apna actor</b> (sirf local observation), ek <b>centralized critic</b> (global state, sirf
     training mein). Team reward, GAE, PPO clipped updates. Experiments: M0 (γ 0.99), M1 (γ 0.95), LONG (120k) — LONG frozen (stable).</p>
     <h4>Nateeja (test seeds)</h4><p>Teeno scenarios mein sab se behtar; overload cost {s9_mappo} vs ECT {s9_ect} vs FCFS {s9_fcfs}.</p>
     <h4>Kamzori</h4><p>Load balancing <b>Cloud</b> se hoti hai, padosiyon se nahi (overload Edge 0%) — kyunke overload mein padosi bhi
     full hain aur Cloud free hai. Verification tab mein iski jaanch.</p><h4>Files</h4><p><code>agents/mappo.py</code>,
     <code>train_mappo.py</code>, <code>Step_Documentation/Step9_MAPPO.md</code></p>"""),
    ("step10", "Step 10", "Priority-aware reward", "Critical tasks ko zyada importance — Critical misses taqreeban khatam.",
     """<h4>Kya kiya</h4><p>Reward = w<sub>priority</sub> × (−latency − 10 × miss). Weights deadlines se nikale (andaze se nahi): W1 1/1/1,
     W5 5/2/1 (10 / offset), W25 25/4/1. Validation par pre-registered rule (Critical violations) se <b>W5</b> chuna gaya.</p>
     <h4>Nateeja (test seeds)</h4><p>Slow network: Critical violations {s10_crit_w1} → {s10_crit_w5}, Critical latency {s10_clat_w1} → {s10_clat_w5};
     Routine latency {s10_rout_w1} → {s10_rout_w5} (zyada wait), Routine violation {s10_routviol_w5}. Overall return taqreeban same.</p><h4>Keemat</h4>
     <p>Cloud ka istemal zyada ({s10_cloud_w5} vs {s10_cloud_w1}). Weights placeholder deadlines se hain, clinical data se nahi.</p><h4>Files</h4><p><code>env/gym_env.py</code>
     (<code>_task_reward</code>), <code>Step_Documentation/Step10_Priority_Reward.md</code></p>"""),
    ("results", "Results", "Final comparison", "FCFS, ECT, DQN, MAPPO aur Priority MAPPO — same test workloads, same metrics.",
     """<h4>Kaise parhein</h4><p><b>Cost = −return</b>: har task ki latency + 10 × deadline miss ka jor; <b>kam = behtar</b>. Har scenario
     ka scale alag hai (overload ke numbers default se ~50 guna bade), is liye har scenario ka alag panel. Error bar = 95% CI (5 training
     seeds).</p><h4>Khulasa</h4><p>MAPPO aur Priority MAPPO har stress scenario mein sab se behtar; Priority MAPPO Critical violations sab
     se kam. DQN ECT se kamzor. Sab se bada caveat: learned policies <b>free, unlimited Cloud</b> par tiki hain.</p>"""),
    ("verification", "Checks", "Verification (supervisor ke sawal)", "Independent actors, reproducibility, cloud sensitivity, imbalance.",
     """<h4>Kya check hua</h4><ul><li><b>V1</b>: har department ka alag actor; critic hata kar bhi evaluation bilkul same.</li>
     <li><b>V2</b>: saare methods same test workloads par.</li><li><b>V3</b>: saare automated tests, teeno test evaluations aur ek model ki retraining
     byte-for-byte same.</li><li><b>V4</b>: Cloud capacity 50 → 6 ya delay +4 par frozen policy kaafi kharab (overload cost {v4_cost50} → {v4_cost6},
     Critical {v4_crit50} → {v4_crit6}) — faisle Cloud par bohat munhasir.</li><li><b>V5</b>: weights ka chunao aur per-priority nateeje.</li>
     <li><b>V6</b>: sirf ICU bhara ho to default-trained MAPPO padosiyon ko use karta hai (ICU ke {v6_edge_default} tasks); overload-trained nahi ({v6_edge_overload}).</li></ul>
     <p>V4 aur V6 <b>frozen</b> models ki generalization hai, retraining ke nateeje nahi.</p>"""),
]


# ============================================================================
# Frontend (single page; no external libraries, works offline)
# ============================================================================
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Thesis Dashboard</title><style>
:root{color-scheme:light;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);--accent:#2a78d6;--good:#0ca30c;--crit:#d03b3b;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;
--q1:#b7d3f6;--q2:#5598e7;--q3:#184f95}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--accent:#3987e5;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;
--q1:#86b6ef;--q2:#3987e5;--q3:#1c5cab}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--border);background:var(--surface)}header h1{margin:0;font-size:18px}
header p{margin:2px 0 0;color:var(--ink2);font-size:13px}
.wrap{display:flex;min-height:calc(100vh - 70px)}nav{width:250px;flex:none;border-right:1px solid var(--border);padding:10px;background:var(--surface)}
nav button{display:block;width:100%;text-align:left;border:0;background:none;color:var(--ink);padding:8px 10px;border-radius:8px;cursor:pointer;font:inherit}
nav button small{display:block;color:var(--muted);font-size:12px}nav button:hover{background:var(--grid)}
nav button.on{background:var(--accent);color:#fff}nav button.on small{color:#e8eefb}
main{flex:1;padding:20px 24px;max-width:1240px}
.top{display:flex;gap:12px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap}
.top h2{margin:0;font-size:20px}.top p{margin:4px 0 0;color:var(--ink2)}
.runs{display:flex;gap:8px;flex-wrap:wrap}.runs button{border:1px solid var(--accent);background:var(--accent);color:#fff;border-radius:8px;
padding:7px 12px;cursor:pointer;font:inherit}.runs button.alt{background:none;color:var(--accent)}.runs button:disabled{opacity:.45;cursor:default}
details.ex{margin:14px 0;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 14px}
details.ex summary{cursor:pointer;font-weight:600}details.ex h4{margin:12px 0 2px;font-size:13px}details.ex p,details.ex ul{margin:2px 0;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:12px 0}
.card h3{margin:0 0 8px;font-size:14px}.note{color:var(--ink2);background:var(--surface);border-left:3px solid var(--accent);padding:10px 14px;border-radius:6px;margin:12px 0}
.empty{color:var(--ink2);padding:18px;text-align:center;border:1px dashed var(--axis);border-radius:10px;margin:12px 0}
.kpis{display:flex;gap:10px;flex-wrap:wrap}.kpi{flex:1;min-width:150px;border:1px solid var(--border);border-radius:10px;padding:10px 12px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}.kpi i{font-style:normal;color:var(--ink2);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}.scroll{overflow-x:auto}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.panel{min-width:0}.panel svg{display:block;max-width:100%;height:auto}.panel h5{margin:0 0 4px;font-size:13px;color:var(--ink2)}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:0 0 8px;font-size:12px;color:var(--ink2)}.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}.swl{width:16px;height:0;border-top:2px dashed;display:inline-block}
svg text{fill:var(--muted);font-size:11px}svg .lab{fill:var(--ink2)}.tog{float:right;font-size:12px;color:var(--accent);cursor:pointer;border:0;background:none}
pre{white-space:pre-wrap;font:12px/1.45 ui-monospace,Consolas,monospace;color:var(--ink2);max-height:420px;overflow:auto;margin:0}
.badge{display:inline-flex;gap:6px;align-items:center;border-radius:8px;padding:6px 10px;margin:4px 0;font-size:13px;width:100%;border:1px solid var(--border)}
.badge.ok b{color:var(--good)}.badge.bad b{color:var(--crit)}.status{margin:12px 0}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:6px 9px;
font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.15);display:none;z-index:9;max-width:320px}
.md h1,.md h2{font-size:14px;margin:10px 0 4px}.md p{color:var(--ink2);margin:4px 0}
@media(max-width:820px){.wrap{flex-direction:column}nav{width:auto;display:flex;overflow-x:auto;border-right:0;border-bottom:1px solid var(--border)}
nav button{min-width:150px}main{padding:16px}}
</style></head><body><header><h1>Privacy-Preserving SplitFed-MADRL — Thesis Dashboard</h1>
<p>Priority-aware task scheduling in healthcare edge computing · Usama Manzoor · Steps 1–10 · saare numbers committed result files se</p></header>
<div class="wrap"><nav id="nav"></nav><main id="main"></main></div><div id="tip"></div>
<script>
const $=s=>document.querySelector(s), tip=$('#tip');let STEPS=[],CUR=null,POLL=null;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue('--'+v).trim();
const fmt=(v,f)=>v==null||isNaN(v)?'—':f==='pct1'?(100*v).toFixed(1)+'%':f==='pct2'?(100*v).toFixed(2)+'%':f==='n2'?v.toFixed(2):
  Math.abs(v)>=100?v.toLocaleString(undefined,{maximumFractionDigits:1}):v.toFixed(1);
function showTip(e,html){tip.innerHTML=html;tip.style.display='block';const x=Math.min(e.clientX+14,innerWidth-330);tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px'}
function hideTip(){tip.style.display='none'}
const el=(t,a={},h='')=>{const n=document.createElement(t);for(const k in a)n.setAttribute(k,a[k]);if(h)n.innerHTML=h;return n};
const S='http://www.w3.org/2000/svg',sv=(t,a)=>{const n=document.createElementNS(S,t);for(const k in a)n.setAttribute(k,a[k]);return n};
function nice(lo,hi){if(lo===hi){hi=lo+1}const span=hi-lo,step=Math.pow(10,Math.floor(Math.log10(span/4)));const m=[1,2,2.5,5,10].find(m=>span/(m*step)<=5)*step;
  return {lo:Math.floor(lo/m)*m,hi:Math.ceil(hi/m)*m,step:m}}
function legend(items){const d=el('div',{class:'legend'});items.forEach(i=>{const s=el('span');s.append(el('i',{class:i.line?'swl':'sw',style:(i.line?'border-color:':'background:')+css(i.color)}),document.createTextNode(i.name));d.append(s)});return d}
function tableOf(cols,rows){const w=el('div',{class:'scroll'}),t=el('table');t.append(el('tr',{},cols.map(c=>`<th>${c}</th>`).join('')));
  rows.forEach(r=>t.append(el('tr',{},r.map(c=>`<td>${c}</td>`).join(''))));w.append(t);return w}
function barPanel(p,f,ref,W){W=Math.max(260,Math.round(W||560));const H=240,L=58,R=(ref&&ref.length)?60:12,T=12,B=34,svg=sv('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:'img'});
  let vals=[];p.series.forEach(s=>s.values.forEach((v,i)=>{if(v!=null){vals.push(v);if(s.err&&s.err[i]!=null){vals.push(v+s.err[i],v-s.err[i])}}}));(ref||[]).forEach(r=>vals.push(r.value));
  const nonneg=p.series.every(s=>s.values.every(v=>v==null||v>=0));if(nonneg)vals=vals.map(v=>Math.max(0,v));
  const sc=nice(Math.min(0,...vals),Math.max(0,...vals)),y=v=>T+(H-T-B)*(sc.hi-v)/(sc.hi-sc.lo);
  for(let v=sc.lo;v<=sc.hi+1e-9;v+=sc.step){svg.append(sv('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:css('grid')}));const t=sv('text',{x:L-6,y:y(v)+4,'text-anchor':'end'});t.textContent=fmt(v,f);svg.append(t)}
  const nC=p.categories.length,nS=p.series.length,gw=(W-L-R)/nC,bw=Math.max(4,Math.min(34,(gw-14)/nS-2));
  p.categories.forEach((c,ci)=>{const g0=L+ci*gw+(gw-nS*(bw+2))/2;
    p.series.forEach((s,si)=>{const v=s.values[ci];if(v==null)return;const x=g0+si*(bw+2),y0=y(0),y1=y(v),top=Math.min(y0,y1),h=Math.max(1,Math.abs(y1-y0));
      svg.append(sv('rect',{x,y:top,width:bw,height:h,rx:Math.min(3,bw/3),fill:css(s.color)}));
      const e=s.err&&s.err[ci];if(e){const cx=x+bw/2,lo=nonneg?Math.max(0,v-e):v-e;svg.append(sv('line',{x1:cx,x2:cx,y1:y(lo),y2:y(v+e),stroke:css('ink2'),'stroke-width':1.5}));
        [lo,v+e].forEach(z=>svg.append(sv('line',{x1:cx-4,x2:cx+4,y1:y(z),y2:y(z),stroke:css('ink2'),'stroke-width':1.5})))}
      const hit=sv('rect',{x:x-1,y:T,width:bw+2,height:H-T-B,fill:'transparent'});
      hit.addEventListener('mousemove',ev=>showTip(ev,`<b>${s.name}</b>${c?' · '+c:''}<br>${fmt(v,f)}${e?' ± '+fmt(e,f):''}`));hit.addEventListener('mouseleave',hideTip);svg.append(hit)});
    if(c){const t=sv('text',{x:L+ci*gw+gw/2,y:H-B+16,'text-anchor':'middle',class:'lab'});t.textContent=c;svg.append(t)}});
  svg.append(sv('line',{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:css('axis')}));
  (ref||[]).forEach(r=>{svg.append(sv('line',{x1:L,x2:W-R,y1:y(r.value),y2:y(r.value),stroke:css('ink2'),'stroke-dasharray':'4 4'}));
    const t=sv('text',{x:W-R+6,y:y(r.value)+4,'text-anchor':'start'});t.textContent=r.label;svg.append(t)});return svg}
function blockBars(b,box){const names=[];b.panels.forEach(p=>p.series.forEach(s=>{if(!names.find(n=>n.name===s.name))names.push({name:s.name,color:s.color})}));
  if(names.length>1)box.append(legend(names));const g=el('div',{class:'panels'});
  box.append(g);const ds=b.panels.map(p=>{const d=el('div',{class:'panel'});if(p.title)d.append(el('h5',{},p.title));g.append(d);return d});
  b.panels.forEach((p,i)=>ds[i].append(barPanel(p,p.fmt||b.fmt,b.ref,ds[i].clientWidth)));
  const rows=[];b.panels.forEach(p=>p.categories.forEach((c,ci)=>p.series.forEach(s=>rows.push([p.title||'',c,s.name,fmt(s.values[ci],p.fmt||b.fmt)+(s.err&&s.err[ci]!=null?' ± '+fmt(s.err[ci],p.fmt||b.fmt):'')]))));
  addTable(box,['Panel','Category','Series','Value'],rows)}
function addTable(box,cols,rows){const t=el('button',{class:'tog'},'Table dikhao');const w=tableOf(cols,rows);w.style.display='none';
  t.onclick=()=>{const o=w.style.display==='none';w.style.display=o?'block':'none';t.textContent=o?'Table chhupao':'Table dikhao'};box.querySelector('h3').append(t);box.append(w)}
function linePanel(p,color,name,W){W=Math.max(260,Math.round(W||560));const H=230,L=58,R=12,T=12,B=34,svg=sv('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:'img'});const s=p.series[0];
  const vals=[...s.lo,...s.hi,...p.refs.map(r=>r.value)],sc=nice(Math.min(...vals),Math.max(...vals)),y=v=>T+(H-T-B)*(sc.hi-v)/(sc.hi-sc.lo);
  const xm=Math.max(...p.x),x=v=>L+(W-L-R)*v/xm;
  for(let v=sc.lo;v<=sc.hi+1e-9;v+=sc.step){svg.append(sv('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:css('grid')}));const t=sv('text',{x:L-6,y:y(v)+4,'text-anchor':'end'});t.textContent=fmt(v,'n1');svg.append(t)}
  [0,.25,.5,.75,1].forEach(q=>{const t=sv('text',{x:x(q*xm),y:H-B+16,'text-anchor':'middle'});t.textContent=Math.round(q*xm/1000)+'k';svg.append(t)});
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
function blockLines(b,box){box.append(legend([{name:b.series_name+' (mean; band = range over 5 seeds)',color:b.color},{name:'ECT',color:'s3',line:1},{name:'FCFS',color:'s4',line:1}]));
  const g=el('div',{class:'panels'});box.append(g);const ds=b.panels.map(p=>{const d=el('div',{class:'panel'});d.append(el('h5',{},p.title));g.append(d);return d});
  b.panels.forEach((p,i)=>ds[i].append(linePanel(p,b.color,b.series_name,ds[i].clientWidth)));
  const rows=[];b.panels.forEach(p=>p.x.forEach((v,i)=>rows.push([p.title,v,fmt(p.series[0].mean[i]),fmt(p.series[0].lo[i])+' – '+fmt(p.series[0].hi[i])])));
  addTable(box,['Scenario','Decisions','Mean cost','Range'],rows)}
function blockStacked(b,box){box.append(legend(b.panels[0].parts.map(p=>({name:p.name,color:p.color}))));const g=el('div',{class:'panels'});
  box.append(g);const ds=b.panels.map(p=>{const d=el('div',{class:'panel'});d.append(el('h5',{},p.title));g.append(d);return d});
  b.panels.forEach((p,pi)=>{const d=ds[pi];const W=Math.max(260,d.clientWidth||560),rowH=26,L=130,H=p.categories.length*rowH+10;
    const svg=sv('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H});p.categories.forEach((c,i)=>{let x0=L;const t=sv('text',{x:L-8,y:i*rowH+20,'text-anchor':'end',class:'lab'});t.textContent=c;svg.append(t);
      p.parts.forEach(part=>{const v=part.values[i]||0,w=(W-L-10)*v;if(w>0){const r=sv('rect',{x:x0,y:i*rowH+6,width:Math.max(0,w-2),height:rowH-10,rx:2,fill:css(part.color)});
        r.addEventListener('mousemove',ev=>showTip(ev,`<b>${c}</b> · ${part.name}: ${(100*v).toFixed(1)}%`));r.addEventListener('mouseleave',hideTip);svg.append(r)}x0+=w})});
    d.append(svg)});
  const rows=[];b.panels.forEach(p=>p.categories.forEach((c,i)=>rows.push([p.title,c].concat(p.parts.map(x=>(100*x.values[i]).toFixed(1)+'%')))));
  addTable(box,['Scenario','Method'].concat(b.panels[0].parts.map(p=>p.name)),rows)}
function mdToHtml(t){const lines=t.split('\n');let h='',tbl=null;const flush=()=>{if(tbl){h+='<div class="scroll"><table>'+tbl.map((r,i)=>'<tr>'+r.map(c=>i?`<td>${c}</td>`:`<th>${c}</th>`).join('')+'</tr>').join('')+'</table></div>';tbl=null}};
  const inl=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/\*\*(.+?)\*\*/g,'<b>$1</b>').replace(/`(.+?)`/g,'<code>$1</code>');
  lines.forEach(l=>{if(/^\|/.test(l)){if(/^\|[-| ]+\|$/.test(l))return;(tbl=tbl||[]).push(l.slice(1,-1).split('|').map(c=>inl(c.trim())));return}flush();
    if(/^#+ /.test(l))h+='<h2>'+inl(l.replace(/^#+ /,''))+'</h2>';else if(/^- /.test(l))h+='<p>• '+inl(l.slice(2))+'</p>';else if(l.trim())h+='<p>'+inl(l)+'</p>'});flush();return h}
function render(blocks){const out=$('#results');out.innerHTML='';blocks.forEach(b=>{
  if(b.type==='note'){out.append(el('div',{class:'note'},b.text));return}if(b.type==='empty'){out.append(el('div',{class:'empty'},b.text));return}
  const box=el('div',{class:'card'});box.append(el('h3',{},b.title||''));out.append(box);
  if(b.type==='kpis'){const k=el('div',{class:'kpis'});b.items.forEach(i=>k.append(el('div',{class:'kpi'},`<span>${i.label}</span><b>${i.value}</b><i>${i.note||''}</i>`)));box.append(k)}
  else if(b.type==='table')box.append(tableOf(b.columns,b.rows));else if(b.type==='bars')blockBars(b,box);else if(b.type==='lines')blockLines(b,box);
  else if(b.type==='stacked')blockStacked(b,box);else if(b.type==='pre')box.append(el('pre',{},b.text.replace(/&/g,'&amp;').replace(/</g,'&lt;')));
  else if(b.type==='md'){const d=el('details');d.append(el('summary',{},'Output dikhao'));d.append(el('div',{class:'md'},mdToHtml(b.text)));box.append(d)}})}
async function load(id){CUR=id;history.replaceState(null,'','#'+id);document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.id===id));const s=STEPS.find(x=>x.id===id);
  $('#main').innerHTML=`<div class="top"><div><h2>${s.label}: ${s.title}</h2><p>${s.short}</p></div><div class="runs" id="runs"></div></div>
  <details class="ex"><summary>Is step mein kya hua — poori explanation</summary>${s.explain}</details><div class="status" id="status"></div><div id="results"><div class="empty">Load ho raha hai…</div></div>`;
  s.runs.forEach((r,i)=>{const b=el('button',{class:i?'alt':''},'▶ '+r.label);b.onclick=()=>run(id,r.id);$('#runs').append(b)});
  const r=await fetch('/api/data/'+id).then(r=>r.json());if(r.error)$('#results').innerHTML=`<div class="empty">Error: ${r.error}</div>`;else render(r.blocks);poll()}
async function run(id,opt){const r=await fetch(`/api/run/${id}/${opt}`,{method:'POST'});if(r.status===409){alert('Ek run pehle se chal raha hai — us ke khatam hone ka intezar karein.');return}poll()}
async function poll(){clearTimeout(POLL);const j=await fetch('/api/job').then(r=>r.json());document.querySelectorAll('.runs button').forEach(b=>b.disabled=j.running);
  const st=$('#status');if(!st)return;if(j.step===CUR&&(j.running||j.verdicts.length)){const secs=Math.round((j.finished||Date.now()/1000)-j.started);
    st.innerHTML=(j.running?`<div class="note">⏳ Backend par chal raha hai… ${secs} s</div>`:'')+j.verdicts.map(v=>`<div class="badge ${v.ok?'ok':'bad'}"><b>${v.ok?'✓ PASS':'✗ FAIL'}</b> ${v.check} — ${v.detail}</div>`).join('');
    const d=el('details');d.append(el('summary',{},'Run log'));d.append(el('pre',{},j.log.join('\n').replace(/&/g,'&amp;').replace(/</g,'&lt;')));if(j.running)d.open=true;st.append(d);
    if(!j.running&&j.just_finished!==window.LASTFIN){window.LASTFIN=j.just_finished;const r=await fetch('/api/data/'+CUR).then(r=>r.json());if(!r.error)render(r.blocks)}}
  else st.innerHTML='';if(j.running)POLL=setTimeout(poll,1000)}
let RT=null;addEventListener('resize',()=>{clearTimeout(RT);RT=setTimeout(()=>{if(CUR)load(CUR)},300)});
fetch('/api/steps').then(r=>r.json()).then(s=>{STEPS=s;s.forEach(x=>{const b=el('button',{'data-id':x.id},`${x.label}: ${x.title}<small>${x.short}</small>`);b.onclick=()=>load(x.id);$('#nav').append(b)});const h=location.hash.slice(1);load(s.find(x=>x.id===h)?h:s[0].id)});
</script></body></html>"""


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
            F = facts()
            return self._send(200, json.dumps([{"id": i, "label": l, "title": t, "short": s, "explain": e.format(**F),
                                                "runs": [{"id": o, "label": lab} for o, lab, _ in RUNS[i]]} for i, l, t, s, e in STEPS]))
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
        if not m or m.group(1) not in RUNS or m.group(2) not in [o for o, _, _ in RUNS[m.group(1)]]:
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
