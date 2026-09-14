# MS Thesis Project: Privacy-Preserving SplitFed-MADRL

Practical implementation for: *Privacy-Preserving SplitFed-MADRL for Priority-Aware
Task Scheduling in Healthcare Edge Computing* (Usama Manzoor, MSIT71S25S008,
University of Sargodha).

This code follows the 18-step roadmap in `My Research Practical ROADMAP.docx`.
**Do not skip steps** — each step tests exactly one new idea before the next
is added, so bugs stay easy to isolate.

## Progress

- [x] **Step 0 — Research setup.** Python 3.11, NumPy, PyYAML, Gymnasium 1.3.0,
      PyTorch 2.14.0 installed and verified. See `requirements.txt`.
- [x] **Step 1 — Mini hospital simulator (no AI).** Three department edge
      nodes (Emergency, ICU, General Ward), each with a fixed per-step compute
      capacity and a FIFO task queue. No scheduling intelligence, no priority,
      no deadlines yet — this step only proves the queueing + capacity
      mechanics are correct. All 6 verification tests pass (see below).
      **Post-Step-3 correction:** `EdgeNode.utilization` originally only
      looked at the queue's front task, so 3 small tasks that together
      filled a node's capacity were misreported as barely busy. Fixed to
      mirror `step()`'s actual FIFO capacity-consumption loop (see
      `test_utilization_reflects_whole_queue_not_just_front_task`). It was
      unused dead code at the time (no test/demo called it), so this did
      not affect any prior result — but it matters going forward since
      node utilization is a likely RL observation feature (Steps 7+).
- [x] **Step 2 — Generate medical tasks (Poisson arrivals).** `MedicalTaskGenerator`
      (`env/task_generator.py`) draws Critical/High/Routine arrivals per
      department from independent Poisson processes, with compute-demand drawn
      from a per-priority range. `Task` now carries an optional `priority`
      field. **Data source note:** arrival rates and compute-demand ranges in
      `configs/environment.yaml` (`task_generation` section) are PLACEHOLDER
      values, clinically plausible but not yet derived from MIMIC-IV —
      PhysioNet credentialed access is still pending (see the roadmap's
      Objective 1 admin action). Only that config section will need updating
      once access is granted; no code changes required. Deadlines are still
      NOT included (deferred to Step 4, as originally planned). All 6
      verification tests pass (see below).
      **Post-Step-3 correction:** Routine's `compute_demand_range` upper
      bound was raised from 4.0 to 8.0 in every department. With the old 4.0
      cap and Step 3's `edge_threshold: 6.0`, no generated task could ever
      exceed `edge_threshold`, so the Cloud tier — fully implemented and
      tested in Step 3 — was silently unreachable by any realistic,
      generator-driven simulation (only reachable via manually-crafted test
      tasks). Routine tasks (batch analytics/reporting) are the plausible
      place for occasional heavy compute_demand, so only their upper bound
      changed. Still a placeholder pending real MIMIC-IV-derived values.
- [x] **Step 3 — Local / Edge / Cloud processing choice.** `TaskRouter`
      (`env/task_router.py`) is a **fixed, non-AI rule** that picks a tier
      from a task's `compute_demand`: small tasks stay `Local` (on-device,
      instant, tiny capacity), medium tasks go to the department's existing
      `Edge` node (unchanged from Step 1/2), and large tasks go to a single
      shared `Cloud` node (`env/cloud_node.py`) with much bigger capacity but
      a configurable `network_delay` — the task waits in transit for that
      many steps before any compute happens. `HospitalSimulator.route_task()`
      / `route_generated_arrivals()` are new, **opt-in** methods; the
      original `add_task()` / `add_generated_arrivals()` from Steps 1–2 are
      completely unchanged and still always use the Edge node (verified by
      `test_step1_add_task_still_bypasses_routing`). Thresholds and tier
      capacities live in `configs/environment.yaml` (`offloading` section).
      All 7 verification tests pass (see below). This is **not** learned —
      the actual RL agent that will make this choice intelligently is
      Steps 7+.
- [x] **Step 4 — Priority and deadline system.** `Task` now carries an
      optional `deadline` (arrival_time + a priority-based offset) and a
      derived `met_deadline` property (`None` until complete, then
      `True`/`False`). `HospitalSimulator._compute_deadline()` is the one
      place that computes it, called from both `add_task()` and
      `route_task()`; offsets live in `configs/environment.yaml`'s new
      `deadlines` section (Critical: 2, High: 5, Routine: 10 steps —
      PLACEHOLDER, pending MIMIC-IV). Fully **opt-in**, same pattern as
      Step 3's `offloading`: no `deadlines` section, or no `priority` on a
      task, and `deadline` just stays `None` — nothing before Step 4
      changes behaviour. This is still **tracking only** — no queue
      reordering, no preemption; FIFO order from Step 1 is untouched. All 8
      verification tests pass (see below).
      **Demo observation:** with the current placeholder offsets, the
      seed=42 demo run meets 21/21 deadlines (0 missed) — the offsets are
      generous relative to current capacities/demand ranges, so this
      dataset doesn't yet exercise a missed deadline in practice. Correctness
      of the miss case itself is verified directly by
      `test_met_deadline_all_three_states` (unit-level, not seed-dependent).
      Worth revisiting once real MIMIC-IV-derived offsets are available.
- [ ] Step 5 — Baseline 1: FCFS
- [ ] Steps 6–18 — see the roadmap document

## Folder structure

```
MS_Thesis_Project/
├── configs/
│   └── environment.yaml      # Step 0/1 values (seed, capacities, n_steps) + Step 2 task_generation + Step 4 deadlines sections
├── env/
│   ├── task.py                # Task data class (id, arrival_time, compute_demand, priority since Step 2, tier since Step 3, deadline since Step 4)
│   ├── edge_node.py            # EdgeNode: limited-capacity FIFO queue + processing
│   ├── cloud_node.py           # Step 3: CloudNode (EdgeNode + network_delay before processing)
│   ├── task_router.py          # Step 3: TaskRouter (fixed Local/Edge/Cloud rule, not AI yet)
│   ├── hospital_env.py         # HospitalSimulator: ties departments + Local/Cloud tiers + Step 4 deadlines together
│   └── task_generator.py       # Step 2: MedicalTaskGenerator (Poisson arrivals per dept/priority)
├── tests/
│   ├── test_step1_simulator.py        # verification tests for Step 1
│   ├── test_step2_task_generator.py   # verification tests for Step 2
│   ├── test_step3_routing.py          # verification tests for Step 3
│   └── test_step4_deadlines.py        # verification tests for Step 4
└── requirements.txt
```

## How to run the tests

```bash
cd MS_Thesis_Project
pip install -r requirements.txt
python3 -m tests.test_step1_simulator
python3 -m tests.test_step2_task_generator
python3 -m tests.test_step3_routing
python3 -m tests.test_step4_deadlines
```

Expected output: 6 `[PASS]` lines + `ALL STEP 1 TESTS PASSED.`, then 6 `[PASS]`
lines + `ALL STEP 2 TESTS PASSED.`, then 7 `[PASS]` lines + `ALL STEP 3 TESTS
PASSED.`, then 8 `[PASS]` lines + `ALL STEP 4 TESTS PASSED.`

## What Step 1 proves (and does NOT prove)

Proves: tasks placed in a department's queue are processed within that
department's fixed capacity, tasks larger than one step's capacity correctly
carry over to later steps, departments do not interfere with each other's
capacity, `EdgeNode.utilization` correctly reflects the total work the whole
queue would consume this step (not just the front task), and invalid inputs
raise clear errors.

## What Step 2 proves (and does NOT prove)

Proves: arrivals are generated by an independent Poisson process per
(department, priority) pair; the same random seed always reproduces the same
sequence of tasks; over many steps the average arrival count converges to the
configured `arrival_rate`; generated compute-demand always stays inside its
configured range; and generated arrivals flow correctly into the Step 1
simulator (same queue/capacity mechanics apply, priority label preserved).

Does NOT yet include: any learning/scheduling agent (Steps 7+) — intentionally
deferred. Also note the Step 2 data source limitation above (placeholder
arrival rates, pending MIMIC-IV). (Deadlines were deferred to Step 4 when this
was written; they now exist — see below.)

## What Step 3 proves (and does NOT prove)

Proves: `TaskRouter` correctly partitions tasks into Local/Edge/Cloud by
`compute_demand` against two configured thresholds (including exact
boundary values); each tier dispatches to the right node (`local_nodes`,
`nodes`/Edge, `cloud_node`); the Cloud tier's `network_delay` correctly
holds a task untouched until enough steps have passed, then processes it
under the Cloud's own (larger) capacity using the same proven FIFO/capacity
mechanics as Step 1; Poisson-generated arrivals (Step 2) route correctly
end-to-end and, since the post-Step-3 Routine range correction above, can
actually reach all three tiers including Cloud (see `demo_simulation.py`);
and Step 1/2's `add_task()` / `add_generated_arrivals()` are provably
unchanged (still always Edge, no tier set).

Does NOT yet include: an intelligent/learned routing decision (the
`TaskRouter` rule is fixed and hand-written — the RL agent that will learn
this decision is Steps 7+), priority-aware routing (priority still plays no
part in `choose_tier()` — only `compute_demand` does), or per-department
Cloud capacity (Cloud is intentionally ONE shared resource for the whole
hospital, matching a centralized datacenter). (Task deadlines were deferred
to Step 4 when this was written; they now exist — see below.)

## What Step 4 proves (and does NOT prove)

Proves: `deadline = arrival_time + <priority's configured offset>`, computed
identically by both `add_task()` and `route_task()`; a task with no priority,
or a config with no `deadlines` section, correctly keeps `deadline = None`
(opt-in, nothing before Step 4 changes); `met_deadline` correctly resolves to
`None` before completion, `True` on time (including the exact boundary
`completion_time == deadline`), and `False` when late; a deadline before
`arrival_time` is rejected; a `deadlines` section missing a priority class or
holding a negative offset is rejected with a clear error; and Poisson-generated,
Step-3-routed arrivals get the correct deadline end-to-end.

Does NOT yet include: any use of `deadline`/`met_deadline` to change a
decision — queue order is still Step 1's plain FIFO, `TaskRouter` still
ignores priority entirely, and there is no preemption or admission control.
Deadlines are tracked, not enforced; enforcement is a scheduling *policy*
question left to the baselines (Step 5+) and the RL agent (Steps 7+). Also
note the offsets in `configs/environment.yaml`'s `deadlines` section are the
same kind of clinically-plausible PLACEHOLDER as Step 2's arrival rates —
pending MIMIC-IV-derived values.
