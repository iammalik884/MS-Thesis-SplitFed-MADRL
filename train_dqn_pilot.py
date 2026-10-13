"""
Step 7 (PILOT) - Train a single-agent DQN in the Step 6 environment.
================================================================================
ROMAN URDU: Yeh PRELIMINARY pilot hai -- training pipeline ko debug karne aur
learning dynamics dekhne ke liye. Final thesis results NAHI (design note,
Section 6). Environment, reward aur simulator bilkul wahi hain jo Step 6
mein verified hue.

Seeds (design note, Section 5):
  training workloads : har episode ke liye 10,000-19,999 se ek seed
  validation         : 1,000-1,029 (checkpoint chunne ke liye)
  test               : 2,000-2,029 -- pilot mein ISTEMAL NAHI hota
Report: undiscounted return, protocol metrics, action proportions; saath mein
saare baselines (FCFS, always Local/Edge/Cloud, random, ECT) same validation
seeds par.

Run:  py train_dqn_pilot.py --config configs/scenarios/overload.yaml --train-seed 0 --total-steps 40000
      [--set reward_scale=0.01]   (one DQNConfig override per experiment)
Output: results/runs/<name>_<UTC time>_<commit>/ (git-ignored)
"""

import argparse
import csv
import dataclasses
import json
import os
import time

import numpy as np
import torch

from agents.dqn import DQNAgent, DQNConfig
from agents.policy_eval import baseline_policies, evaluate, q_vs_monte_carlo
from env.gym_env import HospitalSchedulingEnv
from env.hospital_env import load_config
from env.run_records import collect_provenance, unique_run_dir, write_json

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SEED_RANGE = (10_000, 19_999)
VALIDATION_SEEDS = range(1_000, 1_030)
TEST_SEEDS = range(2_000, 2_030)  # reserved for final Step 7 results; never used here


def train(config_path, train_seed, total_steps, eval_every, dqn_config, log=print):
    env = HospitalSchedulingEnv(config_path)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, dqn_config, seed=train_seed,
                     unbounded_mask=np.isinf(env.observation_space.high))
    workload_rng = np.random.default_rng([train_seed, 7])  # separate stream for training workload seeds
    learning_curve, validation_curve = [], []
    best = {"mean_return": -np.inf, "state_dict": None, "step": None}
    step, episode, next_eval = 0, 0, eval_every

    def validate():
        result = evaluate(config_path, lambda e, obs: agent.greedy_action(obs), VALIDATION_SEEDS)
        row = {"step": step, "episode": episode, "mean_return": result["mean_return"],
               **{f"mean_{k}": v for k, v in result["means"].items()},
               **{f"share_{a}": p for a, p in result["action_proportions"].items()}}
        validation_curve.append(row)
        if result["mean_return"] > best["mean_return"]:
            best.update(mean_return=result["mean_return"], step=step, result=result,
                        state_dict={k: v.clone() for k, v in agent.q.state_dict().items()})
        log(f"  step {step:6d}  episode {episode:5d}  validation return {result['mean_return']:9.1f}  "
            f"shares L/E/C {result['action_proportions']['Local']:.2f}/"
            f"{result['action_proportions']['Edge']:.2f}/{result['action_proportions']['Cloud']:.2f}")

    while step < total_steps:
        workload_seed = int(workload_rng.integers(TRAIN_SEED_RANGE[0], TRAIN_SEED_RANGE[1] + 1))
        obs, info = env.reset(seed=workload_seed)
        if info["no_tasks_in_episode"]:      # no decision to learn from
            env.step(0)
            continue
        terminated, episode_return, losses = False, 0.0, []
        epsilon = agent.epsilon(step, total_steps)
        while not terminated:
            epsilon = agent.epsilon(step, total_steps)
            action = agent.act(obs, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Only the TRAINING signal is scaled (positive constant, no clipping);
            # all reported returns/metrics stay in original units.
            agent.buffer.add(obs, action, reward * dqn_config.reward_scale, next_obs, terminated)
            obs, episode_return, step = next_obs, episode_return + reward, step + 1
            if step >= dqn_config.learning_starts:
                losses.append(agent.update())
        episode += 1
        learning_curve.append({"episode": episode, "step": step, "workload_seed": workload_seed,
                               "train_return": episode_return, "epsilon": epsilon,
                               "mean_loss": float(np.mean(losses)) if losses else None})
        if step >= next_eval:
            validate()
            next_eval += eval_every
    if not validation_curve or validation_curve[-1]["step"] != step:
        validate()
    return agent, learning_curve, validation_curve, best


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="PILOT: train a DQN in the Step 6 environment")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "environment.yaml"))
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=40_000)
    parser.add_argument("--eval-every", type=int, default=4_000)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results", "runs"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one DQNConfig field, e.g. --set reward_scale=0.01")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)  # deterministic, and the network is small
    fields = {f.name: f.type for f in dataclasses.fields(DQNConfig)}
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in fields:
            raise SystemExit(f"--set: unknown DQNConfig field {key!r}; known: {sorted(fields)}")
        kind = fields[key]
        if kind in (bool, "bool"):
            if value not in ("0", "1", "true", "false"):
                raise SystemExit(f"--set {key}: use 1/0 or true/false, got {value!r}")
            overrides[key] = value in ("1", "true")
        else:
            overrides[key] = (int if kind in (int, "int") else float)(value)
    dqn_config = DQNConfig(**overrides)
    scenario = os.path.splitext(os.path.basename(args.config))[0]
    name = args.name or f"dqn_pilot_{scenario}_trainseed{args.train_seed}"
    print(f"PILOT DQN [{scenario}] train seed {args.train_seed}, {args.total_steps} steps (decisions)")
    start = time.time()
    agent, learning_curve, validation_curve, best = train(
        args.config, args.train_seed, args.total_steps, args.eval_every, dqn_config)

    final_state = {k: v.clone() for k, v in agent.q.state_dict().items()}
    dqn_eval = {}
    for label, state in (("dqn_best", best["state_dict"]), ("dqn_final", final_state)):
        agent.q.load_state_dict(state)
        dqn_eval[label] = {
            "evaluation": evaluate(args.config, lambda e, obs: agent.greedy_action(obs), VALIDATION_SEEDS,
                                   keep_reports=True),
            "q_vs_mc": q_vs_monte_carlo(args.config, agent, VALIDATION_SEEDS),
        }
    agent.q.load_state_dict(final_state)

    print("Validation seeds 1000-1029 (undiscounted return):")
    baselines = {}
    for bname, policy in baseline_policies(random_seed=args.train_seed).items():
        baselines[bname] = evaluate(args.config, policy, VALIDATION_SEEDS, keep_reports=True)
    for policy_name, result in [(k, v["evaluation"]) for k, v in dqn_eval.items()] + list(baselines.items()):
        print(f"  {policy_name:13s} return {result['mean_return']:9.1f}")
    for label in ("dqn_best", "dqn_final"):
        qm = dqn_eval[label]["q_vs_mc"]["unscaled"]
        print(f"  {label} Q vs MC (unscaled): mean Q {qm['mean_q']:.1f}, mean MC return {qm['mean_mc_return']:.1f}, "
              f"bias {qm['mean_bias']:.1f}, RMSE {qm['rmse']:.1f}")
    print(f"Best DQN checkpoint: step {best['step']} ({time.time() - start:.0f} s)")

    provenance = collect_provenance()
    out = unique_run_dir(args.out_dir, name, provenance["git_commit"])
    write_json(os.path.join(out, "manifest.json"), {
        "method": "DQN (PILOT / debugging, preliminary -- not a thesis result)",
        "config_path": os.path.relpath(os.path.abspath(args.config), ROOT),
        "config_snapshot": load_config(args.config),
        "dqn_config": dqn_config.to_dict(),
        "dqn_overrides": overrides,
        "train_seed": args.train_seed, "total_steps": args.total_steps, "eval_every": args.eval_every,
        "training_workload_seed_range": TRAIN_SEED_RANGE,
        "validation_seeds": [VALIDATION_SEEDS.start, VALIDATION_SEEDS.stop - 1],
        "test_seeds_reserved_unused": [TEST_SEEDS.start, TEST_SEEDS.stop - 1],
        "provenance": provenance,
    })
    _write_csv(os.path.join(out, "learning_curve.csv"), learning_curve)
    _write_csv(os.path.join(out, "validation_curve.csv"), validation_curve)
    torch.save(best["state_dict"], os.path.join(out, "model_best.pt"))
    torch.save(final_state, os.path.join(out, "model_final.pt"))

    all_results = {**{k: v["evaluation"] for k, v in dqn_eval.items()}, **baselines}
    os.makedirs(os.path.join(out, "evaluation"), exist_ok=True)
    per_seed = []
    for policy_name, result in all_results.items():
        write_json(os.path.join(out, "evaluation", policy_name.replace(" ", "_") + ".json"), result)
        per_seed += [{"policy": policy_name, **{k: v for k, v in row.items() if k != "report"}}
                     for row in result["per_seed"]]
    _write_csv(os.path.join(out, "per_seed.csv"), per_seed)
    strip = lambda r: {k: v for k, v in r.items() if k != "per_seed"}
    write_json(os.path.join(out, "summary.json"), {
        "note": "Validation seeds only; undiscounted returns in original units",
        "dqn_best": {"step": best["step"], **strip(dqn_eval["dqn_best"]["evaluation"])},
        "dqn_final": {"step": step_of(learning_curve), **strip(dqn_eval["dqn_final"]["evaluation"])},
        "q_vs_mc": {k: v["q_vs_mc"] for k, v in dqn_eval.items()},
        "baselines": {k: strip(v) for k, v in baselines.items()},
    })
    print(f"Saved to {out}")


def step_of(learning_curve):
    return learning_curve[-1]["step"] if learning_curve else 0


if __name__ == "__main__":
    main()
