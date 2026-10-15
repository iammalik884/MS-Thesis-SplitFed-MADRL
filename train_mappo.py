"""
Step 9 - Train MAPPO (CTDE) in the Step 8 three-agent environment.
================================================================================
ROMAN URDU: Har department ka actor apni local observation se faisla karta
hai; centralized critic sirf training mein global state dekhta hai. Seeds,
validation, checkpoint (best + final), baselines aur evaluation BILKUL Step 7
ke DQN jaise hain (agents/policy_eval.evaluate), taake comparison fair ho.
Yeh script sirf VALIDATION seeds par evaluate karta hai; test seeds nahi.

Roadmap checkpoint: "Do agents begin to use spare capacity in other
departments rather than overload one node?" -> har policy ke liye har node
par aane wale tasks aur un ka average queue wait (summary.json:
"destination_load") aur Edge ka share.

Run:  py train_mappo.py --config configs/scenarios/overload.yaml --train-seed 0 [--set gamma=0.95]
Output: results/runs/<name>_<UTC time>_<commit>/ (git-ignored)
"""

import argparse
import dataclasses
import json
import os
import statistics
import time

import numpy as np
import torch

from agents.mappo import MAPPOAgent, MAPPOConfig
from agents.policy_eval import baseline_policies, evaluate
from env.hospital_env import load_config
from env.multi_agent_env import MultiAgentHospitalEnv, local_observation
from env.run_records import collect_provenance, unique_run_dir, write_json
from train_dqn_pilot import TEST_SEEDS, TRAIN_SEED_RANGE, VALIDATION_SEEDS, _write_csv, step_of

ROOT = os.path.dirname(os.path.abspath(__file__))


def greedy_policy(agent):
    """policy(env, obs) for agents/policy_eval.evaluate(): the acting
    department's actor on its LOCAL observation (decentralized execution)."""
    def policy(env, obs):
        department = env._pending[0].department
        return agent.act(department, local_observation(env, department, with_task=True), greedy=True)[0]
    return policy


def train(config_path, train_seed, total_steps, eval_every, cfg, log=print):
    env = MultiAgentHospitalEnv(config_path)
    agent = MAPPOAgent(env.agents, env.observation_spaces[env.agents[0]].shape[0], env.state_space.shape[0],
                       env.action_spaces[env.agents[0]].n, cfg, seed=train_seed)
    workload_rng = np.random.default_rng([train_seed, 7])  # same training-workload stream as the DQN
    learning_curve, validation_curve = [], []
    best = {"mean_return": -np.inf, "state_dict": None, "step": None}
    step, episode, next_eval = 0, 0, eval_every
    batch = {k: [] for k in ("state", "agent", "obs", "action", "log_prob", "reward", "done")}

    def validate():
        result = evaluate(config_path, greedy_policy(agent), VALIDATION_SEEDS)
        row = {"step": step, "episode": episode, "mean_return": result["mean_return"],
               **{f"mean_{k}": v for k, v in result["means"].items()},
               **{f"share_{a}": p for a, p in result["action_proportions"].items()}}
        validation_curve.append(row)
        if result["mean_return"] > best["mean_return"]:
            best.update(mean_return=result["mean_return"], step=step,
                        state_dict={k: {n: t.clone() for n, t in v.items()} for k, v in
                                    {"actors": agent.actors.state_dict(), "critic": agent.critic.state_dict()}.items()})
        log(f"  step {step:6d}  episode {episode:5d}  validation return {result['mean_return']:9.1f}  "
            f"shares L/E/C {result['action_proportions']['Local']:.2f}/"
            f"{result['action_proportions']['Edge']:.2f}/{result['action_proportions']['Cloud']:.2f}")

    while step < total_steps:
        workload_seed = int(workload_rng.integers(TRAIN_SEED_RANGE[0], TRAIN_SEED_RANGE[1] + 1))
        env.reset(seed=workload_seed)
        episode_return = 0.0
        while not env.done:
            a = env.agent_selection
            state, obs = env.state(), env.observe(a)
            action, log_prob = agent.act(a, obs)
            rewards, terminated, info = env.step(action)
            team = info["team_reward"]
            for key, value in zip(batch, (state, a, obs, action, log_prob, team * cfg.reward_scale, terminated)):
                batch[key].append(value)
            episode_return += team
            step += 1
        episode += 1
        stats = {"critic_loss": None, "actor_loss": None, "entropy": None}   # filled on update episodes
        if len(batch["action"]) >= cfg.rollout_decisions or (step >= total_steps and batch["action"]):
            # update on whole episodes only; the last, shorter rollout is used too
            stats = agent.update(batch)
            batch = {k: [] for k in batch}
        learning_curve.append({"episode": episode, "step": step, "workload_seed": workload_seed,
                               "train_return": episode_return, **stats})
        if step >= next_eval:
            validate()
            next_eval += eval_every
    if not validation_curve or validation_curve[-1]["step"] != step:
        validate()
    return agent, learning_curve, validation_curve, best


def value_vs_monte_carlo(config, agent, seeds):
    """Greedy team play; critic V(state) at each decision vs the discounted
    Monte Carlo team return from that decision (same gamma and reward scale
    as training). Also reported divided by reward_scale (latency-steps)."""
    gamma, scale = agent.config.gamma, agent.config.reward_scale
    env = MultiAgentHospitalEnv(config)
    v_all, g_all = [], []
    for seed in seeds:
        env.reset(seed=seed)
        states, rewards = [], []
        while not env.done:
            a = env.agent_selection
            states.append(env.state())
            _, _, info = env.step(agent.act(a, env.observe(a), greedy=True)[0])
            rewards.append(scale * info["team_reward"])
        g, returns = 0.0, []
        for r in reversed(rewards):
            g = r + gamma * g
            returns.append(g)
        v_all += list(agent.value(states)) if states else []
        g_all += returns[::-1]
    v_all, g_all = np.array(v_all, dtype=np.float64), np.array(g_all, dtype=np.float64)
    bias = v_all - g_all
    return {"gamma": gamma, "reward_scale": scale, "decisions": int(len(v_all)),
            "mean_v": float(v_all.mean()), "mean_mc_return": float(g_all.mean()),
            "correlation": float(np.corrcoef(v_all, g_all)[0, 1]) if len(v_all) > 1 else None,
            "unscaled": {"mean_v": float(v_all.mean() / scale), "mean_mc_return": float(g_all.mean() / scale),
                         "mean_bias": float(bias.mean() / scale), "rmse": float(np.sqrt((bias ** 2).mean()) / scale)}}


def destination_load(result):
    """Tasks and mean queue wait per destination node, averaged over seeds
    (needs evaluate(..., keep_reports=True))."""
    nodes = sorted({n for row in result["per_seed"] for n in row["report"]["final"]["by_destination"]})
    out = {}
    for n in nodes:
        rows = [row["report"]["final"]["by_destination"].get(n) for row in result["per_seed"]]
        waits = [r["avg_queue_wait"] for r in rows if r and r["avg_queue_wait"] is not None]
        out[n] = {"mean_tasks": statistics.mean(r["tasks"] if r else 0 for r in rows),
                  "mean_avg_queue_wait": statistics.mean(waits) if waits else None}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 9: train MAPPO in the three-agent environment (validation seeds only)")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "environment.yaml"))
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=40_000)
    parser.add_argument("--eval-every", type=int, default=4_000)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results", "runs"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one MAPPOConfig field, e.g. --set gamma=0.95")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)  # deterministic, and the networks are small
    fields = {f.name: f.type for f in dataclasses.fields(MAPPOConfig)}
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in fields:
            raise SystemExit(f"--set: unknown MAPPOConfig field {key!r}; known: {sorted(fields)}")
        overrides[key] = (int if fields[key] in (int, "int") else float)(value)
    cfg = MAPPOConfig(**overrides)
    scenario = os.path.splitext(os.path.basename(args.config))[0]
    name = args.name or f"mappo_{scenario}_trainseed{args.train_seed}"
    print(f"MAPPO [{scenario}] train seed {args.train_seed}, {args.total_steps} decisions")
    start = time.time()
    agent, learning_curve, validation_curve, best = train(
        args.config, args.train_seed, args.total_steps, args.eval_every, cfg)

    final_state = {k: {n: t.clone() for n, t in v.items()} for k, v in agent.state_dict().items()}
    results, v_check = {}, {}
    for label, state in (("mappo_best", best["state_dict"]), ("mappo_final", final_state)):
        agent.load_state_dict(state)
        results[label] = evaluate(args.config, greedy_policy(agent), VALIDATION_SEEDS, keep_reports=True)
        v_check[label] = value_vs_monte_carlo(args.config, agent, VALIDATION_SEEDS)
    agent.load_state_dict(final_state)
    for bname, policy in baseline_policies(random_seed=args.train_seed).items():
        results[bname] = evaluate(args.config, policy, VALIDATION_SEEDS, keep_reports=True)
    print("Validation seeds 1000-1029 (undiscounted return):")
    for label, result in results.items():
        print(f"  {label:13s} return {result['mean_return']:9.1f}")
    print(f"Best MAPPO checkpoint: step {best['step']} ({time.time() - start:.0f} s)")

    provenance = collect_provenance()
    out = unique_run_dir(args.out_dir, name, provenance["git_commit"])
    write_json(os.path.join(out, "manifest.json"), {
        "method": "MAPPO with CTDE (Step 9 training script; evaluation here on validation seeds only)",
        "config_path": os.path.relpath(os.path.abspath(args.config), ROOT),
        "config_snapshot": load_config(args.config),
        "mappo_config": cfg.to_dict(), "mappo_overrides": overrides,
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
    os.makedirs(os.path.join(out, "evaluation"), exist_ok=True)
    per_seed = []
    for policy_name, result in results.items():
        write_json(os.path.join(out, "evaluation", policy_name.replace(" ", "_") + ".json"), result)
        per_seed += [{"policy": policy_name, **{k: v for k, v in row.items() if k != "report"}}
                     for row in result["per_seed"]]
    _write_csv(os.path.join(out, "per_seed.csv"), per_seed)
    strip = lambda r: {k: v for k, v in r.items() if k != "per_seed"}
    write_json(os.path.join(out, "summary.json"), {
        "note": "Validation seeds only; undiscounted returns in original units",
        "mappo_best": {"step": best["step"], **strip(results["mappo_best"])},
        "mappo_final": {"step": step_of(learning_curve), **strip(results["mappo_final"])},
        "v_vs_mc": v_check,
        "destination_load": {k: destination_load(v) for k, v in results.items()},
        "baselines": {k: strip(v) for k, v in results.items() if not k.startswith("mappo_")},
    })
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
