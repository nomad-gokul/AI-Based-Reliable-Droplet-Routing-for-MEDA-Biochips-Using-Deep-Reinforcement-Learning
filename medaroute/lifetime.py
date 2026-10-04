"""
Chip-lifetime benchmark (Experiments 3 and 8).

A fresh chip executes a long sequence of routing tasks between a few fixed sites
(reservoirs / mixers), as a real bioassay would, and its electrodes wear out as they
are used. Every router gets the same chip and the same task sequence. Besides the
success of every task, the benchmark records how the chip degrades over time.

  run_lifetime(base_env, route, n_tasks)  ->  per-task results and wear snapshots

`route(env)` runs one task on the env (already reset, deadline set) and returns a
dict with "success"; `planner_route` and `agent_route` adapt a classical planner or
a trained agent.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .env import MEDARoutingEnv
from .routers import run_router


def lifetime_chip(seed, size=30, frac=0.5, ports=6):
    """A fresh chip with fixed sites; seed 500 + c are Experiment 3's test chips."""
    env = MEDARoutingEnv(size, size, frac_degradable=frac, task_mode="ports", n_ports=ports)
    env.reset(seed=seed, options={"new_chip": True})
    return env


def planner_route(planner):
    return lambda env: run_router(env, planner)


def agent_route(model, features=(), health=True):
    """A trained PPO agent (Experiment 7) as a route function."""
    import gymnasium as gym
    from .rl import run_agent

    def route(env):
        if env.observe_health != health:
            env.observe_health = health
            env.observation_space = gym.spaces.Box(0.0, 1.0, (4 if health else 3, env.H, env.W), np.float32)
        return run_agent(env, model, features)
    return route


def wear_snapshot(env, dead=0.1, weak=0.5):
    """How worn the chip is. An electrode is 'dead' (blocked) below `dead` health."""
    h = env.health
    used = env.actuations > 0
    return {"dead_frac": float((h < dead).mean()),          # blocked % of the chip
            "weak_frac": float((h < weak).mean()),
            "mean_health": float(h.mean()),
            "mean_health_used": float(h[used].mean()) if used.any() else 1.0,
            "electrodes_used": int(used.sum()),
            "actuations": int(env.actuations.sum())}


def run_lifetime(base_env, route, n_tasks, slack=1.5, snapshot_every=50, dead=0.1):
    """Run `n_tasks` tasks on a copy of `base_env` (same chip and task sequence for
    every router, since the copy carries the random generator). The deadline of each
    task is `slack` x its shortest path."""
    env = copy.deepcopy(base_env)
    success = np.zeros(n_tasks, bool)
    steps = np.zeros(n_tasks, np.int64)
    snaps = [{"task": 0, **wear_snapshot(env, dead)}]
    for t in range(n_tasks):
        env.reset()
        env.max_steps = math.ceil(slack * env._cheb(env.pos, env.goal))
        out = route(env)
        success[t], steps[t] = out["success"], env.steps
        if (t + 1) % snapshot_every == 0:
            snaps.append({"task": t + 1, **wear_snapshot(env, dead)})
    return {"success": success, "steps": steps, "snapshots": snaps, "env": env}


def usable_life(success, level, window=100):
    """Tasks until the rolling success rate over `window` tasks first falls below
    `level` (the full horizon if it never does)."""
    if len(success) < window:
        return len(success)
    c = np.concatenate([[0], np.cumsum(success.astype(np.int64))])
    wins = c[window:] - c[:-window]                 # successes in each window (exact integers)
    below = np.nonzero(wins < level * window - 1e-9)[0]
    return int(below[0] + window) if len(below) else len(success)


def summarise(result, levels=(0.9, 0.7, 0.5), window=100):
    s = result["success"]
    last = result["snapshots"][-1]
    out = {"overall_success": float(s.mean()),
           "tasks_before_first_failure": int(np.argmax(~s)) if (~s).any() else len(s),
           **{f"life_{int(l * 100)}": usable_life(s, l, window) for l in levels},
           "total_steps": int(result["steps"].sum()),
           "final_dead_frac": last["dead_frac"], "final_weak_frac": last["weak_frac"],
           "final_mean_health_used": last["mean_health_used"],
           "electrodes_used": last["electrodes_used"]}
    return out
