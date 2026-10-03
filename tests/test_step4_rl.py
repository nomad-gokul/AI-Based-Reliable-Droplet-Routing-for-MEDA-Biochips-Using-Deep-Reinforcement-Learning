"""Step 4 checks: cost-to-go map, deadline-optimal oracle, PPO observation wrapper."""
import copy
import math
import numpy as np
from medaroute.env import MEDARoutingEnv, ACTIONS
from medaroute.blockage import BlockageEnv
from medaroute.routers import (astar, health_astar, cost_to_go, greedy_action, deadline_optimal,
                               run_deadline_optimal, run_router)


def env_with(start, goal, size=20):
    env = MEDARoutingEnv(height=size, width=size)
    env.reset(seed=0, options={"new_chip": True, "start": start, "goal": goal})
    env.health = np.ones((size, size))
    return env


def worn_chip(seed, size=12):
    env = MEDARoutingEnv(size, size, frac_degradable=0.5, pre_age_max=15)
    env.reset(seed=seed, options={"new_chip": True})
    env.max_steps = math.ceil(1.5 * env._cheb(env.pos, env.goal))
    return env


def test_cost_to_go_is_chebyshev_on_healthy_chip():
    env = env_with((3, 4), (15, 9))
    d = cost_to_go(env)
    for pos in [(3, 4), (1, 1), (18, 18), (15, 9)]:
        assert d[pos] == max(abs(pos[0] - 15), abs(pos[1] - 9))
    assert np.isinf(d[0, 0])                     # droplet would leave the chip


def test_greedy_on_cost_to_go_follows_health_astar():
    env = env_with((2, 10), (17, 10))
    env.health[:, 9:12] = 0.2
    d = cost_to_go(env)
    pos, steps = env.pos, 0
    while pos != env.goal and steps < 100:
        a = greedy_action(env, d, pos)
        pos = (pos[0] + ACTIONS[a][0], pos[1] + ACTIONS[a][1])
        steps += 1
    assert pos == env.goal
    # expected cycles of the greedy route equal health-aware A*'s optimum from the start
    pm, p, total = env.move_prob_map(), env.pos, 0.0
    for a in health_astar(env, env.pos, env.goal):
        total += 1 / pm[p]
        p = (p[0] + ACTIONS[a][0], p[1] + ACTIONS[a][1])
    assert math.isclose(d[env.pos], total)


def test_oracle_is_certain_on_healthy_chip_and_impossible_if_too_short():
    env = env_with((3, 3), (16, 12))
    V, _ = deadline_optimal(env, 13)
    assert math.isclose(V[env.pos], 1.0)
    V, _ = deadline_optimal(env, 12)              # shortest path is 13 moves
    assert V[env.pos] == 0.0


def test_oracle_bounds_health_astar():
    for seed in range(20):
        env = worn_chip(seed)
        V, _ = deadline_optimal(env, env.max_steps)
        # success probability of health-aware A*'s fixed path within the same budget (exact)
        pm, pos, dist = env.move_prob_map(), env.pos, np.zeros(env.max_steps + 1)
        dist[0] = 1.0                             # distribution over steps used so far
        for a in health_astar(env, env.pos, env.goal):
            q, new = pm[pos], np.zeros_like(dist)
            for t in range(env.max_steps):        # geometric number of tries per move
                new[t + 1:] += dist[t] * q * (1 - q) ** np.arange(env.max_steps - t)
            dist = new
            pos = (pos[0] + ACTIONS[a][0], pos[1] + ACTIONS[a][1])
        assert V[env.pos] >= dist.sum() - 1e-9


def test_run_deadline_optimal():
    env = env_with((3, 3), (16, 12))
    env.max_steps = 13
    out = run_deadline_optimal(env)
    assert out["success"] and out["steps"] == 13


def test_routing_task_channels_and_deadline():
    from medaroute.rl import RoutingTask
    e = RoutingTask(MEDARoutingEnv(12, 12, frac_degradable=0.5, pre_age_max=15),
                    slack=1.5, features=("time", "prob", "cost"))
    obs, _ = e.reset(seed=3)
    u = e.unwrapped
    assert obs.shape == (7, 12, 12) and e.observation_space.contains(obs)
    assert u.max_steps == math.ceil(1.5 * u._cheb(u.pos, u.goal))
    assert np.allclose(obs[4], 1.0)                                 # full budget left
    assert np.allclose(obs[5], np.nan_to_num(u.move_prob_map()))    # move-success map
    assert obs[6][u.goal] == 0.0                                    # cost-to-go is 0 at the goal
    obs, *_ = e.step(0)
    assert np.allclose(obs[4], 1 - 1 / u.max_steps)


def test_routing_task_shaping_rewards_progress():
    from medaroute.rl import RoutingTask
    e = RoutingTask(MEDARoutingEnv(20, 20), slack=1.5, shaping=0.1)
    e.reset(seed=0)
    u = e.unwrapped
    u.health[:] = 1.0
    u.pos, u.goal, u.steps, u.max_steps = (10, 10), (10, 16), 0, 50
    _, r_toward, *_ = e.step(1)                   # E: one step closer
    u.pos, u.steps = (10, 10), 0
    _, r_away, *_ = e.step(3)                     # W: one step further
    assert r_toward > -0.05 and r_away < -0.1     # base rewards are -0.05 / -0.1


def test_routing_task_skips_cut_off_tasks():
    from medaroute.rl import RoutingTask
    e = RoutingTask(BlockageEnv(0.7, "hard", height=12, width=12, droplet_radius=0,
                                min_task_distance=3), slack=1.5)
    for s in range(10):
        e.reset(seed=s)
        u = e.unwrapped
        assert astar(u, u.pos, u.goal) is not None


def test_collect_expert_shapes():
    from medaroute.rl import RoutingTask, collect_expert
    e = RoutingTask(MEDARoutingEnv(12, 12, frac_degradable=0.5, pre_age_max=15), slack=1.5)
    e.reset(seed=1)
    obs, act, ret = collect_expert(e, 20)
    assert len(obs) == len(act) == len(ret) and obs.shape[1:] == (4, 12, 12)
    assert act.min() >= 0 and act.max() < 8
