"""Step 5 checks: actuation counts, wear-aware A*, the lifetime benchmark."""
import math
import numpy as np
from medaroute.env import MEDARoutingEnv, ACTIONS
from medaroute.routers import (astar, health_astar, estimate_decay, wear_cost_map, make_wear_astar,
                               expected_cycles, run_router)
from medaroute.lifetime import lifetime_chip, run_lifetime, planner_route, summarise, usable_life


def env_with(start, goal, size=20):
    env = MEDARoutingEnv(height=size, width=size)
    env.reset(seed=0, options={"new_chip": True, "start": start, "goal": goal})
    env.health = np.ones((size, size))
    env.decay = np.full((size, size), 0.8)
    env.actuations[:] = 0
    return env


def test_actuations_count_every_cycle():
    env = env_with((10, 10), (2, 2))
    env.step(1)                                   # E: footprint around (10, 11) is actuated
    env.step(1)
    assert env.actuations.sum() == 2 * 9
    assert env.actuations[10, 12] == 2 and env.actuations[10, 9] == 0


def test_estimate_decay_from_counts_and_health():
    env = env_with((10, 10), (2, 2))
    env.actuations[5, 5], env.health[5, 5] = 100, 0.64     # two wear events: 0.8^2
    env.actuations[6, 6], env.health[6, 6] = 120, 1.0      # worn twice, still healthy: never degrades
    d = estimate_decay(env, prior=0.9)
    assert math.isclose(d[5, 5], 0.8) and d[6, 6] == 1.0 and d[0, 0] == 0.9


def test_wear_cost_zero_lambda_is_health_astar():
    for seed in range(5):
        env = lifetime_chip(900 + seed, size=20)
        env.reset()
        assert make_wear_astar(0.0)(env, env.pos, env.goal) == health_astar(env, env.pos, env.goal)


def test_wear_astar_avoids_electrodes_that_degrade():
    env = env_with((2, 10), (17, 10))
    env.actuations[:] = 100                       # every electrode has been worn twice ...
    env.health[:] = 0.81                          # ... and decays by 0.9 each time,
    env.health[:, 3:8] = 1.0                      # except a band that never degrades
    straight = health_astar(env, env.pos, env.goal)
    wear = make_wear_astar(4.0)(env, env.pos, env.goal)
    cols = lambda path: [env.pos[1] + sum(ACTIONS[a][1] for a in path[:i + 1]) for i in range(len(path))]
    assert max(cols(straight)) >= 9               # health-aware A* goes straight down
    assert min(cols(wear)) <= 7                   # wear-aware A* detours into the durable band


def test_wear_astar_budget_respects_deadline():
    env = env_with((2, 10), (17, 10))
    env.actuations[:] = 100
    env.health[:] = 0.81
    env.health[:, 3:8] = 1.0
    env.max_steps, env.steps = 15, 0              # no slack at all: must race
    fast = health_astar(env, env.pos, env.goal)
    path = make_wear_astar(4.0, budget=0.5)(env, env.pos, env.goal)
    assert expected_cycles(env, env.pos, path) <= expected_cycles(env, env.pos, fast) + 1e-9


def test_port_placement_restarts_when_stuck():
    for c in range(20):                           # 4 sites 6 apart on 12x12 always fit after restarts
        ports = lifetime_chip(500 + c, size=12, ports=4).ports
        assert len(ports) == 4
        assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 6 for a in ports for b in ports if a != b)


def test_lifetime_same_tasks_for_every_router_and_summary():
    base = lifetime_chip(500, size=20)
    r1 = run_lifetime(base, planner_route(astar), 60, snapshot_every=20)
    r2 = run_lifetime(base, planner_route(astar), 60, snapshot_every=20)
    assert (r1["success"] == r2["success"]).all() and len(r1["snapshots"]) == 4
    s = summarise(r1)
    assert 0 <= s["overall_success"] <= 1 and s["life_90"] == 60     # horizon shorter than the window
    assert base.actuations.sum() == 0             # the benchmark works on a copy


def test_usable_life():
    s = np.array([True] * 200 + [False] * 100)
    assert usable_life(s, 0.9) == 200 + 11        # rolling mean of 100 drops below 0.9 after 11 failures
    assert usable_life(np.ones(300, bool), 0.9) == 300
