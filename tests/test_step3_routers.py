"""Step 3 checks: A*, health-aware A*, executor."""
import copy
import numpy as np
from medaroute.env import MEDARoutingEnv, ACTIONS
from medaroute.routers import astar, health_astar, run_router


def env_with(start, goal, size=20):
    env = MEDARoutingEnv(height=size, width=size)
    env.reset(seed=0, options={"new_chip": True, "start": start, "goal": goal})
    env.health = np.ones((size, size))
    return env


def follow(start, path):
    pos = start
    for a in path:
        pos = (pos[0] + ACTIONS[a][0], pos[1] + ACTIONS[a][1])
    return pos


def expected_cycles(env, start, path):
    pm, pos, total = env.move_prob_map(), start, 0.0
    for a in path:
        total += 1 / max(pm[pos], 1e-3)
        pos = (pos[0] + ACTIONS[a][0], pos[1] + ACTIONS[a][1])
    return total


def test_astar_shortest_path():
    for s, g in [((2, 2), (15, 9)), ((17, 3), (3, 17)), ((5, 10), (5, 2))]:
        env = env_with(s, g)
        path = astar(env, s, g)
        assert follow(s, path) == g
        assert len(path) == max(abs(s[0] - g[0]), abs(s[1] - g[1])), "8-directional shortest path"


def test_health_astar_avoids_weak_band():
    env = env_with((2, 10), (17, 10))
    env.health[:, 9:12] = 0.2          # weak band straight down the middle
    p_plain = astar(env, env.pos, env.goal)
    p_health = health_astar(env, env.pos, env.goal)
    assert follow(env.pos, p_health) == env.goal
    assert expected_cycles(env, env.pos, p_health) < expected_cycles(env, env.pos, p_plain)


def test_health_astar_equals_astar_on_healthy_chip():
    env = env_with((3, 3), (16, 12))
    assert len(health_astar(env, env.pos, env.goal)) == len(astar(env, env.pos, env.goal))


def test_run_router():
    env = env_with((3, 3), (16, 12))
    out = run_router(env, astar)
    assert out["success"] and out["steps"] == out["path_len"] == 13
