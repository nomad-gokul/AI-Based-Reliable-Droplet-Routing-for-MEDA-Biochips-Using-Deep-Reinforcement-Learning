"""Step 2 checks: reset, step, rewards, wear, Gymnasium compatibility."""
import numpy as np
from gymnasium.utils.env_checker import check_env
from medaroute.env import MEDARoutingEnv

N, E, S, W, NE, SE, SW, NW = range(8)


def healthy_env(start, goal, **kw):
    env = MEDARoutingEnv(height=20, width=20, **kw)
    env.reset(seed=0, options={"new_chip": True, "start": start, "goal": goal})
    env.health = np.ones((20, 20))
    return env


def test_reset_samples_valid_task():
    env = MEDARoutingEnv(height=20, width=20)
    for s in range(20):
        obs, info = env.reset(seed=s, options={"new_chip": True})
        assert obs.shape == env.observation_space.shape
        assert env.valid(env.pos) and env.valid(env.goal)
        cheb = max(abs(env.pos[0] - env.goal[0]), abs(env.pos[1] - env.goal[1]))
        assert cheb >= env.min_task_distance


def test_moves_on_healthy_chip():
    env = healthy_env((10, 10), (2, 2))
    env.step(E);  assert env.pos == (10, 11)
    env.step(N);  assert env.pos == (9, 11)
    env.step(SW); assert env.pos == (10, 10)


def test_dead_electrodes_block_movement():
    env = healthy_env((10, 10), (2, 2))
    env.health[:] = 0.0
    for _ in range(20):
        _, _, _, _, info = env.step(N)
        assert not info["moved"]
    assert env.pos == (10, 10)


def test_wall_is_a_no_op():
    env = healthy_env((1, 10), (15, 10))
    env.step(N)
    assert env.pos == (1, 10)


def test_rewards():
    env = healthy_env((10, 10), (10, 13))
    _, r, term, _, _ = env.step(E); assert abs(r - (-0.05)) < 1e-9 and not term   # closer
    _, r, _, _, _ = env.step(W);    assert abs(r - (-0.1)) < 1e-9                  # farther
    env.step(E); env.step(E)
    _, r, term, _, _ = env.step(E); assert r == 1.0 and term                       # goal


def test_wear_degrades_after_threshold():
    env = healthy_env((10, 10), (2, 2), usage_threshold=50)
    env.decay[:] = 0.5
    for _ in range(49):
        env._wear(env.pos)
    assert np.allclose(env.health, 1.0), "no decay before the threshold"
    env._wear(env.pos)
    assert abs(env.health[10, 10] - 0.5) < 1e-9 and env.usage[10, 10] == 0
    assert env.health[0, 0] == 1.0, "electrodes not under the droplet are unaffected"


def test_truncation():
    env = healthy_env((10, 10), (2, 2), max_steps=5)
    env.health[:] = 0.0
    for i in range(5):
        _, _, term, trunc, _ = env.step(N)
    assert trunc and not term


def test_observation_channels():
    env = healthy_env((5, 5), (12, 12))
    obs = env._obs()
    assert obs.shape == (4, 20, 20) and obs.dtype == np.float32
    assert obs[0].sum() == 9 and obs[0, 5, 5] == 1      # droplet
    assert obs[1].sum() == 9 and obs[1, 12, 12] == 1    # goal
    assert np.allclose(obs[3], env.health)              # health
    env2 = MEDARoutingEnv(height=20, width=20, observe_health=False)
    assert env2.observation_space.shape == (3, 20, 20)


def test_gymnasium_checker():
    check_env(MEDARoutingEnv(), skip_render_check=True)
