"""Step 1 checks: chip creation, geometry, move probability."""
import numpy as np
from medaroute.env import MEDARoutingEnv


def make(**kw):
    env = MEDARoutingEnv(**kw)
    env.reset(seed=0, options={"new_chip": True})
    return env


def test_new_chip_shapes():
    env = make(height=20, width=25)
    for arr in (env.health, env.decay, env.usage):
        assert arr.shape == (20, 25)
    assert (env.usage == 0).all()


def test_decay_factors():
    env = make(height=100, width=100, frac_degradable=0.3)
    degr = env.decay < 1.0
    assert 0.25 < degr.mean() < 0.35, "about 30% of electrodes should be degradable"
    assert env.decay.min() >= 0.6 and env.decay.max() <= 1.0


def test_fresh_chip_is_healthy():
    env = make(pre_age_max=0)
    assert np.allclose(env.health, 1.0)


def test_pre_aged_chip():
    env = make(height=60, width=60, frac_degradable=0.5, pre_age_max=15)
    assert env.health.max() <= 1.0
    assert (env.health >= env.decay ** 15 - 1e-12).all()
    assert env.health.mean() < 1.0, "a pre-aged chip should have some worn electrodes"


def test_valid_centres():
    env = make(height=30, width=30, droplet_radius=1)
    assert env.valid((1, 1)) and env.valid((28, 28))
    assert not env.valid((0, 5)) and not env.valid((29, 5)) and not env.valid((5, 0))


def test_move_prob_is_footprint_mean():
    env = make(height=10, width=10, droplet_radius=1)
    env.health = np.ones((10, 10))
    env.health[4:7, 4:7] = 0.1          # exactly the footprint of centre (5, 5)
    assert abs(env.move_prob((5, 5)) - 0.1) < 1e-9
    env.health[4, 4] = 1.0              # one of nine cells healthy again
    assert abs(env.move_prob((5, 5)) - (8 * 0.1 + 1.0) / 9) < 1e-9


def test_move_prob_map_matches():
    env = make(height=15, width=18, droplet_radius=1)
    env.health = np.random.default_rng(1).random((15, 18))
    pm = env.move_prob_map()
    for y in range(1, 14):
        for x in range(1, 17):
            assert abs(pm[y, x] - env.move_prob((y, x))) < 1e-9
    assert np.isnan(pm[0, 5]) and np.isnan(pm[5, 17])
