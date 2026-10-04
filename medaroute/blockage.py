"""
Binary-blockage chip (the supervisor's definition of chip health), used by Experiments 6 and 7.

  * every microelectrode is either GOOD (health 1) or BLOCKED (health 0)
  * blockage b = (# blocked electrodes) / (# all electrodes)
  * chip health = (# good electrodes) / (# all electrodes) = 1 - b

mode="soft": the droplet may sit on blocked electrodes; a move succeeds with probability =
fraction of good electrodes under it. mode="hard": a blocked electrode is a wall the droplet
may never cover. worn=True: the good electrodes are also partly worn (50 % degradable,
pre-aged 15), so their health is between 0 and 1.
"""
from __future__ import annotations

import numpy as np

from .env import MEDARoutingEnv


class BlockageEnv(MEDARoutingEnv):
    """Binary chip: each electrode is blocked with probability `blockage`
    (a float, or a (lo, hi) range sampled per chip for training)."""
    def __init__(self, blockage, mode="soft", worn=False, **kw):
        if worn:
            kw.update(frac_degradable=0.5, pre_age_max=15)
        else:
            kw.update(frac_degradable=0.0, pre_age_max=0)
        super().__init__(**kw)
        self.blockage, self.mode, self.worn = blockage, mode, worn

    def new_chip(self):
        rng = self.np_random
        b = self.blockage if np.isscalar(self.blockage) else rng.uniform(*self.blockage)
        self.blocked = np.zeros((self.H, self.W), bool)   # so valid() works inside new_chip
        super().new_chip()
        self.blocked = rng.random((self.H, self.W)) < b
        self.health = np.where(self.blocked, 0.0, self.health)
        if self.mode == "hard":
            self._free = [(y, x) for y in range(self.H) for x in range(self.W) if self.valid((y, x))]

    # hard mode: a centre is usable only if its footprint has no blocked cell
    def valid(self, pos):
        if not super().valid(pos):
            return False
        if self.mode == "hard":
            ys, xs = self._fp(pos)
            return not self.blocked[ys, xs].any()
        return True

    def sample_task(self):
        if self.mode == "soft":
            return super().sample_task()
        rng, free = self.np_random, self._free
        if len(free) < 2:
            return None, None
        for _ in range(200):
            s, g = (free[i] for i in rng.choice(len(free), 2, replace=False))
            if self._cheb(s, g) >= self.min_task_distance:
                return s, g
        s, g = (free[i] for i in rng.choice(len(free), 2, replace=False))
        return s, g

    def _obs(self):
        obs = super()._obs()
        obs[2] = self.blocked            # channel 2: blocked electrodes
        return obs
