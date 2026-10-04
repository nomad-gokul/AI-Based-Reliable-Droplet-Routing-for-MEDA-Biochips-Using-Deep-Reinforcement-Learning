"""
MEDA biochip droplet-routing environment (single droplet), Gymnasium API.

Chip model follows Liang et al., "Parallel Droplet Control in MEDA Biochips
using Multi-Agent Reinforcement Learning" (ICML 2021) and their meda-env code:
  * a droplet is a square block of microelectrodes (side = 2r + 1)
  * it can move in 8 directions (N, E, S, W and the diagonals)
  * a move succeeds with probability = mean health of the electrodes under it
  * a fraction of electrodes is "degradable": each gets a decay factor in
    [0.6, 1.0]; every `usage_threshold` actuations its health is multiplied
    by that factor (charge trapping)
  * reward: +1 at the goal, -0.05 for a step that gets closer, -0.1 otherwise

Following Elfar et al. (IEEE TCAD 2023), the observation can include the
electrode health map as an extra channel (observe_health=True).

Deliberate differences from meda-env (documented for the report):
  * the droplet moves one microelectrode per step (meda-env: 3 orthogonal,
    2 diagonal), which is closer to MEDA's fine-grained control
  * success means the droplet exactly covers the goal footprint
  * health is updated after every step, so wear accumulates within a task
  * `pre_age_max` optionally starts a chip already partly worn, and
    `task_mode="ports"` draws tasks between fixed reservoir/mixer sites
  * `actuations` counts every actuation of every electrode since the chip was
    made (the controller drives the electrodes, so it knows these counts)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# (dy, dx) for N, E, S, W, NE, SE, SW, NW  (y grows downward)
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
ACTION_NAMES = ["N", "E", "S", "W", "NE", "SE", "SW", "NW"]


class MEDARoutingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        height: int = 30,
        width: int = 30,
        droplet_radius: int = 1,
        frac_degradable: float = 0.3,
        decay_range: tuple[float, float] = (0.6, 1.0),
        usage_threshold: int = 50,
        pre_age_max: int = 0,
        observe_health: bool = True,
        max_steps: int | None = None,
        degrade_penalty: float = 0.0,
        task_mode: str = "random",
        n_ports: int = 6,
        min_task_distance: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.H, self.W, self.r = height, width, droplet_radius
        self.frac_degradable = frac_degradable
        self.decay_range = decay_range
        self.usage_threshold = usage_threshold
        self.pre_age_max = pre_age_max
        self.observe_health = observe_health
        self.max_steps = max_steps or 2 * (height + width)
        self.degrade_penalty = degrade_penalty
        self.task_mode = task_mode
        self.n_ports = n_ports
        self.min_task_distance = min_task_distance or 2 * (2 * droplet_radius + 1)
        self.render_mode = render_mode

        n_ch = 4 if observe_health else 3
        self.observation_space = spaces.Box(0.0, 1.0, (n_ch, height, width), np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self._chip_ready = False

    # ------------------------------------------------------------------ chip
    def new_chip(self):
        """Create a fresh chip: decay factors, usage counters and health."""
        rng = self.np_random
        degradable = rng.random((self.H, self.W)) < self.frac_degradable
        self.decay = np.ones((self.H, self.W))
        lo, hi = self.decay_range
        self.decay[degradable] = rng.uniform(lo, hi, degradable.sum())
        self.usage = np.zeros((self.H, self.W), dtype=np.int64)
        self.actuations = np.zeros((self.H, self.W), dtype=np.int64)   # total, never reset
        if self.pre_age_max > 0:
            k = rng.integers(0, self.pre_age_max + 1, (self.H, self.W))
            self.health = self.decay ** k
        else:
            self.health = np.ones((self.H, self.W))
        self.ports = self._make_ports() if self.task_mode == "ports" else None
        self._chip_ready = True

    def _make_ports(self):
        """Fixed sites (reservoirs on the edges, mixers inside)."""
        rng, r = self.np_random, self.r
        ports, misses = [], 0
        while len(ports) < self.n_ports:
            p = (int(rng.integers(r, self.H - r)), int(rng.integers(r, self.W - r)))
            if all(self._cheb(p, q) >= self.min_task_distance for q in ports):
                ports.append(p)
            else:
                misses += 1
                if misses >= 1000:      # the sites so far leave no room: start again
                    ports, misses = [], 0
        return ports

    # ------------------------------------------------------------- geometry
    def _fp(self, pos):
        """Slices covering the droplet footprint centred at pos=(y, x)."""
        y, x = pos
        return slice(y - self.r, y + self.r + 1), slice(x - self.r, x + self.r + 1)

    def valid(self, pos):
        y, x = pos
        return self.r <= y < self.H - self.r and self.r <= x < self.W - self.r

    @staticmethod
    def _cheb(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @staticmethod
    def _euclid(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def move_prob(self, pos):
        ys, xs = self._fp(pos)
        return float(self.health[ys, xs].mean())

    def move_prob_map(self):
        """Move-success probability for every droplet centre (vectorised).
        Invalid centres (droplet would leave the chip) are NaN."""
        k = 2 * self.r + 1
        c = np.pad(self.health, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        box = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
        out = np.full((self.H, self.W), np.nan)
        out[self.r:self.H - self.r, self.r:self.W - self.r] = box
        return out

    # ---------------------------------------------------------------- tasks
    def sample_task(self):
        rng = self.np_random
        if self.task_mode == "ports":
            i, j = rng.choice(len(self.ports), size=2, replace=False)
            return self.ports[i], self.ports[j]
        while True:
            s = (int(rng.integers(self.r, self.H - self.r)), int(rng.integers(self.r, self.W - self.r)))
            g = (int(rng.integers(self.r, self.H - self.r)), int(rng.integers(self.r, self.W - self.r)))
            if self._cheb(s, g) >= self.min_task_distance:
                return s, g

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if options.get("new_chip", False) or not self._chip_ready:
            self.new_chip()
        if "start" in options and "goal" in options:
            self.pos, self.goal = tuple(options["start"]), tuple(options["goal"])
        else:
            self.pos, self.goal = self.sample_task()
        self.steps = 0
        return self._obs(), self._info(moved=False)

    def step(self, action):
        self.steps += 1
        dy, dx = ACTIONS[int(action)]
        old_d = self._euclid(self.pos, self.goal)
        p = self.move_prob(self.pos)
        nxt = (self.pos[0] + dy, self.pos[1] + dx)
        moved = False
        if self.valid(nxt) and self.np_random.random() < p:
            self.pos, moved = nxt, True
        self._wear(self.pos)

        reached = self.pos == self.goal
        if reached:
            reward = 1.0
        elif self._euclid(self.pos, self.goal) < old_d:
            reward = -0.05
        else:
            reward = -0.1
        if self.degrade_penalty:
            reward -= self.degrade_penalty * (1.0 - self.move_prob(self.pos))

        terminated = reached
        truncated = (not reached) and self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, self._info(moved=moved, p=p)

    def _wear(self, pos):
        """Electrodes under the droplet are actuated; degrade past threshold."""
        ys, xs = self._fp(pos)
        self.usage[ys, xs] += 1
        self.actuations[ys, xs] += 1
        hit = self.usage >= self.usage_threshold
        if hit.any():
            self.health[hit] *= self.decay[hit]
            self.usage[hit] = 0

    def _obs(self):
        n_ch = self.observation_space.shape[0]
        obs = np.zeros((n_ch, self.H, self.W), dtype=np.float32)
        ys, xs = self._fp(self.pos)
        obs[0, ys, xs] = 1.0          # channel 0: droplet
        ys, xs = self._fp(self.goal)
        obs[1, ys, xs] = 1.0          # channel 1: goal
        # channel 2: blocked cells / other droplets (empty for one droplet)
        if self.observe_health:
            obs[3] = self.health       # channel 3: electrode health
        return obs

    def _info(self, moved, p=None):
        return {
            "pos": self.pos, "goal": self.goal, "moved": moved,
            "move_prob": p, "reached": self.pos == self.goal,
        }

    def render(self):
        img = (np.stack([self.health] * 3, -1) * 220 + 20).astype(np.uint8)
        ys, xs = self._fp(self.goal)
        img[ys, xs] = (40, 200, 40)
        ys, xs = self._fp(self.pos)
        img[ys, xs] = (40, 90, 230)
        return img
