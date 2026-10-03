"""
Reinforcement-learning building blocks for the health-aware PPO agent (Experiment 7).

* SmallCNN           : 3-layer CNN feature extractor for small chips
* RoutingTask        : Gymnasium wrapper; every episode is a new chip and task, the step limit
                       is `slack` x the shortest path, with optional extra observation channels,
                       a health-based reward term and a curriculum hook
* add_channels       : the extra observation channels
      "time" : fraction of the step budget left (lets the agent know the deadline)
      "prob" : move-success probability at every droplet centre (mean health under the droplet)
      "cost" : health-aware A*'s cost-to-go to the goal (expected cycles, scaled to [0, 1])
* collect_expert / pretrain_bc : behaviour cloning from a classical router (warm start for PPO)
* CurriculumCallback : raises the chip difficulty from easy to the target during training
* run_agent          : run a trained model on an environment (closed loop)

The observation always starts with the environment's own channels (droplet, goal, blocked,
and electrode health when observe_health=True); the extra channels are appended after them.
"""
from __future__ import annotations

import math

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .routers import astar, cost_to_go, greedy_action

FEATURES = ("time", "prob", "cost")


class SmallCNN(BaseFeaturesExtractor):
    """3 conv layers (3x3, padding 1) -> linear. Works on small grids,
    unlike SB3's default NatureCNN which expects >= 36x36 images."""
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))


# ------------------------------------------------------------ observation
class CostCache:
    """Health-aware cost-to-go for the current goal, recomputed only when health changes."""
    def __init__(self):
        self.key, self.dist = None, None

    def get(self, env):
        key = (env.goal, float(env.health.sum()))
        if key != self.key:
            self.key, self.dist = key, cost_to_go(env, health_aware=True)
        return self.dist


def add_channels(env, obs, features, cache: CostCache | None = None):
    """Append the requested feature channels to the environment's observation."""
    if not features:
        return obs
    extra = []
    for f in features:
        if f == "time":
            left = (env.max_steps - env.steps) / max(1, env.max_steps)
            extra.append(np.full((env.H, env.W), np.clip(left, 0.0, 1.0)))
        elif f == "prob":
            extra.append(np.nan_to_num(env.move_prob_map(), nan=0.0))
        elif f == "cost":
            dist = (cache or CostCache()).get(env)
            extra.append(np.clip(dist / (4 * max(env.H, env.W)), 0.0, 1.0))   # inf -> 1
        else:
            raise ValueError(f"unknown feature {f!r}, choose from {FEATURES}")
    return np.concatenate([obs, np.stack(extra).astype(np.float32)])


# ---------------------------------------------------------------- wrapper
class RoutingTask(gym.Wrapper):
    """One episode = a new chip and task. The step limit is `slack` x the shortest path,
    so training can use the same deadline as testing.

    shaping > 0 adds potential-based reward shaping with phi(s) = -(health-aware cost-to-go):
    r' = r + shaping * (gamma * phi(s') - phi(s)). Moving onto worn electrodes (or towards a
    worn region) raises the expected remaining time and is penalised immediately, yet by
    Ng et al. (1999) the optimal policy is unchanged. (The simpler health term is the
    environment's own `degrade_penalty`.)

    curriculum(env_unwrapped, difficulty) is called before every new chip; `difficulty`
    (0 = easiest, 1 = target task) is raised during training by CurriculumCallback."""
    def __init__(self, env, slack=1.5, features=(), shaping=0.0, gamma=0.99, curriculum=None):
        super().__init__(env)
        self.slack, self.features, self.shaping, self.gamma = slack, tuple(features), shaping, gamma
        self.curriculum, self.difficulty = curriculum, 1.0
        self.cache = CostCache()
        c, h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, (c + len(self.features), h, w), np.float32)

    def set_difficulty(self, d):
        self.difficulty = float(d)

    def reset(self, *, seed=None, options=None):
        e = self.env.unwrapped
        if self.curriculum is not None:
            self.curriculum(e, self.difficulty)
        while True:
            obs, info = self.env.reset(seed=seed, options={"new_chip": True})
            seed = None
            path = astar(e, e.pos, e.goal) if e.pos is not None else None
            if path:                       # skip chips with no task / goal cut off (hard blockage)
                break
        e.max_steps = math.ceil(self.slack * len(path))
        return add_channels(e, obs, self.features, self.cache), info

    def _phi(self):
        d = self.cache.get(self.env.unwrapped)[self.env.unwrapped.pos]
        return -min(d, 4 * max(self.env.unwrapped.H, self.env.unwrapped.W))

    def step(self, action):
        phi0 = self._phi() if self.shaping else 0.0
        obs, r, term, trunc, info = self.env.step(action)
        if self.shaping:
            phi1 = 0.0 if term else self._phi()
            r += self.shaping * (self.gamma * phi1 - phi0)
        return add_channels(self.env.unwrapped, obs, self.features, self.cache), r, term, trunc, info


class CurriculumCallback(BaseCallback):
    """Raise the environments' difficulty linearly from `start` to 1 over the first
    `frac` of training, then keep it at 1 (the test task)."""
    def __init__(self, total_timesteps, frac=0.5, start=0.0):
        super().__init__()
        self.total, self.frac, self.start = total_timesteps, frac, start

    def _on_step(self):
        d = min(1.0, self.start + (1 - self.start) * self.num_timesteps / max(1, self.frac * self.total))
        self.training_env.env_method("set_difficulty", d)
        return True


# --------------------------------------------------------- imitation warm start
def collect_expert(env: RoutingTask, n_episodes, health_aware=True, gamma=0.99):
    """Roll out the classical router (health-aware A* or plain A*, replanned every step)
    in a RoutingTask env. Returns observations, expert actions and discounted returns."""
    obs_l, act_l, ret_l = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        e = env.unwrapped
        rews = []
        while True:
            dist = env.cache.get(e) if health_aware else cost_to_go(e, health_aware=False)
            a = greedy_action(e, dist)
            a = 0 if a is None else a
            obs_l.append(obs); act_l.append(a)
            obs, r, term, trunc, _ = env.step(a)
            rews.append(r)
            if term or trunc:
                break
        g, rets = 0.0, []
        for r in reversed(rews):
            g = r + gamma * g
            rets.append(g)
        ret_l.extend(rets[::-1])
    return np.array(obs_l, np.float32), np.array(act_l), np.array(ret_l, np.float32)


def pretrain_bc(model, obs, actions, returns, epochs=15, value_epochs=5, batch_size=256, lr=3e-4, seed=0):
    """Behaviour cloning in two stages, so that PPO fine-tuning starts from a working
    router instead of a random one:
      1. the whole policy network imitates the expert's actions (cross-entropy)
      2. with the shared CNN frozen, the value head learns the expert's returns
    (Training both at once lets the value loss swamp the imitation loss.)"""
    policy = model.policy
    rng = np.random.default_rng(seed)
    n, dev = len(obs), policy.device
    obs_t = torch.as_tensor(obs, device=dev)
    act_t = torch.as_tensor(actions, device=dev)
    ret_t = torch.as_tensor(returns, device=dev)
    value_params = list(policy.mlp_extractor.value_net.parameters()) + list(policy.value_net.parameters())
    policy.set_training_mode(True)
    for stage, n_ep, params in [("policy", epochs, policy.parameters()), ("value", value_epochs, value_params)]:
        opt = torch.optim.Adam(params, lr=lr)
        for ep in range(n_ep):
            idx = rng.permutation(n)
            tot, acc = 0.0, 0.0
            for i in range(0, n, batch_size):
                b = idx[i:i + batch_size]
                values, log_prob, _ = policy.evaluate_actions(obs_t[b], act_t[b])
                if stage == "policy":
                    loss = -log_prob.mean()
                    acc += (policy.get_distribution(obs_t[b]).distribution.probs.argmax(1)
                            == act_t[b]).sum().item()
                else:
                    loss = ((values.flatten() - ret_t[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
            msg = f", expert-action accuracy {acc / n:.3f}" if stage == "policy" else ""
            print(f"  BC {stage} epoch {ep + 1}/{n_ep}: loss {tot / n:.3f}{msg}", flush=True)
    policy.set_training_mode(False)


# --------------------------------------------------------------- evaluation
def run_agent(env, model, features=()):
    """Run the model on `env` (already reset, with max_steps set) until the episode ends."""
    cache = CostCache()
    obs = add_channels(env, env._obs(), features, cache)
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(int(a))
        obs = add_channels(env, obs, features, cache)
        if term or trunc:
            return {"success": bool(term), "steps": env.steps}
