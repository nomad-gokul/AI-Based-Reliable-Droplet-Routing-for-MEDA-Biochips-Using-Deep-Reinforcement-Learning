"""
Experiment 4 - PPO agent, with vs without the electrode-health map.

  * "PPO (no health)" : observation = droplet, goal, blocked  (Liang et al. style)
  * "PPO (health)"    : observation + electrode health map    (Elfar et al. style)

Each agent is trained with PPO (Stable-Baselines3) and a small CNN on random
pre-aged chips (a new chip every episode). Both are then evaluated next to
A* and health-aware A* on the SAME set of test chips and tasks.

Outputs (results/):
  exp4_model_health.zip, exp4_model_nohealth.zip  trained agents
  exp4_monitor_*.monitor.csv                      training logs
  exp4_learning_curve.png                         training success rate
  exp4_eval.csv, exp4_eval.png                    test comparison

Examples:
  python experiments/exp4_ppo.py --timesteps 20000           # quick smoke test
  python experiments/exp4_ppo.py                             # full run (use a GPU)
  python experiments/exp4_ppo.py --only health --timesteps 1000000
  python experiments/exp4_ppo.py --eval_only                 # re-evaluate saved models
"""
import argparse, copy, csv, math, os, sys
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from medaroute.env import MEDARoutingEnv
from medaroute.routers import ROUTERS, run_router

p = argparse.ArgumentParser()
p.add_argument("--size", type=int, default=12, help="chip is size x size microelectrodes")
p.add_argument("--frac", type=float, default=0.5, help="fraction of degradable electrodes")
p.add_argument("--pre_age", type=int, default=15)
p.add_argument("--slack", type=float, default=1.5, help="test deadline = slack x shortest path")
p.add_argument("--train_slack", type=float, default=3.0, help="episode limit during training")
p.add_argument("--timesteps", type=int, default=500_000)
p.add_argument("--n_envs", type=int, default=8)
p.add_argument("--eval_episodes", type=int, default=500)
p.add_argument("--only", choices=["health", "nohealth"], default=None)
p.add_argument("--eval_only", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# --------------------------------------------------------------- network
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


# ----------------------------------------------------------- env helpers
class FreshChipEachEpisode(gym.Wrapper):
    """New random pre-aged chip + task every episode; step limit scaled to task."""
    def __init__(self, env, slack):
        super().__init__(env)
        self.slack = slack

    def reset(self, **kwargs):
        obs, info = self.env.reset(seed=kwargs.get("seed"), options={"new_chip": True})
        e = self.env.unwrapped
        e.max_steps = math.ceil(self.slack * e._cheb(e.pos, e.goal))
        return obs, info


def make_env(observe_health, rank, log_file=None):
    def _f():
        e = MEDARoutingEnv(args.size, args.size, frac_degradable=args.frac,
                           pre_age_max=args.pre_age, observe_health=observe_health)
        e = FreshChipEachEpisode(e, args.train_slack)
        e = Monitor(e, filename=log_file, info_keywords=("reached",))
        e.reset(seed=args.seed * 1000 + rank)
        return e
    return _f


def clone_view(env, observe_health):
    """Same chip & task, observation with or without the health channel."""
    e = copy.deepcopy(env)
    e.observe_health = observe_health
    n_ch = 4 if observe_health else 3
    e.observation_space = gym.spaces.Box(0.0, 1.0, (n_ch, e.H, e.W), np.float32)
    return e


def run_agent(env, model):
    obs = env._obs()
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            return {"success": bool(term), "steps": env.steps}


# ----------------------------------------------------------------- train
variants = {"nohealth": False, "health": True}
if args.only:
    variants = {args.only: variants[args.only]}

models = {}
for name, oh in variants.items():
    path = os.path.join(args.out, f"exp4_model_{name}.zip")
    if args.eval_only:
        models[name] = PPO.load(path, device=device)
        continue
    # env 0 writes a log file with the success of every training episode
    log = os.path.join(args.out, f"exp4_monitor_{name}")
    venv = DummyVecEnv([make_env(oh, r, log if r == 0 else None) for r in range(args.n_envs)])
    model = PPO(
        "MlpPolicy", venv, device=device, seed=args.seed, verbose=0,
        n_steps=256, batch_size=512, n_epochs=4, learning_rate=3e-4,
        gamma=0.99, ent_coef=0.01,
        policy_kwargs=dict(features_extractor_class=SmallCNN,
                           features_extractor_kwargs=dict(features_dim=256),
                           net_arch=dict(pi=[128], vf=[128])),
    )
    print(f"training PPO ({name}) for {args.timesteps} steps ...")
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(path)
    models[name] = model

# ------------------------------------------------------- learning curves
if not args.eval_only:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in variants:
        f = os.path.join(args.out, f"exp4_monitor_{name}.monitor.csv")
        with open(f) as fh:
            fh.readline()                                   # skip the JSON header line
            recs = list(csv.DictReader(fh))
        s = np.array([r["reached"] == "True" for r in recs], dtype=float)
        lens = np.array([float(r["l"]) for r in recs])
        w = max(1, min(200, len(s) // 10))
        roll = np.convolve(s, np.ones(w) / w, mode="valid")
        steps = np.cumsum(lens)[w - 1:] * args.n_envs      # approx. total env steps
        ax.plot(steps, roll, label=f"PPO ({'with' if variants[name] else 'without'} health map)")
    ax.set_xlabel("Training steps (approx.)")
    ax.set_ylabel(f"Episode success (rolling, limit {args.train_slack}x)")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"PPO training on {args.size}x{args.size} chips, {args.frac:.0%} degradable")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "exp4_learning_curve.png"), dpi=200)
    plt.close(fig)

# ------------------------------------------------------------ evaluation
methods = list(ROUTERS) + [f"PPO ({n})" for n in models]
res = {m: [] for m in methods}
for i in range(args.eval_episodes):
    base = MEDARoutingEnv(args.size, args.size, frac_degradable=args.frac, pre_age_max=args.pre_age)
    base.reset(seed=999_000 + i, options={"new_chip": True})   # unseen test chips
    base.max_steps = math.ceil(args.slack * base._cheb(base.pos, base.goal))
    for m, planner in ROUTERS.items():
        res[m].append(run_router(copy.deepcopy(base), planner))
    for n, model in models.items():
        res[f"PPO ({n})"].append(run_agent(clone_view(base, n == "health"), model))

rows = []
print(f"\nTest: {args.eval_episodes} unseen chips, deadline {args.slack}x shortest path")
for m in methods:
    s = np.array([x["success"] for x in res[m]])
    st = np.array([x["steps"] for x in res[m]])
    rows.append({"method": m, "success_rate": s.mean(),
                 "mean_steps_success": st[s].mean() if s.any() else float("nan")})
    print(f"  {m:<20} success {s.mean():.3f}   steps(success) {rows[-1]['mean_steps_success']:.1f}")
with open(os.path.join(args.out, "exp4_eval.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar([r["method"] for r in rows], [r["success_rate"] for r in rows],
              color=["tab:blue", "tab:orange", "tab:gray", "tab:green"][:len(rows)])
ax.bar_label(bars, fmt="%.2f")
ax.set_ylabel(f"Success rate (deadline {args.slack}x)")
ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"{args.size}x{args.size} chips, {args.frac:.0%} degradable, {args.eval_episodes} test tasks")
plt.setp(ax.get_xticklabels(), rotation=12)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "exp4_eval.png"), dpi=200); plt.close(fig)
print("saved to", args.out)
