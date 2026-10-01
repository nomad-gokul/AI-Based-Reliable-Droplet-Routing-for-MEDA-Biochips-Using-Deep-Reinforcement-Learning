"""
Experiment 6 - Routing performance vs. blockage percentage.

Chip model (the professor's definition):
  * every microelectrode is either GOOD (health 1) or BLOCKED (health 0)
  * blockage b = (# blocked electrodes) / (# all electrodes), swept 10 ... 90 %
  * chip health = (# good electrodes) / (# all electrodes) = 1 - b
    (with binary cells this equals the mean electrode health)

Two ways a blocked electrode can affect the droplet (--mode):
  soft : the droplet may sit on blocked electrodes; a move succeeds with
         probability = fraction of good electrodes under the droplet
         (the repo's MEDA model, Liang et al. 2021)
  hard : the droplet may never cover a blocked electrode (a wall); a task whose
         goal is cut off counts as a failure (see column reachable_tasks)
With --worn, the good electrodes are also partly worn as in exp2 (50 %
degradable, pre-aged), so their health is between 0 and 1.

Methods: A*, health-aware A*, and (with --ppo) a PPO agent trained on chips
with random blockage 0-90 % that sees the blockage map.

A task succeeds if the droplet arrives within `slack` x the shortest possible
path (the bioassay deadline). "Success (no deadline)" uses a 10x budget.

Outputs in --out (file names end with the mode, chip size and options):
  blockage_results_*.csv, blockage_success_*.png,
  blockage_success_nodeadline_*.png, blockage_steps_*.png, blockage_example_*.png

  python experiments/exp6_blockage_sweep.py --mode soft
  python experiments/exp6_blockage_sweep.py --mode hard --worn
  python experiments/exp6_blockage_sweep.py --size 12 --ppo --ppo_steps 500000
Outputs go to results/blockage_sweep/ by default.
"""
import argparse, copy, csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("REPO", os.path.join(HERE, "..")))
from medaroute.env import MEDARoutingEnv, ACTIONS
from medaroute.routers import ROUTERS, run_router, astar

p = argparse.ArgumentParser()
p.add_argument("--levels", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
p.add_argument("--mode", choices=["soft", "hard"], default="soft")
p.add_argument("--n", type=int, default=300, help="chips (one task each) per level")
p.add_argument("--size", type=int, default=30)
p.add_argument("--radius", type=int, default=None, help="droplet radius (default 1 soft, 0 hard)")
p.add_argument("--slack", type=float, default=1.5)
p.add_argument("--worn", action="store_true",
               help="good electrodes are also partly worn (50%% degradable, pre-aged 15)")
p.add_argument("--ppo", action="store_true")
p.add_argument("--ppo_steps", type=int, default=500_000)
p.add_argument("--out", default=os.path.join(HERE, "..", "results", "blockage_sweep"))
args = p.parse_args()
if args.radius is None:
    args.radius = 1 if args.mode == "soft" else 0
os.makedirs(args.out, exist_ok=True)
NO_DEADLINE = 10.0


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
        if self.worn:     # good electrodes are also partly worn (repo's decay model)
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


def shortest_len(env):
    path = astar(env, env.pos, env.goal)
    return None if path is None else max(1, len(path))


def run_agent(env, model):
    obs = env._obs()
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            return {"success": bool(term), "steps": env.steps}


def make_kw(size):
    return dict(height=size, width=size, droplet_radius=args.radius,
                observe_health=False, min_task_distance=max(3, size // 6))


# ------------------------------------------------------------ optional PPO
model = None
if args.ppo:
    import gymnasium as gym
    import torch, torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class SmallCNN(BaseFeaturesExtractor):
        def __init__(self, space, features_dim=256):
            super().__init__(space, features_dim)
            c, h, w = space.shape
            self.cnn = nn.Sequential(nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Flatten())
            with torch.no_grad():
                n = self.cnn(torch.zeros(1, c, h, w)).shape[1]
            self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())
        def forward(self, x):
            return self.linear(self.cnn(x))

    class Fresh(gym.Wrapper):
        def reset(self, **kw):
            e = self.env.unwrapped
            while True:
                obs, info = self.env.reset(options={"new_chip": True})
                if e.pos is not None:
                    break
            e.max_steps = math.ceil(3.0 * e._cheb(e.pos, e.goal)) + 5
            return obs, info

    def mk(rank):
        def f():
            e = Fresh(BlockageEnv((0.0, 0.9), args.mode, args.worn, **make_kw(args.size)))
            e.reset(seed=7000 + rank)
            return e
        return f

    path = os.path.join(args.out, f"ppo_blockage_{args.mode}_{args.size}.zip")
    if os.path.exists(path):
        model = PPO.load(path, device="cpu"); print("loaded", path)
    else:
        venv = DummyVecEnv([mk(r) for r in range(8)])
        model = PPO("MlpPolicy", venv, device="cpu", seed=0, verbose=0, n_steps=256,
                    batch_size=512, n_epochs=4, learning_rate=3e-4, gamma=0.99, ent_coef=0.01,
                    policy_kwargs=dict(features_extractor_class=SmallCNN,
                                       net_arch=dict(pi=[128], vf=[128])))
        print(f"training PPO for {args.ppo_steps} steps ...", flush=True)
        model.learn(total_timesteps=args.ppo_steps)
        model.save(path)

# --------------------------------------------------------------- sweep
methods = list(ROUTERS) + (["PPO"] if model else [])
rows, example = [], None
for li, b in enumerate(args.levels):
    res = {m: [] for m in methods}; n_tasks = 0; seed = 0
    while n_tasks < args.n:
        env = BlockageEnv(b, args.mode, args.worn, **make_kw(args.size))
        env.reset(seed=100_000 * li + seed, options={"new_chip": True}); seed += 1
        if env.pos is None:
            continue
        n_tasks += 1
        L = shortest_len(env)
        if L is None:                 # goal cut off by blocked electrodes: nobody can succeed
            for m in methods:
                res[m].append({t: {"success": False, "steps": 0, "reachable": False} for t in ("dl", "nodl")})
            continue
        if example is None and abs(b - 0.3) < 1e-9:
            example = copy.deepcopy(env)
        for m in methods:
            out = {}
            for tag, sl in (("dl", args.slack), ("nodl", NO_DEADLINE)):
                e = copy.deepcopy(env); e.max_steps = math.ceil(sl * L)
                out[tag] = run_agent(e, model) if m == "PPO" else run_router(e, ROUTERS[m])
            res[m].append(out)
    for m in methods:
        s = np.array([x["dl"]["success"] for x in res[m]])
        s2 = np.array([x["nodl"]["success"] for x in res[m]])
        st = np.array([x["nodl"]["steps"] for x in res[m]])
        rows.append({"blockage": b, "chip_health": 1 - b, "method": m, "mode": args.mode,
                     "success_deadline": s.mean(), "success_no_deadline": s2.mean(),
                     "mean_steps_success": st[s2].mean() if s2.any() else float("nan"),
                     "reachable_tasks": np.mean([x["dl"].get("reachable", True) for x in res[m]]),
                     "n": len(s)})
        print(f"{b:>4.0%}  {m:<16} deadline {s.mean():.3f}  no-deadline {s2.mean():.3f}  "
              f"steps {rows[-1]['mean_steps_success']:.1f}", flush=True)

tag = f"{args.mode}_{args.size}" + ("_worn" if args.worn else "") + ("_ppo" if model else "")
with open(os.path.join(args.out, f"blockage_results_{tag}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

x = np.array(args.levels) * 100
title = (f"{args.size}x{args.size} chip, {args.mode} blockage{' + worn good cells' if args.worn else ''}, "
         f"{2*args.radius+1}x{2*args.radius+1} droplet, {args.n} tasks/level")
for metric, ylab, fn in [
        ("success_deadline", f"Success rate (deadline {args.slack}x shortest path)", "success"),
        ("success_no_deadline", f"Success rate (budget {NO_DEADLINE:.0f}x)", "success_nodeadline"),
        ("mean_steps_success", "Mean routing time of successful tasks (steps)", "steps")]:
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, mk_ in zip(methods, "osd"):
        ax.plot(x, [r[metric] for r in rows if r["method"] == m], marker=mk_, label=m)
    ax.set_xlabel("Blocked electrodes (% of chip)"); ax.set_ylabel(ylab)
    ax.set_xticks(x)
    if "success" in metric:
        ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3); ax.legend(); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"blockage_{fn}_{tag}.png"), dpi=200)
    plt.close(fig)

if example is not None:   # one 30 % chip with both A* routes drawn
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(example.health, cmap="gray", vmin=0, vmax=1)
    for m, col in zip(ROUTERS, ["tab:blue", "tab:orange"]):
        path = ROUTERS[m](example, example.pos, example.goal)
        pts = [example.pos]
        for a in path or []:
            pts.append((pts[-1][0] + ACTIONS[a][0], pts[-1][1] + ACTIONS[a][1]))
        ys, xs = zip(*pts); ax.plot(xs, ys, color=col, lw=2, label=m)
    ax.plot(example.pos[1], example.pos[0], "go", ms=9); ax.plot(example.goal[1], example.goal[0], "r*", ms=12)
    ax.set_title("30% blocked (black = blocked, grey = worn)"); ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"blockage_example_{tag}.png"), dpi=200); plt.close(fig)
print("saved to", args.out)
