"""
Experiment 2 - Baselines across degradation levels.

For each fraction of degradable electrodes, generate N pre-aged chips with
one random routing task each. Both routers solve the SAME chip and task.
A task succeeds if the droplet arrives within `slack` x its shortest-path
length (a bioassay deadline).

Outputs (results/):
  exp2_results.csv, exp2_success.png, exp2_steps.png
"""
import argparse, copy, csv, math, os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from medaroute.env import MEDARoutingEnv
from medaroute.routers import ROUTERS, run_router

p = argparse.ArgumentParser()
p.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5, 0.7])
p.add_argument("--n", type=int, default=500, help="chips (tasks) per level")
p.add_argument("--size", type=int, default=30)
p.add_argument("--pre_age", type=int, default=15)
p.add_argument("--slack", type=float, default=1.5)
p.add_argument("--out", default="results")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

rows = []
for li, frac in enumerate(args.levels):
    res = {k: [] for k in ROUTERS}
    for i in range(args.n):
        env = MEDARoutingEnv(args.size, args.size, frac_degradable=frac, pre_age_max=args.pre_age)
        env.reset(seed=10_000 * li + i, options={"new_chip": True})
        env.max_steps = math.ceil(args.slack * env._cheb(env.pos, env.goal))
        for name, planner in ROUTERS.items():
            res[name].append(run_router(copy.deepcopy(env), planner))
    for name, v in res.items():
        succ = np.array([x["success"] for x in v])
        steps = np.array([x["steps"] for x in v])
        plen = np.array([x["path_len"] for x in v])
        rows.append({
            "frac_degradable": frac, "router": name,
            "success_rate": succ.mean(),
            "mean_steps_success": steps[succ].mean() if succ.any() else float("nan"),
            "mean_path_len": plen.mean(),
            "n": len(v),
        })
        print(f"{frac:>4.0%}  {name:<16} success={succ.mean():.3f}  "
              f"steps(success)={rows[-1]['mean_steps_success']:.1f}  path={plen.mean():.1f}")

with open(os.path.join(args.out, "exp2_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

# ---- plots
names = list(ROUTERS)
x = np.arange(len(args.levels)); bw = 0.38
labels = [f"{l:.0%}" for l in args.levels]
for metric, ylabel, fname in [
    ("success_rate", f"Success rate (deadline = {args.slack}x shortest path)", "exp2_success.png"),
    ("mean_steps_success", "Mean routing time of successful tasks (steps)", "exp2_steps.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, name in enumerate(names):
        vals = [r[metric] for r in rows if r["router"] == name]
        bars = ax.bar(x + (k - 0.5) * bw, vals, bw, label=name)
        ax.bar_label(bars, fmt="%.2f" if metric == "success_rate" else "%.1f", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Fraction of degradable microelectrodes")
    ax.set_ylabel(ylabel)
    if metric == "success_rate":
        ax.set_ylim(0, 1.1)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, fname), dpi=200); plt.close(fig)
print("saved to", args.out)
