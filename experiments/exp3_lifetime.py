"""
Experiment 3 - Chip lifetime.

A fresh chip executes a long sequence of routing tasks between a few fixed
sites (reservoirs / mixers), as a real bioassay would. Electrodes wear out as
they are used. Both routers get the same chip and the same task sequence.

Outputs (results/):
  exp3_results.csv    per-chip: tasks before first failure, success rate
  exp3_lifetime.png   rolling success rate vs number of tasks executed
  exp3_health.png     final electrode health for each router (one chip)
"""
import argparse, copy, csv, math, os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from medaroute.env import MEDARoutingEnv
from medaroute.routers import ROUTERS, run_router

p = argparse.ArgumentParser()
p.add_argument("--chips", type=int, default=10)
p.add_argument("--tasks", type=int, default=3000)
p.add_argument("--size", type=int, default=30)
p.add_argument("--frac", type=float, default=0.5)
p.add_argument("--ports", type=int, default=6)
p.add_argument("--slack", type=float, default=1.5)
p.add_argument("--window", type=int, default=100)
p.add_argument("--out", default="results")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

succ_curves = {k: [] for k in ROUTERS}
rows, final = [], {}
for c in range(args.chips):
    base = MEDARoutingEnv(args.size, args.size, frac_degradable=args.frac,
                          task_mode="ports", n_ports=args.ports)
    base.reset(seed=500 + c, options={"new_chip": True})
    for name, planner in ROUTERS.items():
        env = copy.deepcopy(base)  # same chip, same RNG -> same task sequence
        s = np.zeros(args.tasks, dtype=bool)
        for t in range(args.tasks):
            env.reset()
            env.max_steps = math.ceil(args.slack * env._cheb(env.pos, env.goal))
            s[t] = run_router(env, planner)["success"]
        first_fail = int(np.argmax(~s)) if (~s).any() else args.tasks
        succ_curves[name].append(s)
        rows.append({"chip": c, "router": name, "tasks_before_first_failure": first_fail,
                     "overall_success": s.mean(), "final_mean_health": env.health.mean()})
        if c == 0:
            final[name] = (env.health.copy(), env.ports)
        print(f"chip {c}  {name:<16} first failure at task {first_fail:>4}  "
              f"overall success {s.mean():.3f}")

with open(os.path.join(args.out, "exp3_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print("\nSummary (mean +- std over chips)")
for name in ROUTERS:
    ff = [r["tasks_before_first_failure"] for r in rows if r["router"] == name]
    os_ = [r["overall_success"] for r in rows if r["router"] == name]
    print(f"  {name:<16} tasks before first failure {np.mean(ff):7.1f} +- {np.std(ff):6.1f}"
          f"   overall success {np.mean(os_):.3f} +- {np.std(os_):.3f}")

# ---- rolling success curve
fig, ax = plt.subplots(figsize=(7, 4))
kern = np.ones(args.window) / args.window
for name, curves in succ_curves.items():
    roll = np.array([np.convolve(s.astype(float), kern, mode="valid") for s in curves])
    t = np.arange(roll.shape[1]) + args.window
    m, sd = roll.mean(0), roll.std(0)
    ax.plot(t, m, label=name); ax.fill_between(t, m - sd, m + sd, alpha=0.2)
ax.set_xlabel("Routing tasks executed on the same chip")
ax.set_ylabel(f"Success rate (rolling {args.window} tasks)")
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()
ax.set_title(f"Chip lifetime ({args.frac:.0%} degradable electrodes, {args.chips} chips)")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "exp3_lifetime.png"), dpi=200); plt.close(fig)

# ---- final health maps (chip 0)
fig, axes = plt.subplots(1, len(final), figsize=(10, 4.5))
for ax, (name, (h, ports)) in zip(axes, final.items()):
    im = ax.imshow(h, cmap="RdYlGn", vmin=0, vmax=1)
    for (y, x) in ports:
        ax.plot(x, y, "ks", ms=5)
    ax.set_title(f"{name}\nmean health {h.mean():.2f}")
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.8, label="Electrode health")
fig.suptitle(f"Electrode health after {args.tasks} tasks (squares = reservoirs / mixers)")
fig.savefig(os.path.join(args.out, "exp3_health.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("saved to", args.out)
