"""
Experiment 3b - Why mean chip health hides the lifetime gain.

Re-runs Experiment 3 (same chips, seeds and task sequences) and records how
each router wears the chip: total actuations, how many electrodes it uses,
how many electrodes die, and the health of the electrodes it actually used.

Outputs (results/exp3_wear_analysis/):
  exp3_wear_analysis.csv   one row per chip and router
  exp3_wear_analysis.png   the key wear metrics side by side

  python experiments/exp3_wear_analysis.py            # 10 chips x 3000 tasks, ~3 min
"""
import argparse, copy, csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from medaroute.env import MEDARoutingEnv
from medaroute.routers import ROUTERS, run_router

p = argparse.ArgumentParser()
p.add_argument("--chips", type=int, default=10)
p.add_argument("--tasks", type=int, default=3000)
p.add_argument("--size", type=int, default=30)
p.add_argument("--frac", type=float, default=0.5)
p.add_argument("--ports", type=int, default=6)
p.add_argument("--slack", type=float, default=1.5)
p.add_argument("--dead", type=float, default=0.1, help="an electrode is dead below this health")
p.add_argument("--out", default=os.path.join(HERE, "..", "results", "exp3_wear_analysis"))
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

rows = []
for c in range(args.chips):
    base = MEDARoutingEnv(args.size, args.size, frac_degradable=args.frac,
                          task_mode="ports", n_ports=args.ports)
    base.reset(seed=500 + c, options={"new_chip": True})     # same seeds as Experiment 3
    degradable = base.decay < 1.0
    for name, planner in ROUTERS.items():
        env = copy.deepcopy(base)
        actuations = np.zeros((args.size, args.size), dtype=np.int64)
        wear = env._wear
        def counted_wear(pos, env=env, actuations=actuations, wear=wear):
            ys, xs = env._fp(pos); actuations[ys, xs] += 1; wear(pos)
        env._wear = counted_wear
        succ = np.zeros(args.tasks, bool); total_steps = 0
        for t in range(args.tasks):
            env.reset()
            env.max_steps = math.ceil(args.slack * env._cheb(env.pos, env.goal))
            succ[t] = run_router(env, planner)["success"]
            total_steps += env.steps
        h, used = env.health, actuations > 0
        rows.append({
            "chip": c, "router": name,
            "overall_success": succ.mean(),
            "tasks_before_first_failure": int(np.argmax(~succ)) if (~succ).any() else args.tasks,
            "total_steps": total_steps,
            "electrodes_used": int(used.sum()),
            "mean_health_all": h.mean(),
            "mean_health_degradable": h[degradable].mean(),
            "mean_health_used": h[used].mean(),
            "dead_electrodes": int((h < args.dead).sum()),
            "electrodes_below_0_5": int((h < 0.5).sum()),
        })
        print(f"chip {c}  {name:<16} success {succ.mean():.3f}  used {used.sum():>3}  "
              f"dead {rows[-1]['dead_electrodes']:>3}  mean health {h.mean():.3f}", flush=True)

with open(os.path.join(args.out, "exp3_wear_analysis.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

metrics = [("overall_success", "Overall success"),
           ("total_steps", "Total actuations"),
           ("electrodes_used", "Electrodes used"),
           ("dead_electrodes", f"Dead electrodes (health < {args.dead})"),
           ("mean_health_all", "Mean health, all electrodes"),
           ("mean_health_used", "Mean health, used electrodes")]
print("\nMean over chips")
fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6))
for ax, (k, label) in zip(axes, metrics):
    vals = [np.mean([r[k] for r in rows if r["router"] == n]) for n in ROUTERS]
    errs = [np.std([r[k] for r in rows if r["router"] == n]) for n in ROUTERS]
    bars = ax.bar(range(len(ROUTERS)), vals, yerr=errs, color=["tab:blue", "tab:orange"], capsize=3)
    ax.bar_label(bars, fmt="%.2f" if max(vals) < 2 else "%.0f", fontsize=8)
    ax.set_xticks(range(len(ROUTERS)), list(ROUTERS), fontsize=8)
    ax.set_title(label, fontsize=9); ax.grid(axis="y", alpha=0.3)
    print(f"  {label:<36}" + "".join(f"{n}: {v:9.3f}   " for n, v in zip(ROUTERS, vals)))
fig.suptitle(f"How each router wears the chip ({args.chips} chips x {args.tasks} tasks, mean +- std)")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "exp3_wear_analysis.png"), dpi=200); plt.close(fig)
print("saved to", args.out)
