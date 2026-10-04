"""
Experiment 8 - Chip-lifetime benchmark: can a router keep the chip usable for longer?

A fresh chip runs 3,000 tasks between 6 fixed sites (Experiment 3's chips and task
sequences) and wears out as it is used. Parts (--part):

  routers : A*, health-aware A*, wear-aware A* (charges routes for the wear they cause,
            estimated from actuation counts and observed health) and, for analysis only,
            wear-aware A* that knows the true decay factors (an optimistic bound for
            wear-aware routing). Records success and how the chip wears over time.
  sites   : the same chips with the electrodes within r cells of the fixed sites made
            wear-free (r = 0, 1, 2, 4): how much of the lost lifetime comes from the
            sites, which every task must start and end on?
  agents  : the trained PPO agents of Experiment 7 on a 12x12 lifetime benchmark with 4 sites (needs
            their models in results/exp7/runs/, which are not committed).

Wear-aware A*'s settings (lam = 1, budget 0.3) were chosen on separate validation chips
(seeds 900-903), not on these test chips.

  python experiments/exp8_lifetime.py --part routers      # ~4 min on 4 cores
  python experiments/exp8_lifetime.py --part sites        # ~4 min
  python experiments/exp8_lifetime.py --part agents       # ~5 min
Outputs: results/exp8/
"""
import argparse, csv, json, os, sys
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from medaroute.lifetime import lifetime_chip, run_lifetime, planner_route, agent_route, summarise
from medaroute.routers import astar, health_astar, make_wear_astar

p = argparse.ArgumentParser()
p.add_argument("--part", choices=["routers", "sites", "agents"], default="routers")
p.add_argument("--chips", type=int, default=10)
p.add_argument("--tasks", type=int, default=None, help="default 3000 (30x30) or 1000 (agents, 12x12)")
p.add_argument("--slack", type=float, default=1.5)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--window", type=int, default=100)
p.add_argument("--out", default=os.path.join(HERE, "..", "results", "exp8"))
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
EXP7 = os.path.join(HERE, "..", "results", "exp7", "runs")

PLANNERS = {
    "A*": astar,
    "Health-aware A*": health_astar,
    "Wear-aware A*": make_wear_astar(1.0, budget=0.3),
    "Wear-aware A* (true decay, analysis)": make_wear_astar(2.0, budget=0.3, oracle=True),
}
AGENTS = {   # Experiment 7 models: (file, observation channels, sees health)
    "PPO, Exp. 4 recipe": ("degraded_base_s0.zip", (), True),
    "Imitation + fine-tuning, health map (seed 0)": ("degraded_bc-ft_s0.zip", ("time", "prob"), True),
    "Imitation + fine-tuning, health map (seed 1)": ("degraded_bc-ft_s1.zip", ("time", "prob"), True),
    "Imitation + fine-tuning, no health map (seed 0)": ("degraded_bc-ft-nohealth_s0.zip", ("time",), False),
    "Imitation + fine-tuning, no health map (seed 1)": ("degraded_bc-ft-nohealth_s1.zip", ("time",), False),
}


def chip(seed, size, protect=None):
    # 6 sites at least 6 cells apart do not fit on a 12x12 chip: use 4 there
    env = lifetime_chip(seed, size=size, ports=6 if size >= 20 else 4)
    if protect:      # electrodes within `protect` cells of a site never wear
        for (y, x) in env.ports:
            env.decay[max(0, y - protect):y + protect + 1, max(0, x - protect):x + protect + 1] = 1.0
    return env


def job(a):
    name, seed, size, protect, tasks = a
    if name in AGENTS:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        f, feats, health = AGENTS[name]
        route = agent_route(PPO.load(os.path.join(EXP7, f), device="cpu"), feats, health)
    else:
        route = planner_route(PLANNERS[name])
    r = run_lifetime(chip(seed, size, protect), route, tasks, slack=args.slack)
    return {"router": name, "chip": seed - 500, "protect": protect or 0, **summarise(r, window=args.window),
            "success_seq": r["success"].tolist(), "snapshots": r["snapshots"]}


def run(jobs):
    with Pool(args.workers) as pool:
        return pool.map(job, jobs)


def save(rows, tag):
    keys = [k for k in rows[0] if k not in ("success_seq", "snapshots")]
    with open(os.path.join(args.out, f"exp8_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in rows])
    groups = {}
    for r in rows:
        groups.setdefault((r["router"], r["protect"]), []).append(r)
    summ = []
    for (name, prot), g in groups.items():
        summ.append({"router": name, "protect": prot, "chips": len(g),
                     **{k: round(float(np.mean([r[k] for r in g])), 3) for k in keys
                        if k not in ("router", "chip", "protect")},
                     "overall_success_std": round(float(np.std([r["overall_success"] for r in g])), 3)})
        print(summ[-1])
    with open(os.path.join(args.out, f"exp8_{tag}_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summ[0].keys()); w.writeheader(); w.writerows(summ)
    return groups


def curves(groups, names, tag, title, wear=True):
    fig, axes = plt.subplots(1, 2 if wear else 1, figsize=(13 if wear else 7, 4.2), squeeze=False)
    kern = np.ones(args.window) / args.window
    for name in names:
        g = groups[(name, 0)]
        roll = np.array([np.convolve(r["success_seq"], kern, mode="valid") for r in g])
        t = np.arange(roll.shape[1]) + args.window
        ls = ":" if "analysis" in name or "no health" in name else "-"
        line, = axes[0, 0].plot(t, roll.mean(0), ls=ls, label=name)
        axes[0, 0].fill_between(t, roll.mean(0) - roll.std(0), roll.mean(0) + roll.std(0),
                                color=line.get_color(), alpha=0.12)
        if wear:
            tt = [s["task"] for s in g[0]["snapshots"]]
            dead = np.array([[s["dead_frac"] for s in r["snapshots"]] for r in g]) * 100
            axes[0, 1].plot(tt, dead.mean(0), ls=ls, color=line.get_color(), label=name)
    axes[0, 0].set_xlabel("Tasks executed on the same chip")
    axes[0, 0].set_ylabel(f"Success rate (rolling {args.window} tasks, deadline {args.slack}x)")
    axes[0, 0].set_ylim(0, 1.05); axes[0, 0].grid(alpha=0.3); axes[0, 0].legend(fontsize=7)
    if wear:
        axes[0, 1].set_xlabel("Tasks executed on the same chip")
        axes[0, 1].set_ylabel("Blocked electrodes (% with health < 0.1)")
        axes[0, 1].grid(alpha=0.3); axes[0, 1].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"exp8_{tag}.png"), dpi=200); plt.close(fig)


if args.part == "routers":
    tasks = args.tasks or 3000
    rows = run([(n, 500 + c, 30, None, tasks) for n in PLANNERS for c in range(args.chips)])
    groups = save(rows, "routers")
    curves(groups, list(PLANNERS), "routers",
           f"30x30 chips, 50% degradable, 6 fixed sites, {args.chips} chips x {tasks} tasks (mean, band = std)")

elif args.part == "sites":
    tasks = args.tasks or 3000
    radii = [0, 1, 2, 4]
    names = ["A*", "Health-aware A*"]
    rows = run([(n, 500 + c, 30, r, tasks) for r in radii for n in names for c in range(args.chips)])
    groups = save(rows, "sites")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, col in zip(names, ["tab:blue", "tab:orange"]):
        for ax, k, lab in [(axes[0], "overall_success", "Overall success over 3,000 tasks"),
                           (axes[1], "life_70", "Tasks until rolling success < 70%")]:
            m = [np.mean([r[k] for r in groups[(name, rr)]]) for rr in radii]
            ax.plot(radii, m, "o-", color=col, label=name)
            ax.set_xlabel("Electrodes made wear-free: within r cells of each site")
            ax.set_ylabel(lab); ax.set_xticks(radii); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylim(0, 1.02)
    fig.suptitle("How much of the chip's lifetime is lost at the fixed sites "
                 "(r = 0: normal chip; r = 1: the sites' own electrodes)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "exp8_sites.png"), dpi=200); plt.close(fig)

else:
    tasks = args.tasks or 1000
    names = ["A*", "Health-aware A*", "Wear-aware A*"] + list(AGENTS)
    rows = run([(n, 500 + c, 12, None, tasks) for n in names for c in range(args.chips)])
    groups = save(rows, "agents")
    curves(groups, names, "agents", f"12x12 chips, 4 fixed sites, {args.chips} chips x {tasks} tasks "
           "(Experiment 7 agents, trained on single tasks)", wear=False)
print("saved to", args.out)
