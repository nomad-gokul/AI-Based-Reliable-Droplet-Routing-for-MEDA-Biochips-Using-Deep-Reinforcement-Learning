"""
Experiment 1 - Environment demonstration.

Shows a partly worn MEDA chip (electrode health heatmap) with one routing
task, and the paths chosen by plain A* and health-aware A*.

Output: results/exp1_paths.png
"""
import argparse, copy, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from medaroute.env import MEDARoutingEnv, ACTIONS
from medaroute.routers import ROUTERS

p = argparse.ArgumentParser()
p.add_argument("--size", type=int, default=30)
p.add_argument("--frac", type=float, default=0.5)
p.add_argument("--pre_age", type=int, default=15)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--out", default="results")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

env = MEDARoutingEnv(args.size, args.size, frac_degradable=args.frac, pre_age_max=args.pre_age)
env.reset(seed=args.seed, options={"new_chip": True,
                                   "start": (3, args.size // 2), "goal": (args.size - 4, args.size // 2)})

fig, ax = plt.subplots(figsize=(6.5, 6))
im = ax.imshow(env.health, cmap="RdYlGn", vmin=0, vmax=1)
colors = {"A*": "tab:blue", "Health-aware A*": "black"}
styles = {"A*": "--", "Health-aware A*": "-"}
for name, planner in ROUTERS.items():
    path = planner(copy.deepcopy(env), env.pos, env.goal)
    pts = [env.pos]
    for a in path:
        dy, dx = ACTIONS[a]
        pts.append((pts[-1][0] + dy, pts[-1][1] + dx))
    pts = np.array(pts)
    pmap = env.move_prob_map()
    exp_cycles = sum(1 / max(pmap[tuple(q)], 1e-3) for q in pts[:-1])
    ax.plot(pts[:, 1], pts[:, 0], styles[name], color=colors[name], lw=2.2,
            label=f"{name}: {len(path)} moves, ~{exp_cycles:.0f} expected cycles")
r = env.r
for pos, col, lab in [(env.pos, "tab:blue", "start"), (env.goal, "tab:purple", "goal")]:
    ax.add_patch(Rectangle((pos[1] - r - .5, pos[0] - r - .5), 2 * r + 1, 2 * r + 1,
                           fill=False, ec=col, lw=2.5))
    ax.text(pos[1], pos[0] - r - 1.2, lab, ha="center", fontsize=9, color=col, weight="bold")
fig.colorbar(im, ax=ax, shrink=0.8, label="Electrode health (1 = healthy)")
ax.set_title(f"{args.size}x{args.size} MEDA chip, {args.frac:.0%} degradable electrodes")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=8.5)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(args.out, "exp1_paths.png"), dpi=200); plt.close(fig)
print("saved", os.path.join(args.out, "exp1_paths.png"))
