"""
Experiment 7 - Improving the health-aware PPO agent.

Each run trains one PPO agent (one variant, one seed) and tests it on a fixed set of unseen
chips. Variants switch on the planned improvements one at a time, so their effect can be
measured (see VARIANTS below):

  slack 1.5      train with the same 1.5x deadline used at test time (Experiment 4 used 3x)
  "time"         observation channel: fraction of the step budget left
  "prob"         observation channel: move-success probability at every position
  "cost"         observation channel: health-aware A*'s cost-to-go map
  penalty        reward term: -penalty * (1 - move-success probability) every step
  shaping        potential-based reward shaping with health-aware cost-to-go
  bc             warm start by imitating health-aware A* (plain A* for no-health agents)
  curriculum     start on easy chips, reach the target difficulty halfway through training

Settings (--setting):
  degraded : Experiment 4's task. 12x12 chips, 50% degradable, pre-aged; the same 500 test chips
  blockage : Experiment 6's soft, worn 12x12 chip; trained on 0-90% blockage, tested at 10...90%
             (200 chips per level, the same chips as exp6), so health vs no-health can be
             compared at each blockage level

Every test also runs A*, health-aware A* and the deadline-optimal oracle (an exact dynamic
programme, the best success rate any router can reach) on the same chips.

  python experiments/exp7_health_ppo.py --variant base --timesteps 20000 --eval_episodes 50  # smoke test
  python experiments/exp7_health_ppo.py --variant prob --seed 0                               # one run
  python experiments/exp7_health_ppo.py --report                                              # tables and figures

Outputs: results/exp7/runs/<setting>_<variant>_s<seed>.json (+ model .zip and training log,
not committed), and with --report results/exp7/exp7_<setting>.csv and figures.
"""
import argparse, copy, csv, glob, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from medaroute.env import MEDARoutingEnv
from medaroute.blockage import BlockageEnv
from medaroute.routers import ROUTERS, run_router, run_deadline_optimal, deadline_optimal, astar

PPO_KW = dict(n_steps=256, batch_size=512, n_epochs=4, learning_rate=3e-4, gamma=0.99, ent_coef=0.01)
T = ("time",)
TP = ("time", "prob")
TPC = ("time", "prob", "cost")
VARIANTS = {
    # Experiment 4's recipe (3x training step limit, no extra channels) as the reference
    "base":             dict(health=True, slack=3.0),
    "base-nohealth":    dict(health=False, slack=3.0),
    # one change at a time
    "penalty":          dict(health=True, slack=3.0, penalty=0.1),
    "slack":            dict(health=True, slack=1.5),
    "slack-nohealth":   dict(health=False, slack=1.5),
    "time":             dict(health=True, slack=1.5, features=T),
    "time-nohealth":    dict(health=False, slack=1.5, features=T),
    "prob":             dict(health=True, slack=1.5, features=TP),
    "cost":             dict(health=True, slack=1.5, features=TPC),
    "shaping":          dict(health=True, slack=1.5, features=TP, shaping=0.05),
    "curriculum":       dict(health=True, slack=1.5, features=TP, curriculum=True),
    "bc":               dict(health=True, slack=1.5, features=TP, bc=True),
    "cost-bc":          dict(health=True, slack=1.5, features=TPC, bc=True),
    "bc-nohealth":      dict(health=False, slack=1.5, features=T, bc=True),
}
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

p = argparse.ArgumentParser()
p.add_argument("--setting", choices=["degraded", "blockage"], default="degraded")
p.add_argument("--variant", choices=list(VARIANTS), default="prob")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--timesteps", type=int, default=500_000)
p.add_argument("--n_envs", type=int, default=8)
p.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 lets several runs share a CPU)")
p.add_argument("--bc_episodes", type=int, default=3000)
p.add_argument("--eval_episodes", type=int, default=None,
               help="test chips (default 500 for degraded, 200 per level for blockage)")
p.add_argument("--eval_only", action="store_true", help="re-test a saved model")
p.add_argument("--report", action="store_true", help="collect all runs into tables and figures")
p.add_argument("--out", default=os.path.join(HERE, "..", "results", "exp7"))
args = p.parse_args()
RUNS = os.path.join(args.out, "runs")
os.makedirs(RUNS, exist_ok=True)
SIZE, SLACK_TEST = 12, 1.5


# ---------------------------------------------------------------- chips
def train_chip(health):
    if args.setting == "degraded":
        return MEDARoutingEnv(SIZE, SIZE, frac_degradable=0.5, pre_age_max=15, observe_health=health)
    return BlockageEnv((0.0, 0.9), "soft", worn=True, height=SIZE, width=SIZE, droplet_radius=1,
                       observe_health=health, min_task_distance=max(3, SIZE // 6))


def curriculum(e, d):
    """d = 0: easiest chips, d = 1: the test distribution."""
    if args.setting == "degraded":
        e.pre_age_max = round(15 * d)
    else:
        e.blockage = (0.0, 0.1 + 0.8 * d)


def test_chips():
    """(level, env) pairs: reset, deadline set, identical for every method and run."""
    if args.setting == "degraded":
        n = args.eval_episodes or 500
        for i in range(n):           # Experiment 4's test chips
            e = MEDARoutingEnv(SIZE, SIZE, frac_degradable=0.5, pre_age_max=15)
            e.reset(seed=999_000 + i, options={"new_chip": True})
            e.max_steps = math.ceil(SLACK_TEST * e._cheb(e.pos, e.goal))
            yield None, e
        return
    n = args.eval_episodes or 200
    for li, b in enumerate(LEVELS):  # Experiment 6's test chips (soft, worn, 12x12)
        for s in range(n):
            e = BlockageEnv(b, "soft", worn=True, height=SIZE, width=SIZE, droplet_radius=1,
                            observe_health=False, min_task_distance=max(3, SIZE // 6))
            e.reset(seed=100_000 * li + s, options={"new_chip": True})
            e.max_steps = math.ceil(SLACK_TEST * max(1, len(astar(e, e.pos, e.goal))))
            yield b, e


def view(e, health):
    """The same chip and task, observed with or without the health channel."""
    import gymnasium as gym
    e = copy.deepcopy(e)
    e.observe_health = health
    e.observation_space = gym.spaces.Box(0.0, 1.0, (4 if health else 3, e.H, e.W), np.float32)
    return e


def summarise(results):
    """results: list of (level, success, steps) -> {level: {...}} (level None = all)."""
    out = {}
    for lv in sorted({r[0] for r in results}, key=lambda x: -1 if x is None else x):
        s = np.array([r[1] for r in results if r[0] == lv])
        out["all" if lv is None else f"{lv:.1f}"] = {"success": float(s.mean()), "n": int(len(s))}
    return out


# ------------------------------------------------------------- baselines
def baselines():
    path = os.path.join(args.out, f"baselines_{args.setting}_{args.eval_episodes or 'default'}.json")
    if os.path.exists(path):
        return json.load(open(path))
    res = {"A*": [], "Health-aware A*": [], "Deadline-optimal (oracle)": []}
    oracle_p = []
    for lv, e in test_chips():
        for m, planner in ROUTERS.items():
            res[m].append((lv, run_router(copy.deepcopy(e), planner)["success"], 0))
        res["Deadline-optimal (oracle)"].append((lv, run_deadline_optimal(copy.deepcopy(e))["success"], 0))
        V, _ = deadline_optimal(e, e.max_steps)
        oracle_p.append((lv, V[e.pos], 0))
    out = {m: summarise(r) for m, r in res.items()}
    out["Oracle success probability"] = summarise(oracle_p)
    json.dump(out, open(path, "w"), indent=1)
    return out


# ------------------------------------------------------------------- run
def train_and_test():
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from medaroute.rl import (SmallCNN, RoutingTask, CurriculumCallback, collect_expert,
                              pretrain_bc, run_agent)
    torch.set_num_threads(args.threads)
    cfg = dict(VARIANTS[args.variant])
    feats = tuple(cfg.get("features", ()))
    name = f"{args.setting}_{args.variant}_s{args.seed}"
    model_path = os.path.join(RUNS, name + ".zip")

    def make(rank, log=None):
        def f():
            base = train_chip(cfg["health"])
            base.degrade_penalty = cfg.get("penalty", 0.0)
            e = RoutingTask(Monitor(base, filename=log, info_keywords=("reached",)), slack=cfg["slack"],
                            features=feats, shaping=cfg.get("shaping", 0.0), gamma=PPO_KW["gamma"],
                            curriculum=curriculum if cfg.get("curriculum") else None)
            e.reset(seed=args.seed * 1000 + rank)
            return e
        return f

    def test(model):
        return summarise([(lv, run_agent(view(e, cfg["health"]), model, feats)["success"], 0)
                          for lv, e in test_chips()])

    t0, bc_test = time.time(), None
    if args.eval_only:
        model = PPO.load(model_path, device="cpu")
    else:
        venv = DummyVecEnv([make(r, os.path.join(RUNS, name) if r == 0 else None) for r in range(args.n_envs)])
        model = PPO("MlpPolicy", venv, device="cpu", seed=args.seed, verbose=0, **PPO_KW,
                    policy_kwargs=dict(features_extractor_class=SmallCNN,
                                       features_extractor_kwargs=dict(features_dim=256),
                                       net_arch=dict(pi=[128], vf=[128])))
        if cfg.get("bc"):
            expert = "health-aware A*" if cfg["health"] else "A*"
            print(f"collecting {args.bc_episodes} {expert} episodes for behaviour cloning ...", flush=True)
            data = collect_expert(make(10_000)(), args.bc_episodes, health_aware=cfg["health"],
                                  gamma=PPO_KW["gamma"])
            pretrain_bc(model, *data, seed=args.seed)
            bc_test = test(model)                     # the imitation policy before any RL
            print("after behaviour cloning:", json.dumps(bc_test), flush=True)
        cb = CurriculumCallback(args.timesteps) if cfg.get("curriculum") else None
        print(f"training {name} for {args.timesteps} steps ...", flush=True)
        model.learn(total_timesteps=args.timesteps, callback=cb)
        model.save(model_path)
    train_min = (time.time() - t0) / 60

    out = {"setting": args.setting, "variant": args.variant, "seed": args.seed,
           "timesteps": args.timesteps, "config": {k: list(v) if isinstance(v, tuple) else v
                                                   for k, v in cfg.items()},
           "train_minutes": round(train_min, 1), "test": test(model)}
    if bc_test:
        out["test_bc_only"] = bc_test
    json.dump(out, open(os.path.join(RUNS, name + ".json"), "w"), indent=1)
    print(json.dumps(out["test"]))


# ---------------------------------------------------------------- report
def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = baselines()
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RUNS, f"{args.setting}_*.json")))]
    groups = {}
    for r in runs:
        groups.setdefault((r["variant"], r["timesteps"]), []).append(r)
    order = {v: i for i, v in enumerate(VARIANTS)}
    keys = sorted(groups, key=lambda k: (k[1], order.get(k[0], 99)))
    levels = [k for k in base["A*"] if k != "all"] or ["all"]
    rows = []
    for m in ["A*", "Health-aware A*", "Deadline-optimal (oracle)"]:
        rows.append({"method": m, "timesteps": "", "seeds": "", "health_map": "",
                     **{f"success_{lv}": round(base[m][lv]["success"], 3) for lv in levels},
                     **({"success_mean": round(np.mean([base[m][lv]["success"] for lv in levels]), 3)}
                        if len(levels) > 1 else {}), "std_over_seeds": ""})
    for v, ts in keys:
        g = groups[(v, ts)]
        per_seed = np.array([[r["test"][lv]["success"] for lv in levels] for r in g])
        rows.append({"method": f"PPO [{v}]", "timesteps": ts, "seeds": len(g),
                     "health_map": g[0]["config"]["health"],
                     **{f"success_{lv}": round(per_seed[:, i].mean(), 3) for i, lv in enumerate(levels)},
                     **({"success_mean": round(per_seed.mean(), 3)} if len(levels) > 1 else {}),
                     "std_over_seeds": round(per_seed.mean(1).std(), 3) if len(g) > 1 else ""})
    csv_path = os.path.join(args.out, f"exp7_{args.setting}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)

    if args.setting == "degraded":
        fig, ax = plt.subplots(figsize=(10, 4.5))
        labels = [r["method"] + (f"\n{r['timesteps'] // 1000}k x{r['seeds']}" if r["timesteps"] else "")
                  for r in rows]
        vals = [r["success_all"] for r in rows]
        err = [r["std_over_seeds"] or 0 for r in rows]
        cols = ["tab:blue", "tab:orange", "black"] + [
            "tab:green" if r["health_map"] else "tab:gray" for r in rows[3:]]
        bars = ax.bar(range(len(rows)), vals, yerr=err, color=cols, capsize=3)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.axhline(base["Deadline-optimal (oracle)"]["all"]["success"], color="black", ls="--", lw=0.8)
        ax.set_xticks(range(len(rows)), labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(f"Test success (deadline {SLACK_TEST}x)")
        lo = min(vals) - 0.1
        ax.set_ylim(max(0, lo), 1.0)
        ax.set_title("12x12 chips, 50% degradable, 500 unseen test chips "
                     "(green = sees health map, grey = does not)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "exp7_degraded.png"), dpi=200); plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = [float(lv) * 100 for lv in levels]
        for m, st in [("A*", "o-"), ("Health-aware A*", "s-"), ("Deadline-optimal (oracle)", "k--")]:
            ax.plot(x, [base[m][lv]["success"] for lv in levels], st, label=m, lw=1.2)
        for r in rows[3:]:
            ax.plot(x, [r[f"success_{lv}"] for lv in levels], "^-" if r["health_map"] else "v:",
                    label=f"{r['method']} ({r['seeds']} seed{'s' if r['seeds'] > 1 else ''})")
        ax.set_xlabel("Blocked electrodes (% of chip)"); ax.set_ylabel(f"Success rate (deadline {SLACK_TEST}x)")
        ax.set_xticks(x); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_title("12x12 soft blockage + worn good electrodes, 200 test chips per level", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "exp7_blockage.png"), dpi=200); plt.close(fig)

    # learning curves (training episode success, env 0) of the first seed of each variant
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for v, ts in keys:
        log = os.path.join(RUNS, f"{args.setting}_{v}_s{groups[(v, ts)][0]['seed']}.monitor.csv")
        if not os.path.exists(log):
            continue
        with open(log) as fh:
            fh.readline()
            recs = list(csv.DictReader(fh))
        s = np.array([r["reached"] == "True" for r in recs], float)
        lens = np.array([float(r["l"]) for r in recs])
        w = max(1, min(500, len(s) // 10))
        ax.plot(np.cumsum(lens)[w - 1:] * args.n_envs, np.convolve(s, np.ones(w) / w, "valid"),
                label=f"{v} ({VARIANTS[v]['slack']}x limit)", lw=1)
    ax.set_xlabel("Training steps (approx.)"); ax.set_ylabel("Training episode success (rolling)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"exp7_{args.setting}_learning.png"), dpi=200)
    plt.close(fig)
    print("saved to", args.out)


if args.report:
    report()
else:
    train_and_test()
