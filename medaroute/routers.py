"""
Classical baseline routers for the MEDA environment.

* astar        : plain A*; every move costs 1 (ignores electrode health)
* health_astar : health-aware A*; leaving a position costs 1 / p, where p is
                 the move-success probability there. Since failed moves are
                 retried, 1 / p is the expected number of actuation cycles
                 needed, so the planner minimises expected routing time.

* wear_astar   : wear-aware A* (see make_wear_astar); also charges each position
                 for the electrode wear the droplet will cause there, estimated from
                 the actuation counts and the observed health

All use the Chebyshev distance as heuristic, which is admissible for
8-directional moves with cost >= 1.
"""
from __future__ import annotations

import heapq
import itertools
import numpy as np

from .env import ACTIONS, MEDARoutingEnv


def _plan(env: MEDARoutingEnv, start, goal, health_aware: bool, p_floor: float = 1e-3, leave=None):
    """A* from start to goal. leave[u] (optional) overrides the cost of leaving u."""
    pmap = env.move_prob_map() if health_aware and leave is None else None
    h = lambda s: max(abs(s[0] - goal[0]), abs(s[1] - goal[1]))
    tie = itertools.count()
    open_ = [(h(start), next(tie), start)]
    g = {start: 0.0}
    parent = {start: (None, None)}
    closed = set()
    while open_:
        _, _, u = heapq.heappop(open_)
        if u == goal:
            break
        if u in closed:
            continue
        closed.add(u)
        if leave is not None:
            step_cost = leave[u]
        else:
            step_cost = 1.0 / max(pmap[u], p_floor) if health_aware else 1.0
        for a, (dy, dx) in enumerate(ACTIONS):
            v = (u[0] + dy, u[1] + dx)
            if not env.valid(v) or v in closed:
                continue
            nv = g[u] + step_cost
            if nv < g.get(v, np.inf):
                g[v] = nv
                parent[v] = (u, a)
                heapq.heappush(open_, (nv + h(v), next(tie), v))
    if goal not in parent:
        return None
    actions, s = [], goal
    while parent[s][0] is not None:
        s, a = parent[s]
        actions.append(a)
    return actions[::-1]


def astar(env, start, goal):
    return _plan(env, start, goal, health_aware=False)


def health_astar(env, start, goal):
    return _plan(env, start, goal, health_aware=True)


ROUTERS = {"A*": astar, "Health-aware A*": health_astar}


# ------------------------------------------------------------- wear-aware A*
def footprint_mean(env: MEDARoutingEnv, arr):
    """Mean of `arr` under the droplet for every droplet centre (NaN where invalid)."""
    k = 2 * env.r + 1
    c = np.pad(arr, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    box = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    out = np.full((env.H, env.W), np.nan)
    out[env.r:env.H - env.r, env.r:env.W - env.r] = box
    return out


def estimate_decay(env: MEDARoutingEnv, prior: float = 0.9):
    """Per-electrode decay factor estimated from what the controller observes.
    After k = actuations // usage_threshold wear events an electrode's health is
    decay^k, so decay = health^(1/k); electrodes not yet worn get the prior.
    (Assumes the chip started fresh, as in the lifetime benchmark.)"""
    k = env.actuations // env.usage_threshold
    est = np.full((env.H, env.W), prior)
    worn = k > 0
    est[worn] = np.power(np.clip(env.health[worn], 0.0, 1.0), 1.0 / k[worn])
    return est


def wear_cost_map(env: MEDARoutingEnv, lam: float, prior: float = 0.9, p_floor: float = 1e-3,
                  oracle: bool = False):
    """Cost of leaving each position: expected cycles there (1 / p) times
    (1 + lam * expected wear per cycle). Wear is the health an electrode loses
    at its next wear event, h * (1 - decay), averaged under the droplet and scaled
    so that a fresh electrode of unknown type counts 1. Electrodes known not to
    degrade (still at health 1 after a wear event) or already dead count 0, so the
    planner learns to route over them."""
    p = np.maximum(np.nan_to_num(env.move_prob_map(), nan=0.0), p_floor)
    decay = env.decay if oracle else estimate_decay(env, prior)   # oracle: true decay (analysis only)
    loss = env.health * (1.0 - decay) / (1.0 - prior)
    wear = np.nan_to_num(footprint_mean(env, loss), nan=0.0)
    return (1.0 + lam * wear) / p


def expected_cycles(env: MEDARoutingEnv, start, path, p_floor: float = 1e-3):
    """Expected actuation cycles to execute `path` from `start` (retrying failed moves)."""
    pm, pos, total = env.move_prob_map(), start, 0.0
    for a in path:
        total += 1.0 / max(pm[pos], p_floor)
        pos = (pos[0] + ACTIONS[a][0], pos[1] + ACTIONS[a][1])
    return total


def make_wear_astar(lam: float = 1.0, prior: float = 0.9, budget: float | None = None, oracle: bool = False):
    """Wear-aware A*: minimises expected time plus lam x expected electrode wear.
    lam = 0 is health-aware A*.

    With `budget` (0..1) the wear term may only spend that share of the slack the
    deadline leaves: with E0 = expected cycles of health-aware A*'s route and T the
    step budget, a route is accepted only if its expected cycles are at most
    E0 + budget * (T - E0). lam is halved until a route fits (lam = 0 always does),
    so the planner saves wear when there is time to spare and races when there is not."""
    def wear_astar(env, start, goal):
        if budget is None:
            return _plan(env, start, goal, health_aware=True, leave=wear_cost_map(env, lam, prior, oracle=oracle))
        fast = _plan(env, start, goal, health_aware=True)
        if fast is None:
            return None
        e0 = expected_cycles(env, start, fast)
        limit = e0 + budget * max(0.0, (env.max_steps - env.steps) - e0)
        l = lam
        while l > lam / 16:
            path = _plan(env, start, goal, health_aware=True, leave=wear_cost_map(env, l, prior, oracle=oracle))
            if path is not None and expected_cycles(env, start, path) <= limit:
                return path
            l /= 2
        return fast
    wear_astar.__name__ = (f"wear_astar_{lam:g}" + ("" if budget is None else f"_b{budget:g}")
                           + ("_oracle" if oracle else ""))
    return wear_astar


def run_router(env: MEDARoutingEnv, planner):
    """Plan once on the current chip state, then execute on the environment.
    A failed move is retried until it succeeds or the step budget runs out.
    Returns dict(success, steps, path_len)."""
    path = planner(env, env.pos, env.goal)
    if path is None:
        return {"success": False, "steps": 0, "path_len": 0}
    term = trunc = False
    for a in path:
        while True:
            _, _, term, trunc, info = env.step(a)
            if info["moved"] or term or trunc:
                break
        if term or trunc:
            break
    return {"success": bool(term), "steps": env.steps, "path_len": len(path)}


# ------------------------------------------------------------------ cost maps
def cost_to_go(env: MEDARoutingEnv, goal=None, health_aware: bool = True, p_floor: float = 1e-3):
    """Expected actuation cycles from every droplet centre to `goal` (default: env.goal),
    with the same costs as the planners above (1 / p to leave a position, or 1 for plain A*).
    Computed with Dijkstra backwards from the goal. Invalid or cut-off centres are inf.

    Moving to the neighbour with the lowest cost-to-go is health-aware A*'s policy
    (plain A*'s with health_aware=False)."""
    goal = env.goal if goal is None else tuple(goal)
    if health_aware:
        leave = 1.0 / np.maximum(np.nan_to_num(env.move_prob_map(), nan=0.0), p_floor)
    else:
        leave = np.ones((env.H, env.W))
    dist = np.full((env.H, env.W), np.inf)
    dist[goal] = 0.0
    heap = [(0.0, goal)]
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        for dy, dx in ACTIONS:                  # u --(dy, dx)--> v
            u = (v[0] - dy, v[1] - dx)
            if not env.valid(u):
                continue
            nd = d + leave[u]
            if nd < dist[u]:
                dist[u] = nd
                heapq.heappush(heap, (nd, u))
    return dist


def greedy_action(env: MEDARoutingEnv, dist, pos=None):
    """Action towards the valid neighbour with the lowest cost-to-go (None if no route)."""
    pos = env.pos if pos is None else pos
    best, best_a = np.inf, None
    for a, (dy, dx) in enumerate(ACTIONS):
        v = (pos[0] + dy, pos[1] + dx)
        if env.valid(v) and dist[v] < best:
            best, best_a = dist[v], a
    return best_a


# --------------------------------------------------- deadline-optimal oracle
def deadline_optimal(env: MEDARoutingEnv, budget: int):
    """Exact dynamic programme for the policy that maximises the probability of reaching
    the goal within `budget` steps, assuming electrode health stays as it is now.

    With one droplet a move from u succeeds with probability p(u) whatever its direction,
    so V_k(u) = max_a [ p(u) V_{k-1}(u + a) + (1 - p(u)) V_{k-1}(u) ], V_0 = 1 at the goal.
    Returns (V, policy): V[u] = success probability from u with the full budget, and
    policy[k - 1][u] = best action with k steps left. Wear during one task is ignored
    (no electrode reaches the usage threshold within a 1.5x deadline), so this is an
    upper bound on the deadline success of ANY router, learned or classical."""
    H, W = env.H, env.W
    p = np.nan_to_num(env.move_prob_map(), nan=0.0)
    valid = np.array([[env.valid((y, x)) for x in range(W)] for y in range(H)])
    ys, xs = np.mgrid[0:H, 0:W]
    nxt = []                                   # for each action: index of u + a (or u if invalid)
    for dy, dx in ACTIONS:
        ny, nx = ys + dy, xs + dx
        ok = valid & (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        ok[ok] = valid[ny[ok], nx[ok]]
        nxt.append((np.where(ok, ny, ys), np.where(ok, nx, xs), ok))
    V = np.zeros((H, W))
    V[env.goal] = 1.0
    policy = []
    for _ in range(budget):
        Q = np.stack([np.where(ok, p * V[ny, nx] + (1 - p) * V, V) for ny, nx, ok in nxt])
        policy.append(Q.argmax(0))
        V = np.where(valid, Q.max(0), 0.0)
        V[env.goal] = 1.0
    return V, policy


def run_deadline_optimal(env: MEDARoutingEnv):
    """Execute the deadline-optimal policy for the env's step budget (env.max_steps)."""
    _, policy = deadline_optimal(env, env.max_steps - env.steps)
    while True:
        a = policy[env.max_steps - env.steps - 1][env.pos]     # policy for the steps left
        _, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            return {"success": bool(term), "steps": env.steps}
