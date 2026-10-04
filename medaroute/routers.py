"""
Classical baseline routers for the MEDA environment.

* astar        : plain A*; every move costs 1 (ignores electrode health)
* health_astar : health-aware A*; leaving a position costs 1 / p, where p is
                 the move-success probability there. Since failed moves are
                 retried, 1 / p is the expected number of actuation cycles
                 needed, so the planner minimises expected routing time.

Both use the Chebyshev distance as heuristic, which is admissible for
8-directional moves with cost >= 1.
"""
from __future__ import annotations

import heapq
import itertools
import numpy as np

from .env import ACTIONS, MEDARoutingEnv


def _plan(env: MEDARoutingEnv, start, goal, health_aware: bool, p_floor: float = 1e-3):
    pmap = env.move_prob_map() if health_aware else None
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
