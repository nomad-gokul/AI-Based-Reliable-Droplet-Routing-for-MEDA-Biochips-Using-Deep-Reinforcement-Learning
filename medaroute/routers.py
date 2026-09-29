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
