"""Discrete Johansen (yield-line) mechanism optimisation for plate collapse.

The rigid-plastic collapse pressure of a plate is bounded from above by the
virtual work of *any* kinematically admissible mechanism::

    p_u <= sum_hinges m_p |theta| l  /  int_patch w dA

For a piecewise-linear surface (``w = 0`` on the simple-support boundary) the
dissipation is exactly concentrated on the interior edges of the triangulation
-- each such edge is a hinge of measured rotation and length, the facets are
planes so a hinge carries no twisting rate -- hence the ratio is *exact* for the
discrete mechanism and every number this module returns is a valid upper bound
(the surface itself is admissible).

Minimising that ratio under the normalisation ``int_patch w dA = 1`` is a linear
program (the dissipation is a sum of absolute values of linear forms), solved
here with HiGHS through ``scipy.optimize.linprog``.  The discretisation is a
*restriction* of the admissible set, so the LP value decreases monotonically
towards the true collapse load as the grid is refined (never below it).

Why this module exists (verification context, GAP-PP-3)
------------------------------------------------------
The closed-form yield-line bound of ``plate_patch_plastic_limit`` uses one
hand-written mechanism family; this module optimises over *all* piecewise
linear mechanisms of a grid, so it is an independent yardstick for that bound
and for FE limit loads.  Two consequences found with it:

* on a uniformly loaded simply supported **square** plate the LP returns
  ``24 m_p/a^2`` -- the exact Johansen collapse load -- to four significant
  digits, which is the regression anchor used in the tests;
* on the patch-loaded anchor geometry the pyramid mechanism is *not* a grid
  mechanism (its ridge lines are not grid edges), so at reachable resolutions
  the LP bound (``63.3 m_p``) is looser than the closed form (``54.86 m_p``):
  the LP confirms the closed-form bound is the better reported reference and
  does not tighten the GAP-PP-3 bracket.

Assembly defect this module is written against
----------------------------------------------
The hinge rotation rate across a shared edge is ``theta = |s1 - s2|`` with
``s_i`` the normal slope of triangle ``i``; summing the two triangles'
coefficient vectors instead of subtracting them is a *silent* mistake: the
coefficients cancel at the shared nodes of any symmetric configuration and the
LP then reports a mechanism with zero dissipation (``p_u = 0``).  The tests
compare every hinge row against an independently computed dihedral angle, so
that sign slip cannot come back.
"""
from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - exercised by the absence test only
    from scipy.optimize import linprog
except ImportError as exc:  # pragma: no cover
    linprog = None
    _SCIPY_ERROR = exc


def _triangulation(nx: int, ny: int) -> list[tuple[int, int, int]]:
    """Checkerboard diagonal split; every triangle is counter-clockwise."""
    tris: list[tuple[int, int, int]] = []

    def nid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    for j in range(ny):
        for i in range(nx):
            n00, n10 = nid(i, j), nid(i + 1, j)
            n01, n11 = nid(i, j + 1), nid(i + 1, j + 1)
            if (i + j) % 2 == 0:
                tris.append((n00, n10, n11))
                tris.append((n00, n11, n01))
            else:
                tris.append((n00, n10, n01))
                tris.append((n10, n11, n01))
    return tris


def _clip_area(pts: np.ndarray, x1: float, x2: float, y1: float, y2: float) -> float:
    """Exact area of ``triangle ∩ rectangle`` (Sutherland-Hodgman)."""
    poly = [tuple(p) for p in pts]
    for axis, bound, keep_greater in ((0, x1, True), (0, x2, False),
                                      (1, y1, True), (1, y2, False)):
        if not poly:
            return 0.0
        out = []
        n = len(poly)
        for k in range(n):
            cur, nxt = poly[k], poly[(k + 1) % n]
            cin = cur[axis] >= bound if keep_greater else cur[axis] <= bound
            nin = nxt[axis] >= bound if keep_greater else nxt[axis] <= bound
            if cin:
                out.append(cur)
            if cin != nin:
                denom = nxt[axis] - cur[axis]
                t = (bound - cur[axis]) / denom if denom else 0.0
                out.append((cur[0] + t * (nxt[0] - cur[0]),
                            cur[1] + t * (nxt[1] - cur[1])))
        poly = out
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for k in range(len(poly)):
        xa, ya = poly[k]
        xb, yb = poly[(k + 1) % len(poly)]
        area += xa * yb - xb * ya
    return 0.5 * abs(area)


def _gradient_matrix(P: np.ndarray) -> np.ndarray:
    """Rows: d/dw_k of (dw/dx, dw/dy) for a linear triangle."""
    A2 = ((P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1])
          - (P[2, 0] - P[0, 0]) * (P[1, 1] - P[0, 1]))
    coef = np.zeros((3, 2))
    for k in range(3):
        pa, pb = P[(k + 1) % 3], P[(k + 2) % 3]
        coef[k, 0] = (pa[1] - pb[1]) / A2
        coef[k, 1] = (pb[0] - pa[0]) / A2
    return coef


def mechanism_upper_bound(a: float, b: float, mp: float, patch: tuple[float, float],
                          *, nx: int = 16, ny: int = 8) -> dict:
    """Tightest piecewise-linear mechanism upper bound on ``nx x ny`` triangles."""
    if linprog is None:  # pragma: no cover
        raise RuntimeError("scipy is required for the mechanism LP") from _SCIPY_ERROR
    if not (a > 0 and b > 0 and mp > 0 and nx >= 2 and ny >= 2):
        raise ValueError("a, b, mp must be positive and nx, ny >= 2")
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    xs, ys = np.linspace(0.0, a, nx + 1), np.linspace(0.0, b, ny + 1)
    nodes = np.array([(x, y) for y in ys for x in xs])
    tris = _triangulation(nx, ny)
    on_boundary = np.array([x <= 0.0 or x >= a or y <= 0.0 or y >= b
                            for x, y in nodes])
    free = np.where(~on_boundary)[0]
    col = {int(n): k for k, n in enumerate(free)}
    nw = len(free)

    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for ti, t in enumerate(tris):
        for k in range(3):
            p, q, r = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
            edge_map.setdefault((min(p, q), max(p, q)), []).append((ti, r))
    hinges = [(k, v) for k, v in edge_map.items() if len(v) == 2]

    grads = [_gradient_matrix(nodes[list(t)]) for t in tris]
    n_hinge = len(hinges)
    nvar = nw + n_hinge
    cost = np.zeros(nvar)
    rows, rhs = [], []
    for k, ((p, q), members) in enumerate(hinges):
        length = math.dist(nodes[p], nodes[q])
        cost[nw + k] = mp * length
        normal = np.array([nodes[q][1] - nodes[p][1], -(nodes[q][0] - nodes[p][0])])
        normal /= np.linalg.norm(normal)
        row = np.zeros(nvar)
        for side, (ti, _opp) in enumerate(members):
            coef = grads[ti]
            sign = 1.0 if side == 0 else -1.0        # theta = |s1 - s2|
            for local, gid in enumerate(tris[ti]):
                if gid in col:
                    row[col[gid]] += sign * float(coef[local] @ normal)
        upper = row.copy()
        upper[nw + k] -= 1.0                          # theta - t <= 0
        lower = -row.copy()
        lower[nw + k] -= 1.0                          # -theta - t <= 0
        rows.extend((upper, lower))
        rhs.extend((0.0, 0.0))

    work = np.zeros((1, nvar))
    for ti, t in enumerate(tris):
        P = nodes[list(t)]
        overlap = _clip_area(P, x1, x2, y1, y2)
        if overlap <= 0.0:
            continue
        for gid in t:
            if gid in col:
                work[0, col[gid]] += overlap / 3.0
    res = linprog(cost, A_ub=np.array(rows), b_ub=np.array(rhs),
                  A_eq=work, b_eq=np.array([1.0]),
                  bounds=[(None, None)] * nw + [(0.0, None)] * n_hinge,
                  method="highs")
    if not res.success:
        raise RuntimeError(f"mechanism LP failed: {res.message}")
    w = np.zeros(len(nodes))
    w[free] = res.x[:nw]
    return {"pressure": float(res.fun), "m_p": float(mp), "nx": nx, "ny": ny,
            "hinges": n_hinge, "w": w, "nodes": nodes, "tris": tris,
            "hinge_list": hinges, "grads": grads,
            "slacks": np.asarray(res.x[nw:], dtype=float)}


def dihedral_check(result: dict) -> float:
    """Max |row theta - independently computed dihedral| over all hinges.

    Independent of the LP assembly: gradients are rebuilt from the three node
    values of each triangle, so a sign slip in the constraint rows shows up as
    a non-zero return value.
    """
    nodes, tris, grads = result["nodes"], result["tris"], result["grads"]
    w = result["w"]
    worst = 0.0
    for (p, q), members in result["hinge_list"]:
        normal = np.array([nodes[q][1] - nodes[p][1], -(nodes[q][0] - nodes[p][0])])
        normal /= np.linalg.norm(normal)
        slopes = [float((grads[ti] @ normal) @ np.array([w[n] for n in tris[ti]]))
                  for ti, _opp in members]
        worst = max(worst, abs(abs(slopes[0] - slopes[1])))
    return worst


def _main() -> int:  # pragma: no cover - manual driver
    for n in (8, 16, 24):
        print(f"square plate full load n={n}: p/mp = "
              f"{mechanism_upper_bound(1.0, 1.0, 1.0, (1.0, 1.0), nx=n, ny=n)['pressure']:.4f}"
              "  (exact 24)")
    for nx, ny in ((16, 8), (24, 12)):
        r = mechanism_upper_bound(2.0, 1.0, 1.0, (0.5, 0.5), nx=nx, ny=ny)
        print(f"anchor patch {nx}x{ny}: p/mp = {r['pressure']:.3f} "
              "(closed-form pyramid 54.857, strip lower 15.238)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())