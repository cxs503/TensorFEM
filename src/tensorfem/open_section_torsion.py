"""Open thin-walled section torsion: Vlasov sectorial properties, shear centre.

Why this module exists
----------------------
``tensorfem.saint_venant_torsion`` qualifies the merged HEX8 path against exact
Saint-Venant fields and ``tensorfem.box_torsion`` adds the *closed* thin-walled
(box girder) oracle with warping.  Neither covers the **open** thin-walled
profile, which is what a ship stiffener (flat bar, angle, T, L2) actually is.
The gap was visible in production::

    tensorfem.marine_stiffener_failure.lateral_torsional_buckling_screen(
        torsion_constant=1.2e-7, warping_constant=2.5e-8)      # hard-coded

The elastic lateral-torsional buckling moment

    Mcr = (pi/L) sqrt(E Iy G J) sqrt(1 + pi^2 E Cw / (L^2 G J))

is *entirely* driven by ``J`` (Saint-Venant torsion constant) and ``Cw``
(warping constant), and the repository had no way to obtain either from
scantlings.  Hand-typed numbers are exactly the kind of unanchored input that
turns a strength check into a paper exercise, so this module supplies the
classical thin-walled (Vlasov) calculator together with its verification face
(``docs/open-section-torsion.md``, ``tests/test_open_section_torsion.py``).

Model and conventions
---------------------
A section is a **tree of straight walls** (node coordinates + one thickness per
wall).  Stiffener profiles branch at the web/flange junction, so the wall list is
an edge list, not a single polyline; the tree requirement is what makes the
warping function single valued (a closed loop needs the ``box_torsion``
compatibility constant instead and is refused for warping here).

With ``z`` the bar axis, in the ``(x, y)`` section plane and right handed:

* sectorial coordinate (sectorial area) about pole ``P``

      w(s) = int_{s0}^{s} r_t ds',   r_t(s) = (x - x_P) y'(s) - (y - y_P) x'(s)

* geometric integrals over the wall area ``dA = t ds``: ``A``, first moments,
  ``Ixx``, ``Iyy``, ``Ixy``;
* Saint-Venant torsion constant of the open profile: ``j_thin = sum L t^3 / 3``
  (leading order) and ``j_rect = sum J_rect(L, t)`` with ``J_rect`` taken from
  the exponential Timoshenko series already used by
  :class:`tensorfem.saint_venant_torsion.RectSection` (the same function, so
  there is one rectangle oracle in the repo, not two);
* **shear centre** from the Vlasov principal-pole conditions
  ``int w dA = int x w dA = int y w dA = 0`` (``x``, ``y`` from the centroid),
  and **warping constant** ``Cw = int w^2 dA`` with that normalization.

Those three conditions are the thin-walled statement of "a transverse shear
resultant acting through the shear centre produces no twist", i.e. precisely the
quantity that stiffener tripping / lateral-torsional buckling needs.

All integrals use 5-point Gauss quadrature per wall, which is exact for these
integrands (at most quadratic in the arc length), so the closed-form
comparisons reach machine precision instead of "a few digits".

Scope / honesty
---------------
* Thin-walled theory only: the wall is not resolved through the thickness.  The
  ``O(t/b)`` error of ``j_thin`` against the exact rectangle series is *measured*
  (:func:`fem_torsion_constant` gives the 3-D HEX8 number as well), not hidden.
* Open sections for warping; closed profiles get ``j_bredt`` only.
* Junctions are centreline intersections: no fillet and no junction overlap
  penalty, so multi-wall ``J`` carries the classical "sum of rectangles" bias.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .saint_venant_torsion import RectSection

__all__ = [
    "OpenSectionProperties",
    "properties",
    "section_properties",
    "sectorial_coordinates",
    "wall_quadrature",
    "flat_bar",
    "angle_section",
    "tee_section",
    "i_section",
    "z_section",
    "channel_section",
    "box_section",
    "stiffener_profile",
    "stiffener_properties",
    "fem_torsion_constant",
    "rect_block_mesh",
    "prism_extrude",
    "weld_meshes",
]

_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(5)


# --------------------------------------------------------------------------- #
# wall graph
# --------------------------------------------------------------------------- #
def _wall_graph(nodes, edges):
    """Validate ``nodes``/``edges`` and return ``(nodes, edges)`` in tree order."""
    pts = np.asarray(nodes, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("nodes must be an (n, 2) sequence of section coordinates")
    if len(pts) < 2:
        raise ValueError("a section needs at least two nodes")
    if np.any(~np.isfinite(pts)):
        raise ValueError("non-finite section coordinate")

    raw = list(edges)
    if not raw:
        raise ValueError("at least one wall is required")
    triples = []
    for k, item in enumerate(raw):
        if len(item) == 2:
            raise ValueError(f"wall {k} needs (node_a, node_b, thickness)")
        a, b, t = int(item[0]), int(item[1]), float(item[2])
        if not (0 <= a < len(pts)) or not (0 <= b < len(pts)):
            raise ValueError(f"wall {k} references a node outside 0..{len(pts) - 1}")
        if a == b:
            raise ValueError(f"wall {k} joins node {a} to itself")
        if not math.isfinite(t) or t <= 0.0:
            raise ValueError(f"wall {k} thickness must be positive and finite")
        if float(np.hypot(*(pts[b] - pts[a]))) <= 0.0:
            raise ValueError(f"wall {k} has zero length")
        triples.append((a, b, t))

    # breadth first traversal orients every wall (start node first) and proves
    # the graph is a connected tree, which is what makes warping single valued
    adjacency = {i: [] for i in range(len(pts))}
    for k, (a, b, _) in enumerate(triples):
        adjacency[a].append((b, k))
        adjacency[b].append((a, k))
    visited, used = {0}, set()
    ordered = []
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for other, k in adjacency[node]:
            if k in used:
                continue
            if other in visited:
                raise ValueError("open section walls must form a tree: wall "
                                 f"{k} closes a loop (use closed=True/Bredt)")
            used.add(k)
            visited.add(other)
            queue.append(other)
            ordered.append((node, other, triples[k][2]))
    if len(used) != len(triples) or len(visited) != len(pts):
        raise ValueError("disconnected section: not every wall/node is reachable")
    return pts, ordered


def wall_quadrature(nodes, edges):
    """Quadrature data of ``int f dA`` for every wall.

    Returns ``(per_wall, lengths)`` where each entry is a dict with the abscissae
    ``points`` (m, 2), the weights ``weights`` (= ``t ds``) and the arc length
    ``s`` measured from the wall's start node.
    """
    pts, walls = _wall_graph(nodes, edges)
    per_wall, lengths = [], []
    for a, b, t in walls:
        p0, p1 = pts[a], pts[b]
        length = float(np.hypot(*(p1 - p0)))
        s = 0.5 * length * (_GAUSS_X + 1.0)
        points = p0[None, :] + (p1 - p0)[None, :] * (s / length)[:, None]
        per_wall.append({"points": points, "weights": 0.5 * length * _GAUSS_W * t,
                         "s": s, "start": a, "end": b, "thickness": t,
                         "length": length})
        lengths.append(length)
    return per_wall, lengths


def sectorial_coordinates(nodes, edges, pole) -> tuple[list[np.ndarray], np.ndarray]:
    """Sectorial coordinate ``w`` at wall quadrature points and at the nodes.

    ``pole`` is the Vlasov pole; ``w`` is zero at the root node and accumulates
    ``r_t ds`` along the wall tree, so it is single valued by construction.
    """
    pts, walls = _wall_graph(nodes, edges)
    px, py = float(pole[0]), float(pole[1])
    node_w = np.zeros(len(pts))
    known = np.zeros(len(pts), dtype=bool)
    known[walls[0][0]] = True
    node_w[walls[0][0]] = 0.0
    per_wall = []
    for a, b, _ in walls:
        p0, p1 = pts[a], pts[b]
        length = float(np.hypot(*(p1 - p0)))
        tx, ty = (p1 - p0) / length
        r0 = (p0[0] - px) * ty - (p0[1] - py) * tx
        r1 = (p1[0] - px) * ty - (p1[1] - py) * tx
        slope = (r1 - r0) / length
        s = 0.5 * length * (_GAUSS_X + 1.0)
        w = node_w[a] + r0 * s + 0.5 * slope * s * s
        node_w[b] = node_w[a] + 0.5 * (r0 + r1) * length
        known[b] = True
        per_wall.append(w)
    return per_wall, node_w


def _flatten(per_wall):
    return np.concatenate([w["points"] for w in per_wall]), \
        np.concatenate([w["weights"] for w in per_wall])


# --------------------------------------------------------------------------- #
# properties
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OpenSectionProperties:
    """Thin-walled properties of a wall-tree (open) or wall-loop (closed) section."""

    name: str
    closed: bool
    area: float
    centroid: tuple[float, float]
    i_xx: float
    i_yy: float
    i_xy: float
    j_thin: float
    j_rect: float
    j_bredt: float | None
    shear_centre: tuple[float, float] | None
    warping_constant: float | None
    node_warping: tuple[float, ...] | None
    orthogonality_residuals: tuple[float, float] | None
    wall_lengths: tuple[float, ...]
    notes: dict = field(default_factory=dict)

    @property
    def j(self) -> float:
        """Recommended Saint-Venant constant (rectangle-series refinement)."""
        return self.j_bredt if self.closed else self.j_rect

    @property
    def radius_of_gyration_polar(self) -> float:
        """``sqrt((Ixx + Iyy)/A)``: the torsional-buckling length scale."""
        return math.sqrt((self.i_xx + self.i_yy) / self.area)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "closed": self.closed,
            "area": self.area,
            "centroid": list(self.centroid),
            "i_xx": self.i_xx,
            "i_yy": self.i_yy,
            "i_xy": self.i_xy,
            "j_thin": self.j_thin,
            "j_rect": self.j_rect,
            "j_bredt": self.j_bredt,
            "j_recommended": self.j,
            "shear_centre": None if self.shear_centre is None else list(self.shear_centre),
            "warping_constant": self.warping_constant,
            "orthogonality_residuals": (None if self.orthogonality_residuals is None
                                        else list(self.orthogonality_residuals)),
            "node_warping": None if self.node_warping is None else list(self.node_warping),
            "wall_lengths": list(self.wall_lengths),
            "notes": dict(self.notes),
        }


def _polyline_edges(count: int, thickness) -> list[tuple[int, int, float]]:
    thick = np.atleast_1d(np.asarray(thickness, dtype=np.float64))
    if thick.size == 1:
        thick = np.full(count - 1, float(thick[0]))
    if thick.size != count - 1:
        raise ValueError("thickness needs one entry per consecutive wall")
    return [(i, i + 1, float(thick[i])) for i in range(count - 1)]


def _loop_area(points) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))


def section_properties(nodes, thickness=None, *, edges=None, name: str = "section",
                       closed: bool = False) -> OpenSectionProperties:
    """Thin-walled area/centroid/inertias, ``J``, shear centre and ``Cw``.

    ``edges`` is an explicit ``[(node_a, node_b, t), ...]`` wall list (use it for
    branching profiles); the positional ``thickness`` argument describes a
    polyline path instead.  ``closed=True`` restricts the result to the Bredt
    torsion constant plus the geometric integrals.
    """
    if edges is None:
        pts = np.asarray(nodes, dtype=np.float64)
        edges = _polyline_edges(len(pts), thickness)
    else:
        if thickness is not None:
            raise ValueError("pass either thickness (polyline) or edges, not both")
        pts = np.asarray(nodes, dtype=np.float64)
    if closed:
        pts_arr = pts
        if not np.allclose(pts_arr[0], pts_arr[-1]):
            raise ValueError("a closed section must repeat its first node last")
        walls = [(a, b, float(t)) for a, b, t in edges]
        per_wall = []
        for a, b, t in walls:
            p0, p1 = pts_arr[a], pts_arr[b]
            length = float(np.hypot(*(p1 - p0)))
            s = 0.5 * length * (_GAUSS_X + 1.0)
            per_wall.append({"points": p0[None, :] + (p1 - p0)[None, :] * (s / length)[:, None],
                             "weights": 0.5 * length * _GAUSS_W * t, "s": s,
                             "start": a, "end": b, "thickness": t, "length": length})
        points, weights = _flatten(per_wall)
        area = float(weights.sum())
        sx = float((weights * points[:, 1]).sum())
        sy = float((weights * points[:, 0]).sum())
        centroid = (sy / area, sx / area)
        dx = points[:, 0] - centroid[0]
        dy = points[:, 1] - centroid[1]
        ixx = float((weights * dy * dy).sum())
        iyy = float((weights * dx * dx).sum())
        ixy = float((weights * dx * dy).sum())
        perim = float(sum(w["length"] / w["thickness"] for w in per_wall))
        enclosed = _loop_area(pts_arr)
        return OpenSectionProperties(
            name=name, closed=True, area=area, centroid=centroid, i_xx=ixx,
            i_yy=iyy, i_xy=ixy, j_thin=float("nan"), j_rect=float("nan"),
            j_bredt=4.0 * enclosed ** 2 / perim, shear_centre=None,
            warping_constant=None, node_warping=None, orthogonality_residuals=None,
            wall_lengths=tuple(w["length"] for w in per_wall),
            notes={"enclosed_area": enclosed, "perimeter_over_thickness": perim,
                   "j_convention": "Bredt 4 A^2 / oint(ds/t)",
                   "warping": "closed-section warping lives in box_torsion"})

    per_wall, lengths = wall_quadrature(pts, edges)
    points, weights = _flatten(per_wall)
    area = float(weights.sum())
    sx = float((weights * points[:, 1]).sum())
    sy = float((weights * points[:, 0]).sum())
    centroid = (sy / area, sx / area)
    dx = points[:, 0] - centroid[0]
    dy = points[:, 1] - centroid[1]
    ixx = float((weights * dy * dy).sum())
    iyy = float((weights * dx * dx).sum())
    ixy = float((weights * dx * dy).sum())

    j_thin = float(sum(w["length"] * w["thickness"] ** 3 for w in per_wall) / 3.0)
    j_rect = float(sum(RectSection(0.5 * max(w["length"], w["thickness"]),
                                   0.5 * min(w["length"], w["thickness"])).j
                       for w in per_wall))

    w_walls, node_w = sectorial_coordinates(pts, edges, centroid)
    w = np.concatenate(w_walls)
    w0 = float((weights * w).sum())
    wx = float((weights * w * dx).sum())
    wy = float((weights * w * dy).sum())
    # principal pole conditions (x, y measured from the centroid, so the
    # additive start-point constant drops out):
    #   wx - xp ixy + yp iyy = 0        (int x w dA = 0)
    #   wy - xp ixx + yp ixy = 0        (int y w dA = 0)
    mat = np.array([[-ixy, iyy], [-ixx, ixy]], dtype=np.float64)
    rhs = np.array([-wx, -wy], dtype=np.float64)
    degenerate = abs(float(np.linalg.det(mat))) <= 1e-18 * max(1.0, ixx * iyy)
    if degenerate:
        # single straight wall: the section is a line, so the pole is only
        # determined along the wall; take the minimum-norm (centroid) choice and
        # say so instead of raising -- flat bars are the commonest stiffener.
        xp, yp = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    else:
        xp, yp = np.linalg.solve(mat, rhs)
    w = w - xp * dy + yp * dx
    node_w = node_w - xp * (pts[:, 1] - centroid[1]) + yp * (pts[:, 0] - centroid[0])
    mean = float((weights * w).sum()) / area             # start-point constant
    w = w - mean
    node_w = node_w - mean
    cw = float((weights * w * w).sum())
    norm_x = math.sqrt(max(cw * iyy, 0.0))
    norm_y = math.sqrt(max(cw * ixx, 0.0))
    res_x = float((weights * w * dx).sum()) / norm_x if norm_x > 0.0 else 0.0
    res_y = float((weights * w * dy).sum()) / norm_y if norm_y > 0.0 else 0.0
    return OpenSectionProperties(
        name=name, closed=False, area=area, centroid=centroid, i_xx=ixx,
        i_yy=iyy, i_xy=ixy, j_thin=j_thin, j_rect=j_rect, j_bredt=None,
        shear_centre=(float(centroid[0] + xp), float(centroid[1] + yp)),
        warping_constant=cw, node_warping=tuple(float(v) for v in node_w),
        orthogonality_residuals=(res_x, res_y),
        wall_lengths=tuple(lengths),
        notes={"j_convention": "sum L t^3 / 3 (j_thin); j_rect = exact rectangle series",
               "warping": "Cw = int w^2 dA with int w dA = int x w dA = int y w dA = 0",
               "degenerate_pole": bool(degenerate)})


# --------------------------------------------------------------------------- #
# profiles (centreline wall trees, ship scantling vocabulary)
# --------------------------------------------------------------------------- #
def flat_bar(width: float, thickness: float, *, name: str = "flat_bar") -> dict:
    """Plate stiffener without a flange: one wall along ``x``."""
    return {"name": name, "nodes": [(0.0, 0.0), (float(width), 0.0)],
            "edges": [(0, 1, float(thickness))]}


def angle_section(leg_a: float, leg_b: float, thickness: float,
                  *, name: str = "angle") -> dict:
    """L profile: legs along ``+x`` and ``+y`` from the corner centreline."""
    a, b, t = float(leg_a), float(leg_b), float(thickness)
    return {"name": name, "nodes": [(a, 0.0), (0.0, 0.0), (0.0, b)],
            "edges": [(0, 1, t), (1, 2, t)]}


def tee_section(web_depth: float, flange_width: float, web_t: float,
                flange_t: float, *, name: str = "tee") -> dict:
    """T profile: flange centreline at ``y = web_depth``, web foot at ``y = 0``."""
    h, b = float(web_depth), float(flange_width)
    nodes = [(0.0, 0.0), (0.0, h), (-0.5 * b, h), (0.5 * b, h)]
    return {"name": name, "nodes": nodes,
            "edges": [(0, 1, float(web_t)), (1, 2, float(flange_t)),
                      (1, 3, float(flange_t))]}


def i_section(web_depth: float, flange_width: float, web_t: float,
              flange_t: float, *, name: str = "i_section") -> dict:
    """Doubly symmetric I profile; ``web_depth`` is the flange-centreline distance."""
    h, b = float(web_depth), float(flange_width)
    nodes = [(0.0, 0.0), (0.0, h), (-0.5 * b, 0.0), (0.5 * b, 0.0),
             (-0.5 * b, h), (0.5 * b, h)]
    return {"name": name, "nodes": nodes,
            "edges": [(0, 1, float(web_t)), (0, 2, float(flange_t)),
                      (0, 3, float(flange_t)), (1, 4, float(flange_t)),
                      (1, 5, float(flange_t))]}


def z_section(web_depth: float, flange_width: float, web_t: float,
              flange_t: float, *, name: str = "z_section") -> dict:
    """Point-symmetric Z profile: flanges point in opposite directions."""
    h, b = float(web_depth), float(flange_width)
    nodes = [(0.0, 0.0), (0.0, h), (b, h), (-b, 0.0)]
    return {"name": name, "nodes": nodes,
            "edges": [(0, 1, float(web_t)), (1, 2, float(flange_t)),
                      (0, 3, float(flange_t))]}


def channel_section(web_depth: float, flange_width: float, web_t: float,
                    flange_t: float, *, name: str = "channel") -> dict:
    """Channel (U) profile: web on ``x = 0`` at ``y = -h/2``, flanges towards ``+x``."""
    h, b = float(web_depth), float(flange_width)
    nodes = [(0.0, 0.0), (0.0, h), (b, h), (b, 0.0)]
    return {"name": name, "nodes": nodes,
            "edges": [(0, 1, float(web_t)), (1, 2, float(flange_t)),
                      (0, 3, float(flange_t))]}


def box_section(width: float, depth: float, thickness: float,
                *, name: str = "box") -> dict:
    """Closed rectangular tube, ``width``/``depth`` on the centreline."""
    w, d = float(width), float(depth)
    nodes = [(-0.5 * w, -0.5 * d), (0.5 * w, -0.5 * d), (0.5 * w, 0.5 * d),
             (-0.5 * w, 0.5 * d), (-0.5 * w, -0.5 * d)]
    return {"name": name, "nodes": nodes,
            "edges": [(0, 1, float(thickness)), (1, 2, float(thickness)),
                      (2, 3, float(thickness)), (3, 4, float(thickness))],
            "closed": True}


def stiffener_profile(web_depth: float, web_t: float, flange_width: float = 0.0,
                      flange_t: float = 0.0, *, name: str = "") -> dict:
    """Map a shipyard stiffener description onto a profile."""
    if flange_width <= 0.0 or flange_t <= 0.0:
        return flat_bar(web_depth, web_t, name=name or "flat_bar")
    return tee_section(web_depth, flange_width, web_t, flange_t,
                       name=name or "tee_stiffener")


def properties(profile: dict) -> OpenSectionProperties:
    """Thin-walled properties of a profile dict from this module."""
    return section_properties(profile["nodes"], edges=profile["edges"],
                              name=profile.get("name", "section"),
                              closed=bool(profile.get("closed", False)))


def stiffener_properties(web_depth: float, web_t: float, flange_width: float = 0.0,
                         flange_t: float = 0.0) -> OpenSectionProperties:
    """``J``/``Cw``/shear centre of a stiffener straight from its scantlings."""
    return properties(stiffener_profile(web_depth, web_t, flange_width, flange_t))


# --------------------------------------------------------------------------- #
# 3-D HEX8 torsion cross-check (independent of the thin-wall algebra)
# --------------------------------------------------------------------------- #
def rect_block_mesh(xs: Iterable[float], ys: Iterable[float]):
    """Structured quad block on the tensor grid ``xs x ys``."""
    xs = np.asarray(list(xs), dtype=np.float64)
    ys = np.asarray(list(ys), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    nx, ny = len(xs), len(ys)

    def idx(i, j):
        return i * ny + j

    quads = [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))
             for i in range(nx - 1) for j in range(ny - 1)]
    return nodes, np.asarray(quads, dtype=np.int64)


def weld_meshes(meshes, tol: float = 1e-12):
    """Merge coincident nodes of several 2-D blocks (exact shared junctions)."""
    nodes = np.vstack([m[0] for m in meshes])
    key = np.round(nodes / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    merged = nodes[first]
    remap = inverse.reshape(-1)
    quads, offset = [], 0
    for block_nodes, block_quads in meshes:
        quads.append(remap[offset + block_quads])
        offset += len(block_nodes)
    return merged, np.vstack(quads)


def prism_extrude(nodes_2d, quads, length: float, layers: int):
    """Extrude a 2-D quad mesh into a HEX8 prism (bar axis ``z``)."""
    nodes_2d = np.asarray(nodes_2d, dtype=np.float64)
    m = len(nodes_2d)
    zs = np.linspace(0.0, float(length), int(layers) + 1)
    nodes = np.vstack([np.column_stack([nodes_2d, np.full(m, z)]) for z in zs])
    hexes = np.vstack([np.hstack([quads + k * m, quads + (k + 1) * m])
                       for k in range(int(layers))])
    return nodes, hexes


def fem_torsion_constant(nodes3d, hexes, *, young: float = 2.6, poisson: float = 0.3,
                         twist: float = 1.0, rtol: float = 1e-9) -> dict:
    """``J_FE = T/(G theta)`` of a prismatic mesh under Saint-Venant kinematics.

    Both end faces carry the *exact* transverse rotation field
    ``u_x = -theta z y``, ``u_y = theta z x`` (``z = 0`` therefore removes the
    rigid rotation), ``u_z`` (warping) is free everywhere and the lateral surface
    is traction free.  The torque is read from the end reactions, so the result
    never uses the thin-wall formula it is meant to check.
    """
    import torch

    from .solid3d import elasticity_matrix_3d, hex8_stiffness
    from .sparse_core import assemble_coo, solve_sparse_static

    nodes = torch.as_tensor(np.asarray(nodes3d), dtype=torch.float64)
    conn = torch.as_tensor(np.asarray(hexes), dtype=torch.long)
    ne = len(conn)
    ndof = 3 * len(nodes)
    shear = young / (2.0 * (1.0 + poisson))
    D = elasticity_matrix_3d(nodes.new_full((ne,), young), nodes.new_full((ne,), poisson))
    ke, _ = hex8_stiffness(nodes[conn], D)
    edofs = (3 * conn[:, :, None] + torch.arange(3)).reshape(ne, 24)
    K = assemble_coo(ke, edofs, ndof)
    dirichlet = {}
    zmax = float(nodes[:, 2].max())
    for zval in (0.0, zmax):
        sl = torch.nonzero(torch.isclose(nodes[:, 2], torch.tensor(zval,
                                                                  dtype=nodes.dtype))).flatten()
        xy = nodes[sl]
        for pos, node in enumerate(sl.tolist()):
            dirichlet[3 * node] = float(-twist * zval * xy[pos, 1])
            dirichlet[3 * node + 1] = float(twist * zval * xy[pos, 0])
    sol = solve_sparse_static(K, nodes.new_zeros(ndof), dirichlet=dirichlet, rtol=rtol)
    u, reaction = sol.displacement, sol.reaction
    tip = torch.nonzero(torch.isclose(nodes[:, 2], torch.tensor(zmax,
                                                               dtype=nodes.dtype))).flatten()
    torque = float((nodes[tip, 0] * reaction[3 * tip + 1]
                    - nodes[tip, 1] * reaction[3 * tip]).sum())
    return {"j_fem": abs(torque) / (shear * twist), "torque": abs(torque),
            "dofs": int(ndof), "elements": int(ne),
            "warping_max": float(u[2::3].abs().max())}