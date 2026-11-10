"""ASTM E399 specimens (C(T), SE(B)) and Griffith-Irwin anchors for AT2 phase-field fracture.

:mod:`tensorfem.phase_field_fracture` carries the damage physics and the coupled
solver; this module adds the *specimen* side that turns a phase-field simulation
into a fracture-mechanics measurement:

* standard specimen geometries -- single-edge-notched tension (Miehe), SE(B)
  three-point bending and C(T) compact tension with pin-loaded holes carved out
  of a structured Q4 mesh by element removal;
* consistent pin loading: the classical cosine contact pressure
  ``p(phi) = p0 cos(phi)`` on the loaded half of a circular hole, integrated
  element-wise so that the resultant equals the applied force *exactly*
  (independent of the stair-cased hole boundary);
* rigid-pin / rigid-roller displacement control dof sets, so the post-peak
  softening branch is reachable;
* the ASTM E399 stress-intensity shape factors and the plane-strain Irwin
  relation ``K_Ic = sqrt(Gc E')`` that anchor the phase-field peak load against
  linear-elastic fracture mechanics (LEFM);
* measurement post-processing: peak-load identification, energy-ledger residual,
  crack-path extraction and an ``l -> 0`` extrapolation of the peak load.

Everything here is geometry and bookkeeping: the constitutive model, the tangent
and the solvers are untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .phase_field_fracture import (
    PhaseFieldProblem,
    crack_line_elements,
    nodes_on_segment,
    structured_quad_mesh,
)

__all__ = [
    "plane_strain_modulus",
    "irwin_k_from_gc",
    "gc_from_irwin_k",
    "griffith_stress",
    "tada_edge_crack_factor",
    "sent_critical_stress",
    "astm_ct_factor",
    "astm_ct_k",
    "astm_ct_predict_load",
    "astm_seb_factor",
    "astm_seb_k",
    "astm_seb_predict_load",
    "circular_hole_elements",
    "hole_boundary_ring",
    "pin_load_forces",
    "lock_circle_damage",
    "Specimen",
    "ct_setup",
    "seb_setup",
    "sent_setup",
    "rigid_control_dofs",
    "initial_damage_field",
    "relax_damage",
    "peak_load",
    "ledger_residuals",
    "damage_path",
    "fit_limit",
]


# --------------------------------------------------------------------------------------
# LEFM / Griffith anchors
# --------------------------------------------------------------------------------------
def plane_strain_modulus(E: float, nu: float) -> float:
    """Plane-strain effective modulus ``E' = E / (1 - nu^2)``."""
    return E / (1.0 - nu ** 2)


def irwin_k_from_gc(Gc: float, E: float, nu: float) -> float:
    """Irwin relation ``K_Ic = sqrt(Gc E')`` (plane strain)."""
    return math.sqrt(max(float(Gc), 0.0) * plane_strain_modulus(E, nu))


def gc_from_irwin_k(K: float, E: float, nu: float) -> float:
    """Inverse Irwin relation ``Gc = K^2 (1 - nu^2) / E``."""
    return K * K / plane_strain_modulus(E, nu)


def griffith_stress(Gc: float, E: float, nu: float, a: float) -> float:
    """Griffith critical stress of an infinite plate, ``sigma = K_Ic / sqrt(pi a)``."""
    return irwin_k_from_gc(Gc, E, nu) / math.sqrt(math.pi * a)


def tada_edge_crack_factor(alpha: float) -> float:
    """Tada single-edge-notch tension shape factor (``H/W >= 1``, error < 1 %)."""
    a = float(alpha)
    return 1.12 - 0.231 * a + 10.55 * a ** 2 - 21.72 * a ** 3 + 30.39 * a ** 4


def sent_critical_stress(Gc: float, E: float, nu: float, a: float, W: float) -> float:
    """Griffith anchor with the finite-width correction of a SENT specimen.

    ``sigma_c = K_Ic / (sqrt(pi a) F(a/W))``.  The ``F = 1`` value is the strict
    (infinite-plate) Griffith stress :func:`griffith_stress`.
    """
    return irwin_k_from_gc(Gc, E, nu) / (math.sqrt(math.pi * a) * tada_edge_crack_factor(a / W))


# --------------------------------------------------------------------------------------
# ASTM E399 shape factors
# --------------------------------------------------------------------------------------
def astm_ct_factor(alpha: float) -> float:
    """C(T) shape factor of ASTM E399, ``K = P / (B sqrt(W)) * f(a/W)``.

    Valid for ``0.2 <= a/W <= 1``; ``P`` is the load carried by one pin.
    """
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("a/W must lie in (0, 1)")
    return ((2.0 + a) * (0.886 + 4.64 * a - 13.32 * a ** 2 + 14.72 * a ** 3 - 5.60 * a ** 4)
            / (1.0 - a) ** 1.5)


def astm_ct_k(P: float, B: float, W: float, alpha: float) -> float:
    """ASTM E399 C(T) stress-intensity factor for the pin load ``P``."""
    return P / (B * math.sqrt(W)) * astm_ct_factor(alpha)


def astm_ct_predict_load(K_ic: float, B: float, W: float, alpha: float) -> float:
    """Pin load at which ``K_I = K_Ic`` in a C(T) specimen (peak-load prediction)."""
    return K_ic * B * math.sqrt(W) / astm_ct_factor(alpha)


def astm_seb_factor(alpha: float) -> float:
    """SE(B) shape factor of ASTM E399 for ``S = 4 W``.

    ``K = 3 P S / (2 B W^{3/2}) * f(a/W)``; ``P`` is the total roller load.
    """
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("a/W must lie in (0, 1)")
    return (3.0 * math.sqrt(a) / (2.0 * (1.0 + 2.0 * a) * (1.0 - a) ** 1.5)
            * (1.99 - a * (1.0 - a) * (2.15 - 3.93 * a + 2.70 * a ** 2)))


def astm_seb_k(P: float, B: float, W: float, alpha: float, S: float | None = None) -> float:
    """ASTM E399 SE(B) stress-intensity factor (span ``S`` defaults to ``4 W``)."""
    S = 4.0 * W if S is None else S
    return 3.0 * P * S / (2.0 * B * W ** 1.5) * astm_seb_factor(alpha)


def astm_seb_predict_load(K_ic: float, B: float, W: float, alpha: float,
                          S: float | None = None) -> float:
    """Roller load at which ``K_I = K_Ic`` in a SE(B) specimen."""
    S = 4.0 * W if S is None else S
    return K_ic * 2.0 * B * W ** 1.5 / (3.0 * S * astm_seb_factor(alpha))


# --------------------------------------------------------------------------------------
# hole carving and pin loading
# --------------------------------------------------------------------------------------
def circular_hole_elements(nodes: np.ndarray, elements: np.ndarray, center,
                           radius: float) -> np.ndarray:
    """Element indices whose centroid lies inside the circle (hole + pin bore)."""
    c = np.asarray(center, dtype=np.float64)
    centroid = np.asarray(nodes, dtype=np.float64)[np.asarray(elements)].mean(axis=1)
    d2 = ((centroid - c) ** 2).sum(axis=1)
    return np.where(d2 < float(radius) ** 2)[0]


def hole_boundary_ring(nodes: np.ndarray, elements: np.ndarray, void_elements,
                       center, radius: float, tol: float | None = None):
    """Stair-cased hole boundary: nodes shared by a kept and a void element.

    Returns ``(node_ids, polar_angles)`` sorted by increasing angle measured from
    the global ``+x`` axis about ``center``.
    """
    void = np.zeros(len(elements), dtype=bool)
    void[np.asarray(void_elements, dtype=np.int64)] = True
    node_in_void = np.zeros(len(nodes), dtype=bool)
    node_in_void[np.asarray(elements)[void].ravel()] = True
    node_in_keep = np.zeros(len(nodes), dtype=bool)
    node_in_keep[np.asarray(elements)[~void].ravel()] = True
    ring = np.where(node_in_void & node_in_keep)[0]
    if tol is not None:
        r = np.linalg.norm(np.asarray(nodes)[ring] - np.asarray(center, dtype=np.float64), axis=1)
        ring = ring[np.abs(r - float(radius)) <= float(tol)]
    v = np.asarray(nodes)[ring] - np.asarray(center, dtype=np.float64)
    angle = np.arctan2(v[:, 1], v[:, 0])
    order = np.argsort(angle)
    return ring[order], angle[order]


def pin_load_forces(nodes: np.ndarray, elements: np.ndarray, void_elements, center,
                    radius: float, force: float, direction=(0.0, 1.0),
                    half_angle: float = 0.5 * math.pi):
    """Consistent nodal forces of a pin (cosine contact pressure) in a bored hole.

    The contact pressure acts on the half of the hole that faces the load
    direction, ``p(phi) = p0 cos(phi)`` with ``phi`` measured from the load
    direction and ``p0 = 2 F / (pi r)``.  The pressure is integrated analytically
    over the arc segment attached to every boundary node, so the nodal vector
    sums to exactly ``force * direction`` (the resultant, and therefore the
    applied stress intensity, does not depend on the stair-casing of the hole).

    Returns ``(force_vector, info)`` where the vector has length ``2 n_nodes``.
    """
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-30)
    # the pin presses the material in the direction of the applied force, so the
    # contact arc is centred on sign(force) * direction
    drive = d if float(force) >= 0.0 else -d
    e2 = np.array([-drive[1], drive[0]])
    ring, angle = hole_boundary_ring(nodes, elements, void_elements, center, radius)
    if ring.size < 4:
        raise ValueError("hole boundary ring not resolved by the mesh")
    phi0 = math.atan2(drive[1], drive[0])
    rel = np.mod(angle - phi0 + math.pi, 2.0 * math.pi) - math.pi
    keep = np.abs(rel) <= half_angle
    ring, rel = ring[keep], rel[keep]
    if ring.size < 3:
        raise ValueError("no boundary nodes inside the contact arc")
    order = np.argsort(rel)
    ring, rel = ring[order], rel[order]
    # arc-segment edges: midpoints between consecutive angles, clipped to the contact arc
    edges = np.concatenate(([-half_angle], 0.5 * (rel[1:] + rel[:-1]), [half_angle]))
    p0 = 2.0 * abs(float(force)) / (math.pi * radius)
    f = np.zeros(2 * len(nodes))
    arcs = []
    for k in range(ring.size):
        pa, pb = float(edges[k]), float(edges[k + 1])
        if pb <= pa:
            continue
        arc_e1 = 0.5 * (pb - pa) + 0.25 * (math.sin(2.0 * pb) - math.sin(2.0 * pa))
        arc_e2 = 0.5 * (math.sin(pb) ** 2 - math.sin(pa) ** 2)
        fx = p0 * radius * (arc_e1 * drive[0] + arc_e2 * e2[0])
        fy = p0 * radius * (arc_e1 * drive[1] + arc_e2 * e2[1])
        half = 0.5 * np.array([fx, fy])
        f[2 * ring[k]:2 * ring[k] + 2] += half
        f[2 * ring[k] + 2:2 * ring[k] + 4] += half
        arcs.append({"node": int(ring[k]), "phi": float(rel[k]), "arc": pb - pa,
                     "fx": float(fx), "fy": float(fy)})
    info = {"ring": ring, "phi": rel, "arcs": arcs,
            "resultant": f.reshape(-1, 2).sum(axis=0)}
    return f, info


def lock_circle_damage(nodes: np.ndarray, center, inner: float, outer: float) -> np.ndarray:
    """Nodes in the annulus ``inner <= r <= outer`` -- used to freeze ``d`` near a pin.

    The rigid-pin idealisation produces a contact stress singularity that a
    linear-elastic phase-field model would otherwise turn into spurious damage.
    """
    r = np.linalg.norm(np.asarray(nodes) - np.asarray(center, dtype=np.float64), axis=1)
    return np.where((r >= inner) & (r <= outer))[0]


# --------------------------------------------------------------------------------------
# specimens
# --------------------------------------------------------------------------------------
@dataclass
class Specimen:
    """A meshed fracture specimen plus the bookkeeping needed to load it."""

    problem: PhaseFieldProblem
    nodes: np.ndarray
    elements: np.ndarray
    fixed: dict
    control: np.ndarray                     # node ids driven by the actuator
    notch: np.ndarray                       # nodes pre-set to d = 1
    a: float = 0.0
    W: float = 1.0
    B: float = 1.0
    l: float = 0.0
    h: float = 0.0
    control2: np.ndarray | None = None         # second actuator (rigid pin pair)
    crack_line: tuple | None = None            # (mouth, tip) of the initial crack
    meta: dict = field(default_factory=dict)

    @property
    def control_dofs(self):
        return [2 * int(n) + 1 for n in self.control]

    @property
    def control2_dofs(self):
        return [] if self.control2 is None else [2 * int(n) + 1 for n in self.control2]

    @property
    def alpha(self) -> float:
        return self.a / self.W

    def summary(self) -> dict:
        out = {"a_over_W": self.alpha, "W": self.W, "B": self.B, "l": self.l, "h": self.h,
               "elements": int(self.problem.n_elem), "nodes": int(self.problem.n_nodes),
               "notch_nodes": int(len(self.notch)),
               "control_nodes": int(len(self.control)),
               "void_elements": int(0 if self.problem.void_elements is None
                                    else len(self.problem.void_elements))}
        out.update(self.meta)
        return out


def _slit_nodes(nodes: np.ndarray, elements: np.ndarray, slit_elements, tip,
                axis: int) -> np.ndarray:
    """Nodes of the removed slit that lie strictly behind the crack tip.

    The nodes of the crack-tip element itself are *not* pre-damaged, so the
    initial crack is sharp and its length is exactly the nominal ``a``.
    """
    nds = np.unique(np.asarray(elements)[np.asarray(slit_elements, dtype=np.int64)].ravel())
    tip_coord = float(np.asarray(tip)[axis])
    return nds[np.asarray(nodes)[nds, axis] < tip_coord - 1e-9]


def _pin_orphans(fixed: dict, n_nodes: int, orphan_nodes) -> None:
    """Kinematic constraints for the nodes left inside a removed element row.

    Nodes with no incident kept element carry neither stiffness nor a damage
    equation; leaving them free makes the tangential matrix singular.  They are
    pinned in both fields (the value of ``d`` is irrelevant, the fracture energy
    of their void element is excluded by the ``keep`` mask).
    """
    for nd in np.asarray(orphan_nodes, dtype=np.int64):
        fixed[2 * int(nd)] = 0.0
        fixed[2 * int(nd) + 1] = 0.0
        fixed[2 * n_nodes + int(nd)] = 0.0


def _odd_cells(span: float, target_h: float) -> tuple[int, float]:
    """Number of cells in ``span`` (forced odd) and the resulting cell size."""
    n = max(2, int(round(span / target_h)))
    if n % 2 == 0:
        n += 1
    return n, span / n


def ct_setup(a_over_w: float = 0.5, W: float = 1.0, l: float = 0.02, h: float | None = None,
             E: float = 1000.0, nu: float = 0.3, Gc: float = 1.0e-3, thickness: float = 1.0,
             hole_diameter: float = 0.25, hole_offset: float = 0.25,
             hole_spacing: float = 0.55) -> Specimen:
    """ASTM E399 C(T) compact-tension specimen with two pin-loaded holes.

    Lays the load line through the hole centres at ``x = 0``: the notched front
    face is at ``x = -hole_offset * W``, the far edge at ``x = W``, the specimen
    half height is ``0.6 W`` (``H = 1.2 W``), the holes have diameter
    ``hole_diameter * W`` and their centres are ``hole_spacing * W`` apart.  The
    blunt notch (one element thick) and the two holes are removed from the mesh
    by element deletion -- the notch is a *slit*, so no notch root radius is
    introduced by the geometry itself.

    The upper pin is driven in ``+y`` and the lower pin in ``-y``;
    :func:`rigid_control_dofs` returns the two rigid-pin displacement sets.
    """
    h = float(h) if h is not None else 0.5 * l
    H = 1.2 * W
    x_front = -hole_offset * W
    ny, hy = _odd_cells(H, h)
    nx = max(4, int(round((W - x_front) / hy)))
    nodes, elements = structured_quad_mesh(x_front, W, -0.5 * H, 0.5 * H, nx, ny)
    # ASTM E399 measures a from the *load line* (x = 0, through the pin centres),
    # so the notch cut from the front face (x = -hole_offset W) is longer by the
    # hole offset: a_notch = a + hole_offset * W
    a = a_over_w * W
    a_notch = a + hole_offset * W
    void = [crack_line_elements(nodes, elements, (x_front, 0.0), (x_front + a_notch, 0.0),
                               tol=0.3 * hy)]
    radius = 0.5 * hole_diameter * W
    centers = [(0.0, 0.5 * hole_spacing * W), (0.0, -0.5 * hole_spacing * W)]
    for c in centers:
        void.append(circular_hole_elements(nodes, elements, c, radius))
    void = np.unique(np.concatenate(void))
    problem = PhaseFieldProblem(nodes, elements, E, nu, Gc=Gc, l=l, thickness=thickness,
                               void_elements=void)
    fixed = {}
    _pin_orphans(fixed, problem.n_nodes, problem.orphan_nodes)
    # Both pins are x-fixed (a pin resists the horizontal force at its bore and
    # this removes the two rigid-body modes of the specimen): the vertical load
    # path stays free so that the specimen can be loaded by force or by a rigid
    # pin displacement.
    for c in centers:
        ring, _ = hole_boundary_ring(nodes, elements, void, c, radius)
        rr = nodes[ring]
        pick = ring[int(np.argmin((rr[:, 0] - c[0]) ** 2 + 0.01 * (rr[:, 1] - c[1]) ** 2))]
        fixed[2 * int(pick)] = 0.0
    upper, _ = hole_boundary_ring(nodes, elements, void, centers[0], radius)
    lower, _ = hole_boundary_ring(nodes, elements, void, centers[1], radius)
    up_arc = np.sort(upper[nodes[upper, 1] > centers[0][1] - 0.5 * hy])
    low_arc = np.sort(lower[nodes[lower, 1] < centers[1][1] + 0.5 * hy])
    notch = _slit_nodes(nodes, elements, void[0], (x_front + a_notch, 0.0), axis=0)
    return Specimen(problem=problem, nodes=nodes, elements=elements, fixed=fixed,
                    control=up_arc, control2=low_arc, notch=notch, a=a, W=W, B=thickness,
                    l=l, h=hy, crack_line=((x_front, 0.0), (x_front + a_notch, 0.0)),
                    meta={"specimen": "C(T)", "hole_radius": radius,
                          "hole_centers": [list(c) for c in centers],
                          "lower_pin_nodes": int(low_arc.size),
                          "crack_tip": [x_front + a_notch, 0.0],
                          "notch_from_front": a_notch,
                          "front_face_x": x_front})


def seb_setup(a_over_w: float = 0.45, W: float = 1.0, S: float | None = None,
              l: float = 0.02, h: float | None = None, E: float = 1000.0, nu: float = 0.3,
              Gc: float = 1.0e-3, thickness: float = 1.0,
              roller_width: float = 0.30, support_width: float = 0.10) -> Specimen:
    """ASTM E399 SE(B) three-point-bend specimen, span ``S`` (default ``4 W``).

    The notch (one element thick, ``d = 1``) runs from the bottom mid-span up to
    ``a``.  Both supports and the loading roller are idealised as rigid rollers:
    every node of the top face (resp. bottom face) within ``roller_width`` of the
    roller centre is driven to the same vertical displacement, which spreads the
    contact and avoids a spurious damage nucleus under a single point load.
    """
    h = float(h) if h is not None else 0.5 * l
    S = 4.0 * W if S is None else S
    nx, hx = _odd_cells(S, h)
    ny = max(3, int(round(W / hx)))
    nodes, elements = structured_quad_mesh(-0.5 * S, 0.5 * S, 0.0, W, nx, ny)
    a = a_over_w * W
    void = crack_line_elements(nodes, elements, (0.0, 0.0), (0.0, a), tol=0.3 * hx)
    problem = PhaseFieldProblem(nodes, elements, E, nu, Gc=Gc, l=l, thickness=thickness,
                               void_elements=void)
    fixed = {}
    _pin_orphans(fixed, problem.n_nodes, problem.orphan_nodes)
    bottom = nodes[:, 1] <= 1e-12
    left = np.where(bottom & (np.abs(nodes[:, 0] + 0.5 * S) <= 0.5 * support_width * W))[0]
    right = np.where(bottom & (np.abs(nodes[:, 0] - 0.5 * S) <= 0.5 * support_width * W))[0]
    for nd in np.concatenate((left, right)):
        fixed[2 * int(nd) + 1] = 0.0
    fixed[2 * int(left[0])] = 0.0                       # remove the horizontal rigid mode
    top = nodes[:, 1] >= W - 1e-12
    ctrl = np.where(top & (np.abs(nodes[:, 0]) <= 0.5 * roller_width * W))[0]
    notch = _slit_nodes(nodes, elements, void, (0.0, a), axis=1)
    return Specimen(problem=problem, nodes=nodes, elements=elements, fixed=fixed,
                    control=np.sort(ctrl), notch=notch, a=a, W=W, B=thickness, l=l, h=hx,
                    crack_line=((0.0, 0.0), (0.0, a)),
                    meta={"specimen": "SE(B)", "span": S,
                          "support_nodes": int(len(left) + len(right)),
                          "crack_tip": [0.0, a]})


def sent_setup(a_over_w: float = 0.3, W: float = 1.0, H: float | None = None,
               l: float = 0.02, h: float | None = None, E: float = 1000.0, nu: float = 0.3,
               Gc: float = 1.0e-3, thickness: float = 1.0) -> Specimen:
    """Single-edge-notched tension specimen (Miehe et al. 2010 geometry).

    Square plate ``W x 2H`` (``H`` defaults to ``W``), horizontal notch of length
    ``a`` from the mid-height of the left edge.  Grips are displacement
    controlled: the whole top and bottom faces are driven to ``+-delta``.
    """
    h = float(h) if h is not None else 0.5 * l
    H = W if H is None else H
    nx = max(4, int(round(W / h)))
    ny, hy = _odd_cells(2.0 * H, h)          # odd -> y = 0 is a cell centre line
    nodes, elements = structured_quad_mesh(0.0, W, -H, H, nx, ny)
    a = a_over_w * W
    void = crack_line_elements(nodes, elements, (0.0, 0.0), (a, 0.0), tol=0.3 * hy)
    problem = PhaseFieldProblem(nodes, elements, E, nu, Gc=Gc, l=l, thickness=thickness,
                               void_elements=void)
    fixed = {}
    _pin_orphans(fixed, problem.n_nodes, problem.orphan_nodes)
    bottom = np.where(nodes[:, 1] <= -H + 1e-12)[0]
    for nd in bottom:                       # lower grip: roller (u_y = 0)
        fixed[2 * int(nd) + 1] = 0.0
    bl = int(np.argmin(nodes[:, 0] ** 2 + (nodes[:, 1] + H) ** 2))
    fixed[2 * bl] = 0.0                     # one horizontal restraint
    top = np.where(nodes[:, 1] >= H - 1e-12)[0]
    notch = _slit_nodes(nodes, elements, void, (a, 0.0), axis=0)
    return Specimen(problem=problem, nodes=nodes, elements=elements, fixed=fixed,
                    control=np.sort(top), notch=notch, a=a, W=W, B=thickness, l=l, h=hy,
                    crack_line=((0.0, 0.0), (a, 0.0)),
                    meta={"specimen": "SENT", "height": 2.0 * H, "crack_tip": [a, 0.0]})


def initial_damage_field(specimen: "Specimen", l: float | None = None) -> np.ndarray:
    """Analytical AT2 profile ``d = exp(-r / l)`` around the initial crack.

    ``r`` is the distance from the crack line (mouth -> tip).  Starting from the
    sharp ``0/1`` notch instead would let the phase field relax at the first load
    step and book that relaxation to the fracture energy, which spoils the energy
    ledger of the first increments.
    """
    l = float(specimen.l if l is None else l)
    nodes = np.asarray(specimen.nodes, dtype=np.float64)
    if specimen.crack_line is None:
        d = np.zeros(len(nodes))
    else:
        p0 = np.asarray(specimen.crack_line[0], dtype=np.float64)
        p1 = np.asarray(specimen.crack_line[1], dtype=np.float64)
        ab = p1 - p0
        L2 = float(ab @ ab)
        t = ((nodes - p0) @ ab) / max(L2, 1e-30)
        t = np.clip(t, 0.0, 1.0)
        proj = p0 + t[:, None] * ab
        r = np.linalg.norm(nodes - proj, axis=1)
        d = np.exp(-r / max(l, 1e-30))
    d[np.asarray(specimen.notch, dtype=np.int64)] = 1.0
    return np.clip(d, 0.0, 1.0)


def relax_damage(solver, tol: float = 1e-8, max_iter: int = 100):
    """Bring the initial phase field into equilibrium at zero load (``u = 0``, ``H = 0``).

    With the analytical ``exp(-r/l)`` profile as the starting point and the notch
    nodes held at ``d = 1``, this solves the ``d`` sub-problem alone; the result is
    the discrete AT2 equilibrium profile of the initial crack, so the first load
    step no longer spends (or releases) fracture energy on relaxing the notch.
    """
    n2 = 2 * solver.problem.n_nodes
    fixed = solver.fixed.copy()
    values = solver.fixed_values.copy()
    solver.fixed[:n2] = True
    solver.fixed_values[:n2] = 0.0
    solver.u[:] = 0.0
    solver.history[:] = 0.0
    solver._apply_fixed()
    iters, hist, converged = solver.alternating_step(tol=tol, max_iter=max_iter)
    solver.fixed[:] = fixed
    solver.fixed_values[:] = values
    solver._apply_fixed()
    solver.update_history()
    return iters, converged, hist[-1]


def rigid_control_dofs(specimen: Specimen, delta: float, sign: float = 1.0,
                       second: np.ndarray | None = None):
    """Rigid-actuator boundary conditions for a displacement-controlled step.

    Returns ``(fixed_values, control_dofs, reaction_sign)`` where ``fixed_values``
    maps dof -> prescribed value (a superset of ``specimen.fixed``), and the
    physical actuator force is ``reaction_sign * sum(R[control_dofs])``.
    """
    fixed = {int(k): float(v) for k, v in specimen.fixed.items()}
    control = list(specimen.control_dofs if not hasattr(specimen, "control_dofs")
                   else specimen.control_dofs)
    for dof in control:
        fixed[int(dof)] = float(sign) * float(delta)
    if second is not None:
        for nd in np.atleast_1d(second):
            fixed[2 * int(nd) + 1] = -float(sign) * float(delta)
    return fixed, control, 1.0


def rigid_two_sided(specimen: Specimen, upper: np.ndarray, lower: np.ndarray,
                    delta: float):
    """Rigid-pin pair: upper pin ``+delta``, lower pin ``-delta`` (C(T))."""
    fixed = {int(k): float(v) for k, v in specimen.fixed.items()}
    for nd in np.atleast_1d(upper):
        fixed[2 * int(nd) + 1] = float(delta)
    for nd in np.atleast_1d(lower):
        fixed[2 * int(nd) + 1] = -float(delta)
    control = [2 * int(n) + 1 for n in np.atleast_1d(upper)]
    return fixed, control


# --------------------------------------------------------------------------------------
# runtime configuration
# --------------------------------------------------------------------------------------
def configure_threads(n_threads: int = 1) -> int:
    """Pin the torch thread pool (returns the previous setting).

    The assembly of :mod:`tensorfem.phase_field_fracture` is a chain of small
    tensor operations; on a many-core host the default pool (128 threads here)
    makes it 6x slower than single-threaded work (3.5 s vs 0.6 s per assembly of
    a 15k-element mesh).  Verification suites call this explicitly rather than
    relying on the environment.
    """
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(int(n_threads))
    return previous


# --------------------------------------------------------------------------------------
# displacement-controlled load path with an exact external-work integral
# --------------------------------------------------------------------------------------
def run_control_path(problem, solver, control_sets, measure_dofs=None, scheme: str = "monolithic",
                     max_iter: int = 25, tol: float = 1e-9, stop=None, abscissa=None):
    """Step a sequence of *multi-dof* prescribed displacement sets.

    :func:`tensorfem.phase_field_fracture.run_load_path` drives one set of dofs
    with a common value, which cannot represent a C(T) specimen (the two pins
    move in opposite directions) and therefore also mis-states the external
    work.  Here every step supplies its own ``dict`` of dof -> prescribed value,
    so the external work is integrated dof by dof

        W = sum_dof trapezoid(R_dof, u_dof),

    which is exact for the piecewise-linear load path.  Per step the record
    carries the measured force (``sum(R[measure_dofs])``), the elastic and
    fracture energies relative to the initial state, the residual history and
    the irreversibility audit.
    """
    from .phase_field_fracture import NewtonRecord, energies

    measure = list(solver.fixed.nonzero()[0]) if measure_dofs is None else list(measure_dofs)
    reference = energies(problem, solver.u, solver.d, solver.history)
    e_ref, f_ref = reference["elastic"], reference["fracture"]
    w_ref = energies(problem, solver.u, solver.d, solver.history)
    del w_ref
    free_d = ~solver.fixed[2 * problem.n_nodes:]
    prev_x = solver._x().copy()
    prev_R = solver.reactions()
    prev_d = solver.d.copy()
    work = 0.0
    records = []
    for step, prescribed in enumerate(control_sets):
        for dof, val in prescribed.items():
            solver.fixed[int(dof)] = True
            solver.fixed_values[int(dof)] = float(val)
        solver._apply_fixed()
        if scheme == "alternating":
            iters, hist, conv = solver.alternating_step(tol=1e-6, max_iter=max_iter)
            rec = NewtonRecord(hist, iters, conv)
        else:
            rec = solver.newton_step(max_iter=max_iter, tol=tol)
        solver.update_history()
        solver._apply_fixed()
        R = solver.reactions()
        x = solver._x()
        for dof in prescribed:
            work += 0.5 * (R[dof] + prev_R[dof]) * (x[dof] - prev_x[dof])
        prev_x, prev_R = x.copy(), R.copy()
        en = energies(problem, solver.u, solver.d, solver.history)
        drop = prev_d - solver.d
        value = float(abscissa[step]) if abscissa is not None else float(step + 1)
        records.append({
            "step": step,
            "value": value,
            "reaction": float(sum(R[int(d)] for d in measure)),
            "elastic": en["elastic"] - e_ref,
            "fracture": en["fracture"] - f_ref,
            "total": (en["elastic"] - e_ref) + (en["fracture"] - f_ref),
            "work": work,
            "d_max": float(solver.d.max()),
            "d_max_free": float(solver.d[free_d].max()) if bool(free_d.any()) else 0.0,
            "newton": rec.residual_norms,
            "newton_iters": rec.iterations,
            "converged": rec.converged,
            "irreversibility_violations": int((drop > 1e-10).sum()),
            "damage_drop_max": float(max(drop.max(), 0.0)),
        })
        prev_d = solver.d.copy()
        if stop is not None and stop(solver, records[-1]):
            break
    return records


# --------------------------------------------------------------------------------------
# measurement post-processing
# --------------------------------------------------------------------------------------
def peak_load(records, key: str = "reaction") -> dict:
    """Peak of a load record list, its abscissa and the post-peak drop."""
    if not records:
        raise ValueError("no load records")
    vals = np.abs(np.asarray([float(r[key]) for r in records]))
    i = int(np.argmax(vals))
    post = vals[i + 1:]
    drops = 100.0 * (vals[i] - post.min()) / max(vals[i], 1e-30) if post.size else 0.0
    diffs = np.diff(vals[: i + 1])
    return {"peak": float(vals[i]),
            "index": i,
            "value_at_peak": float(records[i]["value"]),
            "post_peak_drop_percent": float(drops),
            "rising_monotone": bool(np.all(diffs >= -1e-12 * max(vals[i], 1e-30))),
            "n_records": len(records)}


def ledger_residuals(records) -> np.ndarray:
    """Relative energy-balance residual ``|W_ext - (U_el + W_frac)| / W_ext``."""
    out = []
    for r in records:
        work = abs(float(r["work"]))
        total = float(r["elastic"]) + float(r["fracture"])
        out.append(abs(work - total) / max(work, 1e-30))
    return np.asarray(out)


def damage_path(nodes: np.ndarray, d: np.ndarray, origin, direction, length: float,
                n_bands: int = 24, threshold: float = 0.5) -> dict:
    """Extract the crack path from a damage field.

    The path is sampled in ``n_bands`` slices along ``direction`` starting at
    ``origin``; in every slice the damage-weighted transverse coordinate of the
    nodes with ``d >= threshold`` gives one path point.  A straight, crack-path
    following the maximum-principal-stress direction keeps ``max |t|`` at the
    level of the damage band width ``l``.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    e1 = np.asarray(direction, dtype=np.float64)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-30)
    e2 = np.array([-e1[1], e1[0]])
    rel = nodes - np.asarray(origin, dtype=np.float64)
    s = rel @ e1
    t = rel @ e2
    edges = np.linspace(0.0, float(length), n_bands + 1)
    bands = []
    for k in range(n_bands):
        sel = (s >= edges[k]) & (s < edges[k + 1]) & (d >= threshold)
        if not np.any(sel):
            bands.append({"s": 0.5 * (edges[k] + edges[k + 1]), "t": None, "d_max": 0.0,
                          "count": 0})
            continue
        w = d[sel]
        bands.append({"s": float(0.5 * (edges[k] + edges[k + 1])),
                      "t": float((t[sel] * w).sum() / w.sum()),
                      "d_max": float(w.max()),
                      "count": int(sel.sum())})
    pop = [b for b in bands if b["t"] is not None]
    hot = np.where(d >= threshold)[0]
    tip_s = float(s[hot].max()) if hot.size else 0.0
    if pop:
        tt = np.array([b["t"] for b in pop])
        return {"bands": bands,
                "max_abs_t": float(np.abs(tt).max()),
                "mean_abs_t": float(np.abs(tt).mean()),
                "rms_t": float(math.sqrt(float((tt ** 2).mean()))),
                "filled_fraction": len(pop) / max(n_bands, 1),
                "tip_s": tip_s,
                "path": [[b["s"], b["t"]] for b in pop]}
    return {"bands": bands, "max_abs_t": float("nan"), "mean_abs_t": float("nan"),
            "rms_t": float("nan"), "filled_fraction": 0.0, "tip_s": tip_s, "path": []}


def fit_limit(x_values, values, power: float = 1.0) -> dict:
    """Least-squares limit ``value(l -> 0)`` of a monotone mesh-refinement series.

    Fits ``v(l) = v0 + C l^power`` and reports the extrapolated limit, the
    monotonicity of the raw series and the relative change between the finest
    point and the limit (the quantity the convergence criterion bounds).
    """
    x = np.asarray(x_values, dtype=np.float64) ** float(power)
    y = np.asarray(values, dtype=np.float64)
    A = np.vstack([np.ones_like(x), x]).T
    (v0, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    diffs = np.diff(y)
    monotone = bool(np.all(diffs > 0) or np.all(diffs < 0))
    finest_gap = 100.0 * abs(y[np.argmin(np.asarray(x_values))] - v0) / max(abs(v0), 1e-30)
    resid = y - (v0 + slope * x)
    return {"limit": float(v0), "slope": float(slope), "power": float(power),
            "monotone": monotone,
            "finest_gap_percent": float(finest_gap),
            "fit_residual_rms": float(np.sqrt((resid ** 2).mean()))}