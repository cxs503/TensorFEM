"""Grillage plastic collapse (limit analysis) under an ice patch + lateral pressure.

Why this module exists
----------------------
``tensorfem.grillage_ice_ultimate`` answers the "which member hits its limit
first" question with an **elastic** load share (Navier orthotropic modal
partition) plus limiting-stress member checks.  Its own docstring names the
consequence:

    "Load sharing is the *elastic* modal partition: once a member yields, load
    redistribution is not traced."

For a ductile grillage that limitation is the difference between a *member
check* and a *structural limit*: after the first plastic hinge the structure
keeps carrying load by redistribution until a collapse mechanism forms.  The
ice-belt assessment in the JOUR-ICE-STRUCT chain stops at the plate-panel
bending stress; the missing final jump (panel stress -> structural ultimate
strength with a safety margin) is what this module supplies, at the level of the
actual member layout and not of a smeared plate.

Method and lineage
------------------
Rigid-plastic limit analysis by the classical **method of successive plastic
hinge formation** (Neal, *The Plastic Methods of Structural Analysis*, 3rd ed.,
ch. 9 "Collapse of grids and beams"; Baker & Heyman 1969; the same hinge-by-hinge
procedure underlies the IACS CSR incremental-iterative collapse formulation and
Smith's 1977 progressive-collapse idealisation for stiffened panels):

1. the member line is a continuous Euler-Bernoulli beam over its real spans,
   discretised with 2-node (w, theta) bending elements and *consistent* nodal
   loads from the actual patch geometry;
2. a plastic hinge is an exact moment release at an element end (static
   condensation of the rotational DOF), which preserves shear transfer and lets
   the moment redistribute;
3. at each stage the structure is linear, so the load increment that brings the
   next section to its plastic moment is closed form; the section with the
   smallest increment is hinged and the process repeats;
4. when no section can reach the plastic moment with a positive load increment
   -- equivalently the released structure has lost its load path (singular
   tangent stiffness) -- a **collapse mechanism** exists and the load at that
   point is the limit load.

The plastic moment is reduced by the coexisting axial compression with the
standard full-section interaction ``M_p (1 - (P/P_y)^2)`` (rectangular
idealisation of a fully yielded section; see Neal ch. 2 and CSR Pt.1 Ch.5 for
the same reduction).  Second-order amplification of the *elastic* path is left
to ``grillage_ice_ultimate``; here the axial force enters only through the
plastic capacity, which is the accepted rigid-plastic convention.

Two orthogonal member systems
-----------------------------
The grillage is resolved into the two systems the plating actually feeds --
longitudinal stiffeners (x, continuous over the transverse frames) and
transverse frames (y, simply supported at the girders) -- and the applied ice
patch is split between them by the *same* orthotropic modal share used by
``grillage_ice_ultimate`` (``system_load_shares``), renormalised to
``f_x + f_y = 1``.  Both systems therefore see the same total patch force, and
the two modules are directly comparable: the difference between them is exactly
the elastic-share vs plastic-redistribution effect.  The union of the two
limiting mechanisms is reported as the grillage ultimate state, together with
the plate-field / beam-column limits already carried by
``grillage_ice_ultimate.ultimate_ice_pressure`` (which is the binding constraint
when the plate field is stocky).

Verification (``tensorfem.grillage_plastic_verification``)
---------------------------------------------------------
``V1`` single-span simply supported beam, uniform load: ``w_u = 8 M_p / L^2``
(exact plastic theory, 0.5 % gate).
``V2`` single-span beam, central patch ``c``: ``w_u = 8 M_p / c^2`` (1 % gate).
``V3`` clamped-clamped beam, uniform load: ``w_u = 16 M_p / L^2`` (1 % gate).
``V4`` two-span continuous beam, uniform load: ``w_u = 11.6569 M_p / L^2`` from
an independent virtual-work minimisation over the hinge position
(``two_span_collapse_upper_bound``), 2 % gate.
``V5`` elastic path against closed form (midspan moment and deflection, 0.2 %).
``V6`` axial interaction: the reduced plastic moment reproduces
``M_p (1 - (P/P_y)^2)`` and the collapse load scales with it.
``V7`` mesh convergence in the number of elements per span.
``V8`` patch-size and coexisting-compression sensitivity of the case study.

Scope and limitations (explicit)
--------------------------------
* Ideal rigid-plastic material, monotonic proportional ice loading on top of a
  dead hydrostatic load; no unloading, no strain hardening, no cyclic
  (ratcheting) path -- the ice-belt ratcheting question is a separate module.
* Hinges can only form at element boundaries, i.e. at multiples of
  ``span / elements_per_span``; the mesh convergence study bounds that error.
* Plate buckling between stiffeners, web local buckling / tripping, frame
  tripping, welding residual stress and initial deflections are *not* in the
  collapse mechanism; they enter only through the plate-field limit taken from
  ``grillage_ice_ultimate`` in the combined result and through the elastic
  checks there.
* The patch is quasi-static and uniform over its rectangle; it is placed at the
  centre of the representative grillage (the worst span for the member lines
  modelled here).
* The frame lines are taken as rigid vertical supports for the stiffener system
  and the girder lines as rigid supports for the frame system; the elastic
  support flexibility of the girder is not modelled.
* No rule table is transcribed: ``design_pressure`` arrives with a provenance
  string and the margin is ``p_ult / design_pressure``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    MemberSection,
    case_study_geometry,
    plate_patch_response,
    stiffener_axial_capacity,
    system_load_shares,
    ultimate_ice_pressure,
)

__all__ = [
    "MemberLine",
    "member_collapse_pressure",
    "grillage_plastic_collapse",
    "elastic_reference_beam",
    "two_span_collapse_upper_bound",
    "write_plastic_collapse_report",
    "case_study_load_case",
]


# --------------------------------------------------------------------------- #
# validation helpers (house style)
# --------------------------------------------------------------------------- #
def _positive(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return v


def _non_negative(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


# --------------------------------------------------------------------------- #
# member line
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemberLine:
    """One continuous member line of the grillage, over its real spans.

    ``span_lengths`` are the distances between the rigid supports of the line
    (transverse-frame spacing for a stiffener line, frame span for a frame line).
    ``section`` must include the attached plating (``GrillageGeometry`` builds
    those with ``stiffener_section`` / ``frame_section``).  ``axial_force`` is
    the coexisting compression in the member (positive).
    """

    name: str
    span_lengths: Sequence[float]
    section: MemberSection
    yield_stress: float
    young: float = 2.06e11
    axial_force: float = 0.0
    elements_per_span: int = 4
    fixed_ends: bool = False
    """Clamp the end rotations (built-in ends); a plastic hinge releases them."""

    def __post_init__(self) -> None:
        spans = tuple(_positive(f"span_lengths[{i}]", s)
                      for i, s in enumerate(self.span_lengths))
        if not spans:
            raise ValueError("a member line needs at least one span")
        object.__setattr__(self, "span_lengths", spans)
        self.section.validate()
        _positive("yield_stress", self.yield_stress)
        _positive("young", self.young)
        _non_negative("axial_force", self.axial_force)
        if int(self.elements_per_span) < 2:
            raise ValueError("elements_per_span must be >= 2 (a midspan hinge must exist)")
        object.__setattr__(self, "elements_per_span", int(self.elements_per_span))
        if self.axial_force >= self.axial_yield:
            raise ValueError("axial_force must stay below the squash load")

    @property
    def total_length(self) -> float:
        return float(sum(self.span_lengths))

    @property
    def area(self) -> float:
        return self.section.area

    @property
    def axial_yield(self) -> float:
        return self.section.area * self.yield_stress

    @property
    def plastic_moment_full(self) -> float:
        """``M_p = Z_p sigma_y`` (N m)."""
        return self.section.plastic_modulus * self.yield_stress

    @property
    def axial_reduction(self) -> float:
        """``1 - (P/P_y)^2`` of the full-section plastic interaction."""
        return 1.0 - (self.axial_force / self.axial_yield) ** 2

    @property
    def plastic_moment(self) -> float:
        """Plastic moment reduced by the coexisting axial compression (N m)."""
        return self.plastic_moment_full * self.axial_reduction

    @property
    def column_slenderness(self) -> float:
        """``lambda = (a / (pi r)) sqrt(sigma_y / E)`` on the longest span."""
        inertia = self.section.inertia
        area = self.section.area
        r = math.sqrt(inertia / area)
        a = max(self.span_lengths)
        return a / (math.pi * r) * math.sqrt(self.yield_stress / self.young)

    def support_positions(self) -> tuple[float, ...]:
        xs = [0.0]
        for s in self.span_lengths:
            xs.append(xs[-1] + s)
        return tuple(xs)


# --------------------------------------------------------------------------- #
# beam assembly with exact moment releases
# --------------------------------------------------------------------------- #
class _Mechanism(Exception):
    """Raised when the released structure has lost its load path."""


class _BeamSystem:
    """2-node Euler-Bernoulli beam line with exact plastic hinges at the nodes.

    Each element carries its **own** rotational DOF at each end; the two ends of
    two neighbouring elements are merged into one shared rotation DOF unless a
    plastic hinge sits between them.  This is the only correct way to represent
    a plastic hinge in a beam FE: the hinge is a pair of independent rotations
    joined by a *prescribed* couple ``M_p`` (applied as equal-and-opposite nodal
    moments), so the section keeps transmitting its plastic moment while the
    relative rotation is free.  Merging the rotations and merely releasing one
    element end would drop the node's moment equilibrium and make the section
    moment vanish.

    Section moment convention: the physical bending moment equals ``+m`` for an
    element-start end and ``-m`` for an element-end end, ``m`` being the nodal
    moment conjugate to that end's rotation.
    """

    def __init__(self, member: MemberLine) -> None:
        self.member = member
        self.ei = member.young * member.section.inertia
        lengths: list[float] = []
        for span in member.span_lengths:
            n = member.elements_per_span
            lengths.extend([span / n] * n)
        self.lengths = lengths
        xs = [0.0]
        for L in lengths:
            xs.append(xs[-1] + L)
        self.xs = np.asarray(xs, dtype=float)
        self.n_nodes = len(self.xs)
        self.n_elem = len(lengths)
        self.supports = {int(np.argmin(np.abs(self.xs - s)))
                         for s in member.support_positions()}
        self.clamped = ({0, self.n_nodes - 1} if member.fixed_ends else set())
        self.scale = max(self.ei / L ** 3 for L in lengths)

    # -- element matrices -------------------------------------------------
    def _bending(self, L: float) -> np.ndarray:
        e = self.ei / L ** 3
        return e * np.array([
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2],
        ])

    def incident(self, node: int) -> list[tuple[int, int]]:
        """Element ends attached to ``node``: ``(element, end)`` pairs."""
        out: list[tuple[int, int]] = []
        if node > 0:
            out.append((node - 1, 1))
        if node < self.n_elem:
            out.append((node, 0))
        return out

    # -- DOF bookkeeping --------------------------------------------------
    def dof_map(self, hinges: Mapping[int, float]) -> tuple[dict, int]:
        """Global DOF of every ``(node)`` / ``(element, end)`` pair.

        Returns ``(map, n_dof)``.  ``map[("w", node)]`` is the translation DOF and
        ``map[("r", element, end)]`` the rotation DOF; rotations are merged across
        a node unless a hinge sits there.
        """
        parent: dict[tuple[int, int], tuple[int, int]] = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        ends = [(e, end) for e in range(self.n_elem) for end in (0, 1)]
        for key in ends:
            parent[key] = key
        for n in range(self.n_nodes):
            if n in hinges:
                continue
            attached = self.incident(n)
            for other in attached[1:]:
                union(attached[0], other)
        dof: dict = {}
        nxt = self.n_nodes
        roots: dict[tuple[int, int], int] = {}
        for key in ends:
            root = find(key)
            if root not in roots:
                roots[root] = nxt
                nxt += 1
            dof[("r", key[0], key[1])] = roots[root]
        for n in range(self.n_nodes):
            dof[("w", n)] = n
        return dof, nxt

    # -- assembly ---------------------------------------------------------
    def assemble(self, hinges: Mapping[int, float]
                 ) -> tuple[np.ndarray, np.ndarray, dict, int]:
        """Stiffness, hinge couples, DOF map and DOF count."""
        dof, n_dof = self.dof_map(hinges)
        K = np.zeros((n_dof, n_dof))
        f_h = np.zeros(n_dof)
        for e, L in enumerate(self.lengths):
            Ke = self._bending(L)
            idx = [dof[("w", e)], dof[("r", e, 0)],
                   dof[("w", e + 1)], dof[("r", e, 1)]]
            K[np.ix_(idx, idx)] += Ke
        # hinge couples: equal and opposite nodal moments on the two sides
        for node, moment in hinges.items():
            for (e, end) in self.incident(node):
                # internal nodal moment must equal the applied couple, and the
                # physical section moment is +m at an element start, -m at an end
                f_h[dof[("r", e, end)]] += moment if end == 0 else -moment
        # boundary conditions
        for n in sorted(self.supports):
            j = dof[("w", n)]
            K[j, :] = 0.0
            K[:, j] = 0.0
            K[j, j] = self.scale
            f_h[j] = 0.0
        for n in sorted(self.clamped - set(hinges)):
            for (e, end) in self.incident(n):
                j = dof[("r", e, end)]
                if K[j, :].any() or K[:, j].any():
                    K[j, :] = 0.0
                    K[:, j] = 0.0
                    K[j, j] = self.scale
                    f_h[j] = 0.0
        # orphaned rotations (both ends hinged) must not make the system singular
        for n in range(self.n_nodes):
            for (e, end) in self.incident(n):
                j = dof[("r", e, end)]
                if not (K[j, :].any() or K[:, j].any()):
                    K[j, j] = self.scale
                    f_h[j] = 0.0
        return K, f_h, dof, n_dof

    # -- loads ------------------------------------------------------------
    def _shape(self, xi: float, L: float) -> np.ndarray:
        return np.array([
            1.0 - 3.0 * xi ** 2 + 2.0 * xi ** 3,
            L * (xi - 2.0 * xi ** 2 + xi ** 3),
            3.0 * xi ** 2 - 2.0 * xi ** 3,
            L * (-xi ** 2 + xi ** 3),
        ])

    def load_vector(self, q: Callable[[float], float],
                    breakpoints: Sequence[float] = ()) -> np.ndarray:
        """Consistent nodal loads of the piecewise-constant line load ``q``.

        Returned in the *canonical* numbering -- translations ``n`` and one
        rotation per element end ``n_nodes + 2 e + end`` -- so the vector can be
        mapped onto any hinge pattern without mixing the two sides of a hinge.
        """
        f = np.zeros(self.n_nodes + 2 * self.n_elem)
        gx, gw = np.polynomial.legendre.leggauss(3)
        for e, L in enumerate(self.lengths):
            x_lo, x_hi = self.xs[e], self.xs[e + 1]
            cuts = sorted({float(x_lo), float(x_hi)}
                          | {b for b in breakpoints if x_lo < b < x_hi})
            fe = np.zeros(4)
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b - a <= 0.0:
                    continue
                mid, half = 0.5 * (a + b), 0.5 * (b - a)
                for g, w in zip(gx, gw):
                    x = mid + half * g
                    fe += self._shape((x - x_lo) / L, L) * q(float(x)) * w * half
            f[[e, self.n_nodes + 2 * e, e + 1, self.n_nodes + 2 * e + 1]] += fe
        for n in self.supports:
            f[n] = 0.0
        return f

    # -- solution ---------------------------------------------------------
    def solve(self, hinges: Mapping[int, float], f: np.ndarray,
              couples: bool = True) -> np.ndarray:
        """Solve the current (hinged) structure under the canonical load ``f``.

        ``couples=True`` includes the plastic-hinge couples (the *total* state);
        ``couples=False`` is the response to ``f`` alone (the unit-load
        increment), so that ``M(p) = M_dead + p M_unit`` stays linear.
        """
        dof, n_dof = self.dof_map(hinges)
        K, f_h = self.assemble(hinges)[:2]
        rhs = f_h.copy() if couples else np.zeros(n_dof)
        for n in range(self.n_nodes):
            rhs[dof[("w", n)]] += f[n]
        for e in range(self.n_elem):
            for end in (0, 1):
                rhs[dof[("r", e, end)]] += f[self.n_nodes + 2 * e + end]
        for n in self.supports:
            rhs[dof[("w", n)]] = 0.0
        if np.linalg.cond(K) > 1e13:
            raise _Mechanism
        try:
            u = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise _Mechanism from exc
        if not np.all(np.isfinite(u)):
            raise _Mechanism
        return u

    # -- sections ---------------------------------------------------------
    def section_moment(self, hinges: Mapping[int, float], u: np.ndarray,
                       node: int) -> float:
        """Physical bending moment (N m) at ``node`` for the current state."""
        if node in hinges:
            return float(hinges[node])
        dof, _ = self.dof_map(hinges)
        for (e, end) in self.incident(node):
            L = self.lengths[e]
            wi = u[dof[("w", e)]]
            ti = u[dof[("r", e, 0)]]
            wj = u[dof[("w", e + 1)]]
            tj = u[dof[("r", e, 1)]]
            if end == 0:
                m = self.ei / L ** 2 * (6.0 * wi + 4.0 * L * ti
                                        - 6.0 * wj + 2.0 * L * tj)
                return float(m)
            m = self.ei / L ** 2 * (6.0 * wi + 2.0 * L * ti
                                    - 6.0 * wj + 4.0 * L * tj)
            return float(-m)
        return 0.0

    def section_moments(self, hinges: Mapping[int, float],
                        u: np.ndarray) -> dict[int, float]:
        return {n: self.section_moment(hinges, u, n) for n in range(self.n_nodes)}

    def deflection(self, hinges: Mapping[int, float], u: np.ndarray) -> np.ndarray:
        dof, _ = self.dof_map(hinges)
        return np.array([u[dof[("w", n)]] for n in range(self.n_nodes)])


# --------------------------------------------------------------------------- #
# collapse of one member line
# --------------------------------------------------------------------------- #
def member_collapse_pressure(
    member: MemberLine,
    *,
    ice_tributary: float,
    ice_interval: tuple[float, float],
    hydro_pressure: float = 0.0,
    hydro_tributary: float = 0.0,
    max_hinges: int = 80,
    ratio_tol: float = 1e-10,
) -> dict[str, object]:
    """Limit ice pressure of one member line by successive hinge formation.

    The hydrostatic pressure is a *dead* load; the ice pressure is then raised
    monotonically.  ``ice_tributary``/``hydro_tributary`` are the widths (m) over
    which the respective pressures reach the member, so the line loads are
    ``p * tributary``.  Returns a JSON-ready dict with the limit pressure, the
    hinge sequence, the first-hinge pressure and the plastic reserve ratio
    ``p_ult / p_first_hinge`` (the redistribution reserve of the member line).
    """
    _positive("ice_tributary", ice_tributary)
    _non_negative("hydro_pressure", hydro_pressure)
    _non_negative("hydro_tributary", hydro_tributary)
    x0, x1 = (float(v) for v in ice_interval)
    if not (x1 > x0):
        raise ValueError("ice_interval must satisfy x1 > x0")
    system = _BeamSystem(member)
    mp = member.plastic_moment
    if mp <= 0.0:  # pragma: no cover - guarded by MemberLine
        raise ValueError("the reduced plastic moment vanished (axial_force too high)")

    f_dead = system.load_vector(lambda x: hydro_pressure * hydro_tributary)
    f_ice = system.load_vector(lambda x: ice_tributary if x0 <= x <= x1 else 0.0,
                               breakpoints=(x0, x1))
    hinges: dict[int, float] = {}
    sequence: list[dict[str, object]] = []
    pressure = 0.0
    first_hinge: float | None = None
    mechanism = False
    hydro_moment = 0.0
    for _ in range(int(max_hinges)):
        try:
            u_dead = system.solve(hinges, f_dead)
            u_ice = system.solve(hinges, f_ice, couples=False)
        except _Mechanism:
            mechanism = True
            break
        m_dead = system.section_moments(hinges, u_dead)
        m_ice = system.section_moments(hinges, u_ice)
        if not sequence:
            hydro_moment = max(abs(v) for v in m_dead.values())
        best: tuple[float, int] | None = None
        for n in range(system.n_nodes):
            if n in hinges:
                continue
            base, dm = m_dead[n], m_ice[n]
            if abs(dm) < 1e-30:
                continue
            dp = (mp - base) / dm if dm > 0.0 else (-mp - base) / dm
            if dp > ratio_tol and (best is None or dp < best[0]):
                best = (dp, n)
        if best is None:
            mechanism = True
            break
        dp, node = best
        # with the hinge set fixed the moment field is linear in the ice
        # pressure, so ``dp`` is the *total* pressure at which the next section
        # reaches M_p -- not an increment on top of the previous hinge pressure
        if first_hinge is None:
            first_hinge = dp
        pressure = dp
        section_moment = m_dead[node] + dp * m_ice[node]
        hinges[node] = float(section_moment)
        sequence.append({
            "node": int(node),
            "x": float(system.xs[node]),
            "section_moment": float(section_moment),
            "ice_pressure": float(pressure),
        })

    return {
        "member": member.name,
        "ice_pressure_ultimate": float(pressure),
        "ice_pressure_first_hinge": None if first_hinge is None else float(first_hinge),
        "plastic_reserve_ratio": (pressure / first_hinge) if first_hinge else None,
        "mechanism": mechanism,
        "hinges": sequence,
        "plastic_moment": mp,
        "plastic_moment_full": member.plastic_moment_full,
        "axial_reduction": member.axial_reduction,
        "hydro_moment_max": hydro_moment,
        "elements_per_span": member.elements_per_span,
        "fixed_ends": bool(member.fixed_ends),
        "n_elements": system.n_elem,
        "ice_interval": [x0, x1],
        "ice_tributary": ice_tributary,
        "hydro_tributary": hydro_tributary,
        "hydro_pressure": hydro_pressure,
    }


def _overlap_members(extent: float, spacing: float, n_members: int) -> int:
    """Number of parallel members a patch of ``extent`` touches (>= 2)."""
    return int(min(max(2, int(round(extent / spacing)) + 1), n_members))


def grillage_plastic_collapse(geom: GrillageGeometry,
                             case: CombinedLoadCase,
                             *, elements_per_span: int = 4,
                             terms: int = 61) -> dict[str, object]:
    """Grillage ultimate ice pressure from the two limiting mechanisms.

    The ice patch force is split between the longitudinal-stiffener system and
    the transverse-frame system with the orthotropic modal share of
    ``grillage_ice_ultimate`` (renormalised to sum to one) and each system is
    taken to its own collapse mechanism.  The grillage collapse pressure is the
    smaller of the two; the combined result also carries the plate-field /
    beam-column limit from ``ultimate_ice_pressure`` so the reported ultimate is
    the governing one over all three limit states.
    """
    geom.validate()
    case.validate(geom)
    if case.patch_length <= 0.0 or case.patch_width <= 0.0:
        raise ValueError("grillage_plastic_collapse needs a positive patch size")
    lx, ly = case.patch_extents(geom)
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    dx, dy, d3 = geom.plate_rigidities()
    patch = (0.5 - 0.5 * lx / span_x, 0.5 + 0.5 * lx / span_x,
             0.5 - 0.5 * ly / span_y, 0.5 + 0.5 * ly / span_y)

    ice_share = plate_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                                     d3=d3, pressure=1.0, patch=patch,
                                     terms=terms)
    f_x, f_y = system_load_shares(ice_share)
    if case.hydrostatic_pressure > 0.0:
        hydro_share = plate_patch_response(span_x=span_x, span_y=span_y, d1=dx,
                                          d2=dy, d3=d3, pressure=1.0,
                                          patch=(0.0, 1.0, 0.0, 1.0), terms=terms)
        f_x_h, f_y_h = system_load_shares(hydro_share)
    else:
        f_x_h = f_y_h = 0.0

    cap = stiffener_axial_capacity(geom)
    stiff_axial = case.axial_stress * geom.stiffener_section.area
    n_over_x = _overlap_members(ly, geom.bay_spacing, geom.n_stiffeners)
    n_over_y = _overlap_members(lx, geom.frame_spacing, geom.n_frames)

    x_member = MemberLine(
        name="longitudinal_stiffener",
        span_lengths=[geom.frame_spacing] * int(geom.n_spans),
        section=geom.stiffener_section,
        yield_stress=geom.yield_stress,
        young=geom.young,
        axial_force=stiff_axial,
        elements_per_span=elements_per_span,
    )
    y_member = MemberLine(
        name="transverse_frame",
        span_lengths=[span_y],
        section=geom.frame_section,
        yield_stress=geom.yield_stress,
        young=geom.young,
        axial_force=0.0,
        elements_per_span=max(4, int(elements_per_span)),
    )

    x_patch = (0.5 * (span_x - lx), 0.5 * (span_x + lx))
    y_patch = (0.5 * (span_y - ly), 0.5 * (span_y + ly))

    x_res = member_collapse_pressure(
        x_member,
        ice_tributary=f_x * ly / n_over_x,
        ice_interval=x_patch,
        hydro_pressure=case.hydrostatic_pressure,
        hydro_tributary=f_x_h * geom.bay_spacing,
    )
    y_res = member_collapse_pressure(
        y_member,
        ice_tributary=f_y * lx / n_over_y,
        ice_interval=y_patch,
        hydro_pressure=case.hydrostatic_pressure,
        hydro_tributary=f_y_h * geom.frame_spacing,
    )

    systems = {x_res["member"]: x_res, y_res["member"]: y_res}
    governing_system = min(systems, key=lambda k: systems[k]["ice_pressure_ultimate"])
    p_plastic = float(systems[governing_system]["ice_pressure_ultimate"])

    panel = ultimate_ice_pressure(geom, case, terms=terms)
    p_panel = float(panel["ice_pressure_ultimate"])
    p_ult = min(p_plastic, p_panel)
    governing_mode = ("grillage_plastic_collapse"
                      if p_plastic <= p_panel else str(panel["governing_member"]))

    beta = geom.plate_slenderness
    alpha_x = geom.frame_spacing / geom.bay_spacing
    return {
        "ice_pressure_ultimate": p_ult,
        "ice_pressure_grillage_plastic": p_plastic,
        "ice_pressure_panel_limit": p_panel,
        "governing_mode": governing_mode,
        "governing_member_line": governing_system,
        "systems": systems,
        "load_shares": {"x": f_x, "y": f_y,
                        "hydro_x": f_x_h, "hydro_y": f_y_h,
                        "modal_x": ice_share["load_share_x"],
                        "modal_y": ice_share["load_share_y"],
                        "modal_torsion": ice_share["load_share_torsion"]},
        "overlap_members": {"x_system": n_over_x, "y_system": n_over_y},
        "patch": {"length": lx, "width": ly,
                  "area": lx * ly,
                  "x_fraction": lx / span_x, "y_fraction": ly / span_y},
        "dimensionless": {
            "plate_slenderness_beta": beta,
            "stiffener_column_slenderness": x_member.column_slenderness,
            "frame_column_slenderness": y_member.column_slenderness,
            "frame_spacing_over_bay": alpha_x,
            "stiffener_plastic_moment": x_member.plastic_moment,
            "frame_plastic_moment": y_member.plastic_moment,
            "axial_stress_ratio": case.axial_stress / geom.yield_stress,
            "axial_reduction_stiffener": x_member.axial_reduction,
            "plate_strength_ratio_faulkner": geom.faulkner_plate_strength,
            "panel_ultimate_stress": cap["panel_ultimate_stress"],
            "hydro_line_load": case.hydrostatic_pressure * (
                f_x_h * geom.bay_spacing if governing_system == x_res["member"]
                else f_y_h * geom.frame_spacing),
            "hydro_over_collapse_line_load": (
                case.hydrostatic_pressure
                * (f_x_h * geom.bay_spacing if governing_system == x_res["member"]
                   else f_y_h * geom.frame_spacing)
                / max(p_plastic * float(systems[governing_system]["ice_tributary"]), 1e-30)),
        },
        "design_pressure": case.design_pressure,
        "design_provenance": case.design_provenance,
        "margin_vs_design": ((p_ult / case.design_pressure)
                             if case.design_pressure else None),
        "elements_per_span": int(elements_per_span),
        "terms": int(terms),
    }


def case_study_load_case(*, ice_pressure: float = 6.0e6,
                         axial_stress: float = 0.3 * 3.55e8,
                         hydrostatic_pressure: float = 1.0e5,
                         design_pressure: float = 4.0e6,
                         design_provenance: str = "caller supplied design value"
                         ) -> CombinedLoadCase:
    """Representative combined ice + hydrostatic + hull-girder load case.

    Patch ``1.05 m x 1.05 m`` (half a frame spacing square, an ice-class
    pressure-area style patch on the case-study grillage), coexisting hull-girder
    compression ``0.30 sigma_y`` and ``0.1 MPa`` hydrostatic head.  The design
    pressure is a *caller-supplied* placeholder: no rule table is transcribed.
    """
    geom = case_study_geometry()
    return CombinedLoadCase(
        axial_stress=axial_stress,
        ice_pressure=ice_pressure,
        patch_length=0.5 * geom.frame_spacing,
        patch_width=3.0 * geom.bay_spacing,
        hydrostatic_pressure=hydrostatic_pressure,
        design_pressure=design_pressure,
        design_provenance=design_provenance,
    ).validate(geom)


# --------------------------------------------------------------------------- #
# independent reference solutions (used by the verification module)
# --------------------------------------------------------------------------- #
def elastic_reference_beam(*, span: float, section: MemberSection,
                          yield_stress: float, load: float = 1.0) -> dict[str, float]:
    """Closed-form SSSS beam under a uniform load (independent of the FE)."""
    ei = section.young * section.inertia
    return {
        "moment_mid": load * span ** 2 / 8.0,
        "deflection_mid": 5.0 * load * span ** 4 / (384.0 * ei),
        "first_hinge_load": 8.0 * section.plastic_modulus * yield_stress / span ** 2,
    }


def two_span_collapse_upper_bound(*, span: float, plastic_moment: float,
                                  samples: int = 200001) -> dict[str, float]:
    """Exact plastic-collapse load of a two-equal-span beam under uniform load.

    Virtual-work upper bound for the combined mechanism (hinges at the interior
    support and at ``a`` from each outer support), minimised over ``a``:
    ``w_u = 2 M_p (L + a) / ((L - a) a L)``.  The minimum sits at
    ``a/L = sqrt(2) - 1`` giving ``w_u = 11.65685 M_p / L^2``; the closed form is
    evaluated here and the discrete scan is returned as an independent check.
    """
    _positive("span", span)
    _positive("plastic_moment", plastic_moment)
    u = math.sqrt(2.0) - 1.0
    exact = 2.0 * plastic_moment / (u * (1.0 - u) / (1.0 + u) * span ** 2)
    us = np.linspace(1e-4, 1.0 - 1e-4, int(samples))
    w = 2.0 * plastic_moment * (1.0 + us) / ((1.0 - us) * us * span ** 2)
    return {
        "collapse_load_exact": exact,
        "collapse_load_scan": float(np.min(w)),
        "hinge_position_ratio": u,
    }


# --------------------------------------------------------------------------- #
# report writer
# --------------------------------------------------------------------------- #
def write_plastic_collapse_report(out_dir: str | Path,
                                  result: Mapping[str, object]) -> Path:
    """Write the collapse result (and the case-study sweep) as JSON."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "grillage_plastic_collapse.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    return out