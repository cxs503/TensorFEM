"""Anchor-free closure operators for plate/shell buckling coefficients.

Why this module exists
----------------------
A buckling *coefficient* (``k`` in ``sigma_cr = k pi^2 D / (b^2 t)``) and a
member *reduction curve* (``chi(lambdabar)``) are the two numbers that ship a
plate/shell buckling check into a design report.  Both are normally taken from a
code or a handbook, i.e. they are *transcribed* values, and a transcription is
only evidence when the source text can be quoted verbatim at a known location.

This module therefore keeps apart the two things a "coefficient register" tends
to mix:

* **anchor transcription** -- never done here.  An anchor (Timoshenko & Gere
  plate coefficients, DNV-RP-C201 plate buckling factors, EN 1993-1-6 shell
  reduction curves) can only be registered with a verbatim quote plus a
  location; when the source text is not available in the environment the entry
  is registered as ``WITHHELD`` and carries *no* number (:func:`anchor_entry`).
  Nothing here reconstructs a code value from memory or from a secondary
  transcription.
* **operator closure** -- what can be established without any anchor, and what
  this module implements:

  1. the coefficient ``k`` is *derived* from the governing plate equation by two
     routes that share no code (a closed-form single-mode minimisation and the
     independent full-series Navier solver in
     :mod:`tensorfem.marine_plate_buckling`), so the residual between them
     measures operator consistency at machine precision
     (:func:`coefficient_closure`);
  2. the same coefficient is recomputed by two *discretisations* that share no
     approximation space -- a Bogner--Fox--Schmit C1 Hermite element
     (:func:`bfs_uniaxial_coefficient`) and a clamped-admissible polynomial
     Rayleigh--Ritz basis (:func:`ritz_uniaxial_coefficient`) -- for the
     boundary conditions that have no closed form (clamped edges and edge
     rotational restraint).  Those two agree only within their discretisation
     error, so they are reported with a convergence ladder and are never
     claimed at ``1e-10``;
  3. the influence of edge rotational restraint is a *continuous* ladder from
     the simply supported coefficient to the clamped one
     (:func:`restraint_ladder_uniaxial`), a property of the operator family that
     needs no literature value;
  4. a reduction curve ``chi(lambdabar)`` is checked against structural clauses
     that hold for *any* admissible curve (:func:`check_reduction_curve`):
     dimensionless argument, range ``0 < chi <= 1``, monotone non-increasing,
     continuous, and the **elastic ceiling** ``chi <= 1 / lambdabar^2`` (no
     reduction curve may return a capacity above the elastic critical load).
     Code-specific clauses (stocky plateau length, imperfection factor, plastic
     branch) are *parameters* of the checker; when the anchor is withheld the
     parameter is ``None`` and the clause is reported as skipped, not guessed.

Nothing in this module mutates solver state; it assembles and diagonalises small
dense matrices with numpy (a declared dev dependency, imported by the existing
:mod:`tensorfem.plate_buckling_hermite` module the same way).  Every number it
produces is operator-level evidence: it is *not* validated against an
experiment, a code or a report while the anchors are withheld.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .marine_plate_buckling import isotropic_plate_rigidities, navier_uniaxial_buckling_load
from .plate_buckling_hermite import bfs_plate_element_matrices, hermite_cubic_1d

__all__ = [
    "ANCHOR_REQUESTS",
    "RestraintLadder",
    "anchor_entry",
    "bfs_uniaxial_coefficient",
    "check_reduction_curve",
    "closed_form_uniaxial_coefficient",
    "coefficient_closure",
    "coefficient_from_critical_stress",
    "critical_stress_from_coefficient",
    "elastic_ceiling_violations",
    "navier_biaxial_coefficient",
    "navier_uniaxial_coefficient",
    "register_digest",
    "restraint_ladder_uniaxial",
    "ritz_uniaxial_coefficient",
    "slenderness",
    "synthetic_reduction_specimen",
]


def _gauss_legendre(count: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return 0.5 * (nodes + 1.0), 0.5 * weights


#: Four-point rule on ``[0, 1]``: exact to degree 7, enough for cubic Hermite
#: products (degree 6) in the geometric and edge matrices.
_GAUSS04, _GAUSS04_W = _gauss_legendre(4)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _poisson(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("poisson must be a real number")
    value = float(value)
    if not math.isfinite(value) or not -1.0 < value < 0.5:
        raise ValueError("poisson must be finite and lie between -1 and 0.5")
    return value


def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _material(young: float, poisson: float, thickness: float) -> tuple[float, float]:
    young = _positive("young", young)
    _poisson(poisson)
    thickness = _positive("thickness", thickness)
    rigidity, _, _ = isotropic_plate_rigidities(young, poisson, thickness)
    return thickness, rigidity


# ---------------------------------------------------------------------------
# 1. coefficient <-> critical stress operator (the only shared algebra)
# ---------------------------------------------------------------------------
def critical_stress_from_coefficient(
    *, coefficient: float, young: float, poisson: float, thickness: float, width: float
) -> float:
    """Return ``sigma_cr = k pi^2 D / (b^2 t)`` in Pa.

    ``width`` is the plate span entering the coefficient normalisation (``b``)
    and ``t`` the plate thickness, with ``D = E t^3 / (12 (1 - nu^2))``.  This is
    the normalisation used by the classical plate literature and by the Navier
    route in :mod:`tensorfem.marine_plate_buckling`, which reports a line load
    ``N_cr`` with ``sigma_cr = N_cr / t``.
    """
    coefficient = _positive("coefficient", coefficient)
    width = _positive("width", width)
    thickness, rigidity = _material(young, poisson, thickness)
    return coefficient * math.pi**2 * rigidity / (width**2 * thickness)


def coefficient_from_critical_stress(
    *, critical_stress: float, young: float, poisson: float, thickness: float, width: float
) -> float:
    """Inverse of :func:`critical_stress_from_coefficient` (round-trip partner)."""
    critical_stress = _positive("critical_stress", critical_stress)
    width = _positive("width", width)
    thickness, rigidity = _material(young, poisson, thickness)
    return critical_stress * width**2 * thickness / (math.pi**2 * rigidity)


def slenderness(*, yield_stress: float, critical_stress: float) -> float:
    """Dimensionless plate slenderness ``lambdabar = sqrt(fy / sigma_cr)``.

    A definition (the square root of a stress ratio), not a code value: it is the
    argument every reduction curve consumes, and its dimensionless guard is
    checkable without an anchor.
    """
    yield_stress = _positive("yield_stress", yield_stress)
    critical_stress = _positive("critical_stress", critical_stress)
    return math.sqrt(yield_stress / critical_stress)


# ---------------------------------------------------------------------------
# 2. two independent routes to the simply supported uniaxial coefficient
# ---------------------------------------------------------------------------
def closed_form_uniaxial_coefficient(
    *, aspect_ratio: float, max_half_waves: int = 64
) -> tuple[float, int, int]:
    """Return ``(k, m, n)`` for a simply supported plate under uniaxial load.

    Derivation (no literature value): the Navier series solution of
    ``D lap lap w + N_x w_,xx = 0`` gives, for half-wave counts ``m`` (along the
    loaded span) and ``n`` (transverse),

        N_x(m, n) = pi^2 D / b^2 * (m^2 / alpha^2 + n^2)^2 / (m^2 / alpha^2)

    hence ``k(m, n) = (u + n^2 / u)^2`` with ``u = m / alpha``,
    ``alpha = a / b``.  ``n = 1`` minimises that expression for every ``m`` (it
    enters like ``n^4``), which leaves the single-mode envelope
    ``k(alpha) = min_m (m / alpha + alpha / m)^2``.  For ``alpha >= 1`` the
    minimising mode satisfies ``m <= alpha``, so the envelope lies in ``[4, 4.5]``
    with the global minimum ``k = 4`` at integer ``alpha`` and the maximum
    ``k = 4.5`` at ``alpha = sqrt(2)``; for ``alpha < 1`` no integer mode is
    available below ``u = 1`` and the envelope exceeds ``4.5`` (a plate shorter
    than it is wide is stiffer in this normalisation).  Both facts follow from
    the algebra above and are asserted in the tests, not read from a table.
    """
    alpha = _positive("aspect_ratio", aspect_ratio)
    limit = _positive_int("max_half_waves", max_half_waves)
    best = (math.inf, 0, 1)
    for m in range(1, limit + 1):
        u = m / alpha
        value = (u + 1.0 / u) ** 2
        if value < best[0]:
            best = (value, m, 1)
    if best[1] == limit:
        raise ValueError("max_half_waves too small: minimising mode at the enumeration edge")
    return best


def navier_uniaxial_coefficient(
    *,
    length: float,
    width: float,
    young: float,
    poisson: float,
    thickness: float,
    max_half_waves: int = 64,
) -> tuple[float, int, int]:
    """Return ``(k, m, n)`` from the repo's independent full-series Navier solver.

    That solver enumerates ``(m, n)`` on a two-dimensional grid in *dimensional*
    rigidities and returns a line load; here only the normalisation
    ``k = N_cr b^2 / (pi^2 D)`` is applied.  Sharing the PDE but not the code
    path is what makes the pair a closure check rather than a tautology.
    """
    length = _positive("length", length)
    width = _positive("width", width)
    _, rigidity = _material(young, poisson, thickness)
    load, m, n = navier_uniaxial_buckling_load(
        length=length,
        width=width,
        rigidity_x=rigidity,
        rigidity_y=rigidity,
        coupling_rigidity=rigidity,
        maximum_half_waves=_positive_int("max_half_waves", max_half_waves),
    )
    return load * width**2 / (math.pi**2 * rigidity), m, n


def navier_biaxial_coefficient(
    *, aspect_ratio: float, load_ratio: float = 1.0, max_half_waves: int = 64
) -> tuple[float, int, int]:
    """Return ``(k, m, n)`` for simply supported biaxial compression.

    ``load_ratio`` is ``psi = N_y / N_x``.  The same Navier series gives

        N_x(m, n) = pi^2 D/b^2 * (m^2/alpha^2 + n^2)^2 / (m^2/alpha^2 + psi n^2),
        k(m, n)   = (m^2/alpha^2 + n^2)^2 / (m^2/alpha^2 + psi n^2),

    which reduces to the uniaxial envelope at ``psi = 0`` and gives ``k = 2`` for
    ``psi = 1`` on a square plate (``m = n = 1``).  It exists to show that the
    coefficient operator is not hard-wired to one load case; the value is still
    derived here, not transcribed.
    """
    alpha = _positive("aspect_ratio", aspect_ratio)
    if isinstance(load_ratio, bool) or not isinstance(load_ratio, (int, float)):
        raise TypeError("load_ratio must be a real number")
    psi = float(load_ratio)
    if not math.isfinite(psi) or psi < 0.0:
        raise ValueError("load_ratio must be finite and non-negative")
    limit = _positive_int("max_half_waves", max_half_waves)
    best = (math.inf, 0, 0)
    for m in range(1, limit + 1):
        am2 = (m / alpha) ** 2
        for n in range(1, limit + 1):
            bn2 = float(n * n)
            denominator = am2 + psi * bn2
            if denominator <= 0.0:
                continue
            value = (am2 + bn2) ** 2 / denominator
            if value < best[0]:
                best = (value, m, n)
    if best[1] == limit or best[2] == limit:
        raise ValueError("max_half_waves too small: minimising mode at the enumeration edge")
    return best


def coefficient_closure(
    *,
    cases: list[dict[str, float]],
    young: float,
    poisson: float,
    thickness: float,
    tolerance: float = 1e-10,
) -> list[dict[str, object]]:
    """Run the coefficient closure check over a list of analytic cases.

    Each case carries ``length`` and ``width`` in metres.  Every row holds both
    routes, both critical stresses and the *relative* residual
    ``|sigma_A - sigma_B| / sigma_B`` against ``tolerance``.  The residual
    measures algebraic closure between a coefficient and a solved PDE -- it is
    not a validation against an experiment or a code value, and the register
    must say so.
    """
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        width = _positive("width", case["width"])
        length = _positive("length", case["length"])
        alpha = length / width
        k_a, m_a, n_a = closed_form_uniaxial_coefficient(aspect_ratio=alpha)
        k_b, m_b, n_b = navier_uniaxial_coefficient(
            length=length, width=width, young=young, poisson=poisson, thickness=thickness
        )
        sigma_a = critical_stress_from_coefficient(
            coefficient=k_a, young=young, poisson=poisson, thickness=thickness, width=width
        )
        sigma_b = critical_stress_from_coefficient(
            coefficient=k_b, young=young, poisson=poisson, thickness=thickness, width=width
        )
        residual = abs(sigma_a - sigma_b) / sigma_b
        rows.append(
            {
                "case": case.get("id", f"case-{index}"),
                "load_case": case.get("load_case", "uniaxial"),
                "length_m": length,
                "width_m": width,
                "aspect_ratio": alpha,
                "coefficient_route_a": k_a,
                "mode_route_a": [m_a, n_a],
                "coefficient_route_b": k_b,
                "mode_route_b": [m_b, n_b],
                "critical_stress_route_a_pa": sigma_a,
                "critical_stress_route_b_pa": sigma_b,
                "relative_residual": residual,
                "tolerance": tolerance,
                "pass": bool(residual <= tolerance),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 3. discretisations: BFS C1 Hermite element and polynomial Rayleigh--Ritz
# ---------------------------------------------------------------------------
def _num_dof(nodes_x: int, i: int, j: int, k: int) -> int:
    return 4 * (j * nodes_x + i) + k


def _bfs_element_gx(a_e: float, b_e: float) -> np.ndarray:
    """Unit-compression geometric element matrix ``integral w_,x w_,x dA``."""
    gx = np.zeros((16, 16))
    for gi, s in enumerate(_GAUSS04):
        _, dx, _ = hermite_cubic_1d(float(s), a_e)
        for gj, t in enumerate(_GAUSS04):
            hy, _, _ = hermite_cubic_1d(float(t), b_e)
            weight = _GAUSS04_W[gi] * _GAUSS04_W[gj] * a_e * b_e
            derivative = np.zeros(16)
            for ny in (0, 1):
                for nx in (0, 1):
                    base = 4 * (2 * ny + nx)
                    hi_x, dj_x = (0, 1) if nx == 0 else (2, 3)
                    hi_y, dj_y = (0, 1) if ny == 0 else (2, 3)
                    derivative[base] = dx[hi_x] * hy[hi_y]
                    derivative[base + 1] = dx[dj_x] * hy[hi_y]
                    derivative[base + 2] = dx[hi_x] * hy[dj_y]
                    derivative[base + 3] = dx[dj_x] * hy[dj_y]
            gx += weight * np.outer(derivative, derivative)
    return gx


def _bfs_matrices(
    n_x: int, n_y: int, a: float, b: float, young: float, poisson: float, thickness: float
) -> tuple[np.ndarray, np.ndarray, int]:
    nodes_x = n_x + 1
    ndof = 4 * nodes_x * (n_y + 1)
    kb = np.zeros((ndof, ndof))
    gx = np.zeros((ndof, ndof))
    a_e, b_e = a / n_x, b / n_y
    ke_b, _ = bfs_plate_element_matrices(a_e, b_e, young, poisson, thickness)
    ke_g = _bfs_element_gx(a_e, b_e)
    for j in range(n_y):
        for i in range(n_x):
            dofs = np.array(
                [
                    _num_dof(nodes_x, ii, jj, k)
                    for jj in (j, j + 1)
                    for ii in (i, i + 1)
                    for k in range(4)
                ]
            )
            kb[np.ix_(dofs, dofs)] += ke_b
            gx[np.ix_(dofs, dofs)] += ke_g
    return kb, gx, ndof


def _support_dofs(n_x: int, n_y: int) -> set[int]:
    """Supported edges: ``w = 0`` and the tangential slope stays zero.

    The tangential-slope condition keeps the edge straight, which is the standard
    simple-support idealisation and is *satisfied exactly* by the Navier mode
    ``sin(m pi x / a) sin(n pi y / b)`` (its tangential derivative vanishes along
    every edge), so imposing it cannot bias the comparison with the closed-form
    coefficient.
    """
    nodes_x = n_x + 1
    fixed: set[int] = set()
    for j in range(n_y + 1):
        for i in (0, n_x):
            fixed.add(_num_dof(nodes_x, i, j, 0))  # w
            fixed.add(_num_dof(nodes_x, i, j, 2))  # tangential slope w_,y
    for i in range(nodes_x):
        for j in (0, n_y):
            fixed.add(_num_dof(nodes_x, i, j, 0))  # w
            fixed.add(_num_dof(nodes_x, i, j, 1))  # tangential slope w_,x
    return fixed


def _normal_slope_dofs(n_x: int, n_y: int) -> set[int]:
    """Normal-slope dofs on the four edges: ``w_,x`` on x-edges, ``w_,y`` on y-edges."""
    nodes_x = n_x + 1
    fixed: set[int] = set()
    for j in range(n_y + 1):
        for i in (0, n_x):
            fixed.add(_num_dof(nodes_x, i, j, 1))
    for i in range(nodes_x):
        for j in (0, n_y):
            fixed.add(_num_dof(nodes_x, i, j, 2))
    return fixed


def _edge_block(length: float) -> np.ndarray:
    """``integral h_a(s) h_b(s) ds`` for the four 1D Hermite functions on a segment."""
    matrix = np.zeros((4, 4))
    for gi, s in enumerate(_GAUSS04):
        values, _, _ = hermite_cubic_1d(float(s), length)
        matrix += (_GAUSS04_W[gi] * length) * np.outer(values, values)
    return matrix


def _edge_restraint_matrix(n_x: int, n_y: int, a: float, b: float) -> np.ndarray:
    """Assemble ``integral (w_,n)^2 ds`` on all four edges with unit stiffness.

    Scaled by ``k_r`` before it is added to the bending matrix: ``k_r -> 0``
    leaves the normal slope free (simple support), ``k_r -> inf`` clamps it.

    Along an edge the normal slope of a BFS element is a *cubic Hermite*
    interpolant of four nodal values: the normal slope ``w_,n`` and the twist
    ``w_,nt`` at the two nodes of the segment, with the value functions
    ``(h0, h2)`` and the derivative functions ``(h1, h3)`` respectively.  Both
    families share the same derivative at the edge (``dx[1] = 1`` at the left
    node, ``dx[3] = 1`` at the right node), so penalising only one of them --
    a mistake that looks plausible because ``h0, h2`` are the functions that
    vanish at the edge in the *value* sense -- would leave the edge free to
    bulge and under-restrain the plate.  The block below keeps all four.

    Corner nodes carry the twist dof of two edges; it is therefore penalised
    twice, which only makes the constraint stronger (the limit is unchanged).
    """
    nodes_x = n_x + 1
    ndof = 4 * nodes_x * (n_y + 1)
    p = np.zeros((ndof, ndof))
    a_e, b_e = a / n_x, b / n_y

    def add(dofs: tuple[int, int, int, int], length: float) -> None:
        index = np.array(dofs, dtype=int)
        p[np.ix_(index, index)] += _edge_block(length)

    for j in range(n_y):
        for i in (0, n_x):
            add(
                (
                    _num_dof(nodes_x, i, j, 1),
                    _num_dof(nodes_x, i, j, 3),
                    _num_dof(nodes_x, i, j + 1, 1),
                    _num_dof(nodes_x, i, j + 1, 3),
                ),
                b_e,
            )
    for i in range(n_x):
        for j in (0, n_y):
            add(
                (
                    _num_dof(nodes_x, i, j, 2),
                    _num_dof(nodes_x, i, j, 3),
                    _num_dof(nodes_x, i + 1, j, 2),
                    _num_dof(nodes_x, i + 1, j, 3),
                ),
                a_e,
            )
    return p


def _smallest_positive_eigenvalue(
    stiffness: np.ndarray, geometric: np.ndarray, rcond: float = 1.0e-9
) -> float:
    """Smallest ``mu`` with ``K v = mu G v`` over the numerical range of ``G``.

    The sign convention is the buckling one: ``K_b`` is the positive bending
    operator and ``G`` the positive unit-compression geometric operator
    ``integral w_,x w_,x dA``, so the smallest positive generalised eigenvalue
    *is* the critical compressive line load ``N_cr = mu``.

    ``G`` only has to be positive *semi*-definite: directions with a vanishing
    geometric stiffness carry no compression and are projected out (they would
    otherwise return an infinite ratio).  The projection is done through the
    eigen-decomposition of ``G`` rather than a Cholesky factorisation because the
    Rayleigh--Ritz route builds a badly conditioned Gram matrix at high degree,
    where a Cholesky test fails on round-off alone.
    """
    stiffness = 0.5 * (stiffness + stiffness.T)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (geometric + geometric.T))
    ceiling = float(np.max(eigenvalues))
    if ceiling <= 0.0:
        raise ValueError("geometric matrix has no positive direction: check the load case")
    keep = eigenvalues > rcond * ceiling
    if not np.any(keep):
        raise ValueError("geometric matrix is numerically singular: refine or add supports")
    transform = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
    reduced = transform.T @ stiffness @ transform
    reduced = 0.5 * (reduced + reduced.T)
    candidate = np.linalg.eigvalsh(reduced)
    positive = candidate[candidate > 0.0]
    if positive.size == 0:
        raise ValueError("no positive buckling eigenvalue found")
    return float(positive[0])


@dataclass(frozen=True)
class RestraintLadder:
    """One point of the edge-restraint ladder (:func:`restraint_ladder_uniaxial`)."""

    restraint: float
    coefficient: float
    n_x: int
    n_y: int

    def as_dict(self) -> dict[str, object]:
        return {
            "restraint_ratio": None if math.isinf(self.restraint) else self.restraint,
            "clamped": math.isinf(self.restraint),
            "coefficient": self.coefficient,
            "n_x": self.n_x,
            "n_y": self.n_y,
        }


def bfs_uniaxial_coefficient(
    *,
    aspect_ratio: float = 1.0,
    n: int = 8,
    restraint: float | None = None,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.012,
    width: float = 1.0,
) -> RestraintLadder:
    """Uniaxial buckling coefficient ``k`` from the BFS C1 Hermite discretisation.

    ``restraint`` is the dimensionless edge rotational stiffness
    ``k_r = restraint * D / b`` (edge energy ``k_r/2 * integral (w_,n)^2 ds``):
    ``0`` is a simple support (normal slope free) and ``None`` clamps it
    (normal-slope dofs eliminated).  ``n`` is the number of elements across the
    width; the loaded span gets ``round(n * alpha)`` elements.  The edges are
    held ``w = 0`` throughout, so every ladder point is a supported plate and
    only the rotational restraint changes.
    """
    alpha = _positive("aspect_ratio", aspect_ratio)
    n = _positive_int("n", n)
    width = _positive("width", width)
    length = alpha * width
    n_x = max(1, int(round(n * alpha)))
    n_y = n
    _, rigidity = _material(young, poisson, thickness)
    kb, gx, ndof = _bfs_matrices(n_x, n_y, length, width, young, poisson, thickness)
    free = np.ones(ndof, dtype=bool)
    for dof in _support_dofs(n_x, n_y):
        free[dof] = False
    if restraint is None:
        for dof in _normal_slope_dofs(n_x, n_y):
            free[dof] = False
        ratio = math.inf
    else:
        if isinstance(restraint, bool) or not isinstance(restraint, (int, float)):
            raise TypeError("restraint must be a real number or None")
        ratio = float(restraint)
        if not math.isfinite(ratio) or ratio < 0.0:
            raise ValueError("restraint must be finite and non-negative (or None to clamp)")
        if ratio > 0.0:
            kb = kb + ratio * (rigidity / width) * _edge_restraint_matrix(n_x, n_y, length, width)
    index = np.flatnonzero(free)
    mu = _smallest_positive_eigenvalue(kb[np.ix_(index, index)], gx[np.ix_(index, index)])
    coefficient = mu * width**2 / (math.pi**2 * rigidity)
    return RestraintLadder(restraint=ratio, coefficient=coefficient, n_x=n_x, n_y=n_y)


def restraint_ladder_uniaxial(
    *,
    restraints: tuple[float | None, ...] = (0.0, 1.0, 10.0, 100.0, 1000.0),
    aspect_ratio: float = 1.0,
    n: int = 10,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.012,
    width: float = 1.0,
) -> list[RestraintLadder]:
    """Evaluate the restraint ladder; the sequence must be monotone increasing.

    Monotonicity in the restraint is the *operator-level* statement behind
    "clamping raises the plate coefficient".  It is checked here, not assumed.

    The ladder is the *penalty* family.  ``restraint=None`` (nodal clamping) is
    a different constraint set -- it fixes the normal slope at the nodes only,
    while the penalty drives ``integral (w_,n)^2 ds`` to zero along the whole
    edge -- so mixing ``None`` into a finite ladder may legitimately break
    monotonicity; the two variants are reported separately instead.
    """
    ladder = [
        bfs_uniaxial_coefficient(
            aspect_ratio=aspect_ratio,
            n=n,
            restraint=r,
            young=young,
            poisson=poisson,
            thickness=thickness,
            width=width,
        )
        for r in restraints
    ]
    values = [point.coefficient for point in ladder]
    for lower, upper in zip(values, values[1:]):
        if upper < lower - 1.0e-9:
            raise ValueError("restraint ladder is not monotone: check the penalty scaling")
    return ladder


def ritz_uniaxial_coefficient(
    *, aspect_ratio: float = 1.0, degree: int = 5, quadrature: int = 24
) -> float:
    """Clamped-edge coefficient ``k`` from a polynomial Rayleigh--Ritz basis.

    The admissible space is ``W(X, Y) = sum c_ij phi_i(X) phi_j(Y)`` with
    ``phi_i(t) = t^2 (1 - t)^2 t^i``, so ``W = dW/dt = 0`` on all four edges by
    construction (clamped).  With ``X = x/a``, ``Y = y/b`` the normalised
    coefficient is

        k = (alpha^2 / pi^2) * int[(W_,XX / alpha^2 + W_,YY)^2] / int[W_,X^2].

    This basis shares no approximation space and no code with the BFS element, so
    agreement between the two routes is evidence about the clamped coefficient
    rather than self-consistency.  A Ritz quotient is an upper bound on the true
    eigenvalue, so the FE ladder and this value bracket it from either side.
    """
    alpha = _positive("aspect_ratio", aspect_ratio)
    degree = _positive_int("degree", degree)
    quadrature = _positive_int("quadrature", quadrature)
    nodes, weights = _gauss_legendre(quadrature)
    count = degree + 1

    def derivatives(order: int) -> np.ndarray:
        columns = []
        for i in range(count):
            poly = np.polynomial.Polynomial([0.0] * (i + 2) + [1.0, -2.0, 1.0])
            columns.append(poly.deriv(order)(nodes))
        return np.stack(columns, axis=-1)

    phi = derivatives(0)
    d1 = derivatives(1)
    d2 = derivatives(2)

    def field(x_field: np.ndarray, y_field: np.ndarray) -> np.ndarray:
        product = np.einsum("ap,bq->abpq", x_field, y_field)
        return product.reshape(nodes.size * nodes.size, count * count)

    family_xx = field(d2, phi)
    family_yy = field(phi, d2)
    family_x = field(d1, phi)
    weight_grid = (weights[:, None] * weights[None, :]).ravel()

    def integrate(family: np.ndarray) -> np.ndarray:
        return (family * weight_grid[:, None]).T @ family

    stiffness = integrate(family_xx / alpha**2 + family_yy)
    geometric = integrate(family_x)
    mu = _smallest_positive_eigenvalue(stiffness, geometric)
    return alpha**2 * mu / math.pi**2


# ---------------------------------------------------------------------------
# 4. reduction-curve contracts (structural clauses only)
# ---------------------------------------------------------------------------
def elastic_ceiling_violations(
    *, lambdas, chi_values, tolerance: float = 1.0e-12
) -> list[tuple[float, float, float]]:
    """Return the points where ``chi > 1 / lambdabar^2`` (capacity above elastic load).

    The one clause of a reduction-curve contract that holds for every admissible
    curve, independently of any code: the reduced strength ``chi * fy`` cannot
    exceed the elastic critical stress ``sigma_cr = fy / lambdabar^2``.  A curve
    that violates it is not merely different from a code, it is inconsistent.
    """
    violations: list[tuple[float, float, float]] = []
    for lam, chi in zip(lambdas, chi_values):
        lam = float(lam)
        chi = float(chi)
        if lam <= 0.0:
            raise ValueError("lambdabar must be positive")
        ceiling = 1.0 / lam**2
        if chi > ceiling + tolerance:
            violations.append((lam, chi, ceiling))
    return violations


def check_reduction_curve(
    curve,
    *,
    lambdas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0),
    tolerance: float = 1.0e-9,
    flat_limit: float | None = None,
) -> dict[str, object]:
    """Machine-check a candidate reduction curve against the structural contract.

    ``curve`` is a callable ``lambdabar -> chi``.  Clauses checked:

    ``range``           ``0 < chi <= 1`` (a reduction factor never amplifies);
    ``stocky_limit``    ``chi(0) == 1`` within tolerance;
    ``monotone``        non-increasing on the sampled grid;
    ``continuity``      a centrally refined difference is small at every interior
                        sample;
    ``elastic_ceiling`` ``chi <= 1 / lambdabar^2``.

    ``flat_limit`` is the code-specific end of the stocky plateau (with the
    imperfection parameter and plastic-branch constants): it is a *parameter
    supplied by the anchor*.  With the anchor withheld it stays ``None`` and that
    clause is reported as ``SKIPPED_PARAMETER_WITHHELD`` instead of being guessed
    -- which is the point: the checker can be qualified now, the code clause
    cannot.
    """
    values = [float(curve(float(lam))) for lam in lambdas]
    positive = [float(lam) for lam in lambdas if lam > 0.0]
    report: dict[str, object] = {
        "samples": [
            {"lambdabar": float(lam), "chi": float(chi)} for lam, chi in zip(lambdas, values)
        ],
        "tolerance": tolerance,
    }
    clauses: dict[str, object] = {}
    clauses["range"] = {
        "pass": all(0.0 < chi <= 1.0 + tolerance for chi in values),
        "min_chi": min(values),
        "max_chi": max(values),
    }
    zero = float(curve(0.0))
    clauses["stocky_limit"] = {"pass": abs(zero - 1.0) <= tolerance, "chi_at_zero": zero}
    worst_increase = max((upper - lower for lower, upper in zip(values, values[1:])), default=0.0)
    clauses["monotone"] = {"pass": worst_increase <= tolerance, "worst_increase": worst_increase}
    jump = 0.0
    for lam in positive[:-1]:
        h = 1.0e-4 * lam
        jump = max(jump, abs(float(curve(lam + h)) - float(curve(lam - h))))
    threshold = 1.0e-3 * max(values)
    clauses["continuity"] = {"pass": jump <= threshold, "max_jump": jump, "threshold": threshold}
    violations = elastic_ceiling_violations(
        lambdas=positive,
        chi_values=[float(curve(lam)) for lam in positive],
        tolerance=1.0e-12,
    )
    clauses["elastic_ceiling"] = {
        "pass": not violations,
        "violation_count": len(violations),
        "sample_violations": violations[:5],
    }
    if flat_limit is None:
        clauses["stocky_plateau_length"] = {
            "status": "SKIPPED_PARAMETER_WITHHELD",
            "pass": None,
            "reason": "plateau end and imperfection constants come from the anchor text",
        }
    else:
        plateau = max(float(flat_limit), 0.0)
        clauses["stocky_plateau_length"] = {
            "status": "CHECKED",
            "flat_limit": plateau,
            "pass": all(abs(float(curve(lam)) - 1.0) <= tolerance for lam in (0.0, 0.5 * plateau)),
        }
    report["clauses"] = clauses
    report["pass"] = all(clause.get("pass") is not False for clause in clauses.values())
    return report


def synthetic_reduction_specimen(lambdabar: float, *, decay_exponent: float = 4.0) -> float:
    """A **synthetic** contract specimen -- explicitly *not* a code curve.

    ``chi = 1 / sqrt(1 + lambdabar^decay_exponent)`` qualifies
    :func:`check_reduction_curve` and its negative controls.  With the default
    exponent 4 the curve is dimensionless, monotone, ``chi(0) = 1`` and it
    satisfies the elastic ceiling *exactly in the limit* because it decays like
    ``1/lambdabar^2``.  The exponent is a knob for the negative controls: a
    ``(1 + lambdabar^2)``-shaped curve (exponent 2) decays only like
    ``1/lambdabar`` and therefore crosses the ceiling, which is exactly the
    mistake the clause exists to catch.

    Its values carry no design meaning and must never be reported as a
    DNV-RP-C201 or EN 1993-1-6 result.
    """
    lam = float(lambdabar)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("lambdabar must be finite and non-negative")
    exponent = float(decay_exponent)
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("decay_exponent must be positive and finite")
    return 1.0 / math.sqrt(1.0 + lam**exponent)


# ---------------------------------------------------------------------------
# 5. anchor register entries (transcription, or honest withholding)
# ---------------------------------------------------------------------------
ANCHOR_REQUESTS: tuple[dict[str, str], ...] = (
    {
        "id": "A1",
        "anchor": "Timoshenko & Gere, Theory of Elastic Stability -- flat plate buckling",
        "quantity": (
            "plate buckling coefficient k and the closed-form critical stress for "
            "simply supported and clamped rectangular plates"
        ),
        "needed_for": "k in sigma_cr = k pi^2 D / (b^2 t) for at least one aspect ratio",
    },
    {
        "id": "A2",
        "anchor": "DNV-RP-C201 Buckling strength of plated structures",
        "quantity": (
            "plate buckling coefficient and the reduction factor expression with its coefficients"
        ),
        "needed_for": "code-path reduction compared with the derived coefficient operator",
    },
    {
        "id": "A3",
        "anchor": "EN 1993-1-6 Shell buckling",
        "quantity": (
            "shell buckling reduction curve chi(lambdabar): plateau end, imperfection "
            "factor, plastic branch constants"
        ),
        "needed_for": "shell reduction curve chi(lambdabar) clause checks",
    },
)


def anchor_entry(
    request: dict[str, str],
    *,
    verbatim_quote: str | None = None,
    quote_location: str | None = None,
    source_bytes: bytes | None = None,
    source_name: str | None = None,
) -> dict[str, object]:
    """Build one register entry, refusing to record a number without a quote.

    An entry is ``VERBATIM`` only when a quote, a location and the source bytes
    (for the sha256) are supplied together.  Otherwise it is ``WITHHELD`` and
    carries no transcribed value, which is the honest state when the standard or
    handbook text is not present in the environment.  Second-hand transcriptions
    (papers, reports, downstream code) are rejected by construction: the digest
    has to cover the source text itself.
    """
    entry: dict[str, object] = dict(request)
    if verbatim_quote is not None and quote_location is not None and source_bytes is not None:
        entry.update(
            {
                "status": "VERBATIM",
                "verbatim_quote": verbatim_quote,
                "quote_location": quote_location,
                "source_name": source_name,
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )
        return entry
    entry.update(
        {
            "status": "WITHHELD",
            "verbatim_quote": None,
            "quote_location": None,
            "source_name": source_name,
            "source_sha256": None,
            "reason": (
                "anchor source text not present in the execution environment; a value "
                "filled from memory or from a secondary transcription is not evidence"
            ),
            "closure_path": (
                "place the standard/handbook text (with page or clause anchors) in the "
                "evidence corpus, then rebuild the register with --source for that file"
            ),
            "numeric_value_recorded": False,
        }
    )
    return entry


def register_digest(payload: dict[str, object]) -> str:
    """Canonical sha256 of a register payload (sorted keys, stable separators)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
