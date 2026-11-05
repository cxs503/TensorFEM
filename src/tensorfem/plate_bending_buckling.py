"""Buckling of simply supported panels under linearly varying edge compression.

``tensorfem.marine_plate_buckling`` qualifies *uniform* uniaxial compression and
``tensorfem.plate_shear_buckling`` adds *uniform shear* and compression-shear
combinations.  ``docs/benchmark-qualification-status.md`` records the remaining
gap as "non-uniform prebuckling stress": a panel whose in-plane compression
varies across the width.  That is the situation in a hull panel under overall
longitudinal bending combined with local load, where ``N_x`` varies linearly
from compression to tension across one plate field, and the buckling strength is
several times the uniform-compression value.

This module closes the *linearly varying, y-only* subset of that gap.  The
in-plane load is

    N_x(y) = -N_0 * (1 - gradient * y / b),   y in [0, b],  N_0 > 0

so ``gradient = 0`` is uniform compression, ``gradient = 1`` runs from uniform
compression at ``y = 0`` to a stress-free edge at ``y = b``, and
``gradient = 2`` is pure in-plane bending (equal compression and tension).  The
stress ratio used by plate design rules is ``psi = 1 - gradient`` (``sigma_2 /
sigma_1``).  Results are reported as

    k = N_0 b^2 / (pi^2 D),   N_0,cr = k pi^2 D / b^2

where ``D`` is the panel bending rigidity, so ``k`` is directly comparable with
the classical simply supported coefficients: ``k = 4.0`` for uniform
compression, ``k = 7.81`` for the triangular distribution and ``k = 23.9`` for
pure bending of a long panel (Timoshenko & Gere, *Theory of Elastic Stability*,
2nd ed., 1961, plate buckling tables; the same numbers are quoted in ship
structural design codes as ``k = 4.0 / 7.8 / 23.9``).

Three independent verification legs are provided:

* **closed form vs quadrature** - the width integrals
  ``int sin(n pi y/b) sin(q pi y/b) dy`` and
  ``int (y/b) sin(n pi y/b) sin(q pi y/b) dy`` are evaluated in closed form and
  recomputed by Gauss-Legendre quadrature of the energy functionals only
  (:func:`stress_energy_quadrature`), so a parity/sign error in the closed form
  cannot hide;
* **independent strip discretisation** - because ``N_x`` depends on ``y`` only,
  ``w = sin(m pi x/a) f(y)`` is an exact separation, so a plain finite-difference
  discretisation of ``f(y)`` (:func:`strip_buckling_coefficient`) gives a second,
  fully independent route to the same eigenvalue;
* **classical coefficients** - the 3 %-gated comparison in
  :func:`run_linear_stress_qualification`.

Scope and conventions
---------------------
* Lengths in m, rigidities ``Dx, Dy, H = D12 + 2 D66`` in N m, loads in N/m.
* ``N_0`` is the compressive membrane load at the loaded edge ``y = 0``.
* The panel is flat, of constant thickness and simply supported on all four
  edges.  The in-plane stress is uniform in ``x`` and linear in ``y``.
* ``D16/D26`` bending-twisting coupling, non-linear stress variation,
  prebuckling redistribution, shell curvature, stiffener tripping, post-buckled
  (tension-field) reserve, plasticity, shear or transverse load interaction and
  initial imperfections are **not** covered.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .marine_plate_buckling import _positive_finite, isotropic_plate_rigidities

__all__ = [
    "LITERATURE_K_UNIFORM",
    "LITERATURE_K_TRIANGULAR",
    "LITERATURE_K_PURE_BENDING",
    "LITERATURE_K_PURE_BENDING_SQUARE",
    "LinearStressMode",
    "buckling_coefficient",
    "stress_energy_quadrature",
    "sine_stress_matrices",
    "ritz_linear_stress_buckling",
    "strip_buckling_coefficient",
    "stress_ratio",
    "run_linear_stress_qualification",
]

#: Classical simply supported coefficients ``k = N_0 b^2 / (pi^2 D)`` for
#: ``N_x`` uniform (exact Navier value), linear from ``N_0`` to zero (psi = 0)
#: and pure bending (psi = -1, long panel).  Timoshenko & Gere (1961).
LITERATURE_K_UNIFORM = 4.0
LITERATURE_K_TRIANGULAR = 7.81
LITERATURE_K_PURE_BENDING = 23.9
#: Pure bending of a *square* panel; the classical tables give a higher value
#: than the long-panel asymptote because the mode cannot spread over several
#: half waves.
LITERATURE_K_PURE_BENDING_SQUARE = 25.6

#: Above this degree of variation the compressed zone is no longer a straight
#: linearly varying field in the sense of the design rules; kept as a guard.
_MAX_GRADIENT = 3.0


@dataclass(frozen=True)
class LinearStressMode:
    """Critical state of a panel under linearly varying edge compression."""

    #: ``N_0,cr`` - compressive membrane load at the loaded edge ``y = 0``, N/m.
    multiplier: float
    #: ``k = N_0 b^2 / (pi^2 Dx)``; dimensionless buckling coefficient.
    coefficient: float
    #: ``psi = 1 - gradient`` - edge stress ratio of the applied field.
    psi: float
    #: Truncation used, ``(basis_x, basis_y)``.
    basis: tuple[int, int]
    #: Dominant half-wave numbers ``(m, n)``.
    dominant_mode: tuple[int, int]
    #: ``False`` when no positive buckling load was found for the field.
    converged: bool


def stress_ratio(gradient: float) -> float:
    """Edge stress ratio ``psi = sigma_2 / sigma_1 = 1 - gradient``."""
    g = float(gradient)
    if not math.isfinite(g) or g < 0.0 or g > _MAX_GRADIENT:
        raise ValueError(
            f"stress_gradient must be finite and within [0, {_MAX_GRADIENT}], got {gradient!r}"
        )
    return 1.0 - g


def _width_moment(n: int, q: int) -> float:
    """``int_0^1 t sin(n pi t) sin(q pi t) dt`` in closed form."""
    if n == q:
        return 0.25
    a = 0.5 * (((-1.0) ** (n - q) - 1.0) / ((n - q) ** 2 * math.pi**2))
    b = 0.5 * (((-1.0) ** (n + q) - 1.0) / ((n + q) ** 2 * math.pi**2))
    return a - b


def _mode_indices(basis_x: int, basis_y: int) -> list[tuple[int, int]]:
    if int(basis_x) < 2 or int(basis_y) < 2:
        raise ValueError("basis_x and basis_y must be at least 2")
    return [(m, n) for m in range(1, int(basis_x) + 1) for n in range(1, int(basis_y) + 1)]


def sine_stress_matrices(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    stress_gradient: float,
    basis_x: int = 16,
    basis_y: int = 16,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Bending and geometric matrices of the sine basis, closed form.

    Both matrices correspond to the energy functional

        ``Pi = 1/2 c^T Kb c - 1/2 N_0 c^T Kg c``

    with ``N_0`` the compressive edge load, so the smallest buckling load is
    ``1 / lambda_max(Kb^-1 Kg)``.  The geometric matrix is *not* diagonal: the
    linear part of ``N_x(y)`` couples the ``y`` half-wave numbers through
    :func:`_width_moment`.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    grad = float(stress_gradient)
    stress_ratio(grad)  # validation only
    modes = _mode_indices(basis_x, basis_y)
    size = len(modes)
    scale = a * b / 4.0
    bending = np.zeros((size, size))
    geometric = np.zeros((size, size))
    for i, (m, n) in enumerate(modes):
        kx = m * math.pi / a
        ky = n * math.pi / b
        bending[i, i] = (dx * kx**4 + 2.0 * h * kx**2 * ky**2 + dy * ky**4) * scale
        # int_0^a cos(kx x) cos(px x) dx = a/2 for p == m, so only p == m survives.
        y_part = (b / 2.0) * (1.0 if n == n else 0.0) - grad * b * _width_moment(n, n)
        geometric[i, i] = kx**2 * (a / 2.0) * y_part
        for j in range(size):
            p, q = modes[j]
            if p != m or q == n:
                continue
            geometric[i, j] = kx**2 * (a / 2.0) * (-grad * b * _width_moment(n, q))
    return bending, geometric, modes


def stress_energy_quadrature(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    stress_gradient: float,
    basis_x: int = 16,
    basis_y: int = 16,
    points: int = 240,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Same two matrices, rebuilt by Gauss-Legendre quadrature of the energies.

    Independent of every closed-form integral in :func:`sine_stress_matrices`:
    the bending matrix is integrated as ``Dx w,xx^2 + 2 H w,xy^2 + Dy w,yy^2``
    (the integration-by-parts twin of the strong form) and the geometric matrix
    as ``int N_x(y) w,x_i w,x_j dA``.  Only the basis functions and their
    derivatives are shared with the closed form.  ``points`` Gauss abscissae in
    each direction resolve the polynomial-times-trigonometric integrand to
    machine precision at the basis sizes used here.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    grad = float(stress_gradient)
    stress_ratio(grad)
    npts = int(points)
    if npts < 16:
        raise ValueError("points must be at least 16")
    modes = _mode_indices(basis_x, basis_y)
    size = len(modes)
    xi, wq = np.polynomial.legendre.leggauss(npts)
    x = 0.5 * (xi + 1.0) * a
    y = 0.5 * (xi + 1.0) * b
    wx = 0.5 * wq * a
    wy = 0.5 * wq * b
    weight = np.outer(wx, wy)
    strain = 1.0 - grad * np.outer(np.ones(npts), y / b)
    dxx = np.zeros((size, npts, npts))
    dyy = np.zeros_like(dxx)
    dxy = np.zeros_like(dxx)
    dwx = np.zeros_like(dxx)
    for i, (m, n) in enumerate(modes):
        kx = m * math.pi / a
        ky = n * math.pi / b
        sx = np.sin(np.outer(kx * x, np.ones(npts)))
        sy = np.sin(np.outer(np.ones(npts), ky * y))
        cx = np.cos(np.outer(kx * x, np.ones(npts)))
        cy = np.cos(np.outer(np.ones(npts), ky * y))
        field = sx * sy
        dwx[i] = kx * cx * sy
        dxx[i] = -(kx**2) * field
        dyy[i] = -(ky**2) * field
        dxy[i] = kx * ky * cx * cy
    flat = lambda f: f.reshape(size, -1)  # noqa: E731
    w = weight.ravel()
    dxx2 = flat(dxx) @ (flat(dxx) * w).T
    dyy2 = flat(dyy) @ (flat(dyy) * w).T
    dxy2 = flat(dxy) @ (flat(dxy) * w).T
    bending = dx * dxx2 + 2.0 * h * dxy2 + dy * dyy2
    geometric = flat(dwx) @ ((flat(dwx) * strain.reshape(1, -1)) * w).T
    return bending, geometric, modes


def buckling_coefficient(*, edge_load: float, rigidity: float, width: float) -> float:
    """``k = N_0 b^2 / (pi^2 Dx)`` for comparison with tabulated values."""
    load = _positive_finite("edge_load", edge_load)
    d = _positive_finite("rigidity", rigidity)
    b = _positive_finite("width", width)
    return load * b**2 / (math.pi**2 * d)


def ritz_linear_stress_buckling(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    stress_gradient: float,
    basis_x: int = 16,
    basis_y: int = 16,
) -> LinearStressMode:
    """Critical edge load of a panel under linearly varying compression."""
    bending, geometric, modes = sine_stress_matrices(
        length=length,
        width=width,
        rigidity_x=rigidity_x,
        rigidity_y=rigidity_y,
        coupling_rigidity=coupling_rigidity,
        stress_gradient=stress_gradient,
        basis_x=basis_x,
        basis_y=basis_y,
    )
    basis = (int(basis_x), int(basis_y))
    psi = stress_ratio(stress_gradient)
    try:
        chol = np.linalg.cholesky(bending)
    except np.linalg.LinAlgError:  # pragma: no cover - bending is SPD by construction
        return LinearStressMode(math.inf, math.inf, psi, basis, (0, 0), False)
    inv = np.linalg.inv(chol)
    reduced = inv @ geometric @ inv.T
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues = np.linalg.eigvalsh(reduced)
    lam = float(eigenvalues[-1])
    if not math.isfinite(lam) or lam <= 0.0:
        return LinearStressMode(math.inf, math.inf, psi, basis, (0, 0), False)
    multiplier = 1.0 / lam
    _, vectors = np.linalg.eigh(reduced)
    dominant = modes[int(np.argmax(np.abs(vectors[:, -1])))]
    coefficient = buckling_coefficient(
        edge_load=multiplier, rigidity=rigidity_x, width=width
    )
    return LinearStressMode(multiplier, coefficient, psi, basis, dominant, True)


def strip_buckling_coefficient(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    stress_gradient: float,
    nodes: int = 200,
    half_waves: int | None = None,
) -> tuple[float, int]:
    """Independent route: finite differences in ``y`` on the separated strip.

    With ``N_x = N_x(y)`` the operator leaves ``w = sin(m pi x/a) f(y)``
    invariant, so for each half-wave number ``m`` the energy

        ``Pi(f) = 1/2 int_0^b [ Dx k^4 f^2 + 2 H k^2 f'^2 + Dy f''^2
        - N_0 (1 - grad y/b) k^2 f^2 ] dy``,  ``k = m pi / a``

    is a one-dimensional generalised eigenvalue problem.  ``f(y)`` is
    discretised by second-order central differences with the simply supported
    conditions ``f = 0``, ``f'' = 0`` enforced through zero boundary values.
    Returns ``(k, critical_m)``.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    grad = float(stress_gradient)
    stress_ratio(grad)
    n = int(nodes)
    if n < 8:
        raise ValueError("nodes must be at least 8")
    steps = n
    hh = b / steps
    interior = n - 1
    y = hh * np.arange(1, n)
    # first and second difference operators on the interior nodes, boundary zero
    first = np.zeros((interior, interior))
    second = np.zeros((interior, interior))
    for i in range(interior):
        second[i, i] = -2.0
        if i + 1 < interior:
            second[i, i + 1] = 1.0
        if i > 0:
            second[i, i - 1] = 1.0
        if i + 1 < interior:
            first[i, i + 1] = 1.0 / (2.0 * hh)
        if i > 0:
            first[i, i - 1] = -1.0 / (2.0 * hh)
    second /= hh**2
    m_max = int(half_waves) if half_waves is not None else max(4, int(2.0 * a / b) + 3)
    m_max = max(1, m_max)
    best = (math.inf, 0)
    mass = hh * np.eye(interior)
    curvature = mass @ (second.T @ second)
    slope = mass @ (first.T @ first)
    # f'(0) and f'(b) do not vanish, so the composite trapezium rule needs the
    # half-weighted boundary contributions; omitting them leaves an O(h) error
    # in the slope energy (measured: 5.0e-3 at 400 nodes, first order), which is
    # far larger than the discretisation error of the Ritz route.
    slope[0, 0] += 0.5 / hh
    slope[-1, -1] += 0.5 / hh
    gradient_field = 1.0 - grad * (y / b)
    for m in range(1, m_max + 1):
        kx = m * math.pi / a
        stiff = dx * kx**4 * mass + 2.0 * h * kx**2 * slope + dy * curvature
        geom = kx**2 * (mass * gradient_field[None, :])
        geom = 0.5 * (geom + geom.T)
        try:
            chol = np.linalg.cholesky(stiff)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        inv = np.linalg.inv(chol)
        lam = float(np.linalg.eigvalsh(inv @ geom @ inv.T)[-1])
        if lam <= 0.0 or not math.isfinite(lam):
            continue
        load = 1.0 / lam
        if load < best[0]:
            best = (load, m)
    if not math.isfinite(best[0]):
        return math.inf, 0
    return buckling_coefficient(edge_load=best[0], rigidity=dx, width=b), best[1]


def run_linear_stress_qualification(
    *,
    young: float = 210.0e9,
    poisson: float = 0.3,
    thickness: float = 0.012,
    tolerance: float = 0.03,
    basis: int = 16,
) -> dict[str, object]:
    """Qualify linearly varying compression against the classical coefficients.

    The gate is 3 % on ``k`` against the tabulated values.  Cases without a
    tabulated anchor (intermediate stress ratios, orthotropic panels) are
    reported as self-consistency or convergence evidence only and never as a
    passing accuracy claim.
    """
    d, _, h = isotropic_plate_rigidities(young, poisson, thickness)
    rows: list[dict[str, object]] = []
    cases = (
        ("uniform_compression", 1.0, 0.0, LITERATURE_K_UNIFORM),
        ("triangular_compression", 2.0, 1.0, LITERATURE_K_TRIANGULAR),
        ("pure_bending_long_panel", 6.0, 2.0, LITERATURE_K_PURE_BENDING),
        ("pure_bending_square", 1.0, 2.0, LITERATURE_K_PURE_BENDING_SQUARE),
    )
    for name, aspect, gradient, literature in cases:
        mode = ritz_linear_stress_buckling(
            length=aspect,
            width=1.0,
            rigidity_x=d,
            rigidity_y=d,
            coupling_rigidity=h,
            stress_gradient=gradient,
            basis_x=basis,
            basis_y=basis,
        )
        error = abs(mode.coefficient / literature - 1.0) if mode.converged else math.inf
        strip, strip_m = strip_buckling_coefficient(
            length=aspect,
            width=1.0,
            rigidity_x=d,
            rigidity_y=d,
            coupling_rigidity=h,
            stress_gradient=gradient,
        )
        rows.append(
            {
                "case": name,
                "aspect_ratio": aspect,
                "stress_gradient": gradient,
                "psi": mode.psi,
                "literature_k": literature,
                "ritz_k": mode.coefficient,
                "ritz_dominant_mode": list(mode.dominant_mode),
                "strip_k": strip,
                "strip_dominant_mode": strip_m,
                "ritz_vs_strip_relative": abs(strip / mode.coefficient - 1.0),
                "relative_error": error,
                "passed": bool(error < tolerance and abs(strip / mode.coefficient - 1.0) < tolerance),
            }
        )
    # Intermediate stress ratios: no tabulated anchor, so only monotonicity and
    # the two converged limits are asserted.
    scans = []
    previous = 0.0
    for gradient in (0.0, 0.5, 1.0, 1.5, 2.0):
        mode = ritz_linear_stress_buckling(
            length=4.0,
            width=1.0,
            rigidity_x=d,
            rigidity_y=d,
            coupling_rigidity=h,
            stress_gradient=gradient,
            basis_x=basis,
            basis_y=basis,
        )
        scans.append(
            {
                "stress_gradient": gradient,
                "psi": mode.psi,
                "k": mode.coefficient,
                "monotone_increase": bool(mode.coefficient > previous),
            }
        )
        previous = mode.coefficient
    passed = all(bool(row["passed"]) for row in rows) and all(
        bool(row["monotone_increase"]) for row in scans
    )
    if not passed:
        raise AssertionError("linearly varying compression qualification failed")
    return {
        "schema": "tensorfem.plate-linear-stress-buckling/1.0",
        "scope": "TensorFEM variational plate buckling; no TensorLBM/CFD",
        "tolerance": tolerance,
        "young": young,
        "poisson": poisson,
        "thickness": thickness,
        "basis": basis,
        "cases": rows,
        "stress_ratio_scan": scans,
        "passed": passed,
    }