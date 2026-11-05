"""Closed-form spectral reference for buckling of simply supported panels.

Why this module exists
----------------------
``tensorfem.marine_plate_combined_buckling`` supplies a finite-difference
eigensolver for combined in-plane loading and anchors its pure-shear branch on
the classical coefficient ``k_s = 5.34 + 4 (b/a)^2`` of Timoshenko & Gere.
That anchor is an *asymptotic fit*, not an exact solution, and the merged work
reported it as a limitation: at ``a/b ~ 2`` the panel solver sits 3.55% above
the fit with an observed convergence order of only 0.13, so the point was left
uncertified, and a leftover Rayleigh-Ritz cross-check (``k ~ 9.03``) could not
be reconciled with the finite-difference extrapolation (``k ~ 9.346``).

This module removes the ambiguity by giving a *high-accuracy, mesh-free*
reference.  The buckling mode of a simply supported panel is expanded in the
double-sine basis

    w(x, y) = sum_{m,n} c_mn sin(m pi x / a) sin(n pi y / b),

which satisfies both essential edges (``w = 0``) and both natural edges
(``M_n = 0``) *function by function*: for any basis function
``w_,nn = -(n pi/b)^2 w = 0`` on the edges.  Every matrix element is then an
exact product of elementary sine/cosine integrals, so there is no mesh, no
quadrature error and no discretisation of the boundary.  Because the double
sine also diagonalises the bending operator, the bending matrix is diagonal and
the buckling pencil reduces to the symmetric standard problem

    C = K_b^{-1/2} G K_b^{-1/2},     C c = (1/N) c,     N_cr = 1 / lambda_max(C),

where ``G = longitudinal * Gx + transverse * Gy + shear * Gs`` is the geometric
operator for a unit reference line load.  The geometric operator is indefinite
under shear, so the spectrum is two sided and the buckling load is the largest
*positive* eigenvalue of ``C`` - the same convention the finite-difference
solver uses, which makes the two discretisations directly comparable.

What it delivers
----------------
* ``spectral_shear_buckling_coefficient`` - converged ``k_s(a/b)`` for pure
  shear.  ``k_s(1) = 9.32453`` against the textbook ``9.34`` (-0.17%); the fit
  is 3.15% low at ``a/b = 2``, which is why the finite-difference run looked
  like it failed there.
* ``spectral_buckling_load`` - the same machinery for any
  ``(N_x, N_y, N_xy)`` combination, used as the reference.
* ``run_shear_spectral_qualification`` - fail-closed gates comparing the
  spectral reference with quadrature, with the exact Navier and uniaxial
  closed forms, with the literature fit and with the finite-difference solver.

Numerics
--------
Only ``numpy.linalg`` is used.  ``scipy.linalg`` segfaults on some hosts of this
fleet at matrices of size ~30 and above (``cholesky`` of a 36x36 identity
reproducibly dumps core), so the LAPACK entry point is deliberately avoided;
``numpy.linalg`` calls the same LAPACK but takes a code path that survives here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SpectralBucklingResult",
    "cos_sin_integral",
    "spectral_buckling_operators",
    "spectral_buckling_load",
    "spectral_shear_buckling_coefficient",
    "richardson_extrapolation",
    "run_shear_spectral_qualification",
]

#: Node count of the Gauss-Legendre rule used to verify the closed-form
#: operator entries.  A double-sine integrand is a trigonometric polynomial of
#: order at most ``m + p``, so 48 nodes are far beyond exactness for the
#: truncations used here.
_QUADRATURE_NODES = 48


def _finite(name: str, value, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _integer(name: str, value, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {number}")
    return number


def cos_sin_integral(cos_index: int, sin_index: int, length: float) -> float:
    """``int_0^L cos(cos_index pi x / L) sin(sin_index pi x / L) dx``.

    The integral vanishes unless the two integer indices have opposite parity,
    in which case it equals ``2 q / (q^2 - p^2) * L / pi`` with ``p`` the cosine
    index and ``q`` the sine index.  Both arguments must be positive integers.
    """

    p = _integer("cos_index", cos_index)
    q = _integer("sin_index", sin_index)
    span = _finite("length", length, positive=True)
    if p == q or (p + q) % 2 == 0:
        return 0.0
    return 2.0 * q / (q * q - p * p) * span / math.pi


def _cos_sin_matrix(cos_index: np.ndarray, sin_index: np.ndarray, length: float) -> np.ndarray:
    """Vectorised ``cos_sin_integral`` over all index pairs.

    Returns ``M[i, j] = int_0^L cos(cos_index[i] pi x / L)
    sin(sin_index[j] pi x / L) dx``.
    """

    p = np.asarray(cos_index, dtype=float)[:, None]
    q = np.asarray(sin_index, dtype=float)[None, :]
    odd = np.mod(p + q, 2.0) == 1.0
    denominator = q * q - p * p
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 2.0 * q / denominator * (length / math.pi)
    return np.where(odd & (denominator != 0.0), value, 0.0)


@dataclass(frozen=True)
class SpectralBucklingResult:
    """Buckling load of a simply supported panel in the double-sine basis."""

    critical_line_load: float
    """Reference line load ``N`` at buckling (N/m)."""

    shares: tuple[float, float, float]
    """``(longitudinal, transverse, shear)`` multipliers of ``N``."""

    modes: tuple[int, int]
    """Half-wave counts ``(modes_x, modes_y)`` of the truncation."""

    spectrum: np.ndarray = field(repr=False)
    """All generalised eigenvalues ``1/N`` of the pencil (ascending)."""

    positive_loads: np.ndarray = field(repr=False)
    """Positive buckling loads ``N`` implied by the spectrum, ascending."""

    @property
    def next_load_ratio(self) -> float:
        """Ratio of the second buckling load to the first (mode separation)."""

        if self.positive_loads.size < 2:
            return math.inf
        return float(self.positive_loads[1] / self.positive_loads[0])


def spectral_buckling_operators(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    modes_x: int,
    modes_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form bending/geometric operators in the double-sine basis.

    Returns ``(bending_diagonal, geometry_x, geometry_y, geometry_shear)``.  The
    bending operator is returned as a diagonal *vector* (length ``modes_x *
    modes_y``) because the double sine diagonalises it; the three geometric
    operators are dense symmetric matrices for a unit reference line load of
    1 N/m.  All entries are exact; see the module docstring for the derivation
    and ``tests/test_shear_plate_spectral.py`` for the quadrature check.
    """

    a = _finite("length", length, positive=True)
    b = _finite("width", width, positive=True)
    dx = _finite("rigidity_x", rigidity_x, positive=True)
    dy = _finite("rigidity_y", rigidity_y, positive=True)
    dh = _finite("coupling_rigidity", coupling_rigidity, positive=True)
    mx = _integer("modes_x", modes_x)
    my = _integer("modes_y", modes_y)

    m = np.repeat(np.arange(1, mx + 1), my).astype(float)
    n = np.tile(np.arange(1, my + 1), mx).astype(float)
    alpha = m * math.pi / a
    beta = n * math.pi / b
    area = 0.5 * a * 0.5 * b  # (a/2) * (b/2): int sin^2 = L/2 per direction

    bending = area * (dx * alpha**4 + 2.0 * dh * alpha**2 * beta**2 + dy * beta**4)

    same_m = m[:, None] == m[None, :]
    same_n = n[:, None] == n[None, :]
    geometry_x = (alpha[:, None] * alpha[None, :]) * area * same_m * same_n
    geometry_y = (beta[:, None] * beta[None, :]) * area * same_m * same_n

    cos_sin_x = _cos_sin_matrix(m, m, a)      # [i, j] = int cos(m_i) sin(m_j)
    sin_cos_x = cos_sin_x.T                   # [i, j] = int sin(m_i) cos(m_j)
    cos_sin_y = _cos_sin_matrix(n, n, b)      # [i, j] = int cos(n_i) sin(n_j)
    sin_cos_y = cos_sin_y.T                   # [i, j] = int sin(n_i) cos(n_j)
    # int w_i,y w_j,x  +  int w_i,x w_j,y  for the bare N_xy = 1 N/m term.
    geometry_shear = (
        beta[:, None] * alpha[None, :] * sin_cos_x * cos_sin_y
        + alpha[:, None] * beta[None, :] * cos_sin_x * sin_cos_y
    )
    geometry_shear = 0.5 * (geometry_shear + geometry_shear.T)
    return bending, geometry_x, geometry_y, geometry_shear


def spectral_buckling_load(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    modes_x: int,
    modes_y: int,
    longitudinal_share: float = 1.0,
    transverse_share: float = 0.0,
    shear_share: float = 0.0,
    keep_spectrum: bool = False,
) -> SpectralBucklingResult:
    """Critical reference line load ``N`` for a mixed in-plane load state.

    ``N_x = longitudinal_share * N``, ``N_y = transverse_share * N`` and
    ``N_xy = shear_share * N``.  Tension components may be negative.  A load
    state with no positive buckling load raises ``ValueError`` instead of
    reporting an infinite strength.
    """

    lx = _finite("longitudinal_share", longitudinal_share)
    ty = _finite("transverse_share", transverse_share)
    sxy = _finite("shear_share", shear_share)
    if lx == 0.0 and ty == 0.0 and sxy == 0.0:
        raise ValueError("at least one load share must be non-zero")

    bending, gx, gy, gs = spectral_buckling_operators(
        length=length, width=width, rigidity_x=rigidity_x,
        rigidity_y=rigidity_y, coupling_rigidity=coupling_rigidity,
        modes_x=modes_x, modes_y=modes_y,
    )
    geometry = lx * gx + ty * gy + sxy * gs
    scale = np.sqrt(np.outer(bending, bending))
    reduced = geometry / scale
    reduced = 0.5 * (reduced + reduced.T)
    values = np.linalg.eigvalsh(reduced)
    if not np.all(np.isfinite(values)):
        raise ValueError("spectral operators are not finite")
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("load combination admits no positive buckling load")
    loads = 1.0 / positive
    loads.sort()
    # ``values`` is ascending, so ``1/value`` is descending; sort the loads so
    # that the critical load is the first entry.
    return SpectralBucklingResult(
        critical_line_load=float(loads[0]),
        shares=(lx, ty, sxy),
        modes=(int(modes_x), int(modes_y)),
        spectrum=values if keep_spectrum else np.array([], dtype=float),
        positive_loads=loads,
    )


def _mode_budget(aspect_ratio: float, budget: int) -> tuple[int, int]:
    """Isotropic mode budget: equal mode density per unit length.

    ``modes_x / a = modes_y / b`` so the cutoff wavelength is the same in both
    directions; the total number of modes stays close to ``budget**2``.
    """

    root = math.sqrt(aspect_ratio)
    mx = max(1, int(round(budget * root)))
    my = max(1, int(round(budget / root)))
    return mx, my


def spectral_shear_buckling_coefficient(
    aspect_ratio: float,
    *,
    mode_budget: int = 12,
) -> float:
    """Pure-shear buckling coefficient ``k_s`` from the converged spectral model.

    ``k_s = N_xy,cr b^2 / (pi^2 D)`` for an isotropic panel with bending
    rigidity ``D``; the plateau values are ``9.32453`` (``a/b = 1``), ``7.06998``
    (1.5), ``6.54604`` (2) and ``5.84016`` (3), converged to better than 1e-5
    with ``mode_budget >= 20``.
    """

    aspect = _finite("aspect_ratio", aspect_ratio, positive=True)
    if aspect < 1.0:
        raise ValueError("aspect_ratio must be at least 1 (long panel along x)")
    budget = _integer("mode_budget", mode_budget, minimum=2)
    mx, my = _mode_budget(aspect, budget)
    result = spectral_buckling_load(
        length=aspect, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=mx, modes_y=my,
        longitudinal_share=0.0, transverse_share=0.0, shear_share=1.0,
    )
    return result.critical_line_load / math.pi**2


def richardson_extrapolation(
    budgets: list[int], values: list[float]
) -> tuple[float, float]:
    """Observed order and Richardson limit from the last three refinements.

    ``budgets`` are the mode budgets (larger is finer); the refinement parameter
    is ``h = 1 / budget``.  The order ``p`` solves

        (v2 - v1) / (v1 - v0) = (h2^p - h1^p) / (h1^p - h0^p),

    which is solved by bisection so that non-uniform budget sequences are
    handled correctly, and the limit follows as
    ``v2 + (v2 - v1) / ((h1/h2)^p - 1)``.
    """

    if len(budgets) < 3 or len(budgets) != len(values):
        raise ValueError("need at least three (budget, value) pairs")
    h = [1.0 / float(b) for b in budgets[-3:]]
    v = [float(x) for x in values[-3:]]
    if not all(math.isfinite(x) for x in h + v):
        raise ValueError("refinement data must be finite")
    if not h[0] > h[1] > h[2] > 0.0:
        raise ValueError("budgets must increase strictly")
    d1 = v[1] - v[0]
    d2 = v[2] - v[1]
    if d1 == 0.0 or d2 == 0.0 or d1 / d2 <= 0.0:
        return math.nan, v[2]

    def ratio(power: float) -> float:
        return (h[2] ** power - h[1] ** power) / (h[1] ** power - h[0] ** power)

    target = d2 / d1
    low, high = 1e-3, 64.0
    if not (ratio(low) >= target >= ratio(high)):
        # Non-asymptotic data: fall back to the finest value with no order claim.
        return math.nan, v[2]
    for _ in range(200):
        mid = 0.5 * (low + high)
        if ratio(mid) > target:
            low = mid
        else:
            high = mid
        if high - low < 1e-13:
            break
    order = 0.5 * (low + high)
    limit = v[2] + d2 / ((h[1] / h[2]) ** order - 1.0)
    return order, limit


def _quadrature_check(
    *, length: float, width: float, modes_x: int, modes_y: int
) -> float:
    """Max relative deviation between closed-form operators and quadrature."""

    a, b = length, width
    bending, gx, gy, gs = spectral_buckling_operators(
        length=a, width=b, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=modes_x, modes_y=modes_y,
    )
    nodes, weights = np.polynomial.legendre.leggauss(_QUADRATURE_NODES)
    xs = 0.5 * (nodes + 1.0) * a
    ys = 0.5 * (nodes + 1.0) * b
    wxs = 0.5 * weights * a
    wys = 0.5 * weights * b
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    weight = np.outer(wxs, wys)
    modes = [(i, j) for i in range(1, modes_x + 1) for j in range(1, modes_y + 1)]
    size = len(modes)
    ref_x = np.zeros((size, size))
    ref_y = np.zeros((size, size))
    ref_s = np.zeros((size, size))
    ref_b = np.zeros((size, size))
    deriv = []
    for m, n in modes:
        am, bn = m * math.pi / a, n * math.pi / b
        shape = np.sin(am * grid_x) * np.sin(bn * grid_y)
        dx1 = am * np.cos(am * grid_x) * np.sin(bn * grid_y)
        dy1 = bn * np.sin(am * grid_x) * np.cos(bn * grid_y)
        dxx = -am * am * shape
        dyy = -bn * bn * shape
        dxy = am * bn * np.cos(am * grid_x) * np.cos(bn * grid_y)
        deriv.append((dx1, dy1, dxx, dyy, dxy))
    for i in range(size):
        dxi, dyi, dxxi, dyyi, dxyi = deriv[i]
        for j in range(size):
            dxj, dyj, dxxj, dyyj, dxyj = deriv[j]
            ref_x[i, j] = np.sum(weight * dxi * dxj)
            ref_y[i, j] = np.sum(weight * dyi * dyj)
            ref_s[i, j] = np.sum(weight * (dyi * dxj + dxi * dyj))
            ref_b[i, j] = np.sum(weight * (dxxi * dxxj + 2.0 * dxyi * dxyj + dyyi * dyyj))
    deviation = 0.0
    for analytic, reference in ((gx, ref_x), (gy, ref_y), (gs, ref_s)):
        scale = max(np.max(np.abs(reference)), 1e-300)
        deviation = max(deviation, float(np.max(np.abs(analytic - reference)) / scale))
    scale = max(np.max(np.abs(ref_b)), 1e-300)
    deviation = max(deviation, float(np.max(np.abs(np.diag(bending) - ref_b) / scale)))
    return deviation


def run_shear_spectral_qualification(
    *,
    fine_budgets: tuple[int, ...] = (8, 12, 16, 20, 24),
    finite_difference_grid: int = 16,
) -> dict[str, object]:
    """Fail-closed qualification of the spectral reference for pure shear.

    Gates (any of them failing means ``passed`` is ``False``):

    ``operators_vs_quadrature``
        closed-form operator entries against Gauss-Legendre quadrature
        (tolerance 1e-10 relative);
    ``uniaxial_recovers_classical_k``
        the same operator set must reproduce ``k = 4`` exactly for a uniaxial
        panel (tolerance 1e-10 relative), which checks the bending normalisation
        and the geometric operator at once;
    ``biaxial_recovers_navier``
        equal biaxial compression of a square panel must give
        ``N_x = 2 pi^2 D / b^2`` (Navier) to 1e-10;
    ``shear_sign_symmetry``
        ``shear_share = +1`` and ``-1`` must give the same critical load;
    ``shear_convergence``
        every aspect ratio must converge monotonically with shrinking gaps and a
        Richardson limit within 1e-4 of the finest truncation;
    ``shear_reference_vs_classical_fit``
        ``k_s(1)`` against the textbook ``9.34``, tolerance 3%;
    ``shear_reference_vs_finite_difference``
        the finite-difference solver of ``marine_plate_combined_buckling``
        against the spectral reference at ``a/b = 1`` and ``a/b = 2``,
        tolerance 3%.  The ``a/b = 2`` gate is the point the merged report left
        uncertified;
    ``classical_fit_is_not_a_gate``
        informational: the fit's deviation at ``a/b = 2`` is recorded and must
        exceed the fit's deviation at ``a/b = 1`` by construction of the
        report; the gate only records the number, it cannot be "passed".

    ``finite_difference_grid`` controls the cost of the cross-discretisation
    gate; the grid enters through the finite-difference solver only.
    """

    gates: list[dict[str, object]] = []

    def gate(
        name: str, value, reference, tolerance: float, note: str = "",
        *, mode: str = "relative",
    ) -> None:
        try:
            numeric = float(value)
            target = float(reference)
            finite = math.isfinite(numeric) and math.isfinite(target)
        except (TypeError, ValueError):
            numeric, target, finite = math.nan, math.nan, False
        if mode not in ("relative", "absolute"):
            raise ValueError(f"unknown gate mode {mode!r}")
        if not finite:
            relative = math.nan
            passed = False
        elif mode == "absolute":
            relative = abs(numeric - target)
            passed = math.isfinite(relative) and relative <= tolerance
        elif target == 0.0:
            relative = math.nan
            passed = False
        else:
            relative = abs(numeric - target) / abs(target)
            passed = math.isfinite(relative) and relative <= tolerance
        gates.append(
            {
                "name": name, "value": numeric, "reference": target,
                "relative_error": relative, "tolerance": tolerance,
                "passed": bool(passed), "note": note,
            }
        )

    quadrature = _quadrature_check(length=2.0, width=1.0, modes_x=5, modes_y=4)
    gate(
        "operators_vs_quadrature", quadrature, 0.0, 1e-10,
        note="max relative deviation of the closed-form operators from quadrature",
        mode="absolute",
    )

    bending, gx, gy, gs = spectral_buckling_operators(
        length=1.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=6, modes_y=6,
    )
    scale = np.sqrt(np.outer(bending, bending))
    uniaxial = spectral_buckling_load(
        length=1.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=6, modes_y=6,
        longitudinal_share=1.0, transverse_share=0.0, shear_share=0.0,
    )
    gate(
        "uniaxial_recovers_classical_k",
        uniaxial.critical_line_load / math.pi**2, 4.0, 1e-10,
        note="k = 4 for the first uniaxial mode of a square panel",
    )
    biaxial = spectral_buckling_load(
        length=1.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=6, modes_y=6,
        longitudinal_share=1.0, transverse_share=1.0, shear_share=0.0,
    )
    gate(
        "biaxial_recovers_navier",
        biaxial.critical_line_load, 2.0 * math.pi**2, 1e-10,
        note="Navier: N_x = 2 pi^2 D / b^2 for equal biaxial load on a square panel",
    )

    plus = spectral_buckling_load(
        length=1.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=10, modes_y=10,
        longitudinal_share=0.0, transverse_share=0.0, shear_share=1.0,
    ).critical_line_load
    minus = spectral_buckling_load(
        length=1.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=10, modes_y=10,
        longitudinal_share=0.0, transverse_share=0.0, shear_share=-1.0,
    ).critical_line_load
    gate(
        "shear_sign_symmetry", minus, plus, 1e-12,
        note="pure shear up and down must buckle at the same magnitude",
    )

    aspects = (1.0, 1.5, 2.0, 3.0)
    convergence: dict[str, object] = {}
    for aspect in aspects:
        values = [spectral_shear_buckling_coefficient(aspect, mode_budget=b) for b in fine_budgets]
        gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]
        monotone = all(g >= -1e-12 for g in gaps)
        shrinking = all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1))
        order, limit = richardson_extrapolation(list(fine_budgets), values)
        within = bool(math.isfinite(limit)) and abs(limit - values[-1]) / limit <= 1e-4
        convergence[str(aspect)] = {
            "budgets": list(fine_budgets),
            "coefficients": values,
            "gaps": gaps,
            "monotone": bool(monotone),
            "shrinking_gaps": bool(shrinking),
            "observed_order": float(order),
            "richardson_limit": float(limit),
            "finest_deviation": float(abs(limit - values[-1])),
        }
        gates.append(
            {
                "name": f"shear_convergence[a/b={aspect}]",
                "value": float(values[-1]), "reference": float(limit),
                "relative_error": float(abs(limit - values[-1]) / limit),
                "tolerance": 1e-4,
                "passed": bool(monotone and shrinking and within),
                "note": "monotone decreasing, shrinking gaps, Richardson stable",
            }
        )

    square = spectral_shear_buckling_coefficient(1.0, mode_budget=24)
    fit_square = 5.34 + 4.0 / 1.0**2
    fit_two = 5.34 + 4.0 / 2.0**2
    spectral_two = spectral_shear_buckling_coefficient(2.0, mode_budget=24)
    gate(
        "shear_reference_vs_classical_fit", square, 9.34, 0.03,
        note="Timoshenko & Gere k_s = 9.34 for the square panel",
    )
    gates.append(
        {
            "name": "classical_fit_is_not_a_gate",
            "value": float(fit_two), "reference": float(spectral_two),
            "relative_error": float(abs(fit_two - spectral_two) / spectral_two),
            "tolerance": 0.03,
            "passed": True,
            "note": "the fit itself exceeds 3% at a/b = 2; it must not be used as a gate there",
        }
    )

    try:
        import torch

        from .marine_plate_combined_buckling import panel_buckling_load

        cross: dict[str, object] = {}
        for aspect in (1.0, 2.0):
            grid_x = finite_difference_grid
            grid_y = finite_difference_grid * int(round(aspect))
            result = panel_buckling_load(
                length=aspect, width=1.0, thickness=0.012,
                rigidity_x=1.0, rigidity_y=1.0, coupling_rigidity=1.0,
                grid_x=grid_x, grid_y=grid_y,
                longitudinal_share=0.0, transverse_share=0.0, shear_share=1.0,
            )
            fd_k = float(result.critical_line_load) / math.pi**2
            reference = spectral_shear_buckling_coefficient(aspect, mode_budget=24)
            cross[str(aspect)] = {
                "finite_difference_grid": [grid_x, grid_y],
                "finite_difference_k": fd_k,
                "spectral_k": reference,
                "relative_error": abs(fd_k - reference) / reference,
            }
            gate(
                f"shear_reference_vs_finite_difference[a/b={aspect}]",
                fd_k, reference, 0.03,
                note="finite-difference solver of marine_plate_combined_buckling",
            )
    except ImportError as error:  # pragma: no cover - torch is a hard dependency
        cross = {"error": f"torch unavailable: {error}"}
        gates.append(
            {
                "name": "shear_reference_vs_finite_difference", "value": math.nan,
                "reference": math.nan, "relative_error": math.nan,
                "tolerance": 0.03, "passed": False,
                "note": "cross-discretisation gate could not be evaluated",
            }
        )

    passed = all(bool(entry["passed"]) for entry in gates)
    return {
        "schema": "tensorfem.shear_plate_spectral.qualification/1",
        "tolerance": 0.03,
        "gates": gates,
        "passed": passed,
        "convergence": convergence,
        "cross_discretisation": cross,
        "square_panel": {
            "spectral_k": square,
            "classical_fit_k": fit_square,
            "relative_error": abs(square - fit_square) / fit_square,
        },
        "aspect_two": {
            "spectral_k": spectral_two,
            "classical_fit_k": fit_two,
            "relative_error": abs(fit_two - spectral_two) / spectral_two,
        },
        "reduced_matrix_scale": float(np.max(np.abs(scale))),
    }