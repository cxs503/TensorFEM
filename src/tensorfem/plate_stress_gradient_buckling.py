"""Buckling of a simply supported panel under a *linearly varying* in-plane load.

Why this module exists
----------------------
``tensorfem.marine_plate_buckling`` covers uniaxial compression, and
``tensorfem.plate_shear_buckling`` adds uniform shear and combined
compression--shear.  Both assume a **uniform** in-plane stress field, and the
scope note of the latter module records "prebuckling stress gradients ... not
covered" -- an open verification debt in
``docs/benchmark-qualification-status.md``.

The load case added here is the classical stress gradient: the compressive
membrane load varies linearly across the panel, one edge carrying ``N0`` and the
opposite edge ``psi * N0``, with ``psi`` the edge stress ratio
(``psi = 1`` uniform compression, ``psi = 0`` triangular, ``psi = -1`` pure
in-plane bending).  The load is *not* separable: the destabilising term
``N_x(y) w,x^2`` couples the half-wave numbers through the ramp mass matrix, so
the one-term Navier solution of the uniform case does not apply and a truncated
variational solution is required.

Formulation
-----------
Kirchhoff strain energy and load potential::

    U = 1/2 int [ Dx w,xx^2 + 2 H w,xx w,yy + Dy w,yy^2 ] dA
    V = 1/2 int N_x(y) w,x^2 dA,    N_x(y) = N0 g(y/b),  g(t) = 1 + (psi-1) t

with the double-sine trial space ``w = sum_mn c_mn sin(m pi x/a) sin(n pi y/b)``
(``m, n >= 1``, simply supported on all four edges).  The bending matrix is
reused unchanged from :func:`tensorfem.plate_shear_buckling.sine_energy_matrices`;
the new ingredient is the geometric matrix, whose ``y`` factor is the exact ramp
integral::

    Yhat[n, q] = I(|n-q|) - I(n+q),     I(k) = int_0^1 g(t) cos(k pi t) dt

with ``I(0) = (1+psi)/2`` and ``I(k) = (psi-1) ((-1)^k - 1) / (k^2 pi^2)``.
``Yhat`` is diagonal only for the uniform case, and for ``psi = -1`` its diagonal
vanishes, so the geometric matrix becomes indefinite and the eigenvalue driver
must select the smallest *positive* multiplier.

Verification legs
-----------------
1. *Exact*: for ``psi = 1`` the machinery must reproduce both the closed-form
   Navier load of ``marine_plate_buckling.navier_uniaxial_buckling_load`` and the
   analytic single-mode ``k = (a/b + b/a)^2``.
2. *Independent quadrature*: the geometric matrix is recomputed from
   Gauss--Legendre quadrature of the load potential with no closed integral.
3. *Literature*: the tabulated simply supported coefficients ``k = 4.00``
   (uniform), ``7.81`` (triangular), ``23.9`` (pure bending, minimum over the
   aspect ratio) with ``k = N0 b^2 / (pi^2 D)`` referenced to the larger edge.
4. *Basis refinement*: the critical coefficient must approach the limit from
   above (Rayleigh quotient), so a coarse basis cannot pass as converged.

Conventions
-----------
Lengths in m, rigidities ``Dx, Dy, H = D12 + 2 D66`` in N m, membrane loads in
N/m.  ``N0 > 0`` is compression; the reported multiplier is the critical value of
the *largest* edge compression, the convention of the tabulated coefficient.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .marine_plate_buckling import (
    _positive_finite,
    isotropic_plate_rigidities,
    navier_uniaxial_buckling_load,
)
from .plate_shear_buckling import (
    _modes,
    _smallest_positive_from_symmetric_indefinite,
    sine_energy_matrices,
)

__all__ = [
    "LITERATURE_GRADIENT_COEFFICIENTS",
    "REFERENCE_GRADIENT_SWEEP",
    "StressGradientBucklingMode",
    "gradient_buckling_mode_field",
    "gradient_cos_integral",
    "plate_buckling_coefficient",
    "reference_gradient_sweep",
    "ritz_gradient_buckling_load",
    "run_stress_gradient_qualification",
    "sine_gradient_geometric_matrix",
    "sine_gradient_geometric_matrix_quadrature",
    "stress_gradient_ramp",
]


@dataclass(frozen=True)
class StressGradientBucklingMode:
    """Critical load level of a panel under a linearly varying compression."""

    #: critical largest-edge compression ``N0`` [N/m].
    multiplier: float
    #: buckling coefficient ``k = N0 b^2 / (pi^2 Dx)``.
    coefficient: float
    #: edge stress ratio of the applied load.
    stress_gradient: float
    basis_shape: tuple[int, int]
    #: dominant half-wave numbers ``(m, n)`` of the critical mode.
    dominant_mode: tuple[int, int]
    buckling: bool


#: Classical simply supported buckling coefficients ``k = N0 b^2 / (pi^2 D)``,
#: referenced to the larger compression edge, keyed by ``(psi, a/b)``:
#:
#: * ``(1, 1)`` -- uniform compression, ``k = 4`` (exact single-term Navier);
#: * ``(0, 1)`` -- triangular load vanishing at one edge, ``k = 7.81``;
#: * ``(-1, 2)`` -- pure in-plane bending, ``k = 23.9``.  That tabulated value is
#:   the *minimum over the aspect ratio*, which a long panel reaches; a square
#:   panel buckles at ``k = 25.53`` instead, so the anchor is applied at
#:   ``a/b = 2`` and the square-panel value is reported as reference-only
#:   (Timoshenko & Gere, *Theory of Elastic Stability*, 2nd ed. 1961, Table 9.2).
#:   The discrete reference brackets that minimum: ``23.8818`` at ``a/b = 2`` and
#:   ``23.9123`` at ``a/b = 0.7``.
LITERATURE_GRADIENT_COEFFICIENTS: dict[tuple[float, float], float] = {
    (1.0, 1.0): 4.00,
    (0.0, 1.0): 7.81,
    (-1.0, 2.0): 23.9,
}

#: Edge stress ratios of the reported reference sweep.  Only the three ratios in
#: :data:`LITERATURE_GRADIENT_COEFFICIENTS` have tabulated anchors; the
#: intermediate ones are a certified *reference*, not a literature comparison.
REFERENCE_GRADIENT_SWEEP: tuple[float, ...] = (
    1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.25, -0.5, -0.75, -1.0,
)


def stress_gradient_ramp(name: str, value: float) -> float:
    """Validate an edge stress ratio ``psi`` in ``[-1, 1]``.

    ``psi = 1`` is uniform compression, ``psi = 0`` a triangular load vanishing at
    one edge, ``psi = -1`` pure in-plane bending.  Ratios outside ``[-1, 1]``
    (net tensile load on the opposite edge) are rejected rather than silently
    reinterpreted.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    ratio = float(value)
    if not math.isfinite(ratio):
        raise ValueError(f"{name} must be finite")
    if ratio < -1.0 or ratio > 1.0:
        raise ValueError(f"{name} must lie in [-1, 1]")
    return ratio


def gradient_cos_integral(harmonic: int, stress_gradient: float) -> float:
    """``int_0^1 g(t) cos(k pi t) dt`` for the ramp ``g(t) = 1 + (psi-1) t``."""
    k = int(harmonic)
    if k < 0:
        raise ValueError("harmonic must be non-negative")
    psi = float(stress_gradient)
    if k == 0:
        return 0.5 * (1.0 + psi)
    sign = 1.0 if k % 2 == 0 else -1.0
    return (psi - 1.0) * (sign - 1.0) / (k * k * math.pi * math.pi)


def sine_gradient_geometric_matrix(
    *,
    length: float,
    width: float,
    basis_x: int,
    basis_y: int,
    stress_gradient: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Geometric (load) matrix of unit largest-edge compression, closed form.

    Normalised like ``sine_energy_matrices`` (common factor ``a*b/4``) so it can
    be paired with that function's bending matrix.  For ``stress_gradient = 1``
    the matrix reduces to the uniform ``diag(alpha)``.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    nb_x = int(basis_x)
    nb_y = int(basis_y)
    if nb_x < 2 or nb_y < 2:
        raise ValueError("basis_x and basis_y must be at least 2")
    psi = stress_gradient_ramp("stress_gradient", stress_gradient)
    modes = _modes(nb_x, nb_y)
    alpha = np.array([(m * math.pi / a) ** 2 for m, _ in modes])
    ramp = np.empty((nb_y, nb_y))
    for n in range(1, nb_y + 1):
        for q in range(1, nb_y + 1):
            ramp[n - 1, q - 1] = (
                gradient_cos_integral(abs(n - q), psi)
                - gradient_cos_integral(n + q, psi)
            )
    # ``alpha`` is indexed per *mode* ``(m, n)`` while ``ramp`` is indexed per
    # ``y`` harmonic, so the coupling is block diagonal in ``m``:
    # ``G[(m,n),(p,q)] = alpha_m delta_mp Yhat[n,q]``.  A plain
    # ``kron(diag(alpha), ramp)`` inflates the matrix to ``(M*N)^2`` -- the first
    # version of this routine did exactly that; the shape check below pins it.
    geometric = np.diag(alpha) @ np.kron(np.eye(nb_x), ramp)
    geometric = 0.5 * (geometric + geometric.T)
    if geometric.shape != (len(modes), len(modes)):
        raise AssertionError("gradient geometric matrix has the wrong shape")
    return geometric, modes


def sine_gradient_geometric_matrix_quadrature(
    *,
    length: float,
    width: float,
    basis_x: int,
    basis_y: int,
    stress_gradient: float,
    samples: int = 90,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Same matrix as :func:`sine_gradient_geometric_matrix`, integrated numerically.

    Gauss--Legendre quadrature of ``int N_x(y) w,x^i w,x^j dA`` with no closed
    ramp integral involved; an index or parity error in the closed form is
    detectable by comparing the two.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    nb_x = int(basis_x)
    nb_y = int(basis_y)
    if nb_x < 2 or nb_y < 2:
        raise ValueError("basis_x and basis_y must be at least 2")
    psi = stress_gradient_ramp("stress_gradient", stress_gradient)
    modes = _modes(nb_x, nb_y)
    nodes, weights = np.polynomial.legendre.leggauss(int(samples))
    xs = 0.5 * (nodes + 1.0) * a
    ys = 0.5 * (nodes + 1.0) * b
    wq = np.outer(0.5 * a * weights, 0.5 * b * weights)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    sqrt_w = np.sqrt(wq.ravel())[:, None]
    columns = []
    for m, n in modes:
        kx, ky = m * math.pi / a, n * math.pi / b
        columns.append(kx * np.cos(kx * X) * np.sin(ky * Y))
    gx = np.column_stack(columns) * sqrt_w
    ramp_values = 1.0 + (psi - 1.0) * (Y / b)
    weighted = gx * ramp_values[:, None]
    geometric = (weighted.T @ gx) / (a * b / 4.0)
    return 0.5 * (geometric + geometric.T), modes


def plate_buckling_coefficient(
    *, critical_load: float, rigidity: float, width: float
) -> float:
    """``k = N_cr b^2 / (pi^2 D)`` for comparison with tabulated values."""
    load = _positive_finite("critical_load", critical_load)
    bending = _positive_finite("rigidity", rigidity)
    span = _positive_finite("width", width)
    return load * span**2 / (math.pi**2 * bending)


def ritz_gradient_buckling_load(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    stress_gradient: float,
    basis_x: int = 16,
    basis_y: int = 16,
) -> StressGradientBucklingMode:
    """Critical largest-edge compression of a linearly loaded panel (Galerkin).

    The generalised problem ``K_b phi = lambda G phi`` uses the stiffness-reduced
    driver shared with the uniform and shear modules; ``lambda`` is the critical
    ``N0`` because ``G`` is assembled for unit edge compression.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    coupling = _positive_finite("coupling_rigidity", coupling_rigidity)
    nb_x = int(basis_x)
    nb_y = int(basis_y)
    psi = stress_gradient_ramp("stress_gradient", stress_gradient)
    bending, _, _, _ = sine_energy_matrices(
        length=a, width=b, rigidity_x=dx, rigidity_y=dy,
        coupling_rigidity=coupling, basis_x=nb_x, basis_y=nb_y,
    )
    geometric, modes = sine_gradient_geometric_matrix(
        length=a, width=b, basis_x=nb_x, basis_y=nb_y, stress_gradient=psi,
    )
    multiplier, phi = _smallest_positive_from_symmetric_indefinite(bending, geometric)
    if not math.isfinite(multiplier):
        return StressGradientBucklingMode(
            math.inf, math.inf, psi, (nb_x, nb_y), (0, 0), False)
    dominant = modes[int(np.argmax(np.abs(phi)))]
    return StressGradientBucklingMode(
        multiplier,
        plate_buckling_coefficient(critical_load=multiplier, rigidity=dx, width=b),
        psi,
        (nb_x, nb_y),
        dominant,
        True,
    )


def gradient_buckling_mode_field(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    stress_gradient: float,
    basis_x: int = 24,
    basis_y: int = 24,
    samples: int = 41,
) -> dict[str, object]:
    """Critical mode shape ``w(x, y)`` of a linearly loaded panel on a grid.

    The eigenvector is mapped back through the sine basis, so the returned field
    can be plotted and its Rayleigh quotient re-evaluated independently::

        N0 = phi^T K_b phi / phi^T G phi

    which must equal the returned critical load.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    coupling = _positive_finite("coupling_rigidity", coupling_rigidity)
    nb_x = int(basis_x)
    nb_y = int(basis_y)
    count = int(samples)
    if count < 3:
        raise ValueError("samples must be at least 3")
    psi = stress_gradient_ramp("stress_gradient", stress_gradient)
    bending, _, _, modes = sine_energy_matrices(
        length=a, width=b, rigidity_x=dx, rigidity_y=dy,
        coupling_rigidity=coupling, basis_x=nb_x, basis_y=nb_y,
    )
    geometric, _ = sine_gradient_geometric_matrix(
        length=a, width=b, basis_x=nb_x, basis_y=nb_y, stress_gradient=psi,
    )
    multiplier, phi = _smallest_positive_from_symmetric_indefinite(bending, geometric)
    if not math.isfinite(multiplier):
        raise ValueError("no positive buckling multiplier for this load")
    energy_ratio = float(phi @ bending @ phi) / float(phi @ geometric @ phi)
    coefficients = np.asarray(phi, dtype=float).reshape(nb_x, nb_y)
    peak = float(np.abs(coefficients).max())
    if peak <= 0.0:
        raise ValueError("critical mode has an empty coefficient vector")
    coefficients = coefficients / peak
    xs = np.linspace(0.0, a, count)
    ys = np.linspace(0.0, b, count)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    shape = np.zeros_like(X)
    for m, n in modes:
        shape += coefficients[m - 1, n - 1] * np.sin(
            m * math.pi * X / a) * np.sin(n * math.pi * Y / b)
    peak_field = float(np.abs(shape).max())
    if peak_field > 0.0:
        shape = shape / peak_field
    return {
        "x": xs,
        "y": ys,
        "shape": shape,
        "normalised": True,
        "critical_load": multiplier,
        "coefficient_k": plate_buckling_coefficient(
            critical_load=multiplier, rigidity=dx, width=b),
        "energy_ratio_load": energy_ratio,
        "energy_ratio_relative_deviation": abs(energy_ratio / multiplier - 1.0),
        "dominant_mode": list(modes[int(np.argmax(np.abs(phi)))]),
        "stress_gradient": psi,
        "basis_shape": [nb_x, nb_y],
    }


def reference_gradient_sweep(
    *,
    young: float = 210.0e9,
    poisson: float = 0.3,
    thickness: float = 0.012,
    aspect_ratio: float = 1.0,
    basis: int = 24,
    stress_gradients: tuple[float, ...] = REFERENCE_GRADIENT_SWEEP,
) -> list[dict[str, object]]:
    """Critical coefficient of an isotropic panel for each edge stress ratio."""
    d, _, h = isotropic_plate_rigidities(young, poisson, thickness)
    rows: list[dict[str, object]] = []
    for psi in stress_gradients:
        mode = ritz_gradient_buckling_load(
            length=aspect_ratio, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=psi,
            basis_x=basis, basis_y=basis,
        )
        rows.append({
            "stress_gradient": stress_gradient_ramp("stress_gradient", psi),
            "aspect_ratio": aspect_ratio,
            "coefficient_k": mode.coefficient,
            "critical_load": mode.multiplier,
            "dominant_mode": list(mode.dominant_mode),
        })
    return rows


def run_stress_gradient_qualification(
    *,
    young: float = 210.0e9,
    poisson: float = 0.3,
    thickness: float = 0.012,
    tolerance: float = 0.03,
    basis: int = 24,
) -> dict[str, object]:
    """Qualify the stress-gradient reference on four legs and fail closed.

    Leg 1 is *exact*: with ``psi = 1`` the ramp machinery must reproduce the
    closed-form Navier load of the merged uniform module, i.e. the analytic
    ``k = ((a/m b) + (m b/a))^2`` minimised over the axial half-wave count.  Leg 2 compares the
    tabulated coefficients at ``tolerance`` (3 % by default).  Leg 3 pins the
    closed-form geometric matrix against Gauss--Legendre quadrature of the same
    functional.  Leg 4 requires the critical coefficient to approach the limit
    from above under basis refinement.  A failure raises instead of returning a
    downgraded claim.
    """
    d, _, h = isotropic_plate_rigidities(young, poisson, thickness)

    # ---- leg 1: exact single-mode Navier identity -------------------------
    exact_rows: list[dict[str, object]] = []
    for aspect in (0.5, 0.75, 1.0, 2.0):
        mode = ritz_gradient_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=1.0, basis_x=basis, basis_y=basis,
        )
        navier_load, navier_m, navier_n = navier_uniaxial_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h,
        )
        # exact classical value: the uniform load case keeps one transverse
        # half-wave, so ``k = ((a/m b) + (m b/a))^2`` minimised over the axial
        # half-wave count ``m``.  Taking only ``m = 1`` is correct for
        # ``a/b <= 1`` but wrong for longer panels, where the optimum is
        # ``m = round(a/b)``.
        identity = min(
            (aspect / m + m / aspect) ** 2 for m in range(1, 9)
        )
        error = abs(mode.coefficient / identity - 1.0)
        exact_rows.append({
            "case": f"uniform_navier_a_over_b_{aspect:g}",
            "aspect_ratio": aspect,
            "coefficient_k": mode.coefficient,
            "navier_coefficient_k": navier_load / (math.pi**2 * d),
            "analytic_k_single_mode": identity,
            "navier_mode": [navier_m, navier_n],
            "dominant_mode": list(mode.dominant_mode),
            "relative_error": error,
            "passed": bool(error < 1e-8),
        })

    # ---- leg 2: tabulated coefficients -----------------------------------
    literature_rows: list[dict[str, object]] = []
    for (psi, aspect), literature in LITERATURE_GRADIENT_COEFFICIENTS.items():
        mode = ritz_gradient_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, stress_gradient=psi, basis_x=basis, basis_y=basis,
        )
        error = abs(mode.coefficient / literature - 1.0)
        literature_rows.append({
            "case": f"gradient_psi_{psi:+.2f}_a_over_b_{aspect:g}",
            "stress_gradient": psi,
            "aspect_ratio": aspect,
            "literature_k": literature,
            "coefficient_k": mode.coefficient,
            "dominant_mode": list(mode.dominant_mode),
            "relative_error": error,
            "passed": bool(error < tolerance),
        })

    # ---- leg 3: quadrature of the load potential -------------------------
    quadrature_rows: list[dict[str, object]] = []
    for psi in (1.0, 0.0, -0.5, -1.0):
        closed, _ = sine_gradient_geometric_matrix(
            length=1.0, width=1.0, basis_x=4, basis_y=4, stress_gradient=psi)
        numeric, _ = sine_gradient_geometric_matrix_quadrature(
            length=1.0, width=1.0, basis_x=4, basis_y=4, stress_gradient=psi)
        scale = float(np.abs(closed).max())
        difference = float(np.abs(closed - numeric).max()) / scale
        quadrature_rows.append({
            "case": f"quadrature_psi_{psi:+.2f}",
            "stress_gradient": psi,
            "max_relative_deviation": difference,
            "passed": bool(difference < 1e-10),
        })

    # ---- leg 4: basis refinement ----------------------------------------
    refinement: list[dict[str, object]] = []
    for psi, aspect in ((0.0, 1.0), (-1.0, 2.0)):
        series: list[dict[str, object]] = []
        for order in sorted({4, 8, 12, basis}):
            mode = ritz_gradient_buckling_load(
                length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
                coupling_rigidity=h, stress_gradient=psi,
                basis_x=order, basis_y=order,
            )
            series.append({
                "basis": order,
                "coefficient_k": mode.coefficient,
                "dominant_mode": list(mode.dominant_mode),
            })
        coefficients = [float(row["coefficient_k"]) for row in series]
        monotone = all(
            later <= earlier * (1.0 + 1e-12)
            for earlier, later in zip(coefficients, coefficients[1:])
        )
        refinement.append({
            "stress_gradient": psi,
            "aspect_ratio": aspect,
            "series": series,
            "monotone_from_above": monotone,
            "passed": bool(monotone),
        })

    passed = all(
        bool(row["passed"])
        for row in (*exact_rows, *literature_rows, *quadrature_rows, *refinement)
    )
    result = {
        "schema": "tensorfem.plate-stress-gradient-buckling/1.0",
        "scope": (
            "Kirchhoff variational reference for a flat, simply supported, "
            "isotropic panel under a linearly varying in-plane load; shell "
            "geometry, imperfections, tension-field action, plasticity and "
            "D16/D26 coupling are not covered.  No element in this repository "
            "carries a geometric stiffness for a non-uniform membrane field, so "
            "this qualifies the reference only"
        ),
        "tolerance": tolerance,
        "young": young,
        "poisson": poisson,
        "thickness": thickness,
        "basis": basis,
        "exact_navier": exact_rows,
        "literature": literature_rows,
        "quadrature": quadrature_rows,
        "refinement": refinement,
        "reference_sweep": reference_gradient_sweep(
            young=young, poisson=poisson, thickness=thickness, basis=basis),
        "passed": passed,
    }
    if not passed:
        raise AssertionError("stress gradient buckling qualification failed")
    return result