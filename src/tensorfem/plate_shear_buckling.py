"""Shear and combined in-plane buckling of simply supported orthotropic panels.

``tensorfem.marine_plate_buckling`` qualifies the *uniaxial compression* case of
a simply supported orthotropic plate.  That load case is separable, so a one-term
Navier solution is exact and the finite-difference check collapses to the
eigenvalues of a second-order difference operator.  Pure in-plane **shear** is
not separable: the destabilising term ``2 N_xy w,xy`` couples the half-wave
numbers, so only a truncated-series variational solution resolves it.  This
module adds that load case and qualifies it against three independent legs:

* closed-form assembly - the shear coupling integral
  ``int sin(p pi x/a) cos(m pi x/a) dx`` is evaluated in closed form, and
  :func:`sine_energy_quadrature` recomputes the same three energy matrices with
  Gauss-Legendre quadrature of the energy functionals only, so an index-order or
  parity error in the closed form is detectable;
* the convergence of the variational eigenvalue under basis refinement;
* the classical simply supported shear coefficients ``k_s = 9.34`` at
  ``a/b = 1`` and ``k_s -> 5.34`` for a long panel (Timoshenko & Gere, *Theory of
  Elastic Stability*, 2nd ed., 1961, Table 9.1).

Scope and conventions
---------------------
* Lengths in m, rigidities ``Dx, Dy, H = D12 + 2 D66`` in N m, loads in N/m.
* ``N_x > 0`` and ``N_xy > 0`` denote compression and positive shear; the
  reported values are magnitudes of the applied load.
* The panel is flat and of constant thickness, simply supported on all four
  edges, with uniform in-plane stress.  Shell elements, prebuckling stress
  gradients, imperfections, tension-field (post-buckled) action, plasticity,
  ``D16/D26`` bending--twisting coupling and discrete stiffener tripping are
  **not** covered.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .marine_plate_buckling import isotropic_plate_rigidities, _positive_finite

__all__ = [
    "InPlaneBucklingMode",
    "classical_shear_coefficient",
    "critical_shear_stress",
    "ritz_inplane_buckling_load",
    "run_shear_buckling_qualification",
    "shear_buckling_coefficient",
    "sine_energy_matrices",
    "sine_energy_quadrature",
]

#: Classical simply supported isotropic shear buckling coefficients,
#: ``N_xy,cr = k_s pi^2 D / b^2`` (Timoshenko & Gere 1961, Table 9.1).
LITERATURE_KS_SQUARE = 9.34
LITERATURE_KS_LONG_PANEL = 5.34


@dataclass(frozen=True)
class InPlaneBucklingMode:
    """Critical multiplier of an applied in-plane load direction."""

    #: multiplier on the applied load direction; ``inf`` when no load level
    #: of that sign can buckle the panel.
    multiplier: float
    #: compressive axial component ``N_x`` at buckling [N/m] (0 for pure shear).
    compression_x: float
    #: shear component ``N_xy`` at buckling [N/m] (0 for pure compression).
    shear: float
    basis_shape: tuple[int, int]
    #: dominant half-wave numbers ``(m, n)`` of the critical mode.
    dominant_mode: tuple[int, int]
    buckling: bool


def _validate_direction(compression_ratio: float | None) -> float:
    if compression_ratio is None:
        return 0.0
    if isinstance(compression_ratio, bool) or not isinstance(compression_ratio, (int, float)):
        raise TypeError("compression_ratio must be a real number or None")
    ratio = float(compression_ratio)
    if not math.isfinite(ratio):
        raise ValueError("compression_ratio must be finite")
    if ratio < 0.0:
        raise ValueError("compression_ratio must be non-negative")
    return ratio


def _smallest_positive_from_symmetric_indefinite(
    stiffness: np.ndarray, geometric: np.ndarray
) -> tuple[float, np.ndarray]:
    """Smallest positive ``lambda`` of ``stiffness phi = lambda geometric phi``.

    ``stiffness`` must be symmetric positive definite and ``geometric``
    symmetric, possibly indefinite (shear).  The pencil is reduced with a
    Cholesky factor of the stiffness, so only a symmetric eigenproblem remains;
    the buckling load is the reciprocal of the largest positive eigenvalue of
    the reduced problem.  ``(inf, zeros)`` means nothing buckles.

    Note
    ----
    Only ``numpy.linalg`` kernels are used.  The BLAS/LAPACK build of
    ``scipy.linalg.cholesky``/``cho_factor`` in the qualification environment
    segfaults for matrices of order 64 and above (reproduced with scipy 1.16.0
    against numpy 2.4.6; no exception, the process dies), which killed the first
    version of this routine at ``n = 100``.  ``numpy.linalg`` is unaffected.
    """
    n = stiffness.shape[0]
    factor = np.linalg.cholesky(stiffness)
    # congruent transform L^-1 G L^-T of the L L^T = K_b pencil.  The earlier
    # version built L^-T G L^-T (``solve(factor.T, reduced.T)`` on the
    # unsymmetrised product), which is a *different* matrix: it coincides only
    # when the Cholesky factor of the bending matrix is diagonal, i.e. exactly
    # in the isotropic/orthotropic cases this module ships. Any pencil with a
    # non-diagonal stiffness (the Hermite-element reference of
    # ``plate_bending_buckling.infinite_plate_bending_coefficient``, for one)
    # was therefore solved for the wrong eigenvalues.
    reduced = np.linalg.solve(factor, geometric)
    reduced = np.linalg.solve(factor, reduced.T).T
    reduced = 0.5 * (reduced + reduced.T)
    mu, vectors = np.linalg.eigh(reduced)
    positive = mu > 0.0
    if not np.any(positive):
        return math.inf, np.zeros(n)
    index = int(np.argmax(np.where(positive, mu, -np.inf)))
    if not math.isfinite(mu[index]) or mu[index] <= 0.0:
        return math.inf, np.zeros(n)
    phi = np.linalg.solve(factor.T, vectors[:, index])
    return 1.0 / float(mu[index]), phi


def _sine_shear_integral(p: int, m: int) -> float:
    """``(1/a) int_0^a sin(p pi x/a) cos(m pi x/a) dx``.

    Non-zero only for ``p`` and ``m`` of opposite parity, where the closed value
    is ``2p / (pi (p^2 - m^2))``.
    """
    if p == m or (p + m) % 2 == 0:
        return 0.0
    return 2.0 * p / (math.pi * (p * p - m * m))


def _modes(basis_x: int, basis_y: int) -> list[tuple[int, int]]:
    return [(m, n) for m in range(1, basis_x + 1) for n in range(1, basis_y + 1)]


def sine_energy_matrices(
    *, length: float, width: float, rigidity_x: float, rigidity_y: float,
    coupling_rigidity: float, basis_x: int, basis_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Bending, uniaxial and shear energy matrices in the double-sine basis.

    The returned matrices are ``int`` of ``Dx w,xx^2 + 2H w,xx w,yy + Dy w,yy^2``
    (bending), ``w,x^2`` (uniaxial) and ``2 w,x w,y`` (shear) over the panel,
    normalised by the common factor ``a*b/4``; the generalised eigenproblem is
    unchanged by that factor.  ``K_b`` is diagonal because the sine modes are
    orthogonal under the orthotropic bending energy.
    """
    a, b = length, width
    modes = _modes(basis_x, basis_y)
    size = len(modes)
    alpha = np.array([(m * math.pi / a) ** 2 for m, _ in modes])
    beta = np.array([(n * math.pi / b) ** 2 for _, n in modes])
    alpha_root = np.sqrt(alpha)
    beta_root = np.sqrt(beta)
    bending = np.diag(
        rigidity_x * alpha**2
        + 2.0 * coupling_rigidity * alpha * beta
        + rigidity_y * beta**2
    )
    uniaxial = np.diag(alpha)
    shear = np.zeros((size, size))
    for row, (m, n) in enumerate(modes):
        # int cos(alpha_m x) sin(alpha_p x) dx -> sine index (column) first
        fx = np.array([_sine_shear_integral(p, m) for p, _ in modes])
        # int sin(beta_n y) cos(beta_q y) dy -> sine index (row) first
        fy = np.array([_sine_shear_integral(n, q) for _, q in modes])
        shear[row] = 8.0 * alpha_root[row] * beta_root * fx * fy
    # ``int w,x w,y dA`` is symmetric as a bilinear form of the field, but this
    # basis matrix is not symmetric pair by pair (swapping the two modes swaps
    # which one carries the x derivative), so stationarity of the total
    # potential uses the symmetric part only.
    shear = 0.5 * (shear + shear.T)
    return bending, uniaxial, shear, modes


def sine_energy_quadrature(
    *, length: float, width: float, rigidity_x: float, rigidity_y: float,
    coupling_rigidity: float, basis_x: int, basis_y: int,
    samples: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Same matrices as :func:`sine_energy_matrices`, integrated numerically.

    Gauss-Legendre quadrature of the energy functionals for every basis pair,
    with no closed-form integral involved; used to verify the closed assembly.
    """
    a, b = length, width
    modes = _modes(basis_x, basis_y)
    nodes, weights = np.polynomial.legendre.leggauss(int(samples))
    xs = 0.5 * (nodes + 1.0) * a
    ys = 0.5 * (nodes + 1.0) * b
    wq = np.outer(0.5 * a * weights, 0.5 * b * weights)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    WQ = wq.ravel()
    X, Y = X.ravel(), Y.ravel()
    phis, phix, phiy, phixx, phiyy = [], [], [], [], []
    for m, n in modes:
        kx, ky = m * math.pi / a, n * math.pi / b
        s_x, c_x = np.sin(kx * X), np.cos(kx * X)
        s_y, c_y = np.sin(ky * Y), np.cos(ky * Y)
        phis.append(s_x * s_y)
        phix.append(kx * c_x * s_y)
        phiy.append(ky * s_x * c_y)
        phixx.append(-kx**2 * s_x * s_y)
        phiyy.append(-ky**2 * s_x * s_y)
    sqrt_w = np.sqrt(WQ)[:, None]
    xx = np.column_stack(phixx) * sqrt_w
    yy = np.column_stack(phiyy) * sqrt_w
    gx = np.column_stack(phix) * sqrt_w
    gy = np.column_stack(phiy) * sqrt_w
    scale = a * b / 4.0
    bending = (rigidity_x * (xx.T @ xx) + rigidity_y * (yy.T @ yy)) / scale
    cross = coupling_rigidity * (xx.T @ yy) / scale
    bending = bending + cross + cross.T
    uniaxial = (gx.T @ gx) / scale
    shear = (gx.T @ gy) / scale
    shear = shear + shear.T
    return bending, uniaxial, shear, modes


def ritz_inplane_buckling_load(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    basis_x: int = 16,
    basis_y: int = 16,
    compression_ratio: float | None = None,
) -> InPlaneBucklingMode:
    """Galerkin buckling load of a simply supported panel.

    ``compression_ratio`` is ``N_xy / N_x`` of the applied load direction.  With
    ``None`` the load is pure shear and ``multiplier`` is ``N_xy,cr``; otherwise
    ``multiplier`` is ``N_x,cr`` and the shear component is
    ``compression_ratio * multiplier``.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    nb_x, nb_y = int(basis_x), int(basis_y)
    if nb_x < 2 or nb_y < 2:
        raise ValueError("basis_x and basis_y must be at least 2")
    ratio = _validate_direction(compression_ratio)
    bending, uniaxial, shear, modes = sine_energy_matrices(
        length=a, width=b, rigidity_x=dx, rigidity_y=dy,
        coupling_rigidity=h, basis_x=nb_x, basis_y=nb_y,
    )
    if compression_ratio is None:
        direction, compression = shear, 0.0
    else:
        direction = uniaxial + ratio * shear
        compression = 1.0
    multiplier, phi = _smallest_positive_from_symmetric_indefinite(bending, direction)
    if not math.isfinite(multiplier):
        return InPlaneBucklingMode(math.inf, 0.0, 0.0, (nb_x, nb_y), (0, 0), False)
    dominant = modes[int(np.argmax(np.abs(phi)))]
    return InPlaneBucklingMode(
        multiplier,
        compression * multiplier,
        multiplier if compression_ratio is None else ratio * multiplier,
        (nb_x, nb_y),
        dominant,
        True,
    )


def shear_buckling_coefficient(
    *, shear_load: float, rigidity: float, width: float
) -> float:
    """``k_s = N_xy b^2 / (pi^2 D)`` for comparison with tabulated values."""
    load = _positive_finite("shear_load", shear_load)
    bending = _positive_finite("rigidity", rigidity)
    span = _positive_finite("width", width)
    return load * span**2 / (math.pi**2 * bending)


def classical_shear_coefficient(aspect_ratio: float) -> float:
    """Classical approximation ``k_s = 5.34 + 4/(a/b)^2`` (Timoshenko & Gere)."""
    ratio = _positive_finite("aspect_ratio", aspect_ratio)
    return 5.34 + 4.0 / ratio**2


def critical_shear_stress(*, shear_load: float, thickness: float) -> float:
    """Mean shear stress ``tau_cr = N_xy / t`` in Pa."""
    load = _positive_finite("shear_load", shear_load)
    t = _positive_finite("thickness", thickness)
    return load / t


def run_shear_buckling_qualification(
    *, young: float = 210.0e9, poisson: float = 0.3, thickness: float = 0.012,
    tolerance: float = 0.03, basis: int = 16,
) -> dict[str, object]:
    """Qualify shear buckling against the classical coefficients (3 % gate)."""
    d, _, h = isotropic_plate_rigidities(young, poisson, thickness)
    rows: list[dict[str, object]] = []
    for name, aspect, literature in (
        ("square_shear", 1.0, LITERATURE_KS_SQUARE),
        ("long_panel_shear", 6.0, LITERATURE_KS_LONG_PANEL),
    ):
        mode = ritz_inplane_buckling_load(
            length=aspect, width=1.0, rigidity_x=d, rigidity_y=d,
            coupling_rigidity=h, basis_x=basis, basis_y=basis,
        )
        coefficient = shear_buckling_coefficient(
            shear_load=mode.multiplier, rigidity=d, width=1.0)
        error = abs(coefficient / literature - 1.0)
        rows.append({
            "case": name,
            "aspect_ratio": aspect,
            "literature_k_s": literature,
            "coefficient_k_s": coefficient,
            "dominant_mode": list(mode.dominant_mode),
            "critical_shear_load": mode.multiplier,
            "critical_shear_stress": critical_shear_stress(
                shear_load=mode.multiplier, thickness=thickness),
            "relative_error": error,
            "passed": bool(error < tolerance),
        })
    # Orthotropic (smeared longitudinally stiffened) panel: no tabulated anchor
    # exists for this combination, so only self-consistency is reported.
    stiffened = d + young * 3.0e-4 * 0.04**2 / 0.5
    ortho = ritz_inplane_buckling_load(
        length=2.0, width=1.0, rigidity_x=stiffened, rigidity_y=d,
        coupling_rigidity=h, basis_x=basis, basis_y=basis,
    )
    isotropic_2to1 = ritz_inplane_buckling_load(
        length=2.0, width=1.0, rigidity_x=d, rigidity_y=d,
        coupling_rigidity=h, basis_x=basis, basis_y=basis,
    )
    gain = ortho.multiplier / isotropic_2to1.multiplier - 1.0
    rows.append({
        "case": "orthotropic_panel_shear",
        "aspect_ratio": 2.0,
        "literature_k_s": None,
        "coefficient_k_s": shear_buckling_coefficient(
            shear_load=ortho.multiplier, rigidity=d, width=1.0),
        "dominant_mode": list(ortho.dominant_mode),
        "critical_shear_load": ortho.multiplier,
        "critical_shear_stress": critical_shear_stress(
            shear_load=ortho.multiplier, thickness=thickness),
        "relative_error": None,
        "stiffened_to_isotropic_gain": gain,
        "passed": bool(gain > 0.0),
    })
    passed = all(bool(row["passed"]) for row in rows)
    if not passed:
        raise AssertionError("shear buckling qualification failed")
    return {
        "schema": "tensorfem.plate-shear-buckling/1.0",
        "scope": "TensorFEM variational plate buckling; no TensorLBM/CFD",
        "tolerance": tolerance,
        "young": young,
        "poisson": poisson,
        "thickness": thickness,
        "basis": basis,
        "cases": rows,
        "passed": passed,
    }