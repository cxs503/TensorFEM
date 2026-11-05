"""Converged double-sine Galerkin reference for simply supported plate buckling.

Why this module exists
----------------------
``tensorfem.marine_plate_combined_buckling`` solves the combined in-plane buckling
eigenproblem with a finite-difference pencil and anchors its pure-shear branch on
the *asymptotic* coefficient ``k_s = 5.34 + 4 (b/a)**2``.  Its own report records
that the ``a/b = 2`` branch misses that anchor by 3.55 % and that an earlier
Rayleigh--Ritz cross-check disagreed with the finite-difference value by 3.4 %.
Neither discrepancy could be adjudicated, because the asymptotic fit is *not* a
solution of the eigenvalue problem and the earlier Ritz implementation was never
validated against an exactly known case.

This module supplies the missing adjudicator: a whole-domain Galerkin solution in
the exact eigenfunction space of the simply supported plate.  For ``w = 0`` and
``M_n = 0`` on all four edges the Kirchhoff buckling problem

    D_x w,xxxx + 2 H w,xxyy + D_y w,yyyy
        = N_x w,xx + N_y w,yy + 2 N_xy w,xy                     (1)

is diagonalised by ``phi_mn = sin(m pi x / a) sin(n pi y / b)``: every basis
function already satisfies both boundary conditions, so the sine series is
*complete* in the admissible space and the Galerkin truncation converges to the
exact eigenvalue from above.  The in-plane coefficients break separability only
through the twist term, which couples ``(m, n)`` to ``(i, j)`` whenever both index
parities flip.

Weak form used here (``w = 0`` on the boundary kills every boundary term)::

    int [ D_x w,xx phi,xx + H (w,xx phi,yy + w,yy phi,xx) + D_y w,yy phi,yy ]
      = int [ N_x w,x phi,x + N_y w,y phi,y + N_xy (w,x phi,y + w,y phi,x) ]

so the two pencils are

    K_ij,mn = (a b / 4) [ D_x kx_m^4 + 2 H kx_m^2 ky_n^2 + D_y ky_n^4 ] delta
    G_ij,mn = (a b / 4) [ N_x kx_m^2 + N_y ky_n^2 ] delta
              + N_xy * 2 * g_ij,mn

with ``kx_m = m pi / a``, ``ky_n = n pi / b`` and the exact twist coupling

    g_ij,mn = - m n i j chi_mi chi_nj / ((m^2 - i^2) (n^2 - j^2)),
    chi_mi  = 0 if m + i is even else 2,      g_ij,mn = 0 if m == i or n == j.

``g`` is symmetric by construction, so the pencil ``(K, G)`` is symmetric even
though ``G`` is indefinite for pure shear.  Because ``G`` is indefinite the
buckling load is *not* the smallest algebraic eigenvalue: the panel buckles at the
smallest positive ``N`` for which ``K - N G`` is singular, i.e. at
``1 / max{positive eigenvalues of (K, G)}``.

Two numerical traps that this module handles explicitly
------------------------------------------------------
1. The parity factor ``chi = 1 - (-1)^(m + i)`` is ``0`` or ``2``, **not** a
   ``0/1`` indicator.  Using ``1`` scales the whole twist block by four and moves
   the square-panel coefficient from 9.3245 to 37.298 (a perfectly smooth, fully
   converged, four-times-wrong answer).  ``twist_matrix_quadrature_check``
   compares the closed form against Gauss--Legendre quadrature of the same
   integrals and is run by the test suite.
2. ``K`` is symmetric positive definite but ``G`` is not, so the reduced matrix
   ``K^-1 G`` is **non-symmetric**; ``numpy.linalg.eigvalsh`` on it silently reads
   only one triangle and returns a spectrum that does not even contain the
   values of principal sub-blocks.  The Cholesky congruence
   ``L^-1 G L^-T`` is used instead (``scipy.linalg.eigh`` on the pencil segfaults
   in this environment, see ``plate_ritz_reference``).

Evidence and limits
-------------------
* Isotropic uniaxial compression reduces to the exact Navier load (``4 pi^2 D /
  b^2`` for the square) to machine precision; equal biaxial to ``2 pi^2 D / b^2``.
* The square pure-shear coefficient converges to 9.3245, i.e. 0.17 % below the
  classical tabulated ``k_s = 9.34``.
* The long-panel limit reproduces the classical asymptote (5.4146 at ``a/b = 8``,
  decreasing toward 5.34).
* The module covers *thin-plate linearised buckling only*.  It says nothing about
  plasticity, post-buckling, stiffeners, elastically restrained edges, or 3-D
  shell stress.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SpectralBucklingResult",
    "spectral_buckling_load",
    "shear_buckling_coefficient",
    "converged_shear_buckling_coefficient",
    "twist_matrix_quadrature_check",
    "classical_asymptotic_coefficient",
    "SQUARE_SHEAR_LITERATURE_COEFFICIENT",
]

#: Classical tabulated pure-shear coefficient of a simply supported square panel
#: (Timoshenko & Gere, *Theory of Elastic Stability*, 2nd ed., Art. 9.3).
SQUARE_SHEAR_LITERATURE_COEFFICIENT = 9.34

_DEFAULT_BUDGETS: tuple[int, ...] = (8, 12, 16, 20, 24, 28, 32)


@dataclass(frozen=True)
class SpectralBucklingResult:
    """Buckling load of one Galerkin truncation.

    ``critical_line_load`` is the smallest positive ``N`` such that
    ``N_x = longitudinal_share * N``, ``N_y = transverse_share * N`` and
    ``N_xy = shear_share * N`` make the plate buckle.
    """

    critical_line_load: float
    coefficients: tuple[float, float, float]
    modes: tuple[int, int]
    adopted: bool = False
    history: tuple[tuple[int, int, float], ...] = field(default_factory=tuple)

    @property
    def longitudinal_share(self) -> float:
        return self.coefficients[0]

    @property
    def transverse_share(self) -> float:
        return self.coefficients[1]

    @property
    def shear_share(self) -> float:
        return self.coefficients[2]


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {number}")
    return number


def _parity(index_a: int, index_b: int) -> float:
    """``1 - (-1)^(a + b)``; ``0`` or ``2``, never a ``0/1`` indicator."""
    return 0.0 if (index_a + index_b) % 2 == 0 else 2.0


def _cos_sin_integral(mode: int, shape: int, length: float) -> float:
    """``int_0^L cos(mode pi x / L) sin(shape pi x / L) dx`` in closed form."""
    if mode == shape:
        return 0.0
    return -length * shape * _parity(mode, shape) / (math.pi * (mode**2 - shape**2))


def _sin_cos_integral(mode: int, shape: int, length: float) -> float:
    """``int_0^L sin(mode pi x / L) cos(shape pi x / L) dx`` in closed form."""
    if mode == shape:
        return 0.0
    return length * mode * _parity(mode, shape) / (math.pi * (mode**2 - shape**2))


def twist_matrix_quadrature_check(modes: int = 4) -> float:
    """Largest absolute discrepancy between the closed-form twist block and quadrature.

    The twist entries are the only non-diagonal ones, and they are the entry point
    for the parity-factor trap.  Recomputing a subset with Gauss--Legendre
    quadrature is cheap and catches a wrong normalisation before any eigensolve.
    """
    size = _integer("modes", modes, minimum=2)
    a = b = 1.0
    nodes, weights = np.polynomial.legendre.leggauss(64)
    x = 0.5 * (nodes + 1.0) * a
    y = 0.5 * (nodes + 1.0) * b
    wx = 0.5 * weights * a
    wy = 0.5 * weights * b
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    weight = np.meshgrid(wx, wy, indexing="ij")
    weight = weight[0] * weight[1]
    worst = 0.0
    for m in range(1, size + 1):
        for n in range(1, size + 1):
            for i in range(1, size + 1):
                for j in range(1, size + 1):
                    if m == i or n == j:
                        continue
                    # int phi_mn,x phi_ij,y  with phi = sin.sin
                    integrand = (
                        (m * math.pi / a) * np.cos(m * math.pi * grid_x / a)
                        * np.sin(n * math.pi * grid_y / b)
                        * (j * math.pi / b) * np.sin(i * math.pi * grid_x / a)
                        * np.cos(j * math.pi * grid_y / b)
                    )
                    quadrature = float((integrand * weight).sum())
                    closed = (
                        -m * n * i * j * _parity(m, i) * _parity(n, j)
                        / ((m**2 - i**2) * (n**2 - j**2))
                    )
                    worst = max(worst, abs(quadrature - closed))
    return worst


def _pencils(
    length: float, width: float, rigidity_x: float, rigidity_y: float,
    coupling_rigidity: float, modes_x: int, modes_y: int,
    longitudinal_share: float, transverse_share: float, shear_share: float,
) -> tuple[np.ndarray, np.ndarray]:
    a, b = length, width
    indices = [(m, n) for m in range(1, modes_x + 1) for n in range(1, modes_y + 1)]
    size = len(indices)
    stiffness = np.zeros((size, size))
    geometry = np.zeros((size, size))
    area = 0.25 * a * b
    wave_x = [m * math.pi / a for m in range(1, modes_x + 1)]
    wave_y = [n * math.pi / b for n in range(1, modes_y + 1)]
    for p, (i, j) in enumerate(indices):
        kx_i, ky_j = wave_x[i - 1], wave_y[j - 1]
        stiffness[p, p] = area * (
            rigidity_x * kx_i**4
            + 2.0 * coupling_rigidity * kx_i**2 * ky_j**2
            + rigidity_y * ky_j**4
        )
        geometry[p, p] = area * (
            longitudinal_share * kx_i**2 + transverse_share * ky_j**2
        )
    if shear_share != 0.0:
        geometry += 2.0 * shear_share * _twist_block(modes_x, modes_y)
    return stiffness, geometry


def _twist_block(modes_x: int, modes_y: int) -> np.ndarray:
    """Exact twist-coupling block ``g_ij,mn`` as a Kronecker product.

    ``g_ij,mn = -T[i, m] * T[j, n]`` with ``T[i, m] = i m chi / (m^2 - i^2)``;
    ``T`` is antisymmetric and ``g`` is symmetric, which is what keeps the
    pencil symmetric for a purely shear-loaded panel.
    """
    def factor(count: int) -> np.ndarray:
        index = np.arange(1, count + 1)
        difference = index[None, :] ** 2 - index[:, None] ** 2
        parity = np.where((index[None, :] + index[:, None]) % 2 == 0, 0.0, 2.0)
        numerator = index[None, :] * index[:, None] * parity
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.where(difference == 0, 0.0, numerator / difference)
        return value

    return -np.kron(factor(modes_x), factor(modes_y))


def _smallest_positive_inverse_eigenvalue(
    stiffness: np.ndarray, geometry: np.ndarray,
) -> float:
    """``1 / max{positive eigenvalue of (K, G)}`` = smallest positive buckling load."""
    try:
        factor = np.linalg.cholesky(stiffness)
    except np.linalg.LinAlgError as error:  # pragma: no cover - defensive
        raise ValueError("stiffness pencil is not positive definite") from error
    left = np.linalg.solve(factor, geometry)
    reduced = np.linalg.solve(factor, left.T).T
    # Cholesky congruence keeps the spectrum but drops symmetry at round-off level.
    reduced = 0.5 * (reduced + reduced.T)
    values = np.linalg.eigvalsh(reduced)
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("load combination admits no positive buckling load")
    return 1.0 / float(positive.max())


def spectral_buckling_load(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    modes_x: int = 24,
    modes_y: int = 24,
    longitudinal_share: float = 1.0,
    transverse_share: float = 0.0,
    shear_share: float = 0.0,
) -> SpectralBucklingResult:
    """Buckling load of one double-sine Galerkin truncation of problem (1)."""
    a = _finite("length", length, positive=True)
    b = _finite("width", width, positive=True)
    dx = _finite("rigidity_x", rigidity_x, positive=True)
    dy = _finite("rigidity_y", rigidity_y, positive=True)
    dh = _finite("coupling_rigidity", coupling_rigidity, positive=True)
    mx = _integer("modes_x", modes_x, minimum=2)
    my = _integer("modes_y", modes_y, minimum=2)
    lx = _finite("longitudinal_share", longitudinal_share)
    ty = _finite("transverse_share", transverse_share)
    sxy = _finite("shear_share", shear_share)
    if lx == 0.0 and ty == 0.0 and sxy == 0.0:
        raise ValueError("at least one load share must be non-zero")
    if lx < 0.0 or ty < 0.0:
        raise ValueError(
            "shares are compressive (line loads `share * N` with `N > 0`); "
            "tension is outside the scope of this reference"
        )
    stiffness, geometry = _pencils(a, b, dx, dy, dh, mx, my, lx, ty, sxy)
    load = _smallest_positive_inverse_eigenvalue(stiffness, geometry)
    return SpectralBucklingResult(
        critical_line_load=load, coefficients=(lx, ty, sxy), modes=(mx, my))


def shear_buckling_coefficient(
    aspect_ratio: float, *, modes_x: int = 32, modes_y: int = 24, width: float = 1.0,
) -> float:
    """Pure-shear coefficient ``k_s = N_xy,cr b^2 / (pi^2 D)`` for one truncation.

    Isotropic plate (``D_x = D_y = H = D``); ``aspect_ratio = a / b``.
    """
    aspect = _finite("aspect_ratio", aspect_ratio, positive=True)
    b = _finite("width", width, positive=True)
    a = aspect * b
    result = spectral_buckling_load(
        length=a, width=b, rigidity_x=1.0, rigidity_y=1.0, coupling_rigidity=1.0,
        modes_x=modes_x, modes_y=modes_y, longitudinal_share=0.0,
        transverse_share=0.0, shear_share=1.0)
    return result.critical_line_load * b**2 / (math.pi**2)


def converged_shear_buckling_coefficient(
    aspect_ratio: float,
    *,
    budgets: tuple[int, ...] = _DEFAULT_BUDGETS,
    width: float = 1.0,
    tolerance: float = 1e-4,
) -> dict[str, object]:
    """Refine the Galerkin budget until ``k_s`` stops moving.

    The truncation error of the sine basis decreases monotonically (it is a Ritz
    upper bound for the buckling load), so the last pair of budgets gives an
    honest residual instead of a manufactured one.  ``converged`` is ``False``
    when the residual stays above ``tolerance`` -- never silently rounded away.
    """
    aspect = _finite("aspect_ratio", aspect_ratio, positive=True)
    if len(budgets) < 2:
        raise ValueError("at least two mode budgets are required")
    table: list[tuple[int, int, float]] = []
    for budget in budgets:
        modes = _integer("budget", budget, minimum=2)
        # A square truncation is used for every aspect ratio: the twist coupling
        # spreads over all index pairs, and refining both directions together is
        # what makes the residual in the returned table meaningful.  Scaling the
        # x budget with ``aspect`` only inflates the dense pencil (192 x 24 modes
        # at a/b = 8) without changing the converged value.
        modes_x = modes
        coefficient = shear_buckling_coefficient(
            aspect, modes_x=modes_x, modes_y=modes, width=width)
        table.append((modes_x, modes, coefficient))
    residual = abs(table[-1][2] / table[-2][2] - 1.0)
    converged = residual <= tolerance and table[-1][2] <= table[-2][2] + 1e-12
    return {
        "aspect_ratio": aspect,
        "coefficient": table[-1][2],
        "modes": (table[-1][0], table[-1][1]),
        "residual": residual,
        "converged": converged,
        "tolerance": tolerance,
        "table": tuple(table),
    }


def classical_asymptotic_coefficient(aspect_ratio: float) -> float:
    """``k_s = 5.34 + 4 (b/a)^2`` -- quoted for comparison, not as a reference."""
    aspect = _finite("aspect_ratio", aspect_ratio, positive=True)
    return 5.34 + 4.0 / aspect**2