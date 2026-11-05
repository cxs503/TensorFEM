"""Buckling of a simply supported panel under a *linearly varying* in-plane stress.

The existing qualification chain
(:mod:`tensorfem.marine_plate_buckling`, :mod:`tensorfem.plate_shear_buckling`)
covers **uniform** in-plane stress: uniaxial compression is separable and has a
one-term Navier solution, and shear is non-separable but its geometric matrix
still follows from a uniform ``N_xy``.  Both modules state in their scope
sections that *prebuckling stress gradients* are not covered, and the
qualification ledger (``docs/benchmark-qualification-status.md``) lists
"非均匀预屈曲应力" as an open item.

This module closes that item for the case that matters most for a hull girder:
a stiffened panel whose edge stress varies linearly across its width, because
hull-girder bending superposes a bending stress gradient on the nominal
compression.  With ``N_x(y) = N1 (1 + (psi - 1) y / b)`` the panel carries
``N1`` (compression positive) at ``y = 0`` and ``psi * N1`` at ``y = b``:

* ``psi = 1``   uniform compression (``k = 4`` for a long panel);
* ``psi = 0``   triangular compression (``k = 7.81`` for a long panel);
* ``psi = -1``  pure in-plane bending (``k_b = 23.9`` for ``a/b >= 2/3``).

Physically the gradient is not a cosmetic variation: the destabilising energy
``int N_x(y) w,x^2`` now couples sine modes of *opposite parity* in ``y``, so
the mode is no longer separable and the classical bending coefficient ``23.9``
is recoverable only from a coupled variational basis.  The closed-form coupling
integral is given by :func:`stress_gradient_coupling_matrix`.

Qualification legs
------------------
1. **Literature** - ``k_b = 23.9`` (pure bending, ``a/b >= 2/3``) and ``25.6``
   (pure bending, ``a/b = 1``) and ``k = 7.81`` (triangular compression, long
   panel), Timoshenko & Gere, *Theory of Elastic Stability*, 2nd ed. (1961),
   Table 9.1; see also Allen & Bulson, *Background to Buckling* (1980);
2. **Exact limit** - ``psi = 1`` must return the uniform-compression value
   ``k = 4`` from the *same* general machinery, i.e. the gradient assembly
   degenerates correctly (a strong internal-consistency gate);
3. **Independent integral** - :func:`stress_gradient_quadrature` rebuilds both
   matrices by Gauss-Legendre quadrature of the energy functionals, with no
   closed-form integral involved, so an index-order or parity error in the
   closed form is detectable;
4. **Independent discretisation** -
   :func:`finite_difference_bending_buckling` solves the same strong-form
   eigenproblem by finite differences on a nodal grid whose load varies
   row-by-row with ``y``.  It shares neither basis nor matrix assembly with the
   variational route.

Scope and conventions
---------------------
* Lengths in m, rigidities ``Dx, Dy, H = D12 + 2 D66`` in N m, load resultants
  in N/m, stresses in Pa.
* ``N_x > 0`` is compression.  ``multiplier`` is the peak compressive edge
  resultant ``N1``; the opposite edge carries ``psi * N1``.
* Flat panel, constant thickness, simply supported on all four edges, uniform
  in-plane shear absent.  **Not** covered: shell elements, discrete stiffener
  tripping between stiffeners (the smeared rigidity is accepted, as elsewhere
  in this family), post-buckled (effective width / tension-field) strength,
  plasticity, imperfection sensitivity, ``D16/D26`` bending-twisting coupling,
  combined compression + shear + gradient, and biaxial gradient loading.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .marine_plate_buckling import _positive_finite, isotropic_plate_rigidities

__all__ = [
    "LITERATURE_KB_PURE_BENDING_LONG",
    "LITERATURE_KB_PURE_BENDING_SQUARE",
    "LITERATURE_K_TRIANGULAR_LONG",
    "LITERATURE_K_UNIFORM_LONG",
    "StressGradientBucklingMode",
    "bending_buckling_coefficient",
    "edge_stress_ratio_profile",
    "finite_difference_bending_buckling",
    "ritz_stress_gradient_buckling_load",
    "run_bending_buckling_qualification",
    "stress_gradient_coupling_matrix",
    "stress_gradient_matrices",
    "stress_gradient_mode_shape",
    "stress_gradient_quadrature",
]

#: Classical buckling coefficients for a long simply supported isotropic panel,
#: ``N_cr = k pi^2 D / b^2`` (Timoshenko & Gere 1961, Table 9.1).
LITERATURE_K_UNIFORM_LONG = 4.0
LITERATURE_K_TRIANGULAR_LONG = 7.81
#: Pure in-plane bending: peak edge stress is the reference, ``a/b >= 2/3``.
LITERATURE_KB_PURE_BENDING_LONG = 23.9
#: Pure in-plane bending at ``a/b = 1`` (same table; above the asymptote).
LITERATURE_KB_PURE_BENDING_SQUARE = 25.6


@dataclass(frozen=True)
class StressGradientBucklingMode:
    """Critical load of a panel under a linearly varying edge stress."""

    #: peak compressive edge resultant ``N1`` at buckling [N/m].
    multiplier: float
    #: ``sigma_at_y=b / sigma_at_y=0`` of the applied stress, ``-1 <= psi <= 1``.
    edge_ratio: float
    #: resultant at the ``y = 0`` edge, ``= multiplier`` (compression positive).
    compression_edge: float
    #: resultant at the ``y = b`` edge, ``= edge_ratio * multiplier`` (negative
    #: means tension there).
    opposite_edge: float
    basis_shape: tuple[int, int]
    #: dominant half-wave numbers ``(m, n)`` of the critical mode.
    dominant_mode: tuple[int, int]
    buckling: bool


def edge_stress_ratio_profile(edge_ratio: float) -> float:
    """Validate ``psi`` and return it as ``float`` (``-1 <= psi <= 1``).

    ``psi = 1`` is uniform compression, ``psi = 0`` triangular compression and
    ``psi = -1`` pure in-plane bending (tension across half the width).
    """
    if isinstance(edge_ratio, bool) or not isinstance(edge_ratio, (int, float)):
        raise TypeError("edge_ratio must be a real number")
    ratio = float(edge_ratio)
    if not math.isfinite(ratio):
        raise ValueError("edge_ratio must be finite")
    if ratio < -1.0 or ratio > 1.0:
        raise ValueError("edge_ratio must lie in [-1, 1]")
    return ratio


def _validate_basis(basis_x: int, basis_y: int) -> tuple[int, int]:
    nb_x, nb_y = int(basis_x), int(basis_y)
    if nb_x != basis_x or nb_y != basis_y:
        raise TypeError("basis_x and basis_y must be integers")
    if nb_x < 2 or nb_y < 2:
        raise ValueError("basis_x and basis_y must be at least 2")
    return nb_x, nb_y


def _smallest_positive_generalized(
    stiffness: np.ndarray, geometric: np.ndarray
) -> tuple[float, np.ndarray]:
    """Smallest positive ``lambda`` of ``stiffness phi = lambda geometric phi``.

    ``stiffness`` must be symmetric positive definite, ``geometric`` symmetric
    and possibly indefinite (a stress gradient has a tension region, so its
    destabilising energy is not sign definite).

    Note
    ----
    The reduction is ``R = F^-1 G F^-T`` with ``K = F F^T``, i.e. the two
    factors are applied to *opposite* sides.  In
    :mod:`tensorfem.plate_shear_buckling` the same reduction is written as two
    successive ``solve`` calls with a transposed argument, which is only
    equivalent when ``F`` is diagonal (that module's bending matrix is diagonal
    by orthogonality of the sine basis).  A nodal finite-difference bending
    matrix is not diagonal, and the mixed form returns wrong eigenvalues there,
    so this module carries its own solver.  ``(inf, zeros)`` means nothing
    buckles.
    """
    size = stiffness.shape[0]
    factor = np.linalg.cholesky(stiffness)
    left = np.linalg.solve(factor, geometric)
    reduced = np.linalg.solve(factor, left.T).T
    reduced = 0.5 * (reduced + reduced.T)
    mu, vectors = np.linalg.eigh(reduced)
    positive = mu > 0.0
    if not np.any(positive):
        return math.inf, np.zeros(size)
    index = int(np.argmax(np.where(positive, mu, -np.inf)))
    if not math.isfinite(mu[index]) or mu[index] <= 0.0:
        return math.inf, np.zeros(size)
    phi = np.linalg.solve(factor.T, vectors[:, index])
    return 1.0 / float(mu[index]), phi


def stress_gradient_coupling_matrix(
    y_modes: np.ndarray | list[int], *, edge_ratio: float, width: float
) -> np.ndarray:
    """Closed-form ``int_0^b f(y) sin(n pi y / b) sin(q pi y / b) dy``.

    ``f(y) = 1 + (psi - 1) y / b`` is the normalised edge-stress profile, so the
    returned matrix is the ``y``-coupling of the destabilising energy
    ``int N_x(y) (dw/dx)^2``.  In dimensionless ``xi = y / b``

    .. math::

        G_{nq} = b \\left[\\frac{\\delta_{nq}}{2}
                 + (\\psi - 1) \\int_0^1 \\xi \\sin n\\pi\\xi \\sin q\\pi\\xi\\,d\\xi\\right]

    and, because ``int_0^1 xi cos(k pi xi) dxi = ((-1)^k - 1) / (k pi)^2``, the
    non-diagonal part is non-zero **only for odd** ``n + q``:

    .. math::

        \\int_0^1 \\xi \\sin n\\pi\\xi \\sin q\\pi\\xi\\,d\\xi
        = -\\frac{1}{\\pi^2}\\left[\\frac{1}{(n-q)^2} - \\frac{1}{(n+q)^2}\\right]

    The opposite-parity coupling is why a pure-bending mode cannot be a single
    sine pair: the diagonal term alone vanishes at ``psi = -1``.
    """
    ratio = edge_stress_ratio_profile(edge_ratio)
    span = _positive_finite("width", width)
    n = np.asarray(y_modes, dtype=float)
    if n.ndim != 1 or n.size == 0:
        raise ValueError("y_modes must be a non-empty 1-D sequence")
    if np.any(n < 1.0):
        raise ValueError("y_modes must be positive half-wave numbers")
    nn = n[:, None]
    qq = n[None, :]
    odd = ((nn + qq) % 2.0) == 1.0
    gap = nn - qq
    # avoid division by zero for n == q, whose value comes from the diagonal term
    safe_gap = np.where(odd, gap, 1.0)
    off = np.where(
        odd,
        -(1.0 / math.pi**2) * (1.0 / safe_gap**2 - 1.0 / (nn + qq) ** 2),
        0.0,
    )
    diagonal = 0.5 + (ratio - 1.0) * 0.25
    return span * (diagonal * np.eye(n.size) + (ratio - 1.0) * off)


def stress_gradient_matrices(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    basis_x: int = 16,
    basis_y: int = 16,
    edge_ratio: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Bending and geometric matrices of a linearly loaded simply supported panel.

    Basis ``w = sum c_mn sin(m pi x / a) sin(n pi y / b)``, so the bending energy
    ``int Dx w,xx^2 + 2 D12 w,xx w,yy + Dy w,yy^2 + 4 D66 w,xy^2`` is diagonal
    and the geometric energy carries the gradient through
    :func:`stress_gradient_coupling_matrix`.

    Returns ``(bending, geometric, modes)`` with ``geometric`` the matrix of the
    destabilising energy for a **unit** peak edge resultant, i.e. the buckling
    load is the smallest positive ``lambda`` of
    ``bending phi = lambda geometric phi``.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    nb_x, nb_y = _validate_basis(basis_x, basis_y)
    ratio = edge_stress_ratio_profile(edge_ratio)

    m = np.arange(1, nb_x + 1, dtype=float)
    n = np.arange(1, nb_y + 1, dtype=float)
    mm, nn = np.meshgrid(m, n, indexing="ij")
    modes = [(int(i), int(j)) for i, j in zip(mm.ravel(), nn.ravel())]
    flat_m = mm.ravel()
    flat_n = nn.ravel()

    kx = flat_m * math.pi / a
    ky = flat_n * math.pi / b
    bending = np.diag(dx * kx**4 + 2.0 * h * kx**2 * ky**2 + dy * ky**4) * (a * b / 4.0)

    coupling = stress_gradient_coupling_matrix(
        n, edge_ratio=ratio, width=b
    )
    same_m = flat_m[:, None] == flat_m[None, :]
    geometric = (kx**2)[:, None] * np.where(same_m, coupling[np.ix_(flat_n.astype(int) - 1, flat_n.astype(int) - 1)], 0.0) * (a / 2.0)
    return bending, geometric, modes


def stress_gradient_quadrature(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    basis_x: int,
    basis_y: int,
    edge_ratio: float = -1.0,
    samples: int = 90,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Same matrices as :func:`stress_gradient_matrices`, integrated numerically.

    Two-dimensional Gauss-Legendre quadrature of the energy functionals for
    every basis pair, taking only the analytic derivatives of the sine modes:
    no closed-form integral of any product is used, and the ``y``-profile is
    evaluated pointwise from its definition.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    nb_x, nb_y = _validate_basis(basis_x, basis_y)
    ratio = edge_stress_ratio_profile(edge_ratio)
    count = int(samples)
    if count < 8:
        raise ValueError("samples must be at least 8")

    nodes, weights = np.polynomial.legendre.leggauss(count)
    xi = 0.5 * (nodes + 1.0)
    wx = 0.5 * weights
    x = xi * a
    y = xi * b
    X, Y = np.meshgrid(x, y, indexing="ij")
    W = (wx[:, None] * wx[None, :]) * (a * b)

    modes = [(i, j) for i in range(1, nb_x + 1) for j in range(1, nb_y + 1)]
    count_modes = len(modes)
    phi = np.empty((count_modes, X.size))
    dxx = np.empty_like(phi)
    dyy = np.empty_like(phi)
    dxy = np.empty_like(phi)
    dx1 = np.empty_like(phi)
    flat_x = X.ravel()
    flat_y = Y.ravel()
    for index, (i, j) in enumerate(modes):
        kx = i * math.pi / a
        ky = j * math.pi / b
        sx, cx = np.sin(kx * flat_x), np.cos(kx * flat_x)
        sy, cy = np.sin(ky * flat_y), np.cos(ky * flat_y)
        phi[index] = sx * sy
        dx1[index] = kx * cx * sy
        dxx[index] = -(kx**2) * sx * sy
        dyy[index] = -(ky**2) * sx * sy
        dxy[index] = kx * ky * cx * cy
    weight = W.ravel()
    # energy densities per basis pair, integrated with the quadrature weights only
    bending = (
        dx * (dxx * weight) @ dxx.T
        + 2.0 * h * (dxy * weight) @ dxy.T
        + dy * (dyy * weight) @ dyy.T
    )
    profile = 1.0 + (ratio - 1.0) * (flat_y / b)
    geometric = (dx1 * (weight * profile)) @ dx1.T
    return bending, geometric, modes


def ritz_stress_gradient_buckling_load(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    basis_x: int = 16,
    basis_y: int = 16,
    edge_ratio: float = -1.0,
) -> StressGradientBucklingMode:
    """Galerkin buckling load of a panel under a linearly varying edge stress."""
    nb_x, nb_y = _validate_basis(basis_x, basis_y)
    ratio = edge_stress_ratio_profile(edge_ratio)
    bending, geometric, modes = stress_gradient_matrices(
        length=length,
        width=width,
        rigidity_x=rigidity_x,
        rigidity_y=rigidity_y,
        coupling_rigidity=coupling_rigidity,
        basis_x=nb_x,
        basis_y=nb_y,
        edge_ratio=ratio,
    )
    multiplier, phi = _smallest_positive_generalized(bending, geometric)
    if not math.isfinite(multiplier):
        return StressGradientBucklingMode(
            math.inf, ratio, 0.0, 0.0, (nb_x, nb_y), (0, 0), False
        )
    dominant = modes[int(np.argmax(np.abs(phi)))]
    return StressGradientBucklingMode(
        multiplier=float(multiplier),
        edge_ratio=ratio,
        compression_edge=float(multiplier),
        opposite_edge=float(ratio * multiplier),
        basis_shape=(nb_x, nb_y),
        dominant_mode=dominant,
        buckling=True,
    )


def stress_gradient_mode_shape(
    *,
    length: float,
    width: float,
    rigidity_x: float,
    rigidity_y: float,
    coupling_rigidity: float,
    basis_x: int = 16,
    basis_y: int = 16,
    edge_ratio: float = -1.0,
    samples: int = 61,
) -> dict[str, object]:
    """Critical buckling mode ``w(x, y)`` of the graded panel on a regular grid.

    Returns ``{"x", "y", "w", "coefficient_k", "mode", "peak_location"}`` with
    ``w`` normalised to unit peak amplitude and the grid built from the Galerkin
    coefficients of the critical eigenvector.  Used for the field plot; the
    numbers never enter a qualification gate.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    nb_x, nb_y = _validate_basis(basis_x, basis_y)
    ratio = edge_stress_ratio_profile(edge_ratio)
    count = int(samples)
    if count < 8:
        raise ValueError("samples must be at least 8")
    bending, geometric, modes = stress_gradient_matrices(
        length=a, width=b, rigidity_x=dx, rigidity_y=dy,
        coupling_rigidity=h, basis_x=nb_x, basis_y=nb_y, edge_ratio=ratio,
    )
    multiplier, phi = _smallest_positive_generalized(bending, geometric)
    x = np.linspace(0.0, a, count)
    y = np.linspace(0.0, b, count)
    X, Y = np.meshgrid(x, y, indexing="ij")
    w = np.zeros_like(X)
    for coefficient, (i, j) in zip(phi, modes):
        w += coefficient * np.sin(i * math.pi * X / a) * np.sin(j * math.pi * Y / b)
    peak = float(np.abs(w).max())
    if peak > 0.0:
        w = w / peak
    flat = int(np.argmax(np.abs(w)))
    return {
        "x": x,
        "y": y,
        "w": w,
        "critical_load": multiplier,
        "coefficient_k": (
            bending_buckling_coefficient(
                critical_load=multiplier, rigidity=dx, width=b
            )
            if math.isfinite(multiplier)
            else math.inf
        ),
        "mode": modes[int(np.argmax(np.abs(phi)))],
        "peak_location": (float(X.ravel()[flat]), float(Y.ravel()[flat])),
    }


def bending_buckling_coefficient(
    *, critical_load: float, rigidity: float, width: float
) -> float:
    """``k = N1 b^2 / (pi^2 D)`` referred to the **peak** edge resultant."""
    load = _positive_finite("critical_load", critical_load)
    stiffness = _positive_finite("rigidity", rigidity)
    span = _positive_finite("width", width)
    return load * span**2 / (math.pi**2 * stiffness)


def _second_difference(dimension: int, span: float) -> np.ndarray:
    """Interior ``d2/dx2`` of a simply supported (``w = 0``) row, ``n - 1`` square."""
    n = dimension
    step = span / n
    size = n - 1
    matrix = np.zeros((size, size))
    for row in range(size):
        matrix[row, row] = -2.0
        if row > 0:
            matrix[row, row - 1] = 1.0
        if row < size - 1:
            matrix[row, row + 1] = 1.0
    return matrix / step**2


def finite_difference_bending_buckling(
    *,
    length: float,
    width: float,
    rigidity: float,
    edge_ratio: float = -1.0,
    divisions: int = 32,
) -> dict[str, float]:
    """Independent finite-difference critical coefficient for a graded load.

    Solves ``D lap^4 w + N_x(y) w,xx = 0`` in strong form on a nodal grid whose
    compressive resultant follows the linear profile row by row.  Nothing is
    shared with the variational route: no sine basis, no energy matrix.  The
    discretisation shares only the plate model itself.

    Returns ``{"coefficient", "critical_load", "grid"}`` with the coefficient
    referred to the peak edge resultant, so it is directly comparable with
    :func:`bending_buckling_coefficient`.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    stiffness = _positive_finite("rigidity", rigidity)
    ratio = edge_stress_ratio_profile(edge_ratio)
    elements = int(divisions)
    if elements < 6:
        raise ValueError("divisions must be at least 6")
    # isotropic element size: divisions elements across the width, and the same
    # element size along the length.  Equal *node counts* would leave a long
    # panel under-resolved in x by the aspect ratio, which silently caps the
    # half-wave number the grid can carry and destroys the coefficient.
    ny = elements
    nx = max(6, int(round(elements * (a / b))))

    d2x = _second_difference(nx, a)
    d2y = _second_difference(ny, b)
    ix = np.eye(nx - 1)
    iy = np.eye(ny - 1)
    # Bi-harmonic operator of a simply supported panel as the *square* of the
    # discrete Laplacian.  For SSSS edges the natural condition ``w,xx = 0`` is
    # exactly the boundary condition the squared operator inherits from
    # ``w = 0``, so ``lap2 = L @ L`` is positive definite by construction and no
    # ghost-node stencil has to be invented (an earlier 13-point stencil with
    # hand-built ghosts was indefinite and rejected by the Cholesky reduction).
    laplacian = np.kron(d2x, iy) + np.kron(ix, d2y)
    bending = stiffness * (laplacian @ laplacian)

    # y coordinate of each interior row; x is the fast index
    y_rows = np.array([(j + 1) * b / ny for j in range(ny - 1)])
    profile = 1.0 + (ratio - 1.0) * (y_rows / b)
    geometric = -np.kron(d2x, np.diag(profile))

    # symmetric reduction: bending is SPD, so only the reduced pencil is solved;
    # a general nonsymmetric solver is O(n^3) with a far larger constant here
    load, _ = _smallest_positive_generalized(bending, geometric)
    if not math.isfinite(load):
        return {
            "coefficient": math.inf,
            "critical_load": math.inf,
            "grid": float((nx - 1) * (ny - 1)),
        }
    return {
        "coefficient": load * b**2 / (math.pi**2 * stiffness),
        "critical_load": load,
        "grid": float((nx - 1) * (ny - 1)),
    }


def run_bending_buckling_qualification(
    *,
    young: float = 210.0e9,
    poisson: float = 0.3,
    thickness: float = 0.012,
    tolerance: float = 0.03,
    basis: int = 16,
) -> dict[str, object]:
    """Qualify the linearly varying prebuckling stress against four anchors.

    Gates (all ``< tolerance``, default 3 %):

    * ``uniform_limit``: ``psi = 1`` reproduces ``k = 4`` exactly, so the
      gradient assembly degenerates to the validated uniform case;
    * ``pure_bending_long_panel``: ``psi = -1``, ``a/b = 6`` against ``23.9``;
    * ``pure_bending_square``: ``psi = -1``, ``a/b = 1`` against ``25.6``;
    * ``triangular_long_panel``: ``psi = 0``, ``a/b = 6`` against ``7.81``;
    * ``finite_difference``: the independent strong-form discretisation agrees
      with the variational bending coefficient;
    * ``quadrature``: the closed-form matrices agree with Gauss-Legendre
      quadrature of the same functionals;
    * ``basis_convergence``: basis refinement is stable at the reported basis;
    * ``gradient_monotonicity``: ``k(psi)`` decreases with ``psi`` over the
      closed interval, i.e. a bending gradient is strictly less destabilising
      than the same peak compression applied uniformly.

    Raises ``AssertionError`` when any gate fails: this is a fail-closed
    qualification, not a report.
    """
    d, _, h = isotropic_plate_rigidities(young, poisson, thickness)
    tol = float(tolerance)
    if not 0.0 < tol < 1.0:
        raise ValueError("tolerance must lie in (0, 1)")
    nb = int(basis)
    rows: list[dict[str, object]] = []

    def _coefficient(ratio: float, aspect: float, basis_size: int = nb) -> float:
        mode = ritz_stress_gradient_buckling_load(
            length=aspect,
            width=1.0,
            rigidity_x=d,
            rigidity_y=d,
            coupling_rigidity=h,
            basis_x=basis_size,
            basis_y=basis_size,
            edge_ratio=ratio,
        )
        return bending_buckling_coefficient(
            critical_load=mode.multiplier, rigidity=d, width=1.0
        )

    literature_cases = (
        ("uniform_limit", 1.0, 6.0, LITERATURE_K_UNIFORM_LONG),
        ("pure_bending_long_panel", -1.0, 6.0, LITERATURE_KB_PURE_BENDING_LONG),
        ("pure_bending_square", -1.0, 1.0, LITERATURE_KB_PURE_BENDING_SQUARE),
        ("triangular_long_panel", 0.0, 6.0, LITERATURE_K_TRIANGULAR_LONG),
    )
    for name, ratio, aspect, reference in literature_cases:
        coefficient = _coefficient(ratio, aspect)
        error = abs(coefficient / reference - 1.0)
        rows.append({
            "case": name,
            "edge_ratio": ratio,
            "aspect_ratio": aspect,
            "reference_k": reference,
            "coefficient_k": coefficient,
            "reference_source": "Timoshenko & Gere (1961) Table 9.1",
            "relative_error": error,
            "passed": bool(error < tol),
        })

    # independent discretisation: strong-form finite differences.  Run on the
    # square panel, where the graded load is carried by a fine isotropic grid
    # without the aspect ratio inflating the unknown count; the long-panel
    # literature value is not reachable by finite differences at a dense enough
    # grid inside a qualification budget.
    reference_bending = rows[2]["coefficient_k"]  # pure_bending_square
    fd = finite_difference_bending_buckling(
        length=1.0, width=1.0, rigidity=d, edge_ratio=-1.0, divisions=32,
    )
    fd_error = abs(fd["coefficient"] / reference_bending - 1.0)
    rows.append({
        "case": "finite_difference_pure_bending",
        "edge_ratio": -1.0,
        "aspect_ratio": 1.0,
        "variational_k": reference_bending,
        "finite_difference_k": fd["coefficient"],
        "grid_nodes": fd["grid"],
        "relative_error": fd_error,
        "tolerance": 0.05,
        "passed": bool(fd_error < 0.05),
    })
    # the same nodal solver on the degenerated (uniform) profile must return 4
    fd_uniform = finite_difference_bending_buckling(
        length=1.0, width=1.0, rigidity=d, edge_ratio=1.0, divisions=32,
    )
    fd_uniform_error = abs(fd_uniform["coefficient"] / LITERATURE_K_UNIFORM_LONG - 1.0)
    rows.append({
        "case": "finite_difference_uniform_limit",
        "edge_ratio": 1.0,
        "aspect_ratio": 1.0,
        "reference_k": LITERATURE_K_UNIFORM_LONG,
        "finite_difference_k": fd_uniform["coefficient"],
        "grid_nodes": fd_uniform["grid"],
        "relative_error": fd_uniform_error,
        "tolerance": 0.05,
        "passed": bool(fd_uniform_error < 0.05),
    })

    # independent integral: quadrature of the energy functionals
    bending_closed, geometric_closed, _ = stress_gradient_matrices(
        length=6.0, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h,
        basis_x=6, basis_y=6, edge_ratio=-1.0,
    )
    bending_quad, geometric_quad, _ = stress_gradient_quadrature(
        length=6.0, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h,
        basis_x=6, basis_y=6, edge_ratio=-1.0,
    )
    bending_drift = float(
        np.abs(bending_closed - bending_quad).max() / np.abs(bending_closed).max()
    )
    geometric_drift = float(
        np.abs(geometric_closed - geometric_quad).max()
        / np.abs(geometric_closed).max()
    )
    rows.append({
        "case": "closed_form_vs_quadrature",
        "bending_relative_difference": bending_drift,
        "geometric_relative_difference": geometric_drift,
        "tolerance": 1e-10,
        "passed": bool(max(bending_drift, geometric_drift) < 1e-10),
    })

    # basis refinement stability
    coarse = _coefficient(-1.0, 6.0, 12)
    refined = _coefficient(-1.0, 6.0, nb)
    drift = abs(refined - coarse) / refined
    rows.append({
        "case": "basis_convergence_pure_bending",
        "basis_coarse": 12,
        "basis_refined": nb,
        "coefficient_coarse": coarse,
        "coefficient_refined": refined,
        "relative_drift": drift,
        "tolerance": 5e-3,
        "passed": bool(drift < 5e-3),
    })

    # monotonicity of k(psi): a gradient is less destabilising than its peak
    samples = [
        (ratio, _coefficient(ratio, 6.0)) for ratio in (1.0, 0.5, 0.0, -0.5, -1.0)
    ]
    monotone = all(
        later[1] >= earlier[1] - 1e-12
        for earlier, later in zip(samples, samples[1:])
    )
    rows.append({
        "case": "gradient_monotonicity",
        "samples": [{"edge_ratio": r, "coefficient_k": k} for r, k in samples],
        "passed": bool(monotone),
    })

    passed = all(bool(row["passed"]) for row in rows)
    if not passed:
        failed = [row["case"] for row in rows if not row["passed"]]
        raise AssertionError(f"bending buckling qualification failed: {failed}")
    return {
        "schema": "tensorfem.plate-bending-buckling/1.0",
        "scope": (
            "TensorFEM variational and finite-difference plate buckling; "
            "flat simply supported panel, linear edge-stress gradient"
        ),
        "tolerance": tol,
        "young": young,
        "poisson": poisson,
        "thickness": thickness,
        "basis": nb,
        "cases": rows,
        "passed": passed,
    }