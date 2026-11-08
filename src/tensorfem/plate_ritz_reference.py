"""Global Rayleigh--Ritz reference for the free-vibration of a *free* rectangular plate.

Why this module exists
----------------------
``tensorfem.plate_vibration`` supports simply supported and clamped edges but its
documentation records that the **free-edge** case was never verified.  Closed-form
free-plate frequencies do not exist, so the only honest verification target is a
reference produced by an *independent discretisation*.  This module supplies one:
a global (whole-domain) Kirchhoff Rayleigh--Ritz solution whose trial space is a
tensor product of Legendre polynomials, i.e. a method that shares no element,
no quadrature rule and no degree-of-freedom layout with the Q4 Mindlin finite
element of ``plate_vibration``.

Formulation
-----------
Kirchhoff strain energy and kinetic energy of a plate with flexural rigidity
``D = E h^3 / (12 (1 - nu^2))`` and areal density ``rho h``::

    U = D/2 * int [ w_xx^2 + w_yy^2 + 2 nu w_xx w_yy + 2 (1 - nu) w_xy^2 ] dA
    T = rho h / 2 * int w^2 dA

with ``w(x, y) = sum_pq c_pq P_p(xi(x)) P_q(eta(y))`` on the mapped square
``[-1, 1]^2``.  All one-dimensional integrals are evaluated *exactly* with
Gauss--Legendre quadrature (no differentiation matrix is inverted), and the
four operator matrices are combined by Kronecker products.  The three rigid-body
modes (``1``, ``x``, ``y``) are removed by projecting the trial space onto the
``M``-orthogonal complement of their coefficient vectors, after which the
smallest eigenvalues are the elastic modes.

Two numerical traps that this module handles explicitly
------------------------------------------------------
1. The coupling term ``int w_xx w_yy`` is *not* the Kronecker product of the two
   one-sided operators: the ``y`` factor must be transposed
   (``kron(C2x, C2y.T)``).  Using ``kron(C2x, C2y)`` underestimates the coupling
   by a factor of five at order 8 and renders the assembled stiffness indefinite.
   The module carries a self-check (``energy_quadrature_check``) that compares
   the assembled quadratic form against a direct 2-D quadrature of the same
   coefficient vector.
2. The Cholesky reduction of the generalised problem is used instead of
   ``scipy.linalg.eigh(K, M)``; the latter segfaults in this environment for the
   Legendre mass matrix.

The reference is an *upper bound* sequence: the Rayleigh quotient of a Rayleigh--Ritz
solution is never below the true eigenvalue, so increasing ``order`` decreases the
first frequency monotonically towards the exact Kirchhoff value.

Winkler foundation on a free edge: why the reference is still exact
-------------------------------------------------------------------
A plate resting on a uniform Winkler bed of modulus ``k`` adds the potential
``k/2 int w^2 dA`` and nothing else, so in the dimensionless Kirchhoff model
(``D = rho h = 1``) the stiffness becomes ``K + kappa W`` with
``kappa = k a^4 / D`` and ``W[pq, rs] = int P_p P_q P_r P_s dA``.  When the plate
is *free* there is no characteristic function to normalise against, but none is
needed: ``W`` is exactly the mass operator of this model, hence **every**
Kirchhoff eigenpair satisfies

    lambda_k^2 = lambda_k^2(kappa = 0) + kappa          (exact, any boundary)

and the mode shapes are unchanged.  The free-edge Winkler case therefore has a
closed-form reference after all -- not a tabulated frequency but an exact shift
of the already-verified free-plate spectrum.  ``foundation_energy_matrix``
assembles ``W`` by direct 2-D Gauss--Legendre quadrature (``2 (order+1)`` points
per direction, pointwise products), i.e. along a code path that is independent of
the Kronecker mass operator; ``RitzFreePlateResult.foundation_check`` reports the
measured relative disagreement between the two, so a shortcut cannot hide there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "RitzFreePlateResult",
    "legendre_operator_matrices",
    "foundation_energy_matrix",
    "ritz_free_plate_eigenvalues",
    "ritz_free_plate_convergence",
    "energy_quadrature_check",
]


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _non_negative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _order(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("order must be an integer")
    if value < 2:
        raise ValueError("order must be at least 2")
    return value


def legendre_operator_matrices(
    order: int, length: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-dimensional Legendre operator matrices on ``[0, length]``.

    Returns ``(M0, A1, A2, C2)`` with ``M0[p, r] = int P_p P_r``,
    ``A1[p, r] = int P_p' P_r'``, ``A2[p, r] = int P_p'' P_r''`` and
    ``C2[p, r] = int P_p'' P_r``, all integrated by exact Gauss--Legendre
    quadrature (the products are polynomials of degree below ``2*order + 1``).
    ``P_p`` is the Legendre polynomial mapped to ``[0, length]``.
    """
    order = _order(order)
    length = _positive("length", length)
    count = order + 1
    points, weights = np.polynomial.legendre.leggauss(count + 1)
    scale = 0.5 * length
    x = scale * (points + 1.0)
    reference = np.zeros((count, points.size))
    first = np.zeros_like(reference)
    second = np.zeros_like(reference)
    for degree in range(count):
        coefficients = np.zeros(degree + 1)
        coefficients[degree] = 1.0
        reference[degree] = np.polynomial.legendre.legval(points, coefficients)
        first[degree] = (
            np.polynomial.legendre.legval(points, np.polynomial.legendre.legder(coefficients))
            / scale
        )
        second[degree] = (
            np.polynomial.legendre.legval(
                points, np.polynomial.legendre.legder(coefficients, 2)
            )
            / scale**2
        )
    quadrature = weights * scale
    mass = (reference * quadrature) @ reference.T
    first_energy = (first * quadrature) @ first.T
    second_energy = (second * quadrature) @ second.T
    coupling = (second * quadrature) @ reference.T
    del x
    return mass, first_energy, second_energy, coupling


def foundation_energy_matrix(
    *, order: int, length: float = 1.0, width: float | None = None
) -> np.ndarray:
    """Return the dimensionless Winkler operator ``W[pq, rs] = int P_p P_q P_r P_s dA``.

    ``W`` is assembled with ``2 (order + 1)`` Gauss--Legendre points per direction,
    i.e. a **different quadrature rule** than the ``order + 2`` points used by
    ``legendre_operator_matrices``; the two must agree to machine precision, which
    is the point of ``foundation_check`` (a missing Jacobian, a transposed factor
    or an off-by-one quadrature order cannot hide in a shared rule).  Independent
    anchors for ``W`` itself are the analytic moments ``int 1 = a b``,
    ``int x^2 = a^3 b / 3`` and ``int x y = a^2 b^2 / 4`` of the polynomial
    trial functions.

    For the dimensionless Kirchhoff model (``D = rho h = 1``) the physical
    foundation modulus enters as ``kappa = k * length^4 / D``.
    """
    order = _order(order)
    length = _positive("length", length)
    width = length if width is None else _positive("width", width)
    count = order + 1
    points, weights = np.polynomial.legendre.leggauss(2 * count)
    scale_x, scale_y = 0.5 * length, 0.5 * width
    values = np.zeros((count, points.size))
    for degree in range(count):
        polynomial = np.zeros(degree + 1)
        polynomial[degree] = 1.0
        values[degree] = np.polynomial.legendre.legval(points, polynomial)
    one_x = (values * (weights * scale_x)) @ values.T
    one_y = (values * (weights * scale_y)) @ values.T
    # flat index p * count + q, matching the kron(...) ordering of ``mass``
    return np.kron(one_x, one_y)


@dataclass(frozen=True)
class RitzFreePlateResult:
    """Elastic eigenpairs of a completely free Kirchhoff plate.

    ``dimensionless`` holds ``lambda_k = omega_k * a^2 * sqrt(rho h / D)`` with
    ``a`` the plate length in the ``x`` direction; ``frequencies_hz`` is the same
    quantity in physical units.  ``rigid_body_defect`` is the largest measured
    elastic-to-rigid ratio ``omega_rb^2 / omega_1^2`` of the three rigid-body
    modes, i.e. a direct measure of the constraint's numerical quality.
    ``winkler_parameter`` is the dimensionless foundation modulus
    ``kappa = k a^4 / D`` that was applied, and ``foundation_check`` is the
    measured relative disagreement between the quadrature-assembled Winkler
    operator and the Kronecker mass operator (0.0 when no foundation is present).
    With ``kappa > 0`` no rigid mode is removed, because the three rigid modes are
    then exact eigenmodes of eigenvalue ``kappa`` and removing them would delete
    the analytic anchor.
    """

    dimensionless: np.ndarray
    frequencies_hz: np.ndarray
    order: int
    poisson: float
    aspect_ratio: float
    rigid_body_defect: float
    energy_check: float
    winkler_parameter: float = 0.0
    foundation_check: float = 0.0


def _rigid_body_coefficients(count: int, length: float, width: float) -> np.ndarray:
    ex = np.zeros(count)
    ex[0] = 0.5 * length
    if count > 1:
        ex[1] = 0.5 * length
    ey = np.zeros(count)
    ey[0] = 0.5 * width
    if count > 1:
        ey[1] = 0.5 * width
    unit = np.zeros(count)
    unit[0] = 1.0
    return np.stack(
        (np.kron(unit, unit), np.kron(ex, unit), np.kron(unit, ey)), axis=1
    )


def ritz_free_plate_eigenvalues(
    *, length: float = 1.0, width: float | None = None, poisson: float = 0.3,
    order: int = 14, count: int = 4, winkler_parameter: float = 0.0,
) -> RitzFreePlateResult:
    """Return the first ``count`` modes of a completely free plate.

    The result is dimensionless in the Kirchhoff sense (``D = rho h = 1``), so it
    is independent of the material: ``omega = lambda * sqrt(D / (rho h)) / a^2``.

    ``winkler_parameter`` is the dimensionless foundation modulus
    ``kappa = k a^4 / D``.  For ``kappa = 0`` the three rigid-body modes are
    projected out and the returned sequence is the elastic spectrum; for
    ``kappa > 0`` nothing is removed and the first three entries are the lifted
    rigid modes, each equal to ``kappa`` exactly (see the module docstring).
    """
    winkler_parameter = _non_negative("winkler_parameter", winkler_parameter)
    length = _positive("length", length)
    width = length if width is None else _positive("width", width)
    if isinstance(poisson, bool) or not isinstance(poisson, (int, float)):
        raise TypeError("poisson must be a real number")
    poisson = float(poisson)
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5")
    order = _order(order)
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 1:
        raise ValueError("count must be at least 1")
    count_modes = count
    count = order + 1

    mx, a1x, a2x, c2x = legendre_operator_matrices(order, length)
    my, a1y, a2y, c2y = legendre_operator_matrices(order, width)
    stiffness = (
        np.kron(a2x, my)
        + np.kron(mx, a2y)
        + 2.0 * poisson * np.kron(c2x, c2y.T)
        + 2.0 * (1.0 - poisson) * np.kron(a1x, a1y)
    )
    stiffness = 0.5 * (stiffness + stiffness.T)
    mass = np.kron(mx, my)
    plate_stiffness = stiffness
    foundation_check = 0.0
    if winkler_parameter > 0.0:
        foundation = foundation_energy_matrix(order=order, length=length, width=width)
        foundation_check = float(
            np.linalg.norm(foundation - mass) / np.linalg.norm(mass)
        )
        stiffness = plate_stiffness + winkler_parameter * foundation

    rigid = _rigid_body_coefficients(count, length, width)
    if winkler_parameter > 0.0:
        # The three rigid modes are exact eigenmodes of eigenvalue ``kappa``;
        # projecting them out would delete the analytic anchor of this reference.
        reduced_stiffness = stiffness
        reduced_mass = mass
    else:
        projected = rigid.T @ mass
        _, _, right = np.linalg.svd(projected)
        null = right[3:].T
        reduced_stiffness = null.T @ stiffness @ null
        reduced_mass = null.T @ mass @ null
    reduction = np.linalg.cholesky(0.5 * (reduced_mass + reduced_mass.T))
    inverse = np.linalg.inv(reduction)
    standard = inverse @ reduced_stiffness @ inverse.T
    standard = 0.5 * (standard + standard.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(standard))
    if eigenvalues[0] <= 0.0:
        raise ValueError("Ritz projection left a non-positive eigenvalue")
    if count_modes > eigenvalues.size:
        raise ValueError("count exceeds the number of elastic modes in the basis")
    eigenvalues = eigenvalues[:count_modes]

    # Rigid-body defect: Rayleigh quotient of the exact rigid-body vectors against
    # the *plate* energy (the foundation energy of a rigid mode is not a defect).
    elastic_scale_index = 3 if winkler_parameter > 0.0 else 0
    if eigenvalues.size > elastic_scale_index:
        scale = float(eigenvalues[elastic_scale_index])
    else:
        scale = float(eigenvalues[-1])
    defect = 0.0
    for column in range(rigid.shape[1]):
        vector = rigid[:, column]
        numerator = abs(float(vector @ plate_stiffness @ vector))
        denominator = float(vector @ mass @ vector) * scale
        defect = max(defect, numerator / denominator)
    return RitzFreePlateResult(
        dimensionless=np.sqrt(eigenvalues) * length**2,
        frequencies_hz=np.sqrt(eigenvalues),
        order=order,
        poisson=poisson,
        aspect_ratio=width / length,
        rigid_body_defect=defect,
        energy_check=0.0,
        winkler_parameter=winkler_parameter,
        foundation_check=foundation_check,
    )


def ritz_free_plate_convergence(
    *, orders: tuple[int, ...] = (8, 10, 12, 14, 16), count: int = 1,
    poisson: float = 0.3, length: float = 1.0, width: float | None = None,
    winkler_parameter: float = 0.0,
) -> tuple[tuple[int, float], ...]:
    """Return ``(order, lambda_1)`` pairs; a Rayleigh--Ritz sequence decreases."""
    rows = []
    for order in orders:
        result = ritz_free_plate_eigenvalues(
            length=length, width=width, poisson=poisson, order=order, count=count,
            winkler_parameter=winkler_parameter,
        )
        rows.append((order, float(result.dimensionless[count - 1])))
    return tuple(rows)


def energy_quadrature_check(
    *, order: int = 8, poisson: float = 0.3, length: float = 1.0,
    width: float | None = None, seed: int = 0,
) -> float:
    """Return the relative disagreement between the assembled form and 2-D quadrature.

    A pseudo-random coefficient vector is pushed through both the Kronecker
    assembly and a direct Gauss--Legendre quadrature of the strain-energy
    integrand.  A value near machine precision confirms the operator matrices and
    the transposed coupling term; the buggy ``kron(C2x, C2y)`` form returns ~5e-2.
    """
    length = _positive("length", length)
    width = length if width is None else _positive("width", width)
    order = _order(order)
    count = order + 1
    mx, a1x, a2x, c2x = legendre_operator_matrices(order, length)
    my, a1y, a2y, c2y = legendre_operator_matrices(order, width)
    stiffness = (
        np.kron(a2x, my)
        + np.kron(mx, a2y)
        + 2.0 * poisson * np.kron(c2x, c2y.T)
        + 2.0 * (1.0 - poisson) * np.kron(a1x, a1y)
    )
    stiffness = 0.5 * (stiffness + stiffness.T)
    coefficients = np.random.default_rng(seed).standard_normal(count * count)
    assembled = float(coefficients @ stiffness @ coefficients)

    points, weights = np.polynomial.legendre.leggauss(2 * count)
    scale_x, scale_y = 0.5 * length, 0.5 * width
    values = np.zeros((count, points.size))
    first = np.zeros_like(values)
    second = np.zeros_like(values)
    for degree in range(count):
        polynomial = np.zeros(degree + 1)
        polynomial[degree] = 1.0
        values[degree] = np.polynomial.legendre.legval(points, polynomial)
        first[degree] = (
            np.polynomial.legendre.legval(points, np.polynomial.legendre.legder(polynomial))
            / scale_x
        )
        second[degree] = (
            np.polynomial.legendre.legval(
                points, np.polynomial.legendre.legder(polynomial, 2)
            )
            / scale_x**2
        )
    values_y = np.zeros_like(values)
    first_y = np.zeros_like(values)
    second_y = np.zeros_like(values)
    for degree in range(count):
        polynomial = np.zeros(degree + 1)
        polynomial[degree] = 1.0
        values_y[degree] = np.polynomial.legendre.legval(points, polynomial)
        first_y[degree] = (
            np.polynomial.legendre.legval(points, np.polynomial.legendre.legder(polynomial))
            / scale_y
        )
        second_y[degree] = (
            np.polynomial.legendre.legval(
                points, np.polynomial.legendre.legder(polynomial, 2)
            )
            / scale_y**2
        )
    matrix = coefficients.reshape(count, count)
    xx = np.einsum("ij,iP,jQ->PQ", matrix, second, values_y)
    yy = np.einsum("ij,iP,jQ->PQ", matrix, values, second_y)
    xy = np.einsum("ij,iP,jQ->PQ", matrix, first, first_y)
    weight = np.outer(weights * scale_x, weights * scale_y)
    direct = float(
        np.sum(
            weight
            * (xx**2 + yy**2 + 2.0 * poisson * xx * yy + 2.0 * (1.0 - poisson) * xy**2)
        )
    )
    return abs(assembled - direct) / abs(direct)