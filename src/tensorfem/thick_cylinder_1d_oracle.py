"""Chebyshev-collocation oracle for the axisymmetric thick-cylinder (Lame) problem.

Why this exists
---------------
The Lamé test case compares a two-dimensional finite element solution against a
closed form.  A closed form written next to the FE code in the same repository
is not an *independent* reference by construction: a typo in the elastic
constants or in the sign of the pressure would propagate into both sides.  This
module supplies a second, numerically independent route to the same fields:

* it discretises the **displacement formulation** of the axisymmetric problem,

      (r*sigma_r)' - sigma_theta = 0 ,
      sigma_r = C * ((1-nu) u' + nu u/r) ,  sigma_theta = C * (nu u' + (1-nu) u/r)

  (plane strain, ``C = E/((1+nu)(1-2nu))``; plane stress uses the plane-stress
  ``C`` and ``(1, nu) / (nu, 1)`` coefficients), which collapses to the Euler
  equation ``r u'' + u' - u/r = 0``;

* the boundary conditions are imposed as the physical tractions
  ``sigma_r(a) = -p_i``, ``sigma_r(b) = -p_o`` -- no displacement is prescribed;

* it is solved by Chebyshev collocation (spectral, machine-precision for the
  smooth Lame field) rather than by the algebra of the closed form.

Agreement between this oracle and :func:`tensorfem.lame_reference.lame_state`
therefore tests the closed form against the governing equations, and only then
is the closed form used as the reference for the finite element study.

References
----------
Lamé, G. (1852), *Leçons sur la théorie de l'élasticité*; the same problem is
worked in Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed., Art. 27
("The thick cylinder"), and in Roark's *Formulas for Stress and Strain*,
Table 13.5 (case 1, cylinder with internal pressure).  Only the *equations*
are used here -- every number is produced by the code below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["OracleState", "solve_thick_cylinder_1d"]


@dataclass(frozen=True)
class OracleState:
    """Spectral oracle fields sampled at the requested radii."""

    radius: np.ndarray
    radial_stress: np.ndarray
    hoop_stress: np.ndarray
    axial_stress: np.ndarray
    radial_displacement: np.ndarray
    #: max |PDE residual| / max |sigma_r| over the collocation points
    equilibrium_residual: float


def _cheb_nodes(n: int, lo: float, hi: float) -> np.ndarray:
    """Gauss-Lobatto-Chebyshev nodes mapped to ``[lo, hi]``."""
    j = np.arange(n)
    x = -np.cos(np.pi * j / (n - 1))         # ascending: node 0 sits at lo
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * x


def _cheb_matrix(n: int, lo: float, hi: float) -> np.ndarray:
    """First-derivative Chebyshev collocation matrix on ``[lo, hi]``."""
    j = np.arange(n)
    x = -np.cos(np.pi * j / (n - 1))         # ascending, same ordering as nodes
    c = np.ones(n)
    c[0] = c[-1] = 2.0
    c = c * (-1.0) ** j
    dx = x[:, None] - x[None, :]
    np.fill_diagonal(dx, 1.0)
    d = np.outer(c, 1.0 / c) / dx
    d -= np.diag(d.sum(axis=1))
    return d * 2.0 / (hi - lo)


def solve_thick_cylinder_1d(
    radius,
    *,
    inner_radius: float,
    outer_radius: float,
    inner_pressure: float,
    outer_pressure: float = 0.0,
    young_modulus: float,
    poisson_ratio: float,
    plane: str = "strain",
    n_modes: int = 33,
) -> OracleState:
    """Solve the axisymmetric thick cylinder by Chebyshev collocation.

    ``radius`` is an array of sample radii in ``[inner_radius, outer_radius]``;
    the returned :class:`OracleState` holds the interpolated fields there.
    Pressure is the standard compression-positive convention
    (``sigma_r(a) = -inner_pressure``).
    """
    if not (0.0 < inner_radius < outer_radius):
        raise ValueError("require 0 < inner_radius < outer_radius")
    if inner_pressure < 0.0 or outer_pressure < 0.0:
        raise ValueError("pressures must be non-negative")
    if plane not in ("stress", "strain"):
        raise ValueError("plane must be 'stress' or 'strain'")
    nu, e = float(poisson_ratio), float(young_modulus)
    if plane == "strain":
        stiff = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
        a_r, b_r = (1.0 - nu), nu        # sigma_r = C (a_r u' + b_r u/r)
    else:
        stiff = e / (1.0 - nu * nu)
        a_r, b_r = 1.0, nu
    a, b = float(inner_radius), float(outer_radius)

    r = _cheb_nodes(n_modes, a, b)
    d1 = _cheb_matrix(n_modes, a, b)

    # sigma_r(r_j) = stiff * (a_r * (D u)_j + b_r * u_j / r_j)  ->  matrix T
    traction = stiff * (a_r * d1 + b_r * np.diag(1.0 / r))
    # sigma_theta(r_j) = stiff * (b_r * (D u)_j + a_r * u_j / r_j)  ->  matrix S
    hoop = stiff * (b_r * d1 + a_r * np.diag(1.0 / r))
    # (r sigma_r)' - sigma_theta = 0  at the interior collocation points
    operator = d1 @ np.diag(r) @ traction - hoop

    rhs = np.zeros(n_modes)
    operator[0] = traction[0]
    rhs[0] = -inner_pressure
    operator[-1] = traction[-1]
    rhs[-1] = -outer_pressure

    u = np.linalg.solve(operator, rhs)
    du = d1 @ u
    sigma_r = stiff * (a_r * du + b_r * u / r)
    sigma_theta = stiff * (b_r * du + a_r * u / r)
    axial = np.zeros_like(r) if plane == "stress" else nu * (sigma_r + sigma_theta)
    residual = np.abs((d1 @ (r * sigma_r)) - sigma_theta)
    scale = float(np.max(np.abs(sigma_r))) or 1.0

    sample = np.asarray(radius, dtype=float)
    order = np.argsort(r)
    x_nodes = _to_cheb_x(r[order], a, b)
    x_sample = _to_cheb_x(sample, a, b)

    def interpolate(values):
        coeffs = np.polynomial.chebyshev.chebfit(x_nodes, np.asarray(values)[order], n_modes - 1)
        return np.polynomial.chebyshev.chebval(x_sample, coeffs)

    sr_s = interpolate(sigma_r)
    st_s = interpolate(sigma_theta)
    u_s = interpolate(u)
    sz_s = np.zeros_like(sample) if plane == "stress" else nu * (sr_s + st_s)
    return OracleState(
        radius=sample,
        radial_stress=sr_s,
        hoop_stress=st_s,
        axial_stress=sz_s,
        radial_displacement=u_s,
        equilibrium_residual=float(residual.max() / scale),
    )


def _to_cheb_x(r, a: float, b: float):
    return (2.0 * np.asarray(r, dtype=float) - (a + b)) / (b - a)
