"""Exact solution for pure bending of a rectangular curved bar (Winkler-Bach).

Why this reference
------------------
A curved bar in pure bending is the classical closed-form problem in which the
neutral axis of a curved member is *not* the centroid: it shifts towards the
centre of curvature by ``e = r_bar - r_n`` with
``r_n = h / ln(r_o/r_i)`` for a rectangular section (Winkler-Bach formula, see
Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed., Sec. 30).  The hoop
stress is hyperbolic in ``r``, so the field is strongly non-uniform and cannot
be reproduced by the constant-strain patch tests that the plane-stress path of
:mod:`tensorfem.continuum` ships with.  That makes it a demanding and
independent reference for curved, bending-dominated geometry such as a ship
frame, a ring stiffener or a hatch-coaming insert.

Two independent routes to the same numbers are implemented and must agree:

``curved_bar_coefficients``
    solves the three conditions of the Airy stress function
    ``Phi = A ln r + B r^2 ln r + C r^2``
    (traction-free bore and outer surface, prescribed bending moment) as a
    ``3x3`` linear system;
``winkler_bach_hoop_stress``
    evaluates the textbook closed form.

:func:`curved_bar_state` returns the stress *and* displacement fields.  The
displacement below is derived from the exact strain field and is stated in the
gauge used by the verification study: no rigid rotation (``u_theta = 0`` on the
symmetry ray) and one free translation ``x_translation`` along ``x``.  The
oriented-wedge term ``4 c1 B r theta`` is retained because pure bending of a
curved bar is a wedge solution: the radial sections rotate relative to each
other, so the hoop strain of that term is constant and single valued on an arc
of less than a full circle.  Its compatibility is checked numerically by
:func:`strain_residual`.

All formulas are plane problems on a *curved rectangular* section ``[a, b]`` of
out-of-plane thickness ``width``; SI units throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "CurvedBarState",
    "curved_bar_coefficients",
    "curved_bar_state",
    "curved_bar_end_loads",
    "curved_bar_symmetry_dofs",
    "curved_bar_gauge_dof",
    "eccentricity",
    "neutral_radius",
    "strain_residual",
    "winkler_bach_hoop_stress",
]


def neutral_radius(r_inner: float, r_outer: float) -> float:
    """Neutral-axis radius ``r_n = h / ln(r_o/r_i)`` of a rectangular section."""
    if not 0.0 < r_inner < r_outer:
        raise ValueError("require 0 < r_inner < r_outer")
    return (r_outer - r_inner) / math.log(r_outer / r_inner)


def eccentricity(r_inner: float, r_outer: float) -> float:
    """Shift ``e = r_bar - r_n`` of the neutral axis away from the centroid."""
    r_bar = 0.5 * (r_inner + r_outer)
    return r_bar - neutral_radius(r_inner, r_outer)


def _compliance(young: float, poisson: float, plane: str) -> tuple[float, float]:
    """``(c1, c2)`` with ``eps_r = c1 sigma_r + c2 sigma_theta`` and vice versa."""
    if plane == "stress":
        return 1.0 / young, -poisson / young
    if plane == "strain":
        return (1.0 - poisson * poisson) / young, -poisson * (1.0 + poisson) / young
    raise ValueError("plane must be 'stress' or 'strain'")


def _moment_of(coefficients: torch.Tensor, r_inner: float, r_outer: float, width: float) -> float:
    """Bending moment ``M = width * int_a^b sigma_theta (r_bar - r) dr``.

    Uses the antiderivative of ``sigma_theta = -A/r^2 + 2B ln r + 3B + 2C``
    multiplied by ``(r_bar - r)``; the constant part integrates to zero because
    ``r_bar`` is the centroid of the rectangular section.
    """
    a, b = r_inner, r_outer
    r_bar = 0.5 * (a + b)
    log_ratio = math.log(b / a)
    a_part = -(r_bar * (1.0 / a - 1.0 / b) - log_ratio)

    def ib(x: float) -> float:
        return r_bar * (x * math.log(x) - x) - 0.5 * x * x * math.log(x) + 0.25 * x * x

    b_part = 2.0 * (ib(b) - ib(a))
    return width * float(coefficients[0] * a_part + coefficients[1] * b_part)


def _solve_coefficients(
    r_inner: float, r_outer: float, width: float, moment: float
) -> torch.Tensor:
    """Solve the ``3x3`` Airy system for ``(A, B, C)``."""
    a, b = r_inner, r_outer
    r_bar = 0.5 * (a + b)
    log_ratio = math.log(b / a)
    a_part = -(r_bar * (1.0 / a - 1.0 / b) - log_ratio)

    def ib(x: float) -> float:
        return r_bar * (x * math.log(x) - x) - 0.5 * x * x * math.log(x) + 0.25 * x * x

    b_part = 2.0 * (ib(b) - ib(a))
    system = torch.tensor(
        [
            [1.0 / (a * a), 2.0 * math.log(a) + 1.0, 2.0],
            [1.0 / (b * b), 2.0 * math.log(b) + 1.0, 2.0],
            [width * a_part, width * b_part, 0.0],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor([0.0, 0.0, moment], dtype=torch.float64)
    if abs(float(torch.linalg.det(system))) < 1e-18:
        raise ValueError("degenerate curved-bar system")
    return torch.linalg.solve(system, rhs)


def curved_bar_coefficients(
    r_inner: float, r_outer: float, width: float, moment: float
) -> torch.Tensor:
    """Airy coefficients ``(A, B, C)`` for a rectangular curved bar.

    ``moment > 0`` produces hoop tension on the inner fibre, as in the
    Winkler-Bach convention used by :func:`winkler_bach_hoop_stress`.
    """
    if width <= 0.0:
        raise ValueError("width must be positive")
    coefficients = _solve_coefficients(r_inner, r_outer, width, moment)
    if abs(_moment_of(coefficients, r_inner, r_outer, width) - moment) > 1e-9 * max(abs(moment), 1.0):
        raise ValueError("moment condition not reproduced by the solved coefficients")
    return coefficients


def winkler_bach_hoop_stress(
    radius: torch.Tensor,
    *,
    r_inner: float,
    r_outer: float,
    width: float,
    moment: float,
) -> torch.Tensor:
    """Closed form ``sigma_theta = M (r_n - r) / (A e r)`` with ``A = width * h``."""
    r_n = neutral_radius(r_inner, r_outer)
    e = 0.5 * (r_inner + r_outer) - r_n
    area = width * (r_outer - r_inner)
    return moment * (r_n - radius) / (area * e * radius)


@dataclass(frozen=True)
class CurvedBarState:
    """Exact state of a curved rectangular bar in pure bending."""

    radius: torch.Tensor
    hoop_stress: torch.Tensor
    radial_stress: torch.Tensor
    shear_stress: torch.Tensor
    radial_displacement: torch.Tensor
    hoop_displacement: torch.Tensor
    moment: float
    neutral_radius: float
    eccentricity: float
    area: float
    coefficients: torch.Tensor


def curved_bar_state(
    radius: torch.Tensor,
    *,
    r_inner: float,
    r_outer: float,
    width: float,
    moment: float,
    young: float,
    poisson: float,
    plane: str = "stress",
    polar_angle: torch.Tensor | float = 0.0,
    x_translation: float = 0.0,
) -> CurvedBarState:
    """Exact stress and displacement of a curved bar at ``radius``.

    ``polar_angle`` (radians, measured from the symmetry ray) is only used for
    the displacement field; stresses are angle independent.  ``x_translation``
    is the single remaining rigid-body gauge of a polar model whose
    ``u_theta = 0`` condition has already removed the rotation.
    """
    radius = torch.as_tensor(radius, dtype=torch.float64).reshape(-1)
    coefficients = curved_bar_coefficients(r_inner, r_outer, width, moment)
    a_coef, b_coef, c_coef = (float(x) for x in coefficients)

    radial = a_coef / radius**2 + 2.0 * b_coef * torch.log(radius) + b_coef + 2.0 * c_coef
    hoop = -a_coef / radius**2 + 2.0 * b_coef * torch.log(radius) + 3.0 * b_coef + 2.0 * c_coef

    c1, c2 = _compliance(young, poisson, plane)
    # eps_r = du_r/dr integrated analytically in r
    u_r = (
        -(c1 - c2) * a_coef / radius
        + 2.0 * (c1 + c2) * b_coef * (radius * torch.log(radius) - radius)
        + (c1 + 3.0 * c2) * b_coef * radius
        + 2.0 * (c1 + c2) * c_coef * radius
    )
    theta = torch.as_tensor(polar_angle, dtype=torch.float64)
    theta = theta.reshape(-1) if theta.numel() > 1 else theta.reshape(()).expand(radius.shape[0])
    u_theta = 4.0 * c1 * b_coef * radius * theta

    return CurvedBarState(
        radius=radius,
        hoop_stress=hoop,
        radial_stress=radial,
        shear_stress=torch.zeros_like(radius),
        radial_displacement=u_r + x_translation * torch.cos(theta),
        hoop_displacement=u_theta - x_translation * torch.sin(theta),
        moment=moment,
        neutral_radius=neutral_radius(r_inner, r_outer),
        eccentricity=0.5 * (r_inner + r_outer) - neutral_radius(r_inner, r_outer),
        area=width * (r_outer - r_inner),
        coefficients=coefficients,
    )


def strain_residual(
    radius: torch.Tensor,
    *,
    r_inner: float,
    r_outer: float,
    width: float,
    moment: float,
    young: float,
    poisson: float,
    plane: str = "stress",
) -> dict[str, float]:
    """Check that the closed-form displacement reproduces the closed-form stress.

    Kinematic vs constitutive strains are compared on a radial sample, i.e. the
    displacement field of :func:`curved_bar_state` is differentiated and pushed
    through the constitutive law.  A correct derivation leaves a relative
    residual at round-off; a sign or coefficient error shows up immediately.
    """
    radius = torch.as_tensor(radius, dtype=torch.float64).reshape(-1)
    dx = radius * 1e-6
    state_plus = curved_bar_state(
        radius + dx, r_inner=r_inner, r_outer=r_outer, width=width, moment=moment,
        young=young, poisson=poisson, plane=plane,
    )
    state_minus = curved_bar_state(
        radius - dx, r_inner=r_inner, r_outer=r_outer, width=width, moment=moment,
        young=young, poisson=poisson, plane=plane,
    )
    state = curved_bar_state(
        radius, r_inner=r_inner, r_outer=r_outer, width=width, moment=moment,
        young=young, poisson=poisson, plane=plane,
    )
    eps_r_kinematic = (state_plus.radial_displacement - state_minus.radial_displacement) / (2.0 * dx)
    # u_theta = 4 c1 B r theta, so (1/r) du_theta/dtheta adds the constant 4 c1 B
    eps_theta_kinematic = state.radial_displacement / radius + 4.0 * (
        _compliance(young, poisson, plane)[0] * float(state.coefficients[1])
    )
    c1, c2 = _compliance(young, poisson, plane)
    eps_r = c1 * state.radial_stress + c2 * state.hoop_stress
    eps_theta = c2 * state.radial_stress + c1 * state.hoop_stress
    return {
        "eps_r_relative": float(((eps_r_kinematic - eps_r).abs() / eps_r.abs().max()).max()),
        "eps_theta_relative": float(((eps_theta_kinematic - eps_theta).abs() / eps_theta.abs().max()).max()),
    }


def curved_bar_end_loads(
    nodes: torch.Tensor,
    n_theta: int,
    n_radial: int,
    *,
    r_inner: float,
    r_outer: float,
    width: float,
    moment: float,
    sector_angle: float,
    dtype=torch.float64,
) -> torch.Tensor:
    """Consistent nodal forces for the exact end moment on the ``theta = max`` face.

    The traction on that radial face is ``t = sigma_theta(r) e_theta`` with the
    *exact Airy* hoop stress (not the Winkler-Bach approximation, which is 0.7 %
    off at ``r_o/r_i = 2`` and would cap the achievable convergence).  Each of
    the ``n_radial`` straight radial edges carries ``width * int N_a(r) t(r) dr``
    evaluated with three-point Gauss quadrature.
    """
    if nodes.shape[1] != 2:
        raise ValueError("nodes must have shape [n_nodes, 2]")
    n_nodes = (n_radial + 1) * (n_theta + 1)
    if len(nodes) != n_nodes:
        raise ValueError("node count does not match n_theta/n_radial")
    forces = torch.zeros(2 * n_nodes, dtype=dtype)
    # outward normal of the theta = sector_angle face is +e_theta = (-sin, cos);
    # the traction is t = sigma . e_theta = sigma_theta e_theta (sigma_r has no
    # component through that face), so a pure couple carries zero net force.
    unit = torch.tensor([-math.sin(sector_angle), math.cos(sector_angle)], dtype=dtype)
    gauss = (
        (-math.sqrt(3.0 / 5.0), 5.0 / 9.0),
        (0.0, 8.0 / 9.0),
        (math.sqrt(3.0 / 5.0), 5.0 / 9.0),
    )
    stride = n_theta + 1
    a_coef, b_coef, c_coef = (float(x) for x in curved_bar_coefficients(r_inner, r_outer, width, moment))
    for i in range(n_radial):
        node_a = i * stride + n_theta
        node_b = (i + 1) * stride + n_theta
        r_a = float(torch.linalg.vector_norm(nodes[node_a]))
        r_b = float(torch.linalg.vector_norm(nodes[node_b]))
        half = 0.5 * (r_b - r_a)
        mid = 0.5 * (r_b + r_a)
        for xi, weight in gauss:
            r = mid + half * xi
            # exact Airy hoop stress: -A/r^2 + 2B ln r + 3B + 2C
            sigma = -a_coef / (r * r) + 2.0 * b_coef * math.log(r) + 3.0 * b_coef + 2.0 * c_coef
            # linear edge shape functions: N_a = (1 - xi)/2, N_b = (1 + xi)/2
            contribution = weight * half * width * sigma
            forces[2 * node_a] += 0.5 * (1.0 - xi) * contribution * unit[0]
            forces[2 * node_a + 1] += 0.5 * (1.0 - xi) * contribution * unit[1]
            forces[2 * node_b] += 0.5 * (1.0 + xi) * contribution * unit[0]
            forces[2 * node_b + 1] += 0.5 * (1.0 + xi) * contribution * unit[1]
    return forces


def curved_bar_symmetry_dofs(nodes: torch.Tensor, n_theta: int, n_radial: int, *, dtype=torch.long) -> torch.Tensor:
    """``u_theta = 0`` on the ``theta = 0`` ray, i.e. ``u_y = 0`` there.

    The symmetry ray is the ``x`` axis, so the exact ``u_theta = 0`` condition is
    an axis-aligned constraint on the Cartesian ``y`` degree of freedom.  It
    removes rigid translation in ``y`` and rigid rotation about the origin and
    reproduces the exact symmetry condition of the pure-bending state.
    """
    n_nodes = (n_radial + 1) * (n_theta + 1)
    if len(nodes) != n_nodes:
        raise ValueError("node count does not match n_theta/n_radial")
    fixed = [2 * (i * (n_theta + 1)) + 1 for i in range(n_radial + 1)]
    return torch.tensor(fixed, dtype=dtype)


def curved_bar_gauge_dof(nodes: torch.Tensor, n_theta: int, n_radial: int) -> int:
    """``x`` degree of freedom of the mid-radius node on the symmetry ray.

    With ``u_y = 0`` on the whole ray only the rigid translation along ``x``
    survives; this single dof gauges it.  The verification study pins it to the
    exact value and separately repeats the solve with a zero pin to show that
    the interior field is gauge independent.
    """
    n_nodes = (n_radial + 1) * (n_theta + 1)
    if len(nodes) != n_nodes:
        raise ValueError("node count does not match n_theta/n_radial")
    return 2 * ((n_radial // 2) * (n_theta + 1))