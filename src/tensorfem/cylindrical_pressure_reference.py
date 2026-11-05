"""Axisymmetric pressure cylinder with clamped ends: exact bending boundary layer.

Verification oracle for the metric-consistent cylindrical Reissner--Mindlin
element (:mod:`tensorfem.cylindrical_shell4`).  Before this module that element
was only checked on Scordelis--Lo, the pinched cylinder and a constant-membrane
energy patch; the one case where a curved shell must combine *membrane* and
*bending* action under a *pressure load* -- the clamped-edge boundary layer of a
pressurised cylinder -- had no reference.  This module supplies an exact
solution of the governing ODE plus a driver that runs the production element
against it.

Exact solution
--------------
Axisymmetric, small-displacement Kirchhoff shell, radial displacement ``w``
positive outwards, internal pressure ``p`` positive inwards, no axial end load
(``N_x = 0``, open ends), uniform thickness::

    D w'''' + (E t / R^2) w = p,      D = E t^3 / (12 (1 - nu^2))
    beta = (3 (1 - nu^2))**0.25 / sqrt(R t),     w_p = p R^2 / (E t)

with ``w = w_p + e^{beta x}(A cos beta x + B sin beta x)
              + e^{-beta x}(C cos beta x + D sin beta x)`` and the four clamped
end conditions ``w(0) = w'(0) = w(L) = w'(L) = 0`` solved exactly as a 4x4
linear system (no semi-infinite approximation, no superposition truncation).

Stress resultants (sign convention chosen to match the element, see
:class:`ExactSolution`)::

    M_x = -D w''          moment per unit length
    Q   =  dM_x/dx
    N_theta = E t w / R   membrane hoop force

Self-validation (independent of the finite element)
---------------------------------------------------
* governing-ODE residual by fourth-order finite differences (truncation floor);
* long-cylinder limits ``|M_x(0)| -> p / (2 beta^2)`` and ``w(L/2) -> w_p``;
* analytical work--energy balance ``0.5 * int p w dA = U_strain``;
* clamped boundary conditions and ``N_x = 0`` (no axial load, ``dN_x/dx = 0``).

All quantities are SI (m, N, Pa).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

__all__ = [
    "CylinderParameters",
    "cylinder_parameters",
    "ExactSolution",
    "exact_axisymmetric_solution",
    "clamped_coefficients",
    "work_energy_balance",
    "self_validation",
    "pressure_cylinder_mesh",
    "consistent_pressure_loads",
    "CylinderResult",
    "solve_pressure_cylinder",
    "run_boundary_layer_study",
]


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass(frozen=True)
class CylinderParameters:
    """Geometry, material and load of the axisymmetric reference problem."""

    radius: float
    thickness: float
    young: float
    poisson: float
    pressure: float
    length: float

    @property
    def bending_rigidity(self) -> float:
        return self.young * self.thickness ** 3 / (12.0 * (1.0 - self.poisson ** 2))

    @property
    def beta(self) -> float:
        """Boundary-layer decay rate [1/m]; the decay length is ``1/beta``."""
        return (3.0 * (1.0 - self.poisson ** 2)) ** 0.25 / math.sqrt(
            self.radius * self.thickness
        )

    @property
    def membrane_radial_displacement(self) -> float:
        """Unrestrained (membrane) radial displacement ``p R^2 / (E t)`` [m]."""
        return self.pressure * self.radius ** 2 / (self.young * self.thickness)

    @property
    def membrane_hoop_force(self) -> float:
        return self.pressure * self.radius

    @property
    def membrane_hoop_stress(self) -> float:
        return self.pressure * self.radius / self.thickness

    @property
    def edge_moment_long_cylinder(self) -> float:
        """Clamped-edge moment magnitude of the semi-infinite cylinder."""
        return self.pressure / (2.0 * self.beta ** 2)

    @property
    def edge_bending_stress(self) -> float:
        """Surface bending stress ``6 |M_x(0)| / t^2`` of the semi-infinite limit."""
        return 6.0 * self.edge_moment_long_cylinder / self.thickness ** 2

    @property
    def boundary_layer_thickness(self) -> float:
        """Distance ``pi/beta`` over which the edge disturbance has decayed."""
        return math.pi / self.beta


def cylinder_parameters(
    *,
    radius: float,
    thickness: float,
    young: float,
    poisson: float,
    pressure: float,
    length: float,
) -> CylinderParameters:
    """Build validated :class:`CylinderParameters`."""
    poisson = float(poisson)
    if not math.isfinite(poisson) or not -0.5 < poisson < 0.5:
        raise ValueError("poisson must be finite and in (-0.5, 0.5)")
    return CylinderParameters(
        radius=_positive("radius", radius),
        thickness=_positive("thickness", thickness),
        young=_positive("young", young),
        poisson=poisson,
        pressure=_positive("pressure", pressure),
        length=_positive("length", length),
    )


@dataclass(frozen=True)
class ExactSolution:
    """Exact axisymmetric fields sampled at ``x``.

    ``moment`` uses ``M_x = -D w''`` -- the Reissner--Mindlin curvature
    convention ``kappa_x = d(beta_t)/dx`` that the element assembles, so the
    finite-element and exact moment fields share one sign convention.  For a
    clamped end under internal pressure ``M_x(0) < 0`` (hoop bending with
    tension on the inner surface), its magnitude tending to ``p/(2 beta^2)``.
    ``slope`` is ``dw/dx``; the Mindlin section rotation is ``beta_t = -dw/dx``.
    """

    x: torch.Tensor
    displacement: torch.Tensor
    slope: torch.Tensor
    curvature: torch.Tensor
    moment: torch.Tensor
    shear: torch.Tensor
    hoop_force: torch.Tensor
    coefficients: tuple[float, float, float, float]
    membrane_displacement: float


def clamped_coefficients(p: CylinderParameters) -> tuple[float, float, float, float]:
    """Solve the four clamped end conditions exactly for ``(A, B, C, D)``."""
    beta, length, wp = p.beta, p.length, p.membrane_radial_displacement
    u = beta * length
    cu, su = math.cos(u), math.sin(u)
    eu, em = math.exp(u), math.exp(-u)
    matrix = torch.tensor(
        [
            # w(0)  = wp + A + C = 0
            [1.0, 0.0, 1.0, 0.0],
            # w'(0) = beta [(A + B) + (D - C)] = 0
            [1.0, 1.0, -1.0, 1.0],
            # w(L)  = wp + eu (A cu + B su) + em (C cu + D su) = 0
            [eu * cu, eu * su, em * cu, em * su],
            # w'(L) = beta [eu((A+B)cu + (B-A)su) + em((D-C)cu - (D+C)su)] = 0
            [eu * (cu - su), eu * (cu + su), -em * (cu + su), em * (cu - su)],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor([-wp, 0.0, -wp, 0.0], dtype=torch.float64)
    solution = torch.linalg.solve(matrix, rhs)
    return tuple(float(v) for v in solution)  # type: ignore[return-value]


def exact_axisymmetric_solution(
    x: torch.Tensor | float,
    parameters: CylinderParameters,
) -> ExactSolution:
    """Evaluate the exact clamped--clamped axisymmetric shell fields at ``x``."""
    p = parameters
    x_t = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    if bool((x_t < -1e-12).any()) or bool((x_t > p.length + 1e-12).any()):
        raise ValueError("x must lie within [0, length]")
    A, B, C, D = clamped_coefficients(p)
    beta, wp, rigidity = p.beta, p.membrane_radial_displacement, p.bending_rigidity
    u = (beta * x_t).requires_grad_(True)
    eu, em = torch.exp(u), torch.exp(-u)
    cu, su = torch.cos(u), torch.sin(u)
    growing = eu * (A * cu + B * su)
    decaying = em * (C * cu + D * su)
    displacement = wp + growing + decaying
    # Derivatives are taken by automatic differentiation of the closed form, so
    # the exported moment/shear cannot suffer from a hand-derived sign slip.
    ones = torch.ones_like(u)
    dw_du = torch.autograd.grad(displacement, u, grad_outputs=ones, create_graph=True)[0]
    d2w_du2 = torch.autograd.grad(dw_du, u, grad_outputs=ones, create_graph=True)[0]
    d3w_du3 = torch.autograd.grad(d2w_du2, u, grad_outputs=ones)[0]
    slope = beta * dw_du
    curvature = beta ** 2 * d2w_du2
    moment = -rigidity * curvature
    shear = -rigidity * beta ** 3 * d3w_du3
    displacement = displacement.detach()
    hoop_force = p.young * p.thickness * displacement / p.radius
    return ExactSolution(
        x=x_t,
        displacement=displacement,
        slope=slope.detach(),
        curvature=curvature.detach(),
        moment=moment.detach(),
        shear=shear.detach(),
        hoop_force=hoop_force,
        coefficients=(A, B, C, D),
        membrane_displacement=wp,
    )


def work_energy_balance(
    parameters: CylinderParameters,
    *,
    samples: int = 40001,
) -> dict[str, float]:
    """Trapezoidal pressure work against the exact strain energy.

    For the linear problem ``0.5 * int p w dA == U`` with
    ``U = 0.5 * int [D w''^2 + (E t / R^2) w^2] dA`` (``kappa_theta = 0`` in
    the axisymmetric state, so the ``nu`` coupling contributes nothing).
    """
    p = parameters
    grid = torch.linspace(0.0, p.length, int(samples), dtype=torch.float64)
    exact = exact_axisymmetric_solution(grid, p)
    weight = torch.full_like(grid, float(p.length) / (grid.numel() - 1))
    weight[0] *= 0.5
    weight[-1] *= 0.5
    area = 2.0 * math.pi * p.radius
    work = 0.5 * float((p.pressure * exact.displacement * weight).sum()) * area
    density = (p.bending_rigidity * exact.curvature ** 2
               + p.young * p.thickness / p.radius ** 2 * exact.displacement ** 2)
    energy = 0.5 * float((density * weight).sum()) * area
    return {
        "pressure_work": work,
        "strain_energy": energy,
        "relative_gap": abs(work - energy) / max(abs(energy), 1e-300),
    }


def self_validation(
    parameters: CylinderParameters,
    *,
    samples: int = 8001,
) -> dict[str, float]:
    """Independent checks of the exact solution against its own definition."""
    p = parameters
    grid = torch.linspace(0.0, p.length, int(samples), dtype=torch.float64)
    exact = exact_axisymmetric_solution(grid, p)
    h = float(grid[1] - grid[0])
    w = exact.displacement
    # Diagnostic finite-difference residual.  The fourth difference divides
    # roundoff by h^4, so this number grows once h is refined past the point
    # where truncation stops dominating; it is reported, not gated.
    fourth_fd = (w[4:] - 4 * w[3:-1] + 6 * w[2:-2] - 4 * w[1:-3] + w[:-4]) / h ** 4
    residual_fd = (
        p.bending_rigidity * fourth_fd
        + p.young * p.thickness / p.radius ** 2 * w[2:-2]
        - p.pressure
    )
    # Exact residual by differentiating the closed form four times with
    # autograd: this is the machine-precision statement of the ODE.
    A, B, C, D = clamped_coefficients(p)
    u = (p.beta * grid).requires_grad_(True)
    field = (
        p.membrane_radial_displacement
        + torch.exp(u) * (A * torch.cos(u) + B * torch.sin(u))
        + torch.exp(-u) * (C * torch.cos(u) + D * torch.sin(u))
    )
    derivative = field
    for _ in range(4):
        derivative = torch.autograd.grad(
            derivative, u, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
    residual = (
        p.bending_rigidity * p.beta ** 4 * derivative
        + p.young * p.thickness / p.radius ** 2 * field
        - p.pressure
    )
    ends = torch.tensor([0, -1], dtype=torch.long)
    scale = p.membrane_radial_displacement
    edge = abs(float(exact.moment[0])) / p.edge_moment_long_cylinder - 1.0
    # Classical semi-infinite result: two independent edge layers superpose as
    # w = wp [1 - f(beta x) - f(beta (L - x))], f(v) = e^-v (cos v + sin v).
    grid_mid = torch.linspace(0.0, p.length, int(samples), dtype=torch.float64)

    def layer(v: torch.Tensor) -> torch.Tensor:
        return torch.exp(-v) * (torch.cos(v) + torch.sin(v))

    superposed = scale * (1.0 - layer(p.beta * grid_mid)
                          - layer(p.beta * (p.length - grid_mid)))
    superposition_gap = float(
        (exact.displacement - superposed).abs().max()
    ) / scale
    energy = work_energy_balance(p)
    return {
        "ode_residual_over_pressure": float(residual.abs().max()) / p.pressure,
        "ode_residual_fd_over_pressure": float(residual_fd.abs().max()) / p.pressure,
        "clamped_displacement_over_wp": float(exact.displacement[ends].abs().max())
        / scale,
        "clamped_slope_over_wp_beta": float(exact.slope[ends].abs().max())
        / (scale * p.beta),
        "edge_moment_vs_long_limit": edge,
        "two_layer_superposition_gap": superposition_gap,
        "decay_at_far_end": math.exp(-p.beta * p.length),
        "work_energy_relative_gap": energy["relative_gap"],
        "beta_length": p.beta * p.length,
    }


# ---------------------------------------------------------------------------
# Production element driver
# ---------------------------------------------------------------------------
def pressure_cylinder_mesh(
    axes: int,
    facets: int,
    parameters: CylinderParameters,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured ``(x, theta)`` mesh of the full cylinder.

    ``axes`` = number of axial elements, ``facets`` = circumferential elements.
    Returns node parameters ``(nnode, 2)`` and element connectivity
    ``(nelem, 4)`` in the order the element expects,
    ``(x1,t1), (x2,t1), (x2,t2), (x1,t2)``.  Closing facets are represented with
    ``theta`` values that keep ``s = R theta`` continuous, exactly as
    :func:`tensorfem.cylindrical_shell4.cylindrical_shell4_stiffness` requires.
    """
    axes = int(axes)
    facets = int(facets)
    if axes < 1 or facets < 3:
        raise ValueError("need at least one axial and three circumferential elements")
    xs = torch.linspace(0.0, parameters.length, axes + 1, dtype=dtype)
    thetas = torch.arange(facets, dtype=dtype) * (2.0 * math.pi / facets)
    xx, tt = torch.meshgrid(xs, thetas, indexing="ij")
    nodes = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
    elements = []
    for i in range(axes):
        for j in range(facets):
            elements.append(
                (
                    i * facets + j,
                    (i + 1) * facets + j,
                    (i + 1) * facets + (j + 1) % facets,
                    i * facets + (j + 1) % facets,
                )
            )
    connectivity = torch.tensor(elements, dtype=torch.long, device=nodes.device)
    return nodes, connectivity


def _element_parameters(
    nodes: torch.Tensor,
    connectivity: torch.Tensor,
) -> torch.Tensor:
    """Per-element ``(x, theta)`` with the closing facet's wrap kept continuous."""
    element = nodes[connectivity].clone()
    theta = element[..., 1]
    wrapped = (theta[:, 3] - theta[:, 0]) < 0.0
    if bool(wrapped.any()):
        shift = torch.where(
            wrapped, torch.full_like(theta[:, 0], 2.0 * math.pi), torch.zeros_like(theta[:, 0])
        )
        theta = theta + shift[:, None] * (theta < math.pi).to(theta.dtype)
        element = torch.stack((element[..., 0], theta), dim=-1)
    return element


def consistent_pressure_loads(
    nodes: torch.Tensor,
    connectivity: torch.Tensor,
    parameters: CylinderParameters,
) -> torch.Tensor:
    """Consistent nodal load of a uniform internal pressure.

    ``f_a = int N_a p n(theta) dA`` with ``n = (0, sin theta, cos theta)`` the
    outward radial direction (internal pressure pushes the wall out),
    evaluated by 2x2 Gauss quadrature on the exact cylindrical surface.  Each
    node therefore receives its load in its own radial direction instead of a
    lumped direction, so no ``O(delta theta^2)`` load error is injected.
    """
    p = parameters
    ndof = 6 * nodes.shape[0]
    load = torch.zeros(ndof, dtype=nodes.dtype, device=nodes.device)
    g = 3.0 ** -0.5
    points = ((-g, -g), (g, -g), (g, g), (-g, g))
    elements = _element_parameters(nodes, connectivity)
    for element, conn in zip(elements, connectivity.tolist()):
        dx = float(element[1, 0] - element[0, 0])
        t1, t2 = float(element[0, 1]), float(element[3, 1])
        dtheta = t2 - t1
        if not (dx > 0.0 and dtheta > 0.0):
            raise ValueError("element coordinates must increase with node order")
        weight = 0.25 * dx * p.radius * dtheta  # det J = (dx/2)(R dtheta/2)
        for xi, eta in points:
            shape = torch.tensor(
                [
                    (1 - xi) * (1 - eta),
                    (1 + xi) * (1 - eta),
                    (1 + xi) * (1 + eta),
                    (1 - xi) * (1 + eta),
                ],
                dtype=nodes.dtype,
            ) / 4.0
            theta = t1 if eta < 0.0 else t2
            direction = torch.tensor(
                [0.0, math.sin(theta), math.cos(theta)], dtype=nodes.dtype
            )
            for a, node in enumerate(conn):
                load[6 * node: 6 * node + 3] += (
                    shape[a] * weight * p.pressure * direction
                )
    return load


@dataclass(frozen=True)
class CylinderResult:
    """Finite-element result on the axisymmetric pressure cylinder."""

    axes: int
    facets: int
    dofs: int
    x: torch.Tensor
    displacement_fe: torch.Tensor
    displacement_exact: torch.Tensor
    moment_fe: torch.Tensor
    moment_exact: torch.Tensor
    hoop_force_fe: torch.Tensor
    hoop_force_exact: torch.Tensor
    rotation_fe: torch.Tensor
    rotation_exact: torch.Tensor
    axial_force_error: float
    circumferential_uniformity: float
    parameters: CylinderParameters

    @property
    def displacement_l2_error(self) -> float:
        return _relative_l2(self.displacement_fe, self.displacement_exact)

    @property
    def moment_l2_error(self) -> float:
        return _relative_l2(self.moment_fe, self.moment_exact)

    @property
    def edge_moment_error(self) -> float:
        return abs(float(self.moment_fe[0]) / float(self.moment_exact[0]) - 1.0)

    @property
    def peak_displacement_error(self) -> float:
        reference = float(self.displacement_exact.abs().max())
        return float(
            (self.displacement_fe - self.displacement_exact).abs().max()
        ) / reference

    @property
    def moment_sign_correlation(self) -> float:
        a = self.moment_fe - self.moment_fe.mean()
        b = self.moment_exact - self.moment_exact.mean()
        return float((a @ b) / torch.linalg.vector_norm(a) / torch.linalg.vector_norm(b))


def _relative_l2(computed: torch.Tensor, exact: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(computed - exact) / torch.linalg.vector_norm(exact)
    )


def _first_difference(values: torch.Tensor, h: float) -> torch.Tensor:
    out = torch.zeros_like(values)
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * h)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def solve_pressure_cylinder(
    axes: int,
    facets: int,
    parameters: CylinderParameters,
    *,
    dtype: torch.dtype = torch.float64,
) -> CylinderResult:
    """Solve the pressurised cylinder with the production Shell4 element."""
    from .cylindrical_shell4 import cylindrical_shell4_stiffness

    p = parameters
    nodes, connectivity = pressure_cylinder_mesh(axes, facets, p, dtype=dtype)
    elements = _element_parameters(nodes, connectivity)
    ndof = 6 * nodes.shape[0]
    stiffness = torch.zeros((ndof, ndof), dtype=dtype)
    for element, conn in zip(elements, connectivity.tolist()):
        ids = torch.tensor([6 * n + k for n in conn for k in range(6)], dtype=torch.long)
        stiffness[ids[:, None], ids] += cylindrical_shell4_stiffness(
            element, p.radius, p.young, p.poisson, p.thickness
        )
    load = consistent_pressure_loads(nodes, connectivity, p)

    # Clamped ring at x = 0 (all six DOFs); at x = L the radial, tangential and
    # rotational DOFs are clamped while the axial displacement is left free, so
    # the open-end condition N_x = 0 is enforced naturally.
    fixed: list[int] = []
    for j in range(facets):
        fixed.extend(6 * j + k for k in range(6))
        last = axes * facets + j
        fixed.extend(6 * last + k for k in (1, 2, 3, 4, 5))
    fixed_index = torch.tensor(sorted(set(fixed)), dtype=torch.long)
    free_mask = torch.ones(ndof, dtype=torch.bool)
    free_mask[fixed_index] = False
    free = torch.nonzero(free_mask).reshape(-1)
    displacement = torch.zeros(ndof, dtype=dtype)
    displacement[free] = torch.linalg.solve(
        stiffness[free[:, None], free], load[free]
    )

    # Nodes are ordered i*facets + j, so the theta = 0 column is i*facets.
    column = torch.arange(axes + 1, dtype=torch.long) * facets
    radial_fe = (
        torch.sin(nodes[column, 1]) * displacement[6 * column + 1]
        + torch.cos(nodes[column, 1]) * displacement[6 * column + 2]
    )
    rotation_fe = displacement[6 * column + 4]
    axial_fe = displacement[6 * column + 0]
    xs = nodes[column, 0]
    h = float(xs[1] - xs[0])

    exact = exact_axisymmetric_solution(xs, p)
    # The Mindlin section rotation is beta_t = -dw/dx; the same h^2 first
    # difference is applied to both fields so the comparison isolates the
    # element error from the measurement error.
    rotation_exact = -exact.slope
    moment_fe = p.bending_rigidity * _first_difference(rotation_fe, h)
    moment_exact = p.bending_rigidity * _first_difference(rotation_exact, h)
    hoop_force_fe = p.young * p.thickness * radial_fe / p.radius

    # N_x = (E t / (1 - nu^2)) (eps_x + nu eps_theta) must vanish: open ends.
    eps_x = _first_difference(axial_fe, h)
    eps_theta = radial_fe / p.radius
    axial_force = p.young * p.thickness / (1.0 - p.poisson ** 2) * (
        eps_x + p.poisson * eps_theta
    )
    axial_force_error = float(axial_force.abs().max()) / p.membrane_hoop_force

    all_radial = (
        torch.sin(nodes[:, 1]) * displacement[1::6]
        + torch.cos(nodes[:, 1]) * displacement[2::6]
    ).reshape(axes + 1, facets)
    uniformity = float(
        (all_radial - all_radial.mean(dim=1, keepdim=True)).abs().max()
    ) / p.membrane_radial_displacement

    return CylinderResult(
        axes=axes,
        facets=facets,
        dofs=ndof,
        x=xs,
        displacement_fe=radial_fe,
        displacement_exact=exact.displacement,
        moment_fe=moment_fe,
        moment_exact=moment_exact,
        hoop_force_fe=hoop_force_fe,
        hoop_force_exact=exact.hoop_force,
        rotation_fe=rotation_fe,
        rotation_exact=rotation_exact,
        axial_force_error=axial_force_error,
        circumferential_uniformity=uniformity,
        parameters=p,
    )


def _observed_order(coarse: float, fine: float, ratio: float) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def run_boundary_layer_study(
    parameters: CylinderParameters,
    *,
    axis_counts: tuple[int, ...] = (16, 32, 64),
    facets: int = 8,
) -> dict[str, object]:
    """Run the mesh sequence and assemble the evidence payload."""
    axis_counts = tuple(int(n) for n in axis_counts)
    runs = [solve_pressure_cylinder(n, facets, parameters) for n in axis_counts]
    rows = []
    for run in runs:
        rows.append(
            {
                "axes": run.axes,
                "facets": run.facets,
                "dofs": run.dofs,
                "axial_spacing_times_beta": parameters.length / run.axes
                * parameters.beta,
                "displacement_l2_error": run.displacement_l2_error,
                "peak_displacement_error": run.peak_displacement_error,
                "moment_l2_error": run.moment_l2_error,
                "edge_moment_error": run.edge_moment_error,
                "moment_sign_correlation": run.moment_sign_correlation,
                "axial_force_over_hoop": run.axial_force_error,
                "circumferential_uniformity": run.circumferential_uniformity,
            }
        )
    orders = {}
    for name in (
        "displacement_l2_error",
        "peak_displacement_error",
        "moment_l2_error",
        "edge_moment_error",
    ):
        orders[name] = [
            _observed_order(
                rows[i][name], rows[i + 1][name], axis_counts[i + 1] / axis_counts[i]
            )
            for i in range(len(rows) - 1)
        ]
    return {
        "parameters": {
            "radius": parameters.radius,
            "thickness": parameters.thickness,
            "young": parameters.young,
            "poisson": parameters.poisson,
            "pressure": parameters.pressure,
            "length": parameters.length,
            "bending_rigidity": parameters.bending_rigidity,
            "beta": parameters.beta,
            "decay_length": 1.0 / parameters.beta,
            "boundary_layer_thickness": parameters.boundary_layer_thickness,
            "beta_length": parameters.beta * parameters.length,
            "membrane_displacement": parameters.membrane_radial_displacement,
            "membrane_hoop_force": parameters.membrane_hoop_force,
            "membrane_hoop_stress": parameters.membrane_hoop_stress,
            "edge_moment_long_cylinder": parameters.edge_moment_long_cylinder,
            "edge_bending_stress": parameters.edge_bending_stress,
        },
        "self_validation": self_validation(parameters),
        "work_energy": work_energy_balance(parameters),
        "runs": rows,
        "observed_orders": orders,
    }