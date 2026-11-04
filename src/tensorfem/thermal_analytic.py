"""Independent 2-D analytical verification for the Q4 heat-conduction kernel.

The production kernel in :mod:`tensorfem.thermal` is exercised by a 1-D rod,
a 1-D convection resistance chain, a linear Q4 temperature patch, a 1-D
transient decay and a 1-D thermoelastic bar.  The linear patch is a *patch
test*: any consistent element reproduces a linear field, so it cannot detect a
wrong map, a wrong quadrature or an over-stiff conduction term.  Nothing in the
suite had ever probed the Q4 conduction path with a field that actually has
curvature in two directions.

This module supplies three closed-form two-dimensional cases and one transient
case, all of them independent of the finite-element discretisation:

``A`` harmonic field on the unit square
    ``T = sin(pi x) sinh(pi y) / sinh(pi)`` satisfies ``Laplacian(T) = 0``
    exactly because the two second derivatives cancel.  Dirichlet data are
    applied from the exact field on all four edges, so the discrete solution
    must converge to it at the element order.

``B`` the same field on a sheared (non-orthogonal) quad mesh
    A pure mapping test: if the isoparametric Jacobian is wrong, the error
    does not converge on a skewed mesh even though it may on a Cartesian one.

``C`` annular conduction ``T = T1 + (T2-T1) ln(r/r1) / ln(r2/r1)``
    The logarithmic radial field is harmonic, so it is the exact solution of
    radial flow through an annulus.  The mesh is a genuinely curvilinear quad
    ring, which the linear patch test never touches.  The polygonal idealisation
    of the circular boundaries is *not* exact, so the case is judged by mesh
    refinement and observed order rather than by a single number.

``D`` two-dimensional transient decay
    ``T(x,y,t) = sin(pi x) sin(pi y) exp(-2 pi^2 alpha t)`` is the exact
    solution of the initial/boundary value problem with homogeneous Dirichlet
    data.  Both the decay constant and the field shape are compared.

``E`` element-centre flux recovery
    ``q = -k grad(T)`` at the element centres is compared with the analytic
    gradient, integrated in an area-weighted norm.  A temperature field can be
    accurate while its gradient is not, and heat flux is the engineering
    quantity of interest.

All routines are plain ``torch`` in float64 and reuse the production
``ThermalModel``/``assemble_thermal``/``solve_steady_thermal`` path; nothing
here re-implements the element.

References
----------
* H. S. Carslaw and J. C. Jaeger, *Conduction of Heat in Solids*, 2nd ed.,
  Oxford, 1959 (logarithmic radial solution, 2-D transient eigenfunction).
* O. C. Zienkiewicz and R. L. Taylor, *The Finite Element Method*, vol. 1,
  5th ed., Butterworth-Heinemann, 2000, ch. 3-4 (patch test vs. truncation
  error, isoparametric mapping).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .thermal import (
    ThermalModel,
    assemble_thermal,
    solve_steady_thermal,
    solve_transient_thermal,
)

DEFAULT_DTYPE = torch.float64

# 2x2 Gauss rule on the reference square [-1, 1]^2.
_GAUSS = 3.0 ** -0.5
_GAUSS_POINTS = ((_GAUSS, _GAUSS), (-_GAUSS, _GAUSS),
                 (-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS))


# ---------------------------------------------------------------------------
# analytic fields
# ---------------------------------------------------------------------------
def harmonic_temperature(nodes: torch.Tensor) -> torch.Tensor:
    """``T = sin(pi x) sinh(pi y)/sinh(pi)``; exactly harmonic on the square."""
    x, y = nodes[:, 0], nodes[:, 1]
    return torch.sin(math.pi * x) * torch.sinh(math.pi * y) / math.sinh(math.pi)


def harmonic_gradient(nodes: torch.Tensor) -> torch.Tensor:
    x, y = nodes[:, 0], nodes[:, 1]
    dx = math.pi * torch.cos(math.pi * x) * torch.sinh(math.pi * y) / math.sinh(math.pi)
    dy = math.pi * torch.sin(math.pi * x) * torch.cosh(math.pi * y) / math.sinh(math.pi)
    return torch.stack((dx, dy), dim=1)


def annular_temperature(radius: torch.Tensor, inner: float, outer: float,
                        t_inner: float, t_outer: float) -> torch.Tensor:
    """Logarithmic radial field; exactly harmonic away from the axis."""
    scale = (t_outer - t_inner) / math.log(outer / inner)
    return t_inner + scale * torch.log(radius / inner)


def annular_radial_gradient(radius: torch.Tensor, inner: float, outer: float,
                            t_inner: float, t_outer: float) -> torch.Tensor:
    return ((t_outer - t_inner) / math.log(outer / inner)) / radius


def transient_temperature(nodes: torch.Tensor, time: float, alpha: float) -> torch.Tensor:
    """Exact decaying mode of the unit square with homogeneous Dirichlet data."""
    x, y = nodes[:, 0], nodes[:, 1]
    return (torch.sin(math.pi * x) * torch.sin(math.pi * y)
            * math.exp(-2.0 * math.pi ** 2 * alpha * time))


def transient_decay_rate(alpha: float) -> float:
    """Exact exponential rate of the fundamental mode: ``2 pi^2 alpha``."""
    return 2.0 * math.pi ** 2 * alpha


def laplacian_residual(field, nodes: torch.Tensor, step: float = 1.0e-4) -> float:
    """Central-difference Laplacian of a callable field, scaled by its gradient norm.

    Used as a self-check of the closed forms: an "analytic" reference that is
    not actually harmonic would silently invalidate every error below.
    """
    base = field(nodes)
    lap = torch.zeros_like(base)
    for axis in (0, 1):
        e = torch.zeros_like(nodes)
        e[:, axis] = step
        lap = lap + (field(nodes + e) - 2.0 * base + field(nodes - e)) / step ** 2
    return float(torch.linalg.vector_norm(lap)) / float(torch.linalg.vector_norm(base))


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuadMesh:
    nodes: torch.Tensor
    elements: torch.Tensor

    @property
    def boundary_nodes(self) -> torch.Tensor:
        count = torch.zeros(len(self.nodes), dtype=torch.long, device=self.nodes.device)
        count.index_add_(0, self.elements.reshape(-1),
                         torch.ones(self.elements.numel(), dtype=torch.long,
                                    device=self.nodes.device))
        # A convex quad mesh: interior nodes touch 4 elements, edge nodes 1 or 2.
        return torch.nonzero(count < 4).reshape(-1)


def structured_quad_mesh(nx: int, ny: int, *, lx: float = 1.0, ly: float = 1.0,
                         shear: float = 0.0, dtype=DEFAULT_DTYPE) -> QuadMesh:
    """Structured ``nx`` x ``ny`` quad grid, optionally sheared along x.

    ``shear`` > 0 maps ``x -> x + shear*y`` so every element becomes a
    parallelogram with a non-orthogonal Jacobian; ``shear=0`` is Cartesian.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    xs = torch.linspace(0.0, lx, nx + 1, dtype=dtype)
    ys = torch.linspace(0.0, ly, ny + 1, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    x = (gx + shear * gy).reshape(-1)
    y = gy.reshape(-1)
    nodes = torch.stack((x, y), dim=1)
    ids = torch.arange((nx + 1) * (ny + 1), dtype=torch.long).reshape(ny + 1, nx + 1)
    e0 = ids[:-1, :-1].reshape(-1)
    e1 = ids[:-1, 1:].reshape(-1)
    e2 = ids[1:, 1:].reshape(-1)
    e3 = ids[1:, :-1].reshape(-1)
    return QuadMesh(nodes, torch.stack((e0, e1, e2, e3), dim=1))


def annular_quad_mesh(nr: int, nt: int, *, inner: float = 1.0, outer: float = 2.0,
                      dtype=DEFAULT_DTYPE) -> QuadMesh:
    """Quad ring mesh with radial x circumferential subdivision.

    Nodes sit exactly on the two circles; the element edges are chords, so the
    discretised domain is an inscribed polygon and convergence must be measured
    by refinement.
    """
    if nr < 1 or nt < 4:
        raise ValueError("need at least one radial and four circumferential cells")
    radii = torch.linspace(inner, outer, nr + 1, dtype=dtype)
    angles = torch.arange(nt, dtype=dtype) * (2.0 * math.pi / nt)
    r, a = torch.meshgrid(radii, angles, indexing="ij")
    nodes = torch.stack((r.reshape(-1) * torch.cos(a.reshape(-1)),
                         r.reshape(-1) * torch.sin(a.reshape(-1))), dim=1)
    ids = torch.arange((nr + 1) * nt, dtype=torch.long).reshape(nr + 1, nt)
    j = torch.arange(nt, dtype=torch.long)
    jp = (j + 1) % nt
    # Counter-clockwise winding: inner j -> outer j -> outer j+1 -> inner j+1.
    e0 = ids[:-1, :].reshape(-1)
    e1 = ids[1:, :].reshape(-1)
    e2 = ids[1:, jp].reshape(-1)
    e3 = ids[:-1, jp].reshape(-1)
    return QuadMesh(nodes, torch.stack((e0, e1, e2, e3), dim=1))


# ---------------------------------------------------------------------------
# shape functions and error metrics
# ---------------------------------------------------------------------------
def _q4_shape(xi: float, eta: float, dtype=DEFAULT_DTYPE) -> torch.Tensor:
    return torch.tensor([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)],
                        dtype=dtype) / 4.0


def _q4_dshape(xi: float, eta: float, dtype=DEFAULT_DTYPE) -> torch.Tensor:
    return torch.tensor([[-(1 - eta), -(1 - xi)],
                         [(1 - eta), -(1 + xi)],
                         [(1 + eta), (1 + xi)],
                         [-(1 + eta), (1 - xi)]], dtype=dtype) / 4.0


def _element_jacobians(coords: torch.Tensor, dn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    jac = torch.einsum("eia,ib->eab", coords, dn)
    return jac, torch.linalg.det(jac)


def field_errors(mesh: QuadMesh, temperature: torch.Tensor, exact) -> dict:
    """Quadrature-weighted L2 and nodal pointwise errors of a scalar field."""
    nodes, conn = mesh.nodes, mesh.elements.long()
    coords = nodes[conn]
    te = temperature[conn]
    num = 0.0
    ref = 0.0
    for xi, eta in _GAUSS_POINTS:
        n = _q4_shape(xi, eta, dtype=nodes.dtype)
        dn = _q4_dshape(xi, eta, dtype=nodes.dtype)
        jac, det = _element_jacobians(coords, dn)
        if bool(torch.any(det <= 0)):
            raise ValueError("element with non-positive Jacobian in error quadrature")
        weight = det
        xq = torch.einsum("i,eia->ea", n, coords)
        tq = torch.einsum("i,ei->e", n, te)
        eq = exact(xq)
        num += float(torch.sum((tq - eq) ** 2 * weight))
        ref += float(torch.sum(eq ** 2 * weight))
    exact_nodes = exact(nodes)
    pointwise = torch.abs(temperature - exact_nodes) / torch.max(torch.abs(exact_nodes))
    return {
        "l2_relative": math.sqrt(num / ref),
        "pointwise_max_relative": float(torch.max(pointwise)),
        "n_dof": int(len(nodes)),
        "n_elements": int(len(conn)),
    }


def element_centre_gradients(mesh: QuadMesh, temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodal-coordinate centres and ``grad(T)`` at the element centre."""
    nodes, conn = mesh.nodes, mesh.elements.long()
    coords = nodes[conn]
    dn = _q4_dshape(0.0, 0.0, dtype=nodes.dtype)
    jac, _ = _element_jacobians(coords, dn)
    grad = torch.einsum("ib,ebc->eic", dn, torch.linalg.inv(jac))
    grad_t = torch.einsum("eic,ei->ec", grad, temperature[conn])
    return coords.mean(dim=1), grad_t


def element_areas(mesh: QuadMesh) -> torch.Tensor:
    """Exact polygon area of every quad (shoelace over the four vertices)."""
    p = mesh.nodes[mesh.elements.long()]
    x, y = p[..., 0], p[..., 1]
    return 0.5 * torch.sum(x * y.roll(-1, dims=1) - x.roll(-1, dims=1) * y, dim=1)


def flux_error(mesh: QuadMesh, temperature: torch.Tensor, exact_gradient) -> dict:
    """Area-weighted relative L2 error of the recovered gradient."""
    centres, grad = element_centre_gradients(mesh, temperature)
    area = element_areas(mesh)
    ref = exact_gradient(centres)
    num = float(torch.sum(torch.sum((grad - ref) ** 2, dim=1) * area))
    den = float(torch.sum(torch.sum(ref ** 2, dim=1) * area))
    return {"gradient_l2_relative": math.sqrt(num / den), "n_samples": int(len(centres))}


# ---------------------------------------------------------------------------
# production-path drivers
# ---------------------------------------------------------------------------
def _steady_model(mesh: QuadMesh, *, conductivity: float, dtype=DEFAULT_DTYPE) -> ThermalModel:
    boundary = mesh.boundary_nodes
    return ThermalModel(
        nodes=mesh.nodes,
        elements=mesh.elements,
        conductivity=torch.tensor(conductivity, dtype=dtype),
        density=torch.tensor(1.0, dtype=dtype),
        specific_heat=torch.tensor(1.0, dtype=dtype),
        thickness=torch.tensor(1.0, dtype=dtype),
        fixed_nodes=boundary,
        fixed_temperatures=torch.zeros(len(boundary), dtype=dtype),
        element_type="q4",
    )


def solve_dirichlet(mesh: QuadMesh, exact, *, conductivity: float = 1.0) -> torch.Tensor:
    """Solve the Laplace problem with Dirichlet data from the exact field."""
    model = _steady_model(mesh, conductivity=conductivity)
    values = exact(mesh.nodes[model.fixed_nodes.long()]).to(dtype=mesh.nodes.dtype)
    model = ThermalModel(**{**model.__dict__, "fixed_temperatures": values})
    return solve_steady_thermal(model)


def heat_balance_residual(mesh: QuadMesh, exact, temperature: torch.Tensor, *,
                          conductivity: float = 1.0) -> float:
    """``|sum(K T - f)| / max|K T - f|``; zero for a pure Dirichlet problem."""
    model = _steady_model(mesh, conductivity=conductivity)
    values = exact(mesh.nodes[model.fixed_nodes.long()]).to(dtype=mesh.nodes.dtype)
    model = ThermalModel(**{**model.__dict__, "fixed_temperatures": values})
    reaction = assemble_thermal(model).conductivity @ temperature
    return float(torch.abs(torch.sum(reaction))) / float(torch.max(torch.abs(reaction)))


def observed_order(coarse: float, fine: float, ratio: float) -> float:
    """Richardson estimate of the order from two grid errors."""
    if coarse <= 0.0 or fine <= 0.0:
        raise ValueError("errors must be positive to estimate an order")
    return math.log(coarse / fine) / math.log(ratio)


@dataclass(frozen=True)
class ConvergenceCase:
    name: str
    levels: list
    l2: list
    pointwise: list
    orders: list
    dofs: list

    def as_dict(self) -> dict:
        return {"name": self.name, "levels": self.levels, "n_dof": self.dofs,
                "l2_relative": self.l2, "pointwise_max_relative": self.pointwise,
                "observed_order": self.orders}


def harmonic_convergence(sizes: list[int], *, shear: float = 0.0,
                         conductivity: float = 1.0) -> ConvergenceCase:
    """Mesh study of case A/B on ``n x n`` quad grids."""
    levels, l2, point, orders, dofs = [], [], [], [], []
    for n in sizes:
        mesh = structured_quad_mesh(n, n, shear=shear)
        temperature = solve_dirichlet(mesh, harmonic_temperature,
                                      conductivity=conductivity)
        err = field_errors(mesh, temperature, harmonic_temperature)
        levels.append(n)
        l2.append(err["l2_relative"])
        point.append(err["pointwise_max_relative"])
        dofs.append(err["n_dof"])
    for i in range(1, len(sizes)):
        orders.append(observed_order(l2[i - 1], l2[i], sizes[i] / sizes[i - 1]))
    return ConvergenceCase("harmonic_sheared" if shear else "harmonic_cartesian",
                           levels, l2, point, orders, dofs)


def annular_convergence(circumferential: list[int], *, radial_per_8: int = 1,
                        inner: float = 1.0, outer: float = 2.0,
                        t_inner: float = 100.0, t_outer: float = 20.0) -> ConvergenceCase:
    """Mesh study of case C; ``nr = max(1, nt // 8 * radial_per_8)``."""
    def exact(pts):
        r = torch.linalg.vector_norm(pts, dim=1)
        return annular_temperature(r, inner, outer, t_inner, t_outer)

    levels, l2, point, orders, dofs = [], [], [], [], []
    for nt in circumferential:
        nr = max(1, (nt // 8) * radial_per_8)
        mesh = annular_quad_mesh(nr, nt, inner=inner, outer=outer)
        temperature = solve_dirichlet(mesh, exact)
        err = field_errors(mesh, temperature, exact)
        levels.append(nt)
        l2.append(err["l2_relative"])
        point.append(err["pointwise_max_relative"])
        dofs.append(err["n_dof"])
    for i in range(1, len(circumferential)):
        orders.append(observed_order(l2[i - 1], l2[i],
                                     circumferential[i] / circumferential[i - 1]))
    return ConvergenceCase("annular_logarithmic", levels, l2, point, orders, dofs)


def transient_decay(n: int, *, alpha: float = 1.0, steps_per_rate: int = 100,
                    final_time_factor: float = 1.0) -> dict:
    """Case D: compare the discrete mode against ``exp(-2 pi^2 alpha t)``."""
    mesh = structured_quad_mesh(n, n)
    model = _steady_model(mesh, conductivity=alpha)  # rho*cp = 1 -> alpha = k
    initial = transient_temperature(mesh.nodes, 0.0, alpha)
    rate = transient_decay_rate(alpha)
    tau = 1.0 / rate
    dt = tau / steps_per_rate
    time = torch.arange(0.0, final_time_factor * tau + 0.5 * dt, dt,
                        dtype=mesh.nodes.dtype)
    result = solve_transient_thermal(model, time, initial, theta=0.5)

    amplitude = torch.max(torch.abs(result.temperature), dim=1).values
    design = torch.stack((time, torch.ones_like(time)), dim=1)
    slope = torch.linalg.lstsq(design, torch.log(amplitude)).solution[0]

    checkpoints = []
    for fraction in (0.25, 0.5, 1.0):
        t = fraction * tau
        index = int(round(float(t / dt)))
        exact = transient_temperature(mesh.nodes, float(time[index]), alpha)
        num = float(torch.linalg.vector_norm(result.temperature[index] - exact))
        den = float(torch.linalg.vector_norm(exact))
        reference_amplitude = float(torch.max(torch.abs(exact)))
        checkpoints.append({
            "t_over_tau": fraction,
            "field_l2_relative": num / den,
            "amplitude_relative": abs(float(amplitude[index]) - reference_amplitude)
            / reference_amplitude,
        })
    return {
        "mesh": [n, n],
        "n_dof": int(len(mesh.nodes)),
        "alpha": alpha,
        "theta": 0.5,
        "steps_per_rate": steps_per_rate,
        "exact_rate": rate,
        "fitted_rate": float(slope),
        "rate_relative_error": abs(float(slope) + rate) / rate,
        "checkpoints": checkpoints,
    }