"""Centrifugal (rotational) body loading and the classical rotating-disk solution.

Until this module existed TensorFEM had no rotational body-load path at all: the
2-D continuum solver :mod:`tensorfem.continuum` only accepted *nodal* forces, so a
user who wanted a spinning component had to hand-build the load vector with no
verification anchor.  This module supplies

* ``annular_disk_mesh`` -- a polar Q4 mesh of an annulus (full ring or quarter);
* ``centrifugal_consistent_loads`` -- the consistent nodal load vector of the
  centrifugal body force ``rho * omega**2 * x`` for Q4/CST;
* ``rotating_disk_stress`` / ``rotating_disk_displacement`` -- the closed-form
  axisymmetric solution used as the verification oracle;
* ``solve_rotating_disk`` -- the assembled benchmark that returns the measured
  errors against that oracle.

Reference solution
------------------
S. Timoshenko & J. N. Goodier, *Theory of Elasticity*, 3rd ed., §"Rotating
disks"; S. Timoshenko, *Strength of Materials*, Part II.  With
``C = (3 + nu) / 8 * rho * omega**2`` and ``k = (1 + 3 nu) / (3 + nu)``
(plane stress) the in-plane stresses of an annulus ``a <= r <= b`` are

    sigma_r(r) = C * (b^2 + a^2 - a^2 b^2 / r^2 - r^2)
    sigma_theta(r) = C * (b^2 + a^2 + a^2 b^2 / r^2 - k * r^2)

and the radial displacement follows from the plane-stress law
``u(r) = r * (sigma_theta - nu * sigma_r) / E``.  For the long cylinder
(``plane="strain"``, axial strain suppressed) the same structure holds with
``C = (3 - 2 nu) / (8 (1 - nu)) * rho * omega**2`` and
``k = (1 + 2 nu) / (3 - 2 nu)``; the displacement then picks up the suppressed
axial stress ``sigma_z = nu * (sigma_r + sigma_theta)``.

The oracle is not taken on faith.  ``tests/test_rotating_disk.py`` re-derives it
from three independent statements that pin the solution uniquely:

1. radial equilibrium ``d(sigma_r)/dr + (sigma_r - sigma_theta)/r + rho omega^2 r = 0``
   checked by finite differences;
2. strain compatibility ``d(r eps_theta)/dr = eps_r`` checked numerically;
3. traction-free rims ``sigma_r(a) = sigma_r(b) = 0``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "RotatingDiskReference",
    "annular_disk_mesh",
    "centrifugal_consistent_loads",
    "centrifugal_lumped_loads",
    "radial_hoop_stress",
    "rotating_disk_stress",
    "rotating_disk_displacement",
    "solve_rotating_disk",
]

_G = 1.0 / math.sqrt(3.0)
_GAUSS = ((_G, _G), (-_G, _G), (-_G, -_G), (_G, -_G))


# ---------------------------------------------------------------------------
# Reference solution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RotatingDiskReference:
    """Closed-form rotating annulus (Timoshenko & Goodier)."""

    inner_radius: float
    outer_radius: float
    density: float
    omega: float
    young_modulus: float
    poisson_ratio: float
    plane: str = "stress"

    def __post_init__(self) -> None:
        if self.plane not in ("stress", "strain"):
            raise ValueError("plane must be 'stress' or 'strain'")
        if not 0.0 <= self.inner_radius < self.outer_radius:
            raise ValueError("need 0 <= inner_radius < outer_radius")
        if self.density <= 0.0 or self.omega < 0.0:
            raise ValueError("density must be positive and omega non-negative")

    @property
    def constant(self) -> float:
        """Leading factor ``C`` of the two stress components."""
        nu = self.poisson_ratio
        base = self.density * self.omega**2
        if self.plane == "stress":
            return (3.0 + nu) / 8.0 * base
        return (3.0 - 2.0 * nu) / (8.0 * (1.0 - nu)) * base

    @property
    def radial_quadratic(self) -> float:
        """Coefficient ``k`` multiplying ``r^2`` inside ``sigma_theta``."""
        nu = self.poisson_ratio
        if self.plane == "stress":
            return (1.0 + 3.0 * nu) / (3.0 + nu)
        return (1.0 + 2.0 * nu) / (3.0 - 2.0 * nu)

    def stress(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.inner_radius, self.outer_radius
        a2b2 = (a * b) ** 2
        c, k = self.constant, self.radial_quadratic
        sigma_r = c * (b**2 + a**2 - a2b2 / r**2 - r**2)
        sigma_theta = c * (b**2 + a**2 + a2b2 / r**2 - k * r**2)
        return sigma_r, sigma_theta

    def axial_stress(self, r: torch.Tensor) -> torch.Tensor:
        if self.plane == "stress":
            return torch.zeros_like(r)
        sigma_r, sigma_theta = self.stress(r)
        return self.poisson_ratio * (sigma_r + sigma_theta)

    def displacement(self, r: torch.Tensor) -> torch.Tensor:
        sigma_r, sigma_theta = self.stress(r)
        nu, e = self.poisson_ratio, self.young_modulus
        if self.plane == "stress":
            return r * (sigma_theta - nu * sigma_r) / e
        return r * (1.0 + nu) / e * ((1.0 - nu) * sigma_theta - nu * sigma_r)

    def strain(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Radial and hoop strain implied by the closed form."""
        sigma_r, sigma_theta = self.stress(r)
        nu, e = self.poisson_ratio, self.young_modulus
        if self.plane == "stress":
            return (sigma_r - nu * sigma_theta) / e, (sigma_theta - nu * sigma_r) / e
        sigma_z = self.axial_stress(r)
        return (sigma_r - nu * (sigma_theta + sigma_z)) / e, (
            sigma_theta - nu * (sigma_r + sigma_z)
        ) / e

    def equilibrium_residual(self, r: torch.Tensor, epsilon: float | None = None) -> torch.Tensor:
        """Residual of ``d(sigma_r)/dr + (sigma_r - sigma_theta)/r + rho omega^2 r``."""
        h = epsilon if epsilon is not None else 1e-6 * self.outer_radius
        sigma_r_plus, _ = self.stress(r + h)
        sigma_r_minus, _ = self.stress(r - h)
        d_sigma_r = (sigma_r_plus - sigma_r_minus) / (2.0 * h)
        sigma_r, sigma_theta = self.stress(r)
        return d_sigma_r + (sigma_r - sigma_theta) / r + self.density * self.omega**2 * r


def rotating_disk_stress(
    r: torch.Tensor,
    inner_radius: float,
    outer_radius: float,
    density: float,
    omega: float,
    young_modulus: float,
    poisson_ratio: float,
    plane: str = "stress",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper around :class:`RotatingDiskReference`."""
    ref = RotatingDiskReference(
        inner_radius, outer_radius, density, omega, young_modulus, poisson_ratio, plane
    )
    return ref.stress(r)


def rotating_disk_displacement(
    r: torch.Tensor,
    inner_radius: float,
    outer_radius: float,
    density: float,
    omega: float,
    young_modulus: float,
    poisson_ratio: float,
    plane: str = "stress",
) -> torch.Tensor:
    ref = RotatingDiskReference(
        inner_radius, outer_radius, density, omega, young_modulus, poisson_ratio, plane
    )
    return ref.displacement(r)


# ---------------------------------------------------------------------------
# Mesh and body loads
# ---------------------------------------------------------------------------
def annular_disk_mesh(
    inner_radius: float,
    outer_radius: float,
    n_radial: int,
    n_angular: int,
    *,
    quarter: bool = True,
    grading: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Polar Q4 mesh of an annulus.

    ``quarter=True`` meshes ``x >= 0, y >= 0`` only (symmetry planes included),
    which is the configuration where the exact radial solution satisfies the
    symmetry boundary conditions identically.  ``grading`` > 1 clusters radial
    stations towards the inner radius (geometric progression), which is what a
    near-solid disk with a small bore needs.
    """
    if n_radial < 1 or n_angular < 1:
        raise ValueError("need at least one element in each direction")
    if grading <= 0.0:
        raise ValueError("grading must be positive")
    if grading != 1.0:
        if abs(grading - 1.0) < 1e-12:
            fractions = [i / n_radial for i in range(n_radial + 1)]
        else:
            step = (grading - 1.0) / (grading**n_radial - 1.0)
            fractions = [step * (grading**i - 1.0) / (grading - 1.0) for i in range(n_radial + 1)]
            fractions[-1] = 1.0
    else:
        fractions = [i / n_radial for i in range(n_radial + 1)]
    radii = [inner_radius + (outer_radius - inner_radius) * f for f in fractions]
    angular_span = math.pi / 2.0 if quarter else 2.0 * math.pi
    n_angular_nodes = n_angular + 1 if quarter else n_angular
    angles = [angular_span * j / n_angular for j in range(n_angular_nodes)]
    nodes = []
    for r in radii:
        for theta in angles:
            nodes.append((r * math.cos(theta), r * math.sin(theta)))
    nodes = torch.tensor(nodes, dtype=dtype)
    stride = len(angles)
    cells = []
    for i in range(n_radial):
        for j in range(n_angular):
            n0 = i * stride + j
            cells.append((n0, n0 + stride, n0 + stride + 1, n0 + 1))
    return nodes, torch.tensor(cells, dtype=torch.long)


def _q4_shape(xi: float, eta: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    shape = torch.tensor(
        (
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ),
        dtype=dtype,
    ) / 4.0
    derivatives = torch.tensor(
        (
            (-(1 - eta), -(1 - xi)),
            ((1 - eta), -(1 + xi)),
            ((1 + eta), (1 + xi)),
            (-(1 + eta), (1 - xi)),
        ),
        dtype=dtype,
    ) / 4.0
    return shape, derivatives


def _centrifugal_nodal_loads(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    density: float,
    omega: float,
    thickness: float,
    *,
    lumped: bool,
) -> torch.Tensor:
    if elements.shape[1] != 4:
        raise ValueError("centrifugal loads are implemented for Q4 connectivity only")
    coords = nodes[elements.long()]
    ne = coords.shape[0]
    dtype = nodes.dtype
    loads = torch.zeros((ne, 8), dtype=dtype)
    if lumped:
        area = 0.5 * torch.abs(
            (coords[:, 1, 0] - coords[:, 0, 0]) * (coords[:, 2, 1] - coords[:, 0, 1])
            - (coords[:, 2, 0] - coords[:, 0, 0]) * (coords[:, 1, 1] - coords[:, 0, 1])
        )
        mass = density * area * thickness / 4.0
        loads[:, 0::2] = (mass[:, None] * omega**2 * coords[:, :, 0])
        loads[:, 1::2] = (mass[:, None] * omega**2 * coords[:, :, 1])
    else:
        for xi, eta in _GAUSS:
            shape, derivatives = _q4_shape(xi, eta, dtype)
            jac = torch.einsum("eia,ib->eab", coords, derivatives)
            det = torch.linalg.det(jac)
            if torch.any(det <= 0):
                raise ValueError("element with non-positive Jacobian in centrifugal load")
            x_gp = torch.einsum("i,eia->ea", shape, coords)
            weight = (det * thickness)[:, None] * density * omega**2
            loads[:, 0::2] += weight * shape[None, :] * x_gp[:, 0:1]
            loads[:, 1::2] += weight * shape[None, :] * x_gp[:, 1:2]
    forces = torch.zeros(2 * nodes.shape[0], dtype=dtype)
    dofs = torch.stack(tuple(2 * elements + i for i in (0, 1)), 2).reshape(-1)
    return forces.index_put((dofs,), loads.reshape(-1), accumulate=True)


def centrifugal_consistent_loads(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    density: float,
    omega: float,
    thickness: float,
) -> torch.Tensor:
    """Consistent nodal loads of the centrifugal field ``b = rho omega^2 x``.

    The integrand ``N^T b`` is quadratic on a bilinear quad, so the 2x2 Gauss
    rule used here is exact (no quadrature error in the load vector).
    """
    return _centrifugal_nodal_loads(
        nodes, elements, density, omega, thickness, lumped=False
    )


def centrifugal_lumped_loads(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    density: float,
    omega: float,
    thickness: float,
) -> torch.Tensor:
    """Mass-lumped variant, kept only as a deliberately lower-order control."""
    return _centrifugal_nodal_loads(nodes, elements, density, omega, thickness, lumped=True)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def radial_hoop_stress(
    coordinates: torch.Tensor, stress_xy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate Cartesian plane stresses into radial/hoop components."""
    x, y = coordinates[:, 0], coordinates[:, 1]
    r = torch.hypot(x, y)
    theta = torch.atan2(y, x)
    sxx, syy, sxy = stress_xy[:, 0], stress_xy[:, 1], stress_xy[:, 2]
    mean = 0.5 * (sxx + syy)
    half = 0.5 * (sxx - syy)
    cos2, sin2 = torch.cos(2 * theta), torch.sin(2 * theta)
    sigma_r = mean + half * cos2 + sxy * sin2
    sigma_theta = mean - half * cos2 - sxy * sin2
    return sigma_r, sigma_theta, r


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def _symmetry_dofs(
    nodes: torch.Tensor, tolerance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quarter-model symmetry restraints (``u_y = 0`` on y=0, ``u_x = 0`` on x=0)."""
    on_x_axis = torch.where(nodes[:, 1].abs() <= tolerance)[0]
    on_y_axis = torch.where(nodes[:, 0].abs() <= tolerance)[0]
    dofs_y = 2 * on_x_axis + 1
    dofs_x = 2 * on_y_axis
    return dofs_x, dofs_y


def solve_rotating_disk(
    n_radial: int,
    n_angular: int,
    *,
    inner_radius: float = 0.25,
    outer_radius: float = 0.5,
    density: float = 7850.0,
    omega: float = 100.0,
    young_modulus: float = 200e9,
    poisson_ratio: float = 0.3,
    thickness: float = 0.02,
    plane: str = "stress",
    grading: float = 1.0,
    lumped: bool = False,
    omega_scale: float = 1.0,
    reference_omega: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Solve the quarter annulus and measure the error against the closed form.

    ``omega_scale`` and ``lumped`` exist so the negative controls can be driven
    through exactly the same code path as the accepted run.
    """
    from .continuum import ContinuumModel, solve_continuum

    nodes, elements = annular_disk_mesh(
        inner_radius,
        outer_radius,
        n_radial,
        n_angular,
        quarter=True,
        grading=grading,
        dtype=dtype,
    )
    applied_omega = omega * omega_scale
    builder = centrifugal_lumped_loads if lumped else centrifugal_consistent_loads
    forces = builder(nodes, elements, density, applied_omega, thickness)
    dofs_x, dofs_y = _symmetry_dofs(nodes, tolerance=1e-9 * outer_radius)
    fixed = torch.cat((dofs_x, dofs_y))
    model = ContinuumModel(
        nodes,
        elements,
        torch.tensor(young_modulus, dtype=dtype),
        torch.tensor(poisson_ratio, dtype=dtype),
        torch.tensor(thickness, dtype=dtype),
        forces,
        fixed,
        plane=plane,
    )
    result = solve_continuum(model)

    reference = RotatingDiskReference(
        inner_radius,
        outer_radius,
        density,
        applied_omega if reference_omega is None else reference_omega,
        young_modulus,
        poisson_ratio,
        plane,
    )
    # Element-centre stresses (the quadrature point the Q4 solver reports).
    centres = nodes[elements.long()].mean(1)
    sigma_r, sigma_theta, radii = radial_hoop_stress(centres, result.stress)
    exact_r, exact_theta = reference.stress(radii)
    scale = float(exact_theta.abs().max())
    if scale == 0.0:
        scale = 1.0
    def _rel(numer: torch.Tensor, denom_t: torch.Tensor) -> float:
        denom = float(torch.linalg.vector_norm(denom_t))
        return float(torch.linalg.vector_norm(numer)) / denom if denom else 0.0

    stress_error = _rel(
        torch.stack((sigma_r - exact_r, sigma_theta - exact_theta)),
        torch.stack((exact_r, exact_theta)),
    )
    peak_theta_error = float((sigma_theta - exact_theta).abs().max()) / scale
    peak_radius_error = float((sigma_r - exact_r).abs().max()) / scale

    # Displacement: the exact field is purely radial and satisfies the quarter
    # symmetry conditions identically, so a direct comparison is meaningful.
    node_radius = torch.hypot(nodes[:, 0], nodes[:, 1])
    exact_u = reference.displacement(node_radius)
    exact_vector = torch.stack(
        (exact_u * nodes[:, 0] / node_radius, exact_u * nodes[:, 1] / node_radius), 1
    ).reshape(-1)
    disp_error = _rel(result.displacement - exact_vector, exact_vector)
    radial_peak_displacement = float(exact_u.abs().max()) or 1.0
    peak_disp_error = float((result.displacement - exact_vector).abs().max()) / radial_peak_displacement

    # Global equilibrium: the x-reactions on the x=0 symmetry plane must cancel
    # the net centrifugal force of the quarter model.  Compared both with the
    # quadrature value of the applied load and with the analytic integral
    # rho omega^2 t (b^3 - a^3) / 3, which is independent of the mesh.
    x_reaction = float(result.reaction[dofs_x].sum())
    applied_x = float(forces[0::2].sum())
    analytic_x = density * applied_omega**2 * thickness * (outer_radius**3 - inner_radius**3) / 3.0
    equilibrium_residual = abs(x_reaction + applied_x) / abs(applied_x) if applied_x else 0.0
    cut_balance_error = abs(x_reaction + analytic_x) / abs(analytic_x) if analytic_x else 0.0
    # Global energy balance: the work of the centrifugal load must equal the
    # internal strain energy the solver reports (independent of the closed form).
    external_work = 0.5 * float(forces @ result.displacement)
    strain_energy = float(result.strain_energy)
    return {
        "mesh": [n_radial, n_angular],
        "elements": int(elements.shape[0]),
        "nodes": int(nodes.shape[0]),
        "stress_rel_error": stress_error,
        "peak_hoop_error": peak_theta_error,
        "peak_radial_error": peak_radius_error,
        "displacement_rel_error": disp_error,
        "peak_displacement_error": peak_disp_error,
        "equilibrium_residual": equilibrium_residual,
        "cut_balance_error": cut_balance_error,
        "x_reaction": x_reaction,
        "analytic_resultant": analytic_x,
        "max_hoop_stress_fem": float(sigma_theta.max()),
        "max_hoop_stress_exact": float(exact_theta.max()),
        "cell_hoop_stress": [float(v) for v in sigma_theta],
        "cell_hoop_exact": [float(v) for v in exact_theta],
        "cell_radial_stress": [float(v) for v in sigma_r],
        "cell_radial_exact": [float(v) for v in exact_r],
        "cell_radius": [float(v) for v in radii],
        "displacement": [float(v) for v in result.displacement],
        "nodal_radius": [float(v) for v in node_radius],
        "external_work": external_work,
        "energy_balance_error": abs(external_work - strain_energy) / abs(strain_energy) if strain_energy else 0.0,
        "load_resultant": {
            "fem": [
                float(forces[0::2].sum()),
                float(forces[1::2].sum()),
            ],
            "exact": [
                density * applied_omega**2 * thickness * (outer_radius**3 - inner_radius**3) / 3.0
            ] * 2,
        },
        "strain_energy": strain_energy,
    }