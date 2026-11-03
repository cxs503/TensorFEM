"""Steady-state thermoelasticity of a thick-walled annulus.

Physics
-------
A hollow cylinder carries a steady radial temperature field

    T(r) = T_i + (T_o - T_i) * ln(r / r_i) / ln(r_o / r_i),        r_i <= r <= r_o

(no internal heat generation, conductivity independent of temperature).
Both cylindrical surfaces are traction free.  The problem is the canonical
thermal-stress verification case: the restraint is purely internal
(compatibility of neighbouring fibres), so the resulting stress field is a
sensitive probe for a finite element's thermal-strain path — a wrong
equivalent thermal load vector, a missing ``-alpha*T`` term or a plane
stress/strain mix-up all show up as an O(1) relative stress error, not as a
small perturbation.

The module provides

``radial_temperature``
    the exact temperature profile (also used to impose nodal temperatures);
``annulus_thermal_state``
    the **closed-form** displacement and stress field, derived here from
    axisymmetric equilibrium in the displacement form
    ``(lambda + 2 mu) d/dr[(1/r) d(r u)/dr] = (3 lambda + 2 mu) alpha dT/dr``
    with the particular integral
    ``u_p = (1 + nu) alpha r / (2 (1 - nu)) * (T - K/2)``,
    ``K = (T_o - T_i) / ln(r_o / r_i)``;
    the two homogeneous constants follow from the traction-free conditions
    ``sigma_r(r_i) = sigma_r(r_o) = 0``, which are solved here in closed form;
``thermal_strain_loads``
    the consistent equivalent nodal load vector
    ``f_e = int B^T D eps_th t dA`` with ``eps_th = alpha (T - T_ref) [1, 1, 0]``;
``solve_annulus_thermoelasticity``
    the full Q4 finite element path (assembly, symmetry constraints, solve,
    Gauss-point stress recovery with the thermal strain subtracted) used to
    score the discrete solution against the closed form.

Conventions match :mod:`tensorfem.continuum`: engineering shear strain, stress
vector ``[sigma_xx, sigma_yy, tau_xy]``, ``plane="strain"`` reproduces the
``eps_z = 0`` assumption of the closed form (generalised plane strain with
``sigma_z = lambda (eps_r + eps_theta) - (3 lambda + 2 mu) alpha T``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .continuum import elasticity_matrix, q4_stiffness
from .lame_reference import cartesian_to_polar_stress, sector_q4_mesh, symmetry_fixed_dofs

__all__ = [
    "ThermalAnnulusState",
    "ThermoelasticAnnulusResult",
    "annulus_thermal_state",
    "radial_temperature",
    "solve_annulus_thermoelasticity",
    "thermal_strain_loads",
]

_GAUSS = 1.0 / 3.0 ** 0.5


def _check_radii(inner_radius: float, outer_radius: float) -> None:
    if not (0.0 < inner_radius < outer_radius):
        raise ValueError("require 0 < inner_radius < outer_radius")


def effective_thermal_strain(
    temperature: torch.Tensor,
    *,
    expansion_coefficient: float,
    poisson_ratio: float,
    reference_temperature: float = 0.0,
    plane: str = "strain",
) -> torch.Tensor:
    """Thermal strain vector as seen by the plane constitutive matrix.

    Plane stress needs no correction: ``D_ps (eps - alpha*T*[1,1,0])`` reproduces
    ``E/(1+nu^2)...`` (see the module docstring).  Plane strain does: with
    ``eps_z = 0`` the *total* transverse strain vanishes while the thermal part
    does not, so the in-plane stress carries ``- (3 lambda + 2 mu) alpha T``.
    Condensing the 3-D law onto the in-plane strains scales that term by
    ``(1 + nu)``, hence the effective plane-strain thermal strain is
    ``(1 + nu) * alpha * T``.  (Equivalently: plane strain maps onto plane stress
    with ``E/(1 - nu^2), nu/(1 - nu), alpha (1 + nu)``.)  Using ``alpha*T``
    directly under plane strain understates every thermal stress by ``(1 + nu)``
    - the trap this module exists to close.
    """
    factor = expansion_coefficient * (1.0 + poisson_ratio if plane == "strain" else 1.0)
    theta = temperature - reference_temperature
    return (factor * theta)[:, None] * temperature.new_tensor([1.0, 1.0, 0.0])


def radial_temperature(
    radius: torch.Tensor,
    *,
    inner_radius: float,
    outer_radius: float,
    inner_temperature: float,
    outer_temperature: float,
) -> torch.Tensor:
    """Exact steady-state radial temperature of a hollow cylinder."""
    _check_radii(inner_radius, outer_radius)
    radius = torch.as_tensor(radius)
    if torch.any(radius <= 0):
        raise ValueError("radius must be positive")
    span = math.log(outer_radius / inner_radius)
    return inner_temperature + (outer_temperature - inner_temperature) * torch.log(
        radius / inner_radius
    ) / span


@dataclass(frozen=True)
class ThermalAnnulusState:
    """Closed-form thermoelastic field sampled at ``radius``."""

    radius: torch.Tensor
    radial_stress: torch.Tensor
    hoop_stress: torch.Tensor
    axial_stress: torch.Tensor
    radial_displacement: torch.Tensor
    coefficient_scale: torch.Tensor
    coefficient_inverse: torch.Tensor


def _displacement(
    radius: torch.Tensor,
    scale: torch.Tensor,
    inverse: torch.Tensor,
    *,
    inner_radius: float,
    outer_radius: float,
    inner_temperature: float,
    outer_temperature: float,
    expansion_coefficient: float,
    poisson_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(u_r, eps_r, eps_theta)`` for the closed-form displacement field."""
    span = math.log(outer_radius / inner_radius)
    slope = (outer_temperature - inner_temperature) / span
    theta = inner_temperature + slope * torch.log(radius / inner_radius)
    factor = (1.0 + poisson_ratio) * expansion_coefficient / (1.0 - poisson_ratio)
    u_r = scale * radius + inverse / radius + factor * radius / 2.0 * (theta - slope / 2.0)
    d_theta = slope / radius
    du_dr = (
        scale
        - inverse / radius**2
        + factor / 2.0 * (theta - slope / 2.0)
        + factor * radius / 2.0 * d_theta
    )
    return u_r, du_dr, u_r / radius


def annulus_thermal_state(
    radius: torch.Tensor,
    *,
    inner_radius: float,
    outer_radius: float,
    inner_temperature: float,
    outer_temperature: float,
    young_modulus: float,
    poisson_ratio: float,
    expansion_coefficient: float,
    reference_temperature: float = 0.0,
    plane: str = "strain",
    dtype=torch.float64,
) -> ThermalAnnulusState:
    """Closed-form plane thermoelastic field of a traction-free annulus.

    The two integration constants of the homogeneous solution are eliminated
    with the traction-free boundary conditions, which enter as a 2x2 linear
    system (``sigma_r`` is affine in the constants, so two probes pin it down
    exactly).  The stress vector is obtained by applying the *same*
    constitutive matrix that the finite element path uses,
    ``sigma = D (eps - eps_th)``, so the reference is not an independent
    re-derivation of the constitutive law: it can only differ from the
    discrete solution by discretisation error.
    """
    _check_radii(inner_radius, outer_radius)
    radius = torch.as_tensor(radius, dtype=dtype)
    young = torch.tensor([young_modulus], dtype=dtype)
    poisson = torch.tensor([poisson_ratio], dtype=dtype)
    dmat = elasticity_matrix(young, poisson, plane)[0]
    span = math.log(outer_radius / inner_radius)
    slope = (outer_temperature - inner_temperature) / span

    def radial_stress(scale: float, inverse: float, probe: torch.Tensor):
        def at(point: float) -> torch.Tensor:
            r = torch.tensor([point], dtype=dtype)
            u_r, eps_r, eps_t = _displacement(
                r,
                torch.tensor(scale, dtype=dtype),
                torch.tensor(inverse, dtype=dtype),
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                inner_temperature=inner_temperature,
                outer_temperature=outer_temperature,
                expansion_coefficient=expansion_coefficient,
                poisson_ratio=poisson_ratio,
            )
            theta = inner_temperature + slope * torch.log(r / inner_radius)
            strain = torch.stack((eps_r, eps_t, torch.zeros_like(eps_r)), dim=-1)
            thermal = effective_thermal_strain(
                theta,
                expansion_coefficient=expansion_coefficient,
                poisson_ratio=poisson_ratio,
                reference_temperature=reference_temperature,
                plane=plane,
            )
            stress = (strain - thermal) @ dmat.T
            return stress[0, 0]

        return at(probe)

    # sigma_r is affine in (scale, inverse): evaluate the three linear probes.
    probes = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    rows_inner = [radial_stress(a, b, inner_radius) for a, b in probes]
    rows_outer = [radial_stress(a, b, outer_radius) for a, b in probes]
    matrix = torch.stack(
        (
            torch.stack((rows_inner[1] - rows_inner[0], rows_inner[2] - rows_inner[0])),
            torch.stack((rows_outer[1] - rows_outer[0], rows_outer[2] - rows_outer[0])),
        )
    )
    rhs = -torch.stack((rows_inner[0].reshape(()), rows_outer[0].reshape(())))
    scale, inverse = torch.linalg.solve(matrix, rhs)

    u_r, eps_r, eps_t = _displacement(
        radius,
        scale,
        inverse,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        inner_temperature=inner_temperature,
        outer_temperature=outer_temperature,
        expansion_coefficient=expansion_coefficient,
        poisson_ratio=poisson_ratio,
    )
    theta = inner_temperature + slope * torch.log(radius / inner_radius)
    strain = torch.stack((eps_r, eps_t, torch.zeros_like(eps_r)), dim=-1)
    thermal = effective_thermal_strain(
        theta,
        expansion_coefficient=expansion_coefficient,
        poisson_ratio=poisson_ratio,
        reference_temperature=reference_temperature,
        plane=plane,
    )
    stress = (strain - thermal) @ dmat.T
    lam = young[0] * poisson[0] / ((1.0 + poisson[0]) * (1.0 - 2.0 * poisson[0]))
    bulk = young[0] / (1.0 - 2.0 * poisson[0])
    axial = lam * (eps_r + eps_t) - bulk * expansion_coefficient * (theta - reference_temperature)
    if plane == "stress":
        axial = torch.zeros_like(axial)
    return ThermalAnnulusState(
        radius=radius,
        radial_stress=stress[:, 0],
        hoop_stress=stress[:, 1],
        axial_stress=axial,
        radial_displacement=u_r,
        coefficient_scale=scale.reshape(()),
        coefficient_inverse=inverse.reshape(()),
    )


def _shape_derivatives(xy: torch.Tensor, xi: float, eta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape functions and physical derivatives of a Q4 element at ``(xi, eta)``."""
    natural = xy.new_tensor(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [(1.0 - eta), -(1.0 + xi)],
            [(1.0 + eta), (1.0 + xi)],
            [-(1.0 + eta), (1.0 - xi)],
        ]
    ) / 4.0
    jac = natural.T @ xy
    det = torch.linalg.det(jac)
    if float(det) <= 0.0:
        raise ValueError("Q4 element has non-positive Jacobian")
    # dN/dx_b = sum_a dN/dxi_a * dxi_a/dx_b, with dxi/dx = (dx/dxi)^{-T}
    derivatives = natural @ torch.linalg.inv(jac).T
    shapes = 0.25 * xy.new_tensor(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    return shapes, derivatives, det


def _b_matrix(derivatives: torch.Tensor) -> torch.Tensor:
    b = derivatives.new_zeros((3, 8))
    b[0, 0::2] = derivatives[:, 0]
    b[1, 1::2] = derivatives[:, 1]
    b[2, 0::2] = derivatives[:, 1]
    b[2, 1::2] = derivatives[:, 0]
    return b


def thermal_strain_loads(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    temperature: torch.Tensor,
    *,
    young_modulus,
    poisson_ratio,
    expansion_coefficient,
    thickness=1.0,
    reference_temperature=0.0,
    plane: str = "strain",
    dtype=torch.float64,
) -> torch.Tensor:
    """Consistent equivalent nodal loads ``f = int B^T D eps_th t dA``.

    ``temperature`` holds the nodal values; inside an element the field is
    interpolated with the bilinear Q4 shape functions, i.e. exactly the field
    the element uses when it is scored.  Uniform ``T - T_ref`` must produce a
    stress-free uniform expansion, which the accompanying test suite asserts at
    machine precision.
    """
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]
    young = torch.as_tensor(young_modulus, dtype=dtype).reshape(-1)
    young = young.expand(n_elem) if young.numel() == 1 else young
    poisson = torch.as_tensor(poisson_ratio, dtype=dtype).reshape(-1)
    poisson = poisson.expand(n_elem) if poisson.numel() == 1 else poisson
    thick = torch.as_tensor(thickness, dtype=dtype).reshape(-1)
    thick = thick.expand(n_elem) if thick.numel() == 1 else thick
    dmat = elasticity_matrix(young, poisson, plane)

    loads = torch.zeros(2 * n_nodes, dtype=dtype)
    for index in range(n_elem):
        connectivity = elements[index]
        coords = nodes[connectivity]
        nodal_theta = temperature[connectivity] - reference_temperature
        edofs = torch.stack(tuple(2 * connectivity + i for i in (0, 1)), dim=1).reshape(-1)
        element_load = torch.zeros(8, dtype=dtype)
        for xi, eta in ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS)):
            shapes, derivatives, det = _shape_derivatives(coords, xi, eta)
            b = _b_matrix(derivatives)
            theta_gp = shapes @ nodal_theta
            thermal = effective_thermal_strain(
                theta_gp.reshape(1),
                expansion_coefficient=expansion_coefficient,
                poisson_ratio=float(poisson[index]),
                reference_temperature=0.0,
                plane=plane,
            ).reshape(3)
            element_load = element_load + b.T @ (dmat[index] @ thermal) * (det * thick[index])
        loads = loads.index_put((edofs,), element_load, accumulate=True)
    return loads


@dataclass(frozen=True)
class ThermoelasticAnnulusResult:
    """Finite element thermoelastic annulus solution."""

    nodes: torch.Tensor
    elements: torch.Tensor
    temperature: torch.Tensor
    displacement: torch.Tensor
    sample_radius: torch.Tensor
    radial_stress: torch.Tensor
    hoop_stress: torch.Tensor
    axial_stress: torch.Tensor
    rigid_body_modes: int


def solve_annulus_thermoelasticity(
    inner_radius: float,
    outer_radius: float,
    n_theta: int,
    n_radial: int,
    *,
    young_modulus: float,
    poisson_ratio: float,
    expansion_coefficient: float,
    inner_temperature: float,
    outer_temperature: float,
    reference_temperature: float = 0.0,
    thickness: float = 1.0,
    radial_grading: float = 1.0,
    plane: str = "strain",
    include_thermal_strain: bool = True,
    dtype=torch.float64,
) -> ThermoelasticAnnulusResult:
    """Solve the traction-free annulus with a steady radial temperature field.

    Quarter-sector Q4 mesh with axis-aligned symmetry constraints
    (:func:`tensorfem.lame_reference.symmetry_fixed_dofs`); dense solve;
    stresses recovered at the 2x2 Gauss points of every element with the
    thermal strain subtracted.  ``include_thermal_strain=False`` is an audit
    hook: it drops ``eps_th`` from the recovered stress only and is used by the
    tests to prove that the comparison gate can actually fail.
    """
    nodes, elements = sector_q4_mesh(
        inner_radius, outer_radius, n_theta, n_radial, radial_grading=radial_grading, dtype=dtype
    )
    radius = torch.linalg.vector_norm(nodes, dim=1)
    temperature = radial_temperature(
        radius,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        inner_temperature=inner_temperature,
        outer_temperature=outer_temperature,
    )
    n_elem = elements.shape[0]
    young = torch.full((n_elem,), float(young_modulus), dtype=dtype)
    poisson = torch.full((n_elem,), float(poisson_ratio), dtype=dtype)
    thick = torch.full((n_elem,), float(thickness), dtype=dtype)
    dmat = elasticity_matrix(young, poisson, plane)
    coords = nodes[elements]
    element_stiffness, _ = q4_stiffness(coords, dmat, thick)

    forces = thermal_strain_loads(
        nodes,
        elements,
        temperature,
        young_modulus=young,
        poisson_ratio=poisson,
        expansion_coefficient=expansion_coefficient,
        thickness=thick,
        reference_temperature=reference_temperature,
        plane=plane,
        dtype=dtype,
    )

    n_dofs = 2 * nodes.shape[0]
    edofs = torch.stack(tuple(2 * elements + i for i in (0, 1)), dim=2).reshape(n_elem, 8)
    stiffness = torch.zeros((n_dofs, n_dofs), dtype=dtype)
    rows = edofs[:, :, None].expand(-1, -1, 8).reshape(-1)
    cols = edofs[:, None, :].expand(-1, 8, -1).reshape(-1)
    stiffness = stiffness.index_put((rows, cols), element_stiffness.reshape(-1), accumulate=True)
    fixed = symmetry_fixed_dofs(nodes, n_theta=n_theta, n_radial=n_radial)
    free = torch.ones(n_dofs, dtype=torch.bool)
    free[fixed] = False
    free_index = torch.arange(n_dofs)[free]
    displacement = torch.zeros(n_dofs, dtype=dtype)
    displacement = displacement.index_put(
        (free_index,),
        torch.linalg.solve(stiffness[free_index][:, free_index], forces[free_index]),
    )

    sample_radius = []
    radial_stress = []
    hoop_stress = []
    axial_stress = []
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    bulk = young / (1.0 - 2.0 * poisson)
    for index in range(n_elem):
        connectivity = elements[index]
        element_coords = nodes[connectivity]
        nodal_theta = temperature[connectivity] - reference_temperature
        element_u = displacement[edofs[index]]
        for xi, eta in ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS)):
            shapes, derivatives, _ = _shape_derivatives(element_coords, xi, eta)
            strain = _b_matrix(derivatives) @ element_u
            theta_gp = shapes @ nodal_theta
            thermal = effective_thermal_strain(
                theta_gp.reshape(1),
                expansion_coefficient=expansion_coefficient,
                poisson_ratio=float(poisson[index]),
                reference_temperature=0.0,
                plane=plane,
            ).reshape(3)
            mechanical = strain - thermal if include_thermal_strain else strain
            stress = dmat[index] @ mechanical
            point = (shapes @ element_coords).reshape(1, 2)
            rad, hoop, _ = cartesian_to_polar_stress(stress.reshape(1, 3), point)
            axial = lam[index] * (strain[0] + strain[1]) - bulk[index] * expansion_coefficient * theta_gp
            sample_radius.append(torch.linalg.vector_norm(point).reshape(()))
            radial_stress.append(rad.reshape(()))
            hoop_stress.append(hoop.reshape(()))
            axial_stress.append(
                torch.zeros((), dtype=dtype) if plane == "stress" else axial.reshape(())
            )

    return ThermoelasticAnnulusResult(
        nodes=nodes,
        elements=elements,
        temperature=temperature,
        displacement=displacement,
        sample_radius=torch.stack(sample_radius),
        radial_stress=torch.stack(radial_stress),
        hoop_stress=torch.stack(hoop_stress),
        axial_stress=torch.stack(axial_stress),
        rigid_body_modes=3,
    )