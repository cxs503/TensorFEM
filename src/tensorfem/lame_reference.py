"""Analytic Lame solution and a matching polar Q4 mesh for a thick-walled cylinder.

The Lame problem (thick cylinder under internal/external pressure) is *statically
determinate* in the annulus, so the in-plane stress field is identical for plane
stress and plane strain; only the radial displacement differs, because the
out-of-plane constraint changes the compliance.  That makes the case a clean
discriminator between the two constitutive paths of
:mod:`tensorfem.continuum`, which is exactly what the verification study needs.

Conventions
-----------
* Tension positive, ``r`` the radial coordinate, ``theta`` the hoop direction.
* ``inner_pressure``/``outer_pressure`` are positive for compression of the
  wall (the standard pressure convention): ``sigma_r(a) = -inner_pressure`` and
  ``sigma_r(b) = -outer_pressure``.
* Unit out-of-plane thickness; the annulus plane is ``x - y``.

Everything here is a closed form or a direct discretisation helper; no number
in this module is taken from a finite element result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LameState:
    """Analytic fields evaluated at the supplied radii."""

    radial_stress: torch.Tensor
    hoop_stress: torch.Tensor
    axial_stress: torch.Tensor
    radial_displacement: torch.Tensor


def lame_coefficients(
    inner_radius: float,
    outer_radius: float,
    inner_pressure: float,
    outer_pressure: float = 0.0,
    *,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(A, B)`` of ``sigma_r = A - B/r**2``, ``sigma_theta = A + B/r**2``.

    ``A`` and ``B`` follow from the two traction boundary conditions alone,
    hence they are independent of the elastic constants and of the plane
    stress/strain choice.
    """
    if not (0.0 < inner_radius < outer_radius):
        raise ValueError("require 0 < inner_radius < outer_radius")
    if inner_pressure < 0.0 or outer_pressure < 0.0:
        raise ValueError("pressures must be non-negative")
    a2 = inner_radius * inner_radius
    b2 = outer_radius * outer_radius
    scale = 1.0 / (b2 - a2)
    a_coef = (inner_pressure * a2 - outer_pressure * b2) * scale
    b_coef = a2 * b2 * (inner_pressure - outer_pressure) * scale
    return torch.tensor(a_coef, dtype=dtype), torch.tensor(b_coef, dtype=dtype)


def lame_state(
    radius: torch.Tensor,
    *,
    inner_radius: float,
    outer_radius: float,
    inner_pressure: float,
    outer_pressure: float = 0.0,
    young_modulus: float,
    poisson_ratio: float,
    plane: str = "strain",
) -> LameState:
    """Evaluate the analytic Lame fields at ``radius`` (tensor, any shape)."""
    if plane not in ("stress", "strain"):
        raise ValueError("plane must be 'stress' or 'strain'")
    if young_modulus <= 0.0:
        raise ValueError("young_modulus must be positive")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    r = torch.as_tensor(radius)
    if r.numel() and bool(torch.any(r <= 0.0)):
        raise ValueError("radius must be positive")
    a_coef, b_coef = lame_coefficients(
        inner_radius, outer_radius, inner_pressure, outer_pressure, dtype=r.dtype
    )
    nu = torch.as_tensor(poisson_ratio, dtype=r.dtype)
    inv_r2 = 1.0 / (r * r)
    radial = a_coef - b_coef * inv_r2
    hoop = a_coef + b_coef * inv_r2
    if plane == "strain":
        # eps_z = 0  =>  sigma_z = nu (sigma_r + sigma_theta) = 2 nu A
        axial = 2.0 * nu * a_coef * torch.ones_like(r)
        u_r = (1.0 + nu) / young_modulus * ((1.0 - 2.0 * nu) * a_coef * r + b_coef / r)
    else:
        axial = torch.zeros_like(r)
        u_r = ((1.0 - nu) * a_coef * r + (1.0 + nu) * b_coef / r) / young_modulus
    return LameState(radial, hoop, axial, u_r)


def sector_q4_mesh(
    inner_radius: float,
    outer_radius: float,
    n_theta: int,
    n_radial: int,
    *,
    sector_angle: float = math.pi / 2.0,
    radial_grading: float = 1.0,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured Q4 mesh of an annular sector, counter-clockwise.

    Node ``(i, j)`` sits at ``(r_i, theta_j)`` with ``theta_j = j * sector_angle
    / n_theta``, so the first and last rays lie exactly on the symmetry cuts and
    the innermost rings lie exactly on the bore circle.  ``radial_grading > 1``
    clusters rings towards the bore.

    The mesh is a *geometry* helper: the physical sector angle is a modelling
    choice, and a quarter (``pi/2``) sector reproduces the whole annulus when
    the symmetry conditions of :func:`symmetry_fixed_dofs` are applied.
    """
    if n_theta < 2 or n_radial < 1:
        raise ValueError("require n_theta >= 2 and n_radial >= 1")
    if not (0.0 < inner_radius < outer_radius):
        raise ValueError("require 0 < inner_radius < outer_radius")
    if not (0.0 < sector_angle <= 2.0 * math.pi):
        raise ValueError("sector_angle must lie in (0, 2*pi]")
    if radial_grading <= 0.0:
        raise ValueError("radial_grading must be positive")
    t = torch.linspace(0.0, 1.0, n_radial + 1, dtype=dtype) ** radial_grading
    r = inner_radius + (outer_radius - inner_radius) * t
    theta = torch.arange(n_theta + 1, dtype=dtype) * (sector_angle / n_theta)
    ring_r = r[:, None].expand(n_radial + 1, n_theta + 1)
    nodes = torch.stack((ring_r * torch.cos(theta), ring_r * torch.sin(theta)), dim=-1).reshape(-1, 2)
    cells = []
    for i in range(n_radial):
        for j in range(n_theta):
            n0 = i * (n_theta + 1) + j
            n1 = (i + 1) * (n_theta + 1) + j
            n2 = (i + 1) * (n_theta + 1) + j + 1
            n3 = i * (n_theta + 1) + j + 1
            cells.append((n0, n1, n2, n3))
    return nodes, torch.tensor(cells, dtype=torch.long)


def symmetry_fixed_dofs(
    nodes: torch.Tensor,
    *,
    sector_angle: float = math.pi / 2.0,
    n_theta: int,
    n_radial: int,
    dtype=torch.float64,
) -> torch.Tensor:
    """Axis-aligned symmetry constraints for a quarter (``pi/2``) annulus sector.

    ``u_theta = 0`` on both radial cuts removes every rigid-body mode: the
    ``theta = 0`` cut fixes ``u_y`` and the ``theta = pi/2`` cut fixes ``u_x``.
    Only quarter sectors are supported, because the constraints must be
    axis-aligned to be imposed on Cartesian degrees of freedom.
    """
    if abs(sector_angle - math.pi / 2.0) > 1e-12:
        raise ValueError("symmetry_fixed_dofs supports sector_angle = pi/2 only")
    n_nodes = (n_radial + 1) * (n_theta + 1)
    if len(nodes) != n_nodes:
        raise ValueError("node count does not match n_theta/n_radial")
    fixed = []
    for i in range(n_radial + 1):
        fixed.append(2 * (i * (n_theta + 1)) + 1)              # theta = 0  -> u_y = 0
        fixed.append(2 * (i * (n_theta + 1) + n_theta))        # theta = pi/2 -> u_x = 0
    return torch.tensor(sorted(fixed), dtype=torch.long)


def circular_pressure_loads(
    n_theta: int,
    n_radial: int,
    radius: float,
    pressure: float,
    *,
    inner: bool,
    sector_angle: float = math.pi / 2.0,
    dtype=torch.float64,
) -> torch.Tensor:
    """Work-equivalent nodal forces for a uniform pressure on an annular sector edge.

    Each arc between two adjacent rays carries the resultant
    ``2 * pressure * radius * sin(dtheta/2)`` along its bisector, which is exact
    for the arc; the resultant is split equally between the two end nodes
    (equal split is exact for a straight edge and leaves an ``O(dtheta**2)``
    redistribution error for a curved one).

    Sign convention follows :func:`lame_state`: an inner pressure pushes the
    wall radially outward (``+e_r``), an outer pressure pushes it inward.
    """
    if pressure < 0.0:
        raise ValueError("pressure must be non-negative")
    dtheta = sector_angle / n_theta
    theta = torch.arange(n_theta, dtype=dtype) * dtheta
    bisector = theta + 0.5 * dtheta
    # full resultant of the arc; split equally between its two end nodes
    magnitude = 2.0 * pressure * radius * math.sin(0.5 * dtheta)
    sign = 1.0 if inner else -1.0
    edge = sign * magnitude * torch.stack((torch.cos(bisector), torch.sin(bisector)), dim=-1)
    n_nodes = (n_radial + 1) * (n_theta + 1)
    forces = torch.zeros(2 * n_nodes, dtype=dtype)
    ring = 0 if inner else n_radial
    stride = n_theta + 1
    for j in range(n_theta):
        for node in (ring * stride + j, ring * stride + j + 1):
            forces[2 * node] += 0.5 * edge[j, 0]
            forces[2 * node + 1] += 0.5 * edge[j, 1]
    return forces


def radial_component(vector: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """Radial component of a nodal vector field ``[n_nodes, 2]``."""
    radius = torch.linalg.vector_norm(nodes, dim=1)
    unit = nodes / radius[:, None]
    return torch.einsum("ij,ij->i", vector, unit)

def cartesian_to_polar_stress(stress: torch.Tensor, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate ``[sigma_xx, sigma_yy, tau_xy]`` at ``nodes`` into polar components.

    ``nodes`` are the points at which the stress is sampled (element centres for
    :class:`~tensorfem.continuum.ContinuumResult`); the returned tuple is
    ``(sigma_r, sigma_theta, tau_rtheta)`` with the engineering shear
    convention used by :mod:`tensorfem.continuum`.
    """
    if stress.ndim != 2 or stress.shape[1] != 3:
        raise ValueError("stress must have shape [n, 3]")
    if nodes.shape != (stress.shape[0], 2):
        raise ValueError("nodes must match the stress sampling points")
    radius = torch.linalg.vector_norm(nodes, dim=1)
    c = nodes[:, 0] / radius
    s = nodes[:, 1] / radius
    sxx, syy, txy = stress[:, 0], stress[:, 1], stress[:, 2]
    radial = c * c * sxx + s * s * syy + 2.0 * s * c * txy
    hoop = s * s * sxx + c * c * syy - 2.0 * s * c * txy
    shear = (syy - sxx) * s * c + txy * (c * c - s * s)
    return radial, hoop, shear
