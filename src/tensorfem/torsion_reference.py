"""Saint-Venant torsion of circular and faceted annular tubes.

Why this module exists
----------------------
``tensorfem`` ships three-dimensional HEX8 / HEX20 elasticity
(``solid3d.solve_solid``, ``hex20.solve_hex20``) that were only ever verified in
bending, pressurisation and thermal loading.  Torsion of a tubular member had no
verification at all, although it is the governing load case for rudder stocks,
shafting and tubular bracing.  The circular tube is the cheapest case with an
*exact* three-dimensional solution, because it is the only cross-section that
carries pure torsion without warping:

    u = theta z  z_hat x r      ->      gamma_xz = -theta y, gamma_yz = theta x
    tau_theta_z(r) = G theta r,  J = pi (R_o^4 - R_i^4) / 2,  T = G J theta.

A ring of HEX8 elements can only represent that cylinder as a *faceted*
prism, so the discrete model is the exact solution of a different problem.  The
polar second moment of a regular ``n``-gon tube is known in closed form,

    J_n = (n/2) sin(2 pi / n) (R_o^4 - R_i^4),

which gives a second, sharp reference: it isolates the geometric faceting error
one obtains against ``J`` from the element's own consistency error.  HEX20
elements with mid-edge nodes projected onto the true cylinder close that gap,
because their edges are quadratic.

Everything below is analytic or a thin driver around the existing solvers; no
new element formulation is introduced.
"""
from __future__ import annotations

import math

import torch

from .hex20 import Hex20Model, solve_hex20, upgrade_hex8_to_hex20
from .solid3d import SolidModel, solve_solid


def shear_modulus(young_modulus: float, poisson_ratio: float) -> float:
    """Isotropic shear modulus G = E / (2 (1 + nu))."""
    return young_modulus / (2.0 * (1.0 + poisson_ratio))


def tube_torsional_constant(inner_radius: float, outer_radius: float) -> float:
    """Exact Saint-Venant constant of a circular tube, J = pi (Ro^4 - Ri^4) / 2."""
    if not 0.0 <= inner_radius < outer_radius:
        raise ValueError("need 0 <= inner_radius < outer_radius")
    return math.pi * (outer_radius**4 - inner_radius**4) / 2.0


def faceted_torsional_constant(inner_radius: float, outer_radius: float, n_theta: int) -> float:
    """Polar second moment of a regular ``n_theta``-gon tube.

    For a triangle spanned by the centre and one facet, ``int r^2 dA`` is
    ``A R^2 (2 + cos(dphi)) / 6`` with ``A = R^2 sin(dphi) / 2``, so the ring of
    ``n_theta`` facets gives

        J_n = (n/12) sin(dphi) (2 + cos(dphi)) (R_o^4 - R_i^4),   dphi = 2 pi / n.

    The inscribed polygon loses area *and* loses it at the largest radii, so the
    deficit is about twice the area deficit: 4.96 % at n = 16, 1.28 % at n = 32.
    The rigid-rotation field is admissible for this prism up to an O(deficit^2)
    correction, which makes ``J_n`` the sharp reference for a faceted mesh.
    """
    if n_theta < 3:
        raise ValueError("n_theta must be at least 3")
    delta = 2.0 * math.pi / n_theta
    return (n_theta / 12.0) * math.sin(delta) * (2.0 + math.cos(delta)) * (
        outer_radius**4 - inner_radius**4
    )


def faceting_ratio(n_theta: int) -> float:
    """``J_n / J`` of an unrefined faceted tube; ``1 - ratio`` is the faceting error."""
    delta = 2.0 * math.pi / n_theta
    return (n_theta / (6.0 * math.pi)) * math.sin(delta) * (2.0 + math.cos(delta))


def surface_shear_stress(young_modulus: float, poisson_ratio: float, twist_per_length: float, radius: float):
    """Exact polar shear stress tau_theta_z = G theta r of the circular tube."""
    return shear_modulus(young_modulus, poisson_ratio) * twist_per_length * radius


def annular_hex_mesh(
    inner_radius: float,
    outer_radius: float,
    length: float,
    n_theta: int,
    n_radial: int,
    n_axial: int,
    *,
    dtype=torch.float64,
):
    """Periodic O-grid of HEX8 elements on an annular tube r in [Ri, Ro]."""
    if min(n_theta, n_radial, n_axial) < 1:
        raise ValueError("mesh counts must be positive")
    radii = torch.linspace(inner_radius, outer_radius, n_radial + 1, dtype=dtype)
    angles = torch.arange(n_theta, dtype=dtype) * (2.0 * math.pi / n_theta)
    heights = torch.linspace(0.0, length, n_axial + 1, dtype=dtype)

    def index(i_radial: int, i_theta: int, i_axial: int) -> int:
        return (i_axial * (n_radial + 1) + i_radial) * n_theta + (i_theta % n_theta)

    nodes = torch.empty((n_theta * (n_radial + 1) * (n_axial + 1), 3), dtype=dtype)
    for k in range(n_axial + 1):
        for i in range(n_radial + 1):
            r = radii[i]
            for j in range(n_theta):
                a = angles[j]
                nodes[index(i, j, k)] = torch.stack(
                    (r * torch.cos(a), r * torch.sin(a), heights[k])
                )

    cells = []
    for k in range(n_axial):
        for i in range(n_radial):
            for j in range(n_theta):
                cells.append(
                    (
                        index(i, j, k),
                        index(i + 1, j, k),
                        index(i + 1, j + 1, k),
                        index(i, j + 1, k),
                        index(i, j, k + 1),
                        index(i + 1, j, k + 1),
                        index(i + 1, j + 1, k + 1),
                        index(i, j + 1, k + 1),
                    )
                )
    return nodes, torch.tensor(cells, dtype=torch.long)


def project_hex20_to_cylinders(
    nodes: torch.Tensor,
    inner_radius: float,
    outer_radius: float,
    elements: torch.Tensor | None = None,
    *,
    rtol: float = 1e-9,
) -> torch.Tensor:
    """Move HEX20 mid-edge nodes that sit on a facet chord back onto the cylinder.

    A mid-edge node is projected only when *both* of its parent corners already
    lie on the same cylindrical surface, so interior and end-face mid-edge nodes
    keep their position even when the radial spacing is small.  Corners are
    untouched; the operation is idempotent.
    """
    out = nodes.clone()
    radius = torch.linalg.vector_norm(nodes[:, :2], dim=1)
    targets = [t for t in (inner_radius, outer_radius) if t > 0.0]
    if elements is None:
        raise ValueError("elements are required to identify surface mid-edge nodes")
    from .hex20 import _EDGES

    shared = torch.zeros(nodes.shape[0], dtype=torch.long)  # 0 none, else index into targets+1
    cells = elements.long()
    for corner, (a, b) in enumerate(_EDGES):
        first = radius[cells[:, a]]
        second = radius[cells[:, b]]
        mid = cells[:, 8 + corner]
        for k, target in enumerate(targets):
            pair = (first - target).abs() <= rtol * target
            pair = pair & ((second - target).abs() <= rtol * target)
            shared[mid[pair]] = k + 1
    for k, target in enumerate(targets):
        mask = shared == k + 1
        if not bool(mask.any()):
            continue
        scale = target / radius[mask]
        out[mask, 0] = nodes[mask, 0] * scale
        out[mask, 1] = nodes[mask, 1] * scale
    return out


def prescribed_twist(
    nodes: torch.Tensor,
    twist_per_length: float,
    length: float,
    *,
    atol: float = 1e-12,
):
    """Dirichlet data for the linear twist ramp ``u = theta z (z_hat x r)``.

    Applied on both end planes, where the exact solution prescribes a rigid
    rotation of magnitude ``theta z`` and zero axial displacement.  The lateral
    surface is left free, which is exactly the traction-free Saint-Venant state.
    """
    z = nodes[:, 2]
    on_end = (z.abs() <= atol) | ((z - length).abs() <= atol)
    ids = torch.where(on_end)[0]
    angle = twist_per_length * z[ids]
    values = torch.stack(
        (
            -angle * nodes[ids, 1],
            angle * nodes[ids, 0],
            torch.zeros_like(angle),
        ),
        dim=1,
    )
    dofs = torch.stack((3 * ids, 3 * ids + 1, 3 * ids + 2), dim=1).reshape(-1)
    return dofs, values.reshape(-1)


def reaction_torque(nodes: torch.Tensor, reaction: torch.Tensor, *, at_z: float | None = None,
                    atol: float = 1e-12) -> float:
    """Torque about z from a nodal reaction vector ordered [ux0, uy0, uz0, ...].

    Global equilibrium makes the sum over both loaded ends vanish, so ``at_z``
    restricts the sum to one end plane (one of ``0`` or ``length``).
    """
    forces = reaction.reshape(-1, 3)
    select = torch.ones(nodes.shape[0], dtype=torch.bool, device=nodes.device)
    if at_z is not None:
        select = (nodes[:, 2] - at_z).abs() <= atol
    return float(
        (nodes[select, 0] * forces[select, 1] - nodes[select, 1] * forces[select, 0]).sum()
    )


def polar_shear(stress: torch.Tensor, centres: torch.Tensor):
    """Return (radius, tau_theta_z) from element stresses at the given points.

    ``solid3d`` stores shares as [exx, eyy, ezz, gxy, gyz, gxz], so tau_theta_z =
    -tau_xz sin(phi) + tau_yz cos(phi) with phi the polar angle of the point.
    """
    x = centres[:, 0]
    y = centres[:, 1]
    radius = torch.hypot(x, y).clamp_min(torch.finfo(centres.dtype).eps)
    sin = y / radius
    cos = x / radius
    return radius, -stress[:, 5] * sin + stress[:, 4] * cos


def element_centres(nodes: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Mean of the eight corners; adequate for plotting but not for integrals."""
    return nodes[cells.long()].mean(dim=1)


def section_centroids(nodes: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Exact area centroid of the four in-plane corners of every brick.

    With a uniform stress state -- which the linear twist ramp produces exactly --
    the torque integral becomes ``A (-tau_xz y_c + tau_yz x_c)``, so an exact
    centroid turns the stress-implied torque into a sharp check of the reaction
    torque instead of a one-point quadrature estimate.
    """
    corners = nodes[cells.long()][:, :4]
    nxt = corners[:, [1, 2, 3, 0], 0] * corners[:, [0, 1, 2, 3], 1] - corners[:, [0, 1, 2, 3], 0] * corners[:, [1, 2, 3, 0], 1]
    area = 0.5 * nxt.sum(dim=1)
    keep = area.abs().clamp_min(torch.finfo(area.dtype).tiny)
    centre = corners.mean(dim=1)
    # shoelace centroid; the 4-point formula is exact for a planar quadrilateral
    moment_x = (
        (corners[:, 0, 0] + corners[:, 1, 0]) * nxt[:, 0]
        + (corners[:, 1, 0] + corners[:, 2, 0]) * nxt[:, 1]
        + (corners[:, 2, 0] + corners[:, 3, 0]) * nxt[:, 2]
        + (corners[:, 3, 0] + corners[:, 0, 0]) * nxt[:, 3]
    )
    moment_y = (
        (corners[:, 0, 1] + corners[:, 1, 1]) * nxt[:, 0]
        + (corners[:, 1, 1] + corners[:, 2, 1]) * nxt[:, 1]
        + (corners[:, 2, 1] + corners[:, 3, 1]) * nxt[:, 2]
        + (corners[:, 3, 1] + corners[:, 0, 1]) * nxt[:, 3]
    )
    return torch.stack((moment_x / (6.0 * area), moment_y / (6.0 * area), centre[:, 2]), dim=1)


def plane_twist(nodes: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Least-squares rigid rotation about z of every z-plane of a mesh.

    Returns one angle per distinct z level (ascending).  For the exact tube field
    this reproduces ``theta z``; a deviation indicates z-dependent (end) effects.
    """
    u = displacement.reshape(-1, 3)
    levels, inverse = torch.unique(nodes[:, 2], sorted=True, return_inverse=True)
    numerator = nodes[:, 0] * u[:, 1] - nodes[:, 1] * u[:, 0]
    denominator = nodes[:, 0] ** 2 + nodes[:, 1] ** 2
    angle = torch.zeros_like(levels)
    for order in range(levels.numel()):
        mask = inverse == order
        angle[order] = numerator[mask].sum() / denominator[mask].sum()
    return levels, angle


def _twist_summary(nodes, displacement, stress, cells, elements, dofs, length, twist_per_length):
    probe = element_centres(nodes, cells)
    centres = section_centroids(nodes, cells)
    _, tau_probe = polar_shear(stress, probe)
    _, tau_centre = polar_shear(stress, centres)
    levels, angle = plane_twist(nodes, displacement)
    ramp = torch.stack(
        (
            -twist_per_length * nodes[:, 2] * nodes[:, 1],
            twist_per_length * nodes[:, 2] * nodes[:, 0],
            torch.zeros(nodes.shape[0], dtype=nodes.dtype),
        ),
        dim=1,
    ).reshape(-1)
    return {
        "elements": int(elements),
        "dofs": int(dofs),
        "probe": probe,
        "centres": centres,
        "tau_theta_z": tau_centre,
        "tau_probe": tau_probe,
        "twist_levels": levels,
        "twist_angle": angle,
        "displacement_patch_error": float(
            (displacement - ramp).abs().max() / ramp.abs().max()
        ),
    }


def solve_torsion_hex8(
    inner_radius: float,
    outer_radius: float,
    length: float,
    twist_per_length: float,
    young_modulus: float,
    poisson_ratio: float,
    n_theta: int,
    n_radial: int,
    n_axial: int,
    *,
    dtype=torch.float64,
) -> dict:
    """Prescribed-twist HEX8 model of the faceted annular tube."""
    nodes, cells = annular_hex_mesh(
        inner_radius, outer_radius, length, n_theta, n_radial, n_axial, dtype=dtype
    )
    dofs, values = prescribed_twist(nodes, twist_per_length, length)
    model = SolidModel(
        nodes=nodes,
        elements=cells,
        young_modulus=nodes.new_tensor(young_modulus),
        poisson_ratio=nodes.new_tensor(poisson_ratio),
        forces=nodes.new_zeros(3 * nodes.shape[0]),
        fixed_dofs=dofs,
        element_type="hex8",
        prescribed_values=values,
    )
    result = solve_solid(model)
    torque = reaction_torque(nodes, result.reaction, at_z=length)
    summary = _twist_summary(
        nodes,
        result.displacement,
        result.stress,
        cells,
        cells.shape[0],
        model.n_dofs,
        length,
        twist_per_length,
    )
    centres = summary["centres"]
    areas = element_cross_section_areas(nodes, cells)
    lever = torch.linalg.vector_norm(centres[:, :2], dim=1)
    first_layer = torch.zeros(cells.shape[0], dtype=torch.bool)
    first_layer[: n_radial * n_theta] = True  # elements of the z = 0 layer
    # NOTE: this is a one-point quadrature of the torque integral; int r^2 dA is
    # NOT A r_c^2, so it under-reports the reaction torque by 0.7 % (n=32) to
    # 5.8 % (n=8).  It is kept as a documented quadrature trap, never as a gate.
    summary.update(
        {
            "element": "hex8",
            "n_theta": int(n_theta),
            "n_radial": int(n_radial),
            "n_axial": int(n_axial),
            "geometry": "faceted",
            "torque": torque,
            "stress_torque_one_point": float(
                (summary["tau_theta_z"] * lever * areas)[first_layer].sum()
            ),
            "nodes": nodes,
            "cells": cells,
            "displacement": result.displacement,
        }
    )
    return summary


def solve_torsion_hex20(
    inner_radius: float,
    outer_radius: float,
    length: float,
    twist_per_length: float,
    young_modulus: float,
    poisson_ratio: float,
    n_theta: int,
    n_radial: int,
    n_axial: int,
    *,
    curved: bool = True,
    dtype=torch.float64,
) -> dict:
    """Prescribed-twist HEX20 model; ``curved`` projects mid-edge nodes to the cylinder."""
    hex8_nodes, hex8_cells = annular_hex_mesh(
        inner_radius, outer_radius, length, n_theta, n_radial, n_axial, dtype=dtype
    )
    nodes, cells = upgrade_hex8_to_hex20(hex8_nodes, hex8_cells)
    if curved:
        nodes = project_hex20_to_cylinders(nodes, inner_radius, outer_radius, cells)
    dofs, values = prescribed_twist(nodes, twist_per_length, length)
    model = Hex20Model(
        nodes=nodes,
        elements=cells,
        young_modulus=nodes.new_tensor(young_modulus),
        poisson_ratio=nodes.new_tensor(poisson_ratio),
        forces=nodes.new_zeros(3 * nodes.shape[0]),
        fixed_dofs=dofs,
        prescribed_values=values,
    )
    result = solve_hex20(model)
    torque = reaction_torque(nodes, result.reaction, at_z=length)
    # Element centroids of a serendipity brick: mean of the eight corners.
    corner_cells = cells[:, :8]
    summary = _twist_summary(
        nodes,
        result.displacement,
        result.stress,
        corner_cells,
        cells.shape[0],
        model.n_dofs,
        length,
        twist_per_length,
    )
    summary.update(
        {
            "element": "hex20",
            "n_theta": int(n_theta),
            "n_radial": int(n_radial),
            "n_axial": int(n_axial),
            "geometry": "curved-edges" if curved else "faceted",
            "torque": torque,
            "nodes": nodes,
            "cells": cells,
            "displacement": result.displacement,
        }
    )
    return summary


def element_cross_section_areas(nodes: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Cross-section area of every element, from the shoelace rule on its corners.

    The inputs are the four in-plane corners of the brick, ordered around the
    section, so the flat-facet area (smaller than the analytic sector area) is
    recovered exactly instead of assuming a polar parametrisation.
    """
    corners = nodes[cells.long()]
    return 0.5 * (
        corners[:, 0, 0] * corners[:, 1, 1] - corners[:, 1, 0] * corners[:, 0, 1]
        + corners[:, 1, 0] * corners[:, 2, 1] - corners[:, 2, 0] * corners[:, 1, 1]
        + corners[:, 2, 0] * corners[:, 3, 1] - corners[:, 3, 0] * corners[:, 2, 1]
        + corners[:, 3, 0] * corners[:, 0, 1] - corners[:, 0, 0] * corners[:, 3, 1]
    ).abs()


def torsional_rigidity(summary: dict, young_modulus: float, poisson_ratio: float, twist_per_length: float) -> float:
    """Measured ``G J`` from the reaction torque of a prescribed unit twist."""
    return summary["torque"] / twist_per_length


def torsional_constant(summary: dict, young_modulus: float, poisson_ratio: float, twist_per_length: float) -> float:
    """Measured ``J`` = T / (G theta) from the reaction torque."""
    return torsional_rigidity(summary, young_modulus, poisson_ratio, twist_per_length) / shear_modulus(
        young_modulus, poisson_ratio
    )