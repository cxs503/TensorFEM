"""Closed thin Shell4 sphere under uniform internal pressure: exact membrane oracle.

A closed thin spherical shell loaded by a uniform pressure has an exact
bending-free solution in shell (membrane) theory.  With mid-surface radius ``R``,
thickness ``t``, ``E`` and ``nu`` and internal pressure ``p``::

    N_theta = N_phi = p*R/2                (membrane resultant, N/m)
    sigma   = p*R/(2*t)                   (equal biaxial stress)
    w       = p*R**2*(1-nu)/(2*E*t)       (uniform radial displacement)

Because the shell is closed there is no free edge, so no bending boundary layer
exists and the membrane state is *exact* rather than asymptotic.  This makes the
case the only stress-level analytic oracle for the doubly curved Shell4
discretisation in the library; the 18-degree-hole hemisphere and Cook membrane
benchmarks carry displacement oracles only.

The facets use the unchanged projected Shell4 element. Pressure is integrated
on each bilinear surface and scattered to shared nodes. The sparse direct solve
supports refinement until the worst principal resultant meets the same smooth
sphere oracle. Curvature and quadrature errors remain measurable on coarse grids.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .plate import _shape
from .sparse_core import assemble_coo
from .sphere_pressure_benchmark import cube_sphere_surface
from .spherical_shell import projected_shell4_stiffness

D = torch.float64

PROBLEM = {'radius_m': 10., 'thickness_m': .04, 'young_modulus_pa': 6.825e7,
           'poisson_ratio': .3, 'internal_pressure_pa': 1.}


@dataclass(frozen=True)
class SphereMembraneResult:
    facets_per_face: int
    nodes: torch.Tensor
    elements: torch.Tensor
    displacement: torch.Tensor
    radius_ratios: dict
    element_resultants: dict
    applied_force: torch.Tensor
    reaction: torch.Tensor


def membrane_reference(problem: dict = PROBLEM) -> dict:
    """Exact closed-sphere membrane state (shell theory, small strain)."""
    R = float(problem['radius_m'])
    t = float(problem['thickness_m'])
    E = float(problem['young_modulus_pa'])
    nu = float(problem['poisson_ratio'])
    p = float(problem['internal_pressure_pa'])
    if min(R, t, E, p) <= 0 or not (-1. < nu < .5):
        raise ValueError('positive radius/thickness/modulus/pressure and -1 < nu < 0.5')
    return {'resultant_n_per_m': p * R / 2.,
            'stress_pa': p * R / (2. * t),
            'radial_displacement_m': p * R * R * (1. - nu) / (2. * E * t)}


def sphere_shell_mesh(facets_per_face: int, problem: dict = PROBLEM):
    """Cube-mapped quad mesh of a closed sphere with radius ``radius_m``."""
    if facets_per_face < 2 or facets_per_face % 2:
        raise ValueError('even facets_per_face >= 2 required by the cube mapping')
    unit, faces = cube_sphere_surface(facets_per_face)
    nodes = unit.to(D) * float(problem['radius_m'])
    elements = orient_outward(nodes, faces.to(torch.long))
    return nodes, elements


def orient_outward(nodes: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
    """Flip each quad so its diagonal cross product points away from the centre."""
    xyz = nodes[elements]
    normal = torch.linalg.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 1], dim=1)
    flip = (normal * xyz.mean(1)).sum(1) < 0
    out = elements.clone()
    out[flip] = elements[flip][:, [0, 3, 2, 1]]
    return out


def facet_geometry(nodes: torch.Tensor, elements: torch.Tensor) -> dict:
    """Facet areas, outward normals and facet-plane distance from the centre."""
    xyz = nodes[elements]
    # Quadrilateral area = half the cross product of the two diagonals.
    area_vector = .5 * torch.linalg.cross(xyz[:, 2] - xyz[:, 0],
                                         xyz[:, 3] - xyz[:, 1], dim=1)
    area = torch.linalg.vector_norm(area_vector, dim=1)
    unit = area_vector / area[:, None]
    centroid = xyz.mean(1)
    rho = (centroid * unit).sum(1)  # distance from the centre to the facet plane
    return {'area': area, 'normal': unit, 'centroid': centroid, 'rho': rho}


def pressure_load(nodes: torch.Tensor, elements: torch.Tensor, pressure: float,
                  geometry: dict | None = None) -> torch.Tensor:
    """Consistent nodal load integrated on each bilinear facet with 2x2 quadrature."""
    xyz = nodes[elements]
    face_load = torch.zeros_like(xyz)
    g = 1. / math.sqrt(3.)
    for xi in (-g, g):
        for eta in (-g, g):
            shape, deriv = _shape(nodes.new_tensor(xi), nodes.new_tensor(eta))
            tangent = xyz.transpose(1, 2) @ deriv
            area_vector = torch.linalg.cross(tangent[:, :, 0], tangent[:, :, 1], dim=1)
            face_load += pressure * shape[None, :, None] * area_vector[:, None, :]
    nodal = torch.zeros_like(nodes)
    nodal.index_add_(0, elements.flatten(), face_load.reshape(-1, 3))
    force = nodes.new_zeros(6 * len(nodes))
    force.view(-1, 6)[:, :3] = nodal
    return force


def element_membrane_resultants(nodes: torch.Tensor, elements: torch.Tensor,
                                displacement: torch.Tensor,
                                problem: dict = PROBLEM,
                                geometry: dict | None = None) -> dict:
    """Centre-point membrane resultants in each facet's own tangent frame."""
    geom = facet_geometry(nodes, elements) if geometry is None else geometry
    xyz = nodes[elements]
    e3 = geom['normal']
    raw = xyz[:, 1] - xyz[:, 0]
    e1 = raw - (raw * e3).sum(1)[:, None] * e3
    e1 = e1 / torch.linalg.vector_norm(e1, dim=1)[:, None]
    e2 = torch.linalg.cross(e3, e1, dim=1)
    basis = torch.stack((e1, e2), 1)                        # (ne,2,3)
    xy = torch.einsum('nai,nji->naj', xyz - geom['centroid'][:, None, :], basis)
    # Bilinear shape-function derivatives at the centre (xi=eta=0).
    quarter = xyz.new_tensor(.25)
    dN = torch.stack((torch.stack((-quarter, -quarter)),
                      torch.stack((quarter, -quarter)),
                      torch.stack((quarter, quarter)),
                      torch.stack((-quarter, quarter)))).expand(len(elements), 4, 2)
    jac = torch.einsum('nai,naj->nij', xy, dN)
    grad = torch.einsum('naj,njk->nak', dN, torch.linalg.inv(jac))  # dN_a/dx_k
    trans = displacement.reshape(-1, 6)[elements, :3]                # (ne,4,3)
    u1 = (trans * e1[:, None, :]).sum(-1)
    u2 = (trans * e2[:, None, :]).sum(-1)
    strain = torch.stack(((grad[:, :, 0] * u1).sum(1),
                          (grad[:, :, 1] * u2).sum(1),
                          (grad[:, :, 1] * u1 + grad[:, :, 0] * u2).sum(1)), 1)
    E = float(problem['young_modulus_pa'])
    nu = float(problem['poisson_ratio'])
    t = float(problem['thickness_m'])
    C = nodes.new_tensor([[1., nu, 0.], [nu, 1., 0.],
                          [0., 0., (1. - nu) / 2.]]) * (E / (1. - nu * nu))
    stress = strain @ C.T
    resultant = t * stress
    tensor = torch.stack((torch.stack((resultant[:, 0], resultant[:, 2]), 1),
                          torch.stack((resultant[:, 2], resultant[:, 1]), 1)), 1)
    principal = torch.linalg.eigvalsh(tensor)
    # Azimuth projected onto the facet; its perpendicular is the meridian.
    axis = torch.zeros_like(e3)
    axis[:, 2] = 1.
    azimuth = torch.linalg.cross(axis, geom['centroid'], dim=1)
    azimuth -= (azimuth * e3).sum(1)[:, None] * e3
    norm = torch.linalg.vector_norm(azimuth, dim=1, keepdim=True)
    azimuth = torch.where(norm > 1.e-14, azimuth / norm.clamp_min(1.e-14), e1)
    az = torch.stack(((azimuth * e1).sum(1), (azimuth * e2).sum(1)), 1)
    hoop = torch.einsum('ei,eij,ej->e', az, tensor, az)
    return {'hoop': hoop, 'meridional': resultant[:, 0] + resultant[:, 1] - hoop,
            'nx': resultant[:, 0], 'ny': resultant[:, 1], 'nxy': resultant[:, 2],
            'principal_min': principal[:, 0], 'principal_max': principal[:, 1],
            'mean': .5 * (principal[:, 0] + principal[:, 1]),
            'isotropy_defect': (principal[:, 1] - principal[:, 0]).abs()
                               / (.5 * (principal[:, 0] + principal[:, 1])).abs(),
            'von_mises': torch.sqrt(resultant[:, 0] ** 2 - resultant[:, 0] * resultant[:, 1]
                                    + resultant[:, 1] ** 2 + 3. * resultant[:, 2] ** 2)}


def uniform_expansion_fit(nodes: torch.Tensor, displacement: torch.Tensor) -> dict:
    """Least-squares split of ``u`` into uniform radial growth plus rigid modes.

    The constrained solve fixes six DOFs, so the raw field carries an arbitrary
    rigid motion.  Fitting ``u = w*n + a + omega x x`` removes it exactly and
    yields the same ``w`` for any admissible constraint choice.
    """
    xyz = displacement.reshape(-1, 6)[:, :3]
    nn = len(nodes)
    n = nodes / torch.linalg.vector_norm(nodes, dim=1)[:, None]
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    zero = torch.zeros_like(x)
    translation = ((torch.ones_like(x), zero, zero), (zero, torch.ones_like(x), zero),
                   (zero, zero, torch.ones_like(x)))
    rotation = ((zero, -z, y), (z, zero, -x), (-y, x, zero))
    fields = [n] + [torch.stack(c, 1) for c in translation + rotation]
    design = torch.stack(fields, -1).reshape(nn * 3, 7)        # (3*nn, 7)
    unknown = torch.linalg.lstsq(design, xyz.reshape(-1)).solution.reshape(7)
    fitted = design @ unknown
    residual = xyz.reshape(-1) - fitted
    scale = float(unknown[0].abs()) * math.sqrt(nn)
    return {'radial_displacement': float(unknown[0]),
            'rigid_translation': unknown[1:4],
            'residual_norm': float(torch.linalg.vector_norm(residual)),
            'relative_residual': float(torch.linalg.vector_norm(residual)) / max(scale, 1e-300)}


def _constrained_solve(K: torch.Tensor, force: torch.Tensor, *, fixed) -> tuple:
    """Solve the CPU float64 verification system with zero prescribed DOFs."""
    ndof = force.numel()
    free_mask = torch.ones(ndof, dtype=torch.bool)
    free_mask[list(fixed)] = False
    free = torch.nonzero(free_mask).flatten()
    # Direct factorization avoids densifying the fine-grid 6-DOF shell.
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    coo = K.coalesce()
    rows, cols = coo.indices().numpy()
    matrix = coo_matrix((coo.values().numpy(), (rows, cols)), shape=K.shape).tocsr()
    indices = free.numpy()
    solution = spsolve(matrix[indices][:, indices], force[free].numpy())
    displacement = force.new_zeros(ndof)
    displacement[free] = torch.from_numpy(solution)
    return displacement, torch.from_numpy(matrix @ displacement.numpy()) - force


def solve_sphere_membrane(facets_per_face: int, problem: dict = PROBLEM, *,
                          drilling_factor: float = 1e-6) -> SphereMembraneResult:
    """Solve the pressure-loaded closed sphere and recover the membrane state."""
    nodes, elements = sphere_shell_mesh(facets_per_face, problem)
    geom = facet_geometry(nodes, elements)
    E = float(problem['young_modulus_pa'])
    nu = float(problem['poisson_ratio'])
    t = float(problem['thickness_m'])
    p = float(problem['internal_pressure_pa'])
    local = torch.stack([projected_shell4_stiffness(nodes[el], E, nu, t,
                                                    drilling_factor=drilling_factor)
                         for el in elements])
    dofs = (6 * elements[:, :, None] + torch.arange(6, device=elements.device)).reshape(-1, 24)
    K = assemble_coo(local, dofs, 6 * len(nodes))
    force = pressure_load(nodes, elements, p, geom)
    # Translational 3-2-1 gauge at +x, -x, +y. It admits uniform expansion
    # plus a translation and leaves the independent shell rotations free.
    a = int(nodes[:, 0].argmax()) * 6
    b = int(nodes[:, 0].argmin()) * 6
    c = int(nodes[:, 1].argmax()) * 6
    displacement, reaction = _constrained_solve(K, force, fixed=(a, a+1, a+2, b+1, b+2, c+2))
    axial = uniform_expansion_fit(nodes, displacement)
    resultants = element_membrane_resultants(nodes, elements, displacement,
                                             problem, geom)
    reference = membrane_reference(problem)
    n_facet = p * float(geom['rho'].mean()) / 2.
    mean_n = float(resultants['mean'].mean())
    uniform_w = float(axial['radial_displacement'])
    w_exact = reference['radial_displacement_m']
    return SphereMembraneResult(
        facets_per_face=facets_per_face, nodes=nodes, elements=elements,
        displacement=displacement, applied_force=force, reaction=reaction,
        element_resultants={k: v.detach() for k, v in resultants.items()},
        radius_ratios={
            **{key + '_max_rel_error': float((resultants[key] / reference['resultant_n_per_m'] - 1).abs().max())
               for key in ('principal_min', 'principal_max', 'hoop', 'meridional')},
            'load_moment_norm_Nm': float(torch.linalg.vector_norm(torch.linalg.cross(
                nodes, force.reshape(-1, 6)[:, :3], dim=1).sum(0))),
            'facet_count': int(len(elements)),
            'node_count': int(len(nodes)),
            'mean_facet_plane_radius_m': float(geom['rho'].mean()),
            'total_facet_area_m2': float(geom['area'].sum()),
            'sphere_area_m2': 4. * math.pi * float(problem['radius_m']) ** 2,
            'resultant_mean_n_per_m': mean_n,
            'resultant_isotropy_defect': float(resultants['isotropy_defect'].mean()),
            'resultant_scatter_relative': float(resultants['mean'].std() / mean_n),
            'resultant_reference_exact_n_per_m': reference['resultant_n_per_m'],
            'resultant_reference_facet_n_per_m': n_facet,
            'resultant_error_vs_exact': abs(mean_n / reference['resultant_n_per_m'] - 1.),
            'resultant_error_vs_facet': abs(mean_n / n_facet - 1.),
            'stress_mean_pa': float(resultants['mean'].mean() / t),
            'stress_reference_pa': reference['stress_pa'],
            'radial_displacement_m': uniform_w,
            'radial_displacement_reference_m': w_exact,
            'radial_displacement_error': abs(uniform_w / w_exact - 1.),
            'uniform_expansion_relative_residual': axial['relative_residual'],
            'applied_force_resultant_n': float(torch.linalg.vector_norm(force.reshape(-1, 6)[:, :3].sum(0))),
            'applied_force_total_n': float(torch.linalg.vector_norm(force.reshape(-1, 6)[:, :3]).sum()),
            'reaction_norm_n': float(torch.linalg.vector_norm(reaction)),
            'mesh_defect': mesh_defect(elements),
        })


def mesh_defect(elements: torch.Tensor) -> dict:
    """Closed-manifold audit: every edge twice, opposite orientation, Euler 2."""
    from collections import Counter
    edges, orientation = Counter(), Counter()
    for cell in elements.tolist():
        for a, b in zip(cell, cell[1:] + cell[:1]):
            edges[min(a, b), max(a, b)] += 1
            orientation[min(a, b), max(a, b)] += 1 if a < b else -1
    node_count = int(elements.max()) + 1
    defect = node_count - len(edges) + len(elements) - 2
    return {'boundary_or_nonmanifold_edges': sum(v != 2 for v in edges.values()),
            'orientation_defects': sum(v != 0 for v in orientation.values()),
            'euler_characteristic': defect + 2}