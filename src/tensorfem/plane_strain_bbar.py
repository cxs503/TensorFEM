"""Mean-dilatation (B-bar) bilinear quadrilateral for the 2-D continuum path.

Why this module exists
----------------------
``docs/near-incompressible-locking.md`` cured volumetric locking for the 3-D
HEX8 element and then recorded, in its §6 "outstanding work" list, that the 2-D
``q4`` path had **no** B-bar / EAS / BF mitigation: at ``nu = 0.4999`` the
plane-strain radial-stress L2 error on the thick-cylinder Lame case was 155 %.
Ship structures are solved on exactly that path (rubber bearings, coatings,
composite matrix in plane strain), so a formulation that locks by 100x is a
functional gap, not an academic one.

Formulation
-----------
Hughes (1980) mean dilatation.  With the strain-displacement operator split into
a volumetric and a deviatoric part,

    B( xi) = B_dev( xi) + s . div( xi) / 2 ,      s = (1, 1, 0)^T ,

the volumetric part is replaced by its element average

    div_bar = int div dV / int dV   (2x2 Gauss, exact for the bilinear map),

so that the element stiffness becomes

    K_e = int B_bar^T D B_bar dV,   B_bar = B_dev( xi) + s . div_bar / 2.

The deviatoric part keeps full 2x2 integration, hence no hourglass mode is
introduced: the rank audit in ``tests/test_plane_strain_bbar.py`` shows the
element still has exactly three zero eigenvalues (the rigid-body modes) and five
deformation modes on a *distorted* element.

In plane strain the in-plane dilatation is ``eps_xx + eps_yy`` (because
``eps_zz = 0``), which is what ``div`` above is.  On a constant-strain field
``div_bar == div`` pointwise, so the element passes the patch test exactly; that
is asserted in the test suite rather than assumed.

Stress recovery follows the modified operator: the element-centre strain keeps
its deviatoric part and takes the mean volumetric part.  This is the same choice
``tensorfem.solid3d_bbar`` makes for HEX8 and keeps the recovered stress
consistent with the stiffness that produced the displacement.
"""
from __future__ import annotations

import torch

from .continuum import ContinuumModel, ContinuumResult, _b_matrix, elasticity_matrix

#: 2x2 Gauss abscissa.
_GAUSS = 1.0 / 3.0**0.5
_POINTS = ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS))
#: Volumetric selector for the ``(eps_xx, eps_yy, gamma_xy)`` strain ordering.
_VOLUMETRIC = (1.0, 1.0, 0.0)


def _q4_gradient(coordinates: torch.Tensor, xi: float, eta: float):
    """Physical derivative operator and Jacobian determinant at ``(xi, eta)``."""
    natural = coordinates.new_tensor([
        [-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
        [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)],
    ]) / 4
    jacobian = torch.einsum("eia,ib->eab", coordinates, natural)
    det = torch.linalg.det(jacobian)
    if torch.any(det <= 0):
        raise ValueError("Q4 element has non-positive Jacobian")
    derivatives = torch.einsum("ib,ebc->eic", natural, torch.linalg.inv(jacobian))
    return _b_matrix(derivatives), det


def q4_bbar_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor, thickness: torch.Tensor):
    """Mean-dilatation stiffness and the averaged divergence row.

    Parameters
    ----------
    coordinates : [n_elements, 4, 2]
        Nodal coordinates in counter-clockwise order.
    dmat : [n_elements, 3, 3]
        Engineering-shear elasticity matrix (plane stress or plane strain).
    thickness : [n_elements]
        Unit out-of-plane thickness of the plane model.

    Returns
    -------
    (stiffness, div_average) : ([n_elements, 8, 8], [n_elements, 8])
        ``div_average`` is the element-mean volumetric row, reused by
        :func:`q4_bbar_centre_strain` for consistent stress recovery.
    """
    selector = coordinates.new_tensor(_VOLUMETRIC)
    n_elements = len(coordinates)
    volume = torch.zeros(n_elements, dtype=coordinates.dtype, device=coordinates.device)
    divergence_integral = torch.zeros((n_elements, 8), dtype=coordinates.dtype,
                                      device=coordinates.device)
    sampled = []
    for xi, eta in _POINTS:
        b, det = _q4_gradient(coordinates, xi, eta)
        divergence = b[:, 0] + b[:, 1]
        sampled.append((b, divergence, det))
        volume = volume + det
        divergence_integral = divergence_integral + divergence * det[:, None]
    div_average = divergence_integral / volume[:, None]
    average_volumetric = selector[None, :, None] * div_average[:, None, :] / 2
    stiffness = torch.zeros((n_elements, 8, 8), dtype=coordinates.dtype,
                            device=coordinates.device)
    for b, divergence, det in sampled:
        point_volumetric = selector[None, :, None] * divergence[:, None, :] / 2
        b_bar = b - point_volumetric + average_volumetric
        stiffness = stiffness + (b_bar.transpose(1, 2) @ dmat @ b_bar) * (det * thickness)[:, None, None]
    return stiffness, div_average


def q4_bbar_centre_strain(coordinates: torch.Tensor, div_average: torch.Tensor,
                          element_displacements: torch.Tensor) -> torch.Tensor:
    """Element-centre strain with the volumetric part replaced by its mean.

    ``element_displacements`` has shape ``[n_elements, 8]``.
    """
    b_centre = _q4_gradient(coordinates, 0.0, 0.0)[0]
    strain = torch.einsum("eij,ej->ei", b_centre, element_displacements)
    divergence_mean = torch.einsum("ej,ej->e", div_average, element_displacements)
    divergence_point = strain[:, 0] + strain[:, 1]
    correction = (divergence_mean - divergence_point) / 2
    strain = strain.clone()
    strain[:, 0] = strain[:, 0] + correction
    strain[:, 1] = strain[:, 1] + correction
    return strain


def solve_continuum_bbar(model: ContinuumModel) -> ContinuumResult:
    """Solve an existing :class:`ContinuumModel` with mean-dilatation Q4 elements.

    Everything except the element stiffness and stress recovery is the production
    logic of :func:`tensorfem.continuum.solve_continuum`, so constraints,
    prescribed values and reactions behave identically.
    """
    if model.element_type.lower() != "q4":
        raise ValueError("B-bar formulation is implemented for Q4 only")
    n_elements, n_nodes = model.elements.shape
    expand = lambda values: values.to(dtype=model.nodes.dtype, device=model.nodes.device)\
        .reshape(-1).expand(n_elements)
    young, poisson, thickness = map(expand, (model.young_modulus, model.poisson_ratio,
                                             model.thickness))
    dmat = elasticity_matrix(young, poisson, model.plane)
    coordinates = model.nodes[model.elements.long()]
    stiffness_e, div_average = q4_bbar_stiffness(coordinates, dmat, thickness)
    edofs = torch.stack(tuple(2*model.elements + i for i in (0, 1)), 2).reshape(n_elements, 8).long()
    rows = edofs[:, :, None].expand(-1, -1, 8).reshape(-1)
    cols = edofs[:, None, :].expand(-1, 8, -1).reshape(-1)
    stiffness = torch.zeros((model.n_dofs, model.n_dofs), dtype=model.nodes.dtype,
                            device=model.nodes.device)
    stiffness = stiffness.index_put((rows, cols), stiffness_e.reshape(-1), accumulate=True)
    fixed = model.fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    values = (torch.zeros_like(fixed, dtype=model.nodes.dtype) if model.prescribed_values is None
              else model.prescribed_values.to(dtype=model.nodes.dtype, device=model.nodes.device))
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.arange(model.n_dofs, device=model.nodes.device)[mask]
    forces = model.forces.to(dtype=model.nodes.dtype, device=model.nodes.device)
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    u = u.index_put((fixed,), values)
    if free.numel():
        rhs = forces[free] - stiffness[free][:, fixed] @ values
        u = u.index_put((free,), torch.linalg.solve(stiffness[free][:, free], rhs))
    strain = q4_bbar_centre_strain(coordinates, div_average, u[edofs])
    stress = torch.einsum("eij,ej->ei", dmat, strain)
    return ContinuumResult(u, stiffness @ u - forces, strain, stress,
                           0.5 * u @ stiffness @ u, stiffness)


def bbar_element_eigenvalues(nodes: torch.Tensor, young: float = 210.0e9,
                             poisson: float = 0.4999, plane: str = "strain") -> torch.Tensor:
    """Ascending eigenvalues of a single B-bar element (hourglass audit)."""
    coordinates = nodes.reshape(1, 4, 2)
    dmat = elasticity_matrix(coordinates.new_tensor([young]), coordinates.new_tensor([poisson]),
                             plane)
    stiffness, _ = q4_bbar_stiffness(coordinates, dmat, coordinates.new_tensor([1.0]))
    return torch.linalg.eigvalsh((stiffness[0] + stiffness[0].T) / 2)