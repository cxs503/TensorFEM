"""Mean-dilatation (B-bar) bilinear quadrilateral for 2-D continuum problems.

Why this module exists
----------------------
The near-incompressible qualification of the 3-D family
(:mod:`tensorfem.incompressible_locking`) closed the debt for the B-bar HEX8 but
had to record an **open** debt for the 2-D path: :mod:`tensorfem.continuum`
carries only the standard displacement Q4/CST, so under plane strain at
``nu >= 0.499`` the radial stress error of the standard Q4 reaches 14 %-155 % and
the sweep could quantify the locking but not cure it.

This module supplies the missing mitigation: the classical mean-dilatation
("B-bar", Nagtegaal et al. 1974; Hughes 1980) Q4, in which the **volumetric**
part of the strain-displacement operator is replaced by its element average

``Bv = B[0] + B[1]``,  ``Bv_bar = int Bv dA / int dA``,

``B_bar[0] = B[0] - Bv/2 + Bv_bar/2``, ``B_bar[1] = B[1] - Bv/2 + Bv_bar/2``,
``B_bar[2] = B[2]``,

so that the dilatation ``eps_xx + eps_yy`` is constant per element while the
deviatoric (shear) response keeps full 2x2 Gauss integration.  Consequences that
matter for verification:

* for a **constant strain field** ``Bv_bar == Bv`` pointwise, hence ``B_bar == B``
  and the element reproduces the standard element exactly (patch consistency);
* the volumetric constraint count drops from four independent constraints per
  element (2x2 Gauss) to one, which is what removes the over-stiffening;
* the deviatoric part is *not* under-integrated, so no hourglass zero-energy mode
  is introduced (a single element keeps exactly its three rigid-body modes).

Plane stress is supported for completeness, but note that locking is a
plane-strain / 3-D phenomenon: at ``nu -> 1/2`` the plane-stress bulk modulus is
finite, so the standard Q4 does not lock there.

The solver path mirrors :func:`tensorfem.continuum.solve_continuum`
element-for-element: same Dirichlet handling, same free/fixed partitioning, same
centre-point stress recovery.  That is deliberate - the comparison in
:mod:`tensorfem.plane_strain_locking` must differ **only** in the element
operator, never in the boundary-condition or load code.
"""
from __future__ import annotations

import math

import torch

from .continuum import ContinuumModel, ContinuumResult, _b_matrix, elasticity_matrix

_GAUSS = 1.0 / math.sqrt(3.0)
_GAUSS_POINTS = ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS))


def _gradients(coordinates: torch.Tensor, xi: float, eta: float):
    """Shape-function gradients and Jacobian determinants at one Gauss point."""
    natural = coordinates.new_tensor([
        [-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
        [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)],
    ]) / 4.0
    jacobian = torch.einsum("eia,ib->eab", coordinates, natural)
    determinant = torch.linalg.det(jacobian)
    if torch.any(determinant <= 0):
        raise ValueError("Q4 element has non-positive Jacobian")
    derivatives = torch.einsum("ib,ebc->eic", natural, torch.linalg.inv(jacobian))
    return derivatives, determinant


def averaged_dilatation_row(coordinates: torch.Tensor) -> torch.Tensor:
    """Element-averaged dilatation row ``Bv_bar`` (shape ``[element, 2*n_node]``).

    The average is area weighted (unit out-of-plane thickness, so the thickness
    cancels).  For a constant-strain field it reproduces ``Bv`` exactly.
    """
    numerator = None
    denominator = None
    for xi, eta in _GAUSS_POINTS:
        derivatives, determinant = _gradients(coordinates, xi, eta)
        b = _b_matrix(derivatives)
        row = b[:, 0, :] + b[:, 1, :]
        weighted = row * determinant[:, None]
        numerator = weighted if numerator is None else numerator + weighted
        denominator = determinant if denominator is None else denominator + determinant
    return numerator / denominator[:, None]


def _bar_b_matrix(b: torch.Tensor, dilatation: torch.Tensor, averaged: torch.Tensor) -> torch.Tensor:
    """Replace the volumetric part of ``b`` by the mean dilatation ``averaged``."""
    bar = b.clone()
    correction = 0.5 * (averaged - dilatation)
    bar[:, 0, :] = b[:, 0, :] + correction
    bar[:, 1, :] = b[:, 1, :] + correction
    return bar


def q4_bbar_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor,
                      thickness: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-dilatation Q4 stiffness and the modified centre ``B`` matrix.

    ``coordinates`` is ``[element, 4, 2]`` in counter-clockwise order,
    ``dmat`` is ``[element, 3, 3]`` and ``thickness`` is ``[element]``.  The
    returned centre matrix is the B-bar operator evaluated at the element centre,
    so the stress recovered from it is consistent with the stiffness actually
    assembled (constant dilatation per element).
    """
    ne = coordinates.shape[0]
    averaged = averaged_dilatation_row(coordinates)
    stiffness = torch.zeros((ne, 8, 8), dtype=coordinates.dtype, device=coordinates.device)
    for xi, eta in _GAUSS_POINTS:
        derivatives, determinant = _gradients(coordinates, xi, eta)
        b = _b_matrix(derivatives)
        bar = _bar_b_matrix(b, b[:, 0, :] + b[:, 1, :], averaged)
        stiffness = stiffness + (bar.transpose(1, 2) @ dmat @ bar) * (determinant * thickness)[:, None, None]
    centre_derivatives, _ = _gradients(coordinates, 0.0, 0.0)
    centre = _b_matrix(centre_derivatives)
    centre_bar = _bar_b_matrix(centre, centre[:, 0, :] + centre[:, 1, :], averaged)
    return stiffness, centre_bar


def solve_continuum_bbar(model: ContinuumModel) -> ContinuumResult:
    """Solve a 2-D continuum problem with the mean-dilatation Q4.

    Only ``element_type='q4'`` is supported (``cst`` has a constant strain field,
    for which B-bar is identical to the standard element).  Boundary conditions,
    prescribed values and load handling follow
    :func:`tensorfem.continuum.solve_continuum` line for line so that the two
    solves can be compared without any other modelling difference.
    """
    if model.element_type.lower() != "q4":
        raise ValueError("continuum_bbar supports element_type='q4' only")
    ne, nn = model.elements.shape
    expand = lambda x: x.to(dtype=model.nodes.dtype, device=model.nodes.device).reshape(-1).expand(ne)
    young, poisson, thickness = map(expand, (model.young_modulus, model.poisson_ratio, model.thickness))
    dmat = elasticity_matrix(young, poisson, model.plane)
    coordinates = model.nodes[model.elements.long()]
    ke, b = q4_bbar_stiffness(coordinates, dmat, thickness)
    edofs = torch.stack(tuple(2 * model.elements + i for i in (0, 1)), 2).reshape(ne, 2 * nn).long()
    rows = edofs[:, :, None].expand(-1, -1, 2 * nn).reshape(-1)
    cols = edofs[:, None, :].expand(-1, 2 * nn, -1).reshape(-1)
    stiffness = torch.zeros((model.n_dofs, model.n_dofs), dtype=model.nodes.dtype, device=model.nodes.device)
    stiffness = stiffness.index_put((rows, cols), ke.reshape(-1), accumulate=True)
    fixed = model.fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    values = (torch.zeros_like(fixed, dtype=model.nodes.dtype) if model.prescribed_values is None
              else model.prescribed_values.to(dtype=model.nodes.dtype, device=model.nodes.device))
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.arange(model.n_dofs, device=model.nodes.device)[mask]
    forces = model.forces.to(dtype=model.nodes.dtype, device=model.nodes.device)
    rhs = forces[free] - stiffness[free][:, fixed] @ values
    displacement = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    displacement = displacement.index_put((fixed,), values).index_put(
        (free,), torch.linalg.solve(stiffness[free][:, free], rhs))
    reaction = stiffness @ displacement - forces
    strain = torch.einsum("eij,ej->ei", b, displacement[edofs])
    stress = torch.einsum("eij,ej->ei", dmat, strain)
    return ContinuumResult(displacement, reaction, strain, stress,
                           0.5 * displacement @ stiffness @ displacement, stiffness)


def volumetric_constraint_rank(nodes: torch.Tensor, elements: torch.Tensor, *,
                               formulation: str = "bbar") -> int:
    """Global rank of the assembled dilatation operator (the locking mechanism).

    ``formulation='bbar'`` imposes one averaged dilatation constraint per element,
    ``formulation='standard'`` imposes one per Gauss point (four per Q4).  The
    rank is the number of *independent* volumetric constraints; when it approaches
    the number of displacement degrees-of-freedom the admissible displacement
    space collapses towards the rigid-body modes, which is exactly locking.  The
    rows are scattered into global degree-of-freedom columns before the rank is
    taken, so the count is a property of the mesh, not of one element.
    """
    if formulation not in ("standard", "bbar"):
        raise ValueError("formulation must be 'standard' or 'bbar'")
    ne, nn = elements.shape
    n_dofs = 2 * len(nodes)
    edofs = torch.stack(tuple(2 * elements + i for i in (0, 1)), 2).reshape(ne, 2 * nn).long()
    coordinates = nodes[elements.long()]
    if formulation == "bbar":
        local_rows = [averaged_dilatation_row(coordinates)]
    else:
        local_rows = []
        for xi, eta in _GAUSS_POINTS:
            derivatives, _ = _gradients(coordinates, xi, eta)
            b = _b_matrix(derivatives)
            local_rows.append(b[:, 0, :] + b[:, 1, :])
    assembly = torch.cat(local_rows, dim=0)
    element_of_row = torch.arange(len(assembly)) % ne
    operator = torch.zeros((len(assembly), n_dofs), dtype=nodes.dtype, device=nodes.device)
    operator = operator.index_put((torch.arange(len(assembly))[:, None].expand_as(edofs[element_of_row]),
                                   edofs[element_of_row]), assembly, accumulate=True)
    return int(torch.linalg.matrix_rank(operator))
