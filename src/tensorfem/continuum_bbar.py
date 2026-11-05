"""Locking-resistant plane-strain Q4: the B-bar (mean dilatation) element.

The merged 2-D element set in :mod:`tensorfem.continuum` has a single Q4 path
(fully integrated, displacement formulation).  Under plane strain that path
locks as ``nu -> 1/2``: :func:`tensorfem.incompressible_locking.plane_strain_sweep`
records 14.0 % / 35.5 % (``nu = 0.499``) and 61.9 % / 155.2 % (``nu = 0.4999``)
L2 errors on displacement / radial stress, with no mitigation available in the
repository.  Marine structures that run on the plane-strain path -- rubber
bearings, near-incompressible coatings, composite matrix coupons -- therefore had
no usable element in that limit.

This module adds the missing formulation.  It is the 2-D counterpart of
:mod:`tensorfem.solid3d_bbar`: the dilatational part of the strain-displacement
operator is replaced by its element mean, while the deviatoric part is evaluated
at every integration point, so no hourglass mode is introduced.  With

``kappa = E / (3 (1 - 2 nu))``  (3-D bulk modulus, exact for plane strain),

``m = [1, 1, 0]^T``, the plane-strain constitutive law is split as
``D = D_dev + kappa m m^T`` with ``D_dev = D - kappa m m^T``.  Writing
``b_theta = m^T B`` for the dilatation row of the strain-displacement matrix and
``b_bar_theta = (1/A) int b_theta dA`` for its element mean,

``K = int B^T D_dev B dA + kappa A b_bar_theta^T b_bar_theta``.

The stress is recovered at the element centroid from
``B_bar = B - m (b_theta)/2 + m (b_bar_theta)/2``, the same projection the 3-D
B-bar element uses, so the 2-D results are directly comparable with the
``solid3d_bbar`` qualification.

Plane strain only: the volumetric split above uses the 3-D bulk modulus, which
is the correct plane-strain limit.  For ``plane="stress"`` the thickness strain
is not zero and this split is not the consistent one, so the formulation refuses
the call instead of silently returning a different model.
"""
from __future__ import annotations

import torch

from .continuum import (ContinuumModel, ContinuumResult, _b_matrix, elasticity_matrix)


def _expand(values: torch.Tensor, dtype: torch.dtype, device: torch.device, ne: int):
    return values.to(dtype=dtype, device=device).reshape(-1).expand(ne)


def _natural_derivatives(xi: float, eta: float, coordinates: torch.Tensor) -> torch.Tensor:
    """Shape-function derivatives w.r.t. the natural coordinates at ``(xi, eta)``."""
    return coordinates.new_tensor([
        [-(1 - eta), -(1 - xi)],
        [(1 - eta), -(1 + xi)],
        [(1 + eta), (1 + xi)],
        [-(1 + eta), (1 - xi)],
    ]) / 4


def _gradient(coordinates: torch.Tensor, xi: float, eta: float):
    """Cartesian derivatives, Jacobian determinant and B matrix at ``(xi, eta)``."""
    natural = _natural_derivatives(xi, eta, coordinates)
    jacobian = torch.einsum("eia,ib->eab", coordinates, natural)
    determinant = torch.linalg.det(jacobian)
    if torch.any(determinant <= 0):
        raise ValueError("Q4 element has non-positive Jacobian")
    derivatives = torch.einsum("ib,ebc->eic", natural, torch.linalg.inv(jacobian))
    return _b_matrix(derivatives), determinant


def q4_bbar_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor, thickness: torch.Tensor):
    """B-bar Q4 stiffness and centroid B-bar matrix for plane strain.

    ``coordinates`` is ``[n_element, 4, 2]``, ``dmat`` is ``[n_element, 3, 3]``
    (plane-strain elasticity matrix) and ``thickness`` is ``[n_element]``.  The
    returned ``B`` matrix is the mean-dilatation operator used for stress
    recovery, shaped ``[n_element, 3, 8]``.
    """
    if coordinates.ndim != 3 or coordinates.shape[1:] != (4, 2):
        raise ValueError("coordinates must have shape [n_element, 4, 2]")
    ne = len(coordinates)
    g = 1.0 / 3.0**0.5
    points = ((-g, -g), (g, -g), (g, g), (-g, g))
    m = coordinates.new_tensor([1.0, 1.0, 0.0])
    # The bulk modulus is not stored in ``dmat``; it follows exactly from two
    # entries of the plane-strain matrix: D[0, 1] = kappa - 2G/3 and
    # D[2, 2] = G, hence kappa = D[0, 1] + 2 D[2, 2] / 3 = E / (3 (1 - 2 nu)).
    kappa = dmat[:, 0, 1] + 2.0 * dmat[:, 2, 2] / 3.0
    volumetric = kappa[:, None, None] * (m[None, :, None] * m[None, None, :])
    deviatoric = dmat - volumetric
    stiffness = torch.zeros((ne, 8, 8), dtype=coordinates.dtype, device=coordinates.device)
    dilatation_integral = dmat.new_zeros((ne, 8))
    area = dmat.new_zeros(ne)
    for xi, eta in points:
        b, determinant = _gradient(coordinates, xi, eta)
        weight = (determinant * thickness)[:, None, None]
        stiffness = stiffness + (b.transpose(1, 2) @ deviatoric @ b) * weight
        dilatation_integral = dilatation_integral + (b[:, 0, :] + b[:, 1, :]) * weight[:, :, 0]
        area = area + determinant * thickness
    bbar_theta = dilatation_integral / area[:, None]
    stiffness = stiffness + kappa[:, None, None] * (
        area[:, None, None] * (bbar_theta[:, :, None] * bbar_theta[:, None, :]))
    b_centre, _ = _gradient(coordinates, 0.0, 0.0)
    b_theta_centre = b_centre[:, 0, :] + b_centre[:, 1, :]
    bbar = (b_centre - m[None, :, None] * (b_theta_centre / 2.0)[:, None, :]
            + m[None, :, None] * (bbar_theta / 2.0)[:, None, :])
    return stiffness, bbar


def solve_continuum_bbar(model: ContinuumModel) -> ContinuumResult:
    """Solve an existing :class:`ContinuumModel` with B-bar Q4 plane strain."""
    if model.element_type.lower() != "q4":
        raise ValueError("B-bar formulation is implemented for Q4 only")
    if model.plane != "strain":
        raise ValueError("B-bar Q4 is qualified for plane strain only; "
                         "plane stress needs a different volumetric split")
    ne, nn = model.elements.shape
    dtype, device = model.nodes.dtype, model.nodes.device
    young = _expand(model.young_modulus, dtype, device, ne)
    poisson = _expand(model.poisson_ratio, dtype, device, ne)
    thickness = _expand(model.thickness, dtype, device, ne)
    dmat = elasticity_matrix(young, poisson, "strain")
    coordinates = model.nodes[model.elements.long()]
    stiffness_e, b = q4_bbar_stiffness(coordinates, dmat, thickness)
    edofs = torch.stack(tuple(2 * model.elements + i for i in (0, 1)), 2).reshape(ne, 2 * nn).long()
    rows = edofs[:, :, None].expand(-1, -1, 2 * nn).reshape(-1)
    cols = edofs[:, None, :].expand(-1, 2 * nn, -1).reshape(-1)
    n_dofs = model.n_dofs
    stiffness = torch.zeros((n_dofs, n_dofs), dtype=dtype, device=device)
    stiffness = stiffness.index_put((rows, cols), stiffness_e.reshape(-1), accumulate=True)
    fixed = model.fixed_dofs.to(device=device, dtype=torch.long)
    values = (torch.zeros_like(fixed, dtype=dtype, device=device)
              if model.prescribed_values is None
              else model.prescribed_values.to(dtype=dtype, device=device))
    mask = torch.ones(n_dofs, dtype=torch.bool, device=device)
    mask[fixed] = False
    free = torch.arange(n_dofs, device=device)[mask]
    forces = model.forces.to(dtype=dtype, device=device)
    u = torch.zeros(n_dofs, dtype=dtype, device=device).index_put((fixed,), values)
    if free.numel():
        rhs = forces[free] - stiffness[free][:, fixed] @ values
        u = u.index_put((free,), torch.linalg.solve(stiffness[free][:, free], rhs))
    strain = torch.einsum("eij,ej->ei", b, u[edofs])
    stress = torch.einsum("eij,ej->ei", dmat, strain)
    return ContinuumResult(u, stiffness @ u - forces, strain, stress, 0.5 * u @ stiffness @ u, stiffness)


def bbar_free_element_eigenvalues(nodes, young=1.0, poisson=0.4999):
    """Eigenvalues of one free B-bar Q4 element (three must be zero: rigid body)."""
    coordinates = nodes.reshape(1, 4, 2)
    dmat = elasticity_matrix(coordinates.new_tensor([young]), coordinates.new_tensor([poisson]), "strain")
    thickness = coordinates.new_tensor([1.0])
    stiffness, _ = q4_bbar_stiffness(coordinates, dmat, thickness)
    symmetric = (stiffness[0] + stiffness[0].T) / 2.0
    return torch.linalg.eigvalsh(symmetric)