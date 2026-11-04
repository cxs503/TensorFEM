"""Locking-free volumetric treatment for the 2-D bilinear quad (Q4).

:func:`tensorfem.continuum.q4_stiffness` is a fully integrated Q4.  Near the
incompressible limit (``nu -> 1/2``) the dilatational part of its strain field
over-stiffens the element ("volumetric locking") and the displacement error grows
without bound under mesh refinement.

This module adds the two classical cures as *element-level* kernels, mirroring the
3-D B-bar implementation in :mod:`tensorfem.solid3d_bbar`:

``bbar``
    Hughes' B-bar projection: the dilatational part of the strain-displacement
    matrix is replaced by its volume average (constant per element), while the
    deviatoric part keeps full 2x2 Gauss integration.  No hourglass modes, because
    the deviatoric part is still fully integrated.

``sri``
    Selective reduced integration: the deviatoric part is integrated with 2x2
    Gauss, the dilatational part with the single centre point.  An independent
    route to the same goal, used here as a cross-check on ``bbar``.

Both operate on batches of elements and return the element stiffness together
with the *modified* strain-displacement matrix evaluated at the element centre, so
that strain/stress recovery is consistent with the projected field.

Conventions match :mod:`tensorfem.continuum`: engineering strain
``[eps_x, eps_y, gamma_xy]``, plane strain with ``eps_z = 0``, so the volumetric
strain is ``eps_x + eps_y`` and the volumetric selector is ``m = [1, 1, 0]``
with ``m . m = 2``.
"""
from __future__ import annotations

import torch

FORMULATIONS = ("standard", "bbar", "sri")

_GAUSS = 1.0 / 3.0**0.5
#: 2x2 full Gauss rule, weight 1 each in the reference square
_FULL_POINTS = ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS))
#: single-point rule (centre), weight 4
_REDUCED_POINTS = ((0.0, 0.0),)


def _natural_derivatives(xi: float, eta: float, dtype, device) -> torch.Tensor:
    """dN/dxi, dN/deta of the bilinear quad, shape ``[4, 2]``."""
    return torch.tensor(
        [[-(1 - eta), -(1 - xi)],
         [(1 - eta), -(1 + xi)],
         [(1 + eta), (1 + xi)],
         [-(1 + eta), (1 - xi)]],
        dtype=dtype, device=device) / 4.0


def _gradient(coordinates: torch.Tensor, xi: float, eta: float):
    """Shape-function gradients and Jacobian determinant at ``(xi, eta)``."""
    natural = _natural_derivatives(xi, eta, coordinates.dtype, coordinates.device)
    jacobian = torch.einsum("eia,ib->eab", coordinates, natural)
    det = torch.linalg.det(jacobian)
    if torch.any(det <= 0):
        raise ValueError("Q4 element has non-positive Jacobian")
    return torch.einsum("ib,ebc->eic", natural, torch.linalg.inv(jacobian)), det


def _b_matrix(derivatives: torch.Tensor) -> torch.Tensor:
    ne, nn, _ = derivatives.shape
    b = torch.zeros((ne, 3, 2 * nn), dtype=derivatives.dtype, device=derivatives.device)
    b[:, 0, 0::2] = derivatives[:, :, 0]
    b[:, 1, 1::2] = derivatives[:, :, 1]
    b[:, 2, 0::2] = derivatives[:, :, 1]
    b[:, 2, 1::2] = derivatives[:, :, 0]
    return b


def _volumetric_part(b: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
    """Dilatational component ``m (m^T B) / (m^T m)`` of a strain-displacement matrix."""
    m = selector.reshape(1, 3, 1)
    return m * (torch.einsum("i,eij->ej", selector, b)[:, None, :] / float(selector @ selector))


def q4_bbar_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor, thickness: torch.Tensor):
    """B-bar Q4 stiffness for plane stress / plane strain.

    ``coordinates`` ``[n_elem, 4, 2]``, ``dmat`` ``[n_elem, 3, 3]``,
    ``thickness`` ``[n_elem]``.  Returns ``(ke, b_centroid)`` where
    ``b_centroid`` is the projected ``B`` evaluated at the element centre.
    """
    selector = coordinates.new_tensor([1.0, 1.0, 0.0])
    ne = len(coordinates)
    stiffness = torch.zeros((ne, 8, 8), dtype=coordinates.dtype, device=coordinates.device)
    volume = torch.zeros(ne, dtype=coordinates.dtype, device=coordinates.device)
    dilatational_integral = torch.zeros((ne, 3, 8), dtype=coordinates.dtype, device=coordinates.device)
    points = []
    for xi, eta in _FULL_POINTS:
        deriv, det = _gradient(coordinates, xi, eta)
        weight = det * thickness
        b = _b_matrix(deriv)
        dilatational = _volumetric_part(b, selector)
        points.append((b, dilatational, weight))
        volume = volume + weight
        dilatational_integral = dilatational_integral + dilatational * weight[:, None, None]
    mean_dilatational = dilatational_integral / volume[:, None, None]
    for b, dilatational, weight in points:
        b_bar = b - dilatational + mean_dilatational
        stiffness = stiffness + (b_bar.transpose(1, 2) @ dmat @ b_bar) * weight[:, None, None]
    deriv, _ = _gradient(coordinates, 0.0, 0.0)
    b_centre = _b_matrix(deriv)
    b_centroid = b_centre - _volumetric_part(b_centre, selector) + mean_dilatational
    return stiffness, b_centroid


def q4_sri_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor, thickness: torch.Tensor):
    """Selective reduced integration Q4 stiffness (2x2 deviatoric + 1-point volumetric).

    Returns ``(ke, b_centroid)`` with the same convention as
    :func:`q4_bbar_stiffness`; the centre ``B`` uses the centre dilatational part,
    which is what the single-point rule actually integrates.
    """
    selector = coordinates.new_tensor([1.0, 1.0, 0.0])
    ne = len(coordinates)
    stiffness = torch.zeros((ne, 8, 8), dtype=coordinates.dtype, device=coordinates.device)
    for xi, eta in _FULL_POINTS:
        deriv, det = _gradient(coordinates, xi, eta)
        b = _b_matrix(deriv)
        deviatoric = b - _volumetric_part(b, selector)
        stiffness = stiffness + (deviatoric.transpose(1, 2) @ dmat @ deviatoric) * (det * thickness)[:, None, None]
    deriv_c, det_c = _gradient(coordinates, 0.0, 0.0)
    b_centre = _b_matrix(deriv_c)
    dilatational = _volumetric_part(b_centre, selector)
    stiffness = stiffness + (dilatational.transpose(1, 2) @ dmat @ dilatational) * (4.0 * det_c * thickness)[:, None, None]
    return stiffness, b_centre


def q4_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor, thickness: torch.Tensor,
                 formulation: str = "standard"):
    """Dispatch to the requested Q4 volumetric treatment."""
    if formulation == "standard":
        from .continuum import q4_stiffness as standard

        return standard(coordinates, dmat, thickness)
    if formulation == "bbar":
        return q4_bbar_stiffness(coordinates, dmat, thickness)
    if formulation == "sri":
        return q4_sri_stiffness(coordinates, dmat, thickness)
    raise ValueError(f"formulation must be one of {FORMULATIONS}, got {formulation!r}")