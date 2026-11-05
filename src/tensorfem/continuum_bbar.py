"""Locking-resistant plane-strain Q4: the B-bar (mean dilatation) element.

Two independently developed qualifications of the 2-D near-incompressible debt
were merged here (TensorPlatform mission m3364, three-way union):

* the plane-strain B-bar element with the centroid stress-recovery operator and
  the free-element eigen-spectrum check (``q4_bbar_stiffness``,
  ``solve_continuum_bbar``, ``bbar_free_element_eigenvalues``),
* the plane-stress/sweep diagnostics built on the same element operator
  (``averaged_dilatation_row``, ``_bar_b_matrix``,
  ``volumetric_constraint_rank``).

Both share a single element kernel -- :func:`q4_bbar_stiffness` -- so the
qualification numbers of the two reports are produced by one code path.

Why this module exists
----------------------
The merged 2-D element set in :mod:`tensorfem.continuum` has a single Q4 path
(fully integrated, displacement formulation).  Under plane strain that path
locks as ``nu -> 1/2``: the plane-strain sweep records 14.0 % / 35.5 %
(``nu = 0.499``) and 61.9 % / 155.2 % (``nu = 0.4999``) L2 errors on
displacement / radial stress, with no mitigation available elsewhere in the
repository.  Marine structures that run on the plane-strain path -- rubber
bearings, near-incompressible coatings, composite matrix coupons -- therefore
had no usable element in that limit.

This module adds the missing formulation.  It is the 2-D counterpart of
:mod:`tensorfem.solid3d_bbar`: the dilatational part of the strain-displacement
operator is replaced by its element mean, while the deviatoric part is evaluated
at every integration point, so no hourglass mode is introduced.  With

``kappa = E / (3 (1 - 2 nu))``  (3-D bulk modulus, exact for plane strain),

``m = [1, 1, 0]^T``, the plane-strain constitutive law is split as
``D = D_dev + kappa m m^T`` with ``D_dev = D - kappa m m^T``.  Writing
``b_theta = m^T B`` for the dilatation row of the strain-displacement matrix and
``b_bar_theta = (1/A) int b_theta dA`` for its element mean,

``K = int B^T D_dev B dA + kappa A b_bar_theta^T b_bar_theta``,

which is the classical mean-dilatation ("B-bar", Nagtegaal et al. 1974;
Hughes 1980) element.  Consequences that matter for verification:

* for a **constant strain field** ``b_bar_theta == b_theta`` pointwise, hence
  ``B_bar == B`` and the element reproduces the standard element exactly
  (patch consistency);
* the volumetric constraint count drops from four independent constraints per
  element (2x2 Gauss) to one, which is what removes the over-stiffening;
* the deviatoric part is *not* under-integrated, so no hourglass zero-energy
  mode is introduced (a single element keeps exactly its three rigid-body
  modes).

The stress is recovered at the element centroid from
``B_bar = B - m (b_theta)/2 + m (b_bar_theta)/2``, the same projection the 3-D
B-bar element uses, so the 2-D results are directly comparable with the
``solid3d_bbar`` qualification.

Element operator, in the notation of the original derivation
--------------------------------------------------------------
Writing ``Bv = B[0] + B[1]`` for the dilatation row at a Gauss point and
``Bv_bar = int Bv dA / int dA`` for its area mean, the mean-dilatation operator is

``B_bar[0] = B[0] - Bv/2 + Bv_bar/2``,
``B_bar[1] = B[1] - Bv/2 + Bv_bar/2``,
``B_bar[2] = B[2]``,

so the dilatation ``eps_xx + eps_yy`` is constant per element while the shear
response keeps full 2x2 Gauss integration.  For a constant strain field
``Bv_bar == Bv`` pointwise, hence ``B_bar == B``: the element then reproduces the
standard Q4 exactly, which is what the patch test in
:mod:`tensorfem.plane_strain_locking` asserts.

API shapes
----------
``averaged_dilatation_row`` takes ``[element, 4, 2]`` coordinates and returns the
``[element, 2*n_node]`` mean row; ``q4_bbar_stiffness`` takes coordinates
``[element, 4, 2]``, ``dmat`` ``[element, 3, 3]`` and ``thickness`` ``[element]``
and returns ``[element, 8, 8]`` plus the ``[element, 3, 8]`` centroid operator.
Only ``element_type='q4'`` is supported (``cst`` already has a constant strain
field, for which B-bar is identical to the standard element, so accepting it
would silently relabel the standard element as a cure.

Where the standard element fails: at ``nu >= 0.499`` the plane-strain Q4 records
14 %-155 % radial-stress error in :mod:`tensorfem.incompressible_locking`, which
is the debt this element closes.

Reported in
:doc:`plane-strain-bbar-locking` (plane-strain qualification, 10 gates) and
:doc:`plane-strain-locking-bbar` (formulation sweep, mechanism and refinement
study); the shared open debts are tracked in
:doc:`near-incompressible-locking`.

Scope
-----
Plane strain is the qualified path: the volumetric split above uses the 3-D bulk
modulus, which is the correct plane-strain limit.  For ``plane="stress"`` the
thickness strain is not zero and this split is not the consistent one, so
:func:`solve_continuum_bbar` refuses the call instead of silently returning a
different model.  The plane-stress operator is still reachable for completeness
work through the explicit ``allow_plane_stress=True`` opt-in (locking is a
plane-strain phenomenon: at ``nu -> 1/2`` the plane-stress bulk modulus is
finite, so the standard Q4 does not lock there and the opt-in path is a
sanity-checked, unqualified operator rather than a cured one).
"""
from __future__ import annotations

import torch

from .continuum import (ContinuumModel, ContinuumResult, _b_matrix, elasticity_matrix)

_GAUSS = 1.0 / 3.0 ** 0.5
_GAUSS_POINTS = ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS))


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


def averaged_dilatation_row(coordinates: torch.Tensor) -> torch.Tensor:
    """Element-averaged dilatation row ``Bv_bar`` (shape ``[element, 2*n_node]``).

    The average is area weighted (unit out-of-plane thickness, so the thickness
    cancels).  For a constant-strain field it reproduces ``Bv`` exactly, which is
    why the B-bar element passes the patch test unchanged.
    """
    numerator = None
    denominator = None
    for xi, eta in _GAUSS_POINTS:
        b, determinant = _gradient(coordinates, xi, eta)
        row = b[:, 0, :] + b[:, 1, :]
        weighted = row * determinant[:, None]
        numerator = weighted if numerator is None else numerator + weighted
        denominator = determinant if denominator is None else denominator + determinant
    return numerator / denominator[:, None]


def _bar_b_matrix(b: torch.Tensor, dilatation: torch.Tensor,
                  averaged: torch.Tensor) -> torch.Tensor:
    """Replace the volumetric part of ``b`` by the mean dilatation ``averaged``."""
    bar = b.clone()
    correction = 0.5 * (averaged - dilatation)
    bar[:, 0, :] = b[:, 0, :] + correction
    bar[:, 1, :] = b[:, 1, :] + correction
    return bar


def q4_bbar_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor, thickness: torch.Tensor):
    """B-bar Q4 stiffness and centroid B-bar matrix for plane strain.

    ``coordinates`` is ``[n_element, 4, 2]`` (counter-clockwise), ``dmat`` is
    ``[n_element, 3, 3]`` (plane-strain elasticity matrix) and ``thickness`` is
    ``[n_element]``.  The returned ``B`` matrix is the mean-dilatation operator
    used for stress recovery, shaped ``[n_element, 3, 8]``.

    The purely deviatoric part of ``dmat`` is integrated at the four Gauss points
    and the volumetric part carries the element-averaged dilatation, i.e. the
    classical ``K = int B^T D_dev B dA + kappa A b_bar_theta^T b_bar_theta``.
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


def solve_continuum_bbar(model: ContinuumModel, *,
                         allow_plane_stress: bool = False) -> ContinuumResult:
    """Solve an existing :class:`ContinuumModel` with B-bar Q4 plane strain.

    Boundary conditions, prescribed values and load handling mirror
    :func:`tensorfem.continuum.solve_continuum` line for line so that the two
    solves can be compared without any other modelling difference.

    ``plane="stress"`` is refused by default because the volumetric split above
    is the plane-strain one, and a silent substitution of a different model is
    exactly the failure mode this qualification exists to prevent.  Pass
    ``allow_plane_stress=True`` to exercise the operator outside its
    qualification envelope; the returned field is then *not* covered by the
    plane-strain gates.
    """
    if model.element_type.lower() != "q4":
        raise ValueError("B-bar formulation is implemented for Q4 only")
    if model.plane != "strain" and not allow_plane_stress:
        raise ValueError("B-bar Q4 is qualified for plane strain only; "
                         "plane stress needs a different volumetric split "
                         "(pass allow_plane_stress=True to bypass the gate)")
    ne, nn = model.elements.shape
    dtype, device = model.nodes.dtype, model.nodes.device
    young = _expand(model.young_modulus, dtype, device, ne)
    poisson = _expand(model.poisson_ratio, dtype, device, ne)
    thickness = _expand(model.thickness, dtype, device, ne)
    # ``model.plane`` is ``"strain"`` on the qualified path; the plane-stress
    # opt-in solves the declared model instead of silently substituting strain.
    dmat = elasticity_matrix(young, poisson, model.plane)
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
            b, _ = _gradient(coordinates, xi, eta)
            local_rows.append(b[:, 0, :] + b[:, 1, :])
    assembly = torch.cat(local_rows, dim=0)
    element_of_row = torch.arange(len(assembly)) % ne
    operator = torch.zeros((len(assembly), n_dofs), dtype=nodes.dtype, device=nodes.device)
    operator = operator.index_put((torch.arange(len(assembly))[:, None].expand_as(edofs[element_of_row]),
                                   edofs[element_of_row]), assembly, accumulate=True)
    return int(torch.linalg.matrix_rank(operator))