"""Surface stress recovery for 3-D brick fields (Gauss-point extrapolation).

Why this module exists
----------------------
An isoparametric brick computes its stress at the Gauss points that integrate it:
2x2x2 for the fully integrated ``HEX8``, 3x3x3 for the quadratic ``HEX20``.  The
Gauss points of a brick sit strictly inside the element, so the largest value a
*centroid* or a Gauss-point sample can report is an interior value.  Every load
case whose maximum lives on the boundary (torsion, contact, surface pressure,
free-edge stress concentration) is therefore systematically under-reported, by
an amount that grows as the boundary layer is less resolved.  Nitpicking that
gap by taking the centroid maximum and calling it "the peak" is wrong in
principle: for the Saint-Venant torsion prism of the accompanying benchmark the
centroid maximum is about ``37%`` below the exact surface peak, i.e. far outside
any accuracy gate.

The standard remedy is the extrapolation operator of Zienkiewicz & Zhu: the
Gauss-point values are extrapolated to the element nodes by the tensor product
of 1-D Lagrange polynomials, and the nodal contributions of all incident
elements are averaged.  This module implements exactly that operator for
``HEX8`` and ``HEX20`` plus the face-traction projection needed to read the
shear on a free surface.  Nothing here is torsion-specific: the caller passes a
mesh, a converged displacement field and an elasticity matrix.

Accuracy of the operator
------------------------
Both extrapolation matrices are single-element objects and their exactness is
stated in testable form:

* ``HEX8`` (2-point Gauss per direction) extrapolates polynomials up to degree
  **1** per natural coordinate exactly; a quadratic field is *not* reproduced.
* ``HEX20`` (3-point Gauss per direction) extrapolates polynomials up to degree
  **2** per natural coordinate exactly.

Nodal averaging then reproduces a constant stress field exactly on a distorted
mesh (the patch test of the recovery), and reproduces a uniformly varying field
to the order above.  ``tests/test_surface_stress_recovery.py`` asserts both,
along with the negative control that the ``HEX8`` operator fails a quadratic
field while the ``HEX20`` one passes it.  The recovered stress at a free surface
carries a spurious normal component (the finite-element field only satisfies the
traction-free condition weakly); ``surface_traction`` returns it explicitly so a
caller can decide whether the recovered peak is trustworthy instead of silently
reporting the tangential maximum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .hex20 import _gradient as _hex20_gradient
from .hex20 import hex20_shape_gradients
from .solid3d import _b, _hex_gradient

__all__ = [
    "NodalStress",
    "VoigtIndex",
    "extrapolation_matrix",
    "extrapolation_weights_at",
    "gauss_points_1d",
    "natural_coordinates",
    "recover_nodal_stress",
    "surface_traction",
    "transverse_shear_magnitude",
    "voigt_stress_to_tensor",
]

#: Voigt order used by ``solid3d._b`` and ``elasticity_matrix_3d``.
VoigtIndex = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "xz": 5}

_HEX8_NATURAL = ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
                 (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0))

#: Edge list of the twenty-node serendipity brick, matching ``hex20._EDGES``.
_HEX20_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7))


def natural_coordinates(element: str, *, dtype=torch.float64) -> torch.Tensor:
    """Natural ``(xi, eta, zeta)`` coordinates of the element nodes, in node order."""
    key = element.lower()
    if key == "hex8":
        return torch.tensor(_HEX8_NATURAL, dtype=dtype)
    if key == "hex20":
        corners = _HEX8_NATURAL
        nodes = list(corners)
        for a, b in _HEX20_EDGES:
            nodes.append(tuple(0.5 * (corners[a][d] + corners[b][d]) for d in range(3)))
        return torch.tensor(nodes, dtype=dtype)
    raise ValueError("element must be 'hex8' or 'hex20'")


def gauss_points_1d(element: str) -> tuple[float, ...]:
    """The 1-D Gauss abscissae that the element family integrates with."""
    key = element.lower()
    if key == "hex8":
        q = 1.0 / math.sqrt(3.0)
        return (-q, q)
    if key == "hex20":
        q = math.sqrt(3.0 / 5.0)
        return (-q, 0.0, q)
    raise ValueError("element must be 'hex8' or 'hex20'")


def _lagrange(points: tuple[float, ...], target: float) -> tuple[float, ...]:
    """Lagrange basis values at ``target`` for the interpolation nodes ``points``."""
    values = []
    for j, xj in enumerate(points):
        weight = 1.0
        for m, xm in enumerate(points):
            if m != j:
                weight *= (target - xm) / (xj - xm)
        values.append(weight)
    return tuple(values)


def extrapolation_weights_at(element: str, targets, *, dtype=torch.float64) -> torch.Tensor:
    """``(n_targets, n_gauss)`` Lagrange weights that transport Gauss values to ``targets``.

    ``targets`` are natural coordinates ``(xi, eta, zeta)`` anywhere in the element
    (the nodes are the usual choice, but an interior point is equally valid: the
    operator is just polynomial extrapolation/interpolation of the Gauss values).
    Rows are ordered against the Gauss-point sequence ``(i, j, k)`` with ``k``
    running fastest, matching :func:`gauss_points_1d`.  Every row sums to one, so
    a constant field is reproduced exactly.
    """
    points = gauss_points_1d(element)
    targets = torch.as_tensor(targets, dtype=torch.float64).reshape(-1, 3).tolist()
    rows = []
    for xi, eta, zeta in targets:
        wx = _lagrange(points, xi)
        wy = _lagrange(points, eta)
        wz = _lagrange(points, zeta)
        rows.append([wx[i] * wy[j] * wz[k]
                     for i in range(len(points)) for j in range(len(points)) for k in range(len(points))])
    return torch.tensor(rows, dtype=dtype)


def extrapolation_matrix(element: str, *, dtype=torch.float64) -> torch.Tensor:
    """``(n_nodes, n_gauss)`` matrix mapping Gauss-point values to nodal values.

    Shorthand for :func:`extrapolation_weights_at` evaluated at the element nodes;
    this is the operator that :func:`recover_nodal_stress` applies.
    """
    return extrapolation_weights_at(element, natural_coordinates(element, dtype=torch.float64), dtype=dtype)


@dataclass(frozen=True)
class NodalStress:
    """Averaged nodal stress of a brick mesh.

    ``stress`` has shape ``(n_nodes, 6)`` in the Voigt order of
    :data:`VoigtIndex`; ``counts`` records how many elements contributed to each
    node, so a caller can mask out nodes that only one element touched.
    ``element_max_residual`` is the largest gap between the extrapolated nodal
    value of an element and the averaged value it contributed to, which exposes
    (without hiding it) the discontinuity of the raw element field.
    """

    stress: torch.Tensor
    counts: torch.Tensor
    element_max_residual: float


def _gauss_point_stress(x, u_e, modulus, element: str) -> torch.Tensor:
    """Stress at every Gauss point of every element, shape ``(ne, ng, 6)``."""
    points = gauss_points_1d(element)
    key = element.lower()
    out = []
    for r in points:
        for s in points:
            for t in points:
                if key == "hex8":
                    g, _ = _hex_gradient(x, r, s, t)
                else:
                    _, dn = hex20_shape_gradients(r, s, t, like=x)
                    g, _ = _hex20_gradient(x, dn)
                b = _b(g)
                strain = torch.einsum("eij,ej->ei", b, u_e)
                out.append(torch.einsum("eij,ej->ei", modulus, strain))
    return torch.stack(out, 1)


def recover_nodal_stress(nodes: torch.Tensor, cells: torch.Tensor, displacement: torch.Tensor,
                         modulus: torch.Tensor, element: str = "hex8") -> NodalStress:
    """Extrapolate Gauss-point stresses to the nodes and average over elements.

    Parameters
    ----------
    nodes : ``(n_nodes, 3)`` node coordinates.
    cells : ``(n_elements, nen)`` connectivity, ``nen = 8`` or ``20``, in the node
        order of ``hex20``/``solid3d`` (corners first, then edge midpoints).
    displacement : ``(3*n_nodes,)`` converged global displacement vector.
    modulus : ``(n_elements, 6, 6)`` element elasticity matrices.
    element : ``'hex8'`` or ``'hex20'``.

    Returns
    -------
    :class:`NodalStress` with the averaged nodal stress and contribution counts.
    """
    key = element.lower()
    expected = {"hex8": 8, "hex20": 20}.get(key)
    if expected is None:
        raise ValueError("element must be 'hex8' or 'hex20'")
    if cells.ndim != 2 or cells.shape[1] != expected:
        raise ValueError(f"{key} connectivity must have {expected} columns")
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape [n, 3]")
    displacement = displacement.reshape(-1)
    if displacement.numel() != 3 * len(nodes):
        raise ValueError("displacement must have 3*n_nodes entries")

    cells = cells.long()
    x = nodes[cells]
    edofs = torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(len(cells), 3 * expected)
    u_e = displacement[edofs]
    gauss = _gauss_point_stress(x, u_e, modulus, key)            # (ne, ng, 6)
    matrix = extrapolation_matrix(key, dtype=nodes.dtype)         # (nen, ng)
    nodal = torch.einsum("aq,eqi->eai", matrix, gauss)            # (ne, nen, 6)

    stress_sum = nodes.new_zeros((len(nodes), 6))
    counts = torch.zeros(len(nodes), dtype=torch.long, device=nodes.device)
    flat = cells.reshape(-1)
    stress_sum.index_add_(0, flat, nodal.reshape(-1, 6))
    counts.index_add_(0, flat, torch.ones_like(flat))
    averaged = stress_sum / counts.clamp_min(1).to(nodes.dtype)[:, None]
    residual = float((nodal - averaged[cells]).abs().max()) if len(cells) else 0.0
    return NodalStress(stress=averaged, counts=counts, element_max_residual=residual)


def voigt_stress_to_tensor(stress: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 6)`` Voigt stress to ``(..., 3, 3)`` Cauchy stress."""
    if stress.shape[-1] != 6:
        raise ValueError("Voigt stress must have 6 components in the last axis")
    s = stress
    return torch.stack((
        torch.stack((s[..., 0], s[..., 3], s[..., 5]), -1),
        torch.stack((s[..., 3], s[..., 1], s[..., 4]), -1),
        torch.stack((s[..., 5], s[..., 4], s[..., 2]), -1),
    ), -2)


def transverse_shear_magnitude(stress: torch.Tensor, axis: int = 2) -> torch.Tensor:
    """Magnitude of the shear vector acting on the plane normal to ``axis``.

    This is the quantity that the Saint-Venant torsion peak lives in: the shear
    carried by the section is ``(tau_zx, tau_zy)``, and its magnitude peaks on
    the lateral *boundary curve* while the traction *on* that free surface is
    identically zero (the nonzero components act on planes that are not parallel
    to the surface).  Reading a free-surface shear peak through
    :func:`surface_traction` therefore returns zero or noise, not the peak.

    ``axis = 2`` (default) gives ``|(tau_yz, tau_xz)|``, ``axis = 1`` gives
    ``|(tau_xy, tau_yz)|`` and ``axis = 0`` gives ``|(tau_xy, tau_xz)|``.
    """
    a = axis % 3
    pairs = {0: ("xy", "xz"), 1: ("xy", "yz"), 2: ("yz", "xz")}
    if stress.shape[-1] != 6:
        raise ValueError("Voigt stress must have 6 components in the last axis")
    first, second = pairs[a]
    return torch.hypot(stress[..., VoigtIndex[first]], stress[..., VoigtIndex[second]])


def surface_traction(stress: torch.Tensor, normal) -> dict:
    """Traction ``sigma . n`` of a Voigt stress field split into normal/tangential parts.

    Use this for a *loaded* face (pressure, contact, prescribed traction).  For a
    free surface it must return ~0 by construction, which is a statement about
    the field satisfying the traction-free condition only weakly -- it is not a
    way to read a free-surface shear peak; see
    :func:`transverse_shear_magnitude`.

    Returns a dict with

    * ``traction`` ``(..., 3)``: the traction vector ``sigma . n``,
    * ``normal`` ``(...)``: its component along ``n`` (must vanish on a free surface),
    * ``tangential`` ``(..., 3)``: the in-surface part of the traction,
    * ``tangential_magnitude`` ``(...)``: ``|tangential|``, the quantity a free-surface
      shear peak has to be read from.

    The normal is normalised internally, so a face normal taken from a mesh with
    unscaled geometry is accepted as-is.
    """
    tensor = voigt_stress_to_tensor(stress)
    n = torch.as_tensor(normal, dtype=tensor.dtype, device=tensor.device).reshape(-1)
    if n.numel() != 3:
        raise ValueError("normal must have three components")
    n = n / torch.linalg.vector_norm(n)
    traction = torch.einsum("...ij,j->...i", tensor, n)
    normal_component = torch.einsum("...i,i->...", traction, n)
    tangential = traction - normal_component[..., None] * n
    return {
        "traction": traction,
        "normal": normal_component,
        "tangential": tangential,
        "tangential_magnitude": torch.linalg.vector_norm(tangential, dim=-1),
    }