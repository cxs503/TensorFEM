"""Lateral-surface shear recovery for the Saint-Venant torsion benchmark.

Why this module exists
----------------------
``torsion_bar.TorsionBarResult`` reports the peak ``|tau|`` only at element
centroids, and the exact maximum of the Saint-Venant shear stress sits on the
lateral surface at the midpoint of a long side.  A centroid sample therefore
cannot reach the peak, and the earlier evidence left
``shear_peak_relative_error`` explicitly WITHHELD for that reason.  This module
evaluates the same displacement solution *on the surface* and closes the gap
with a measured number instead of a withheld placeholder.

Two estimators are provided so the closure is not a single tuned choice:

``direct``
    Evaluate the element strain-displacement operator at the surface point
    (natural coordinate ``r = 1`` on the face ``x = side``) for every element
    attached to that point and average them, which is what a nodal
    post-processor does.
``extrapolated``
    Evaluate ``|tau|`` at the two interior Gauss abscissae along the surface
    normal and linearly extrapolate to ``r = 1`` (Barlow-style).  The stress is
    most accurate at the Gauss points, so this isolates how much of the surface
    error is sampling versus interpolation.
``centroid``
    The previous WITHHELD baseline (maximum centroid value in the mid-slab), so
    the improvement is visible in one table.

The reference value is the closed-form series evaluated *at* the boundary
point, cross-checked against the interior grid scan of
``torsion_bar.max_shear_stress`` (which is a lower bound off the boundary).

The bar is a ``side x side x length`` prism, ``x, y in [0, side]`` with the
axis along ``z``; the monitored point is ``(side, side/2, length/2)``, i.e. the
midpoint of a long side on the ``x = side`` face, and its mirror images are
equivalent by symmetry for a square section.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .hex20 import hex20_shape_gradients, hex20_stiffness, structured_hex20_mesh
from .solid3d import (elasticity_matrix_3d, hex8_stiffness, structured_hex_mesh,
                      _b, _hex_gradient)
from .sparse_core import assemble_coo, solve_sparse_static
from .torsion_bar import _face_weights, max_shear_stress, torsion_shear_stress

_GAUSS_2 = 1.0 / math.sqrt(3.0)
_GAUSS_3 = math.sqrt(3.0 / 5.0)


@dataclass(frozen=True)
class TorsionBoundaryPeak:
    """Recovered lateral-surface peak shear stress and its reference."""

    element: str
    segments: int
    n_dofs: int
    twist_rate: float
    exact_twist_rate: float
    direct: float
    extrapolated: float
    centroid: float
    exact_boundary: float
    exact_scan: float
    n_face_elements: int

    @property
    def error_direct(self) -> float:
        return abs(self.direct - self.exact_boundary) / self.exact_boundary

    @property
    def error_extrapolated(self) -> float:
        return abs(self.extrapolated - self.exact_boundary) / self.exact_boundary

    @property
    def error_centroid(self) -> float:
        return abs(self.centroid - self.exact_boundary) / self.exact_boundary

    @property
    def twist_rate_relative_error(self) -> float:
        return abs(self.twist_rate - self.exact_twist_rate) / abs(self.exact_twist_rate)


def _mesh(element: str, side: float, length: float, segments: int):
    if element not in ("hex8", "hex20"):
        raise ValueError("element must be 'hex8' or 'hex20'")
    if segments < 1:
        raise ValueError("segments must be >= 1")
    nz = 4 * segments
    if element == "hex20":
        nodes, cells = structured_hex20_mesh(side, side, length, segments, segments, nz)
        return nodes, cells, hex20_stiffness, 20
    nodes, cells = structured_hex_mesh(side, side, length, segments, segments, nz)
    return nodes, cells, hex8_stiffness, 8


def _stress_at(nodes, cells, displacement, modulus, element_dofs, quadratic, r, s, t):
    """Cauchy stress at one natural point of each listed element."""
    xyz = nodes[cells.long()]
    if quadratic:
        _, dn = hex20_shape_gradients(r, s, t, like=xyz)
        jacobian = torch.einsum("eia,ib->eab", xyz, dn)
        gradient = torch.einsum("ib,ebc->eic", dn, torch.linalg.inv(jacobian))
    else:
        gradient, _ = _hex_gradient(xyz, r, s, t)
    strain = torch.einsum("eij,ej->ei", _b(gradient), displacement[element_dofs])
    return torch.einsum("eij,ej->ei", modulus, strain)


def _magnitude(stress: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(stress[:, 5] ** 2 + stress[:, 4] ** 2)


def torsion_boundary_peak(*, element: str = "hex20", segments: int = 4, side: float = 1.0,
                          length: float = 4.0, young: float = 1.0e9, poisson: float = 0.3,
                          torque: float = 1.0, terms: int = 201, rtol: float = 1e-12
                          ) -> TorsionBoundaryPeak:
    """Solve one prism and recover the surface peak shear stress.

    The load case, constraints and twist-rate extraction are identical to
    :func:`tensorfem.torsion_bar.solve_torsion_bar`; only the post-processing
    point differs (surface versus centroid).
    """
    nodes, cells, stiffness, per_node = _mesh(element, side, length, segments)
    quadratic = element == "hex20"
    n_elements = len(cells)
    modulus = elasticity_matrix_3d(nodes.new_tensor(young).expand(n_elements),
                                   nodes.new_tensor(poisson).expand(n_elements))
    ke, _ = stiffness(nodes[cells.long()], modulus)
    element_dofs = torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(
        n_elements, 3 * per_node).long()
    n_dofs = 3 * len(nodes)
    K = assemble_coo(ke, element_dofs, n_dofs, csr=True)

    xc = nodes[:, 0] - side / 2.0
    yc = nodes[:, 1] - side / 2.0
    loading = torch.where(nodes[:, 2] == length)[0]
    weights = _face_weights(nodes, cells, level=length, side=side, segments=segments,
                            quadratic=quadratic)
    forces = nodes.new_zeros(n_dofs)
    forces[3 * loading] = weights[loading] * (-yc[loading])
    forces[3 * loading + 1] = weights[loading] * xc[loading]
    polar = float(torch.sum(weights[loading] * (xc[loading] ** 2 + yc[loading] ** 2)))
    forces = forces * (torque / polar)
    applied = float(torch.sum(xc[loading] * forces[3 * loading + 1]
                             - yc[loading] * forces[3 * loading]))
    if abs(applied - torque) > 1e-9 * abs(torque):
        raise RuntimeError("end-face traction does not carry the requested torque")
    root = torch.where(nodes[:, 2] == 0)[0]
    fixed = {int(dof): 0.0 for dof in torch.cat((3 * root, 3 * root + 1)).tolist()}
    fixed[3 * int(root[0]) + 2] = 0.0
    displacement = solve_sparse_static(K, forces, dirichlet=fixed, rtol=rtol).displacement

    # twist rate from the interior window, exactly as in torsion_bar
    radius2 = xc ** 2 + yc ** 2
    planes = torch.unique(nodes[:, 2])
    thetas = nodes.new_zeros(len(planes))
    for index, z in enumerate(planes.tolist()):
        plane = torch.where(nodes[:, 2] == z)[0]
        thetas[index] = torch.sum(displacement[3 * plane] * (-yc[plane])
                                  + displacement[3 * plane + 1] * xc[plane]) / torch.sum(radius2[plane])
    inside = (planes >= 0.25 * length) & (planes <= 0.75 * length)
    design = torch.stack((planes[inside], torch.ones(int(inside.sum()), dtype=nodes.dtype)), 1)
    twist_rate = float(torch.linalg.lstsq(design, thetas[inside]).solution[0])
    from .torsion_bar import rectangular_torsion_constant
    j_exact = rectangular_torsion_constant(side, side)
    exact_rate = torque / (young / (2.0 * (1.0 + poisson)) * j_exact)

    # elements attached to the monitored surface point (side, side/2, length/2)
    xyz_all = nodes[cells.long()]
    xhi = xyz_all[:, :, 0].amax(1)
    ylo = xyz_all[:, :, 1].amin(1)
    yhi = xyz_all[:, :, 1].amax(1)
    zlo = xyz_all[:, :, 2].amin(1)
    zhi = xyz_all[:, :, 2].amax(1)
    tol = 1e-12
    sel = ((xhi - side).abs() < tol) & (ylo <= side / 2 + tol) & (yhi >= side / 2 - tol) \
        & (zlo <= length / 2 + tol) & (zhi >= length / 2 - tol)
    face = torch.nonzero(sel).flatten()
    if len(face) == 0:
        raise RuntimeError("no element touches the monitored surface point")
    xyz = xyz_all[face]
    centre = xyz.mean(1)
    width = torch.stack((xyz[:, :, 0].amax(1) - xyz[:, :, 0].amin(1),
                         xyz[:, :, 1].amax(1) - xyz[:, :, 1].amin(1),
                         xyz[:, :, 2].amax(1) - xyz[:, :, 2].amin(1)), 1)
    target = torch.tensor([side, side / 2.0, length / 2.0], dtype=nodes.dtype)
    natural = 2.0 * (target - centre) / width
    if torch.any(natural.abs() > 1.0 + 1e-9):
        raise RuntimeError("monitored point outside the selected elements")

    # The natural derivative tables are shared across elements, so the couple of
    # face elements are evaluated one at a time with scalar natural coordinates.
    g = _GAUSS_3 if quadratic else _GAUSS_2
    fraction = (1.0 - g) / (2.0 * g)
    direct_values, extrapolated_values = [], []
    for index, cell_index in enumerate(face.tolist()):
        r, s, t = (float(natural[index, 0]), float(natural[index, 1]),
                   float(natural[index, 2]))
        one = cells[cell_index:cell_index + 1]
        dofs_one = element_dofs[cell_index:cell_index + 1]

        mod_one = modulus[cell_index:cell_index + 1]

        def face_stress(rr):
            values = _magnitude(_stress_at(nodes, one, displacement, mod_one,
                                           dofs_one, quadratic, rr, s, t)).reshape(-1)
            if values.numel() != 1:
                raise RuntimeError(f"expected one face sample, got {values.numel()}")
            return float(values[0])
        direct_values.append(face_stress(r))
        inner, outer = face_stress(-g), face_stress(+g)
        extrapolated_values.append(outer + fraction * (outer - inner))
    direct = sum(direct_values) / len(direct_values)
    extrapolated = sum(extrapolated_values) / len(extrapolated_values)

    centroid_stress = _magnitude(_stress_at(nodes, cells, displacement, modulus,
                                            element_dofs, quadratic, 0.0, 0.0, 0.0))
    if not quadratic:
        # HEX8: the stored operator is the centroid one; it is the same point.
        pass
    centroids = xyz_all.mean(1)
    slab = (centroids[:, 2] - length / 2.0).abs() < length / (4.0 * segments)
    centroid = float(centroid_stress[slab].max())

    shear_modulus = young / (2.0 * (1.0 + poisson))
    # centred section coordinates of the monitored point are (+side/2, 0)
    tx, ty = torsion_shear_stress(torch.tensor([side / 2.0], dtype=nodes.dtype),
                                  torch.tensor([0.0], dtype=nodes.dtype),
                                  side, side, twist_rate=twist_rate,
                                  shear_modulus=shear_modulus, terms=terms)
    exact_boundary = float(torch.sqrt(tx ** 2 + ty ** 2))
    exact_scan, _ = max_shear_stress(side, side, torque=torque, terms=terms)
    return TorsionBoundaryPeak(
        element=element, segments=segments, n_dofs=n_dofs, twist_rate=twist_rate,
        exact_twist_rate=exact_rate, direct=direct, extrapolated=extrapolated,
        centroid=centroid, exact_boundary=exact_boundary, exact_scan=exact_scan,
        n_face_elements=int(len(face)))