"""Mass matrices for TET4 and HEX8 solids, with analytic verification anchors.

The repository previously offered stiffness + static solve for 3-D solids
(``solid3d``) but no mass matrix at all, so 3-D vibration, transient and
explicit-dynamics work could only be done by hand-assembling a mass matrix
outside the library.  This module supplies

* ``hex8_consistent_mass`` / ``tet4_consistent_mass`` -- element mass matrices
  from the displacement interpolation (:math:`\\int \\rho N_i N_j \\, dV`),
* ``hex8_lumped_mass`` / ``tet4_lumped_mass`` -- the classical row-sum
  (mass-preserving) diagonal companion used by explicit integrators,
* ``assemble_solid_mass`` -- global ``(n_dofs, n_dofs)`` assembly using the
  same degree-of-freedom ordering as ``solid3d.solve_solid``
  (``[ux, uy, uz]`` per node, element by element),
* ``mass_properties`` -- the two exact invariants any mass matrix must satisfy
  (total mass, polar moment of inertia) plus their independent quadrature,

and the analytic reference used to verify the mass formulation physically:

    omega_n = beta_n sqrt(M_c / rho),   beta_n = (2n-1) pi / (2L)

for a fixed--free rod whose lateral displacements are suppressed, where
``M_c = E (1-nu) / ((1+nu)(1-2nu))`` is the confined (uniaxial-strain) modulus
that governs a laterally constrained rod.  With ``nu = 0`` this collapses to
the familiar ``omega_n = (2n-1) pi / (2L) * sqrt(E / rho)``.

Implementation notes
--------------------
* The element mass integral is evaluated with the same 2x2x2 Gauss rule used
  by ``hex8_stiffness``.  ``N_i N_j`` is at most quadratic per natural
  direction, so the rule is exact for bricks and is the standard choice for
  isoparametric hexes.
* The linear TET4 mass matrix is integrated in closed form
  (:math:`m_{ii} = \\rho V / 10`, :math:`m_{ij} = \\rho V / 20`), which is exact
  for any tet and doubles as an oracle for the quadrature path.
* Row-sum lumping conserves total mass exactly but *raises* rotational inertia
  of the discrete model; ``mass_properties`` measures that difference instead
  of hiding it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

_HEX_SIGNS = (
    (-1, -1, -1),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
)


def _expand_scalar(value, count: int, *, dtype, device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
    if tensor.numel() not in (1, count):
        raise ValueError("per-element scalar must have length 1 or n_elements")
    if not bool(torch.isfinite(tensor).all()) or bool((tensor <= 0).any()):
        raise ValueError("density must be positive and finite")
    return tensor.expand(count)


def _dof_map(elements: torch.Tensor) -> torch.Tensor:
    ne, nn = elements.shape
    return torch.stack(tuple(3 * elements + i for i in range(3)), 2).reshape(ne, 3 * nn).long()


def _gauss_to_full(scalar: torch.Tensor) -> torch.Tensor:
    """Lift an ``(ne, nn, nn)`` scalar mass block to ``(ne, 3nn, 3nn)``."""
    ne, nn, _ = scalar.shape
    eye = torch.eye(3, dtype=scalar.dtype, device=scalar.device)
    return torch.einsum("eij,ab->eiajb", scalar, eye).reshape(ne, 3 * nn, 3 * nn)


def _row_sum_lumped(element_mass: torch.Tensor) -> torch.Tensor:
    rows = element_mass.sum(-1)
    if bool((rows <= 0).any()):
        raise ValueError("lumped mass must be positive")
    return torch.diag_embed(rows)


def _hex8_point(x: torch.Tensor, xi: float, eta: float, zeta: float):
    """Return shape functions and Jacobian determinant at one Gauss point."""
    signs = x.new_tensor(_HEX_SIGNS)
    a, b, c = signs[:, 0], signs[:, 1], signs[:, 2]
    one = x.new_tensor(1.0)
    d_nat = torch.stack(
        (
            a * (one + b * eta) * (one + c * zeta),
            b * (one + a * xi) * (one + c * zeta),
            c * (one + a * xi) * (one + b * eta),
        ),
        1,
    ) / 8.0
    jacobian = torch.einsum("eia,ib->eab", x, d_nat)
    det = torch.linalg.det(jacobian)
    if bool((det <= torch.finfo(x.dtype).eps).any()):
        raise ValueError("HEX8 element has non-positive or degenerate Jacobian")
    shape = (one + a * xi) * (one + b * eta) * (one + c * zeta) / 8.0
    return shape.expand(len(x), 8), det


def _tet4_volume(x: torch.Tensor) -> torch.Tensor:
    edges = torch.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), 1)
    volume = torch.linalg.det(edges).abs() / 6.0
    if bool((volume <= torch.finfo(x.dtype).eps).any()):
        raise ValueError("TET4 element has zero volume")
    return volume


def hex8_consistent_mass(x: torch.Tensor, density) -> torch.Tensor:
    """Return the 24x24 consistent mass matrix of HEX8 elements.

    Parameters
    ----------
    x : (n_elements, 8, 3) nodal coordinates of each element.
    density : scalar or ``(n_elements,)`` mass density.
    """
    if x.ndim != 3 or x.shape[1:] != (8, 3):
        raise ValueError("x must have shape (n_elements, 8, 3)")
    rho = _expand_scalar(density, len(x), dtype=x.dtype, device=x.device)
    block = torch.zeros((len(x), 8, 8), dtype=x.dtype, device=x.device)
    q = 1.0 / 3.0 ** 0.5
    for xi in (-q, q):
        for eta in (-q, q):
            for zeta in (-q, q):
                shape, det = _hex8_point(x, xi, eta, zeta)
                block = block + (rho * det)[:, None, None] * torch.einsum(
                    "ei,ej->eij", shape, shape
                )
    return _gauss_to_full(block)


def tet4_consistent_mass(x: torch.Tensor, density) -> torch.Tensor:
    """Return the 12x12 consistent mass matrix of TET4 elements (closed form).

    Linear tets integrate exactly to m_ii = rho V / 10 and m_ij = rho V / 20, so the
    scalar block is (rho V / 20) * (ones + eye) and every row sums to rho V / 4.
    """
    if x.ndim != 3 or x.shape[1:] != (4, 3):
        raise ValueError("x must have shape (n_elements, 4, 3)")
    rho = _expand_scalar(density, len(x), dtype=x.dtype, device=x.device)
    volume = _tet4_volume(x)
    block = torch.ones((len(x), 4, 4), dtype=x.dtype, device=x.device)
    block = block + torch.eye(4, dtype=x.dtype, device=x.device)
    return _gauss_to_full((rho * volume / 20.0)[:, None, None] * block)


def _element_mass(x: torch.Tensor, density, element_type: str, lumped: bool) -> torch.Tensor:
    kind = element_type.lower()
    if kind == "hex8":
        element = hex8_consistent_mass(x, density)
    elif kind == "tet4":
        element = tet4_consistent_mass(x, density)
    elif kind == "tet10":
        from .quadratic_solid import tet10_consistent_mass, tet10_lumped_mass

        return (tet10_lumped_mass if lumped else tet10_consistent_mass)(x, density)
    else:
        raise ValueError("element_type must be 'tet4', 'tet10' or 'hex8'")
    return _row_sum_lumped(element) if lumped else element


def assemble_solid_mass(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    density,
    *,
    element_type: str = "hex8",
    lumped: bool = False,
) -> torch.Tensor:
    """Assemble the global solid mass matrix in ``solid3d`` DOF ordering."""
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (n_nodes, 3)")
    if elements.ndim != 2 or elements.dtype not in (torch.int32, torch.int64):
        raise ValueError("elements must be an integer connectivity array")
    kind = element_type.lower()
    expected = {"hex8": 8, "tet4": 4, "tet10": 10}.get(kind)
    if expected is None:
        raise ValueError("element_type must be 'tet4', 'tet10' or 'hex8'")
    if elements.shape[1] != expected:
        raise ValueError(f"{kind} connectivity must have {expected} nodes per element")
    if int(elements.min()) < 0 or int(elements.max()) >= len(nodes):
        raise ValueError("connectivity index out of range")
    x = nodes[elements.long()]
    element = _element_mass(x, density, kind, lumped)
    dofs = _dof_map(elements)
    ndofs = 3 * len(nodes)
    rows = dofs[:, :, None].expand(-1, -1, dofs.shape[1]).reshape(-1)
    cols = dofs[:, None, :].expand(-1, dofs.shape[1], -1).reshape(-1)
    mass = torch.zeros((ndofs, ndofs), dtype=nodes.dtype, device=nodes.device)
    return mass.index_put((rows, cols), element.reshape(-1), accumulate=True)


@dataclass(frozen=True)
class MassProperties:
    """Discrete versus exact mass and polar moment of inertia.

    ``total_mass`` is ``1^T M_x 1`` (partition of unity makes this ``rho V``).
    ``polar_inertia`` is the quadratic form of the rigid rotation field
    ``v = omega (-y, x, 0)``, i.e. the discrete ``rho * integral (x^2 + y^2)``.
    """

    total_mass: float
    exact_mass: float
    polar_inertia: float
    exact_polar_inertia: float

    @property
    def mass_error(self) -> float:
        return abs(self.total_mass / self.exact_mass - 1.0)

    @property
    def polar_inertia_error(self) -> float:
        return abs(self.polar_inertia / self.exact_polar_inertia - 1.0)


def _exact_integrals(nodes: torch.Tensor, elements: torch.Tensor, density, element_type: str):
    """Quadrature of ``rho``, ``rho * (x^2 + y^2)`` independent of the mass code."""
    x = nodes[elements.long()]
    rho = _expand_scalar(density, len(x), dtype=nodes.dtype, device=nodes.device)
    kind = element_type.lower()
    if kind == "tet4":
        # Symmetric 4-point (Keast) rule, exact for quadratics on a tetrahedron.
        a, b = 0.5854101966249685, 0.1381966011250105
        bary = torch.tensor(
            [[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]],
            dtype=x.dtype,
            device=x.device,
        )
        pts = torch.einsum("kp,epc->ekc", bary, x)
        weights = (_tet4_volume(x) / 4.0)[:, None].expand(-1, 4)
    elif kind == "hex8":
        q = 1.0 / 3.0 ** 0.5
        coords = [(xi, eta, zeta) for xi in (-q, q) for eta in (-q, q) for zeta in (-q, q)]
        shapes = []
        dets = []
        for xi, eta, zeta in coords:
            shape, det = _hex8_point(x, xi, eta, zeta)
            shapes.append(shape)
            dets.append(det)
        shapes = torch.stack(shapes, 1)
        dets = torch.stack(dets, 1)
        pts = torch.einsum("eji,eia->eja", shapes, x)
        weights = dets
    elif kind == "tet10":
        # Independent higher-order rule, distinct from the mass integrator default.
        from .quadratic_solid import _tet10_det_weights, tet10_mass_quadrature

        shape, scaled = _tet10_det_weights(x, tet10_mass_quadrature(6, dtype=x.dtype))
        pts = torch.einsum("qi,eia->eqa", shape, x)
        weights = scaled
    else:
        raise ValueError("element_type must be 'tet4', 'tet10' or 'hex8'")
    mass = float((rho[:, None] * weights).sum())
    radius = pts[..., 0] ** 2 + pts[..., 1] ** 2
    polar = float((rho[:, None] * weights * radius).sum())
    return mass, polar


def mass_properties(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    density,
    *,
    element_type: str = "hex8",
    lumped: bool = False,
) -> MassProperties:
    """Return discrete versus exact mass and polar inertia of the mesh."""
    mass = assemble_solid_mass(
        nodes, elements, density, element_type=element_type, lumped=lumped
    )
    ndofs = mass.shape[0]
    unit = torch.ones(ndofs, dtype=nodes.dtype, device=nodes.device)
    total = float(unit @ mass @ unit) / 3.0
    velocity = torch.zeros((len(nodes), 3), dtype=nodes.dtype, device=nodes.device)
    velocity[:, 0] = -nodes[:, 1]
    velocity[:, 1] = nodes[:, 0]
    velocity = velocity.reshape(-1)
    polar = float(velocity @ mass @ velocity)
    exact_mass, exact_polar = _exact_integrals(nodes, elements, density, element_type)
    return MassProperties(total, exact_mass, polar, exact_polar)


def confined_modulus(young: float, poisson: float) -> float:
    """Uniaxial-strain (confined) modulus ``E(1-nu)/((1+nu)(1-2nu))``."""
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson ratio must lie in (-1, 0.5)")
    return young * (1.0 - poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson))


def rod_axial_angular_frequency(
    mode: int, length: float, modulus: float, density: float, *, fixed_free: bool = True
) -> float:
    """Exact axial frequency of a rod: ``beta_n sqrt(modulus / density)``.

    ``fixed_free=True`` gives ``beta_n = (2n-1) pi / (2L)``; ``False`` (free--
    free / simply supported axial) gives ``beta_n = n pi / L``.
    """
    if mode < 1:
        raise ValueError("mode must be >= 1")
    if length <= 0 or modulus <= 0 or density <= 0:
        raise ValueError("length, modulus and density must be positive")
    beta = (2 * mode - 1) * math.pi / (2 * length) if fixed_free else mode * math.pi / length
    return beta * math.sqrt(modulus / density)