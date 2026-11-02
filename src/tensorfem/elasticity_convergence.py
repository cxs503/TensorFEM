"""Analytic oracles and h-convergence drivers for the 3-D linear-elastic solid elements.

Two independent oracles are provided, and they answer two different questions.

1. :class:`PureBendingOracle` -- the classical Navier pure-bending solution of a
   prismatic bar.  For a bar of length ``L`` in ``x`` with a rectangular section
   (extent ``w`` in ``y``, ``h`` in ``z``) and curvature ``k``

       u_x = -k x yc,
       u_y =  k/2 * (x**2 + nu*yc**2 - nu*zc**2),
       u_z =  k*nu * yc*zc,          yc = y - w/2,  zc = z - h/2

   satisfies ``div(sigma) = 0`` identically (``sigma_xx = -E k yc`` is the only
   non-zero stress, all shears vanish, every component is at most linear in
   ``yc``), so it is the *exact* solution of the elasticity boundary-value
   problem with that field prescribed on the boundary.  Its strain energy is
   known in closed form, ``U = E k**2 / 2 * L * I`` with ``I = h w**3 / 12``.
   On a uniform mesh the discrete solution reproduces its nodal values to
   machine precision, which makes it a strong *consistency* (superconvergence)
   check but not an order-of-accuracy measurement.

2. :class:`ManufacturedElasticity` -- a smooth trigonometric manufactured
   solution whose body force ``b = -div(sigma)`` is obtained by automatic
   differentiation of the analytic field and whose consistent nodal loads are
   integrated with an independent Gauss rule (no Jacobian, ``B`` matrix or
   stiffness from the code under test is involved).  A trilinear HEX8 space
   cannot represent the field, so the discretisation error decays with a
   measurable rate, giving a genuine h-convergence study against an analytic
   reference.

Neither oracle proves a solver is physically right on its own; together with
:mod:`tensorfem.convergence_verification` they quantify consistency, rate and
the remaining error band.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .solid3d import SolidModel, hex_to_tet_mesh, solve_solid, structured_hex_mesh

__all__ = [
    "PureBendingOracle",
    "ManufacturedElasticity",
    "axial_bar_oracle_displacement",
    "hex8_dirichlet_study",
    "hex8_manufactured_study",
]


# ---------------------------------------------------------------------------
# Oracle 1: closed-form pure bending (consistency / superconvergence)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PureBendingOracle:
    """Closed-form pure-bending state of a prismatic bar."""

    length: float
    width: float
    height: float
    young_modulus: float
    poisson_ratio: float
    curvature: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length), ("width", self.width), ("height", self.height),
            ("young_modulus", self.young_modulus), ("curvature", self.curvature),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")

    @property
    def second_moment(self) -> float:
        """Section second moment about the bending axis, ``I = h w**3 / 12``."""
        return self.height * self.width**3 / 12.0

    @property
    def bending_moment(self) -> float:
        """End moment consistent with the imposed curvature, ``M = E I k``."""
        return self.young_modulus * self.second_moment * self.curvature

    @property
    def strain_energy(self) -> float:
        """Exact strain energy ``U = E k**2 / 2 * L * I`` (J)."""
        return self.young_modulus * self.curvature**2 / 2.0 * self.length * self.second_moment

    def displacement(self, nodes: torch.Tensor) -> torch.Tensor:
        """Exact displacement ``[n,3]`` at physical coordinates ``[n,3]``."""
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("nodes must have shape [n,3]")
        x = nodes[:, 0]
        yc = nodes[:, 1] - self.width / 2.0
        zc = nodes[:, 2] - self.height / 2.0
        nu = self.poisson_ratio
        k = self.curvature
        return torch.stack(
            (
                -k * x * yc,
                k / 2.0 * (x * x + nu * yc * yc - nu * zc * zc),
                k * nu * yc * zc,
            ),
            dim=1,
        )

    def axial_stress(self, nodes: torch.Tensor) -> torch.Tensor:
        """Exact ``sigma_xx = -E k yc`` at ``nodes`` (Pa)."""
        return -self.young_modulus * self.curvature * (nodes[:, 1] - self.width / 2.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "young_modulus": self.young_modulus,
            "poisson_ratio": self.poisson_ratio,
            "curvature": self.curvature,
            "second_moment": self.second_moment,
            "bending_moment": self.bending_moment,
            "exact_strain_energy": self.strain_energy,
        }


def axial_bar_oracle_displacement(
    nodes: torch.Tensor, *, stress: float, young_modulus: float, poisson_ratio: float
) -> torch.Tensor:
    """Exact uniaxial-stress displacement field of a bar aligned with ``x``.

    ``u = (sigma/E) * (x, -nu*y, -nu*z)`` is reproduced exactly by TET4/HEX8, so
    it serves as the consistency (patch-test) companion of the bending oracle.
    """
    strain = stress / young_modulus
    return torch.stack(
        (strain * nodes[:, 0], -poisson_ratio * strain * nodes[:, 1],
         -poisson_ratio * strain * nodes[:, 2]),
        dim=1,
    )


def _box_boundary_mask(nodes: torch.Tensor, lx: float, ly: float, lz: float) -> torch.Tensor:
    x, y, z = nodes.unbind(1)
    tol = 1e-12 * max(lx, ly, lz)
    return (
        (x <= tol) | (x >= lx - tol)
        | (y <= tol) | (y >= ly - tol)
        | (z <= tol) | (z >= lz - tol)
    )


def hex8_dirichlet_study(
    oracle: PureBendingOracle,
    divisions: Sequence[int],
    *,
    element_type: str = "hex8",
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, object]]:
    """Solve the bending Dirichlet problem on ``(2n, n, n)`` meshes.

    Every level uses ``n`` from ``divisions`` with ``nx = 2n``, ``ny = nz = n`` and
    the representative mesh size ``h = length / nx``, so a geometric sequence of
    ``n`` gives a constant refinement ratio.  The exact field is prescribed on
    all boundary nodes; interior nodes are solved for, i.e. this is a genuine
    Galerkin solve.  Records carry the relative L2 displacement error and the
    relative strain-energy error against the closed-form values.
    """
    if not divisions:
        raise ValueError("divisions must not be empty")
    exact_energy = oracle.strain_energy
    records: list[dict[str, object]] = []
    for n in divisions:
        if int(n) != n or n < 1:
            raise ValueError("divisions must be positive integers")
        nx, ny, nz = 2 * int(n), int(n), int(n)
        nodes, hexes = structured_hex_mesh(
            oracle.length, oracle.width, oracle.height, nx, ny, nz, dtype=dtype
        )
        elements = hexes if element_type.lower() == "hex8" else hex_to_tet_mesh(nodes, hexes)[1]
        exact = oracle.displacement(nodes)
        boundary_nodes = torch.where(_box_boundary_mask(
            nodes, oracle.length, oracle.width, oracle.height))[0]
        fixed = torch.stack(
            (3 * boundary_nodes, 3 * boundary_nodes + 1, 3 * boundary_nodes + 2), 1
        ).reshape(-1)
        model = SolidModel(
            nodes,
            elements,
            torch.tensor(oracle.young_modulus, dtype=dtype),
            torch.tensor(oracle.poisson_ratio, dtype=dtype),
            torch.zeros(3 * len(nodes), dtype=dtype),
            fixed,
            element_type,
            exact.reshape(-1)[fixed],
        )
        result = solve_solid(model)
        numerical = result.displacement.reshape(-1, 3)
        energy = float(result.strain_energy)
        records.append({
            "divisions": int(n),
            "mesh": [nx, ny, nz],
            "mesh_size": oracle.length / nx,
            "n_nodes": int(len(nodes)),
            "n_free_dofs": int(3 * len(nodes) - fixed.numel()),
            "relative_l2_displacement_error": float(
                torch.linalg.vector_norm(numerical - exact) / torch.linalg.vector_norm(exact)
            ),
            "strain_energy": energy,
            "exact_strain_energy": exact_energy,
            "relative_strain_energy_error": abs(energy - exact_energy) / exact_energy,
            "energy_above_exact": bool(energy >= exact_energy * (1.0 - 1e-12)),
        })
    return records


# ---------------------------------------------------------------------------
# Oracle 2: manufactured trigonometric field with a smooth body force
# ---------------------------------------------------------------------------


def _gauss_rule(n: int) -> tuple[list[float], list[float]]:
    """Gauss-Legendre nodes/weights on ``[-1,1]``."""
    if n < 2:
        raise ValueError("gauss order must be at least 2")
    import numpy as np

    nodes, weights = np.polynomial.legendre.leggauss(n)
    return [float(v) for v in nodes], [float(v) for v in weights]


def _map_rule(
    points: Sequence[float], weights: Sequence[float], lo: float, hi: float
) -> list[tuple[float, float]]:
    half = (hi - lo) / 2.0
    return [(lo + half * (p + 1.0), w * half) for p, w in zip(points, weights)]


def _hex_shape(rst: Sequence[float]) -> list[float]:
    """Trilinear HEX8 shape functions for the node order used by this repo."""
    r, s, t = rst
    return [
        (1 - r) * (1 - s) * (1 - t) / 8,
        (1 + r) * (1 - s) * (1 - t) / 8,
        (1 + r) * (1 + s) * (1 - t) / 8,
        (1 - r) * (1 + s) * (1 - t) / 8,
        (1 - r) * (1 - s) * (1 + t) / 8,
        (1 + r) * (1 - s) * (1 + t) / 8,
        (1 + r) * (1 + s) * (1 + t) / 8,
        (1 - r) * (1 + s) * (1 + t) / 8,
    ]


def _element_counts(nodes: torch.Tensor) -> tuple[int, int, int]:
    """Element counts of a structured box mesh (unique coordinate planes - 1)."""
    return tuple(int(len(torch.unique(nodes[:, axis]))) - 1 for axis in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class ManufacturedElasticity:
    """Smooth manufactured field ``u_i = a cos/sin(k x) cos/sin(l y) cos/sin(m z)``."""

    length: float = 2.0
    width: float = 0.4
    height: float = 0.6
    young_modulus: float = 210e9
    poisson_ratio: float = 0.3
    amplitude: float = 1e-4
    planar: bool = False

    def __post_init__(self) -> None:
        for name in ("length", "width", "height", "young_modulus", "amplitude"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")

    def _wavenumbers(self) -> tuple[float, float, float]:
        """Wavenumbers ``(kx, ly, mz)``; ``mz = 0`` for the planar variant.

        With ``planar=True`` the field is independent of ``z`` and ``u_z == 0``,
        which is still an exact solution (``sigma_zz != 0`` there, supplied by
        the Dirichlet data), so a single element through the thickness is exact
        and the refinement study becomes two-dimensional and cheap.
        """
        mz = 0.0 if self.planar else math.pi / self.height
        return (math.pi / self.length, math.pi / self.width, mz)

    def displacement(self, nodes: torch.Tensor) -> torch.Tensor:
        """Analytic manufactured displacement ``[n,3]``."""
        kx, ly, mz = self._wavenumbers()
        x, y, z = nodes.unbind(1)
        a = self.amplitude
        return torch.stack(
            (
                a * torch.sin(kx * x) * torch.cos(ly * y) * torch.cos(mz * z),
                a * torch.cos(kx * x) * torch.sin(ly * y) * torch.cos(mz * z),
                a * torch.cos(kx * x) * torch.cos(ly * y) * torch.sin(mz * z),
            ),
            dim=1,
        )

    def displacement_gradient(self, nodes: torch.Tensor) -> torch.Tensor:
        """``du_i/dx_j`` from the closed-form derivatives."""
        kx, ly, mz = self._wavenumbers()
        x, y, z = nodes.unbind(1)
        a = self.amplitude
        sx, cx = torch.sin(kx * x), torch.cos(kx * x)
        sy, cy = torch.sin(ly * y), torch.cos(ly * y)
        sz, cz = torch.sin(mz * z), torch.cos(mz * z)
        grad = torch.zeros(len(nodes), 3, 3, dtype=nodes.dtype, device=nodes.device)
        grad[:, 0, 0] = a * kx * cx * cy * cz
        grad[:, 0, 1] = -a * ly * sx * sy * cz
        grad[:, 0, 2] = -a * mz * sx * cy * sz
        grad[:, 1, 0] = -a * kx * sx * sy * cz
        grad[:, 1, 1] = a * ly * cx * cy * cz
        grad[:, 1, 2] = -a * mz * cx * sy * sz
        grad[:, 2, 0] = -a * kx * sx * cy * sz
        grad[:, 2, 1] = -a * ly * cx * sy * sz
        grad[:, 2, 2] = a * mz * cx * cy * cz
        return grad

    def strain(self, nodes: torch.Tensor) -> torch.Tensor:
        """Analytic strain ``[n,6]`` ordered ``(xx, yy, zz, g_xy, g_yz, g_xz)``.

        The shear entries are *engineering* shears ``g_ij = du_i/dx_j +
        du_j/dx_i``, matching the strain vector produced by the element
        ``B``-matrix of this repository (``SolidResult.strain``).
        """
        grad = self.displacement_gradient(nodes)
        return torch.stack(
            (
                grad[:, 0, 0], grad[:, 1, 1], grad[:, 2, 2],
                grad[:, 0, 1] + grad[:, 1, 0],
                grad[:, 1, 2] + grad[:, 2, 1],
                grad[:, 0, 2] + grad[:, 2, 0],
            ),
            dim=1,
        )

    def _lame(self) -> tuple[float, float]:
        nu = self.poisson_ratio
        lam = self.young_modulus * nu / ((1 + nu) * (1 - 2 * nu))
        mu = self.young_modulus / (2 * (1 + nu))
        return lam, mu

    def stress(self, nodes: torch.Tensor) -> torch.Tensor:
        """Analytic stress ``[n,6]`` ordered ``(xx, yy, zz, xy, yz, xz)`` (Pa).

        ``sigma = D @ strain`` with the repository's ``D``; because ``strain``
        carries engineering shears, ``sigma_ij = mu * g_ij`` for the shears.
        """
        strain = self.strain(nodes)
        lam, mu = self._lame()
        tr = strain[:, 0] + strain[:, 1] + strain[:, 2]
        return torch.stack(
            (
                lam * tr + 2 * mu * strain[:, 0],
                lam * tr + 2 * mu * strain[:, 1],
                lam * tr + 2 * mu * strain[:, 2],
                mu * strain[:, 3],
                mu * strain[:, 4],
                mu * strain[:, 5],
            ),
            dim=1,
        )

    def body_force(self, nodes: torch.Tensor) -> torch.Tensor:
        """``b = -div(sigma)`` of the analytic field, by autograd.

        This is the piece that makes the field an exact elasticity solution.  It
        differentiates the analytic displacement expression directly, so the
        load is not derived from the element ``B``-matrix under test.
        """
        xyz = nodes.clone().detach().requires_grad_(True)
        u = self.displacement(xyz)
        eye = torch.eye(3, dtype=nodes.dtype, device=nodes.device)
        ones = torch.ones(len(nodes), dtype=nodes.dtype, device=nodes.device)
        grad_u = torch.stack(
            [
                torch.autograd.grad(u[:, i], xyz, grad_outputs=ones, create_graph=True)[0]
                for i in range(3)
            ],
            dim=1,
        )
        sym = 0.5 * (grad_u + grad_u.transpose(1, 2))
        tr = sym[:, 0, 0] + sym[:, 1, 1] + sym[:, 2, 2]
        lam, mu = self._lame()
        sigma = lam * tr[:, None, None] * eye + 2 * mu * sym
        rows = []
        for i in range(3):
            total = torch.zeros(len(nodes), dtype=nodes.dtype, device=nodes.device)
            for j in range(3):
                total = total + torch.autograd.grad(
                    sigma[:, i, j].sum(), xyz, retain_graph=True
                )[0][:, j]
            rows.append(-total)
        return torch.stack(rows, dim=1)

    def strain_energy_exact(self, gauss: int = 5) -> float:
        """Exact strain energy ``int sigma:eps/2 dV`` by independent quadrature."""
        gl, gw = _gauss_rule(gauss)
        xs = _map_rule(gl, gw, 0.0, self.length)
        ys = _map_rule(gl, gw, 0.0, self.width)
        zs = _map_rule(gl, gw, 0.0, self.height)
        points = [(a[0], b[0], c[0]) for a in xs for b in ys for c in zs]
        weights = [a[1] * b[1] * c[1] for a in xs for b in ys for c in zs]
        coords = torch.tensor(points, dtype=torch.float64)
        weight = torch.tensor(weights, dtype=torch.float64)
        strain = self.strain(coords)
        stress = self.stress(coords)
        return float(((stress * strain).sum(dim=1) / 2.0 * weight).sum())

    def as_dict(self) -> dict[str, float]:
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "young_modulus": self.young_modulus,
            "poisson_ratio": self.poisson_ratio,
            "amplitude": self.amplitude,
            "exact_strain_energy": self.strain_energy_exact(),
        }


def _consistent_body_force(
    case: ManufacturedElasticity,
    nodes: torch.Tensor,
    hexes: torch.Tensor,
    points: Sequence[float],
    weights: Sequence[float],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``f_a = sum_e int_e N_a b dV`` integrated with an independent Gauss rule.

    The mesh is a uniform box, so an element's natural coordinates follow from
    its physical size and lower corner; no Jacobian, ``B``-matrix or stiffness
    from the code under test is used, which keeps the load independent.  All
    quadrature points are evaluated in a single batch because the body force is
    a function of the coordinates only.
    """
    nx, ny, nz = _element_counts(nodes)
    half = torch.tensor(
        [case.length / nx / 2.0, case.width / ny / 2.0, case.height / nz / 2.0], dtype=dtype
    )
    rule = [(px, py, pz, wx * wy * wz)
            for px, wx in zip(points, weights)
            for py, wy in zip(points, weights)
            for pz, wz in zip(points, weights)]
    natural = torch.tensor([r[:3] for r in rule], dtype=dtype)           # (q,3)
    weight = torch.tensor([r[3] for r in rule], dtype=dtype)             # (q,)
    shape = torch.tensor([_hex_shape(tuple(r[:3])) for r in rule], dtype=dtype)  # (q,8)
    origin = nodes[hexes].min(dim=1).values                              # (e,3)
    scale = half.prod()
    coords = origin[:, None, :] + half * (natural[None, :, :] + 1.0)     # (e,q,3)
    body = case.body_force(coords.reshape(-1, 3)).reshape(
        -1, len(rule), 3
    )                                                                    # (e,q,3)
    contribution = torch.einsum("qa,q,eqi->eai", shape, weight * scale, body)
    forces = torch.zeros(len(nodes), 3, dtype=dtype)
    forces.index_add_(0, hexes.reshape(-1), contribution.reshape(-1, 3))
    return forces.reshape(-1)


def hex8_manufactured_study(
    case: ManufacturedElasticity,
    divisions: Sequence[int],
    *,
    gauss: int = 4,
    layers: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, object]]:
    """Solve the manufactured Dirichlet problem on ``(2n, n, n)`` HEX8 meshes.

    Boundary nodes carry the analytic displacement; interior nodes are free and
    loaded by the consistent body-force vector.  Because the field is not in the
    trilinear space, the errors decay at a measurable rate -- that is what the
    h-convergence study needs.
    """
    if not divisions:
        raise ValueError("divisions must not be empty")
    points, weights = _gauss_rule(gauss)
    exact_energy = case.strain_energy_exact()
    records: list[dict[str, object]] = []
    thickness_layers = layers if layers is not None else (3 if case.planar else None)
    for n in divisions:
        if int(n) != n or n < 2:
            raise ValueError("divisions must be integers >= 2 (interior nodes are required)")
        nx, ny = 2 * int(n), int(n)
        nz = thickness_layers if thickness_layers is not None else int(n)
        nodes, hexes = structured_hex_mesh(
            case.length, case.width, case.height, nx, ny, nz, dtype=dtype
        )
        exact = case.displacement(nodes)
        boundary_nodes = torch.where(_box_boundary_mask(
            nodes, case.length, case.width, case.height))[0]
        fixed = torch.stack(
            (3 * boundary_nodes, 3 * boundary_nodes + 1, 3 * boundary_nodes + 2), 1
        ).reshape(-1)
        forces = _consistent_body_force(case, nodes, hexes, points, weights, dtype=dtype)
        model = SolidModel(
            nodes,
            hexes,
            torch.tensor(case.young_modulus, dtype=dtype),
            torch.tensor(case.poisson_ratio, dtype=dtype),
            forces,
            fixed,
            "hex8",
            exact.reshape(-1)[fixed],
        )
        result = solve_solid(model)
        numerical = result.displacement.reshape(-1, 3)
        energy = float(result.strain_energy)
        records.append({
            "divisions": int(n),
            "mesh": [nx, ny, nz],
            "mesh_size": case.length / nx,
            "n_nodes": int(len(nodes)),
            "n_free_dofs": int(3 * len(nodes) - fixed.numel()),
            "relative_l2_displacement_error": float(
                torch.linalg.vector_norm(numerical - exact) / torch.linalg.vector_norm(exact)
            ),
            "strain_energy": energy,
            "exact_strain_energy": exact_energy,
            "relative_strain_energy_error": abs(energy - exact_energy) / exact_energy,
        })
    return records