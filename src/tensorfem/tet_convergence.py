"""h-convergence / GCI evidence for the TET4 and TET10 solid element families.

``docs/h-convergence-gci-verification.md`` §6 records that only ``hex8`` was
covered by the discrete-uncertainty evidence set, leaving ``tet4`` and the
ten-node tetrahedron unmeasured.  This module closes that gap with the same
methodology but no reuse of the element stiffness under test:

* :func:`tet_consistent_body_force` integrates the consistent nodal loads of a
  body force with a collapsed (Duffy) Gauss rule over each tetrahedron.  It uses
  only the affine geometry of the four corners and the element shape functions,
  so neither the ``B`` matrix nor the stiffness matrix of the code under test
  enters the load vector.
* :func:`tet4_manufactured_study` / :func:`tet10_manufactured_study` solve a
  Dirichlet problem with the smooth trigonometric manufactured field of
  :class:`tensorfem.elasticity_convergence.ManufacturedElasticity` on
  ``(2n, n, n)`` tetrahedral meshes and report the L2 displacement error, the
  strain-energy error and the counts needed to interpret them.
* :func:`tet_dirichlet_study` applies the closed-form pure-bending field of
  :class:`~tensorfem.elasticity_convergence.PureBendingOracle` as the Dirichlet
  data with no body force.  The field is a member of the TET10 space, so TET10
  must reproduce its nodal values to roundoff -- a strong consistency check that
  does not certify any bending problem the mesh cannot resolve.

Measuring a rate is not validation: it states that the discrete operator is
consistent and converges at some order.  It says nothing about the suitability
of tetrahedra for a bending-dominated shell or about volumetric locking, which
is a separate, already documented limitation.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .elasticity_convergence import ManufacturedElasticity, PureBendingOracle
from .quadratic_solid import Tet10Model, solve_tet10, tet10_shape, tet4_to_tet10
from .solid3d import SolidModel, hex_to_tet_mesh, solve_solid, structured_hex_mesh

__all__ = [
    "duffy_tet_rule",
    "tet_consistent_body_force",
    "tet4_manufactured_study",
    "tet10_manufactured_study",
    "tet_dirichlet_study",
]


# ---------------------------------------------------------------------------
# Independent quadrature over a tetrahedron
# ---------------------------------------------------------------------------


def duffy_tet_rule(order: int = 6, *, dtype: torch.dtype = torch.float64):
    """Collapsed Gauss rule on the reference tetrahedron.

    Returns points ``(q,3)`` in natural coordinates ``(r,s,t)`` plus weights
    ``(q,)`` that already include the reference volume factor, i.e.
    ``sum(weights) == 1/6``.  The collapse ``r=u``, ``s=v(1-u)``,
    ``t=w(1-u)(1-v)`` has Jacobian ``(1-u)**2 * (1-v)``, so the mapping is
    exact for polynomials that the Gauss order can integrate.  Nothing in this
    rule depends on a finite element implementation.
    """
    if int(order) != order or order < 2:
        raise ValueError("duffy order must be an integer >= 2")
    import numpy as np

    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    u = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    uu = torch.tensor(u, dtype=torch.float64)
    ww = torch.tensor(w, dtype=torch.float64)
    a, b, c = torch.meshgrid(uu, uu, uu, indexing="ij")
    wa, wb, wc = torch.meshgrid(ww, ww, ww, indexing="ij")
    r = a
    s = b * (1.0 - a)
    t = c * (1.0 - a) * (1.0 - b)
    weight = wa * wb * wc * (1.0 - a) ** 2 * (1.0 - b)
    points = torch.stack((r, s, t), dim=-1).reshape(-1, 3)
    return points.to(dtype), weight.reshape(-1).to(dtype)


def tet_consistent_body_force(
    case: ManufacturedElasticity,
    nodes: torch.Tensor,
    corners: torch.Tensor,
    *,
    connectivity: torch.Tensor | None = None,
    quadratic: bool = False,
    order: int = 6,
) -> torch.Tensor:
    """Consistent nodal loads ``f_a = sum_e int_e N_a b dV`` on a tetrahedral mesh.

    ``corners`` holds the four vertex indices of every element (they define the
    geometry); ``connectivity`` optionally holds the full element-to-node map
    (10 entries for TET10) whose first four entries coincide with ``corners``.
    Only affine geometry and shape-function values are used, so the vector is
    independent of the stiffness/``B`` machinery under test.
    """
    dtype, device = nodes.dtype, nodes.device
    conn = corners if connectivity is None else connectivity
    x = nodes[corners.long()].to(dtype)
    origin = x[:, 0]
    edges = torch.stack((x[:, 1] - origin, x[:, 2] - origin, x[:, 3] - origin), dim=2)
    det = torch.linalg.det(edges)
    if torch.any(det <= 0.0):
        raise ValueError("tetrahedron has non-positive Jacobian (check node ordering)")
    points, weights = duffy_tet_rule(order, dtype=dtype)
    points = points.to(device)
    joint = weights.to(dtype).to(device)[None, :] * det[:, None]  # (e,q)
    coords = (
        origin[:, None, :]
        + points[None, :, 0:1] * edges[:, None, :, 0]
        + points[None, :, 1:2] * edges[:, None, :, 1]
        + points[None, :, 2:3] * edges[:, None, :, 2]
    )  # (e,q,3)
    body = case.body_force(coords.reshape(-1, 3)).reshape(len(x), len(points), 3)
    if quadratic:
        shape, _ = tet10_shape(points)  # (q,10)
    else:
        shape = torch.stack(
            (
                1.0 - points[:, 0] - points[:, 1] - points[:, 2],
                points[:, 0],
                points[:, 1],
                points[:, 2],
            ),
            dim=-1,
        )  # (q,4)
    contribution = torch.einsum("qa,eq,eqi->eai", shape, joint, body)
    forces = torch.zeros(len(nodes), 3, dtype=dtype, device=device)
    forces.index_add_(0, conn.reshape(-1).to(device), contribution.reshape(-1, 3))
    return forces.reshape(-1)


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def _boundary_dofs(nodes: torch.Tensor, length: float, width: float, height: float) -> torch.Tensor:
    """All three DOFs of every node on the surface of the box ``[0,L]x[0,W]x[0,H]``.

    The tolerance is a fraction of the smallest node spacing, so it also marks
    midside nodes that lie on a face without collapsing distinct nodes.
    """
    tol = 1e-9 * min(length, width, height)
    x, y, z = nodes.unbind(1)
    on_boundary = (
        (x <= tol)
        | (x >= length - tol)
        | (y <= tol)
        | (y >= width - tol)
        | (z <= tol)
        | (z >= height - tol)
    )
    ids = torch.where(on_boundary)[0]
    return torch.stack((3 * ids, 3 * ids + 1, 3 * ids + 2), dim=1).reshape(-1)


def _tet_mesh(
    length: float,
    width: float,
    height: float,
    n: int,
    *,
    layers: int | None = None,
    dtype: torch.dtype = torch.float64,
):
    """``(2n, n, n)`` box discretised into TET4 elements (six tets per hex)."""
    if int(n) != n or n < 2:
        raise ValueError("divisions must be integers >= 2 (interior nodes are required)")
    nx, ny = 2 * int(n), int(n)
    nz = int(layers) if layers is not None else int(n)
    nodes, hexes = structured_hex_mesh(length, width, height, nx, ny, nz, dtype=dtype)
    _, tets = hex_to_tet_mesh(nodes, hexes)
    return nodes, tets, [nx, ny, nz]


def _relative_l2(numerical: torch.Tensor, exact: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(numerical - exact) / torch.linalg.vector_norm(exact))


def _require_free_dofs(n_free: int, n_nodes: int, mesh: Sequence[int]) -> None:
    """Fail closed on a mesh whose every node is prescribed.

    With all boundary DOFs Dirichlet-constrained and no interior node, the
    "solution" *is* the nodal interpolant and the reported L2 error is exactly
    zero for any element type.  That is a meaningless convergence study, so it
    is rejected instead of being reported as a perfect result.
    """
    if n_free <= 0:
        raise ValueError(
            f"mesh {list(mesh)} has no free DOF ({n_nodes} nodes, all prescribed); "
            "use at least three divisions through every direction so interior "
            "nodes exist, otherwise the L2 error is identically zero"
        )


# ---------------------------------------------------------------------------
# Manufactured-solution studies
# ---------------------------------------------------------------------------


def _manufactured_record(
    case: ManufacturedElasticity,
    nodes: torch.Tensor,
    mesh: Sequence[int],
    numerical: torch.Tensor,
    exact: torch.Tensor,
    energy: float,
    exact_energy: float,
    reference_energy: float,
    n_fixed: int,
    n_elements: int,
    n_nodes: int,
) -> dict[str, object]:
    return {
        "mesh": list(mesh),
        "mesh_size": case.length / mesh[0],
        "n_nodes": int(n_nodes),
        "n_elements": int(n_elements),
        "n_free_dofs": int(3 * n_nodes - n_fixed),
        "relative_l2_displacement_error": _relative_l2(numerical, exact),
        "strain_energy": float(energy),
        "exact_strain_energy": float(exact_energy),
        "reference_energy_independent": float(reference_energy),
        "relative_strain_energy_error": abs(energy - exact_energy) / abs(exact_energy),
    }


def tet4_manufactured_study(
    case: ManufacturedElasticity,
    divisions: Sequence[int],
    *,
    layers: int | None = None,
    order: int = 6,
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, object]]:
    """Manufactured-solution h-convergence on ``(2n, n, n)`` TET4 meshes."""
    if not divisions:
        raise ValueError("divisions must not be empty")
    exact_energy = case.strain_energy_exact()
    reference_energy = _independent_energy(case, order=max(order, 5))
    records: list[dict[str, object]] = []
    for n in divisions:
        nodes, tets, mesh = _tet_mesh(
            case.length, case.width, case.height, n, layers=layers, dtype=dtype
        )
        exact = case.displacement(nodes)
        fixed = _boundary_dofs(nodes, case.length, case.width, case.height)
        _require_free_dofs(int(3 * len(nodes) - fixed.numel()), len(nodes), mesh)
        forces = tet_consistent_body_force(case, nodes, tets, quadratic=False, order=order)
        model = SolidModel(
            nodes,
            tets,
            torch.tensor(case.young_modulus, dtype=dtype),
            torch.tensor(case.poisson_ratio, dtype=dtype),
            forces,
            fixed,
            "tet4",
            exact.reshape(-1)[fixed],
        )
        result = solve_solid(model)
        records.append(
            _manufactured_record(
                case,
                nodes,
                mesh,
                result.displacement.reshape(-1, 3),
                exact,
                float(result.strain_energy),
                exact_energy,
                reference_energy,
                int(fixed.numel()),
                len(tets),
                len(nodes),
            )
        )
    return records


def tet10_manufactured_study(
    case: ManufacturedElasticity,
    divisions: Sequence[int],
    *,
    layers: int | None = None,
    order: int = 6,
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, object]]:
    """Manufactured-solution h-convergence on ``(2n, n, n)`` TET10 meshes.

    The quadratic mesh is upgraded from the conforming TET4 mesh, so midside
    nodes are shared between elements and the discretisation stays conforming.
    """
    if not divisions:
        raise ValueError("divisions must not be empty")
    exact_energy = case.strain_energy_exact()
    reference_energy = _independent_energy(case, order=max(order, 5))
    records: list[dict[str, object]] = []
    for n in divisions:
        nodes, tets, mesh = _tet_mesh(
            case.length, case.width, case.height, n, layers=layers, dtype=dtype
        )
        nodes10, cells = tet4_to_tet10(nodes, tets)
        exact = case.displacement(nodes10)
        fixed = _boundary_dofs(nodes10, case.length, case.width, case.height)
        _require_free_dofs(int(3 * len(nodes10) - fixed.numel()), len(nodes10), mesh)
        forces = tet_consistent_body_force(
            case, nodes10, tets, connectivity=cells, quadratic=True, order=order
        )
        model = Tet10Model(
            nodes10,
            cells,
            torch.tensor(case.young_modulus, dtype=dtype),
            torch.tensor(case.poisson_ratio, dtype=dtype),
            forces,
            fixed,
            exact.reshape(-1)[fixed],
        )
        result = solve_tet10(model)
        records.append(
            _manufactured_record(
                case,
                nodes10,
                mesh,
                result.displacement.reshape(-1, 3),
                exact,
                float(result.strain_energy),
                exact_energy,
                reference_energy,
                int(fixed.numel()),
                len(cells),
                len(nodes10),
            )
        )
    return records


def _independent_energy(case: ManufacturedElasticity, *, order: int = 8) -> float:
    """Strain energy of the manufactured field by a second, higher-order rule.

    ``ManufacturedElasticity.strain_energy_exact`` already uses an independent
    tensor-product Gauss rule; recomputing it at a higher order documents that
    the reference itself is converged rather than tuned to the study.
    """
    return float(case.strain_energy_exact(gauss=max(order, 5)))


# ---------------------------------------------------------------------------
# Pure-bending Dirichlet consistency
# ---------------------------------------------------------------------------


def tet_dirichlet_study(
    oracle: PureBendingOracle,
    divisions: Sequence[int],
    *,
    family: str = "tet4",
    layers: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> list[dict[str, object]]:
    """Pure-bending Dirichlet problem on tetrahedral meshes (no body force).

    Every boundary DOF carries the closed-form displacement; interior DOFs are
    free.  TET4 cannot represent the quadratic field, so its error decays with
    h.  TET10 contains the field exactly, so the discrete solution must equal
    the nodal interpolant to roundoff -- a consistency check, not a bending
    accuracy claim for coarse meshes.
    """
    if family not in ("tet4", "tet10"):
        raise ValueError("family must be 'tet4' or 'tet10'")
    exact_energy = oracle.strain_energy
    records: list[dict[str, object]] = []
    for n in divisions:
        nodes, tets, mesh = _tet_mesh(
            oracle.length, oracle.width, oracle.height, n, layers=layers, dtype=dtype
        )
        cells = tet4_to_tet10(nodes, tets)[1] if family == "tet10" else tets
        active_nodes = tet4_to_tet10(nodes, tets)[0] if family == "tet10" else nodes
        exact = oracle.displacement(active_nodes)
        fixed = _boundary_dofs(active_nodes, oracle.length, oracle.width, oracle.height)
        _require_free_dofs(int(3 * len(active_nodes) - fixed.numel()), len(active_nodes), mesh)
        if family == "tet10":
            model = Tet10Model(
                active_nodes,
                cells,
                torch.tensor(oracle.young_modulus, dtype=dtype),
                torch.tensor(oracle.poisson_ratio, dtype=dtype),
                torch.zeros(3 * len(active_nodes), dtype=dtype),
                fixed,
                exact.reshape(-1)[fixed],
            )
            result = solve_tet10(model)
        else:
            model = SolidModel(
                active_nodes,
                cells,
                torch.tensor(oracle.young_modulus, dtype=dtype),
                torch.tensor(oracle.poisson_ratio, dtype=dtype),
                torch.zeros(3 * len(active_nodes), dtype=dtype),
                fixed,
                "tet4",
                exact.reshape(-1)[fixed],
            )
            result = solve_solid(model)
        numerical = result.displacement.reshape(-1, 3)
        energy = float(result.strain_energy)
        records.append(
            {
                "family": family,
                "mesh": list(mesh),
                "mesh_size": oracle.length / mesh[0],
                "n_nodes": int(len(active_nodes)),
                "n_elements": int(len(cells)),
                "n_free_dofs": int(3 * len(active_nodes) - fixed.numel()),
                "relative_l2_displacement_error": _relative_l2(numerical, exact),
                "relative_max_displacement_error": float(
                    (numerical - exact).abs().max() / exact.abs().max()
                ),
                "strain_energy": energy,
                "exact_strain_energy": float(exact_energy),
                "relative_strain_energy_error": abs(energy - exact_energy) / abs(exact_energy),
            }
        )
    return records
