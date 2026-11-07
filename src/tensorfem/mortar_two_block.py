"""Two-deformable-block mortar segment benchmark driver.

The model is a quarter-symmetric stack of two HEX8 blocks that are pressed
together by a uniform pressure on the top face of the upper block.  The
interface is discretised independently on both sides (``n_upper`` slave facets
against ``n_base`` master facets over the same physical area), so the interface
force has to be exchanged by the mortar segment kernel in
:mod:`tensorfem.mortar_segment_contact3d`.

Reference solution
------------------
Two blocks of *identical* material under a uniform top pressure ``p`` admit the
exact state ``sigma_zz = -p``, ``sigma_xx = sigma_yy = 0``: both blocks expand
laterally by the same amount, so the interface stays closed with a uniform
normal traction ``p`` and zero shear.  That uniform state satisfies the quarter
symmetry planes ``x = 0``, ``y = 0`` and the roller bottom ``u_z = 0`` exactly,
which makes the identical-block case a genuine patch test for the mortar
transfer (the reference traction is analytic, not a self-comparison).

A wider elastic base (``base_extent > upper_extent``) breaks that uniformity:
the base restrains the lateral expansion of the punch, the interface pressure
concentrates towards the punch edge, and the central band approaches ``p``.
For finite blocks with free outer faces no closed-form solution exists, so the
edge peak is reported as mesh dependent (see the documentation).
"""

from __future__ import annotations
from dataclasses import dataclass
import torch

from .mortar_segment_contact3d import (
    MortarSegmentSamples,
    assemble_mortar_segment_contact,
    sample_mortar_segment_contact,
)
from .solid3d import elasticity_matrix_3d, hex8_stiffness, structured_hex_mesh


@dataclass(frozen=True)
class TwoBlockModel:
    nodes: torch.Tensor
    elements: torch.Tensor
    stiffness: torch.Tensor
    forces: torch.Tensor
    fixed: torch.Tensor
    slave_nodes: torch.Tensor
    slave_faces: torch.Tensor
    master_nodes: torch.Tensor
    master_faces: torch.Tensor
    pressure: torch.Tensor
    upper_extent: float
    base_extent: float
    height: float
    base_height: float
    n_upper: int
    n_base: int

    @property
    def n_dofs(self) -> int:
        return 3 * int(self.nodes.shape[0])

    @property
    def slave_dofs(self) -> torch.Tensor:
        return _dofs(self.slave_nodes)

    @property
    def master_dofs(self) -> torch.Tensor:
        return _dofs(self.master_nodes)

    @property
    def applied_load(self) -> float:
        """Total compressive load on the upper block (positive)."""
        return abs(float(self.forces.sum()))


def _dofs(node_ids: torch.Tensor) -> torch.Tensor:
    return (3 * node_ids[:, None] + torch.arange(3, device=node_ids.device)[None, :]).reshape(-1)


def _nid(i, j, k, nx, ny):
    return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i


def _surface(nodes, nx, ny, k, nz):
    """Return (global node ids, local quad connectivity) of a structured face."""
    ids, index = [], {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            index[(i, j)] = len(ids)
            ids.append(_nid(i, j, k, nx, ny))
    faces = [
        (index[(i, j)], index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)])
        for j in range(ny)
        for i in range(nx)
    ]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(faces, dtype=torch.long)


def _assemble_stiffness(nodes, elements, young, poisson):
    e = (
        torch.as_tensor(young, dtype=nodes.dtype, device=nodes.device)
        .reshape(-1)
        .expand(len(elements))
    )
    nu = (
        torch.as_tensor(poisson, dtype=nodes.dtype, device=nodes.device)
        .reshape(-1)
        .expand(len(elements))
    )
    d = elasticity_matrix_3d(e, nu)
    ke, _ = hex8_stiffness(nodes[elements], d)
    ed = torch.stack(tuple(3 * elements + i for i in range(3)), 2).reshape(len(elements), 24)
    rows = ed[:, :, None].expand(-1, -1, 24).reshape(-1)
    cols = ed[:, None, :].expand(-1, 24, -1).reshape(-1)
    k = torch.zeros((3 * len(nodes), 3 * len(nodes)), dtype=nodes.dtype)
    return k.index_put((rows, cols), ke.reshape(-1), accumulate=True)


def build_two_block(
    *,
    n_upper: int = 2,
    n_base: int = 2,
    nz: int = 2,
    upper_extent: float = 0.5,
    base_extent: float = 0.5,
    height: float = 1.0,
    pressure: float = 10.0,
    young: float = 1000.0,
    poisson: float = 0.3,
    base_height: float | None = None,
    base_young: float | None = None,
    nz_base: int | None = None,
    dtype=torch.float64,
) -> TwoBlockModel:
    """Build the quarter-symmetric two-block stack.

    The upper block occupies ``[0, upper_extent]^2 x [0, height]``, the lower one
    ``[0, base_extent]^2 x [-height, 0]``; the interface lies on ``z = 0`` with
    the upper block as slave surface (``n_upper`` facets per direction) and the
    lower block as master surface (``n_base`` facets per direction).
    """
    if n_upper < 1 or n_base < 1 or nz < 1:
        raise ValueError("mesh divisions must be positive")
    if upper_extent <= 0 or base_extent < upper_extent or height <= 0:
        raise ValueError("invalid block geometry")
    if pressure <= 0:
        raise ValueError("pressure must be positive")
    base_height = height if base_height is None else float(base_height)
    base_young = young if base_young is None else float(base_young)
    nz_base = nz if nz_base is None else int(nz_base)
    if base_height <= 0 or base_young <= 0 or nz_base < 1:
        raise ValueError("invalid base properties")
    lower, lower_hex = structured_hex_mesh(
        base_extent, base_extent, base_height, n_base, n_base, nz_base, dtype=dtype
    )
    upper, upper_hex = structured_hex_mesh(
        upper_extent, upper_extent, height, n_upper, n_upper, nz, dtype=dtype
    )
    lower = lower.clone()
    lower[:, 2] -= base_height
    upper = upper.clone()
    offset = len(lower)
    nodes = torch.cat((lower, upper), 0)
    elements = torch.cat((lower_hex, upper_hex + offset), 0)
    stiffness = _assemble_stiffness(
        nodes,
        elements,
        torch.cat(
            (
                nodes.new_tensor([base_young]).expand(len(lower_hex)),
                nodes.new_tensor([young]).expand(len(upper_hex)),
            )
        ),
        poisson,
    )

    # quarter-symmetry planes and roller bottom
    tol = 1e-12 * max(height, base_height, base_extent)
    fixed = []
    for axis, plane in ((0, 0.0), (1, 0.0)):
        sel = torch.nonzero((nodes[:, axis] - plane).abs() <= tol, as_tuple=False).reshape(-1)
        fixed.append(3 * sel + axis)
    bottom = torch.nonzero((nodes[:, 2] + base_height).abs() <= tol, as_tuple=False).reshape(-1)
    fixed.append(3 * bottom + 2)
    fixed = torch.unique(torch.cat(fixed))

    # consistent uniform pressure on the top face of the upper block
    top_ids, top_faces = _surface(upper, n_upper, n_upper, nz, nz)
    area = (upper_extent / n_upper) ** 2
    forces = torch.zeros(3 * len(nodes), dtype=dtype)
    for face in top_faces:
        for lid in face:
            forces[3 * (int(top_ids[lid]) + offset) + 2] -= pressure * area / 4.0

    slave_ids, slave_faces = _surface(upper, n_upper, n_upper, 0, nz)
    master_ids, master_faces = _surface(lower, n_base, n_base, nz_base, nz_base)
    return TwoBlockModel(
        nodes,
        elements,
        stiffness,
        forces,
        fixed,
        slave_ids + offset,
        slave_faces,
        master_ids,
        master_faces,
        nodes.new_tensor([pressure]),
        upper_extent,
        base_extent,
        height,
        base_height,
        n_upper,
        n_base,
    )


@dataclass(frozen=True)
class TwoBlockSolution:
    displacement: torch.Tensor
    residual_history: tuple[float, ...]
    samples: MortarSegmentSamples
    interface_area: torch.Tensor

    @property
    def maximum_penetration(self) -> torch.Tensor:
        return self.samples.maximum_penetration


def _scatter_residual(residual, dofs, n_dofs, dtype):
    out = torch.zeros(n_dofs, dtype=dtype)
    return out.index_add(0, dofs, residual)


def solve_two_block(
    model: TwoBlockModel,
    *,
    penalty: float,
    tolerance: float = 1e-10,
    max_iterations: int = 40,
    search_distance: float | None = None,
) -> TwoBlockSolution:
    """Newton solve of ``K u + r_c(u) = f`` with the mortar segment interface."""
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    dtype = model.nodes.dtype
    slave_dofs, master_dofs = model.slave_dofs, model.master_dofs
    contact_dofs = torch.cat((slave_dofs, master_dofs))
    free_mask = torch.ones(model.n_dofs, dtype=torch.bool)
    free_mask[model.fixed] = False
    free = torch.nonzero(free_mask, as_tuple=False).reshape(-1)
    slave_ref = model.nodes[model.slave_nodes]
    master_ref = model.nodes[model.master_nodes]
    u = torch.zeros(model.n_dofs, dtype=dtype)
    reference_scale = float(torch.linalg.vector_norm(model.forces))
    history = []
    for _ in range(max_iterations):
        assembly = assemble_mortar_segment_contact(
            slave_ref,
            model.slave_faces,
            master_ref,
            model.master_faces,
            u[slave_dofs].reshape(-1, 3),
            u[master_dofs].reshape(-1, 3),
            normal_penalty=penalty,
            search_distance=search_distance,
        )
        residual = model.stiffness @ u + _scatter_residual(
            assembly.residual, contact_dofs, model.n_dofs, dtype
        )
        residual = residual - model.forces
        residual[model.fixed] = 0.0
        relative = float(torch.linalg.vector_norm(residual)) / reference_scale
        history.append(relative)
        if relative <= tolerance:
            break
        jacobian = model.stiffness.clone()
        # scatter the dense contact tangent into the global tangent
        block = torch.zeros((model.n_dofs, model.n_dofs), dtype=dtype)
        block[contact_dofs[:, None], contact_dofs[None, :]] = assembly.tangent
        jacobian = jacobian + block
        step = torch.zeros(model.n_dofs, dtype=dtype)
        step[free] = torch.linalg.solve(jacobian[free][:, free], -residual[free])
        u = u + step
        if float(torch.linalg.vector_norm(step)) <= 1e-14 * max(
            1.0, float(torch.linalg.vector_norm(u))
        ):
            break
    displacement = u
    current_slave = slave_ref + displacement[slave_dofs].reshape(-1, 3)
    current_master = master_ref + displacement[master_dofs].reshape(-1, 3)
    samples = sample_mortar_segment_contact(
        current_slave,
        model.slave_faces,
        current_master,
        model.master_faces,
        normal_penalty=penalty,
        search_distance=search_distance,
        slave_reference=slave_ref,
        master_reference=master_ref,
    )
    return TwoBlockSolution(displacement, tuple(history), samples, samples.integrated_area)


def interface_metrics(
    model: TwoBlockModel, solution: TwoBlockSolution, *, band_margin: float = 0.05
) -> dict:
    """Traction uniformity, equilibrium and penetration metrics.

    ``band`` selects quadrature points whose distance to the punch boundary is
    larger than ``band_margin``; the full-interface metrics include the edge
    concentration region, which is mesh dependent for a finite base.
    """
    p = float(model.pressure)
    pts = solution.samples.points
    w = solution.samples.areas
    t = solution.samples.normal_traction
    gap = solution.samples.gaps
    ext = model.upper_extent
    distance = torch.minimum(
        torch.minimum(pts[:, 0], ext - pts[:, 0]), torch.minimum(pts[:, 1], ext - pts[:, 1])
    )
    band = distance >= band_margin
    area = float(w.sum())
    total = float(solution.samples.total_normal_force)
    applied = model.applied_load

    def l2(mask):
        if float(w[mask].sum()) <= 0:
            return float("nan")
        numerator = float((w[mask] * (t[mask] - p) ** 2).sum())
        denominator = (p**2) * float(w[mask].sum())
        return (numerator / denominator) ** 0.5

    band_area = float(w[band].sum())
    band_mean = float((w[band] * t[band]).sum()) / band_area if band_area > 0 else float("nan")
    edge = distance < band_margin
    return {
        "quadrature_points": int(w.numel()),
        "candidate_area": area,
        "contact_area": float(solution.samples.contact_area),
        "expected_area": ext * ext,
        "area_error": abs(float(solution.samples.contact_area) - ext * ext) / (ext * ext),
        "total_normal_force": total,
        "applied_load": applied,
        "equilibrium_error": abs(total - applied) / applied,
        "maximum_penetration": float(solution.samples.maximum_penetration),
        "penetration_to_pressure": float(solution.samples.maximum_penetration) / p,
        "minimum_gap": float(gap.min()),
        "band_mean_traction": band_mean,
        "band_mean_error": abs(band_mean - p) / p,
        "band_l2_error": l2(band),
        "full_l2_error": l2(torch.ones_like(band)),
        "edge_peak_traction": float(t[edge].max()) if bool(edge.any()) else float("nan"),
        "edge_over_pressure": float(t[edge].max()) / p if bool(edge.any()) else float("nan"),
        "closed_fraction": float((gap <= 0).to(gap.dtype).mean()),
        "zero_tension": bool(torch.all(t >= -1e-14)),
        "newton_iterations": len(solution.residual_history),
        "final_relative_residual": solution.residual_history[-1],
    }
