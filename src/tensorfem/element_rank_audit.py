"""Zero-energy (rank-deficiency) audit of the element library.

A single element assembled without essential boundary conditions must have a
stiffness null space spanned *exactly* by the rigid-body motions the element can
represent.  Every extra null vector is a spurious zero-energy mechanism
("hourglass" mode): a discretisation defect that essential boundary conditions
merely mask, and that becomes a vanishing-stiffness mode of the assembled system.

The audit is deliberately free of empirical tolerances on the *physical* side: the
expected null space is built analytically (:func:`rigid_body_basis_*`) and the
measured null space comes from a symmetric eigendecomposition with a purely
round-off scaled threshold ``|lambda| <= rtol * |lambda|_max``.

Two independent quantities are reported per element:

``spurious``
    ``null_dim - rank(projection of the null space onto the analytic rigid-body
    span)``.  Exact integers, no tolerance tuning: a mode that is 1e-14 away from
    the analytic basis is explained, everything else is counted as spurious.

``first_positive_ratio``
    ``lambda_{r+1} / lambda_max``.  For an assembled mesh this quantity *shrinks*
    with refinement for a near-spurious mechanism, which is the practically
    dangerous signature: the mode is not exactly zero for one element but its
    stiffness vanishes as ``h -> 0``, so it pollutes free/free or rotationally
    unconstrained eigenproblems.

Element ordering conventions are taken from the element modules themselves:
solids ``(ux, uy, uz)`` per node, plane members ``(ux, uy)`` per node, the Mindlin
plate kernel ``(w, theta_x, theta_y)`` per node and the shell facets
``(ux, uy, uz, rx, ry, rz)`` per node.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Callable, Iterable, Sequence

import torch

__all__ = [
    "RankRecord",
    "MeshRecord",
    "rigid_body_basis_solid3d",
    "rigid_body_basis_plane2d",
    "rigid_body_basis_plate",
    "rigid_body_basis_shell",
    "null_space",
    "classify_null_space",
    "audit_element",
    "audit_library",
    "audit_plate_mesh_scan",
    "audit_shell_mesh_scan",
    "summary_table",
]

DEFAULT_RTOL = 1e-10


# --------------------------------------------------------------------------- #
# analytic rigid-body bases
# --------------------------------------------------------------------------- #
def _centroid(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1]).mean(0)


def rigid_body_basis_solid3d(nodes: torch.Tensor) -> torch.Tensor:
    """Return the ``[3*n, 6]`` basis of translations and rotations for 3-D solids."""
    x = nodes.reshape(-1, 3)
    c = _centroid(x)
    d = x - c
    n = len(x)
    basis = torch.zeros((3 * n, 6), dtype=x.dtype, device=x.device)
    for a in range(3):
        basis[a::3, a] = 1.0
    # rotation about the centroid: u = omega x (x - c)
    for k in range(3):
        omega = torch.zeros(3, dtype=x.dtype, device=x.device)
        omega[k] = 1.0
        u = torch.cross(omega.expand(n, 3), d, dim=1)
        basis[:, 3 + k] = u.reshape(-1)
    return basis


def rigid_body_basis_plane2d(nodes: torch.Tensor) -> torch.Tensor:
    """Return the ``[2*n, 3]`` basis for plane elements: 2 translations + 1 rotation."""
    x = nodes.reshape(-1, 2)
    c = _centroid(x)
    d = x - c
    n = len(x)
    basis = torch.zeros((2 * n, 3), dtype=x.dtype, device=x.device)
    basis[0::2, 0] = 1.0
    basis[1::2, 1] = 1.0
    basis[0::2, 2] = -d[:, 1]
    basis[1::2, 2] = d[:, 0]
    return basis


def rigid_body_basis_plate(nodes: torch.Tensor) -> torch.Tensor:
    """Return the ``[3*n, 3]`` basis of the Mindlin kernel in ``(w, tx, ty)`` order.

    Null shear requires ``dw/dx = -tx`` and ``dw/dy = -ty``, so the rigid motions
    are a uniform deflection and the two constant-slope fields.
    """
    x = nodes.reshape(-1, 2)
    c = _centroid(x)
    d = x - c
    n = len(x)
    basis = torch.zeros((3 * n, 3), dtype=x.dtype, device=x.device)
    basis[0::3, 0] = 1.0                       # uniform deflection
    basis[1::3, 1] = 1.0                       # tx constant
    basis[0::3, 1] = -d[:, 0]
    basis[2::3, 2] = 1.0                       # ty constant
    basis[0::3, 2] = -d[:, 1]
    return basis


def rigid_body_basis_shell(nodes: torch.Tensor) -> torch.Tensor:
    """Return the expected null basis of a flat facet shell in ``(u, r)`` per node.

    Six physical rigid-body motions are used: three translations and the two
    rotations about in-plane axes.  Two further zero-energy directions exist by
    construction and must not be counted as defects:

    * the rotation about the facet normal expressed through the membrane
      displacements ``(ux, uy)``;
    * the same rotation expressed through the drilling dof ``rz``, which carries
      no strain at all (only the difference operator stabilises it).

    The expected dimension is therefore 7, not 6.
    """
    x = nodes.reshape(-1, 3)
    c = _centroid(x)
    d = x - c
    n = len(x)
    basis = torch.zeros((6 * n, 7), dtype=x.dtype, device=x.device)
    for a in range(3):
        basis[a::6, a] = 1.0
    # rotations about the in-plane axes: u = omega x (x - c), stored as u and r
    for k, axis in enumerate((0, 1)):
        omega = torch.zeros(3, dtype=x.dtype, device=x.device)
        omega[axis] = 1.0
        u = torch.cross(omega.expand(n, 3), d, dim=1)
        basis[0::6, 3 + k] = u[:, 0]
        basis[1::6, 3 + k] = u[:, 1]
        basis[2::6, 3 + k] = u[:, 2]
        basis[3::6, 3 + k] = omega[0]
        basis[4::6, 3 + k] = omega[1]
        basis[5::6, 3 + k] = omega[2]
    # rotation about the facet normal: membrane representation, then drilling dof
    basis[0::6, 5] = -d[:, 1]
    basis[1::6, 5] = d[:, 0]
    basis[5::6, 6] = 1.0
    return basis


# --------------------------------------------------------------------------- #
# spectra
# --------------------------------------------------------------------------- #
def null_space(k: torch.Tensor, rtol: float = DEFAULT_RTOL) -> dict:
    """Return the round-off scaled null space of a symmetric stiffness matrix."""
    k = 0.5 * (k + k.transpose(-1, -2))
    values = torch.linalg.eigvalsh(k)
    scale = float(values.abs().max())
    threshold = rtol * scale if scale > 0.0 else 0.0
    mask = values.abs() <= threshold
    vectors = torch.linalg.eigh(k).eigenvectors[:, mask]
    first_positive = None
    for value in values[~mask]:
        if float(value) > 0.0:
            first_positive = float(value)
            break
    return {
        "values": values,
        "vectors": vectors,
        "dim": int(mask.sum()),
        "scale": scale,
        "threshold": threshold,
        "first_positive": first_positive,
        "first_positive_ratio": None if first_positive is None else first_positive / scale,
        "min_eigenvalue": float(values.min()),
    }


def classify_null_space(
    k: torch.Tensor, basis: torch.Tensor, rtol: float = DEFAULT_RTOL, explain_rtol: float = 1e-8
) -> dict:
    """Split the null space into the analytic rigid-body span and the rest.

    A zero eigenvalue has multiplicity, so individual eigenvectors inside the null
    space carry no meaning; the comparison therefore uses the principal angles
    between the measured null space and the analytic span.  With ``Q_N`` an
    orthonormal basis of the measured null space and ``Q_R`` of the analytic span,
    the cosines of the principal angles are the singular values of ``Q_N^T Q_R``;
    a rigid-body direction is explained when its cosine is ``1`` to round-off.
    """
    spectrum = null_space(k, rtol)
    vectors = spectrum["vectors"]
    q_rigid, _ = torch.linalg.qr(basis)
    if vectors.shape[1] == 0:
        return {
            "null_dim": 0,
            "explained": 0,
            "spurious": 0,
            "expected_rigid_dim": int(basis.shape[1]),
            "first_positive": spectrum["first_positive"],
            "first_positive_ratio": spectrum["first_positive_ratio"],
            "scale": spectrum["scale"],
            "min_eigenvalue": spectrum["min_eigenvalue"],
            "spurious_vectors": [],
            "cosines": [],
            "values": spectrum["values"],
        }
    cosines = torch.linalg.svdvals(vectors.transpose(0, 1) @ q_rigid)
    explained = int((cosines > 1.0 - explain_rtol).sum())
    # spurious directions: the complement of the rigid span inside the null space
    projector = vectors @ vectors.transpose(0, 1)
    residual_basis = basis - projector @ basis
    spurious_vectors: list[torch.Tensor] = []
    if int(spectrum["dim"]) - explained > 0:
        u, sv, _ = torch.linalg.svd(residual_basis, full_matrices=False)
        keep = torch.nonzero(sv > explain_rtol * max(float(sv.max()), 1e-300)).flatten()
        keep = keep[: int(spectrum["dim"]) - explained]
        for j in keep.tolist():
            spurious_vectors.append(u[:, j])
    return {
        "null_dim": int(spectrum["dim"]),
        "explained": explained,
        "spurious": int(spectrum["dim"]) - explained,
        "expected_rigid_dim": int(basis.shape[1]),
        "first_positive": spectrum["first_positive"],
        "first_positive_ratio": spectrum["first_positive_ratio"],
        "scale": spectrum["scale"],
        "min_eigenvalue": spectrum["min_eigenvalue"],
        "spurious_vectors": spurious_vectors,
        "cosines": [float(c) for c in cosines],
        "values": spectrum["values"],
    }


@dataclass(frozen=True)
class RankRecord:
    """Rank-deficiency record of one element formulation."""

    element: str
    dofs: int
    null_dim: int
    expected_rigid_dim: int
    spurious: int
    first_positive_ratio: float | None
    min_eigenvalue: float
    scale: float
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MeshRecord:
    """Assembled-mesh scan record: how the softest mode scales with refinement."""

    element: str
    n: int
    dofs: int
    null_dim: int
    expected_rigid_dim: int
    spurious: int
    first_positive: float
    first_positive_ratio: float

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# element audit
# --------------------------------------------------------------------------- #
def audit_element(
    element: str, k: torch.Tensor, basis: torch.Tensor, *, note: str = "", rtol: float = DEFAULT_RTOL
) -> RankRecord:
    """Classify the null space of a single free element."""
    report = classify_null_space(k, basis, rtol=rtol)
    return RankRecord(
        element=element,
        dofs=int(k.shape[0]),
        null_dim=report["null_dim"],
        expected_rigid_dim=report["expected_rigid_dim"],
        spurious=report["spurious"],
        first_positive_ratio=report["first_positive_ratio"],
        min_eigenvalue=report["min_eigenvalue"],
        scale=report["scale"],
        note=note,
    )


def _unit_cube(dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ],
        dtype=dtype,
    )


def _distorted_hex(dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.2, 1.0, 0.05], [0.1, 1.0, 0.0],
            [0.05, 0.02, 1.0], [1.0, 0.15, 1.2], [1.1, 1.0, 1.0], [0.0, 1.05, 0.9],
        ],
        dtype=dtype,
    )


def _square(dtype=torch.float64, side: float = 1.0) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]], dtype=dtype
    )


def _distorted_quad(dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0], [2.0, 0.3], [1.5, 1.2], [0.2, 1.0]], dtype=dtype
    )


def audit_library(
    *, young: float = 200.0e9, poisson: float = 0.3, thickness: float = 0.01, rtol: float = DEFAULT_RTOL
) -> list[RankRecord]:
    """Audit every element formulation that exposes an element stiffness matrix."""
    from .continuum import elasticity_matrix, q4_stiffness, cst_stiffness
    from .hex20 import hex20_stiffness
    from .plate import q4_mindlin_stiffness
    from .shell4 import shell4_stiffness
    from .shell4_mitc import shell4_mitc_stiffness
    from .solid3d import elasticity_matrix_3d, hex8_stiffness, tet4_stiffness
    from .solid3d_bbar import hex8_bbar_stiffness

    dtype = torch.float64
    d3 = elasticity_matrix_3d(
        torch.tensor([young], dtype=dtype), torch.tensor([poisson], dtype=dtype)
    )
    d2 = elasticity_matrix(
        torch.tensor([young], dtype=dtype), torch.tensor([poisson], dtype=dtype), "stress"
    )
    t = torch.tensor([thickness], dtype=dtype)
    records: list[RankRecord] = []

    # --- 3-D solids ------------------------------------------------------- #
    cube = _unit_cube(dtype)
    for name, nodes in (("hex8 (2x2x2 integration)", cube), ("hex8 (distorted)", _distorted_hex(dtype))):
        k, _ = hex8_stiffness(nodes.unsqueeze(0), d3)
        records.append(audit_element(name, k[0], rigid_body_basis_solid3d(nodes), rtol=rtol))

    tet = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.1, 0.2, 1.0]], dtype=dtype
    )
    k, _ = tet4_stiffness(tet.unsqueeze(0), d3)
    records.append(audit_element("tet4", k[0], rigid_body_basis_solid3d(tet), rtol=rtol))

    for label, nodes in (("hex8 B-bar (unit cube)", cube), ("hex8 B-bar (distorted)", _distorted_hex(dtype))):
        kb, _ = hex8_bbar_stiffness(nodes.unsqueeze(0), d3)
        records.append(audit_element(label, kb[0], rigid_body_basis_solid3d(nodes), rtol=rtol))

    hex20 = _hex20_cube(dtype)
    k20, _ = hex20_stiffness(hex20.unsqueeze(0), d3)
    records.append(
        audit_element("hex20 (2x2x2 integration)", k20[0], rigid_body_basis_solid3d(hex20), rtol=rtol)
    )

    # --- plane elements --------------------------------------------------- #
    square = _square(dtype)
    km, _ = q4_stiffness(square.unsqueeze(0), d2, t)
    records.append(audit_element("q4 membrane (2x2 integration)", km[0], rigid_body_basis_plane2d(square), rtol=rtol))

    tri = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.2, 1.0]], dtype=dtype)
    kt, _ = cst_stiffness(tri.unsqueeze(0), d2, t)
    records.append(audit_element("cst membrane", kt[0], rigid_body_basis_plane2d(tri), rtol=rtol))

    for treatment in ("sri", "mitc"):
        for shape, nodes in (("square", square), ("distorted", _distorted_quad(dtype))):
            kp = q4_mindlin_stiffness(nodes, young, poisson, thickness, shear_treatment=treatment)
            records.append(
                audit_element(
                    f"q4 Mindlin plate {treatment.upper()} ({shape})",
                    kp,
                    rigid_body_basis_plate(nodes),
                    rtol=rtol,
                )
            )

    # --- flat facet shells ------------------------------------------------ #
    def shell3(nodes2d: torch.Tensor) -> torch.Tensor:
        return torch.cat((nodes2d, torch.zeros(len(nodes2d), 1, dtype=dtype)), 1)

    for builder, name in ((shell4_stiffness, "shell4"), (shell4_mitc_stiffness, "shell4m")):
        for treatment in ("sri", "mitc"):
            for shape, nodes2d in (("square", square), ("distorted", _distorted_quad(dtype))):
                xyz = shell3(nodes2d)
                ks = builder(xyz, young, poisson, thickness, plate_shear=treatment)
                records.append(
                    audit_element(
                        f"{name} facet, plate shear {treatment.upper()} ({shape})",
                        ks,
                        rigid_body_basis_shell(xyz),
                        rtol=rtol,
                    )
                )

    return records


def _hex20_cube(dtype=torch.float64) -> torch.Tensor:
    """Return the 20-node brick on the unit cube in ``hex20.py`` ordering."""
    from .hex20 import structured_hex20_mesh

    nodes, _ = structured_hex20_mesh(1.0, 1.0, 1.0, 1, 1, 1, dtype=dtype)
    return nodes


# --------------------------------------------------------------------------- #
# assembled mesh scans
# --------------------------------------------------------------------------- #
def _assemble(elements: Sequence[torch.Tensor], cells: Iterable[Sequence[int]], ndof: int, per_node: int) -> torch.Tensor:
    k = torch.zeros((ndof, ndof), dtype=torch.float64)
    for cell, ke in zip(cells, elements):
        idx = torch.tensor([per_node * int(n) + a for n in cell for a in range(per_node)])
        k[idx[:, None], idx] = k[idx[:, None], idx] + ke
    return k


def _rect_mesh(n: int, side: float = 1.0) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
    from .plate_vibration import structured_rectangular_mesh

    nodes, cells = structured_rectangular_mesh(n, n, side, side)
    return nodes, [tuple(int(i) for i in row) for row in cells.tolist()]


def audit_plate_mesh_scan(
    *, sizes: Sequence[int] = (1, 2, 3, 4, 6, 8), young: float = 200.0e9, poisson: float = 0.3,
    thickness: float = 0.01, rtol: float = DEFAULT_RTOL, treatment: str = "sri",
) -> list[MeshRecord]:
    """Scan the free (unconstrained) Mindlin plate mesh with refinement."""
    from .plate import q4_mindlin_stiffness

    records: list[MeshRecord] = []
    for n in sizes:
        nodes, cells = _rect_mesh(n)
        elements = [
            q4_mindlin_stiffness(nodes[list(c)], young, poisson, thickness, shear_treatment=treatment)
            for c in cells
        ]
        k = _assemble(elements, cells, 3 * len(nodes), 3)
        report = classify_null_space(k, rigid_body_basis_plate(nodes), rtol=rtol)
        records.append(
            MeshRecord(
                element=f"q4 Mindlin plate {treatment.upper()}",
                n=n,
                dofs=int(k.shape[0]),
                null_dim=report["null_dim"],
                expected_rigid_dim=report["expected_rigid_dim"],
                spurious=report["spurious"],
                first_positive=float(report["first_positive"] or 0.0),
                first_positive_ratio=float(report["first_positive_ratio"] or 0.0),
            )
        )
    return records


def audit_shell_mesh_scan(
    *, sizes: Sequence[int] = (1, 2, 3, 4), formulation: str = "sri", young: float = 200.0e9,
    poisson: float = 0.3, thickness: float = 0.01, rtol: float = DEFAULT_RTOL,
    plate_shear: str = "sri",
) -> list[MeshRecord]:
    """Scan the free (unconstrained) flat facet shell mesh with refinement."""
    from .shell4 import shell4_stiffness
    from .shell4_mitc import shell4_mitc_stiffness

    builder: Callable[..., torch.Tensor] = (
        shell4_stiffness if formulation.lower() == "sri" else shell4_mitc_stiffness
    )
    records: list[MeshRecord] = []
    for n in sizes:
        nodes, cells = _rect_mesh(n)
        xyz = torch.cat((nodes, torch.zeros(len(nodes), 1, dtype=nodes.dtype)), 1)
        elements = [builder(xyz[list(c)], young, poisson, thickness, plate_shear=plate_shear) for c in cells]
        k = _assemble(elements, cells, 6 * len(nodes), 6)
        report = classify_null_space(k, rigid_body_basis_shell(xyz), rtol=rtol)
        records.append(
            MeshRecord(
                element=f"shell4 {'SRI' if formulation.lower() == 'sri' else 'MITC'} facet, plate shear {plate_shear.upper()}",
                n=n,
                dofs=int(k.shape[0]),
                null_dim=report["null_dim"],
                expected_rigid_dim=report["expected_rigid_dim"],
                spurious=report["spurious"],
                first_positive=float(report["first_positive"] or 0.0),
                first_positive_ratio=float(report["first_positive_ratio"] or 0.0),
            )
        )
    return records


def summary_table(records: Sequence[RankRecord | MeshRecord]) -> str:
    """Return a Markdown table of audit records (programmatic, no hand copying)."""
    head = "| element | n | dofs | null dim | expected | spurious | lambda_1/lambda_max |\n"
    head += "|---|---:|---:|---:|---:|---:|---:|\n"
    rows = []
    for r in records:
        n = getattr(r, "n", "-")
        ratio = r.first_positive_ratio
        ratio_text = "-" if ratio is None else f"{ratio:.3e}"
        rows.append(
            f"| {r.element} | {n} | {r.dofs} | {r.null_dim} | {r.expected_rigid_dim} | "
            f"{r.spurious} | {ratio_text} |"
        )
    return head + "\n".join(rows) + "\n"

# --------------------------------------------------------------------------- #
# HEX20 geometry / integration-variant audit (curved, distorted, reduced)
# --------------------------------------------------------------------------- #
#: Gauss rules used by the audit of the twenty-node brick, as
#: ``(points, weights)`` per natural axis.
_HEX20_GAUSS: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {
    2: ((-1.0 / 3.0 ** 0.5, 1.0 / 3.0 ** 0.5), (1.0, 1.0)),
    3: ((-(3.0 / 5.0) ** 0.5, 0.0, (3.0 / 5.0) ** 0.5), (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)),
}


def hex20_gauss_rule(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the ``(points, weights)`` of the ``order``-point 1-D Gauss rule used here."""
    if order not in _HEX20_GAUSS:
        raise ValueError(f"audit supports the 2- and 3-point rules, got {order}")
    return _HEX20_GAUSS[order]


def hex20_jacobian_bounds(nodes: torch.Tensor, *, order: int = 3) -> tuple[float, float]:
    """Return ``(min, max)`` isoparametric Jacobian determinant over the Gauss points.

    The production element only inspects the 3x3x3 quadrature points
    (:func:`tensorfem.hex20._gradient`); the audit reproduces that sampling so the
    reported bounds are the ones the element itself would see.
    """
    from .hex20 import _gradient, hex20_shape_gradients

    x = nodes.reshape(-1, 20, 3)
    points, _ = hex20_gauss_rule(order)
    low, high = float("inf"), float("-inf")
    for r in points:
        for s in points:
            for t in points:
                _, dn = hex20_shape_gradients(r, s, t, like=x)
                _, det = _gradient(x, dn)
                low = min(low, float(det.min()))
                high = max(high, float(det.max()))
    return low, high


def hex20_stiffness_variant(
    nodes: torch.Tensor, d: torch.Tensor, *, order: int = 3
) -> torch.Tensor:
    """Audit-only element stiffness for an arbitrary Gauss order.

    ``order=3`` reproduces :func:`tensorfem.hex20.hex20_stiffness` (the production
    full integration); ``order=2`` is the reduced-integration variant that the
    production module deliberately does *not* offer, instantiated here so its rank
    deficiency can be quantified instead of argued.
    """
    from .hex20 import _b, _gradient, hex20_shape_gradients

    x = nodes.reshape(-1, 20, 3)
    points, weights = hex20_gauss_rule(order)
    ke = x.new_zeros((x.shape[0], 60, 60))
    for i, r in enumerate(points):
        for j, s in enumerate(points):
            for k, t in enumerate(points):
                _, dn = hex20_shape_gradients(r, s, t, like=x)
                g, det = _gradient(x, dn)
                b = _b(g)
                ke = ke + (b.transpose(1, 2) @ d @ b) * (det * weights[i] * weights[j] * weights[k])[:, None, None]
    return ke[0] if nodes.ndim == 2 else ke


def _hex20_corner_grid(dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ],
        dtype=dtype,
    )


def hex20_from_corners(
    corners: torch.Tensor, midside: dict[int, torch.Tensor] | None = None
) -> torch.Tensor:
    """Complete eight corners to the twenty-node ordering of ``hex20.py``.

    ``midside`` maps an edge index (``0..11`` in ``hex20._EDGES`` order) to an explicit
    mid-edge node; the default is the straight-edge midpoint.  Explicit positions are
    how a *curved* (sub-parametric-only-in-position) element is described: the shape
    functions stay the standard 20-node serendipity ones.
    """
    from .hex20 import _EDGES

    corners = corners.reshape(8, 3)
    nodes = [corners[i] for i in range(8)]
    for edge, (a, b) in enumerate(_EDGES):
        if midside is not None and edge in midside:
            nodes.append(midside[edge].reshape(3))
        else:
            nodes.append(0.5 * (corners[a] + corners[b]))
    return torch.stack(nodes)


def _polar(radius: float, angle: float, z: float, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor([radius * torch.cos(torch.tensor(angle)).item(),
                         radius * torch.sin(torch.tensor(angle)).item(), z], dtype=dtype)


def hex20_arc_block(
    angle_deg: float, *, radius: float = 1.0, thickness: float = 0.2, height: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """A single HEX20 block wrapped onto a circular arc of ``angle_deg`` degrees.

    Corner order follows ``hex20.py``: the four ``z = 0`` corners run
    ``(R,0) -> (R,angle) -> (R-t,angle) -> (R-t,0)``, and the arc mid-edge nodes sit on
    the arc at the mean radius, i.e. the element is genuinely curved, not a straight
    chord block.  ``angle_deg = 0`` degenerates to the straight block.
    """
    span = math.radians(abs(angle_deg)) * (1.0 if angle_deg >= 0.0 else -1.0)
    corners = torch.stack(
        [
            _polar(radius, 0.0, 0.0, dtype), _polar(radius, span, 0.0, dtype),
            _polar(radius - thickness, span, 0.0, dtype), _polar(radius - thickness, 0.0, 0.0, dtype),
            _polar(radius, 0.0, height, dtype), _polar(radius, span, height, dtype),
            _polar(radius - thickness, span, height, dtype), _polar(radius - thickness, 0.0, height, dtype),
        ]
    )
    midside: dict[int, torch.Tensor] = {}
    from .hex20 import _EDGES

    for edge, (a, b) in enumerate(_EDGES):
        pa, pb = corners[a], corners[b]
        ra, rb = float(pa[:2].norm()), float(pb[:2].norm())
        same_radius = abs(ra - rb) <= 1e-12 * max(ra, 1.0)
        same_height = abs(float(pa[2] - pb[2])) <= 1e-12
        if same_radius and same_height and abs(span) > 1e-15:
            angle = 0.5 * (math.atan2(float(pa[1]), float(pa[0])) + math.atan2(float(pb[1]), float(pb[0])))
            midside[edge] = _polar(0.5 * (ra + rb), angle, float(pa[2]), dtype)
    return hex20_from_corners(corners, midside)


def hex20_straight_block(
    width: float = 1.0, *, thickness: float = 0.2, height: float = 1.0, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """The flat control block with the same dimensions as :func:`hex20_arc_block`."""
    corners = torch.tensor(
        [
            [0.0, 0.0, 0.0], [width, 0.0, 0.0], [width, thickness, 0.0], [0.0, thickness, 0.0],
            [0.0, 0.0, height], [width, 0.0, height], [width, thickness, height], [0.0, thickness, height],
        ],
        dtype=dtype,
    )
    return hex20_from_corners(corners)


def hex20_arc_mesh(
    sectors: int, angle_deg: float, *, radius: float = 1.0, thickness: float = 0.2, height: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conforming ``sectors``-element HEX20 mesh on an arc (shared mid-edge nodes).

    Conformity matters for the audit: if neighbouring curved elements did not share
    the interpolated arc nodes exactly, the assembled free-free spectrum would gain
    spurious zero modes for reasons unrelated to the element formulation.
    """
    nodes: list[torch.Tensor] = []
    lookup: dict[tuple[float, float, float], int] = {}

    def index(point: torch.Tensor) -> int:
        key = tuple(round(float(v), 12) for v in point.tolist())
        if key not in lookup:
            lookup[key] = len(nodes)
            nodes.append(point)
        return lookup[key]

    cells: list[list[int]] = []
    step = math.radians(abs(angle_deg)) / sectors
    for s in range(sectors):
        element = hex20_arc_block(
            math.degrees(step), radius=radius, thickness=thickness, height=height, dtype=dtype
        )
        shifted = element.clone()
        angle = s * step
        c, sn = math.cos(angle), math.sin(angle)
        rot = torch.tensor([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
        shifted = shifted @ rot.T
        cells.append([index(p) for p in shifted])
    return torch.stack(nodes), torch.tensor(cells, dtype=torch.long)


def _hex20_free_element_record(
    name: str, nodes: torch.Tensor, d: torch.Tensor, *, order: int = 3, rtol: float = DEFAULT_RTOL,
    family: str = "geometry", note: str = "",
) -> dict:
    record = audit_element(name, hex20_stiffness_variant(nodes, d, order=order),
                           rigid_body_basis_solid3d(nodes), note=note, rtol=rtol).as_dict()
    low, high = hex20_jacobian_bounds(nodes, order=order)
    record.update(
        {"family": family, "integration_order": order, "jacobian_min": low, "jacobian_max": high,
         "jacobian_ratio": high / low if low > 0.0 else None}
    )
    return record


def audit_hex20_geometry_sweep(
    *, young: float = 200.0e9, poisson: float = 0.3, rtol: float = DEFAULT_RTOL,
    angles: Sequence[float] = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
    curvature: float = 0.2,
) -> list[dict]:
    """Rank audit of the single HEX20 over distorted / warped / curved geometries."""
    from .solid3d import elasticity_matrix_3d

    dtype = torch.float64
    d = elasticity_matrix_3d(torch.tensor([young], dtype=dtype), torch.tensor([poisson], dtype=dtype))
    records: list[dict] = []

    cube = _hex20_cube(dtype)
    records.append(_hex20_free_element_record("HEX20 unit cube", cube, d, rtol=rtol, note="baseline"))

    rotated = cube @ torch.tensor(
        [[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]], dtype=dtype
    )
    records.append(
        _hex20_free_element_record("HEX20 unit cube, rigidly rotated", rotated, d, rtol=rtol,
                                   note="rank must be rotation invariant")
    )

    perturbed = _hex20_corner_grid(dtype) + 0.6 * torch.tensor(
        [[0.0, 0.0, 0.0], [0.25, 0.1, 0.0], [0.2, 0.05, 0.1], [0.0, 0.2, 0.0],
         [0.05, 0.0, 0.15], [0.3, 0.0, 0.05], [0.2, 0.3, 0.1], [0.0, 0.25, 0.2]], dtype=dtype
    )
    records.append(
        _hex20_free_element_record("HEX20 distorted (straight edges)", hex20_from_corners(perturbed), d,
                                   rtol=rtol)
    )

    warped = _hex20_corner_grid(dtype).clone()
    warped[4:8, 2] = warped[4:8, 2] + torch.tensor([1.0, -0.8, 0.9, -0.7], dtype=dtype)
    records.append(
        _hex20_free_element_record("HEX20 warped (non-planar top face)", hex20_from_corners(warped), d, rtol=rtol)
    )

    for angle in angles:
        nodes = hex20_arc_block(angle, thickness=curvature, dtype=dtype)
        records.append(
            _hex20_free_element_record(f"HEX20 curved arc {angle:g} deg", nodes, d, rtol=rtol,
                                       family="curvature", note="mid-edge nodes on the arc")
        )

    width = 2.0 * math.sin(math.radians(30.0) / 2.0)  # chord of the 30 deg arc at R = 1
    control = hex20_straight_block(width, thickness=curvature, dtype=dtype)
    records.append(
        _hex20_free_element_record("HEX20 straight block, 30 deg chord", control, d, rtol=rtol,
                                   family="curvature", note="aspect-ratio control for the 30 deg arc")
    )
    return records


def audit_hex20_integration_variants(
    *, young: float = 200.0e9, poisson: float = 0.3, rtol: float = DEFAULT_RTOL
) -> list[dict]:
    """Rank audit of full (3x3x3) versus reduced (2x2x2) HEX20 integration."""
    from .solid3d import elasticity_matrix_3d

    dtype = torch.float64
    d = elasticity_matrix_3d(torch.tensor([young], dtype=dtype), torch.tensor([poisson], dtype=dtype))
    records: list[dict] = []
    geometries = (
        ("unit cube", _hex20_cube(dtype)),
        ("distorted", hex20_arc_block(30.0, dtype=dtype)),
    )
    for shape, nodes in geometries:
        for order in (3, 2):
            label = "3x3x3 full" if order == 3 else "2x2x2 reduced"
            records.append(
                _hex20_free_element_record(f"HEX20 {label} ({shape})", nodes, d, order=order, rtol=rtol,
                                           family="integration")
            )
    return records


def audit_hex20_curved_mesh(
    *, sectors: Sequence[int] = (2, 4, 6), angle_deg: float = 90.0, radius: float = 1.0,
    thickness: float = 0.2, height: float = 1.0, young: float = 200.0e9, poisson: float = 0.3,
    rtol: float = DEFAULT_RTOL,
) -> list[MeshRecord]:
    """Free-free spectrum of a conforming curved HEX20 mesh (rigid modes must stay 6)."""
    from .solid3d import elasticity_matrix_3d

    dtype = torch.float64
    d = elasticity_matrix_3d(torch.tensor([young], dtype=dtype), torch.tensor([poisson], dtype=dtype))
    records: list[MeshRecord] = []
    for n in sectors:
        nodes, cells = hex20_arc_mesh(n, angle_deg, radius=radius, thickness=thickness, height=height, dtype=dtype)
        elements = [hex20_stiffness_variant(nodes[cell], d) for cell in cells]
        k = _assemble(elements, [tuple(int(i) for i in cell.tolist()) for cell in cells], 3 * len(nodes), 3)
        report = classify_null_space(k, rigid_body_basis_solid3d(nodes), rtol=rtol)
        records.append(
            MeshRecord(
                element=f"HEX20 curved arc mesh {angle_deg:g} deg",
                n=n,
                dofs=int(k.shape[0]),
                null_dim=report["null_dim"],
                expected_rigid_dim=report["expected_rigid_dim"],
                spurious=report["spurious"],
                first_positive=float(report["first_positive"] or 0.0),
                first_positive_ratio=float(report["first_positive_ratio"] or 0.0),
            )
        )
    return records


def probe_hex20_jacobian_guard(
    *, young: float = 200.0e9, poisson: float = 0.3, angles: Sequence[float] = (30.0, 90.0)
) -> list[dict]:
    """Check that the production HEX20 refuses invalid geometry instead of returning junk.

    Two invalid descriptions are probed: a clockwise-wound arc block (reversed
    corner ordering, which inverts the mapping) and a corner pulled through the
    element so the Jacobian turns negative at a quadrature point.
    """
    from .solid3d import elasticity_matrix_3d

    dtype = torch.float64
    d = elasticity_matrix_3d(torch.tensor([young], dtype=dtype), torch.tensor([poisson], dtype=dtype))
    probes: list[dict] = []
    for angle in angles:
        nodes = hex20_arc_block(-float(angle), dtype=dtype)
        probes.append(_guard_record(f"HEX20 arc {angle:g} deg, reversed winding", nodes, d))
    folded = hex20_arc_block(90.0, dtype=dtype).clone()
    folded[0, 0] -= 0.9
    folded[3, 0] -= 0.9
    probes.append(_guard_record("HEX20 arc 90 deg, corner folded inside", folded, d))
    return probes


def _guard_record(name: str, nodes: torch.Tensor, d: torch.Tensor) -> dict:
    try:
        k = hex20_stiffness_variant(nodes, d, order=3)
    except ValueError as exc:
        return {"element": name, "raised": True, "message": str(exc), "jacobian_min": None}
    low, _ = hex20_jacobian_bounds(nodes, order=3)
    return {"element": name, "raised": False, "message": "accepted", "jacobian_min": low,
            "min_eigenvalue": float(torch.linalg.eigvalsh(0.5 * (k + k.T)).min())}


__all__ = sorted(set(__all__) | {
    "hex20_gauss_rule",
    "hex20_jacobian_bounds",
    "hex20_stiffness_variant",
    "hex20_from_corners",
    "hex20_arc_block",
    "hex20_straight_block",
    "hex20_arc_mesh",
    "audit_hex20_geometry_sweep",
    "audit_hex20_integration_variants",
    "audit_hex20_curved_mesh",
    "probe_hex20_jacobian_guard",
})
