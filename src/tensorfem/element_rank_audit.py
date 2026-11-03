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