"""Comprehensive mesh quality metrics library.

Provides per-element and global quality indicators for FEM meshes:
- aspect_ratio: edge-length ratio (simplices) or condition-number based (quads/hexes)
- skew: angular deviation from ideal cell shape
- non_orthogonality: angle between face normal and cell-center-to-face-center vector
- jacobian: minimum Jacobian determinant over integration points
- watertightness: whether every interior face is shared by exactly 2 elements
- manifoldness: no face owned by >2 elements; consistent orientation
- orientation: sign of Jacobian / outward-normal consistency

All metrics are computed with torch for batch efficiency and are
cross-validatable against Gmsh/meshmagick reference values.

Acceptance criteria (per GAP-CAD-D1-MESH-QUALITY-LIB):
  ≥3 mesh types,指标差异 <1e-6 vs Gmsh/meshmagick or documented explanation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElementQuality:
    """Quality metrics for a single element."""
    element_id: int
    element_type: str
    aspect_ratio: float
    skew: float  # degrees, 0 = perfect
    non_orthogonality: float  # degrees, 0 = perfectly orthogonal
    jacobian_min: float
    jacobian_max: float
    volume: float
    orientation: str  # "positive" | "negative" | "degenerate"


@dataclass
class MeshQualityReport:
    """Aggregate quality report for an entire mesh."""
    elements: list[ElementQuality] = field(default_factory=list)
    watertight: bool = True
    manifold: bool = True
    manifold_violations: list[tuple[tuple[int, ...], list[int]]] = field(default_factory=list)
    orientation_violations: list[int] = field(default_factory=list)
    # Global extrema
    max_aspect_ratio: float = 1.0
    min_aspect_ratio: float = 1.0
    max_skew: float = 0.0
    max_non_orthogonality: float = 0.0
    min_jacobian: float = math.inf
    max_jacobian: float = -math.inf
    mean_jacobian: float = 0.0
    total_volume: float = 0.0
    n_elements: int = 0
    n_nodes: int = 0

    def to_dict(self) -> dict:
        return {
            "n_elements": self.n_elements,
            "n_nodes": self.n_nodes,
            "watertight": self.watertight,
            "manifold": self.manifold,
            "max_aspect_ratio": self.max_aspect_ratio,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_skew_deg": self.max_skew,
            "max_non_orthogonality_deg": self.max_non_orthogonality,
            "min_jacobian": self.min_jacobian,
            "max_jacobian": self.max_jacobian,
            "mean_jacobian": self.mean_jacobian,
            "total_volume": self.total_volume,
            "manifold_violations": len(self.manifold_violations),
            "orientation_violations": len(self.orientation_violations),
        }


# ---------------------------------------------------------------------------
# Element-type metadata
# ---------------------------------------------------------------------------

# Face definitions for 3D elements (node indices, 0-based)
_HEX_FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7),  # bottom, top
    (0, 1, 5, 4), (1, 2, 6, 5),  # front, right
    (2, 3, 7, 6), (3, 0, 4, 7),  # back, left
]
_TET_FACES = [
    (0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3),
]

# Reference element volumes
_REF_VOLUME = {
    "CPS3": 2.0,   # triangle: reference area = 2 (for Jacobian = 2*area)
    "CPE3": 2.0,
    "CPS4": 4.0,   # quad: reference area = 4 ([-1,1]^2)
    "CPE4": 4.0,
    "C3D4": 6.0,   # tet: reference volume = 6 (for Jacobian = 6*volume)
    "C3D8": 8.0,   # hex: reference volume = 8 ([-1,1]^3)
    "C3D10": 8.0,  # use corner-node hex reference
    "C3D20": 8.0,  # use corner-node hex reference
}


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _coords_3d(nodes: dict[int, tuple[float, ...]], conn: Sequence[int]) -> torch.Tensor:
    """Return (n_nodes, 3) tensor of coordinates, padding 2D with z=0."""
    pts = []
    for nid in conn:
        p = nodes[nid]
        if len(p) == 2:
            pts.append([p[0], p[1], 0.0])
        else:
            pts.append([p[0], p[1], p[2]])
    return torch.tensor(pts, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Aspect ratio
# ---------------------------------------------------------------------------

def _aspect_ratio_simplex(x: torch.Tensor) -> float:
    """Aspect ratio = max_edge / min_edge for triangles and tetrahedra."""
    n = x.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append(float(torch.linalg.vector_norm(x[j] - x[i])))
    mn = min(edges)
    if mn < 1e-15:
        return math.inf
    return max(edges) / mn


def _aspect_ratio_quad(x: torch.Tensor) -> float:
    """Aspect ratio for quads: ratio of max to min edge length."""
    edges = [float(torch.linalg.vector_norm(x[(i + 1) % 4] - x[i])) for i in range(4)]
    mn = min(edges)
    if mn < 1e-15:
        return math.inf
    return max(edges) / mn


def _aspect_ratio_hex(x: torch.Tensor) -> float:
    """Aspect ratio for hex: max/min edge length over all 12 edges."""
    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    edges = [float(torch.linalg.vector_norm(x[j] - x[i])) for i, j in edge_pairs]
    mn = min(edges)
    if mn < 1e-15:
        return math.inf
    return max(edges) / mn


def compute_aspect_ratio(elem_type: str, x: torch.Tensor) -> float:
    et = elem_type.upper()
    if et in ("CPS3", "CPE3", "C3D4"):
        return _aspect_ratio_simplex(x)
    if et in ("CPS4", "CPE4"):
        return _aspect_ratio_quad(x)
    if et in ("C3D8", "C3D10", "C3D20"):
        # For higher-order, use corner nodes only
        if et == "C3D10":
            x = x[:4]
        elif et == "C3D20":
            x = x[[0, 1, 2, 3, 4, 5, 6, 7]]
        return _aspect_ratio_hex(x)
    if et == "T2D2":
        return 1.0  # line element
    return 1.0


# ---------------------------------------------------------------------------
# Skew
# ---------------------------------------------------------------------------

def _skew_triangle(x: torch.Tensor) -> float:
    """Skew = max |angle - 60| for triangles."""
    angles = []
    for i in range(3):
        a = x[(i - 1) % 3] - x[i]
        b = x[(i + 1) % 3] - x[i]
        na = torch.linalg.vector_norm(a)
        nb = torch.linalg.vector_norm(b)
        if na < 1e-15 or nb < 1e-15:
            return 60.0  # degenerate
        cos_val = torch.clamp(torch.dot(a, b) / (na * nb), -1.0, 1.0)
        angles.append(math.degrees(math.acos(float(cos_val))))
    return max(abs(a - 60.0) for a in angles)


def _skew_quad(x: torch.Tensor) -> float:
    """Skew = max |angle - 90| for quads."""
    angles = []
    for i in range(4):
        a = x[(i - 1) % 4] - x[i]
        b = x[(i + 1) % 4] - x[i]
        na = torch.linalg.vector_norm(a)
        nb = torch.linalg.vector_norm(b)
        if na < 1e-15 or nb < 1e-15:
            return 90.0
        cos_val = torch.clamp(torch.dot(a, b) / (na * nb), -1.0, 1.0)
        angles.append(math.degrees(math.acos(float(cos_val))))
    return max(abs(a - 90.0) for a in angles)


def compute_skew(elem_type: str, x: torch.Tensor) -> float:
    et = elem_type.upper()
    if et in ("CPS3", "CPE3"):
        return _skew_triangle(x)
    if et in ("CPS4", "CPE4"):
        return _skew_quad(x)
    # For 3D elements, use max face skew
    if et == "C3D4":
        max_sk = 0.0
        for face in _TET_FACES:
            fx = x[list(face)]
            sk = _skew_triangle(fx)
            max_sk = max(max_sk, sk)
        return max_sk
    if et in ("C3D8", "C3D10", "C3D20"):
        if et == "C3D10":
            x = x[:4]
        elif et == "C3D20":
            x = x[[0, 1, 2, 3, 4, 5, 6, 7]]
        max_sk = 0.0
        for face in _HEX_FACES:
            fx = x[list(face)]
            sk = _skew_quad(fx)
            max_sk = max(max_sk, sk)
        return max_sk
    return 0.0


# ---------------------------------------------------------------------------
# Non-orthogonality
# ---------------------------------------------------------------------------

def _face_normal(x_face: torch.Tensor) -> torch.Tensor:
    """Compute outward-pointing face normal (unnormalized) for a planar face."""
    n = x_face.shape[0]
    if n == 3:
        return torch.cross(x_face[1] - x_face[0], x_face[2] - x_face[0], dim=0)
    if n == 4:
        # Average of two triangle normals
        n1 = torch.cross(x_face[1] - x_face[0], x_face[2] - x_face[0], dim=0)
        n2 = torch.cross(x_face[2] - x_face[0], x_face[3] - x_face[0], dim=0)
        return n1 + n2
    return torch.zeros(3, dtype=torch.float64)


def _non_orthogonality_3d(x: torch.Tensor, faces: list[tuple[int, ...]]) -> float:
    """Max non-orthogonality angle over all faces.

    Non-orthogonality = angle between face normal and the vector from
    cell centroid to face centroid.  0° = perfectly orthogonal.
    """
    centroid = x.mean(dim=0)
    max_angle = 0.0
    for face in faces:
        fx = x[list(face)]
        face_centroid = fx.mean(dim=0)
        normal = _face_normal(fx)
        d = face_centroid - centroid
        n_norm = torch.linalg.vector_norm(normal)
        d_norm = torch.linalg.vector_norm(d)
        if n_norm < 1e-15 or d_norm < 1e-15:
            return 90.0
        cos_val = torch.clamp(torch.dot(normal, d) / (n_norm * d_norm), -1.0, 1.0)
        angle = math.degrees(math.acos(abs(float(cos_val))))
        max_angle = max(max_angle, angle)
    return max_angle


def compute_non_orthogonality(elem_type: str, x: torch.Tensor) -> float:
    et = elem_type.upper()
    if et in ("CPS3", "CPE3", "CPS4", "CPE4", "T2D2"):
        return 0.0  # 2D elements: non-orthogonality not defined in the same sense
    if et == "C3D4":
        return _non_orthogonality_3d(x, _TET_FACES)
    if et in ("C3D8", "C3D10", "C3D20"):
        if et == "C3D10":
            x = x[:4]
        elif et == "C3D20":
            x = x[[0, 1, 2, 3, 4, 5, 6, 7]]
        return _non_orthogonality_3d(x, _HEX_FACES)
    return 0.0


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------

def _jacobian_tri(x: torch.Tensor) -> float:
    """Jacobian = 2 * area for linear triangle (constant over element)."""
    return float(torch.cross(x[1] - x[0], x[2] - x[0], dim=0)[2])


def _jacobian_quad(x: torch.Tensor) -> tuple[float, float]:
    """Min and max Jacobian at 4 Gauss points for bilinear quad.

    The Jacobian determinant maps from reference [-1,1]^2 to physical space.
    For a unit square, det(J) = 0.25 at all Gauss points.
    """
    q = 1.0 / math.sqrt(3.0)
    signs = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64)
    dets = []
    for rr in (-q, q):
        for ss in (-q, q):
            d = torch.empty((4, 2), dtype=torch.float64)
            d[:, 0] = signs[:, 0] * (1 + ss * signs[:, 1]) / 4.0
            d[:, 1] = signs[:, 1] * (1 + rr * signs[:, 0]) / 4.0
            det = float(torch.linalg.det(x[:, :2].T @ d))
            dets.append(det)
    return min(dets), max(dets)


def _jacobian_tet(x: torch.Tensor) -> float:
    """Jacobian = 6 * volume for linear tet (constant)."""
    return float(torch.linalg.det(torch.stack((x[1] - x[0], x[2] - x[0], x[3] - x[0]), dim=1)))


def _jacobian_hex(x: torch.Tensor) -> tuple[float, float]:
    """Min and max Jacobian at 8 Gauss points for trilinear hex.

    The Jacobian determinant maps from reference [-1,1]^3 to physical space.
    For a unit cube, det(J) = 0.125 at all Gauss points.
    """
    q = 1.0 / math.sqrt(3.0)
    signs = torch.tensor([
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ], dtype=torch.float64)
    dets = []
    for r in (-q, q):
        for s in (-q, q):
            for t in (-q, q):
                d = torch.empty((8, 3), dtype=torch.float64)
                d[:, 0] = signs[:, 0] * (1 + s * signs[:, 1]) * (1 + t * signs[:, 2]) / 8.0
                d[:, 1] = signs[:, 1] * (1 + r * signs[:, 0]) * (1 + t * signs[:, 2]) / 8.0
                d[:, 2] = signs[:, 2] * (1 + r * signs[:, 0]) * (1 + s * signs[:, 1]) / 8.0
                det = float(torch.linalg.det(x.T @ d))
                dets.append(det)
    return min(dets), max(dets)


def compute_jacobian(elem_type: str, x: torch.Tensor) -> tuple[float, float]:
    """Return (min_jacobian, max_jacobian)."""
    et = elem_type.upper()
    if et in ("CPS3", "CPE3"):
        j = _jacobian_tri(x)
        return j, j
    if et in ("CPS4", "CPE4"):
        return _jacobian_quad(x)
    if et == "C3D4":
        j = _jacobian_tet(x)
        return j, j
    if et in ("C3D8", "C3D10", "C3D20"):
        if et == "C3D10":
            x = x[:4]
        elif et == "C3D20":
            x = x[[0, 1, 2, 3, 4, 5, 6, 7]]
        return _jacobian_hex(x)
    if et == "T2D2":
        return float(torch.linalg.vector_norm(x[1] - x[0])), float(torch.linalg.vector_norm(x[1] - x[0]))
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Volume / area
# ---------------------------------------------------------------------------

def compute_volume(elem_type: str, x: torch.Tensor) -> float:
    et = elem_type.upper()
    if et in ("CPS3", "CPE3"):
        return abs(float(torch.cross(x[1] - x[0], x[2] - x[0], dim=0)[2])) / 2.0
    if et in ("CPS4", "CPE4"):
        # Shoelace formula for quad area
        return abs(float(
            (x[0, 0] * x[1, 1] - x[1, 0] * x[0, 1]) +
            (x[1, 0] * x[2, 1] - x[2, 0] * x[1, 1]) +
            (x[2, 0] * x[3, 1] - x[3, 0] * x[2, 1]) +
            (x[3, 0] * x[0, 1] - x[0, 0] * x[3, 1])
        )) / 2.0
    if et == "C3D4":
        return abs(_jacobian_tet(x)) / 6.0
    if et in ("C3D8", "C3D10", "C3D20"):
        if et == "C3D10":
            x = x[:4]
        elif et == "C3D20":
            x = x[[0, 1, 2, 3, 4, 5, 6, 7]]
        # Volume = det(J) * reference_volume / n_gauss_points
        # For constant Jacobian (perfect hex): vol = det(J) * 8
        j_min, j_max = _jacobian_hex(x)
        return abs(j_max) * 8.0  # reference volume of [-1,1]^3 is 8
    if et == "T2D2":
        return float(torch.linalg.vector_norm(x[1] - x[0]))
    return 0.0


# ---------------------------------------------------------------------------
# Watertightness & manifoldness
# ---------------------------------------------------------------------------

def _face_key(face_nodes: tuple[int, ...]) -> tuple[int, ...]:
    """Canonical face key (sorted node IDs) for face-sharing detection."""
    return tuple(sorted(face_nodes))


def check_watertight_and_manifold(
    nodes: dict[int, tuple[float, ...]],
    elements: list[tuple[int, str, list[int]]],  # (eid, etype, conn)
) -> tuple[bool, bool, list[tuple[tuple[int, ...], list[int]]]]:
    """Check watertightness and manifoldness of a mesh.

    Returns (watertight, manifold, violations).
    watertight: every face is shared by exactly 2 elements (no boundary faces).
    manifold: no face shared by >2 elements.
    violations: list of (face_key, [owner_eids]) for non-manifold faces.

    Note: A mesh with boundary (e.g., a single element or a partial mesh)
    is NOT watertight by this strict definition. Watertight means the mesh
    has no boundary faces at all (e.g., a periodic mesh or a closed volume
    where the boundary is handled by a separate surface mesh).
    """
    face_owners: dict[tuple[int, ...], list[int]] = {}

    for eid, etype, conn in elements:
        et = etype.upper()
        faces: list[tuple[int, ...]] = []
        if et == "C3D4":
            faces = [(conn[i], conn[j], conn[k]) for i, j, k in _TET_FACES]
        elif et in ("C3D8", "C3D10", "C3D20"):
            if et == "C3D10":
                conn = conn[:4]
            elif et == "C3D20":
                conn = [conn[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]]
            faces = [tuple(conn[i] for i in f) for f in _HEX_FACES]
        elif et in ("CPS3", "CPE3"):
            faces = [(conn[i], conn[(i + 1) % 3]) for i in range(3)]
        elif et in ("CPS4", "CPE4"):
            faces = [(conn[i], conn[(i + 1) % 4]) for i in range(4)]
        else:
            continue

        for f in faces:
            key = _face_key(f)
            face_owners.setdefault(key, []).append(eid)

    violations = []
    watertight = True
    manifold = True

    for key, owners in face_owners.items():
        if len(owners) > 2:
            manifold = False
            violations.append((key, owners))
        if len(owners) == 1:
            # Boundary face - mesh is not watertight
            watertight = False

    return watertight, manifold, violations


# ---------------------------------------------------------------------------
# Orientation check
# ---------------------------------------------------------------------------

def check_orientation(
    nodes: dict[int, tuple[float, ...]],
    elements: list[tuple[int, str, list[int]]],
) -> list[int]:
    """Return list of element IDs with negative/degenerate Jacobian."""
    bad = []
    for eid, etype, conn in elements:
        x = _coords_3d(nodes, conn)
        j_min, _ = compute_jacobian(etype, x)
        if j_min <= 0:
            bad.append(eid)
    return bad


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_mesh_quality(
    nodes: dict[int, tuple[float, ...]],
    elements: list[tuple[int, str, list[int]]],
) -> MeshQualityReport:
    """Evaluate comprehensive mesh quality for a set of nodes and elements.

    Parameters
    ----------
    nodes : dict mapping node_id -> (x, y) or (x, y, z)
    elements : list of (element_id, element_type, connectivity)

    Returns
    -------
    MeshQualityReport with per-element and aggregate metrics.
    """
    report = MeshQualityReport(n_nodes=len(nodes), n_elements=len(elements))

    jac_sum = 0.0
    jac_count = 0

    for eid, etype, conn in elements:
        x = _coords_3d(nodes, conn)
        ar = compute_aspect_ratio(etype, x)
        sk = compute_skew(etype, x)
        no = compute_non_orthogonality(etype, x)
        j_min, j_max = compute_jacobian(etype, x)
        vol = compute_volume(etype, x)

        if j_min > 0:
            orient = "positive"
        elif j_min < 0:
            orient = "negative"
        else:
            orient = "degenerate"

        eq = ElementQuality(
            element_id=eid,
            element_type=etype,
            aspect_ratio=ar,
            skew=sk,
            non_orthogonality=no,
            jacobian_min=j_min,
            jacobian_max=j_max,
            volume=vol,
            orientation=orient,
        )
        report.elements.append(eq)

        report.max_aspect_ratio = max(report.max_aspect_ratio, ar)
        report.min_aspect_ratio = min(report.min_aspect_ratio, ar)
        report.max_skew = max(report.max_skew, sk)
        report.max_non_orthogonality = max(report.max_non_orthogonality, no)
        report.min_jacobian = min(report.min_jacobian, j_min)
        report.max_jacobian = max(report.max_jacobian, j_max)
        if j_min > 0:
            jac_sum += j_min
            jac_count += 1
        report.total_volume += abs(vol)

        if orient == "negative":
            report.orientation_violations.append(eid)

    if jac_count > 0:
        report.mean_jacobian = jac_sum / jac_count

    # Watertightness and manifoldness (only meaningful for 3D)
    has_3d = any(e[1].upper().startswith("C3D") for e in elements)
    if has_3d:
        report.watertight, report.manifold, report.manifold_violations = \
            check_watertight_and_manifold(nodes, elements)

    return report


# ---------------------------------------------------------------------------
# Compatibility with mesh_pipeline.ModelDB
# ---------------------------------------------------------------------------

def evaluate_modeldb_quality(db) -> MeshQualityReport:
    """Evaluate mesh quality from a ModelDB instance."""
    elements = []
    for block in db.elements:
        for eid, conn in zip(block.ids, block.connectivity):
            elements.append((eid, block.element_type, conn))
    return evaluate_mesh_quality(db.nodes, elements)
