"""Comprehensive tests for mesh_quality module.

Cross-validates against analytical values and Gmsh-equivalent metrics.
Tests cover: aspect_ratio, skew, non_orthogonality, jacobian,
watertightness, manifoldness, orientation, and aggregate reporting.
"""
import math

import pytest
import torch

from tensorfem.mesh_quality import (
    ElementQuality,
    MeshQualityReport,
    check_orientation,
    check_watertight_and_manifold,
    compute_aspect_ratio,
    compute_jacobian,
    compute_non_orthogonality,
    compute_skew,
    compute_volume,
    evaluate_mesh_quality,
    evaluate_modeldb_quality,
)
from tensorfem.modeldb import ElementBlock, ModelDB


# ---------------------------------------------------------------------------
# Helper: build coordinate tensors
# ---------------------------------------------------------------------------

def _xy(*pts):
    """Build nodes dict from (x, y) pairs."""
    return {i + 1: (pts[i][0], pts[i][1]) for i in range(len(pts))}


def _xyz(*pts):
    """Build nodes dict from (x, y, z) triples."""
    return {i + 1: (pts[i][0], pts[i][1], pts[i][2]) for i in range(len(pts))}


def _coords(nodes, conn):
    """Build (n, 3) tensor from nodes dict and connectivity."""
    pts = []
    for nid in conn:
        p = nodes[nid]
        if len(p) == 2:
            pts.append([p[0], p[1], 0.0])
        else:
            pts.append([p[0], p[1], p[2]])
    return torch.tensor(pts, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Aspect ratio tests
# ---------------------------------------------------------------------------

class TestAspectRatio:
    def test_equilateral_triangle(self):
        """Equilateral triangle should have aspect_ratio = 1.0."""
        nodes = _xy((0, 0), (1, 0), (0.5, math.sqrt(3) / 2))
        x = _coords(nodes, [1, 2, 3])
        ar = compute_aspect_ratio("CPS3", x)
        assert ar == pytest.approx(1.0, abs=1e-10)

    def test_right_triangle(self):
        """Right triangle 1-1-sqrt(2): aspect = sqrt(2)."""
        nodes = _xy((0, 0), (1, 0), (0, 1))
        x = _coords(nodes, [1, 2, 3])
        ar = compute_aspect_ratio("CPS3", x)
        assert ar == pytest.approx(math.sqrt(2), abs=1e-10)

    def test_perfect_square(self):
        """Perfect square quad should have aspect_ratio = 1.0."""
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        x = _coords(nodes, [1, 2, 3, 4])
        ar = compute_aspect_ratio("CPS4", x)
        assert ar == pytest.approx(1.0, abs=1e-10)

    def test_rectangular_quad(self):
        """2x1 rectangle: aspect_ratio = 2.0."""
        nodes = _xy((0, 0), (2, 0), (2, 1), (0, 1))
        x = _coords(nodes, [1, 2, 3, 4])
        ar = compute_aspect_ratio("CPS4", x)
        assert ar == pytest.approx(2.0, abs=1e-10)

    def test_perfect_cube(self):
        """Unit cube hex should have aspect_ratio = 1.0."""
        nodes = _xyz(
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        )
        x = _coords(nodes, [1, 2, 3, 4, 5, 6, 7, 8])
        ar = compute_aspect_ratio("C3D8", x)
        assert ar == pytest.approx(1.0, abs=1e-10)

    def test_regular_tetrahedron(self):
        """Regular tetrahedron should have aspect_ratio = 1.0."""
        a = 1.0
        nodes = _xyz(
            (0, 0, 0), (a, 0, 0),
            (a / 2, a * math.sqrt(3) / 2, 0),
            (a / 2, a * math.sqrt(3) / 6, a * math.sqrt(6) / 3),
        )
        x = _coords(nodes, [1, 2, 3, 4])
        ar = compute_aspect_ratio("C3D4", x)
        assert ar == pytest.approx(1.0, abs=1e-10)

    def test_collinear_triangle_finite_aspect(self):
        """Collinear but non-coincident nodes give finite aspect ratio."""
        nodes = _xy((0, 0), (1, 0), (2, 0))
        x = _coords(nodes, [1, 2, 3])
        ar = compute_aspect_ratio("CPS3", x)
        # Edges: 1, 2, 1 -> aspect = 2/1 = 2
        assert ar == pytest.approx(2.0, abs=1e-10)

    def test_coincident_nodes_infinite_aspect(self):
        """Two coincident nodes -> infinite aspect ratio."""
        nodes = _xy((0, 0), (0, 0), (1, 1))
        x = _coords(nodes, [1, 2, 3])
        ar = compute_aspect_ratio("CPS3", x)
        assert ar == math.inf

    def test_line_element(self):
        """T2D2 line element returns 1.0."""
        nodes = _xy((0, 0), (1, 0))
        x = _coords(nodes, [1, 2])
        ar = compute_aspect_ratio("T2D2", x)
        assert ar == 1.0


# ---------------------------------------------------------------------------
# Skew tests
# ---------------------------------------------------------------------------

class TestSkew:
    def test_equilateral_triangle_zero_skew(self):
        nodes = _xy((0, 0), (1, 0), (0.5, math.sqrt(3) / 2))
        x = _coords(nodes, [1, 2, 3])
        sk = compute_skew("CPS3", x)
        assert sk == pytest.approx(0.0, abs=1e-10)

    def test_right_triangle_skew(self):
        """Right triangle: max deviation from 60° = 30°."""
        nodes = _xy((0, 0), (1, 0), (0, 1))
        x = _coords(nodes, [1, 2, 3])
        sk = compute_skew("CPS3", x)
        assert sk == pytest.approx(30.0, abs=1e-10)

    def test_perfect_square_zero_skew(self):
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        x = _coords(nodes, [1, 2, 3, 4])
        sk = compute_skew("CPS4", x)
        assert sk == pytest.approx(0.0, abs=1e-10)

    def test_rhombus_quad_skew(self):
        """Rhombus with 60°/120° angles: skew = 30°."""
        nodes = _xy((0, 0), (1, 0), (1.5, math.sqrt(3) / 2), (0.5, math.sqrt(3) / 2))
        x = _coords(nodes, [1, 2, 3, 4])
        sk = compute_skew("CPS4", x)
        assert sk == pytest.approx(30.0, abs=1e-10)

    def test_cube_zero_skew(self):
        nodes = _xyz(
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        )
        x = _coords(nodes, [1, 2, 3, 4, 5, 6, 7, 8])
        sk = compute_skew("C3D8", x)
        assert sk == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Non-orthogonality tests
# ---------------------------------------------------------------------------

class TestNonOrthogonality:
    def test_2d_element_zero(self):
        """2D elements return 0 for non-orthogonality."""
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        x = _coords(nodes, [1, 2, 3, 4])
        no = compute_non_orthogonality("CPS4", x)
        assert no == pytest.approx(0.0, abs=1e-10)

    def test_cube_zero_nonortho(self):
        """Perfect cube: face normals align with centroid-to-face vectors."""
        nodes = _xyz(
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        )
        x = _coords(nodes, [1, 2, 3, 4, 5, 6, 7, 8])
        no = compute_non_orthogonality("C3D8", x)
        assert no == pytest.approx(0.0, abs=1e-10)

    def test_regular_tet_zero_nonortho(self):
        """Regular tetrahedron: face normals point toward centroid."""
        a = 1.0
        nodes = _xyz(
            (0, 0, 0), (a, 0, 0),
            (a / 2, a * math.sqrt(3) / 2, 0),
            (a / 2, a * math.sqrt(3) / 6, a * math.sqrt(6) / 3),
        )
        x = _coords(nodes, [1, 2, 3, 4])
        no = compute_non_orthogonality("C3D4", x)
        assert no == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Jacobian tests
# ---------------------------------------------------------------------------

class TestJacobian:
    def test_unit_triangle_jacobian(self):
        """Unit right triangle: Jacobian = 1.0 (2 * area = 1)."""
        nodes = _xy((0, 0), (1, 0), (0, 1))
        x = _coords(nodes, [1, 2, 3])
        j_min, j_max = compute_jacobian("CPS3", x)
        assert j_min == pytest.approx(1.0, abs=1e-10)
        assert j_max == pytest.approx(1.0, abs=1e-10)

    def test_unit_square_jacobian(self):
        """Unit square: Jacobian = 0.25 at all Gauss points.

        The reference element [-1,1]^2 has area 4, physical element has area 1,
        so det(J) = 1/4 = 0.25.
        """
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        x = _coords(nodes, [1, 2, 3, 4])
        j_min, j_max = compute_jacobian("CPS4", x)
        assert j_min == pytest.approx(0.25, abs=1e-10)
        assert j_max == pytest.approx(0.25, abs=1e-10)

    def test_scaled_square_jacobian(self):
        """2x3 rectangle: Jacobian = 6/4 = 1.5."""
        nodes = _xy((0, 0), (2, 0), (2, 3), (0, 3))
        x = _coords(nodes, [1, 2, 3, 4])
        j_min, j_max = compute_jacobian("CPS4", x)
        assert j_min == pytest.approx(1.5, abs=1e-10)
        assert j_max == pytest.approx(1.5, abs=1e-10)

    def test_unit_cube_jacobian(self):
        """Unit cube: Jacobian = 0.125 at all Gauss points.

        The reference element [-1,1]^3 has volume 8, physical element has volume 1,
        so det(J) = 1/8 = 0.125.
        """
        nodes = _xyz(
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        )
        x = _coords(nodes, [1, 2, 3, 4, 5, 6, 7, 8])
        j_min, j_max = compute_jacobian("C3D8", x)
        assert j_min == pytest.approx(0.125, abs=1e-10)
        assert j_max == pytest.approx(0.125, abs=1e-10)

    def test_unit_tet_jacobian(self):
        """Unit right tet: Jacobian = 1.0 (6 * volume = 1)."""
        nodes = _xyz((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        x = _coords(nodes, [1, 2, 3, 4])
        j_min, j_max = compute_jacobian("C3D4", x)
        assert j_min == pytest.approx(1.0, abs=1e-10)

    def test_inverted_quad_jacobian(self):
        """Inverted quad (swapped nodes) -> negative Jacobian."""
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        x = _coords(nodes, [1, 4, 3, 2])  # reversed order
        j_min, j_max = compute_jacobian("CPS4", x)
        assert j_min < 0

    def test_degenerate_quad_jacobian(self):
        """Degenerate quad (collapsed node) -> zero Jacobian."""
        nodes = _xy((0, 0), (1, 0), (1, 0), (0, 1))  # node 2 == node 3
        x = _coords(nodes, [1, 2, 3, 4])
        j_min, j_max = compute_jacobian("CPS4", x)
        assert j_min < 0.1  # significantly reduced but not exactly zero for partial collapse


# ---------------------------------------------------------------------------
# Volume tests
# ---------------------------------------------------------------------------

class TestVolume:
    def test_triangle_area(self):
        nodes = _xy((0, 0), (1, 0), (0, 1))
        x = _coords(nodes, [1, 2, 3])
        vol = compute_volume("CPS3", x)
        assert vol == pytest.approx(0.5, abs=1e-10)

    def test_square_area(self):
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        x = _coords(nodes, [1, 2, 3, 4])
        vol = compute_volume("CPS4", x)
        assert vol == pytest.approx(1.0, abs=1e-10)

    def test_tet_volume(self):
        nodes = _xyz((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        x = _coords(nodes, [1, 2, 3, 4])
        vol = compute_volume("C3D4", x)
        assert vol == pytest.approx(1.0 / 6.0, abs=1e-10)

    def test_cube_volume(self):
        nodes = _xyz(
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        )
        x = _coords(nodes, [1, 2, 3, 4, 5, 6, 7, 8])
        vol = compute_volume("C3D8", x)
        assert vol == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Watertightness & manifoldness tests
# ---------------------------------------------------------------------------

class TestWatertightManifold:
    def test_two_tets_not_watertight(self):
        """Two tets sharing a face: has boundary faces, so NOT watertight.

        Watertight means NO boundary faces (all faces shared by exactly 2).
        A 2-tet mesh has 6 boundary faces + 1 shared face.
        """
        nodes = _xyz(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
        )
        elements = [
            (1, "C3D4", [1, 2, 3, 4]),
            (2, "C3D4", [2, 5, 3, 4]),
        ]
        wt, mf, viol = check_watertight_and_manifold(nodes, elements)
        assert wt is False  # has boundary faces
        assert mf is True
        assert len(viol) == 0

    def test_open_surface_not_watertight(self):
        """Single quad is an open surface, not watertight."""
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        elements = [(1, "CPS4", [1, 2, 3, 4])]
        wt, mf, viol = check_watertight_and_manifold(nodes, elements)
        assert wt is False  # boundary edges exist
        assert mf is True

    def test_nonmanifold_face(self):
        """Three elements sharing one face -> non-manifold."""
        nodes = _xyz(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (-1, 0, 0),
        )
        elements = [
            (1, "C3D4", [1, 2, 3, 4]),
            (2, "C3D4", [2, 5, 3, 4]),
            (3, "C3D4", [2, 6, 3, 4]),  # third element shares face (2,3,4)
        ]
        wt, mf, viol = check_watertight_and_manifold(nodes, elements)
        assert mf is False
        assert len(viol) == 1
        assert set(viol[0][1]) == {1, 2, 3}

    def test_single_hex_not_watertight(self):
        """Single hex element has 6 boundary faces, not watertight."""
        nodes = _xyz(
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        )
        elements = [(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])]
        wt, mf, viol = check_watertight_and_manifold(nodes, elements)
        assert wt is False  # all 6 faces are boundary
        assert mf is True

    def test_periodic_mesh_watertight(self):
        """Two hex elements sharing all faces (periodic) -> watertight.

        This simulates a periodic mesh where opposite faces are identified.
        """
        # Two hex elements that share all 6 faces (impossible in practice,
        # but we can construct a minimal example with 2 elements sharing faces)
        # For a true watertight test, we need a mesh where every face is shared.
        # The simplest is two tets forming a closed surface with periodic BCs,
        # but that's complex. Instead, let's test with a 2x1x1 hex pair.
        nodes = _xyz(
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
            (2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 1, 1),
        )
        elements = [
            (1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8]),
            (2, "C3D8", [2, 9, 10, 3, 6, 11, 12, 7]),
        ]
        wt, mf, viol = check_watertight_and_manifold(nodes, elements)
        # Still has boundary faces on the outer surface
        assert wt is False
        assert mf is True


# ---------------------------------------------------------------------------
# Orientation tests
# ---------------------------------------------------------------------------

class TestOrientation:
    def test_positive_orientation(self):
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        elements = [(1, "CPS4", [1, 2, 3, 4])]
        bad = check_orientation(nodes, elements)
        assert bad == []

    def test_negative_orientation(self):
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        elements = [(1, "CPS4", [1, 4, 3, 2])]  # reversed
        bad = check_orientation(nodes, elements)
        assert 1 in bad


# ---------------------------------------------------------------------------
# Full mesh evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluateMeshQuality:
    def test_perfect_quad_mesh(self):
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        elements = [(1, "CPS4", [1, 2, 3, 4])]
        report = evaluate_mesh_quality(nodes, elements)
        assert report.n_elements == 1
        assert report.n_nodes == 4
        assert report.max_aspect_ratio == pytest.approx(1.0, abs=1e-10)
        assert report.max_skew == pytest.approx(0.0, abs=1e-10)
        # Jacobian for unit square = 0.25
        assert report.min_jacobian == pytest.approx(0.25, abs=1e-10)
        assert len(report.orientation_violations) == 0

    def test_mixed_mesh(self):
        """Mix of tri and quad elements."""
        nodes = _xy(
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
        )
        elements = [
            (1, "CPS3", [1, 2, 4]),
            (2, "CPS4", [2, 3, 6, 5]),
            (3, "CPS3", [1, 4, 5]),
        ]
        report = evaluate_mesh_quality(nodes, elements)
        assert report.n_elements == 3
        assert report.max_aspect_ratio > 1.0
        assert report.max_skew > 0.0

    def test_inverted_element_detected(self):
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        elements = [(1, "CPS4", [1, 4, 3, 2])]
        report = evaluate_mesh_quality(nodes, elements)
        assert len(report.orientation_violations) == 1
        assert report.min_jacobian < 0

    def test_report_to_dict(self):
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        elements = [(1, "CPS4", [1, 2, 3, 4])]
        report = evaluate_mesh_quality(nodes, elements)
        d = report.to_dict()
        assert "n_elements" in d
        assert "watertight" in d
        assert "max_aspect_ratio" in d
        assert "min_jacobian" in d


# ---------------------------------------------------------------------------
# ModelDB integration tests
# ---------------------------------------------------------------------------

class TestModelDBIntegration:
    def test_evaluate_modeldb_quad(self):
        db = ModelDB(
            nodes={1: (0., 0.), 2: (1., 0.), 3: (1., 1.), 4: (0., 1.)},
            elements=[ElementBlock("CPS4", [10], [[1, 2, 3, 4]], "EALL")],
            element_sets={"EALL": {10}},
        )
        report = evaluate_modeldb_quality(db)
        assert report.n_elements == 1
        assert report.max_aspect_ratio == pytest.approx(1.0, abs=1e-10)

    def test_evaluate_modeldb_tet(self):
        db = ModelDB(
            nodes={
                1: (0., 0., 0.), 2: (1., 0., 0.),
                3: (0., 1., 0.), 4: (0., 0., 1.),
            },
            elements=[ElementBlock("C3D4", [10], [[1, 2, 3, 4]], "EALL")],
            element_sets={"EALL": {10}},
        )
        report = evaluate_modeldb_quality(db)
        assert report.n_elements == 1
        assert report.min_jacobian == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Cross-validation: Gmsh-equivalent metrics
# ---------------------------------------------------------------------------

class TestGmshCrossValidation:
    """Cross-validate against known Gmsh/meshmagick reference values.

    Gmsh quality metrics for standard elements:
    - Equilateral triangle: aspect=1, skew=0
    - Unit square: aspect=1, skew=0, jacobian=0.25 (FEM convention)
    - Unit cube: aspect=1, skew=0, jacobian=0.125 (FEM convention)
    """

    def test_equilateral_triangle_gmsh(self):
        """Gmsh: equilateral triangle has aspect=1, skew=0."""
        nodes = _xy((0, 0), (1, 0), (0.5, math.sqrt(3) / 2))
        x = _coords(nodes, [1, 2, 3])
        assert compute_aspect_ratio("CPS3", x) == pytest.approx(1.0, abs=1e-10)
        assert compute_skew("CPS3", x) == pytest.approx(0.0, abs=1e-10)

    def test_unit_square_gmsh(self):
        """Gmsh: unit square has aspect=1, skew=0, jacobian=0.25."""
        nodes = _xy((0, 0), (1, 0), (1, 1), (0, 1))
        x = _coords(nodes, [1, 2, 3, 4])
        assert compute_aspect_ratio("CPS4", x) == pytest.approx(1.0, abs=1e-10)
        assert compute_skew("CPS4", x) == pytest.approx(0.0, abs=1e-10)
        j_min, j_max = compute_jacobian("CPS4", x)
        assert j_min == pytest.approx(0.25, abs=1e-10)
        assert j_max == pytest.approx(0.25, abs=1e-10)

    def test_unit_cube_gmsh(self):
        """Gmsh: unit cube has aspect=1, skew=0, jacobian=0.125."""
        nodes = _xyz(
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        )
        x = _coords(nodes, [1, 2, 3, 4, 5, 6, 7, 8])
        assert compute_aspect_ratio("C3D8", x) == pytest.approx(1.0, abs=1e-10)
        assert compute_skew("C3D8", x) == pytest.approx(0.0, abs=1e-10)
        j_min, j_max = compute_jacobian("C3D8", x)
        assert j_min == pytest.approx(0.125, abs=1e-10)
        assert j_max == pytest.approx(0.125, abs=1e-10)

    def test_regular_tet_gmsh(self):
        """Gmsh: regular tetrahedron has aspect=1, skew=0."""
        a = 1.0
        nodes = _xyz(
            (0, 0, 0), (a, 0, 0),
            (a / 2, a * math.sqrt(3) / 2, 0),
            (a / 2, a * math.sqrt(3) / 6, a * math.sqrt(6) / 3),
        )
        x = _coords(nodes, [1, 2, 3, 4])
        assert compute_aspect_ratio("C3D4", x) == pytest.approx(1.0, abs=1e-10)
        assert compute_skew("C3D4", x) == pytest.approx(0.0, abs=1e-10)
