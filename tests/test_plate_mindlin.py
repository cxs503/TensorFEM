"""Verification tests for tensorfem.plate_mindlin module.

Tests cover:
1. Winkler foundation stiffness matrix properties (symmetry, pattern, PSD)
2. Consistent load vector (partition of unity for uniform pressure)
3. Simply supported square plate under uniform load vs Navier series
4. Mesh convergence study (h-refinement)
5. Face stress computation sanity
6. Export JSON schema
7. Hetenyi reference at origin vs analytical formula w(0) = P/(8*sqrt(D*k_f))
"""
from __future__ import annotations

import math
import pytest
import torch

from tensorfem.plate_mindlin import (
    q4_winkler_foundation_stiffness,
    consistent_load_vector,
    hetenyi_reference,
    solve_plate_on_winkler_foundation,
    compute_face_stresses,
    export_failure_faces_json,
)
from tensorfem.plate import structured_square_mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_square_element(h: float = 1.0):
    """Return a single Q4 element on [0, h]^2."""
    return torch.tensor([[0.0, 0.0], [h, 0.0], [h, h], [0.0, h]], dtype=torch.float64)


def _navier_ss_square(q: float, a: float, D: float, n_terms: int = 51) -> float:
    """Navier double-series for simply supported square plate under uniform load q.

    w_max = (q/D) * sum_{m,n odd} 16 / (pi^6 * m * n * ((m/a)^2 + (n/a)^2)^2)
    For square plate a=b, evaluated at center (a/2, a/2).
    """
    w = 0.0
    for m in range(1, 2 * n_terms, 2):
        for n in range(1, 2 * n_terms, 2):
            w += 1.0 / (m * n * ((m / a) ** 2 + (n / a) ** 2) ** 2)
    w *= 16.0 * q / (math.pi ** 6 * D)
    return w


# ---------------------------------------------------------------------------
# 1. Winkler foundation stiffness
# ---------------------------------------------------------------------------

class TestWinklerFoundationStiffness:
    """Tests for q4_winkler_foundation_stiffness."""

    def test_symmetry(self):
        """Foundation stiffness must be symmetric."""
        xy = _unit_square_element()
        ke = q4_winkler_foundation_stiffness(xy, winkler_k=1e6)
        assert torch.allclose(ke, ke.T, atol=1e-12), "Ke_winkler not symmetric"

    def test_pattern_only_w_dof(self):
        """Only w-dof rows/cols (indices 0,3,6,9) should be non-zero."""
        xy = _unit_square_element()
        ke = q4_winkler_foundation_stiffness(xy, winkler_k=1e6)
        # Check that rotation dof rows are zero
        for i in range(12):
            if i % 3 != 0:  # theta_x or theta_y dof
                assert torch.allclose(ke[i, :], torch.zeros(12, dtype=torch.float64), atol=1e-14), \
                    f"Row {i} (rotation dof) should be zero"

    def test_positive_semidefinite(self):
        """Foundation stiffness should be positive semi-definite."""
        xy = _unit_square_element()
        ke = q4_winkler_foundation_stiffness(xy, winkler_k=1e6)
        # Extract the 4x4 w-block
        w_idx = [0, 3, 6, 9]
        ke_w = ke[w_idx][:, w_idx]
        eigvals = torch.linalg.eigvalsh(ke_w)
        assert torch.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"

    def test_sum_equals_k_times_area(self):
        """Sum of all entries in w-block = k_f * area (partition of unity)."""
        h = 1.5
        xy = _unit_square_element(h)
        k_f = 2e5
        ke = q4_winkler_foundation_stiffness(xy, winkler_k=k_f)
        w_idx = [0, 3, 6, 9]
        ke_w = ke[w_idx][:, w_idx]
        total = ke_w.sum().item()
        expected = k_f * h * h  # k_f * area
        assert abs(total - expected) / expected < 1e-10, \
            f"Sum {total} != expected {expected}"

    def test_zero_foundation(self):
        """k_f=0 should give zero matrix."""
        xy = _unit_square_element()
        ke = q4_winkler_foundation_stiffness(xy, winkler_k=0.0)
        assert torch.allclose(ke, torch.zeros_like(ke))


# ---------------------------------------------------------------------------
# 2. Consistent load vector
# ---------------------------------------------------------------------------

class TestConsistentLoadVector:
    """Tests for consistent_load_vector."""

    def test_uniform_pressure_sum(self):
        """Sum of load vector = pressure * area for uniform load."""
        h = 2.0
        xy = _unit_square_element(h)
        q = 1000.0
        fe = consistent_load_vector(xy, q)
        total = fe.sum().item()
        expected = q * h * h
        assert abs(total - expected) / expected < 1e-10, \
            f"Load sum {total} != expected {expected}"

    def test_only_w_dof_loaded(self):
        """Only w-dof entries should be non-zero."""
        xy = _unit_square_element()
        fe = consistent_load_vector(xy, 500.0)
        for i in range(12):
            if i % 3 != 0:
                assert abs(fe[i].item()) < 1e-14, f"fe[{i}] should be zero"

    def test_callable_pressure(self):
        """Callable pressure should integrate correctly."""
        h = 1.0
        xy = _unit_square_element(h)
        # Uniform via callable
        fe_callable = consistent_load_vector(xy, lambda x, y: 100.0)
        fe_scalar = consistent_load_vector(xy, 100.0)
        assert torch.allclose(fe_callable, fe_scalar, atol=1e-10)

    def test_linear_pressure_symmetry(self):
        """Linear pressure p(x,y) = x should give symmetric nodal distribution."""
        h = 1.0
        xy = _unit_square_element(h)
        fe = consistent_load_vector(xy, lambda x, y: x)
        # For p=x on [0,1]^2: integral = 0.5
        total = fe.sum().item()
        assert abs(total - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# 3. Simply supported plate vs Navier series
# ---------------------------------------------------------------------------

class TestSimplySupportedPlate:
    """Compare FEM solution with Navier analytical series."""

    def test_center_deflection_coarse(self):
        """Center deflection of SS square plate should approach Navier value."""
        E = 210e9  # Pa
        nu = 0.3
        t = 0.01   # m
        a = 1.0    # m
        q = 1000.0 # Pa

        D = E * t**3 / (12.0 * (1.0 - nu**2))
        w_navier = _navier_ss_square(q, a, D)

        result = solve_plate_on_winkler_foundation(
            nx=8, side=a, young=E, poisson=nu, thickness=t,
            pressure=q, winkler_k=0.0, boundary="simply_supported",
        )

        # Find center node
        nodes = result.nodes
        center_mask = ((nodes[:, 0] - a / 2).abs() < (a / 8)) & \
                      ((nodes[:, 1] - a / 2).abs() < (a / 8))
        center_idx = center_mask.nonzero(as_tuple=False)[0, 0]
        w_fem = result.displacement[3 * center_idx].item()

        rel_err = abs(w_fem - w_navier) / abs(w_navier)
        # Coarse mesh should be within 10%
        assert rel_err < 0.10, f"rel_err={rel_err:.4f} > 10%"

    def test_convergence_monotone(self):
        """Verify h-convergence: error should decrease with mesh refinement."""
        E = 210e9
        nu = 0.3
        t = 0.01
        a = 1.0
        q = 1000.0
        D = E * t**3 / (12.0 * (1.0 - nu**2))
        w_navier = _navier_ss_square(q, a, D)

        errors = []
        for nx in [4, 8]:
            result = solve_plate_on_winkler_foundation(
                nx=nx, side=a, young=E, poisson=nu, thickness=t,
                pressure=q, winkler_k=0.0, boundary="simply_supported",
            )
            nodes = result.nodes
            center_mask = ((nodes[:, 0] - a / 2).abs() < (a / nx)) & \
                          ((nodes[:, 1] - a / 2).abs() < (a / nx))
            center_idx = center_mask.nonzero(as_tuple=False)[0, 0]
            w_fem = result.displacement[3 * center_idx].item()
            errors.append(abs(w_fem - w_navier) / abs(w_navier))

        # Error should decrease
        assert errors[1] < errors[0], \
            f"Error not decreasing: {errors[0]:.6f} -> {errors[1]:.6f}"

        # Both should be within 10%
        assert errors[1] < 0.10, f"Finest mesh error {errors[1]:.4f} > 10%"


# ---------------------------------------------------------------------------
# 4. Plate on Winkler foundation - physical sanity
# ---------------------------------------------------------------------------

class TestPlateOnWinklerFoundation:
    """Tests for plate with non-zero Winkler foundation."""

    def test_foundation_reduces_deflection(self):
        """Adding foundation stiffness should reduce center deflection."""
        E, nu, t, a, q = 210e9, 0.3, 0.01, 1.0, 1000.0
        nx = 8

        res_no_foundation = solve_plate_on_winkler_foundation(
            nx=nx, side=a, young=E, poisson=nu, thickness=t,
            pressure=q, winkler_k=0.0, boundary="simply_supported",
        )
        res_with_foundation = solve_plate_on_winkler_foundation(
            nx=nx, side=a, young=E, poisson=nu, thickness=t,
            pressure=q, winkler_k=1e7, boundary="simply_supported",
        )

        # Find center deflections
        nodes = res_no_foundation.nodes
        center_mask = ((nodes[:, 0] - a / 2).abs() < (a / nx)) & \
                      ((nodes[:, 1] - a / 2).abs() < (a / nx))
        ci = center_mask.nonzero(as_tuple=False)[0, 0]

        w0 = res_no_foundation.displacement[3 * ci].item()
        w_k = res_with_foundation.displacement[3 * ci].item()

        assert w_k < w0, f"Foundation should reduce deflection: w_k={w_k} >= w0={w0}"
        assert w_k > 0, "Deflection should still be positive (downward)"

    def test_clamped_less_than_simply_supported(self):
        """Clamped plate should deflect less than simply supported."""
        E, nu, t, a, q = 210e9, 0.3, 0.01, 1.0, 1000.0
        nx = 8

        res_ss = solve_plate_on_winkler_foundation(
            nx=nx, side=a, young=E, poisson=nu, thickness=t,
            pressure=q, winkler_k=0.0, boundary="simply_supported",
        )
        res_cl = solve_plate_on_winkler_foundation(
            nx=nx, side=a, young=E, poisson=nu, thickness=t,
            pressure=q, winkler_k=0.0, boundary="clamped",
        )

        nodes = res_ss.nodes
        center_mask = ((nodes[:, 0] - a / 2).abs() < (a / nx)) & \
                      ((nodes[:, 1] - a / 2).abs() < (a / nx))
        ci = center_mask.nonzero(as_tuple=False)[0, 0]

        w_ss = res_ss.displacement[3 * ci].item()
        w_cl = res_cl.displacement[3 * ci].item()

        assert w_cl < w_ss, f"Clamped ({w_cl}) should be < SS ({w_ss})"


# ---------------------------------------------------------------------------
# 5. Hetenyi reference
# ---------------------------------------------------------------------------

class TestHetenyiReference:
    """Tests for hetenyi_reference (infinite plate point load)."""

    def test_origin_deflection_formula(self):
        """w(0) = P / (8 * sqrt(D * k_f)) for point load at origin."""
        E = 210e9
        nu = 0.3
        t = 0.1
        k_f = 1e7
        P = 1000.0

        D = E * t**3 / (12.0 * (1.0 - nu**2))
        w_analytical = P / (8.0 * math.sqrt(D * k_f))
        w_hetenyi = hetenyi_reference(0.0, 0.0, P, E, nu, t, k_f)

        rel_err = abs(w_hetenyi - w_analytical) / abs(w_analytical)
        assert rel_err < 1e-6, \
            f"Hetenyi w(0) error: {rel_err:.2e} (got {w_hetenyi}, expected {w_analytical})"

    def test_decay_with_distance(self):
        """Deflection should decay with distance from load."""
        E, nu, t, k_f, P = 210e9, 0.3, 0.1, 1e7, 1000.0
        w0 = hetenyi_reference(0.0, 0.0, P, E, nu, t, k_f)
        w1 = hetenyi_reference(1.0, 0.0, P, E, nu, t, k_f)
        w5 = hetenyi_reference(5.0, 0.0, P, E, nu, t, k_f)

        assert abs(w1) < abs(w0), "Deflection should decay"
        assert abs(w5) < abs(w1), "Deflection should continue to decay"


# ---------------------------------------------------------------------------
# 6. Face stresses
# ---------------------------------------------------------------------------

class TestFaceStresses:
    """Tests for compute_face_stresses."""

    def test_face_count(self):
        """Each element should produce 4 faces."""
        E, nu, t, a, q = 210e9, 0.3, 0.01, 1.0, 1000.0
        nx = 4
        result = solve_plate_on_winkler_foundation(
            nx=nx, side=a, young=E, poisson=nu, thickness=t,
            pressure=q, winkler_k=0.0, boundary="simply_supported",
        )
        faces = compute_face_stresses(result)
        n_elements = nx * nx
        assert len(faces) == 4 * n_elements

    def test_face_schema(self):
        """Each face should have required keys."""
        E, nu, t, a, q = 210e9, 0.3, 0.01, 1.0, 1000.0
        result = solve_plate_on_winkler_foundation(
            nx=4, side=a, young=E, poisson=nu, thickness=t,
            pressure=q, winkler_k=0.0, boundary="simply_supported",
        )
        faces = compute_face_stresses(result)
        required_keys = {"id", "element", "edge", "centroid", "normal", "area", "stress"}
        for face in faces:
            assert required_keys.issubset(face.keys()), f"Missing keys in {face.keys()}"
            assert len(face["centroid"]) == 3
            assert len(face["normal"]) == 3
            assert face["area"] > 0


# ---------------------------------------------------------------------------
# 7. Export JSON
# ---------------------------------------------------------------------------

class TestExportFailureFacesJson:
    """Tests for export_failure_faces_json."""

    def test_schema(self):
        """Output should have 'faces' list and 'units' key."""
        faces = [
            {"id": 0, "area": 0.01, "normal": [1, 0, 0], "centroid": [0.5, 0, 0],
             "stress": {"sigma_x": 100, "sigma_y": 50, "tau_xy": 10, "tau_xz": 0, "tau_yz": 0}},
        ]
        result = export_failure_faces_json(faces)
        assert "faces" in result
        assert "units" in result
        assert result["units"] == "SI"
        assert len(result["faces"]) == 1
        assert "G" in result["faces"][0]
        assert "released_energy" in result["faces"][0]

    def test_empty_faces(self):
        """Empty input should produce empty output."""
        result = export_failure_faces_json([])
        assert result["faces"] == []
        assert result["units"] == "SI"


# ---------------------------------------------------------------------------
# 8. Boundary condition validation
# ---------------------------------------------------------------------------

class TestBoundaryValidation:
    """Test boundary condition input validation."""

    def test_invalid_boundary_raises(self):
        """Invalid boundary string should raise ValueError."""
        with pytest.raises(ValueError, match="boundary must be"):
            solve_plate_on_winkler_foundation(
                nx=4, side=1.0, young=210e9, poisson=0.3, thickness=0.01,
                pressure=1000.0, winkler_k=0.0, boundary="invalid",
            )
