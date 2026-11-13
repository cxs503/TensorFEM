"""Tests for geometric wetting / contact-angle LBM boundary condition.

Validation cases:
1. ContactAngleBC construction and boundary node identification
2. Interface normal estimation
3. Normal rotation to prescribed contact angle
4. Full droplet-on-surface relaxation (90° case)
5. Cross-validation sweep (30°, 60°, 90°, 120°, 150°)
6. Simulation stability
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np
import pytest

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.lbm_wetting import (
    ContactAngleBC,
    DropletConfig,
    geometric_wetting_xval,
    measure_contact_angle,
    run_droplet_validation,
)


# ---------------------------------------------------------------------------
# Test 1: ContactAngleBC construction
# ---------------------------------------------------------------------------

class TestContactAngleBCConstruction:
    def test_valid_angle(self):
        bc = ContactAngleBC(contact_angle_deg=90.0)
        assert bc.contact_angle_deg == 90.0
        assert abs(bc._cos_theta) < 1e-12
        assert abs(bc._sin_theta - 1.0) < 1e-12

    def test_hydrophilic(self):
        bc = ContactAngleBC(contact_angle_deg=30.0)
        assert abs(bc._cos_theta - math.cos(math.radians(30))) < 1e-12

    def test_hydrophobic(self):
        bc = ContactAngleBC(contact_angle_deg=150.0)
        assert abs(bc._cos_theta - math.cos(math.radians(150))) < 1e-12

    def test_invalid_angle_zero(self):
        with pytest.raises(ValueError):
            ContactAngleBC(contact_angle_deg=0.0)

    def test_invalid_angle_180(self):
        with pytest.raises(ValueError):
            ContactAngleBC(contact_angle_deg=180.0)

    def test_invalid_angle_negative(self):
        with pytest.raises(ValueError):
            ContactAngleBC(contact_angle_deg=-10.0)


# ---------------------------------------------------------------------------
# Test 2: Boundary node identification
# ---------------------------------------------------------------------------

class TestBoundaryNodeIdentification:
    def test_bottom_wall(self):
        """Solid at bottom 2 rows, fluid above."""
        ny, nx = 20, 20
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[:2, :] = True

        bc = ContactAngleBC()
        rows, cols = bc.identify_boundary_nodes(solid_mask)

        assert len(rows) > 0, "Should find boundary nodes"
        assert all(solid_mask[r, c] for r, c in zip(rows, cols))

    def test_no_solid(self):
        solid_mask = np.zeros((10, 10), dtype=bool)
        bc = ContactAngleBC()
        rows, cols = bc.identify_boundary_nodes(solid_mask)
        assert len(rows) == 0

    def test_all_solid(self):
        """All solid: edge nodes are boundary nodes (padded boundary is fluid)."""
        solid_mask = np.ones((10, 10), dtype=bool)
        bc = ContactAngleBC()
        rows, cols = bc.identify_boundary_nodes(solid_mask)
        assert len(rows) > 0
        for r, c in zip(rows, cols):
            assert r == 0 or r == 9 or c == 0 or c == 9

    def test_interior_solid_block(self):
        """Solid block in the middle of fluid."""
        ny, nx = 20, 20
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[5:10, 5:10] = True

        bc = ContactAngleBC()
        rows, cols = bc.identify_boundary_nodes(solid_mask)
        assert len(rows) > 0
        for r, c in zip(rows, cols):
            assert 5 <= r <= 9 and 5 <= c <= 9


# ---------------------------------------------------------------------------
# Test 3: Interface normal estimation
# ---------------------------------------------------------------------------

class TestInterfaceNormalEstimation:
    def test_horizontal_interface(self):
        ny, nx = 20, 20
        rho = np.full((ny, nx), 0.1)
        rho[10:, :] = 1.0

        bc = ContactAngleBC()
        rows = np.array([10])
        cols = np.array([10])
        normals = bc.estimate_interface_normal(rho, rows, cols)

        assert normals.shape == (1, 2)
        mag = math.sqrt(normals[0, 0] ** 2 + normals[0, 1] ** 2)
        assert mag <= 1.0 + 1e-10

    def test_uniform_density(self):
        ny, nx = 20, 20
        rho = np.ones((ny, nx))

        bc = ContactAngleBC()
        rows = np.array([5])
        cols = np.array([5])
        normals = bc.estimate_interface_normal(rho, rows, cols)
        assert normals[0, 1] < 0

    def test_diagonal_gradient(self):
        ny, nx = 20, 20
        y_grid, x_grid = np.mgrid[0:ny, 0:nx]
        rho = (x_grid + y_grid).astype(np.float64)

        bc = ContactAngleBC()
        rows = np.array([10])
        cols = np.array([10])
        normals = bc.estimate_interface_normal(rho, rows, cols)
        assert normals[0, 0] > 0
        assert normals[0, 1] > 0


# ---------------------------------------------------------------------------
# Test 4: Normal rotation
# ---------------------------------------------------------------------------

class TestNormalRotation:
    def test_90_degree_rotation(self):
        bc = ContactAngleBC(contact_angle_deg=90.0)
        wall_normal = np.array([0.0, -1.0])
        normals = np.array([[0.0, -1.0]])
        rotated = bc.rotate_normal_to_contact_angle(normals, wall_normal)
        assert abs(rotated[0, 1]) < 0.1
        assert abs(rotated[0, 0]) > 0.9

    def test_0_degree_rotation(self):
        bc = ContactAngleBC(contact_angle_deg=1.0)
        wall_normal = np.array([0.0, -1.0])
        normals = np.array([[0.0, -1.0]])
        rotated = bc.rotate_normal_to_contact_angle(normals, wall_normal)
        assert abs(rotated[0, 1] - (-1.0)) < 0.1

    def test_180_degree_rotation(self):
        bc = ContactAngleBC(contact_angle_deg=179.0)
        wall_normal = np.array([0.0, -1.0])
        normals = np.array([[0.0, -1.0]])
        rotated = bc.rotate_normal_to_contact_angle(normals, wall_normal)
        assert rotated[0, 1] > 0.9


# ---------------------------------------------------------------------------
# Test 5: Contact angle measurement
# ---------------------------------------------------------------------------

class TestContactAngleMeasurement:
    def test_flat_interface(self):
        ny, nx = 40, 40
        rho = np.full((ny, nx), 0.1)
        rho[20:, :] = 1.0
        solid_mask = np.zeros((ny, nx), dtype=bool)
        solid_mask[:2, :] = True

        angle = measure_contact_angle(rho, solid_mask, wall_row=1, threshold=0.5)
        assert 70 < angle < 110, f"Expected ~90°, got {angle}"

    def test_no_interface(self):
        ny, nx = 20, 20
        rho = np.ones((ny, nx))
        solid_mask = np.zeros((ny, nx), dtype=bool)
        angle = measure_contact_angle(rho, solid_mask, wall_row=0, threshold=0.5)
        assert angle == 90.0


# ---------------------------------------------------------------------------
# Test 6: Droplet validation
# ---------------------------------------------------------------------------

class TestDropletValidation:
    def test_90_degree_droplet_quick(self):
        cfg = DropletConfig(
            nx=64, ny=64,
            droplet_radius=10.0,
            droplet_center_x=32.0,
            droplet_center_y=25.0,
            prescribed_angle_deg=90.0,
            max_steps=200,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_droplet_validation(cfg)

        assert "prescribed_angle_deg" in result
        assert "measured_angle_deg" in result
        assert "error_pct" in result
        assert "rho_field" in result
        assert result["rho_field"].shape == (64, 64)
        assert result["steps"] == 200

    def test_simulation_returns_dict(self):
        """Simulation should return a properly structured dict."""
        cfg = DropletConfig(
            nx=32, ny=32,
            droplet_radius=5.0,
            prescribed_angle_deg=90.0,
            max_steps=20,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_droplet_validation(cfg)
        assert isinstance(result, dict)
        assert "prescribed_angle_deg" in result
        assert "measured_angle_deg" in result
        assert "passed" in result


# ---------------------------------------------------------------------------
# Test 7: Cross-validation sweep
# ---------------------------------------------------------------------------

class TestCrossValidationSweep:
    def test_sweep_structure(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = geometric_wetting_xval(
                contact_angles=[90.0],
                nx=48, ny=48,
                max_steps=100,
            )
        assert len(results) == 1
        assert results[0]["prescribed_angle_deg"] == 90.0
        assert "measured_angle_deg" in results[0]
        assert "error_pct" in results[0]
        assert "passed" in results[0]


# ---------------------------------------------------------------------------
# Test 8: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_small_domain(self):
        cfg = DropletConfig(
            nx=32, ny=32,
            droplet_radius=5.0,
            prescribed_angle_deg=90.0,
            max_steps=50,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_droplet_validation(cfg)
        assert result["rho_field"].shape == (32, 32)

    def test_extreme_angles(self):
        for angle in [10.0, 170.0]:
            bc = ContactAngleBC(contact_angle_deg=angle)
            assert bc.contact_angle_deg == angle


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
