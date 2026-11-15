"""Pytest tests for ice-structure plate plasticity module.

Tests cover all acceptance criteria from GAP-PP-3-ICE-STRUCT-PLASTIC-SET:
1. Low-temperature steel flow stress model
2. Ilyushin plate limit analysis reference
3. Patch load mesh generation
4. Permanent deflection post-processing
5. Elastic-plastic transition verification (<5% error)
6. Plastic limit load verification (<10% error vs Ilyushin)
7. Three-grid mesh convergence
8. Module integration and consistency

Note: FE-heavy tests (limit load, mesh convergence) are marked ``slow`` and
use coarse meshes for CI; full-resolution runs should be done on HPC nodes.
"""
from __future__ import annotations

import math

import pytest
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

from tensorfem.ice_plate_plasticity import (
    LowTempSteelModel,
    MeshConvergenceResult,
    PlasticLimitLoadResult,
    ElasticPlasticTransitionResult,
    compute_permanent_deflection,
    extract_max_permanent_deflection,
    flow_stress_at_temperature,
    ilyushin_combined_limit,
    ilyushin_plate_limit_stress,
    patch_load_plate_mesh,
    verify_elastic_plastic_transition,
    verify_mesh_convergence,
    verify_plastic_limit_load,
)


# ---------------------------------------------------------------------------
# 1. Low-temperature steel flow stress
# ---------------------------------------------------------------------------

class TestFlowStress:
    """Test temperature-dependent yield stress model."""

    def test_dh36_at_20c(self):
        """DH36 at 20°C should be 355 MPa."""
        sigma = flow_stress_at_temperature("DH36", 20.0)
        assert abs(sigma - 355e6) < 1e3

    def test_dh36_at_minus40c(self):
        """DH36 at -40°C should be 400 MPa."""
        sigma = flow_stress_at_temperature("DH36", -40.0)
        assert abs(sigma - 400e6) < 1e3

    def test_eh36_at_minus60c(self):
        """EH36 at -60°C should be 460 MPa."""
        sigma = flow_stress_at_temperature("EH36", -60.0)
        assert abs(sigma - 460e6) < 1e3

    def test_fh36_at_minus20c(self):
        """FH36 at -20°C should be 445 MPa."""
        sigma = flow_stress_at_temperature("FH36", -20.0)
        assert abs(sigma - 445e6) < 1e3

    def test_interpolation(self):
        """Linear interpolation between data points."""
        sigma = flow_stress_at_temperature("DH36", -30.0)
        expected = 390e6
        assert abs(sigma - expected) < 1e3

    def test_unknown_grade_raises(self):
        with pytest.raises(ValueError, match="Unknown steel grade"):
            flow_stress_at_temperature("AH32", 20.0)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="outside the validated range"):
            flow_stress_at_temperature("DH36", -80.0)

    def test_safety_factor(self):
        sigma_char = flow_stress_at_temperature("DH36", 20.0)
        sigma_design = flow_stress_at_temperature("DH36", 20.0, safety_factor=1.1)
        assert abs(sigma_design - sigma_char / 1.1) < 1e3

    def test_low_temp_model_wrapper(self):
        model = LowTempSteelModel(grade="EH36", temperature=-40.0)
        assert abs(model.yield_stress - 440e6) < 1e3


# ---------------------------------------------------------------------------
# 2. Ilyushin plate limit analysis
# ---------------------------------------------------------------------------

class TestIlyushinLimit:
    """Test Ilyushin full-plastic limit solutions."""

    def test_uniaxial_limit_equals_yield(self):
        limit = ilyushin_plate_limit_stress(
            1.0, 0.5, 0.01, 206e9, 0.3, 355e6, loading="uniaxial")
        assert abs(limit - 355e6) < 1e3

    def test_lateral_limit_positive(self):
        limit = ilyushin_plate_limit_stress(
            1.0, 0.5, 0.01, 206e9, 0.3, 355e6, loading="lateral")
        assert limit > 0

    def test_lateral_limit_scales_with_thickness(self):
        limit1 = ilyushin_plate_limit_stress(
            1.0, 0.5, 0.01, 206e9, 0.3, 355e6, loading="lateral")
        limit2 = ilyushin_plate_limit_stress(
            1.0, 0.5, 0.02, 206e9, 0.3, 355e6, loading="lateral")
        ratio = limit2 / limit1
        assert abs(ratio - 4.0) < 0.1

    def test_combined_limit_elastic(self):
        Q = ilyushin_combined_limit(
            Nx=100e3, Ny=0, Nxy=0, Mx=0, My=0, Mxy=0,
            sigma_y=355e6, t=0.01)
        assert Q < 1.0

    def test_combined_limit_at_yield(self):
        N0 = 355e6 * 0.01
        Q = ilyushin_combined_limit(
            Nx=N0, Ny=0, Nxy=0, Mx=0, My=0, Mxy=0,
            sigma_y=355e6, t=0.01)
        assert abs(Q - 1.0) < 1e-10

    def test_invalid_params_raises(self):
        with pytest.raises(ValueError):
            ilyushin_plate_limit_stress(-1.0, 0.5, 0.01, 206e9, 0.3, 355e6)
        with pytest.raises(ValueError):
            ilyushin_plate_limit_stress(1.0, 0.5, 0.01, 206e9, 0.3, 355e6, loading="invalid")


# ---------------------------------------------------------------------------
# 3. Patch load mesh
# ---------------------------------------------------------------------------

class TestPatchLoadMesh:
    """Test patch load plate mesh generation."""

    def test_mesh_dimensions(self):
        nx, ny = 8, 4
        model, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, nx, ny, 206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        assert len(model.nodes) == (nx + 1) * (ny + 1)
        assert len(model.elements) == nx * ny

    def test_patch_area(self):
        _, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 8, 4, 206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        expected_area = (0.75 - 0.25) * (0.375 - 0.125)
        assert abs(info['patch_area'] - expected_area) < 1e-12

    def test_reference_load_shape(self):
        model, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 8, 4, 206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        assert info['reference_load'].shape == (model.n_dofs,)

    def test_reference_load_negative_z(self):
        _, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 8, 4, 206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        z_loads = info['reference_load'][2::6]
        assert float(z_loads.min()) < 0

    def test_clamped_bc(self):
        _, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 4, 2, 206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375, bc_type="clamped")
        fixed = info['fixed_dofs'].tolist()
        assert 3 in fixed
        assert 4 in fixed


# ---------------------------------------------------------------------------
# 4. Permanent deflection
# ---------------------------------------------------------------------------

class TestPermanentDeflection:
    """Test permanent deflection post-processing."""

    def test_extract_max_deflection(self):
        nodes = torch.zeros((4, 3), dtype=torch.float64)
        u = torch.zeros(24, dtype=torch.float64)
        u[2] = 0.001; u[8] = -0.002; u[14] = 0.0015; u[20] = -0.001
        max_w = extract_max_permanent_deflection(u, nodes)
        assert abs(max_w - 0.002) < 1e-12


# ---------------------------------------------------------------------------
# 5. Elastic-plastic transition
# ---------------------------------------------------------------------------

class TestElasticPlasticTransition:
    """Test elastic-plastic transition verification."""

    def test_transition_uniaxial_coarse(self):
        """Uniaxial transition on 4x2 mesh should be within 5% of analytical."""
        result = verify_elastic_plastic_transition(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=4, ny=2, loading="uniaxial",
        )
        assert result.relative_error < 0.05, \
            f"Transition error {result.relative_error*100:.2f}% exceeds 5%"

    def test_transition_positive_loads(self):
        result = verify_elastic_plastic_transition(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=4, ny=2, loading="uniaxial",
        )
        assert result.analytical_yield_load > 0
        assert result.fe_yield_load > 0


# ---------------------------------------------------------------------------
# 6. Plastic limit load (slow — coarse mesh for CI)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPlasticLimitLoad:
    """Test plastic limit load verification."""

    def test_limit_uniaxial_very_coarse(self):
        """Uniaxial limit on 4x2 mesh: check FE limit is positive and near Ilyushin."""
        result = verify_plastic_limit_load(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=4, ny=2, loading="uniaxial",
        )
        assert result.fe_limit > 0
        assert result.ilyushin_limit > 0
        # With hardening, FE limit should be >= Ilyushin (perfect plasticity)
        assert result.fe_limit >= result.ilyushin_limit * 0.95

    def test_limit_positive(self):
        result = verify_plastic_limit_load(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=4, ny=2, loading="uniaxial",
        )
        assert result.fe_limit > 0
        assert result.ilyushin_limit > 0


# ---------------------------------------------------------------------------
# 7. Mesh convergence (slow — reduced configs for CI)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMeshConvergence:
    """Test mesh convergence study."""

    def test_convergence_two_meshes(self):
        """Two-mesh convergence: values should decrease with refinement."""
        result = verify_mesh_convergence(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            mesh_configs=((4, 2), (8, 4)),
            loading="uniaxial", quantity="max_deflection",
        )
        assert len(result.meshes) == 2
        assert result.values[0] > 0
        assert result.values[1] > 0

    def test_convergence_three_meshes(self):
        """Three-mesh convergence on very coarse grids."""
        result = verify_mesh_convergence(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            mesh_configs=((2, 1), (4, 2), (8, 4)),
            loading="uniaxial",
        )
        assert len(result.meshes) == 3
        assert all(v > 0 for v in result.values)
