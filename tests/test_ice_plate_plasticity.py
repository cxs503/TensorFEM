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
        # DH36 at -30°C should be halfway between -20°C (380) and -40°C (400)
        sigma = flow_stress_at_temperature("DH36", -30.0)
        expected = 390e6
        assert abs(sigma - expected) < 1e3

    def test_unknown_grade_raises(self):
        """Unknown grade should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown steel grade"):
            flow_stress_at_temperature("AH32", 20.0)

    def test_out_of_range_raises(self):
        """Temperature outside validated range should raise."""
        with pytest.raises(ValueError, match="outside the validated range"):
            flow_stress_at_temperature("DH36", -80.0)

    def test_safety_factor(self):
        """Safety factor should reduce yield stress."""
        sigma_char = flow_stress_at_temperature("DH36", 20.0)
        sigma_design = flow_stress_at_temperature("DH36", 20.0, safety_factor=1.1)
        assert abs(sigma_design - sigma_char / 1.1) < 1e3

    def test_low_temp_model_wrapper(self):
        """LowTempSteelModel convenience wrapper."""
        model = LowTempSteelModel(grade="EH36", temperature=-40.0)
        assert abs(model.yield_stress - 440e6) < 1e3


# ---------------------------------------------------------------------------
# 2. Ilyushin plate limit analysis
# ---------------------------------------------------------------------------

class TestIlyushinLimit:
    """Test Ilyushin full-plastic limit solutions."""

    def test_uniaxial_limit_equals_yield(self):
        """Uniaxial full-plastic limit = yield stress."""
        limit = ilyushin_plate_limit_stress(
            1.0, 0.5, 0.01, 206e9, 0.3, 355e6, loading="uniaxial")
        assert abs(limit - 355e6) < 1e3

    def test_lateral_limit_positive(self):
        """Lateral limit pressure should be positive."""
        limit = ilyushin_plate_limit_stress(
            1.0, 0.5, 0.01, 206e9, 0.3, 355e6, loading="lateral")
        assert limit > 0

    def test_lateral_limit_scales_with_thickness(self):
        """Lateral limit should scale with t^2."""
        limit1 = ilyushin_plate_limit_stress(
            1.0, 0.5, 0.01, 206e9, 0.3, 355e6, loading="lateral")
        limit2 = ilyushin_plate_limit_stress(
            1.0, 0.5, 0.02, 206e9, 0.3, 355e6, loading="lateral")
        ratio = limit2 / limit1
        assert abs(ratio - 4.0) < 0.1  # t^2 scaling

    def test_combined_limit_elastic(self):
        """Combined limit function: elastic state should give Q < 1."""
        Q = ilyushin_combined_limit(
            Nx=100e3, Ny=0, Nxy=0,
            Mx=0, My=0, Mxy=0,
            sigma_y=355e6, t=0.01)
        assert Q < 1.0

    def test_combined_limit_at_yield(self):
        """Combined limit function: Nx = N0 should give Q_n = 1."""
        N0 = 355e6 * 0.01
        Q = ilyushin_combined_limit(
            Nx=N0, Ny=0, Nxy=0,
            Mx=0, My=0, Mxy=0,
            sigma_y=355e6, t=0.01)
        assert abs(Q - 1.0) < 1e-10

    def test_invalid_params_raises(self):
        """Invalid parameters should raise."""
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
        """Mesh should have correct number of nodes and elements."""
        nx, ny = 8, 4
        model, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, nx, ny,
            206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        assert len(model.nodes) == (nx + 1) * (ny + 1)
        assert len(model.elements) == nx * ny

    def test_patch_area(self):
        """Patch area should match geometry."""
        _, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 8, 4,
            206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        expected_area = (0.75 - 0.25) * (0.375 - 0.125)
        assert abs(info['patch_area'] - expected_area) < 1e-12

    def test_reference_load_shape(self):
        """Reference load should have correct shape."""
        model, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 8, 4,
            206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        assert info['reference_load'].shape == (model.n_dofs,)

    def test_reference_load_negative_z(self):
        """Patch load should be in negative z direction."""
        _, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 8, 4,
            206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        # Check that some z-direction loads are negative
        n_nodes = len(info['reference_load']) // 6
        z_loads = info['reference_load'][2::6]
        assert float(z_loads.min()) < 0

    def test_clamped_bc(self):
        """Clamped BC should fix rotations on edges."""
        _, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 4, 2,
            206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375,
            bc_type="clamped")
        fixed = info['fixed_dofs'].tolist()
        # Edge node should have rx and ry fixed
        # Node 0 (corner): DOFs 0,1,2,3,4,5 all fixed
        assert 3 in fixed  # rx
        assert 4 in fixed  # ry


# ---------------------------------------------------------------------------
# 4. Permanent deflection
# ---------------------------------------------------------------------------

class TestPermanentDeflection:
    """Test permanent deflection post-processing."""

    def test_extract_max_deflection(self):
        """Extract max permanent deflection from vector."""
        nodes = torch.zeros((4, 3), dtype=torch.float64)
        # Create a deflection vector with known max w
        u = torch.zeros(24, dtype=torch.float64)
        u[2] = 0.001   # node 0, w
        u[8] = -0.002  # node 1, w
        u[14] = 0.0015 # node 2, w
        u[20] = -0.001 # node 3, w
        max_w = extract_max_permanent_deflection(u, nodes)
        assert abs(max_w - 0.002) < 1e-12


# ---------------------------------------------------------------------------
# 5. Elastic-plastic transition
# ---------------------------------------------------------------------------

class TestElasticPlasticTransition:
    """Test elastic-plastic transition verification."""

    def test_transition_uniaxial_coarse(self):
        """Uniaxial transition on coarse mesh should be within 5% of analytical."""
        result = verify_elastic_plastic_transition(
            a=1.0, b=0.5, t=0.01,
            young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=8, ny=4,
            loading="uniaxial",
        )
        assert result.relative_error < 0.05, \
            f"Transition error {result.relative_error*100:.2f}% exceeds 5%"

    def test_transition_positive_loads(self):
        """Both analytical and FE yield loads should be positive."""
        result = verify_elastic_plastic_transition(
            a=1.0, b=0.5, t=0.01,
            young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=4, ny=2,
            loading="uniaxial",
        )
        assert result.analytical_yield_load > 0
        assert result.fe_yield_load > 0


# ---------------------------------------------------------------------------
# 6. Plastic limit load
# ---------------------------------------------------------------------------

class TestPlasticLimitLoad:
    """Test plastic limit load verification."""

    def test_limit_uniaxial_coarse(self):
        """Uniaxial limit on coarse mesh should be within 10% of Ilyushin."""
        result = verify_plastic_limit_load(
            a=1.0, b=0.5, t=0.01,
            young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=8, ny=4,
            loading="uniaxial",
        )
        assert result.relative_error < 0.10, \
            f"Limit load error {result.relative_error*100:.2f}% exceeds 10%"

    def test_limit_positive(self):
        """FE limit load should be positive."""
        result = verify_plastic_limit_load(
            a=1.0, b=0.5, t=0.01,
            young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=4, ny=2,
            loading="uniaxial",
        )
        assert result.fe_limit > 0
        assert result.ilyushin_limit > 0


# ---------------------------------------------------------------------------
# 7. Mesh convergence
# ---------------------------------------------------------------------------

class TestMeshConvergence:
    """Test three-grid mesh convergence."""

    def test_convergence_uniaxial(self):
        """Mesh convergence for uniaxial loading should pass."""
        result = verify_mesh_convergence(
            a=1.0, b=0.5, t=0.01,
            young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            mesh_configs=((4, 2), (8, 4), (16, 8)),
            loading="uniaxial",
            quantity="max_deflection",
        )
        assert result.passed, \
            f"Convergence failed: errors={[f'{e*100:.2f}%' for e in result.relative_errors]}"

    def test_convergence_three_meshes(self):
        """Must have exactly 3 mesh results."""
        result = verify_mesh_convergence(
            a=1.0, b=0.5, t=0.01,
            young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            mesh_configs=((4, 2), (8, 4), (16, 8)),
            loading="uniaxial",
        )
        assert len(result.meshes) == 3
        assert len(result.values) == 3
        assert len(result.relative_errors) == 3

    def test_convergence_monotonic(self):
        """Errors should decrease with mesh refinement."""
        result = verify_mesh_convergence(
            a=1.0, b=0.5, t=0.01,
            young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            mesh_configs=((4, 2), (8, 4), (16, 8)),
            loading="uniaxial",
        )
        # Finest mesh error should be smallest (or zero)
        assert result.relative_errors[-1] <= result.relative_errors[0]


# ---------------------------------------------------------------------------
# 8. Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for the full module."""

    def test_module_imports(self):
        """Module should import without errors."""
        from tensorfem import ice_plate_plasticity
        assert hasattr(ice_plate_plasticity, 'flow_stress_at_temperature')
        assert hasattr(ice_plate_plasticity, 'ilyushin_plate_limit_stress')
        assert hasattr(ice_plate_plasticity, 'patch_load_plate_mesh')
        assert hasattr(ice_plate_plasticity, 'compute_permanent_deflection')

    def test_low_temp_model_creates_valid_shell(self):
        """LowTempSteelModel should create a valid LayeredShell4Model."""
        steel = LowTempSteelModel(grade="EH36", temperature=-40.0)
        nodes = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float64)
        elements = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        model = steel.make_model(nodes, elements, thickness=0.01)
        model.validate()  # Should not raise
        assert abs(model.yield_stress - 440e6) < 1e3

    def test_full_workflow_uniaxial(self):
        """Full workflow: mesh -> load -> permanent deflection."""
        from tensorfem.layered_shell4_plasticity import LayeredShell4State
        from tensorfem.ice_plate_plasticity import (
            patch_load_plate_mesh, compute_permanent_deflection)

        model, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 8, 4,
            206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)

        # Virgin state should give zero permanent deflection
        state = LayeredShell4State.virgin(model)
        perm = compute_permanent_deflection(model, state, info['fixed_dofs'])
        max_perm = extract_max_permanent_deflection(perm, model.nodes)
        assert max_perm < 1e-12  # No plastic strain -> no permanent deflection
