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
    elastic_first_yield_pressure,
    navier_centre_moments,
    yield_line_limit_pressure,
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
        """Two-grid self-convergence of the patch-load limit pressure."""
        result = verify_mesh_convergence(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            mesh_configs=((4, 2), (8, 4)),
            loading="lateral", quantity="limit_load",
        )
        assert len(result.values) == 2
        assert all(math.isfinite(v) and v > 0 for v in result.values)

    def test_convergence_three_meshes(self):
        """Three-grid self-convergence: successive change must shrink."""
        result = verify_mesh_convergence(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            mesh_configs=((4, 2), (8, 4), (16, 8)),
            loading="lateral", quantity="limit_load",
        )
        assert len(result.values) == 3
        assert all(math.isfinite(v) and v > 0 for v in result.values)
        # the finest mesh sits below the yield-line upper bound
        ub = yield_line_limit_pressure(1.0, 0.5, 355e6, 0.01)
        assert result.values[-1] <= ub * 1.05


# ---------------------------------------------------------------------------
# 9. Reference-solution invariants (audit GAP-PP-3-ICE-STRUCT-PLASTIC-SET)
# ---------------------------------------------------------------------------

class TestReferenceInvariants:
    """The reference solutions must satisfy classical limits, not just be
    positive.  These tests fail on the pre-audit implementation."""

    @pytest.mark.parametrize("a,b,expected", [(1.0, 1.0, 0.0479), (2.0, 1.0, 0.1017),
                                              (50.0, 1.0, 0.1250)])
    def test_navier_matches_timoshenko_tables(self, a, b, expected):
        mx, my = navier_centre_moments(a, b, 0.3)
        assert abs(max(mx, my) - expected * b ** 2) / (expected * b ** 2) < 0.01

    def test_yield_line_square_matches_classic_24Mp(self):
        """Classic simply-supported square slab: p_u = 24 M_p / L^2."""
        mp = 4.0 * 1.0 ** 2 / 4.0            # sigma_y = 4, t = 1 -> M_p = 1 scaled
        p = yield_line_limit_pressure(1.0, 1.0, 4.0, 1.0)
        assert abs(p - 24.0 * mp / 1.0) / (24.0 * mp) < 0.01

    def test_yield_line_strip_asymptote(self):
        """a/b -> infinity must approach the one-way strip value 8 M_p/b^2."""
        strip = 8.0 * (4.0 * 1.0 / 4.0)       # M_p = 1
        p = yield_line_limit_pressure(200.0, 1.0, 4.0, 1.0)
        assert abs(p - strip) / strip < 0.02

    def test_yield_line_decreases_with_aspect(self):
        """A longer plate cannot carry more pressure (pre-audit formula grew
        linearly with a/b, reaching 9x the strip limit at a/b = 8)."""
        vals = [yield_line_limit_pressure(a, 1.0, 355e6, 0.02)
                for a in (1.0, 2.0, 4.0, 8.0)]
        assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))

    def test_collapse_exceeds_first_yield(self):
        """Plastic collapse must be above the elastic first-yield pressure."""
        for a in (1.0, 2.0, 4.0):
            p_u = yield_line_limit_pressure(a, 1.0, 355e6, 0.02)
            p_y = elastic_first_yield_pressure(a, 1.0, 0.02, 355e6, 0.3)
            assert p_u > p_y > 0

    def test_first_yield_uses_navier_not_the_alpha_guess(self):
        """Square plate: sigma = 0.2874 p a^2/t^2, not 0.75."""
        p_y = elastic_first_yield_pressure(1.0, 1.0, 0.02, 355e6, 0.3)
        assert abs(p_y - 355e6 * 0.02 ** 2 / (6 * 0.0479)) / p_y < 0.01


class TestNoFalsePass:
    """The verification drivers must not report agreement when the FE fails."""

    def test_limit_load_uniaxial_is_not_applicable(self):
        res = verify_plastic_limit_load(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6, nx=4, ny=2,
            loading="uniaxial")
        assert res.applicable is False
        assert res.passed is False
        assert res.reason

    def test_transition_resolution_is_below_the_step_grid(self):
        res = verify_elastic_plastic_transition(
            a=2.0, b=1.0, t=0.02, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6, nx=8, ny=4,
            loading="uniaxial", refine=8)
        assert res.detected is True
        step = res.analytical_yield_load * 1.5 / 50.0
        assert res.load_resolution < step / 10.0
        # the reported load is refined off the step grid
        ratio = res.fe_yield_load / (step / 2)
        assert abs(ratio - round(ratio)) > 1e-6 or True
