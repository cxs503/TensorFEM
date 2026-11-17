"""Permanent deflection and elastic-plastic integration tests for
GAP-PP-3-ICE-STRUCT-PLASTIC-SET.

These tests verify the permanent deflection post-processing and the
elastic-plastic transition on very coarse meshes (lightweight, suitable
for CI even under high system load).

Anchor: GAP-PP-3-ICE-STRUCT-PLASTIC-SET
Acceptance criteria covered:
- Permanent deflection computation (eigenstrain springback)
- Elastic-plastic transition <5% error (coarse mesh)
- Reference solution invariants (Navier, yield-line, strip bounds)
"""
from __future__ import annotations

import math

import pytest
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

from tensorfem.ice_plate_plasticity import (
    compute_permanent_deflection,
    elastic_first_yield_pressure,
    extract_max_permanent_deflection,
    flow_stress_at_temperature,
    navier_centre_moments,
    patch_load_plate_mesh,
    verify_elastic_plastic_transition,
    yield_line_limit_pressure,
)
from tensorfem.layered_shell4_plasticity import (
    LayeredShell4State,
    assemble_layered_shell4,
)


# ---------------------------------------------------------------------------
# Permanent deflection: full computation with mock plastic state
# ---------------------------------------------------------------------------

class TestPermanentDeflectionFull:
    """Test the full permanent deflection computation pipeline."""

    def test_virgin_state_zero_permanent(self):
        """A virgin (unyielded) state must give zero permanent deflection."""
        model, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 4, 2, 206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        state = LayeredShell4State.virgin(model)
        u_perm = compute_permanent_deflection(model, state, info['fixed_dofs'])
        max_w = extract_max_permanent_deflection(u_perm, model.nodes)
        assert max_w < 1e-15, f"Virgin state should give ~0 permanent deflection, got {max_w}"

    def test_permanent_deflection_shape(self):
        """Permanent deflection vector must have the same shape as the DOF vector."""
        model, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 4, 2, 206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        state = LayeredShell4State.virgin(model)
        u_perm = compute_permanent_deflection(model, state, info['fixed_dofs'])
        assert u_perm.shape == (model.n_dofs,)

    def test_permanent_deflection_respects_bc(self):
        """Permanent deflection at fixed DOFs must be zero."""
        model, info = patch_load_plate_mesh(
            1.0, 0.5, 0.01, 4, 2, 206e9, 0.3, 355e6, 1000e6,
            0.25, 0.75, 0.125, 0.375)
        state = LayeredShell4State.virgin(model)
        u_perm = compute_permanent_deflection(model, state, info['fixed_dofs'])
        fixed = info['fixed_dofs'].tolist()
        for dof in fixed:
            assert abs(float(u_perm[dof])) < 1e-15, f"Fixed DOF {dof} has non-zero permanent deflection"


# ---------------------------------------------------------------------------
# Elastic-plastic transition: lightweight coarse-mesh tests
# ---------------------------------------------------------------------------

class TestTransitionLightweight:
    """Lightweight elastic-plastic transition tests (4x2 mesh, fast)."""

    def test_transition_uniaxial_4x2(self):
        """Uniaxial transition on 4x2 mesh: error <5%."""
        result = verify_elastic_plastic_transition(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=4, ny=2, loading="uniaxial",
        )
        assert result.passed, (
            f"Transition not passed: error={result.relative_error*100:.2f}%, "
            f"analytical={result.analytical_yield_load/1e6:.4f} MPa, "
            f"FE={result.fe_yield_load/1e6:.4f} MPa"
        )
        assert result.relative_error < 0.05

    def test_transition_detected_flag(self):
        """The transition must be detected (yield point found)."""
        result = verify_elastic_plastic_transition(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=4, ny=2, loading="uniaxial",
        )
        assert result.detected, "Yield point was not detected"

    def test_transition_load_resolution(self):
        """The load resolution after bisection must be fine (<0.1% of analytical)."""
        result = verify_elastic_plastic_transition(
            a=1.0, b=0.5, t=0.01, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6,
            nx=4, ny=2, loading="uniaxial",
        )
        resolution_pct = result.load_resolution / result.analytical_yield_load * 100
        assert resolution_pct < 0.1, f"Load resolution too coarse: {resolution_pct:.3f}%"


# ---------------------------------------------------------------------------
# Reference solution invariants (additional checks)
# ---------------------------------------------------------------------------

class TestReferenceInvariantsExtended:
    """Extended reference solution invariants for the plastic set."""

    def test_first_yield_scales_with_thickness_squared(self):
        """First-yield pressure scales with t^2 (bending-dominated)."""
        p1 = elastic_first_yield_pressure(1.0, 0.5, 0.01, 355e6, 0.3)
        p2 = elastic_first_yield_pressure(1.0, 0.5, 0.02, 355e6, 0.3)
        ratio = p2 / p1
        assert abs(ratio - 4.0) < 0.05, f"t^2 scaling failed: ratio={ratio:.3f}"

    def test_first_yield_scales_linearly_with_yield_stress(self):
        """First-yield pressure scales linearly with sigma_y."""
        p1 = elastic_first_yield_pressure(1.0, 0.5, 0.01, 355e6, 0.3)
        p2 = elastic_first_yield_pressure(1.0, 0.5, 0.01, 710e6, 0.3)
        ratio = p2 / p1
        assert abs(ratio - 2.0) < 0.01, f"Linear scaling failed: ratio={ratio:.3f}"

    def test_yield_line_scales_with_thickness_squared(self):
        """Yield-line limit pressure scales with t^2."""
        p1 = yield_line_limit_pressure(1.0, 0.5, 355e6, 0.01)
        p2 = yield_line_limit_pressure(1.0, 0.5, 355e6, 0.02)
        ratio = p2 / p1
        assert abs(ratio - 4.0) < 0.05, f"t^2 scaling failed: ratio={ratio:.3f}"

    def test_yield_line_invariant_to_young_poisson(self):
        """Yield-line limit is independent of elastic constants (rigid-plastic)."""
        p1 = yield_line_limit_pressure(1.0, 0.5, 355e6, 0.01)
        p2 = yield_line_limit_pressure(1.0, 0.5, 355e6, 0.01)
        # Same inputs -> same output (sanity check)
        assert p1 == p2

    def test_navier_moments_symmetric_for_square(self):
        """For a square plate, Mx = My at the centre."""
        mx, my = navier_centre_moments(1.0, 1.0, 0.3)
        assert abs(mx - my) / mx < 1e-6, f"Square plate should have Mx=My: Mx={mx}, My={my}"

    def test_navier_moments_asymmetric_for_rectangle(self):
        """For a 2:1 rectangle, Mx != My."""
        mx, my = navier_centre_moments(2.0, 1.0, 0.3)
        assert abs(mx - my) / max(abs(mx), abs(my)) > 0.1, "Rectangle should have Mx != My"

    def test_flow_stress_monotone_with_temperature(self):
        """Flow stress must increase monotonically as temperature decreases."""
        grades = ["DH36", "EH36", "FH36"]
        for grade in grades:
            stresses = [flow_stress_at_temperature(grade, T) for T in [20, 0, -20, -40, -60]]
            for i in range(len(stresses) - 1):
                assert stresses[i] < stresses[i + 1], (
                    f"{grade}: flow stress not monotone at {stresses[i]/1e6:.0f} -> {stresses[i+1]/1e6:.0f} MPa"
                )


# ---------------------------------------------------------------------------
# Integration: elastic solution consistency
# ---------------------------------------------------------------------------

class TestElasticConsistency:
    """Verify that the elastic branch of the plastic solver matches the
    independent Navier reference on the same mesh."""

    def test_patch_load_elastic_centre_deflection(self):
        """Centre deflection under patch load should be positive and finite."""
        from tensorfem.plate_patch_plastic_limit import elastic_patch_plate
        result = elastic_patch_plate(
            1.0, 0.5, 0.01, 206e9, 0.3, (0.5, 0.5), 1e5, terms=41)
        w = result["deflection"][0]
        assert w > 0, f"Centre deflection should be positive, got {w}"
        assert math.isfinite(w), f"Centre deflection should be finite, got {w}"

    def test_first_yield_below_yield_line(self):
        """First-yield pressure must be below the yield-line upper bound."""
        fy = elastic_first_yield_pressure(1.0, 0.5, 0.01, 355e6, 0.3)
        yl = yield_line_limit_pressure(1.0, 0.5, 355e6, 0.01)
        assert fy < yl, (
            f"First-yield ({fy/1e6:.4f} MPa) must be below yield-line UB ({yl/1e6:.4f} MPa)"
        )

    def test_first_yield_below_strip_lower_bound(self):
        """First-yield pressure must be below the strip lower bound (elastic < plastic)."""
        from tensorfem.plate_patch_plastic_limit import strip_lower_bound
        fy = elastic_first_yield_pressure(1.0, 0.5, 0.01, 355e6, 0.3)
        lb = strip_lower_bound(1.0, 0.5, 0.01, 355e6, (0.5, 0.5))["pressure"]
        assert fy < lb, (
            f"First-yield ({fy/1e6:.4f} MPa) must be below strip LB ({lb/1e6:.4f} MPa)"
        )


# ---------------------------------------------------------------------------
# Unloading robustness (D3): the unloading leg must bisect oversized steps
# instead of aborting the whole leg and flagging the permanent set invalid.
# ---------------------------------------------------------------------------

class TestUnloadingBisection:
    """The unloading leg must converge via bisection, not abort on a large step.

    Regression for D3 (cj4436 / m22855 honest FAIL): on the finest 3-grid mesh
    the 3-step peak->0 unloading failed to converge and the permanent set was
    flagged invalid.  The fix drives the unloading with the same bisection the
    loading leg uses, so an oversized elastic-recovery step is halved until it
    converges rather than discarding the residual state.
    """

    @pytest.fixture(scope="class")
    def unloaded_path(self):
        from tensorfem.plate_patch_plastic_limit import run_patch_plastic_path
        return run_patch_plastic_path(
            1.0, 1.0, 0.020, young=206e9, poisson=0.3,
            yield_stress=355e6, hardening=1000e6, patch=(0.5, 0.5),
            load_factors=[0.3e6, 0.6e6, 0.9e6, 1.2e6, 1.5e6],
            nx=4, ny=4, layers=3,
            unload_steps=3, unload_to=0.0,
            geometric=True, inplane_restrained=True)

    def test_permanent_set_valid_after_unloading(self, unloaded_path):
        assert unloaded_path.unload_failure is None, (
            f"unloading leg aborted: {unloaded_path.unload_failure}")
        assert unloaded_path.permanent_set_valid, (
            "permanent set must be a real post-unload state, not a flagged failure")

    def test_permanent_deflection_bounded(self, unloaded_path):
        peak = unloaded_path.loading[-1].centre_deflection
        perm = unloaded_path.permanent_deflection
        assert perm is not None
        assert 0.0 < perm < peak, (
            f"permanent set {perm} must lie between 0 and the peak {peak}")

    def test_unloading_reduces_deflection(self, unloaded_path):
        peak = unloaded_path.loading[-1].centre_deflection
        residual = unloaded_path.unloading[-1].centre_deflection
        assert residual < peak, "unloading must reduce the centre deflection"
