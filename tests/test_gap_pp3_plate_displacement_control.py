"""Tests for the displacement-controlled plate plastic path (anchor GAP-PP-3).

The module under test replaces the *prescribed pressure* of the archived
load-controlled ladder by a prescribed *centre deflection* with the patch
pressure as an unknown of the augmented Newton system.  The tests assert
invariants rather than executing code:

* on the elastic branch the two drivers must agree on the same state
  (displacement control is not a different mechanics, only a different driver);
* a controlled state is never committed outside the layer section capacity;
* a withheld permanent set stays withheld (no fail-open default);
* the zero-pressure state resolved by bisection really is λ ≈ 0;
* the path-bookkeeping properties (turning point, tangent ratio, first yield)
  behave as documented on synthetic paths, with no FE at all.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import pytest
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

from tensorfem.layered_shell4_plasticity import (  # noqa: E402
    LayeredShell4State,
    solve_layered_shell4_increment,
)
from tensorfem.plate_patch_plastic_displacement import (  # noqa: E402
    DisplacementPath,
    DisplacementPathPoint,
    run_patch_plastic_displacement_path,
)
from tensorfem.plate_patch_plastic_limit import (  # noqa: E402
    analytic_first_yield_pressure,
    patch_plate_mesh,
    section_moment_capacity,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "outputs" / "gap_pp3_displacement_ladder.json"

# A deliberately cheap case (8 elements, 2 layers) so the suite stays in the
# fast lane; the real ladder is a queue job and is checked by the artifact test.
A = B = 1.0
T = 0.020
E, NU, SY = 210e9, 0.3, 355e6
PATCH = (0.5, 0.5)


def _elastic_only(**kwargs):
    return run_patch_plastic_displacement_path(
        A, B, T, young=E, poisson=NU, yield_stress=SY, hardening=10e6,
        patch=PATCH, nx=4, ny=4, layers=2, deflection_ratios=(0.2, 0.4),
        unload_steps=0, **kwargs)


# ---------------------------------------------------------------------------
# 1. Elastic branch: the two drivers must describe the same mechanics
# ---------------------------------------------------------------------------

class TestElasticBranch:

    def test_elastic_branch_is_linear(self):
        path = _elastic_only()
        slope = path.elastic_slope
        assert slope is not None and slope > 0.0
        for point in path.loading:
            # λ = w / k0 on the elastic branch: the control dof and the pressure
            # are the same physical state, only the driver differs
            assert point.load_factor * slope == pytest.approx(
                point.centre_deflection, rel=2e-3)
            assert point.max_alpha < 1e-12        # nothing yielded yet

    def test_displacement_state_agrees_with_load_control(self):
        """Same state, two drivers: load control at λ must give the same w_c."""
        path = _elastic_only()
        first = path.loading[0]
        model = path.model
        reference = path.info["reference_load"]
        virgin = LayeredShell4State.virgin(model)
        step = solve_layered_shell4_increment(
            model, reference, first.load_factor, path.info["fixed_dofs"], virgin,
            torch.zeros(model.n_dofs, dtype=model.nodes.dtype), tolerance=1e-12,
            max_iterations=10)
        nx, ny = path.info["nx"], path.info["ny"]
        centre = 6 * ((nx // 2) * (ny + 1) + ny // 2) + 2   # centre w dof
        assert float(step.displacement[centre]) == pytest.approx(
            first.centre_deflection, rel=1e-8)

    def test_first_yield_reported_above_elastic_range(self):
        """A path that stays elastic reports no first yield, not a guess."""
        path = _elastic_only()
        assert path.first_yield_factor is None
        analytic = analytic_first_yield_pressure(A, B, T, E, NU, SY, PATCH)
        assert path.loading[-1].load_factor < analytic["pressure"]


# ---------------------------------------------------------------------------
# 2. Plastic branch, admissibility and permanence
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plastic():
    """One plastic path shared by the branch tests (cheap mesh, 2 layers)."""
    return run_patch_plastic_displacement_path(
        A, B, T, young=E, poisson=NU, yield_stress=SY, hardening=10e6,
        patch=PATCH, nx=4, ny=4, layers=2,
        deflection_ratios=(0.5, 1.0, 2.0, 3.0), unload_steps=14)


class TestPlasticBranch:

    def test_pressure_increases_with_deflection(self, plastic):
        factors = [p.load_factor for p in plastic.loading]
        assert all(b > a for a, b in zip(factors, factors[1:]))

    def test_states_stay_inside_the_section(self, plastic):
        capacity = section_moment_capacity(T, SY, 2)["ratio"]
        assert capacity == pytest.approx(1.0)
        for point in plastic.loading + plastic.unloading:
            assert 0.0 <= point.max_moment_ratio <= capacity * (1.0 + 1e-3)

    def test_plastic_strain_accumulates(self, plastic):
        assert plastic.loading[0].max_alpha < plastic.loading[-1].max_alpha
        assert plastic.loading[-1].max_alpha > 1e-3

    def test_zero_pressure_state_is_resolved(self, plastic):
        assert plastic.permanent_set_valid
        residual = plastic.permanent_deflection
        peak = plastic.peak_deflection
        assert residual is not None and 0.0 < residual < peak
        # the terminating unload point must be at (numerically) zero pressure
        lam_end = plastic.loading[-1].load_factor
        assert abs(plastic.unloading[-1].load_factor) <= 2e-3 * lam_end

    def test_permanent_set_withheld_when_bracket_unresolved(self):
        """One bisection cannot resolve λ = 0: the set must be withheld."""
        path = run_patch_plastic_displacement_path(
            A, B, T, young=E, poisson=NU, yield_stress=SY, hardening=10e6,
            patch=PATCH, nx=4, ny=4, layers=2,
            deflection_ratios=(0.5, 2.0), unload_steps=1)
        assert path.permanent_set_valid is False
        assert path.permanent_deflection is None
        assert path.unload_failure is not None


# ---------------------------------------------------------------------------
# 3. Input contract
# ---------------------------------------------------------------------------

class TestInputContract:

    @pytest.mark.parametrize("kwargs, match", [
        (dict(deflection_ratios=()), "must not be empty"),
        (dict(deflection_ratios=(1.0, 0.5)), "strictly increasing"),
        (dict(deflection_ratios=(0.0, 1.0)), "must be positive"),
        (dict(layers=0), "layers must be >= 1"),
        (dict(nx=3), "even element counts"),
    ])
    def test_rejects_invalid_input(self, kwargs, match):
        base = dict(young=E, poisson=NU, yield_stress=SY, hardening=10e6,
                    patch=PATCH, nx=4, ny=4, layers=2, unload_steps=0)
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            run_patch_plastic_displacement_path(A, B, T, **base)

    def test_patch_alignment_is_enforced_by_the_mesher(self):
        with pytest.raises(ValueError):
            patch_plate_mesh(2.0, 1.0, T, 12, 6, young=E, poisson=NU,
                             yield_stress=SY, hardening=10e6, layers=2,
                             patch=(0.5, 0.5))


# ---------------------------------------------------------------------------
# 4. Path bookkeeping (no FE): turning point, plateau, tangent ratio
# ---------------------------------------------------------------------------

def _synthetic(pairs):
    path = DisplacementPath(model=None, info={}, elastic_slope=1.0)
    for index, (w, lam) in enumerate(pairs):
        path.loading.append(DisplacementPathPoint(
            centre_deflection=w, load_factor=lam, max_alpha=0.0,
            max_surface_mises=lam * 1e3, max_moment_ratio=0.5,
            newton_iterations=3))
    return path


class TestPathBookkeeping:

    def test_plateau_is_not_a_turning_point(self):
        path = _synthetic([(1.0, 10.0), (2.0, 18.0), (3.0, 19.5), (4.0, 19.9)])
        assert path.limit_is_turning_point is False
        assert path.peak_factor == 19.9
        assert path.plateau_factor == 19.9

    def test_decreasing_branch_is_a_turning_point(self):
        path = _synthetic([(1.0, 10.0), (2.0, 20.0), (3.0, 18.0), (4.0, 15.0)])
        assert path.limit_is_turning_point is True
        assert path.peak_factor == 20.0
        assert path.plateau_factor == 15.0

    def test_tangent_ratio_uses_the_elastic_slope(self):
        path = _synthetic([(1.0, 10.0), (2.0, 20.0), (3.0, 21.0)])
        # dλ/dw = 1 over k0 = 1/1 -> 1/(1/1) ... normalised by 1/k0
        assert path.tangent_ratio == pytest.approx(1.0)

    def test_first_yield_interpolates_the_stress_crossing(self):
        path = _synthetic([(1.0, 10.0), (2.0, 20.0)])
        path.model = type("M", (), {"yield_stress": 15e3})()
        # surface stress = 1e3 λ, so σ_y = 15e3 is met at λ = 15
        assert path.first_yield_factor == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# 5. Queue artifact (skipped until the ladder has run)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACT.exists(), reason="displacement ladder not run yet")
class TestLadderArtifact:

    @pytest.fixture(scope="class")
    def payload(self):
        with open(ARTIFACT) as handle:
            return json.load(handle)

    def test_every_run_is_inside_the_rigid_plastic_bracket(self, payload):
        refs = payload["references"]
        for run in payload["runs"]:
            assert refs["strip_lower"] <= run["plateau_factor"] <= refs["yield_line_upper"]

    def test_reported_numbers_are_finite(self, payload):
        for run in payload["runs"]:
            for key in ("peak_factor", "plateau_factor", "capacity_ratio"):
                assert math.isfinite(run[key])
            assert run["load_points"] >= 2

    def test_mesh_stability_is_reported_not_assumed(self, payload):
        criteria = payload["criteria"]["mesh_stability"]
        assert isinstance(criteria["met"], bool)
        assert criteria["finest_change"] is not None

    def test_permanent_set_withheld_is_explicit(self, payload):
        for run in payload["runs"]:
            if not run["permanent_set_valid"]:
                assert run["permanent_deflection"] is None