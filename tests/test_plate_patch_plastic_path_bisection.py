"""Regression tests for the patch-plate plastic-load *path integrator*.

These tests pin three properties that the GAP-PP-3 limit-load evidence depends
on, none of which the previous suite asserted:

1. an increment that converges onto a state outside the section shell is
   **bisected**, not used to truncate the whole ramp (the pre-fix behaviour
   stopped the path ~7 % below the section capacity, so the physical limit point
   was never reached and ``collapse_factor`` degenerated into "last prescribed
   factor of the ramp");
2. every committed point of a path is section-admissible, so a limit load fitted
   to it is fitted to equilibrium states of the layer model;
3. ``plate_capacity_summary`` forwards the limit-load configuration
   (``geometric``, ``inplane_restrained``) and reports the *measured* limit load
   from :func:`limit_load_from_path`, not just the end of the ramp.

The meshes are deliberately the coarsest admissible ones: the assertions are
about the integrator's control flow, not about discretisation accuracy.
"""
from __future__ import annotations

import os

import pytest
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)

from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    limit_load_from_path,
    low_temperature_flow_stress,
    path_admissibility,
    patch_plate_mesh,
    plate_capacity_summary,
    run_patch_plastic_path,
    section_capacity_ratio_of,
)

#: The bisection makes the near-mechanism path expensive (~1 min per solve on the
#: smallest admissible mesh).  The invariant tests are therefore opt-in, exactly
#: like the 3-grid evidence test that reads the cluster artifact; the cluster
#: study ``cj4232`` runs the same code path on the full matrix.
HEAVY = os.environ.get("TF_RUN_HEAVY_PLASTIC_PATH") == "1"

A, B, T = 2.0, 1.0, 0.020
PATCH = (0.5, 0.5)
E_MOD, NU, HARDENING = 206e9, 0.3, 1000e6
FLOW = low_temperature_flow_stress("DH36", -20.0)
PY = analytic_first_yield_pressure(A, B, T, E_MOD, NU, FLOW.yield_stress, PATCH)["pressure"]


_CACHE: dict = {}


def _path(nx=8, ny=4, layers=3, factors=(1.5, 2.1)):
    """Integrate the path once and reuse it (the bisection makes it expensive)."""
    key = (nx, ny, layers, factors)
    if key not in _CACHE:
        _CACHE[key] = run_patch_plastic_path(
            A, B, T, young=E_MOD, poisson=NU, yield_stress=FLOW.yield_stress,
            hardening=HARDENING, patch=PATCH,
            load_factors=[round(f * PY) for f in factors],
            nx=nx, ny=ny, layers=layers, geometric=True, inplane_restrained=True)
    return _CACHE[key]


@pytest.mark.skipif(not HEAVY, reason="set TF_RUN_HEAVY_PLASTIC_PATH=1 (path solve ~1 min)")
class TestRejectedIncrementIsBisected:
    def test_path_is_refined_instead_of_truncated(self):
        """A section-capacity rejection refines the increment, it does not stop the ramp.

        The load factors present on the path are compared with the prescribed
        ladder: bisection forces intermediate (non-prescribed) factors into the
        history.  If the integrator truncated on the first rejected state the
        history would be a prefix of the ladder.
        """
        path = _path()
        ladder = {round(f * PY) for f in (1.5, 2.1)}
        interior = [q.load_factor for q in path.loading if round(q.load_factor) not in ladder]
        assert interior, (
            "no sub-step was taken: the integrator truncated the ramp at the "
            "first section-capacity rejection instead of bisecting it")
        # the path still advanced past the load level where the pre-fix code stopped
        assert path.loading[-1].load_factor > min(ladder) * 1.05

    def test_committed_points_are_section_admissible(self):
        """No committed point may sit outside the layer model's section capacity."""
        path = _path()
        capacity = section_capacity_ratio_of(path)
        assert capacity is not None
        gate = path_admissibility(path, capacity_ratio=capacity, tolerance=5e-2)
        assert gate["admissible"], gate["violations"]
        peak = max(q.max_moment_ratio for q in path.loading)
        assert peak <= capacity * 1.05, (peak, capacity)

    def test_deflection_increases_monotonically_on_the_path(self):
        path = _path()
        centres = [q.centre_deflection for q in path.loading]
        assert all(b > a for a, b in zip(centres, centres[1:])), centres


class TestStepRejectionRules:
    """The rejection predicate is pure, so its invariants are tested directly."""

    def test_runaway_increment_is_rejected(self):
        from tensorfem.plate_patch_plastic_limit import (
            PatchPathPoint, _step_rejection)
        history = [PatchPathPoint(1.0e5, 0.001, 0.0, 1.0e8, 0.2)]
        reason = _step_rejection(0.5, 0.2, 2.0e5, history, 0.96, None)
        assert reason is not None and "runaway" in reason

    def test_capacity_overshoot_is_rejected(self):
        from tensorfem.plate_patch_plastic_limit import (
            PatchPathPoint, _step_rejection)
        history = [PatchPathPoint(1.0e5, 0.001, 0.0, 1.0e8, 0.2)]
        reason = _step_rejection(0.0012, 1.02, 1.0e5, history, 0.96, None)
        assert reason is not None and "capacity" in reason

    def test_sound_step_is_accepted(self):
        from tensorfem.plate_patch_plastic_limit import (
            PatchPathPoint, _step_rejection)
        history = [PatchPathPoint(1.0e5, 0.001, 0.0, 1.0e8, 0.2)]
        assert _step_rejection(0.0015, 0.3, 1.5e5, history, 0.96, None) is None

    def test_non_finite_state_is_rejected(self):
        from tensorfem.plate_patch_plastic_limit import (
            PatchPathPoint, _step_rejection)
        history = [PatchPathPoint(1.0e5, 0.001, 0.0, 1.0e8, 0.2)]
        assert _step_rejection(float("nan"), 0.3, 1.5e5, history, 0.96, None)


class TestCapacitySummaryLimitReporting:
    @pytest.mark.skipif(not HEAVY, reason="set TF_RUN_HEAVY_PLASTIC_PATH=1")
    def test_summary_reports_measured_limit_load(self):
        """``plate_capacity_summary`` reports the path-measured limit load."""
        summary = plate_capacity_summary(
            a=A, b=B, t=T, young=E_MOD, poisson=NU, grade="DH36",
            temperature_c=-20.0, patch=PATCH, hardening=HARDENING,
            load_factors=[round(f * PY) for f in (1.5, 2.1)],
            nx=8, ny=4, layers=3, unload_steps=1,
            geometric=True, inplane_restrained=True)
        fe = summary["fe"]
        assert "limit_measurements" in fe
        assert "measured_limit_load" in fe
        assert "measured_limit_error_vs_upper" in fe
        # the ramp-end factor and the measured limit are different quantities and
        # the summary says which one the collapse factor is
        assert fe["collapse_factor"] > 0.0

    def test_inplane_restraint_changes_the_boundary_conditions(self):
        """``inplane_restrained=True`` fixes the in-plane DOFs on every boundary node."""
        _, free_info = patch_plate_mesh(A, B, T, 8, 4, young=E_MOD, poisson=NU,
                                        yield_stress=FLOW.yield_stress,
                                        hardening=HARDENING, patch=PATCH,
                                        inplane_restrained=False)
        _, held_info = patch_plate_mesh(A, B, T, 8, 4, young=E_MOD, poisson=NU,
                                        yield_stress=FLOW.yield_stress,
                                        hardening=HARDENING, patch=PATCH,
                                        inplane_restrained=True)
        assert len(held_info["fixed_dofs"]) > len(free_info["fixed_dofs"])

    @pytest.mark.skipif(not HEAVY, reason="set TF_RUN_HEAVY_PLASTIC_PATH=1")
    def test_ramp_end_is_not_a_limit_point(self):
        """A completed ramp must not be reported as a measured limit load."""
        path = _path()
        if path.ramp_completed and not path.collapse_is_limit_point:
            assert path.measured_limit_load is None
        measurements = limit_load_from_path(path)
        assert "limit_load" in measurements
        if path.limit_point_searched:
            assert path.collapse_is_limit_point, (
                "a searched limit point must be reported as a limit point")