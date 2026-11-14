"""Unit tests for the constant-curvature (bending) patch test.

These assert *invariants*, not merely that the code runs:

* affine / skewed patches reproduce a constant-curvature state to round-off
  (the patch test proper) -- displacement, rotations, moments and energy;
* the distorted patch stays inside the 2% case-pool tolerance;
* the deliberate locking control **fails** -- otherwise the passing gates would
  be vacuous;
* the load-driven strip matches the closed form for both lateral conditions, and
  the ratio of the *computed* responses is ``1/(1-nu^2)`` -- grading the ratio of
  the two closed forms instead would be an algebraic identity and is rejected by
  the helper that `run.py` calls (m15654 audit finding F-CRIT);
* the free-edge residual against the 1-D beam closed form is invariant under
  through-width refinement, i.e. it is a plate-vs-beam model difference and not
  discretization error across the width (m15654 audit finding F3);
* the analytic oracle itself is verified before any assembly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorfem.bending_patch import (
    CurvatureState,
    affine_patch_mesh,
    distorted_patch_mesh,
    oracle_selfcheck,
    run_patch_case,
    strip_pure_bending,
    strip_rigidity_crosscheck,
)

E, NU, T = 210.0e3, 0.3, 1.0
STATES = [
    CurvatureState(1.0e-3, 0.0, 0.0),
    CurvatureState(0.0, 0.0, 1.0e-3),
    CurvatureState(1.0e-3, 0.6e-3, 0.4e-3),
]


@pytest.fixture(autouse=True)
def _single_thread():
    torch.set_num_threads(1)
    yield


def test_oracle_is_shear_free_and_constant():
    rep = oracle_selfcheck(STATES[2], E, NU, T)
    assert rep["shear_residual"] < 1e-9
    assert rep["curvature_deviation"] < 1e-9
    assert rep["moment_deviation"] < 1e-6  # central differences, not exact


@pytest.mark.parametrize("state", STATES, ids=lambda s: s.label())
@pytest.mark.parametrize("skew", [0.0, 0.35], ids=["affine", "skew"])
def test_affine_patch_reproduces_constant_curvature_exactly(state, skew):
    patch = affine_patch_mesh(3, skew=skew)
    rep = run_patch_case(patch, state, young=E, poisson=NU, thickness=T,
                         shear_treatment="sri")
    assert rep["displacement_rel_error"] < 1e-12
    assert rep["rotation_rel_error"] < 1e-12
    assert rep["moment_rel_error"] < 1e-12
    assert rep["energy_rel_error"] < 1e-12
    assert rep["equilibrium_residual"] < 1e-10


@pytest.mark.parametrize("state", STATES, ids=lambda s: s.label())
def test_mitc_patch_reproduces_displacements_and_moments(state):
    patch = affine_patch_mesh(3)
    rep = run_patch_case(patch, state, young=E, poisson=NU, thickness=T,
                         shear_treatment="mitc")
    assert rep["displacement_rel_error"] < 1e-12
    assert rep["rotation_rel_error"] < 1e-12
    assert rep["moment_rel_error"] < 1e-12


def test_mitc_energy_on_skew_patch_is_exact():
    """The MITC assumed-natural shear field is sampled in covariant components,
    so the shear-free constant-curvature state survives the skew mapping
    exactly (until 2026-11 the physical sampling left a spurious shear energy
    here; see tests/test_plate_mitc_skew_invariance.py and
    docs/mitc-skew-invariance.md)."""
    patch = affine_patch_mesh(3, skew=0.35)
    rep = run_patch_case(patch, STATES[0], young=E, poisson=NU, thickness=T,
                         shear_treatment="mitc")
    assert rep["displacement_rel_error"] < 1e-12
    assert rep["energy_rel_error"] < 1e-10


@pytest.mark.parametrize("state", STATES, ids=lambda s: s.label())
def test_distorted_patch_within_case_pool_tolerance(state):
    patch = distorted_patch_mesh(4)
    rep = run_patch_case(patch, state, young=E, poisson=NU, thickness=T,
                         shear_treatment="sri")
    assert rep["displacement_rel_error"] < 0.02
    assert rep["moment_rel_error"] < 0.02
    assert rep["energy_rel_error"] < 0.02


def test_distorted_patch_improves_under_refinement():
    state = STATES[0]
    coarse = run_patch_case(distorted_patch_mesh(4), state, young=E, poisson=NU, thickness=T)
    fine = run_patch_case(distorted_patch_mesh(16), state, young=E, poisson=NU, thickness=T)
    assert fine["displacement_rel_error"] < coarse["displacement_rel_error"]
    assert fine["moment_rel_error"] < coarse["moment_rel_error"]


def test_locking_control_fails_the_strip():
    """The negative control must be far outside tolerance, else nothing is proven."""
    locked = strip_pure_bending(8, 2, thickness=0.01, lateral="symmetry",
                                shear_treatment="full", young=E, poisson=NU)
    assert locked["w_rel_error_kirchhoff"] > 0.5


def test_load_driven_strip_symmetry_matches_cylindrical_bending():
    rep = strip_pure_bending(8, 2, thickness=0.01, lateral="symmetry",
                             shear_treatment="sri", young=E, poisson=NU)
    assert rep["w_rel_error_kirchhoff"] < 1e-6
    assert rep["theta_rel_error"] < 1e-6
    assert rep["rigidity_used"] == "D_cyl"


def test_load_driven_strip_free_edges_matches_beam_rigidity():
    rep = strip_pure_bending(16, 2, thickness=0.01, lateral="free",
                             shear_treatment="sri", young=E, poisson=NU)
    assert rep["rigidity_used"] == "D_beam"
    assert rep["w_rel_error_kirchhoff"] < 0.02
    assert rep["theta_rel_error"] < 0.02


def test_poisson_coupling_crosscheck():
    sym = strip_pure_bending(8, 2, thickness=0.01, lateral="symmetry",
                             shear_treatment="sri", young=E, poisson=NU)
    free = strip_pure_bending(8, 2, thickness=0.01, lateral="free",
                              shear_treatment="sri", young=E, poisson=NU)
    measured = free["w_tip"] / sym["w_tip"]
    theory = 1.0 / (1.0 - NU ** 2)
    assert abs(measured - theory) / theory < 0.02


def test_crosscheck_grades_computed_deflections_not_closed_forms():
    """The Poisson cross-check must divide the two *solutions*.

    Dividing the two closed forms returns ``1/(1-nu^2)`` by algebra, so it is
    identically zero error for any element -- the defect repaired here.  The
    helper therefore reports the closed-form ratio separately and flags it.
    """
    sym = strip_pure_bending(8, 2, thickness=0.01, lateral="symmetry",
                             shear_treatment="sri", young=E, poisson=NU)
    free = strip_pure_bending(8, 2, thickness=0.01, lateral="free",
                              shear_treatment="sri", young=E, poisson=NU)
    c = strip_rigidity_crosscheck(sym, free, poisson=NU)
    assert c["operands"].startswith("computed w_tip")
    assert c["is_identity"] is False
    assert 0.0 < c["rel_error"] < 0.02
    # the closed forms *are* an identity: reported, flagged, not used as evidence
    assert c["closed_form_rel_error"] == pytest.approx(0.0, abs=1e-12)
    assert c["measured"] == pytest.approx(free["w_tip"] / sym["w_tip"], rel=0)

    # a 2% perturbation of the computed deflection must move the measured ratio,
    # while the closed-form ratio stays put -- that is what "measured" means
    bumped = strip_rigidity_crosscheck(sym, dict(free, w_tip=1.02 * free["w_tip"]),
                                       poisson=NU)
    # (a +2% change of w_free moves the measured ratio by ~+0.98% of the theory value)
    assert bumped["rel_error"] > c["rel_error"] + 5e-3
    assert bumped["closed_form_rel_error"] == pytest.approx(0.0, abs=1e-12)


def test_crosscheck_rejects_degenerate_input():
    sym = strip_pure_bending(4, 2, thickness=0.01, lateral="symmetry",
                             shear_treatment="sri", young=E, poisson=NU)
    free = strip_pure_bending(4, 2, thickness=0.01, lateral="free",
                              shear_treatment="sri", young=E, poisson=NU)
    with pytest.raises(ValueError):
        strip_rigidity_crosscheck(dict(sym, w_tip=0.0), free, poisson=NU)
    with pytest.raises(KeyError):
        strip_rigidity_crosscheck(sym, {"w_tip": 1.0}, poisson=NU)


def test_free_edge_residual_is_invariant_under_through_width_refinement():
    """F3 of the m15654 audit: the free-edge 2% margin must be understood, not hidden.

    Refining the width (where an anticlastic edge layer lives) does not remove
    the residual against the 1-D beam closed form, so the gate compares two
    models; the test pins that plateau and the 2% bound.
    """
    coarse = strip_pure_bending(16, 2, thickness=0.01, lateral="free",
                                shear_treatment="sri", young=E, poisson=NU)
    fine = strip_pure_bending(16, 16, thickness=0.01, lateral="free",
                              shear_treatment="sri", young=E, poisson=NU)
    finer = strip_pure_bending(16, 32, thickness=0.01, lateral="free",
                               shear_treatment="sri", young=E, poisson=NU)
    e16, e32 = fine["w_rel_error_kirchhoff"], finer["w_rel_error_kirchhoff"]
    assert abs(e32 - e16) / e32 < 0.05            # plateau, not convergence to 0
    assert e32 < 0.02                             # still inside the case-pool tolerance
    assert e32 > 5e-3                             # ... and the residual is real


def test_anchor_artifacts_match_the_driver_and_are_self_consistent():
    """Guard the two repaired evidence defects in the committed artefacts."""
    root = Path(__file__).resolve().parents[1]
    res_p, gate_p = (root / "benchmarks/verified/fem_bending_patch/result.json",
                     root / "benchmarks/verified/fem_bending_patch/gates_verified.json")
    if not (res_p.exists() and gate_p.exists()):
        pytest.skip("anchor artefacts not regenerated in this checkout")
    res = json.loads(res_p.read_text())
    man = json.loads(gate_p.read_text())
    ratio = res["strip_ratio"]
    # F-CRIT: the gate value is a measurement, not the closed-form identity
    assert ratio["operands"] == "computed w_tip"
    assert ratio["is_identity"] is False
    assert ratio["rel_error"] > 0.0
    gate = {g["id"]: g for g in man["gates"]}["poisson_crosscheck"]
    assert gate["value"] == pytest.approx(ratio["rel_error"], rel=1e-12)
    assert gate["passed"] is True
    # F4: one criteria block in both artefacts
    assert res["criteria"] == man["criteria"]
    # the order gate is finite even when a treatment is round-off limited
    assert isinstance({g["id"]: g for g in man["gates"]}["order_w_min"]["value"], float)
    assert man["passed"] is True and man["failed_gates"] == []


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        affine_patch_mesh(1)                       # no free interior node
    with pytest.raises(ValueError):
        distorted_patch_mesh(3, amplitude=5.0)     # inverted elements
    with pytest.raises(ValueError):
        run_patch_case(affine_patch_mesh(2), STATES[0], young=E, poisson=NU,
                       thickness=T, shear_treatment="bogus")
    with pytest.raises(ValueError):
        strip_pure_bending(4, 2, lateral="bogus")


def test_zero_curvature_state_is_rejected():
    with pytest.raises(ValueError):
        run_patch_case(affine_patch_mesh(3), CurvatureState(0.0, 0.0, 0.0),
                       young=E, poisson=NU, thickness=T)