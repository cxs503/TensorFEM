"""Unit tests for the constant-curvature (bending) patch test.

These assert *invariants*, not merely that the code runs:

* affine / skewed patches reproduce a constant-curvature state to round-off
  (the patch test proper) -- displacement, rotations, moments and energy;
* the distorted patch stays inside the 2% case-pool tolerance;
* the deliberate locking control **fails** -- otherwise the passing gates would
  be vacuous;
* the load-driven strip matches the closed form for both lateral conditions, and
  the ratio of the two responses is``1/(1-nu^2)``;
* the analytic oracle itself is verified before any assembly.
"""
from __future__ import annotations

import pytest
import torch

from tensorfem.bending_patch import (
    CurvatureState,
    affine_patch_mesh,
    distorted_patch_mesh,
    oracle_selfcheck,
    run_patch_case,
    strip_pure_bending,
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