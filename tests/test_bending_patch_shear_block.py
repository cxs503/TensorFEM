"""Tests for the constant-shear (transverse-shear block) consistency check.

Why this file exists
--------------------
The round-1/round-2 pool of ``test_bending_patch.py`` grades *shear-free*
states only, so the element's transverse-shear modulus ``Ds = kappa_s G t``
enters none of those quantities.  These tests assert:

* the new check's own oracle is sound (the prescribed field really is a state of
  constant transverse shear with zero bending strain);
* the prescribed-field energy identity holds to round-off on every mesh family
  and every shear treatment -- a pass/fail check with no interpolation caveat;
* the check **can fail**: a 20 % error in ``Ds`` (``kappa_s = 1`` instead of
  ``5/6``) and a flipped shear sign relation are both rejected;
* the balanced split of the field is *necessary*: a ``w``-only state is invariant
  to the shear sign relation and therefore cannot grade it;
* the mutation instrument is the production element (no drift) and the mutation
  context does not leak out of its ``with`` block;
* (finding, as a regression guard) the existing patch/strip legs *pass* with a
  20 % error in ``Ds`` and with ``Ds`` multiplied by 5 -- i.e. the anchor is
  blind to the shear modulus over the whole realistic range, which is what the
  new check repairs.

All tolerances are round-off based, not tuned to the outcome.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.bending_patch import (
    ShearState,
    affine_patch_mesh,
    assemble_plate_stiffness,
    distorted_patch_mesh,
    free_surface_check,
    mutated_q4_stiffness,
    q4_reference_stiffness,
    run_patch_case,
    shear_block_check,
    shear_block_mutation,
    shear_rigidity,
    shear_state_selfcheck,
    strip_pure_bending,
    CurvatureState,
)
from tensorfem.plate import q4_mindlin_stiffness

E, NU, T = 210.0e3, 0.3, 1.0
KAPPA = 5.0 / 6.0
EXACT_TOL = 1.0e-12
SHEAR_STATES = [
    ShearState(2.0e-3, 0.0),
    ShearState(0.0, 2.0e-3),
    ShearState(1.5e-3, -1.0e-3),
]
FAMILIES = {
    "affine": lambda: affine_patch_mesh(2),
    "skew": lambda: affine_patch_mesh(2, skew=0.35),
    "distorted": lambda: distorted_patch_mesh(2),
}


@pytest.fixture(autouse=True)
def _single_thread():
    torch.set_num_threads(1)
    yield


# --------------------------------------------------------------------------- #
# oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state", SHEAR_STATES, ids=lambda s: s.label())
def test_constant_shear_state_is_what_it_claims(state):
    rep = shear_state_selfcheck(state)
    assert rep["shear_deviation"] < 1e-15      # gamma is reproduced exactly
    assert rep["curvature_leak"] < 1e-15       # no bending strain at all
    assert rep["spread"] < 1e-15               # constant over the element
    assert rep["gamma"] == pytest.approx([state.gamma_xz, state.gamma_yz])


def test_degenerate_shear_state_is_rejected():
    with pytest.raises(ValueError):
        ShearState(0.0, 0.0).field(affine_patch_mesh(2).nodes)


def test_shear_rigidity_matches_the_declared_law():
    g = E / (2.0 * (1.0 + NU))
    assert shear_rigidity(E, NU, T) == pytest.approx(KAPPA * g * T)
    assert shear_rigidity(E, NU, T, shear_correction=1.0) == pytest.approx(g * T)


# --------------------------------------------------------------------------- #
# the identity itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("treatment", ["sri", "mitc", "full"])
@pytest.mark.parametrize("state", SHEAR_STATES, ids=lambda s: s.label())
def test_prescribed_energy_identity_is_exact(family, treatment, state):
    rep = shear_block_check(FAMILIES[family](), state, young=E, poisson=NU,
                            thickness=T, shear_treatment=treatment)
    assert rep["energy_rel_error"] <= EXACT_TOL
    assert rep["shear_recovery_error"] <= EXACT_TOL
    assert rep["curvature_leak"] <= 1e-12
    # the graded reference is the declared constitutive law, not a fitted number
    assert rep["energy_exact"] == pytest.approx(
        0.5 * shear_rigidity(E, NU, T) * (state.gamma_xz ** 2
                                          + state.gamma_yz ** 2) * rep["area"])


def test_identity_scales_with_the_state_not_with_the_mesh():
    """The check is a property of the element, not of the mesh: two meshes of
    the same area give the same relative error at round-off, and doubling
    ``gamma`` quadruples the energy."""
    a = shear_block_check(affine_patch_mesh(2), SHEAR_STATES[2], young=E,
                          poisson=NU, thickness=T)
    b = shear_block_check(affine_patch_mesh(3), SHEAR_STATES[2], young=E,
                          poisson=NU, thickness=T)
    assert a["energy_rel_error"] <= EXACT_TOL and b["energy_rel_error"] <= EXACT_TOL
    assert a["energy_exact"] == pytest.approx(b["energy_exact"])
    doubled = ShearState(2.0 * SHEAR_STATES[2].gamma_xz,
                         2.0 * SHEAR_STATES[2].gamma_yz)
    c = shear_block_check(affine_patch_mesh(2), doubled, young=E, poisson=NU,
                          thickness=T)
    assert c["energy"] == pytest.approx(4.0 * a["energy"], rel=1e-12)


# --------------------------------------------------------------------------- #
# discriminating power: the check must be able to fail
# --------------------------------------------------------------------------- #
def test_wrong_shear_factor_is_rejected():
    """``kappa_s = 1`` is a 20 % error in ``Ds``; the check sees it."""
    state = SHEAR_STATES[2]
    nominal = shear_block_check(FAMILIES["skew"](), state, young=E, poisson=NU,
                                thickness=T)
    with shear_block_mutation("none", 1.0 / KAPPA):
        wrong = shear_block_check(FAMILIES["skew"](), state, young=E, poisson=NU,
                                  thickness=T)
    assert nominal["energy_rel_error"] <= EXACT_TOL
    assert wrong["energy_rel_error"] == pytest.approx(0.2, rel=1e-9)
    assert wrong["energy_rel_error"] > 1e10 * nominal["energy_rel_error"]


def test_parabolic_shear_factor_error_is_rejected():
    """``kappa_s = pi^2/12`` is only 1.3 % off -- still rejected by 1e10."""
    state = SHEAR_STATES[0]
    with shear_block_mutation("none", (math.pi ** 2 / 12.0) / KAPPA):
        wrong = shear_block_check(FAMILIES["affine"](), state, young=E,
                                 poisson=NU, thickness=T)
    assert wrong["energy_rel_error"] > 1.0e-2
    assert wrong["energy_rel_error"] / EXACT_TOL > 1e10


def test_flipped_shear_sign_is_rejected():
    state = SHEAR_STATES[2]
    with shear_block_mutation("shear_sign"):
        wrong = shear_block_check(FAMILIES["affine"](), state, young=E,
                                  poisson=NU, thickness=T)
    assert wrong["energy_rel_error"] >= 0.9


def test_deleted_shear_block_is_rejected():
    state = SHEAR_STATES[2]
    with shear_block_mutation("no_shear"):
        wrong = shear_block_check(FAMILIES["affine"](), state, young=E,
                                  poisson=NU, thickness=T)
    assert wrong["energy_rel_error"] >= 0.9


def test_balanced_split_is_necessary():
    """A w-only field is blind to the sign relation; the balanced split is not."""
    patch = affine_patch_mesh(2)
    state = SHEAR_STATES[2]

    def energy(u):
        k = assemble_plate_stiffness(patch, young=E, poisson=NU, thickness=T)
        return 0.5 * float(u @ (k @ u))

    n = patch.nodes.shape[0]
    w_only = torch.zeros(3 * n, dtype=torch.float64)
    w_only[0::3] = (state.gamma_xz * patch.nodes[:, 0]
                    + state.gamma_yz * patch.nodes[:, 1])
    balanced = state.field(patch.nodes).reshape(-1)

    plus_w = energy(w_only)
    plus_b = energy(balanced)
    with shear_block_mutation("shear_sign"):
        flip_w = energy(w_only)
        flip_b = energy(balanced)
    assert abs(flip_w - plus_w) / plus_w < 1e-12      # blind
    assert abs(flip_b - plus_b) / plus_b > 0.9        # maximal separation
    assert plus_b == pytest.approx(
        0.5 * shear_rigidity(E, NU, T)
        * (state.gamma_xz ** 2 + state.gamma_yz ** 2) * patch.quad_area(),
        rel=1e-12)


# --------------------------------------------------------------------------- #
# the instrument must be the element (no drift), and must not leak
# --------------------------------------------------------------------------- #
def test_instrument_reproduces_the_production_element():
    xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                      dtype=torch.float64)
    prod = q4_mindlin_stiffness(xy, young=E, poisson=NU, thickness=T,
                                shear_treatment="sri")
    instr = mutated_q4_stiffness(xy, young=E, poisson=NU, thickness=T)
    assert float((prod - instr).abs().max() / prod.abs().max()) <= 1e-13


def test_mutation_context_only_touches_sri_and_restores():
    xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                      dtype=torch.float64)

    def rel(a, b):
        return float((a - b).abs().max() / a.abs().max())

    base = {tr: q4_reference_stiffness(xy, young=E, poisson=NU, thickness=T,
                                       shear_treatment=tr)
            for tr in ("sri", "mitc", "full")}
    with shear_block_mutation("shear_sign"):
        inside = {tr: q4_reference_stiffness(xy, young=E, poisson=NU, thickness=T,
                                             shear_treatment=tr)
                  for tr in ("sri", "mitc", "full")}
    after = {tr: q4_reference_stiffness(xy, young=E, poisson=NU, thickness=T,
                                        shear_treatment=tr)
             for tr in ("sri", "mitc", "full")}
    assert rel(base["sri"], inside["sri"]) > 0.1     # the context bites
    assert rel(base["mitc"], inside["mitc"]) == 0.0
    assert rel(base["full"], inside["full"]) == 0.0
    for tr in ("sri", "mitc", "full"):               # restored, no leak
        assert rel(base[tr], after[tr]) == 0.0


def test_mutation_names_are_validated():
    with pytest.raises(ValueError):
        with shear_block_mutation("typo"):
            pass
    with pytest.raises(ValueError):
        mutated_q4_stiffness(torch.zeros((4, 2), dtype=torch.float64) + 0.5,
                             young=E, poisson=NU, thickness=T, mutation="typo")


# --------------------------------------------------------------------------- #
# finding: the round-1/2 anchor is blind to the shear modulus
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kappa_scale", [1.0 / KAPPA, 5.0])
def test_existing_anchor_legs_are_blind_to_the_shear_modulus(kappa_scale):
    """Regression guard for the documented coverage gap.

    With ``Ds`` wrong by 20 % (``kappa_s = 1``) *and* with ``Ds`` multiplied by
    five, every shear-sensitive leg of the round-1/2 anchor still passes: the
    constant-curvature states are shear-free by construction, and the strip runs
    at ``t/L = 0.01`` where the shear flexibility is ~0.008 % of the answer.
    The legs are asserted to pass, so that if a future change makes the anchor
    notice a wrong ``Ds`` this test flags that the documented finding changed.
    """
    with shear_block_mutation("none", kappa_scale):
        for state in (CurvatureState(1.0e-3, 0.0, 0.0),
                      CurvatureState(1.0e-3, 0.6e-3, 0.4e-3)):
            for skew in (0.0, 0.35):
                rep = run_patch_case(affine_patch_mesh(3, skew=skew), state,
                                     young=E, poisson=NU, thickness=T)
                assert rep["displacement_rel_error"] <= EXACT_TOL
                assert rep["moment_rel_error"] <= EXACT_TOL
                assert rep["energy_rel_error"] <= EXACT_TOL
        dist = run_patch_case(distorted_patch_mesh(3),
                              CurvatureState(1.0e-3, 0.6e-3, 0.4e-3),
                              young=E, poisson=NU, thickness=T)
        assert dist["displacement_rel_error"] <= 0.02
        assert dist["energy_rel_error"] <= 0.02
        fs = free_surface_check(affine_patch_mesh(3), 1.0e-3, young=E, poisson=NU,
                                thickness=T)
        assert fs["my_over_mx"] <= EXACT_TOL
        assert abs(fs["mx_over_db_kappa"] - 1.0) <= EXACT_TOL
        sym = strip_pure_bending(32, 2, lateral="symmetry", thickness=0.01)
        assert sym["w_rel_error_kirchhoff"] <= 0.02
        free = strip_pure_bending(32, 16, lateral="free", thickness=0.01)
        assert free["w_rel_error_kirchhoff"] <= 0.02


def test_deleted_shear_block_is_caught_by_singularity_not_by_a_gate():
    """Fail-safe, not a gate: with the shear block deleted the Dirichlet patch
    has no stiffness for ``w`` at all, so the solve fails loudly.  Recorded so
    that "the anchor cannot be *silently* wrong about a missing block" is an
    asserted property, not a claim."""
    with shear_block_mutation("no_shear"):
        with pytest.raises(RuntimeError):
            run_patch_case(affine_patch_mesh(3), CurvatureState(1.0e-3, 0.0, 0.0),
                           young=E, poisson=NU, thickness=T)