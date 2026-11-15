"""Unit tests for the constant-curvature (bending) patch test.

These assert *invariants*, not merely that the code runs:

* affine / skewed patches reproduce a constant-curvature state to round-off
  (the patch test proper) -- displacement, rotations, moments and energy;
* the distorted patch stays inside the 2% case-pool tolerance;
* the deliberate locking control **fails** -- otherwise the passing gates would
  be vacuous;
* the load-driven strip matches the closed form for both lateral conditions;
* (round 2) the Poisson coupling is graded on quantities that *can* fail: the
  closed-form ratio `D/D_b` is an algebra identity that cancels any common
  stiffness factor, so it is tested as such, while the `M_y = 0` (plane-stress)
  patch and the *solved* strip ratio carry the content;
* (round 2) the free-edge strip leg is transverse-converged, and its ~1.2%
  offset is a clamped-root end effect that decays with `L/b`;
* the analytic oracle itself is verified before any assembly;
* (round 3) the thin limit: the locking error of the bilinear patch follows the
  `(h/t)^2` law on distorted meshes, MITC4 and the affine SRI path stay at
  round-off at every thickness, and the SRI limitation is recorded rather than
  hidden.
"""
from __future__ import annotations

import pytest
import torch

from tensorfem.bending_patch import (
    CurvatureState,
    affine_patch_mesh,
    distorted_patch_mesh,
    oracle_selfcheck,
    free_surface_check,
    free_surface_state,
    plate_moment_matrix,
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
    """Solved tip deflections, on a transverse-converged mesh (round 2).

    `n_y = 2` (the round-1 mesh) is not used here: it aliases the transverse
    direction and lands ~2.6x closer to the identity than the converged answer.
    """
    sym = strip_pure_bending(8, 8, thickness=0.01, lateral="symmetry",
                             shear_treatment="sri", young=E, poisson=NU)
    free = strip_pure_bending(8, 8, thickness=0.01, lateral="free",
                              shear_treatment="sri", young=E, poisson=NU)
    measured = free["w_tip"] / sym["w_tip"]
    theory = 1.0 / (1.0 - NU ** 2)
    assert abs(measured - theory) / theory < 0.02


def test_closed_form_poisson_ratio_is_blind_to_the_element():
    """Round-2 finding: the round-1 gate compared two closed forms.

    `[M L^2/(2D)] / [M L^2/(2D_b)] = D/D_b = 1/(1-nu^2)` because the rigidity
    enters both operands as a common factor, so the "error" is identically zero
    for any element, any mesh and any corrupted stiffness.  A check that cannot
    fail is not evidence; it is kept as an oracle identity and the coupling is
    graded on solved quantities instead.
    """
    for nu in (0.0, 0.3, 0.45):
        d = float(plate_moment_matrix(E, nu, T)[0, 0])
        d_b = E * T ** 3 / 12.0
        for corruption in (1.0, 7.3):            # any common factor cancels
            ratio = (corruption * d) / (corruption * d_b)
            assert abs(ratio - 1.0 / (1.0 - nu ** 2)) < 1e-12


def test_free_surface_state_is_plane_stress():
    state = free_surface_state(1.0e-3, NU)
    m = state.moments(E, NU, T)
    assert abs(float(m[1])) < 1e-12                    # M_y = 0
    assert abs(float(m[0]) - E * T ** 3 / 12.0 * 1.0e-3) < 1e-12   # D_b kappa_x
    assert float(state.kappa[1]) == pytest.approx(-NU * 1.0e-3)


@pytest.mark.parametrize("nu", [0.3, 0.45])
@pytest.mark.parametrize("skew", [0.0, 0.35], ids=["affine", "skew"])
@pytest.mark.parametrize("treatment", ["sri", "mitc"])
def test_free_surface_patch_is_exact(nu, skew, treatment):
    """`M_y = 0` is reproduced by the element's own `Db`, so this can fail."""
    rep = free_surface_check(affine_patch_mesh(3, skew=skew), 1.0e-3, young=E,
                             poisson=nu, thickness=T, shear_treatment=treatment)
    assert rep["my_over_mx"] < 1e-12
    assert rep["displacement_rel_error"] < 1e-12
    assert rep["rotation_rel_error"] < 1e-12
    assert rep["moment_rel_error"] < 1e-12
    assert rep["mx_over_db_kappa"] == pytest.approx(1.0, abs=1e-12)
    # grading against the cylindrical rigidity D instead of D_b must be rejected
    assert abs(rep["mx_over_db_kappa"] * rep["db_over_d"] - 1.0) > 0.05


def test_free_strip_is_transverse_converged():
    errs = {}
    for n_y in (8, 16):
        rep = strip_pure_bending(32, n_y, thickness=0.01, lateral="free",
                                 shear_treatment="sri", young=E, poisson=NU,
                                 width=0.2)
        errs[n_y] = rep["w_rel_error_kirchhoff"]
    assert abs(errs[16] - errs[8]) / errs[16] < 0.02
    # and the converged offset is still inside the case-pool tolerance
    assert errs[16] < 0.02


def test_free_strip_offset_is_a_clamped_root_end_effect():
    """The offset must decay with L/b at fixed element size, and must not be
    explained by the theta_y restraint of the root."""
    short = strip_pure_bending(32, 8, length=1.0, width=0.2, thickness=0.01,
                               lateral="free", shear_treatment="sri", young=E,
                               poisson=NU)
    long = strip_pure_bending(128, 8, length=4.0, width=0.2, thickness=0.01,
                              lateral="free", shear_treatment="sri", young=E,
                              poisson=NU)
    assert long["w_rel_error_kirchhoff"] < 0.6 * short["w_rel_error_kirchhoff"]
    loose = strip_pure_bending(32, 8, length=1.0, width=0.2, thickness=0.01,
                               lateral="free", shear_treatment="sri", young=E,
                               poisson=NU, clamp_theta_y=False)
    assert abs(loose["w_rel_error_kirchhoff"] - short["w_rel_error_kirchhoff"]) < 2e-3


def _loglog_slope(xs, ys):
    import math
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys)]
    n = len(pts)
    mx = sum(q[0] for q in pts) / n
    my = sum(q[1] for q in pts) / n
    return sum((q[0] - mx) * (q[1] - my) for q in pts) / sum(
        (q[0] - mx) ** 2 for q in pts)


THIN_LADDER = (1.0, 0.5, 0.2, 0.1, 0.05)


def _energy_ladder(patch, treatments, n=4):
    out = {}
    for treat in treatments:
        out[treat] = [
            run_patch_case(patch, STATES[2], young=E, poisson=NU, thickness=tau,
                           shear_treatment=treat)["energy_rel_error"]
            for tau in THIN_LADDER
        ]
    return out


def test_thin_limit_mitc_stays_shear_free():
    """MITC4 must reproduce the state at every thickness, on both mesh families.

    The locking element is 558 % off at the same t/L, so this is a separating
    test, not a round-off curiosity.
    """
    for patch in (affine_patch_mesh(4), distorted_patch_mesh(4)):
        for tau in (1.0, 0.05):
            rep = run_patch_case(patch, STATES[2], young=E, poisson=NU,
                                 thickness=tau, shear_treatment="mitc")
            assert rep["energy_rel_error"] < 1e-10
            assert rep["displacement_rel_error"] < 1e-10


def test_thin_limit_sri_is_exact_on_affine_meshes():
    """On an affine mesh thickness is *not* the trigger: SRI passes at t/L = 0.05."""
    patch = affine_patch_mesh(4)
    for tau in (1.0, 0.05):
        rep = run_patch_case(patch, STATES[2], young=E, poisson=NU, thickness=tau,
                             shear_treatment="sri")
        assert rep["energy_rel_error"] < 1e-10


def test_sri_distorted_thin_limit_follows_the_ht_squared_locking_law():
    """The recorded SRI limitation: on a distorted mesh the spurious shear energy
    grows as (h/t)^2.  Pinning the exponent identifies the mechanism (locking, not
    an implementation defect) and is what makes the thin-limit claim falsifiable.
    """
    e = _energy_ladder(distorted_patch_mesh(4), ("sri",))["sri"]
    assert max(e) > 1e-10                      # measurable, not round-off
    slope = _loglog_slope([1.0 / (4 * tau) for tau in THIN_LADDER], e)
    assert abs(slope - 2.0) < 0.5
    assert e[-1] / e[0] > 100.0                # amplification over h/t = 0.25 -> 5
    assert e[-1] > 0.02                        # honest: SRI is NOT 2 % at h/t = 5


def test_full_integration_locking_amplification_matches_the_pure_law():
    """Negative control, affine mesh: thinning 1 -> 0.05 must inflate the energy
    error by 20^2 = 400 (the law carries no fitted constant)."""
    e = _energy_ladder(affine_patch_mesh(4), ("full",))["full"]
    assert e[0] > 1e-3                         # locking is already visible when thick
    assert e[-1] / e[0] == pytest.approx(400.0, rel=0.02)
    slope = _loglog_slope([1.0 / (4 * tau) for tau in THIN_LADDER], e)
    assert abs(slope - 2.0) < 0.1


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