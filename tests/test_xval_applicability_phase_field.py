"""Machine gates for the XVAL applicability screen (anchor:
``XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT``).

Two kinds of assertions are made:

* **Verdict gates** -- the phase-field AT2 / pseudopotential LBM pairing must screen
  as ``NOT_APPLICABLE`` with *every* blocking predicate failing, while a
  same-class twin pair must screen as ``APPLICABLE`` (a screen that answers "no"
  to everything would be worthless, so the positive control is part of the gate).
* **Substitute-anchor gates** -- the phase-field side must be verified against its
  own closed-form references: the 1-D AT2 profile (L2 and nodal error below the
  3 % self-imposed tolerance, second-order convergence) and the homogeneous
  saturation law ``d = 2Hl/(Gc + 2Hl)`` (exact, because it probes the history
  floor used by the production solver).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensorfem.xval_applicability import (
    ACCEPTANCE_REL_TOL,
    PHASE_FIELD_AT2,
    MethodDescriptor,
    at2_1d_reference_profile,
    convergence_study,
    interface_width_ratio,
    not_applicable_verdict,
    observed_order,
    platform_implementation_inventory,
    resolution_threshold,
    screen_applicability,
    solve_at2_homogeneous,
)

LENGTH = 1.0
ELL = 0.05


@pytest.fixture(scope="module")
def bundle() -> dict:
    return not_applicable_verdict(ell=ELL, length=LENGTH)


@pytest.fixture(scope="module")
def ladder() -> list[dict]:
    return convergence_study(ELL, LENGTH, (10, 20, 40, 80, 160))


# --------------------------------------------------------------------------------------
# verdict gates
# --------------------------------------------------------------------------------------
def test_pairing_screens_not_applicable_with_all_blocking_codes(bundle):
    screen = bundle["screen"]
    assert screen["verdict"] == "NOT_APPLICABLE"
    assert set(screen["blocking_failures"]) == {"X1", "X2", "X3", "X4", "X5"}


def test_each_failure_carries_its_own_evidence(bundle):
    for pred in bundle["screen"]["predicates"]:
        assert pred["passed"] is False
        assert pred["blocking"] is True
        # evidence must name both descriptors, not just restate the verdict
        assert "phase_field_at2" in pred["evidence"]
        assert "pseudopotential_lbm" in pred["evidence"]


def test_positive_control_same_class_twin_is_applicable():
    """A screen that always says NOT_APPLICABLE must fail this test."""
    twin = MethodDescriptor(
        key="phase_field_at2_clone",
        name="same model, second implementation",
        pde_class=PHASE_FIELD_AT2.pde_class,
        state_variable=PHASE_FIELD_AT2.state_variable,
        interface_width_control=PHASE_FIELD_AT2.interface_width_control,
        analytic_benchmarks=PHASE_FIELD_AT2.analytic_benchmarks,
        platform_module=PHASE_FIELD_AT2.platform_module,
    )
    verdict = screen_applicability(PHASE_FIELD_AT2, twin)
    assert verdict.verdict == "APPLICABLE"
    assert verdict.blocking_failures == []
    assert verdict.substitute_path == []


def test_twin_is_rejected_when_inventory_reports_a_stray_implementation():
    """X5 must be driven by the independent inventory scan, not by the descriptor."""
    twin = MethodDescriptor(
        key="phase_field_at2_clone",
        name="same model, second implementation",
        pde_class=PHASE_FIELD_AT2.pde_class,
        state_variable=PHASE_FIELD_AT2.state_variable,
        interface_width_control=PHASE_FIELD_AT2.interface_width_control,
        analytic_benchmarks=PHASE_FIELD_AT2.analytic_benchmarks,
        platform_module=PHASE_FIELD_AT2.platform_module,
    )
    polluted = {
        "package_dir": "/nonexistent",
        "modules_scanned": 3,
        "pseudopotential_hits": ["tensorfem/fake_shan_chen.py"],
        "tokens": ["pseudopotential"],
    }
    verdict = screen_applicability(PHASE_FIELD_AT2, twin, inventory=polluted)
    assert verdict.verdict == "NOT_APPLICABLE"
    assert verdict.blocking_failures == ["X5"]


def test_inventory_finds_no_pseudopotential_in_this_package():
    inv = platform_implementation_inventory()
    assert inv["modules_scanned"] > 100
    assert inv["pseudopotential_hits"] == []


def test_substitute_path_names_the_verified_side_and_the_missing_side(bundle):
    path = " ".join(bundle["screen"]["substitute_path"])
    assert "phase_field_at2" in path
    assert "absent from this repository" in path


def test_verdict_bundle_is_json_serialisable_with_provenance(bundle):
    text = json.dumps(bundle)
    back = json.loads(text)
    assert back["anchor"] == "XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT"
    assert back["screen"]["verdict"] == "NOT_APPLICABLE"
    assert set(back["acceptance"]) >= {
        "at2_profile_l2_ok",
        "at2_profile_nodal_ok",
        "l2_order_ok",
        "at2_homogeneous_ok",
        "width_invariant_under_l_scaling",
        "verdict_is_not_applicable",
    }
    assert all(back["acceptance"].values()), back["acceptance"]


def test_descriptor_validation_rejects_unknown_pde_class():
    with pytest.raises(ValueError):
        MethodDescriptor(
            key="bogus",
            name="bogus",
            pde_class="navier_stokes_spectral",
            state_variable="density",
            interface_width_control="none",
        )


# --------------------------------------------------------------------------------------
# analytic-profile gates
# --------------------------------------------------------------------------------------
def test_reference_profile_satisfies_its_boundary_conditions():
    x = np.linspace(0.0, LENGTH, 4001)
    d = at2_1d_reference_profile(x, ELL, LENGTH)
    assert d[0] == pytest.approx(1.0, rel=0, abs=1e-15)
    # natural BC d'(L) = 0 by central differencing at the far end
    slope = (d[-1] - d[-2]) / (x[-1] - x[-2])
    assert abs(slope) < 1e-6
    # and it is the decaying branch: monotone decreasing, d(L) ~ exp(-L/l)
    assert np.all(np.diff(d) < 0.0)
    # far-field value: cosh(0)/cosh(L/l) = 2 exp(-L/l)/(1 + exp(-2L/l))
    assert d[-1] == pytest.approx(1.0 / math.cosh(LENGTH / ELL), rel=1e-12)


def test_reference_profile_solves_the_at2_strong_form():
    """d = l^2 d'' for the AT2 functional, checked by finite differences."""
    x = np.linspace(0.0, 0.5 * LENGTH, 20001)
    d = at2_1d_reference_profile(x, ELL, LENGTH)
    d2 = np.gradient(np.gradient(d, x), x)
    interior = slice(200, -200)
    assert np.max(np.abs(d[interior] - ELL ** 2 * d2[interior])) < 1e-6


def test_numeric_profile_matches_analytic_within_three_percent(bundle):
    run = bundle["at2_reference_run"]
    assert run["newton_converged"] is True
    assert run["l2_relative_error"] < ACCEPTANCE_REL_TOL
    assert run["nodal_max_relative_error"] < ACCEPTANCE_REL_TOL
    assert ACCEPTANCE_REL_TOL < 0.15  # stricter than the owner's 15 % standard


def test_convergence_is_second_order_in_the_asymptotic_range(ladder):
    order = observed_order(ladder)
    assert 1.8 < order < 2.2, order
    # inside the asymptotic range (h/l <= 0.5) a halving of h must shrink the L2
    # error by ~4x
    asym = [r for r in ladder if r["h_over_l"] <= 0.5]
    assert len(asym) >= 3
    l2 = [r["l2_relative_error"] for r in asym]
    for coarse, fine in zip(l2[:-1], l2[1:]):
        assert 3.7 < coarse / fine < 4.3, (coarse, fine)


def test_pre_asymptotic_levels_are_reported_not_hidden(ladder):
    """h/l >= 1 is outside the asymptotic range; the ladder must say so."""
    coarse = [r for r in ladder if r["h_over_l"] >= 1.0]
    fine = [r for r in ladder if r["h_over_l"] <= 0.125]
    assert coarse and fine
    assert min(r["nodal_max_relative_error"] for r in coarse) > 0.3
    assert max(r["nodal_max_relative_error"] for r in fine) < ACCEPTANCE_REL_TOL
    thr = resolution_threshold(ladder, ACCEPTANCE_REL_TOL)
    assert thr["h_over_l_max"] <= 0.25
    assert thr["n_elem_min"] >= 160


def test_damage_only_problem_is_linear_one_newton_iteration(bundle):
    run = bundle["at2_reference_run"]
    assert run["newton_iterations"] == 1
    assert run["residual_norm"] < 1e-12


def test_unresolved_mesh_reports_failure_instead_of_a_fake_pass():
    """h/l = 4: the linear damage solve does not converge -- must be visible."""
    rows = convergence_study(ELL, LENGTH, (5,))
    assert rows[0]["newton_converged"] is False
    assert rows[0]["nodal_max_relative_error"] > 1.0


# --------------------------------------------------------------------------------------
# homogeneous-history anchor
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("H", [0.05, 0.5, 2.0, 10.0, 50.0])
def test_homogeneous_saturation_law_is_exact(H):
    res = solve_at2_homogeneous(H, ELL)
    assert res["newton_converged"] is True
    assert res["relative_error"] < 1e-12, res
    assert res["d_numeric"] == pytest.approx(2.0 * H * ELL / (1.0 + 2.0 * H * ELL))


# --------------------------------------------------------------------------------------
# width-control evidence (predicate X3)
# --------------------------------------------------------------------------------------
def test_interface_width_is_set_by_the_free_parameter_l():
    """Scale l *and* the domain: the discrete profiles are then affinely identical."""
    ratios = [
        interface_width_ratio(ELL, LENGTH, 160),
        interface_width_ratio(2.0 * ELL, 2.0 * LENGTH, 160),
        interface_width_ratio(4.0 * ELL, 4.0 * LENGTH, 160),
    ]
    for ratio in ratios:
        assert ratio == pytest.approx(1.0, abs=5e-3)
    assert max(ratios) - min(ratios) < 1e-6, ratios


def test_width_ratio_ignores_mesh_density():
    """x(d=1/e)/l must be a property of the model, not of the discretisation."""
    coarse = interface_width_ratio(ELL, LENGTH, 40)
    fine = interface_width_ratio(ELL, LENGTH, 160)
    # the dimensionless width converges from the coarse side; 1 % is the honest bar
    assert abs(coarse - fine) < 1e-2, (coarse, fine)
    assert fine == pytest.approx(1.0, abs=2e-3)


# --------------------------------------------------------------------------------------
# end-to-end harness artefact
# --------------------------------------------------------------------------------------
def test_harness_artefact_matches_the_module_verdict(tmp_path):
    import json as _json
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["python", str(repo / "examples" / "xval_not_applicable_phase_field.py"),
         "--out", str(tmp_path), "--no-figure"],
        capture_output=True, text=True, cwd=str(repo),
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    payload = _json.loads((tmp_path / "xval_not_applicable_phase_field.json").read_text())
    assert payload["screen"]["verdict"] == "NOT_APPLICABLE"
    assert payload["at2_reference_run"]["l2_relative_error"] < ACCEPTANCE_REL_TOL
    assert all(payload["acceptance"].values())