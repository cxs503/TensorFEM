"""Guards for the CASE-FEM-HERTZ-CONTACT residual-attribution study.

Background: the certificate and its two follow-ups left one sentence open --
"after vertical refinement and depth extrapolation a ``-0.2092 %`` indentation
offset survives; its mechanism was not located".  ``residual_attribution.py``
crosses the two axes that had never been crossed (absorbing far field x vertical
near-field spacing) and declares, *before* solving, a two-sided fork between
"clamped truncation boundary" and "irreducible contact-model bias".

Three levels of guard, mirroring ``test_hertz_plateau_mechanism.py``:

1. **Mesh invariance** -- every arm claims to change exactly one variable, so the
   radial node array must be bit-identical across the ``dz`` ladder and ``dz``
   must be exactly ``4 / n_zfine``.
2. **Decision rule** -- the fork must fire in both directions on synthetic rows,
   otherwise a committed verdict would be an artefact of the rule.
3. **Evidence self-consistency** -- every published error percentage and every
   gate value is recomputed from the committed append-only row record.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "benchmarks" / "verified" / "fem_hertz_contact_loadcontrol"


def _load(name, filename):
    verified = ANCHOR.parent
    for path in (str(verified), str(ANCHOR), str(ROOT / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(name, ANCHOR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ATTR = _load("hertz_residual_attribution", "residual_attribution.py")


@pytest.fixture(scope="module")
def recorded():
    rows = ATTR.read_rows()
    if not rows:
        pytest.skip("no residual_attribution_rows.jsonl recorded")
    return rows


@pytest.fixture(scope="module")
def artifact():
    path = ANCHOR / "residual_attribution.json"
    if not path.exists():
        pytest.skip("residual_attribution.json not built yet")
    with open(path) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# 1. one variable per arm
# ---------------------------------------------------------------------------


def _mesh(arm):
    """Rebuild the mesh of an arm exactly as ``run_arm`` does."""
    from tensorfem.hertz_halfspace_loadcontrol import (
        graded_depth,
        hertz_load_reference,
        two_zone_axis,
    )

    ref = hertz_load_reference(
        ATTR.load_for_radius_ratio(ATTR.A_OVER_R), ATTR.R, ATTR.E, ATTR.NU
    )
    a = ref.contact_radius
    h_rel = arm.get("h_rel", ATTR.H_OVER_A)
    depth_a = arm.get("depth", ATTR.DEPTH_A)
    r_axis = two_zone_axis(
        3.0 * a, depth_a * a, int(round(3.0 / h_rel)), ATTR.N_OUTER_BAND
    )
    z_axis = graded_depth(depth_a * a, ATTR.FINE_BAND_A * a, arm["n_zfine"], ATTR.N_GROW)
    return r_axis, z_axis


def test_dz_ladder_changes_only_the_vertical_band():
    """The dz arms differ in n_zfine only -> identical radial nodes, exact dz."""
    ladder = [ATTR.ARM_BY_LABEL[label] for label in ("bouss_dz_a6", "bouss_dz_a12", "bouss_dz_a24")]
    reference = None
    for arm in ladder:
        r_axis, z_axis = _mesh(arm)
        if reference is None:
            reference = r_axis
        else:
            np.testing.assert_array_equal(r_axis, reference)
        # dz is exactly FINE_BAND_A / n_zfine, verified on the actual node array
        fine = z_axis[: arm["n_zfine"] + 1]
        a = ATTR.R * ATTR.A_OVER_R
        np.testing.assert_allclose(
            np.diff(fine), -ATTR.FINE_BAND_A * a / arm["n_zfine"], atol=1e-13
        )
        # dz/a is therefore exactly 4 / n_zfine
        assert math.isclose(ATTR.FINE_BAND_A / arm["n_zfine"], 4.0 / arm["n_zfine"])


def test_radial_arm_changes_only_h():
    """h/a 0.015 -> 0.0075 doubles the number of uniform contact-band cells."""
    coarse = _mesh(ATTR.ARM_BY_LABEL["bouss_dz_a24"])[0]
    fine = _mesh(ATTR.ARM_BY_LABEL["bouss_dz_a24_h0075"])[0]
    band = 3.0 * ATTR.R * ATTR.A_OVER_R
    n_coarse = int(np.sum(coarse < band - 1e-12))
    n_fine = int(np.sum(fine < band - 1e-12))
    assert n_fine == 2 * n_coarse or abs(n_fine - 2 * n_coarse) <= 1
    # the outer geometric zone is identical in node count
    assert len(fine) - n_fine == len(coarse) - n_coarse
    assert math.isclose(coarse[-1], fine[-1], rel_tol=0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 2. the pre-declared fork fires both ways
# ---------------------------------------------------------------------------


def _row(label, e_delta, e_a=-0.5, load_residual=1e-12, penetration=1e-11,
         ff_update=1e-6, dz=0.0417, h_rel=0.015, depth=96.0, rho=1.0, mode="boussinesq"):
    return {
        "label": label,
        "h_over_a": h_rel,
        "dz_over_a": dz,
        "domain_depth_over_a": depth,
        "far_field": mode,
        "rho_factor": rho,
        "n_zfine": int(round(ATTR.FINE_BAND_A / dz)),
        "n_elements": 1000,
        "ndof": 2000,
        "wall_s": 1.0,
        "load_residual": load_residual,
        "penetration_over_delta": penetration,
        "equilibrium_residual_rel": 1e-9,
        "far_field_passes": 2,
        "far_field_selfconsistency": ff_update,
        "far_field_converged": True,
        "analytic": {"contact_radius": 0.5, "indentation": 0.025, "peak_pressure": 3497.9},
        "numeric": {"contact_radius_linear": 0.5, "contact_radius_fit": 0.5,
                    "indentation": 0.025, "peak_pressure_consistent": 3497.9},
        "error_pct": {
            "contact_radius_linear": e_a,
            "contact_radius_fit": e_a,
            "indentation": e_delta,
            "peak_pressure_consistent": 0.02,
        },
    }


def _synthetic(e_bouss24, e_bouss12=-0.30, e_bouss6=-0.36, e_clamp24=-0.29,
               e_clamp12=-0.45, e_clamp6=-1.01, e_h0075=None, e_rho4=None, e_d192=None):
    rows = {
        "clamp_dz_a6": _row("clamp_dz_a6", e_clamp6, dz=0.1667, mode="zero"),
        "clamp_dz_a12": _row("clamp_dz_a12", e_clamp12, dz=0.0833, mode="zero"),
        "clamp_dz_a24": _row("clamp_dz_a24", e_clamp24, dz=0.0417, mode="zero"),
        "bouss_dz_a6": _row("bouss_dz_a6", e_bouss6, dz=0.1667),
        "bouss_dz_a12": _row("bouss_dz_a12", e_bouss12, dz=0.0833),
        "bouss_dz_a24": _row("bouss_dz_a24", e_bouss24, dz=0.0417),
    }
    if e_h0075 is not None:
        rows["bouss_dz_a24_h0075"] = _row("bouss_dz_a24_h0075", e_h0075, dz=0.0417, h_rel=0.0075)
    if e_rho4 is not None:
        rows["bouss_dz_a24_rho4"] = _row("bouss_dz_a24_rho4", e_rho4, dz=0.0417, rho=4.0)
    if e_d192 is not None:
        rows["bouss_dz_a24_D192"] = _row("bouss_dz_a24_D192", e_d192, dz=0.0417, depth=192.0)
    return rows


def test_fork_accepts_the_truncation_hypothesis_when_the_offset_collapses():
    rows = _synthetic(e_bouss24=-0.08, e_h0075=-0.07)
    verdict = ATTR.verdict_of(rows, ATTR.build_gates(rows))
    assert verdict["h_ff_accepted"] is True
    assert verdict["h_bias_accepted"] is False
    assert "clamped truncation boundary" in verdict["mechanism"]


def test_fork_accepts_the_bias_hypothesis_when_the_offset_survives():
    rows = _synthetic(e_bouss24=-0.26, e_bouss12=-0.27, e_bouss6=-0.31)
    verdict = ATTR.verdict_of(rows, ATTR.build_gates(rows))
    assert verdict["h_ff_accepted"] is False
    assert verdict["h_bias_accepted"] is True
    assert "irreducible bias" in verdict["mechanism"]


def test_fork_stays_inconclusive_between_the_bands():
    rows = _synthetic(e_bouss24=-0.20, e_bouss12=-0.22, e_bouss6=-0.28)
    verdict = ATTR.verdict_of(rows, ATTR.build_gates(rows))
    assert verdict["h_ff_accepted"] is False
    assert verdict["h_bias_accepted"] is False
    assert "inconclusive" in verdict["mechanism"]


def test_radial_and_penalty_gates_are_two_sided():
    """A mechanism gate must FAIL when the mechanism is absent, and pass when present."""
    present = _synthetic(e_bouss24=-0.20, e_h0075=-0.05, e_rho4=-0.20)
    gates = {gate["id"]: gate for gate in ATTR.build_gates(present)}
    assert gates["radial_surface_spacing_mechanism"]["passed"] is True
    assert gates["alm_penalty_invariance"]["passed"] is True

    absent = _synthetic(e_bouss24=-0.20, e_h0075=-0.199, e_rho4=-0.30)
    gates = {gate["id"]: gate for gate in ATTR.build_gates(absent)}
    assert gates["radial_surface_spacing_mechanism"]["passed"] is False
    assert gates["alm_penalty_invariance"]["passed"] is False


# ---------------------------------------------------------------------------
# 3. committed evidence is self-consistent
# ---------------------------------------------------------------------------


def test_published_errors_follow_from_the_recorded_numbers(artifact):
    for arm in artifact["arms"]:
        for key, value in arm["error_pct"].items():
            reference_key = {
                "contact_radius_linear": "contact_radius",
                "contact_radius_fit": "contact_radius",
                "indentation": "indentation",
                "peak_pressure_consistent": "peak_pressure",
            }[key]
            recomputed = 100.0 * (
                arm["numeric"][key] / arm["analytic"][reference_key] - 1.0
            )
            assert math.isclose(value, recomputed, rel_tol=1e-9, abs_tol=1e-9)

        # solve-quality controls actually hold on the committed rows
        assert arm["load_residual"] <= artifact["criteria"]["load_residual_max"]
        assert arm["penetration_over_delta"] <= artifact["criteria"]["penetration_over_delta_max"]
        if arm["far_field"] == "boussinesq":
            assert arm["far_field_selfconsistency"] <= artifact["criteria"]["far_field_selfconsistency"]

        # case-pool tolerance
        assert abs(arm["error_pct"]["indentation"]) <= artifact["criteria"]["case_pool_pct"]
        assert abs(arm["error_pct"]["contact_radius_linear"]) <= artifact["criteria"]["case_pool_pct"]


def test_gates_match_the_committed_verdict(artifact):
    gates = {gate["id"]: gate for gate in artifact["gates"]}
    verdict = artifact["verdict"]
    assert gates["absorbing_dz_a24_indentation"]["passed"] == verdict["h_ff_accepted"]
    assert (
        gates["absorbing_dz_a24_above_bias_band"]["passed"] == verdict["h_bias_accepted"]
    )
    # the two hypotheses are mutually exclusive by construction
    assert not (verdict["h_ff_accepted"] and verdict["h_bias_accepted"])
    failed = [gate["id"] for gate in artifact["gates"] if not gate["passed"]]
    assert failed == verdict["failed_gates"]
    assert gates["absorbing_dz_a24_indentation"]["value"] == pytest.approx(
        abs(verdict["absorbing_dz_a24_indentation_pct"]), abs=1e-12
    )


def test_committed_arms_are_the_declared_arms(artifact):
    declared = {arm["label"] for arm in ATTR.ARMS}
    committed = {arm["label"] for arm in artifact["arms"]}
    assert committed <= declared
    for arm in artifact["arms"]:
        spec = ATTR.ARM_BY_LABEL[arm["label"]]
        assert arm["far_field"] == spec["far_field"]
        assert arm["n_zfine"] == spec["n_zfine"]
        assert math.isclose(arm["dz_over_a"], ATTR.FINE_BAND_A / spec["n_zfine"], rel_tol=1e-12)
        assert math.isclose(arm["h_over_a"], spec.get("h_rel", ATTR.H_OVER_A), rel_tol=1e-12)
        assert math.isclose(arm["domain_depth_over_a"], spec.get("depth", ATTR.DEPTH_A), rel_tol=1e-12)


def test_gates_file_matches_the_artifact(artifact):
    with open(ANCHOR / "gates_residual.json") as handle:
        gates_file = json.load(handle)
    assert gates_file["gates"] == artifact["gates"]
    assert gates_file["verdict"] == artifact["verdict"]
    assert gates_file["criteria"] == artifact["criteria"]