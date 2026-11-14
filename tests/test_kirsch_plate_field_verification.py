"""Fail-closed checks for the KIRSCH-PLATE domain-wide field verification.

The artifact ``benchmarks/verified/fem_kirsch_plate/field_verification.json`` is
generated, never hand-edited.  These tests re-derive every gated number from the
stored per-level data, re-check the oracle by its own boundary conditions, prove
that the two closed-form implementations in the repository agree, and re-solve
the coarse annulus end to end so a stale artifact cannot certify anything.
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
BENCH = ROOT / "benchmarks" / "verified" / "fem_kirsch_plate"
ARTIFACT = BENCH / "field_verification.json"
DRIVER = BENCH / "field_verification.py"

sys.path.insert(0, str(ROOT / "src"))


def _load_driver():
    spec = importlib.util.spec_from_file_location("kirsch_field_verification", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


@pytest.fixture(scope="module")
def payload():
    assert ARTIFACT.is_file(), "field_verification.json missing: run the driver"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_is_fresh_and_complete(payload):
    assert payload["schema"] == "tensorfem.verified-field.v1"
    assert payload["anchor"] == "fem_kirsch_plate"
    assert ARTIFACT.stat().st_mtime >= DRIVER.stat().st_mtime - 1.0, (
        "artifact is older than the driver: regenerate instead of certifying a stale file"
    )
    assert len(payload["levels"]) >= payload["criteria"]["levels_min"]
    assert payload["gates_total"] == len(payload["gates"])


def test_every_gate_carries_a_reference_and_threshold(payload):
    ids = set()
    for gate in payload["gates"]:
        assert gate["id"] not in ids, f"duplicate gate id {gate['id']}"
        ids.add(gate["id"])
        assert gate["reference"].strip(), f"{gate['id']} has no reference"
        assert gate["operator"] in ("<=", ">=")
        assert math.isfinite(float(gate["value"]))
        assert math.isfinite(float(gate["threshold"]))
        expected = (
            float(gate["value"]) <= float(gate["threshold"])
            if gate["operator"] == "<="
            else float(gate["value"]) >= float(gate["threshold"])
        )
        assert bool(gate["passed"]) is expected, f"{gate['id']} pass flag is not its own arithmetic"


def test_gated_numbers_recompute_from_stored_levels(payload):
    levels = payload["levels"]
    finest = levels[-1]
    criteria = payload["criteria"]
    gates = {gate["id"]: gate for gate in payload["gates"]}

    assert gates["kt_rel_error_finest"]["value"] == pytest.approx(
        abs(finest["kt"] - 3.0) / 3.0, rel=0, abs=1e-15
    )
    assert finest["kt_rel_error"] == pytest.approx(abs(finest["kt"] - 3.0) / 3.0, rel=0, abs=1e-15)
    assert gates["stress_field_l2_gauss"]["value"] == finest["stress_l2_rel_gauss"]
    assert gates["stress_field_l2_nodal"]["value"] == finest["stress_l2_rel_nodal"]
    assert gates["pointwise_probes"]["value"] == pytest.approx(
        max(p["relative_error"] for p in finest["probes"]), rel=0, abs=1e-15
    )
    assert gates["edge_traction_residual"]["value"] == finest["edge_traction_residual_over_T"]
    assert gates["energy_error"]["value"] == finest["energy_rel_error"]
    assert gates["displacement_field"]["value"] == finest["displacement_l2_rel"]

    rates = [
        math.log(levels[i]["stress_l2_rel_gauss"] / levels[i + 1]["stress_l2_rel_gauss"]) / math.log(2.0)
        for i in range(len(levels) - 1)
    ]
    assert gates["stress_field_rate"]["value"] == pytest.approx(min(rates), rel=1e-12)
    # the ladder must actually refine: both counts double
    for coarse, fine in zip(levels, levels[1:]):
        assert fine["n_theta"] == 2 * coarse["n_theta"]
        assert fine["n_radial"] == 2 * coarse["n_radial"]

    for key, gate_id in (
        ("stress_l2_rel_gauss_max", "stress_field_l2_gauss"),
        ("stress_l2_rel_nodal_max", "stress_field_l2_nodal"),
        ("probe_pointwise_rel_max", "pointwise_probes"),
        ("edge_traction_residual_max", "edge_traction_residual"),
        ("displacement_l2_rel_max", "displacement_field"),
        ("energy_rel_error_max", "energy_error"),
        ("kt_rel_error_max", "kt_rel_error_finest"),
        ("observed_rate_min", "stress_field_rate"),
    ):
        assert gates[gate_id]["threshold"] == criteria[key]


def test_gated_gates_all_pass(payload):
    failed = [gate["id"] for gate in payload["gates"] if not gate["passed"]]
    assert not failed, f"field verification gates failed: {failed}"


def test_probes_are_interior_and_point_at_the_declared_locations(payload):
    declared = [(p["r_over_a"], p["theta_degrees"]) for p in payload["levels"][0]["probes"]]
    for level in payload["levels"]:
        assert [(p["r_over_a"], p["theta_degrees"]) for p in level["probes"]] == declared
        for probe in level["probes"]:
            # a single-valued stress requires a strictly interior parent point
            assert max(abs(probe["parent_xi"]), abs(probe["parent_eta"])) <= 0.7, (
                f"probe {probe['r_over_a']},{probe['theta_degrees']} sits on an element edge"
            )
            assert level["probe_interior_margin_min"] >= 0.3


def test_oracle_self_check_recomputes(driver, payload):
    fresh = driver.oracle_field_selfcheck()
    stored = payload["oracle"]
    assert fresh["hole_edge_traction_free_max_over_T"] <= 1e-12
    assert fresh["kt_exact"] == pytest.approx(3.0, rel=0, abs=1e-12)
    assert fresh["hole_edge_hoop_min_over_T"] == pytest.approx(-1.0, rel=0, abs=1e-12)
    for key, value in stored.items():
        assert fresh[key] == pytest.approx(value, rel=1e-12, abs=1e-15), key


def test_two_independent_closed_forms_agree(driver):
    """``run.py``'s polar oracle and ``tensorfem.kirsch_hole`` must agree."""
    spec = importlib.util.spec_from_file_location("kirsch_anchor_run", BENCH / "run.py")
    anchor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anchor)
    hole = driver.kirsch_hole()
    r = np.linspace(1.0, 6.0, 51)
    theta = np.linspace(0.0, 2.0 * math.pi, 73)
    R, T = np.meshgrid(r, theta, indexing="ij")
    srr, stt, srt = anchor.kirsch_stress(R, T, radius=hole.radius, tension=hole.far_field)
    lrr, ltt, lrt = hole.stress_polar(R, T)
    scale = hole.far_field
    assert np.abs(srr - lrr).max() / scale < 1e-13
    assert np.abs(stt - ltt).max() / scale < 1e-13
    assert np.abs(srt - lrt).max() / scale < 1e-13


def test_coarse_level_reproduces_end_to_end(driver, payload):
    stored = payload["levels"][0]
    fresh = driver.solve_level(stored["n_theta"], stored["n_radial"])
    assert fresh["dofs"] == stored["dofs"]
    assert fresh["elements"] == stored["elements"]
    assert fresh["kt"] == pytest.approx(stored["kt"], rel=1e-10)
    assert fresh["stress_l2_rel_gauss"] == pytest.approx(stored["stress_l2_rel_gauss"], rel=1e-10)
    assert fresh["stress_l2_rel_nodal"] == pytest.approx(stored["stress_l2_rel_nodal"], rel=1e-10)
    assert fresh["edge_traction_residual_over_T"] == pytest.approx(
        stored["edge_traction_residual_over_T"], rel=1e-10
    )
    assert fresh["energy_rel_error"] == pytest.approx(stored["energy_rel_error"], rel=1e-10)


def test_kt_matches_the_anchor_certificate(payload):
    """Overlapping levels must reproduce the anchor's own Q8 cross-check rows.

    ``result.json`` (the anchor certificate) stores ``q8_crosscheck`` for
    ``n_theta`` = 24/48/96 from the same production path; this study re-solves
    those levels with its own measurement code, so the shared ``K_t`` values are
    an independent reproducibility check across the two drivers.
    """
    anchor = json.loads((BENCH / "result.json").read_text(encoding="utf-8"))
    reference = {row["n_theta"]: row["kt"] for row in anchor["q8_crosscheck"]}
    shared = [lv for lv in payload["levels"] if lv["n_theta"] in reference]
    assert len(shared) >= 3, "no overlapping levels with the anchor certificate"
    for level in shared:
        assert level["kt"] == pytest.approx(reference[level["n_theta"]], rel=1e-12), level["n_theta"]


def test_figures_tracked(payload):
    figures = payload.get("figures") or []
    assert figures, "the artifact declares no figures"
    for name in figures:
        path = ROOT / "docs" / "assets" / "kirsch_plate_field" / name
        assert path.is_file(), f"declared figure {name} is not in the tree"
        assert path.stat().st_size > 10_000, f"figure {name} looks empty"