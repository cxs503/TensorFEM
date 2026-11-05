"""Fail-closed checks on the structural-certification material.

The committed ``benchmarks/structural_certification_materials.json`` is the
evidence of record, so the tests recompute every verdict from the stored
numerator/denominator instead of trusting the stored label, and they check the
three-state discipline (a ``WITHHELD`` row must carry a reason, a ``PASS`` row
must be inside the tolerance).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "benchmarks" / "structural_certification_materials.json"
DOC_PATH = ROOT / "docs" / "structural-certification-materials.md"
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.certification_evidence import (  # noqa: E402
    Criterion,
    aggregate_verdicts,
    classify_pointwise,
    combine_envelopes,
    envelope_uncertainty,
    grid_uncertainty,
)


@pytest.fixture(scope="module")
def payload() -> dict:
    assert JSON_PATH.exists(), f"missing evidence file {JSON_PATH}"
    return json.loads(JSON_PATH.read_text())


def _all_rows(payload: dict) -> list[dict]:
    return list(payload["cases"]["kirsch"]["pointwise_rows"]) + list(
        payload["cases"]["finite_width"]["rows"]
    )


def test_every_row_carries_numerator_denominator_and_threshold(payload: dict) -> None:
    rows = _all_rows(payload)
    assert rows, "no comparison rows recorded"
    threshold = payload["criterion"]["relative_tolerance"]
    assert threshold == pytest.approx(0.03)
    for row in rows:
        assert math.isfinite(row["fe"]) and math.isfinite(row["reference"])
        assert row["numerator_abs_error"] == pytest.approx(
            abs(row["fe"] - row["reference"]), rel=1e-12, abs=0.0
        )
        assert row["denominator_abs_reference"] == pytest.approx(
            abs(row["reference"]), rel=1e-12, abs=0.0
        )
        assert row["threshold_relative"] == pytest.approx(threshold)
        assert row["verdict"] in {"PASS", "FAIL", "WITHHELD"}


def test_verdicts_recompute_from_the_stored_numbers(payload: dict) -> None:
    threshold = payload["criterion"]["relative_tolerance"]
    for row in _all_rows(payload):
        if row["verdict"] == "WITHHELD":
            assert row["relative_error"] is None
            assert row["withheld_reason"], f"withheld row without a reason: {row['label']}"
            assert row["denominator_abs_reference"] < row["denominator_floor"]
            continue
        relative = row["numerator_abs_error"] / row["denominator_abs_reference"]
        assert row["relative_error"] == pytest.approx(relative, rel=1e-9)
        expected = "PASS" if relative <= threshold else "FAIL"
        assert row["verdict"] == expected, f"verdict mismatch on {row['label']}"


def test_headline_three_percent_answers_are_arithmetic(payload: dict) -> None:
    quantities = payload["cases"]["kirsch"]["quantities"]
    answers = payload["answers"]["within_3pct"]
    assert set(answers) == set(quantities)
    for name, item in quantities.items():
        relative = abs(item["fe"] - item["reference"]) / abs(item["reference"])
        assert item["relative_error"] == pytest.approx(relative, rel=1e-9)
        assert item["within_3pct"] == (relative <= 0.03)
        assert answers[name]["within_3pct"] == item["within_3pct"]
        assert answers[name]["relative_error"] == pytest.approx(item["relative_error"])
        assert item["verdict"] == ("PASS" if relative <= 0.03 else "FAIL")


def test_uncertainty_budget_is_consistent(payload: dict) -> None:
    for item in payload["cases"]["kirsch"]["quantities"].values():
        combined = item["combined_uncertainty"]
        parts = combined["contributions"]
        assert combined["half_width"] == pytest.approx(
            sum(part["half_width"] for part in parts), rel=1e-12
        )
        assert len(parts) == 2
        disc = item["discretisation"]
        assert disc["status"] in {"ESTIMATED", "WITHHELD"}
        if disc["status"] == "WITHHELD":
            assert disc["reason"]
            assert disc["richardson_value"] is None
        else:
            assert disc["observed_order"] > 0.0
            assert disc["half_width"] > 0.0


def test_mesh_geometry_gates_pass(payload: dict) -> None:
    for record in payload["cases"]["kirsch"]["mesh_ladder"]:
        geometry = record["geometry"]
        assert geometry["verdict"] == "PASS", geometry
        assert geometry["orientation_consistent"] is True
        assert geometry["relative_area_error_vs_polygon"] < 1e-12
        # The deficit itself is O(n^-2) ~ 1e-4, so comparing it with its closed form
        # loses about eight digits to cancellation; 1e-9 is still far below the deficit.
        assert geometry["deficit_closed_form_relative_error"] < 1e-9


def test_sections_state_follows_row_verdicts(payload: dict) -> None:
    kirsch = payload["cases"]["kirsch"]
    summary = kirsch["summary"]["all_pointwise_rows"]
    recomputed = aggregate_verdicts(kirsch["pointwise_rows"])
    assert summary == recomputed
    plate = payload["cases"]["finite_width"]["summary"]
    assert plate == aggregate_verdicts(payload["cases"]["finite_width"]["rows"])


def test_figures_exist_and_are_referenced(payload: dict) -> None:
    figures = payload["figures"]
    assert figures, "no figures recorded"
    assert len(figures) >= 1
    for figure in figures:
        path = ROOT / figure["path"]
        assert path.exists(), f"missing figure {path}"
        assert path.stat().st_size > 10_000, f"figure too small to be a real plot: {path}"
        assert figure["caption"]
    clouds = [figure for figure in figures if "clouds" in figure["name"]]
    assert clouds, "at least one field-cloud figure is required"
    assert clouds[0]["fields"]["samples"] > 1000


def test_document_lists_the_figures(payload: dict) -> None:
    assert DOC_PATH.exists()
    text = DOC_PATH.read_text()
    for figure in payload["figures"]:
        assert figure["path"] in text
    assert "WITHHELD" in text
    assert "3 %" in text


def test_criterion_logic_units() -> None:
    criterion = Criterion(relative_tolerance=0.03, denominator_floor=0.01)
    inside = classify_pointwise("hoop", 3.02, 3.0, scale=1.0, criterion=criterion)
    assert inside["verdict"] == "PASS"
    outside = classify_pointwise("hoop", 3.2, 3.0, scale=1.0, criterion=criterion)
    assert outside["verdict"] == "FAIL"
    ill_conditioned = classify_pointwise("shear", 1e-18, 0.0, scale=1.0, criterion=criterion)
    assert ill_conditioned["verdict"] == "WITHHELD"
    assert ill_conditioned["relative_error"] is None


def test_grid_uncertainty_preconditions() -> None:
    monotone = [
        {"level": 24, "value": 2.67020},
        {"level": 32, "value": 2.67677},
        {"level": 48, "value": 2.68247},
    ]
    estimated = grid_uncertainty(monotone, reference=2.6875684514168476)
    assert estimated["status"] == "ESTIMATED"
    assert estimated["observed_order"] > 0.0
    assert estimated["half_width"] > 0.0
    non_monotone = [
        {"level": 24, "value": 3.001495},
        {"level": 32, "value": 3.002832},
        {"level": 48, "value": 3.003036},
    ]
    withheld = grid_uncertainty(non_monotone, reference=3.0)
    assert withheld["status"] == "WITHHELD"
    assert withheld["reason"]
    assert withheld["half_width"] == envelope_uncertainty(
        [row["value"] for row in non_monotone]
    )["half_width"]


def test_envelope_combination_is_a_labelled_worst_case_sum() -> None:
    combined = combine_envelopes(
        [{"label": "a", "half_width": 0.1}, {"label": "b", "half_width": 0.05}]
    )
    assert combined["half_width"] == pytest.approx(0.15)
    assert "worst case" in combined["method"]
    assert "non-statistical" in combined["method"]


def test_committed_path_reproduces_a_known_hoop_error() -> None:
    """Cheap re-solve proving the committed pipeline is wired to the production path."""
    from tensorfem.kirsch_hole import KirschHole, solve_case

    case = solve_case(
        KirschHole(),
        10.0,
        n_theta=16,
        n_radial=16,
        grading=5.0,
        sample_points=((1.05, math.pi / 2.0),),
    )
    row = case["stress_rows"][0]
    assert row["hoop_relative_error"] < 0.05
    assert row["inside"] is True