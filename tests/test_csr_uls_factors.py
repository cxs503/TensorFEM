"""Tests for the W24 CSR ULS factor anchor and its wiring into the W23 scan.

The tests assert invariants, not just execution: the transcription locators must
be complete, the unit/dimension check must have teeth (a mutation of a declared
coefficient unit must be caught), the factors must be read from the anchor
artifact (a mutation of the artifact must move the utilisation), and the two
utilisation bases must satisfy the analytic attribution identity.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tensorfem import csr_uls_factors as f

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "benchmarks" / "hull_girder_uls"


@pytest.fixture(scope="module")
def anchor() -> dict:
    return json.loads((ARTIFACT_DIR / "csr_uls_anchor.json").read_text())


@pytest.fixture(scope="module")
def rows(anchor: dict) -> list[dict]:
    return f.utilisation_rows(anchor=anchor, scan_rows=f.read_scan_rows(),
                              hull=f.hull_particulars())


@pytest.fixture(scope="module")
def scan_rows() -> list[dict]:
    return f.read_scan_rows()


# ---------------------------------------------------------------------------
# artifacts and transcription completeness
# ---------------------------------------------------------------------------
def test_artifacts_exist(anchor: dict) -> None:
    for name in ("csr_uls_anchor.json", "csr_utilisation_with_factors.csv",
                 "factor_sensitivity.json", "README.md"):
        assert (ARTIFACT_DIR / name).is_file(), name
    assert anchor["anchor_id"] == "AN-IACS-CSR"
    assert anchor["status"] == "TRANSCRIBED"
    assert anchor["mission_id"] == 15972


def test_every_transcribed_clause_has_locator_and_evidence(anchor: dict) -> None:
    entries = anchor["transcription"]["entries"]
    assert len(entries) >= 15
    for entry in entries:
        assert entry["clause"], entry["id"]
        assert isinstance(entry["page"], int) and entry["page"] > 0, entry["id"]
        if entry["status"] == "TRANSCRIBED":
            assert entry.get("quote") or entry.get("relation") or entry.get("values"), entry["id"]
        else:
            assert entry["status"] == "WITHHELD"
            assert entry.get("needed"), entry["id"]


def test_factor_values_are_the_printed_table2_values(anchor: dict) -> None:
    table = anchor["transcription"]["factor_table"]
    assert table["A"] == {"gamma_S": 1.0, "gamma_W": 1.2}
    assert table["B"] == {"gamma_S": 1.0, "gamma_W": 1.3}
    capacity = anchor["transcription"]["capacity_factors"]
    assert capacity["gamma_M"] == 1.1
    assert capacity["gamma_DB"]["hogging_oil_tanker"] == 1.10
    assert capacity["gamma_DB"]["sagging"] == 1.0


def test_source_documents_carry_a_url_and_a_digest(anchor: dict) -> None:
    documents = anchor["source_documents"]
    assert {d["id"] for d in documents} == {"AN-IACS-CSR", "AN-IACS-UR-S7",
                                            "AN-IACS-UR-S11"}
    for document in documents:
        assert document["url"].startswith("https://")
        assert len(document["sha256"]) == 64
        assert document["pdf_pages"] > 0


# ---------------------------------------------------------------------------
# machine check 1: dimensions / printed unit conversions
# ---------------------------------------------------------------------------
def test_dimension_check_passes(anchor: dict) -> None:
    report = anchor["dimension_check"]
    assert report["verdict"] == "PASS"
    assert report["max_residual"] < 1e-6
    assert report["checked_relations"] >= 6
    for entry in report["entries"]:
        assert entry["verdict"] == "PASS", entry
        assert entry["dimension_residual"] < 1e-6
        assert entry["scale_residual"] < 1e-6


def test_dimension_check_catches_a_wrong_coefficient_unit() -> None:
    """Mutating S7's declared m3->cm3 conversion must be detected."""
    mutated = deepcopy(f.CLAUSES)
    for clause in mutated:
        if clause["id"] == "S7-01":
            clause["relation"]["factor_unit"] = "1"      # drop the 1e6 conversion
    report = f.dimension_report(mutated)
    assert report["verdict"] == "FAIL"
    offending = [e for e in report["entries"] if e["clause"] == "S7-01"]
    assert offending and any(e["verdict"] == "FAIL" for e in offending)
    assert max(e["residual"] for e in offending) > 1e3


def test_dimension_check_catches_a_wrong_wave_moment_scale() -> None:
    """The 0.19 wave-moment coefficient is kN/m2; dropping that unit must fail."""
    mutated = deepcopy(f.CLAUSES)
    for clause in mutated:
        if clause["id"] == "CSR-WAV-01":
            clause["relation"]["factor_unit"] = "1"
    report = f.dimension_report(mutated)
    assert report["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# machine check 2: internal consistency
# ---------------------------------------------------------------------------
def test_consistency_checks_pass(anchor: dict) -> None:
    report = anchor["consistency_checks"]
    assert report["verdict"] == "PASS"
    assert report["max_internal_residual"] < 1e-6
    by_id = {check["id"]: check for check in report["checks"]}
    assert by_id["CC-02"]["residual"] < 1e-6          # S7 vs repo UR S11 closure
    assert by_id["CC-01"]["residual"] < 5e-3          # cross-document rounding
    assert by_id["CC-04"]["residual"] < 1e-12         # gamma* reciprocity


def test_s7_minimum_modulus_is_reproduced_by_the_anchor() -> None:
    hull = f.hull_particulars()
    w_min_cm3 = f.s7_coefficient(hull["Lpp"]) * hull["Lpp"] ** 2 * hull["B"] * \
        (hull["Cb"] + 0.7)
    assert w_min_cm3 * 1e-6 == pytest.approx(hull["reference_modulus_m3"], rel=1e-12)


def test_wave_coefficient_and_distribution_factor_branches() -> None:
    assert f.wave_coefficient(320.0) == pytest.approx(10.75)
    assert f.wave_coefficient(300.0) == pytest.approx(10.75)
    assert f.wave_coefficient(280.0) == pytest.approx(10.75 - (20.0 / 100.0) ** 1.5)
    with pytest.raises(ValueError):
        f.wave_coefficient(60.0)
    assert f.distribution_factor_fm(0.2) == pytest.approx(0.5)
    assert f.distribution_factor_fm(0.4) == pytest.approx(1.0)
    assert f.distribution_factor_fm(0.65) == pytest.approx(1.0)
    assert f.distribution_factor_fm(0.70) == pytest.approx(1.0 - 0.05 / 0.35)
    assert f.distribution_factor_fm(0.0) == pytest.approx(0.0)


def test_wave_moment_matches_the_printed_closed_form() -> None:
    hull = f.hull_particulars()
    hog = f.wave_moment_csr(length_m=hull["Lpp"], breadth_m=hull["B"],
                            block_coefficient=hull["Cb"], x_over_l=0.5,
                            condition="hogging")
    sag = f.wave_moment_csr(length_m=hull["Lpp"], breadth_m=hull["B"],
                            block_coefficient=hull["Cb"], x_over_l=0.5,
                            condition="sagging")
    cb = hull["Cb"]
    expected_hog = 0.19 * 1.0 * 1.0 * 1.0 * 10.75 * hull["Lpp"] ** 2 * hull["B"] * cb * 1e3
    expected_sag = -0.19 * 0.58 * (cb + 0.7) / cb * 10.75 * hull["Lpp"] ** 2 * \
        hull["B"] * cb * 1e3
    assert hog == pytest.approx(expected_hog, rel=1e-12)
    assert sag == pytest.approx(expected_sag, rel=1e-12)
    assert sag < 0.0 < hog


# ---------------------------------------------------------------------------
# utilisation wiring
# ---------------------------------------------------------------------------
def test_twelve_scan_cases_carry_a_factored_verdict(rows: list[dict]) -> None:
    assert len(rows) == 24                          # 12 cases x 2 scenarios
    verdicts = [r for r in rows if r["verdict"] in ("PASS", "FAIL")]
    assert len(verdicts) >= 12
    assert {r["verdict"] for r in rows} <= {"PASS", "FAIL", "NOT_APPLICABLE"}
    # honest reporting: the scenario-A sagging cases near the cargo-hold ends fail
    failures = [r for r in verdicts if r["verdict"] == "FAIL"]
    assert failures, "the scan must report failing cases if the check is not met"
    assert all(r["utilisation_with_factors"] > 1.0 for r in failures)
    assert all(r["gamma_R"] == pytest.approx(1.21) for r in rows
               if r["condition"] == "hogging")
    assert all(r["gamma_R"] == pytest.approx(1.10) for r in rows
               if r["condition"] == "sagging")


def test_scenario_b_hogging_rows_are_not_csr_load_cases(rows: list[dict]) -> None:
    for row in rows:
        if row["scenario"] == "B" and row["condition"] == "hogging":
            assert row["verdict"] == "NOT_APPLICABLE"
        else:
            assert row["verdict"] in ("PASS", "FAIL")


def test_factors_are_read_from_the_anchor(rows: list[dict], anchor: dict,
                                          scan_rows: list[dict]) -> None:
    """Doubling gamma_W in the artifact copy must move the utilisation."""
    mutated = deepcopy(anchor)
    mutated["transcription"]["factor_table"]["A"]["gamma_W"] = 2.4
    rows_mutated = f.utilisation_rows(anchor=mutated, scan_rows=scan_rows,
                                      hull=f.hull_particulars(), scenarios=("A",))
    base = {(r["case_id"]): r for r in rows if r["scenario"] == "A"}
    for row in rows_mutated:
        reference = base[row["case_id"]]
        extra = (2.4 - 1.2) * reference["f_beta"] * reference["M_wv_csr_Nm"]
        expected = reference["utilisation_with_factors"] + \
            reference["gamma_R"] * extra / reference["M_u_smith_Nm"]
        assert row["utilisation_with_factors"] == pytest.approx(expected, rel=1e-12)


def test_gamma_star_and_unit_factor_utilisation_are_reciprocal(rows: list[dict]) -> None:
    for row in rows:
        assert row["unit_factor_utilisation"] * row["gamma_star"] == \
            pytest.approx(1.0, rel=1e-12)


def test_attribution_identity_holds_for_every_case(rows: list[dict]) -> None:
    for row in rows:
        assert row["attribution_residual"] < 1e-9, row
        assert row["attribution_wave_factor"] == pytest.approx(1.2 * 1.05) \
            if row["scenario"] == "A" else True
    for row in rows:
        if row["scenario"] == "A":
            expected = row["gamma_R"] * (row["gamma_S"] *
                                         row["attribution_still_water_share"] +
                                         row["attribution_wave_factor"] *
                                         row["attribution_wave_share"])
            assert row["attribution_product"] == pytest.approx(expected, rel=1e-12)


def test_csv_matches_the_anchor_and_the_rows(anchor: dict, rows: list[dict]) -> None:
    with (ARTIFACT_DIR / "csr_utilisation_with_factors.csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == len(rows)
    for csv_row, row in zip(table, rows):
        assert csv_row["case_id"] == row["case_id"]
        assert csv_row["scenario"] == row["scenario"]
        assert float(csv_row["gamma_W"]) == anchor["transcription"]["factor_table"][
            row["scenario"]]["gamma_W"]
        assert float(csv_row["utilisation_with_factors"]) == pytest.approx(
            row["utilisation_with_factors"], rel=1e-12)
        assert float(csv_row["attribution_residual"]) < 1e-9


def test_relative_difference_is_reported_for_every_case(rows: list[dict]) -> None:
    applicable = [r for r in rows if r["verdict"] in ("PASS", "FAIL")]
    for row in applicable:
        assert row["rel_diff_percent"] > 0.0        # the factors raise the demand
        assert row["rel_diff_percent"] == pytest.approx(
            (row["utilisation_with_factors"] / row["unit_factor_utilisation"] - 1.0) * 100.0,
            rel=1e-12)
    # every case exceeds 15 %, so every case must be attributable
    assert all(abs(r["rel_diff_percent"]) > 15.0 for r in applicable)


# ---------------------------------------------------------------------------
# sensitivity envelope
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sensitivity() -> dict:
    return json.loads((ARTIFACT_DIR / "factor_sensitivity.json").read_text())


def test_sensitivity_marks_the_transcribed_points(sensitivity: dict) -> None:
    points = [(g["gamma_S"], g["gamma_W"]) for g in sensitivity["grid"]
              if g["is_transcribed_csr_point"]]
    assert (1.0, 1.2) in points
    assert (1.0, 1.3) in points
    assert sensitivity["status"] == "SENSITIVITY_ENVELOPE"


def test_break_even_factors_sit_exactly_on_the_limit(sensitivity: dict) -> None:
    anchor = {"transcription": {
        "factor_table": f.factor_table(), "capacity_factors": f.capacity_factors(),
        "heading_factor": f.heading_factor(),
        "probability_factors": f.probability_factors()}}
    hull = f.hull_particulars()
    for case in sensitivity["break_even"]:
        gamma_w = case["break_even_gamma_w_at_gamma_s_1"]
        if gamma_w is None:
            continue
        u = case["gamma_R"] * (1.0 * case["M_sw_Nm"] +
                               gamma_w * 1.05 * case["M_wv_csr_Nm"]) / case["M_u_smith_Nm"]
        assert u == pytest.approx(1.0, rel=1e-9)
        assert case["gamma_star"] == pytest.approx(
            case["M_u_smith_Nm"] / (case["M_sw_Nm"] + case["M_wv_csr_Nm"]), rel=1e-12)
    assert hull["Lpp"] == pytest.approx(320.0)


def test_sensitivity_grid_is_monotone_in_gamma_w(sensitivity: dict) -> None:
    grid = {(g["gamma_S"], g["gamma_W"]): g["failures"] for g in sensitivity["grid"]}
    for gamma_s in sensitivity["grid_axes"]["gamma_S"]:
        counts = [grid[(gamma_s, w)] for w in sensitivity["grid_axes"]["gamma_W"]]
        assert counts == sorted(counts)
        assert counts[-1] >= counts[0]