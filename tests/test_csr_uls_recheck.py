"""Tests for the W26 second-pass recheck of the W24 CSR ULS anchor.

The suite locks two things.  First the *mechanism*: quote normalisation, the
granularity ladder (verbatim -> normalised -> cell -> restated), the text-layer
garbling detector and the Table 2 row grouping must all have teeth, so a
mutation of the input must move the verdict.  Second the *artifact*: it must
still describe the anchor and the generator that are in the tree, and its
accounting must add up - ``verify_artifact`` is exercised both ways (clean
artifact passes, a tampered copy is rejected).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "benchmarks" / "hull_girder_uls" / "recheck_csr_uls_anchor.py"
ANCHOR = REPO_ROOT / "benchmarks" / "hull_girder_uls" / "csr_uls_anchor.json"
ARTIFACT = (REPO_ROOT / "papers" / "artifacts" / "w26"
            / "hull-girder-ultimate-strength" / "csr_uls_recheck.json")


@pytest.fixture(scope="module")
def recheck():
    spec = importlib.util.spec_from_file_location("_w26_recheck", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


# ---------------------------------------------------------------------------
# mechanism
# ---------------------------------------------------------------------------
def test_quote_segments_split_on_the_ellipses_the_transcription_uses(recheck):
    quote = ("gamma_R = gamma_M gamma_DB ... gamma_M = 1.1 ... for sagging "
             "condition: gamma_DB = 1.0")
    segments = recheck.quote_segments(quote)
    assert segments == ["gamma_R = gamma_M gamma_DB", "gamma_M = 1.1",
                        "for sagging condition: gamma_DB = 1.0"]
    assert recheck.quote_segments(None) == []
    assert recheck.quote_segments("ab") == []  # too short to be evidence


def test_normalisation_tolerates_the_pdf_symbol_font(recheck):
    printed = "fm = 0.0 for x \u2264 0\n\n  and  \u03b3R = \u03b3M \u03b3DB"
    quoted = "fm = 0.0 for x <= 0 and gamma_R = gamma_M gamma_DB"
    assert recheck.normalise_strict(printed) != quoted
    assert recheck.normalise_loose(printed) == recheck.normalise_loose(quoted)


def test_atomic_evidence_requires_every_number_to_be_printed(recheck):
    pages = ["Cw = 10.75 - ((300 - L)/100) for the wave coefficient"]
    accepted = recheck._atomic_evidence(
        "Cw = 10.75 - ((300 - L)/100)^1.5 for the wave coefficient", pages, {1})
    assert not accepted["accepted"]
    assert accepted["missing_numbers"] == ["1.5"]

    accepted = recheck._atomic_evidence(
        "Cw = 10.75 - ((300 - L)/100) for the wave coefficient", pages, {1})
    assert accepted["accepted"] and accepted["missing_numbers"] == []
    assert accepted["word_coverage"] == 1.0


def test_garbled_detector_finds_reordered_constants_and_prefers_the_short_run(recheck):
    page = "1989 2001 75.10 for 300 m < L < 350 m"
    assert recheck._garbled_fragment(page, "10.75") == "75.10"
    assert recheck._garbled_fragment(page, "10.75") != "1989"
    assert recheck._garbled_fragment("no digits at all", "10.75") is None
    # the digits of 0.19 are all present, out of order, in the UR S11 fragment
    assert recheck._garbled_fragment("BCMCLM 10190", "0.19") == "10190"


def test_quote_granularity_ladder_separates_verbatim_from_restated(recheck):
    pages = ["Table 1 : Design load scenarios", "A\nS+D  Msw-h or Msw-s"]
    verbatim = recheck.locate_quote(pages, "Table 1 : Design load scenarios", {1, 2})
    assert verbatim[0]["level"] == "verbatim"

    restated = recheck.locate_quote(
        pages, "Table 1 : Design load scenarios. A / S+D: Msw-h or Msw-s", {1, 2})
    assert restated[0]["level"] == "restated"

    absent = recheck.locate_quote(pages, "Table 4 : Wave loads 0.31", {1, 2})
    assert absent[0]["level"] == "absent"


def test_table_row_grouping_needs_two_columns_and_orders_by_x(recheck):
    tokens = [{"text": "1.0", "x": 352.4, "y": 699.3},
              {"text": "1.2", "x": 396.6, "y": 699.3},
              {"text": "1.0", "x": 352.4, "y": 682.8},
              {"text": "1.3", "x": 396.6, "y": 682.8},
              {"text": "2.3", "x": 63.0, "y": 656.9}]
    rows = recheck.group_factor_rows(recheck.factor_tokens(tokens))
    assert len(rows) == 2                      # the lone section number is dropped
    assert [row[0]["text"] for row in rows] == ["1.0", "1.0"]
    assert [row[1]["text"] for row in rows] == ["1.2", "1.3"]
    assert rows[0][0]["y"] > rows[1][0]["y"]   # top row first


# ---------------------------------------------------------------------------
# artifact integrity
# ---------------------------------------------------------------------------
def test_artifact_describes_the_anchor_that_is_in_the_tree(artifact):
    anchor_sha = hashlib.sha256(ANCHOR.read_bytes()).hexdigest()
    generator_sha = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
    assert artifact["recheck_of"]["sha256"] == anchor_sha
    assert artifact["generator_sha256"] == generator_sha
    assert artifact["mission_id"] == 17566
    assert artifact["repository"] == "TensorFEM"
    assert artifact["recheck_of"]["anchor_id"] == json.loads(
        ANCHOR.read_text())["anchor_id"]


def test_artifact_verifier_accepts_the_shipped_artifact(recheck):
    report = recheck.verify_artifact(ARTIFACT)
    assert report["verdict"] == "PASS", report["problems"]
    assert report["checks"] >= 10


def test_artifact_verifier_rejects_tampering(recheck, tmp_path, artifact):
    tampered = copy.deepcopy(artifact)
    tampered["criteria"]["W26-C2"]["verdict"] = "FAIL"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    assert recheck.verify_artifact(path)["verdict"] == "FAIL"

    tampered = copy.deepcopy(artifact)
    tampered["error_judgement"][0]["under_3_percent"] = False  # hides a real agreement
    path.write_text(json.dumps(tampered))
    assert any("error judgement" in problem
               for problem in recheck.verify_artifact(path)["problems"])

    tampered = copy.deepcopy(artifact)
    tampered["recheck_of"]["sha256"] = "0" * 64
    path.write_text(json.dumps(tampered))
    assert recheck.verify_artifact(path)["verdict"] == "FAIL"


def test_six_criteria_are_adjudicated_one_by_one(artifact):
    six = artifact["six_criteria_verdicts"]
    assert list(six) == [f"W26-C{index}" for index in range(1, 7)]
    assert all(row["verdict"] == "PASS" for row in six.values())
    assert artifact["criteria_set_definition"]["six"] == [row["entry"]
                                                          for row in six.values()]
    # every entry reports where it was found, and nothing moved off its page
    for row in six.values():
        assert row["found_printed_pages"], row["entry"]
        assert row["numbers_missing_in_window"] == []


def test_every_anchor_entry_is_accounted_for(artifact):
    anchor = json.loads(ANCHOR.read_text())
    anchor_ids = sorted(entry["id"] for entry in anchor["transcription"]["entries"])
    reported = {entry["id"] for entry in artifact["quote_recheck"]["entries"]}
    assert reported == set(anchor_ids)

    summary = artifact["quote_recheck"]["summary"]
    assert summary["withheld"] == len(anchor["transcription"]["summary"]["withheld_ids"])
    assert summary["transcribed"] == len(anchor["transcription"]["summary"]["transcribed_ids"])
    assert not summary["fail"]
    assert summary["absent_segments"] == 0


def test_coefficient_table_carries_a_printed_source_for_every_value(artifact):
    rows = artifact["coefficient_one_to_one"]
    assert {row["id"] for row in rows} >= {"U-gammaS", "U-gammaW", "U-gammaDB",
                                           "U-fbeta", "W-0.19", "W-fnlvs", "W-Cw"}
    for row in rows:
        assert row["verdict"] in ("PASS", "PASS_CROSS_DOCUMENT"), row["id"]
        if row["verdict"] == "PASS_CROSS_DOCUMENT":
            assert row["cross_document_evidence"]["verdict"] == "PASS"
            assert any(token["text_layer_garbled"] for token in row["tokens"])


def test_table2_reconstruction_reading_is_recorded(artifact):
    table2 = artifact["criteria"]["W26-C2"]["table2_coordinate_recheck"]
    assert table2["verdict"] == "PASS"
    assert [row["scenario"] for row in table2["rows"]] == ["A", "B"]
    assert [row["printed_right"] for row in table2["rows"]] == ["1.2", "1.3"]
    assert all(row["gamma_S_matches"] and row["gamma_W_matches"]
               for row in table2["rows"])


def test_smith_baseline_is_recomputed_and_matches_the_stored_capacity(artifact):
    smith = artifact["smith_capacity_baseline"]
    assert smith["recomputation_verdict"] == "PASS"
    assert smith["cases"] == len(smith["cases_table"])
    for row in smith["recomputation"]:
        assert row["relative_difference"] < 1e-9
        assert row["verdict"] == "PASS"
    for row in smith["cases_table"]:
        assert row["M_u_smith_MNm"] > 0 and row["M_y_elastic_MNm"] > 0
        assert row["M_u_over_M_y"] == pytest.approx(row["M_u_smith_MNm"]
                                                    / row["M_y_elastic_MNm"])
        assert 0.0 < row["utilisation"] and row["verdict"] in ("PASS", "FAIL")


def test_error_judgement_answers_the_three_percent_question(artifact):
    judgements = artifact["error_judgement"]
    assert len(judgements) >= 4
    for item in judgements:
        assert item["under_3_percent"] == (abs(item["relative_difference_percent"]) < 3.0)
