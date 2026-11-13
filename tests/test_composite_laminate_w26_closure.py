"""Consistency tests for the W26 composite-laminate verification closure ledger.

The ledger (``papers/artifacts/w26/composite-laminate-verification/closure.json``)
is the artifact that closes the experiment.  These tests re-derive every
criterion verdict from the recorded per-case numbers and check the sha256
provenance, so a hand-edited or silently inconsistent ledger fails the suite
instead of passing review.

Skipped when the artifacts are absent (they are produced by
``benchmarks/composite_laminate_w26/run_closure.py``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parents[1] / "papers" / "artifacts" / "w26" / \
    "composite-laminate-verification"

pytestmark = pytest.mark.skipif(
    not (ART / "closure.json").exists(),
    reason="W26 closure artifacts not generated in this checkout",
)


def _load(name):
    return json.loads((ART / name).read_text())


@pytest.fixture(scope="module")
def closure():
    return _load("closure.json")


def test_status_is_derived_not_declared(closure):
    verdicts = [v["verdict"] for v in closure["criteria"].values()]
    if all(v == "PASS" for v in verdicts):
        assert closure["status"] == "PASS"
    elif any(v == "PASS" for v in verdicts):
        assert closure["status"] == "PARTIAL"
    else:
        assert closure["status"] == "FAIL"


def test_deflection_verdict_matches_rows(closure):
    crit = closure["criteria"]["C2_deflection_vs_anchor"]
    rows = crit["rows"]
    assert rows, "no case rows"
    for r in rows:
        assert r["pass_plate"] == (abs(r["err_plate_pct"]) < r["limit_pct"])
        assert r["pass_hex20"] == (abs(r["err_hex20_pct"]) < r["limit_pct"])
    expected = "PASS" if all(r["pass_plate"] and r["pass_hex20"] for r in rows) else "FAIL"
    assert crit["verdict"] == expected
    assert crit["worst_plate_pct"] == pytest.approx(
        max(abs(r["err_plate_pct"]) for r in rows))


def test_grid_verdict_recomputes_from_gci_and_order(closure):
    crit = closure["criteria"]["C3_grid_convergence"]
    assert crit["rows"]
    for r in crit["rows"]:
        # the criterion is on the GCI magnitude and needs a positive observed order
        expected = "PASS" if (abs(r["gci_fine_pct"]) < 5.0 and r["p_observed"] > 0.0) else "FAIL"
        assert r["gci_verdict"] == expected
        assert r["gci_abs_pct"] == pytest.approx(abs(r["gci_fine_pct"]))
    gci_all = all(r["gci_verdict"] == "PASS" for r in crit["rows"])
    order_all = all(r["order_verdict"] == "PASS" for r in crit["rows"])
    assert crit["gci_all_pass"] == gci_all
    assert crit["order_all_pass"] == order_all
    assert crit["verdict"] == ("PASS" if (gci_all and order_all) else "FAIL")


def test_diagnostics_are_not_scored(closure):
    diag = [k for k, v in closure["criteria"].items() if v["verdict"] == "DIAGNOSTIC"]
    assert "D1_failure_diagnosis" in diag
    # the overall status is derived from the scored criteria only
    scored = [v["verdict"] for k, v in closure["criteria"].items() if k not in diag]
    assert closure["status"] != "PASS" or all(v == "PASS" for v in scored)


def test_shear_verdict_only_scored_where_an_anchor_exists(closure):
    crit = closure["criteria"]["C4_interlaminar_shear"]
    for r in crit["rows"]:
        if r["reference"] is None:
            assert r["verdict"] == "WITHHELD"
        else:
            assert r["verdict"] in ("PASS", "FAIL")
            assert r["verdict"] == ("PASS" if r["worst_abs_rel_pct"] < 10.0 else "FAIL")


def test_cook_membrane_anchor(closure):
    crit = closure["criteria"]["C5_cook_membrane_shear_anchor"]
    cook = _load("cook_membrane.json")
    finest = cook["rows"][-1]
    assert crit["finest"]["nx"] == finest["nx"]
    assert finest["abs_rel_err_pct"] < 3.0
    assert crit["verdict"] == "PASS"
    # Richardson extrapolated value must sit closer to the anchor than the raw solve
    assert abs(cook["richardson"]["extrapolated"] - cook["reference"]) < abs(
        finest["tip_deflection"] - cook["reference"])


def test_artifact_hashes_match_files(closure):
    for name, meta in closure["artifacts"].items():
        p = ART / name
        assert p.exists(), name
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == meta["sha256"], name
        assert len(digest) == 64


def test_every_case_reports_both_paths(closure):
    for row in closure["case_table"]:
        assert row["err_plate_pct"] != ""
        assert row["err_hex_pct"] != ""
    assert len({row["layup"] for row in closure["case_table"]}) >= 3
    assert len({round(float(row["t_over_a"]), 6) for row in closure["case_table"]}) >= 3