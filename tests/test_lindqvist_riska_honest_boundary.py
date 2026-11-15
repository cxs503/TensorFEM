"""Tests for the GAP-ICE-LIT-TRANSCRIPT honest-boundary cross-check.

These tests assert the *honest* boundary of the ice-resistance literature
transcription anchor:

1. The platform's calibrated ``calc_lindqvist`` is a **3-component calibrated
   variant** (submersion + bending + clearing, Fn_ice dependent), structurally
   distinct from the original Lindqvist **4-component** model (crushing +
   bending + submersion + momentum). It must NOT be cited as the original
   verbatim source.
2. The ``GAP-ICE-LIT-TRANSCRIPT`` M1 threshold (>=4 verbatim triples) is
   **NOT met** — no verbatim entry has been fabricated, and the coefficient
   grade stays ``WITHHELD`` (never upgraded to ``REAL`` on the basis of the
   platform's calibrated variant).
3. The source-acquisition readiness artifact is **complete**: every one of the
   6 transcription entries carries a precise closure spec (locator, quote,
   values, verification procedure) so a future agent with an authorised copy
   of the source can close the gap.

These are *honest* tests: they verify that the machinery does not over-claim,
not that it fabricates a source value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ANCHOR_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "anchors"
CROSSCHECK_PATH = ANCHOR_DIR / "lindqvist-riska-calibrated-crosscheck.json"
TRANSCRIPTION_PATH = ANCHOR_DIR / "sources" / "lindqvist_riska_ice_resistance.transcription.json"


@pytest.fixture(scope="module")
def crosscheck():
    return json.loads(CROSSCHECK_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def transcription():
    return json.loads(TRANSCRIPTION_PATH.read_text(encoding="utf-8"))


def test_artifact_exists(crosscheck):
    assert crosscheck["task_key"] == "GAP-ICE-LIT-TRANSCRIPT"
    assert crosscheck["schema_version"] == "1.0.0"


def test_platform_is_variant_not_original(crosscheck):
    """The platform's calibrated Lindqvist must be a 3-component variant,
    structurally distinct from the original 4-component model."""
    sig = crosscheck["structural_signature"]
    assert sig["structural_discrepancy"]["decomposition_mismatch"] is True
    assert sig["structural_discrepancy"]["component_count_original"] == 4
    assert sig["structural_discrepancy"]["component_count_platform"] == 3
    # platform components are submersion/bending/clearing, not crushing/momentum
    plat_components = sig["platform_calibrated_variant"]["components"]
    assert "R_crushing" not in plat_components
    assert "R_momentum" not in plat_components


def test_m1_threshold_still_unmet(crosscheck):
    """No verbatim entry has been fabricated; M1 (>=4 verbatim triples) is
    still unmet, so the coefficient grade must stay WITHHELD."""
    assert crosscheck["verdict"]["m1_threshold_met"] is False
    assert crosscheck["verdict"]["coefficient_grade"] == "WITHHELD"
    assert crosscheck["verdict"]["transcription_gate"] == "WITHHELD"
    assert crosscheck["verdict"]["overall"] == "WITHHELD"


def test_no_false_real_upgrade(crosscheck):
    """The honest-FAIL verdict must be explicit: the platform's calibrated
    variant does NOT upgrade any coefficient to REAL."""
    v = crosscheck["verdict"]
    assert "honest_fail" in v
    assert "NOT upgraded to REAL" in v["honest_fail"]


def test_readiness_artifact_complete(crosscheck):
    """Every one of the 6 transcription entries carries a precise closure
    spec (locator, quote, values, verification)."""
    ready = crosscheck["readiness_artifact"]
    assert ready["m1_threshold"]["required_verbatim_entries"] == 4
    assert ready["m1_threshold"]["current_verbatim_entries"] == 0
    entries = ready["entries_ready"]
    assert len(entries) == 6
    for e in entries:
        assert e["needed_locator"], f"{e['anchor_id']} missing locator"
        assert e["needed_quote"], f"{e['anchor_id']} missing quote spec"
        assert e["needed_values"] is not None, f"{e['anchor_id']} missing values"
        assert e["verification"], f"{e['anchor_id']} missing verification"


def test_transcription_registry_consistent_with_crosscheck(transcription, crosscheck):
    """The transcription registry's entries must agree with the cross-check's
    honest verdict: all 6 entries are WITHHELD (no verbatim source)."""
    entries = transcription["entries"]
    assert len(entries) == 6
    # every entry is WITHHELD — no verbatim source obtained
    for e in entries:
        assert e["verbatim_status"] == "WITHHELD", f"{e['anchor_id']} not WITHHELD"
    # M1 threshold in the registry agrees with the cross-check
    m1 = transcription["m1_threshold"]
    assert m1["current_verbatim_count"] == 0
    assert m1["required_verbatim_entries"] == 4
    assert m1["status"].startswith("WITHHELD")


def test_honest_boundary_checks_all_pass(crosscheck):
    """All three honest-boundary machine checks must PASS."""
    for c in crosscheck["honest_boundary_checks"]:
        assert c["result"] == "PASS", f"{c['id']} did not pass: {c['note']}"
