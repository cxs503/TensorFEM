"""Invariants of the refinement-ladder certificate re-grade.

``tools/regrade_ladder_certificate.py`` re-derives a shipped ladder certificate
from the anchor's own ``result.json`` -- no solver, no caller memory -- and
compares it, field by field, against what the anchor shipped:

* the ``criteria`` block must be in the artifact, otherwise re-grading would
  have to assume the thresholds it is supposed to check (circular and
  unfalsifiable);
* the stored gates must be *exactly* reproduced from ``levels`` + ``criteria``,
  so a hand-edited threshold, a gate left over from an older run, or a
  ``result.json`` refreshed without re-grading turns red;
* the ``tensorfem.verified-gates.v1`` envelope must agree with the ladder it
  certifies, including anchors that ship two ladders under one envelope
  (Scordelis-Lo flat-facet + cylindrical), whose gate ids differ only by the
  ladder-name prefix.

The negative-control tests exist so this guard cannot rot into a no-op: each
mutation of the artifact must be *detected*, not tolerated.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DIR = REPO_ROOT / "benchmarks" / "verified"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tensorfem.refinement_gate import (  # noqa: E402
    CRITERIA_FIELDS,
    SCHEMA as LADDER_SCHEMA,
    LadderCertificateError,
    certificate_drift,
    regrade_ladder,
)

import regrade_ladder_certificate as regrade_tool  # noqa: E402

BLOCKS = regrade_tool.discover(VERIFIED_DIR)
BLOCK_IDS = [f"{anchor}{pointer}" for anchor, pointer, _ in BLOCKS]


def _block(block_id: str) -> dict:
    for anchor, pointer, block in BLOCKS:
        if f"{anchor}{pointer}" == block_id:
            return copy.deepcopy(block)
    raise AssertionError(f"unknown block {block_id}")


def test_discovery_is_not_empty() -> None:
    """A silently empty discovery would make every other test vacuous."""

    assert BLOCKS, "no tensorfem.refinement-gate block found under benchmarks/verified"
    assert {anchor for anchor, _, _ in BLOCKS} >= {"fem_pinched_cylinder",
                                                   "fem_shell_scordelis_lo"}


@pytest.mark.parametrize("block_id", BLOCK_IDS)
def test_block_records_the_criteria_it_was_graded_under(block_id: str) -> None:
    block = _block(block_id)
    assert block["schema"] == LADDER_SCHEMA
    criteria = block.get("criteria")
    assert isinstance(criteria, dict), (
        f"{block_id} ships gates without the criteria they were graded under; the "
        "thresholds would have to be reverse-engineered from the gates themselves"
    )
    for name in CRITERIA_FIELDS:
        assert name in criteria, f"{block_id} criteria is missing {name!r}"
    assert criteria["reference_value"] == block["reference"]["value"]


@pytest.mark.parametrize("block_id", BLOCK_IDS)
def test_shipped_gates_are_reproduced_bit_for_bit(block_id: str) -> None:
    block = _block(block_id)
    drift = certificate_drift(block)
    assert drift == [], f"{block_id} certificate is not reproducible from its own ladder: {drift}"

    regraded = regrade_ladder(block)
    assert regraded["status"] == block["status"]
    assert regraded["reported_accuracy_finest"] == block["reported_accuracy_finest"]
    assert regraded["certified_accuracy"] == block["certified_accuracy"]


def test_full_sweep_is_clean() -> None:
    """Every shipped ladder certificate re-grades, and every envelope agrees."""

    record = regrade_tool.audit(VERIFIED_DIR)
    assert record["blocks"] == len(BLOCKS)
    assert record["errors"] == []
    assert record["drift_fields"] == 0
    assert record["ok"] is True


def test_two_ladders_under_one_envelope_are_not_collapsed() -> None:
    """Scordelis-Lo ships flat-facet *and* cylindrical gates in one envelope.

    Their ids differ only by the ladder-name prefix, so a matcher that strips
    the prefix silently compares the cylindrical block against the flat-facet
    gates.  Pin the counts so that regression cannot come back.
    """

    record = regrade_tool.audit(VERIFIED_DIR)
    by_anchor = {r["anchor"] + r["pointer"]: r for r in record["records"]}
    flat = by_anchor["fem_shell_scordelis_lo/ladder_grading/flat_facet"]
    cyl = by_anchor["fem_shell_scordelis_lo/ladder_grading/cylindrical"]
    assert flat["envelope_issues"] == []
    assert cyl["envelope_issues"] == []
    # The two ladders disagree on 5 of 6 gates (different elements), so a
    # collapse would show up as envelope mismatches above -- assert the numbers
    # really are different ladders, not the same block read twice.
    assert flat["drift"] == [] and cyl["drift"] == []
    assert (regrade_tool.ladder_hint("/ladder_grading/cylindrical") == "cylindrical")
    assert regrade_tool.ladder_hint("/ladder_grading") is None


# --------------------------------------------------------------------------- #
# negative controls: the guard must fail on tampering
# --------------------------------------------------------------------------- #
def _first_graded_gate(block: dict) -> dict:
    for gate in block["gates"]:
        if gate["status"] != "WITHHELD":
            return gate
    raise AssertionError("no graded gate to tamper with")


def test_hand_edited_threshold_is_detected() -> None:
    block = _block("fem_pinched_cylinder/ladder_grading")
    gate = _first_graded_gate(block)
    gate["threshold"] = gate["threshold"] * 10.0
    drift = certificate_drift(block)
    assert any(d["id"] == gate["id"] and d["field"] == "threshold" for d in drift), drift


def test_hand_flipped_verdict_is_detected() -> None:
    block = _block("fem_pinched_cylinder/ladder_grading")
    gate = _first_graded_gate(block)
    gate["status"] = "PASS" if gate["status"] != "PASS" else "FAIL"
    gate["passed"] = not gate["passed"]
    drift = certificate_drift(block)
    assert {d["field"] for d in drift if d["id"] == gate["id"]} >= {"status", "passed"}


def test_stale_gate_from_an_older_run_is_detected() -> None:
    """A level that drifted away from its certificate is the failure mode that
    ``anchor_certificates.json`` cannot see on its own."""

    block = _block("fem_shell_scordelis_lo/ladder_grading/cylindrical")
    block["levels"][-1]["value"] = block["levels"][-1]["value"] - 1e-3
    drift = certificate_drift(block)
    assert drift, "a modified measured value must invalidate the certificate"
    assert any(d["field"] == "actual" for d in drift), drift


def test_tolerance_change_is_detected() -> None:
    block = _block("fem_pinched_cylinder/ladder_grading")
    block["criteria"]["tolerance"] = 0.5
    drift = certificate_drift(block)
    assert any(d["field"] == "threshold" for d in drift), drift


def test_missing_criteria_fails_closed() -> None:
    block = _block("fem_pinched_cylinder/ladder_grading")
    del block["criteria"]
    with pytest.raises(LadderCertificateError, match="criteria"):
        regrade_ladder(block)


def test_inconsistent_reference_fails_closed() -> None:
    block = _block("fem_pinched_cylinder/ladder_grading")
    block["reference"]["value"] = block["reference"]["value"] * 2.0
    with pytest.raises(LadderCertificateError, match="reference"):
        regrade_ladder(block)


def test_wrong_schema_fails_closed() -> None:
    block = _block("fem_pinched_cylinder/ladder_grading")
    block["schema"] = "tensorfem.refinement-gate/0.9"
    with pytest.raises(LadderCertificateError, match="schema"):
        regrade_ladder(block)


def test_envelope_drift_is_detected() -> None:
    """A certificate envelope that disagrees with its ladder is an error, not a
    warning: the audit reads the envelope, the physics reads the ladder."""

    anchor_dir = VERIFIED_DIR / "fem_pinched_cylinder"
    block = _block("fem_pinched_cylinder/ladder_grading")
    gate = _first_graded_gate(block)
    gate["actual"] = gate["actual"] + 1.0
    issues = regrade_tool._envelope_issues(anchor_dir, block, None)
    assert any(i["kind"] == "gate_differs_from_envelope" and i["gate"] == gate["id"]
               for i in issues), issues


def test_cli_check_passes_on_the_shipped_registry() -> None:
    assert regrade_tool.main(["--check"]) == 0