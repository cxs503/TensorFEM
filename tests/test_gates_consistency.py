"""Checks for the verified-gate corpus consistency auditor.

The auditor exists because two merged anchors stored gate files that certify a
PASS the file itself contradicts.  These tests pin the detector on synthetic
inputs (so the rules cannot silently rot) and then enforce a non-regression
budget on the real corpus.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "gates_consistency_audit.py"
BASELINE = REPO / "docs" / "assets" / "gates_consistency" / "gates_consistency.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("gates_consistency_audit", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def _write(tmp_path: Path, name: str, payload) -> Path:
    case = tmp_path / name
    case.mkdir()
    path = case / "gates_verified.json"
    path.write_text(json.dumps(payload))
    return path


def test_detects_pass_with_failed_gate(tool, tmp_path):
    path = _write(
        tmp_path,
        "case_a",
        {
            "gates": [
                {"id": "G1_accuracy", "threshold": 0.03, "actual": 0.01, "passed": True},
                {"id": "G2_monotone_convergence", "threshold": "monotone", "actual": "non-monotone", "passed": False},
            ],
            "overall_verdict": "PASS",
        },
    )
    rules = {f["rule"] for f in tool.audit_file(path)}
    assert "pass_with_failed_gate" in rules


def test_clean_file_yields_no_findings(tool, tmp_path):
    path = _write(
        tmp_path,
        "case_b",
        {
            "gates": [
                {"id": "G1_accuracy", "threshold": 0.03, "actual": 0.01, "passed": True},
                {"id": "G2_monotone_convergence", "threshold": "monotone", "actual": "monotone", "passed": True},
                {"id": "G3_order", "description": "observed order >= 1.0", "threshold": 1.0, "actual": 2.1, "passed": True},
            ],
            "overall_verdict": "PASS",
        },
    )
    assert tool.audit_file(path) == []


def test_lower_bound_gate_direction(tool, tmp_path):
    """An order gate is a minimum gate: actual > threshold must not be flagged."""
    path = _write(
        tmp_path,
        "case_c",
        {
            "gates": [
                {"id": "G3_convergence_order", "description": "observed order >= 1.0", "threshold": 1.0,
                 "actual": 4.3, "passed": True},
            ],
            "overall_verdict": "PASS",
        },
    )
    assert {f["rule"] for f in tool.audit_file(path)} == set()


def test_order_gate_on_non_monotone_sequence(tool, tmp_path):
    path = _write(
        tmp_path,
        "case_d",
        {
            "gates": [
                {"id": "G2_monotone_convergence", "threshold": "monotone", "actual": "non-monotone", "passed": False},
                {"id": "G3_convergence_order", "description": "observed order >= 1.0", "threshold": 1.0,
                 "actual": 4.3, "passed": True},
            ],
            "overall_verdict": {"verdict": "FAIL"},
        },
    )
    assert "order_gate_despite_nonmonotone" in {f["rule"] for f in tool.audit_file(path)}


def test_platform_tolerance_limit(tool, tmp_path):
    path = _write(
        tmp_path,
        "case_e",
        {
            "gates": [
                {"id": "G1_accuracy_error", "description": "error within 20%", "threshold": 0.20,
                 "actual": 0.18, "passed": True},
            ],
            "overall_verdict": "PASS",
        },
    )
    assert "tolerance_above_platform_limit" in {f["rule"] for f in tool.audit_file(path)}


def test_corpus_does_not_regress(tool):
    """The live corpus may not accumulate new inconsistencies."""
    report = tool.audit_corpus()
    if not BASELINE.exists():
        pytest.skip("baseline not generated (run tools/gates_consistency_audit.py --write)")
    baseline = json.loads(BASELINE.read_text())
    assert report["files_scanned"] >= baseline["files_scanned"] - 2
    assert report["total_findings"] <= baseline["total_findings"], (
        "new gate inconsistency introduced: "
        + json.dumps(
            [f for f in report["findings"] if f not in baseline["findings"]], indent=2
        )
    )