"""Regression gate for the thermal verification benchmark."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "thermal_verification_evidence.json"


@pytest.fixture(scope="module")
def evidence():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_thermal_evidence_file_exists():
    assert EVIDENCE.is_file(), "thermal_verification_evidence.json must exist"


def test_thermal_all_anchors_pass(evidence):
    assert evidence["all_passed"], "all thermal verification anchors must pass"
    for anchor in evidence["anchors"]:
        assert anchor["passed"], f"anchor {anchor['anchor']} failed"


def test_thermal_schema_version(evidence):
    assert evidence["schema"] == "tensorfem.thermal-verification/1.0"


def test_thermal_patch_test_machine_zero(evidence):
    """Patch tests must achieve machine-zero error."""
    for anchor in evidence["anchors"]:
        if "patch" in anchor["anchor"].lower() or "prescribed_temperature" in anchor["anchor"]:
            assert anchor["max_absolute_error"] < 1e-10, \
                f"patch test {anchor['anchor']} should be machine-zero"


def test_thermal_energy_balance_machine_zero(evidence):
    """Energy balance must be conserved to machine precision."""
    for anchor in evidence["anchors"]:
        if "energy_balance" in anchor["anchor"]:
            assert anchor["relative_error"] < 1e-10, \
                f"energy balance error {anchor['relative_error']:.2e} exceeds 1e-10"


def test_thermal_transient_below_three_percent(evidence):
    """Transient decay must be within 3% of analytical solution."""
    for anchor in evidence["anchors"]:
        if "transient" in anchor["anchor"]:
            assert anchor["max_relative_error"] < 0.03, \
                f"transient error {anchor['max_relative_error']:.2e} exceeds 3%"


def test_thermal_report_exists():
    report = ROOT / "docs" / "thermal-verification.md"
    assert report.is_file(), "thermal-verification.md must exist"
    text = report.read_text(encoding="utf-8")
    assert "真实计算结果" not in text or "Computed Results" in text
    assert "相对误差" not in text or "relative error" in text.lower() or "Relative error" in text
    assert "✅ ALL PASSED" in text
