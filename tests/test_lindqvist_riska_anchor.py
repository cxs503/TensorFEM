"""Tests for the Lindqvist/Riska ice resistance anchor reconciliation script."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Paths relative to this test file
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
ANCHOR_DIR = REPO_ROOT / "benchmarks" / "anchors"
SCRIPT_PATH = ANCHOR_DIR / "lindqvist_riska_anchor.py"
ARTIFACT_PATH = ANCHOR_DIR / "lindqvist-riska-ice-resistance.json"
TRANSCRIPTION_PATH = ANCHOR_DIR / "sources" / "lindqvist_riska_ice_resistance.transcription.json"


@pytest.fixture(scope="module")
def artifact():
    """Run the anchor script and return the artifact."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    with open(ARTIFACT_PATH) as f:
        return json.load(f)


class TestDimensionalHomogeneity:
    def test_all_steps_homogeneous(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK1_dimensional_homogeneity")
        assert chk["passed"], f"Dimensional homogeneity failed: {chk}"
        assert chk["n_homogeneous"] == chk["n_steps"]

    def test_rc_dimension(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK1_dimensional_homogeneity")
        rc_step = next(s for s in chk["steps"] if s["step"] == "S1_Rc")
        assert rc_step["homogeneous"]
        assert rc_step["lhs"] == rc_step["rhs"]

    def test_lambda_dimension(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK1_dimensional_homogeneity")
        lam_step = next(s for s in chk["steps"] if s["step"] == "S2_lambda")
        assert lam_step["homogeneous"]
        assert lam_step["lhs"] == "m^-1"
        assert lam_step["rhs"] == "m^-1"


class TestScalingExponents:
    def test_crushing_exponent(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK2_scaling_exponents")
        assert chk["thickness_scaling"]["Rc_passed"]
        assert abs(chk["thickness_scaling"]["Rc_exponent_measured"] - 2.0) < 1e-6

    def test_bending_exponent(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK2_scaling_exponents")
        assert chk["thickness_scaling"]["Rb_passed"]
        assert abs(chk["thickness_scaling"]["Rb_exponent_measured"] - 1.25) < 1e-4

    def test_momentum_thickness_exponent(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK2_scaling_exponents")
        assert chk["thickness_scaling"]["Rmom_passed"]
        assert abs(chk["thickness_scaling"]["Rmom_exponent_measured"] - 1.0) < 1e-6

    def test_speed_squared_scaling(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK2_scaling_exponents")
        assert chk["speed_scaling"]["delta_passed"]
        assert chk["speed_scaling"]["Rmom_passed"]


class TestZeroThicknessLimit:
    def test_crushing_vanishes(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK3_zero_thickness_limit")
        assert chk["Rc_tends_to_zero"]
        assert chk["Rc_N"] < 1.0

    def test_bending_vanishes(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK3_zero_thickness_limit")
        assert chk["Rb_tends_to_zero"]
        assert chk["Rb_N"] < 1.0

    def test_momentum_vanishes(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK3_zero_thickness_limit")
        assert chk["Rmom_tends_to_zero"]
        assert chk["Rmom_N"] < 1.0


class TestSpeedDecomposition:
    def test_quadratic_speed_scaling(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK4_speed_decomposition")
        assert chk["passed"]
        assert abs(chk["ratio_dR2_dR1"] - 4.0) < 0.01


class TestUnitInvariance:
    def test_all_components_invariant(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK5_unit_invariance")
        assert chk["passed"]
        assert chk["Rc_relative_error"] < 1e-10
        assert chk["Rb_relative_error"] < 1e-10
        assert chk["Rs_relative_error"] < 1e-10
        assert chk["Rmom_relative_error"] < 1e-10


class TestThicknessSweep:
    def test_monotonic_increase(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK6_thickness_sweep")
        assert chk["monotonic"]

    def test_effective_exponent_in_range(self, artifact):
        chk = next(c for c in artifact["machine_checks"] if c["id"] == "CHK6_thickness_sweep")
        assert chk["exponent_in_range"]
        assert 1.0 < chk["effective_exponent_at_h1m"] < 2.0


class TestTranscriptionGate:
    def test_no_eligible_entries_without_source(self, artifact):
        assert artifact["source_registry"]["claim_eligible_entries"] == 0

    def test_verdict_withheld(self, artifact):
        assert artifact["verdict"]["transcription_gate"] == "WITHHELD"
        assert artifact["verdict"]["overall"] == "WITHHELD"

    def test_m1_threshold_not_met(self, artifact):
        assert not artifact["verdict"]["m1_threshold"]["met"]
        assert artifact["verdict"]["m1_threshold"]["current_verbatim_entries"] == 0
        assert artifact["verdict"]["m1_threshold"]["required_verbatim_entries"] == 4


class TestArtifactStructure:
    def test_has_schema_version(self, artifact):
        assert "schema_version" in artifact

    def test_has_task_key(self, artifact):
        assert artifact["task_key"] == "GAP-ICE-LIT-TRANSCRIPT"

    def test_has_case_caliber(self, artifact):
        assert "case_caliber" in artifact
        assert "quantities" in artifact["case_caliber"]

    def test_has_source_registry(self, artifact):
        assert "source_registry" in artifact
        assert "sources_sought" in artifact["source_registry"]

    def test_has_machine_checks(self, artifact):
        assert "machine_checks" in artifact
        assert len(artifact["machine_checks"]) == 6

    def test_has_dry_run_fixture(self, artifact):
        assert "dry_run_fixture" in artifact
        assert artifact["dry_run_fixture"]["claim_eligible"] is False

    def test_has_verdict(self, artifact):
        assert "verdict" in artifact
        assert "m1_threshold" in artifact["verdict"]


class TestTranscriptionRegistry:
    def test_registry_exists(self):
        assert TRANSCRIPTION_PATH.exists()

    def test_registry_has_sources(self):
        with open(TRANSCRIPTION_PATH) as f:
            registry = json.load(f)
        assert "sources_sought" in registry
        assert len(registry["sources_sought"]) >= 2

    def test_registry_has_entries(self):
        with open(TRANSCRIPTION_PATH) as f:
            registry = json.load(f)
        assert "entries" in registry
        assert len(registry["entries"]) >= 4

    def test_all_entries_withheld(self):
        with open(TRANSCRIPTION_PATH) as f:
            registry = json.load(f)
        for entry in registry["entries"]:
            assert entry["verbatim_status"] == "WITHHELD"

    def test_m1_threshold_documented(self):
        with open(TRANSCRIPTION_PATH) as f:
            registry = json.load(f)
        assert "m1_threshold" in registry
        assert registry["m1_threshold"]["required_verbatim_entries"] == 4
        assert registry["m1_threshold"]["current_verbatim_count"] == 0
