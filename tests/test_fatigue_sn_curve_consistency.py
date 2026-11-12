"""Cross-module S-N curve consistency -- fail-closed regression tests.

Anchors: DNV-RP-C203 Table 2-1 curve D in air (``log a1 = 12.164``, ``m1 = 3``,
knee at ``1e7``; ``log a2 = 15.606``, ``m2 = 5``) and IIW FAT 90.  See
``docs/fatigue-sn-curve-consistency-audit.md``.
"""
from __future__ import annotations

import math
import pathlib

import pytest

from tensorfem.fatigue.sn_curve_consistency import (
    DNV_D_AIR,
    audit,
    curve_rows,
    reference_life,
)
from tensorfem.structural_fatigue_detail import DNVDualSlopeSN, sn_curve_verification_report


def test_dual_slope_report_passes_with_branch_guard() -> None:
    report = sn_curve_verification_report()
    assert report["branch_guard"] is True
    assert report["ratio_m3_error_percent"] < 1e-6
    assert report["ratio_m5_error_percent"] < 1e-6
    assert report["knee_error_percent"] < 1e-6
    assert report["passed"] is True
    # both m=2 probes must sit in the branch they claim to test
    knee = report["stress_knee_pa"]
    assert report["ds1_pa"] > knee and report["ds2_pa"] > knee
    assert report["ds3_pa"] < knee and report["ds4_pa"] < knee


def test_first_release_m5_protocol_straddled_the_knee() -> None:
    """The historical 100/50 MPa protocol: documents why it produced 72.3 % error."""
    curve = DNVDualSlopeSN()
    knee = curve.stress_knee_pa
    assert 100e6 > knee and 50e6 < knee          # the two probes sit in *different* branches
    ratio = curve.cycles_to_failure(50e6) / curve.cycles_to_failure(100e6)
    assert ratio == pytest.approx(8.85421326531684, rel=1e-9)
    assert abs(ratio - 32.0) / 32.0 > 0.7
    # the branch-guarded protocol on the same curve recovers the exact slope
    assert curve.cycles_to_failure(knee * 0.225) / curve.cycles_to_failure(knee * 0.45) == pytest.approx(32.0, rel=1e-12)


def test_docstring_knee_stress_arithmetic() -> None:
    """10**((12.164-7)/3) = 52.64 MPa, not the 169.3 MPa printed in v1."""
    assert 10.0 ** ((12.164 - 7.0) / 3.0) == pytest.approx(52.642115, rel=1e-6)
    assert DNVDualSlopeSN().stress_knee_pa / 1e6 == pytest.approx(52.642115, rel=1e-6)


def test_cross_module_audit_passes() -> None:
    report = audit()
    summary = report["summary"]
    assert summary["n_implementations"] >= 6
    assert summary["n_dnv_curve_d_air"] >= 4
    assert summary["max_knee_continuity_rel"] <= 5e-3
    assert summary["max_abs_delta_log_a1_vs_dnv_d"] <= 1e-6
    assert summary["max_declared_vs_callable_rel"] <= 1e-12
    assert summary["passed"] is True


def test_every_curve_matches_its_declared_table_at_100mpa() -> None:
    for row in curve_rows():
        declared = reference_life(
            100e6, log_a1=row.log_a1, m1=row.m1, log_a2=row.log_a2,
            m2=row.m2, knee_cycles=row.knee_cycles,
        )
        assert row.life(100e6) == pytest.approx(declared, rel=1e-12), row.name


def test_slope_identity_in_each_branch() -> None:
    for row in curve_rows():
        knee = row.knee_stress_mpa * 1e6
        above = row.life(1.4 * knee) / row.life(2.8 * knee)
        # probes below the knee but above the IIW 1e9-cycle cutoff
        below = row.life(0.45 * knee) / row.life(0.9 * knee)
        assert above == pytest.approx(2.0 ** row.m1, rel=1e-10), row.name
        assert below == pytest.approx(2.0 ** row.m2, rel=1e-10), row.name


def test_ice_slamming_matches_dnv_curve_d_air() -> None:
    """Regression: v1 declared C1 = 1.52e12 (log a1 = 12.182, +4.19 % life)."""
    from tensorfem.ice_slamming_whipping import SNCurve

    curve = SNCurve()
    assert math.log10(curve.c1) == pytest.approx(float(DNV_D_AIR["log_a1"]), abs=1e-9)
    assert math.log10(curve.c2) == pytest.approx(float(DNV_D_AIR["log_a2"]), abs=1e-3)
    assert curve.knee_stress / 1e6 == pytest.approx(52.642115, rel=1e-6)
    assert curve.cycles_to_failure(100e6) == pytest.approx(reference_life(100e6), rel=1e-9)


def test_fatigue_life_gate_is_not_a_hardcoded_pass() -> None:
    """The fatigue-life criterion must be computed, not asserted True."""
    source = pathlib.Path("src/tensorfem/structural_fatigue_detail.py").read_text()
    assert "computed correctly if S-N curve is valid" not in source
    assert "life_relative_error" in source
    assert "_knee_anchored_life" in source
