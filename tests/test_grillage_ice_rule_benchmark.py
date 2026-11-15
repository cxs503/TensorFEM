"""Tests for the IACS Polar Class rule-based benchmark for the grillage module.

Verification anchors T14a–T14d: rule plate thickness, safety margin,
monotonicity, and stiffener modulus consistency.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    MemberSection,
    case_study_geometry,
    ultimate_ice_pressure,
)
from tensorfem.grillage_ice_rule_benchmark import (
    IACSPolarClassRule,
    rule_benchmark_table,
    rule_design_pressure,
    rule_frame_modulus,
    rule_plate_thickness,
    rule_stiffener_modulus,
    run_rule_benchmark,
    write_rule_benchmark,
)


# --------------------------------------------------------------------------- #
# unit tests for the rule formulas
# --------------------------------------------------------------------------- #
def test_plate_thickness_formula():
    """C_p * s * sqrt(p / sigma_y) matches hand calculation."""
    s, p, sigma_y = 0.35, 2.0e6, 355.0e6
    rule = IACSPolarClassRule()
    t = rule_plate_thickness(s, p, sigma_y, rule)
    expected = 0.5 * s * math.sqrt(p / sigma_y)
    assert t == pytest.approx(expected, rel=1e-12)


def test_plate_thickness_scales_correctly():
    """Double the pressure → thickness grows by sqrt(2)."""
    s, sigma_y = 0.35, 355.0e6
    t1 = rule_plate_thickness(s, 1.0e6, sigma_y)
    t2 = rule_plate_thickness(s, 2.0e6, sigma_y)
    assert t2 / t1 == pytest.approx(math.sqrt(2.0), rel=1e-10)


def test_stiffener_modulus_formula():
    """Z = p * s * c * (2l - c) / (8 * sigma_a) matches hand calculation."""
    p, s, l, c, sigma_y = 2.0e6, 0.35, 2.1, 0.70, 355.0e6
    rule = IACSPolarClassRule()
    z = rule_stiffener_modulus(p, s, l, c, sigma_y, rule)
    sigma_a = sigma_y / rule.gamma_m
    M = p * s * c * (2.0 * l - c) / 8.0
    expected = M / sigma_a
    assert z == pytest.approx(expected, rel=1e-12)


def test_frame_modulus_formula():
    """Z = p * A * l / (8 * sigma_a) matches hand calculation."""
    p, A, l, sigma_y = 2.0e6, 0.245, 2.8, 355.0e6
    rule = IACSPolarClassRule()
    z = rule_frame_modulus(p, A, l, sigma_y, rule)
    sigma_a = sigma_y / rule.gamma_m
    expected = p * A * l / (8.0 * sigma_a)
    assert z == pytest.approx(expected, rel=1e-12)


def test_design_pressure_back_calculation():
    """p_rule = sigma_y * (t_net / (C_p * s))^2 matches forward formula."""
    geom = case_study_geometry()
    rule = IACSPolarClassRule()
    p_rule = rule_design_pressure(geom, rule)
    t_net = geom.plate_thickness - rule.t_corr
    expected = geom.yield_stress * (t_net / (rule.c_plate * geom.bay_spacing)) ** 2
    assert p_rule == pytest.approx(expected, rel=1e-12)


def test_design_pressure_positive():
    """The rule design pressure for the case-study geometry is positive."""
    geom = case_study_geometry()
    p_rule = rule_design_pressure(geom)
    assert p_rule > 0


def test_rule_validation_rejects_bad_class():
    with pytest.raises(ValueError):
        IACSPolarClassRule(polar_class="PC8").validate()


# --------------------------------------------------------------------------- #
# T14 verification gates
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def benchmark_result():
    return run_rule_benchmark()


def test_benchmark_passes(benchmark_result):
    """All T14 checks pass."""
    failed = [c for c in benchmark_result["checks"] if not c["passed"]]
    assert benchmark_result["passed"], f"Failed checks: {failed}"


def test_t14a_plate_thickness_at_p_ult(benchmark_result):
    """T14a: rule plate thickness at p_ult matches actual within 25%."""
    t14a = [c for c in benchmark_result["checks"] if c["name"].startswith("T14a")]
    assert len(t14a) >= 4
    for check in t14a:
        assert check["passed"], check


def test_t14b_safety_margin_in_range(benchmark_result):
    """T14b: p_ult / p_rule is in [0.5, 2.0] for every combined case."""
    t14b = [c for c in benchmark_result["checks"] if c["name"].startswith("T14b")]
    assert len(t14b) >= 4
    for check in t14b:
        assert check["passed"], check
        assert 0.5 <= check["value"] <= 2.0, check


def test_t14c_margin_monotone(benchmark_result):
    """T14c: safety margin decreases with axial compression."""
    t14c = [c for c in benchmark_result["checks"] if c["name"].startswith("T14c")]
    assert len(t14c) >= 1
    for check in t14c:
        assert check["passed"], check


def test_t14d_stiffener_modulus_consistent(benchmark_result):
    """T14d: rule stiffener modulus at p_ult is finite and positive."""
    t14d = [c for c in benchmark_result["checks"] if c["name"].startswith("T14d")]
    assert len(t14d) >= 4
    for check in t14d:
        assert check["passed"], check


def test_benchmark_table_has_rows(benchmark_result):
    """The benchmark table has the expected number of rows."""
    table = benchmark_result["table"]
    assert len(table["rows"]) == 4 * 3  # 4 axial ratios × 3 hydrostatic pressures


def test_benchmark_table_schema(benchmark_result):
    """The benchmark table has the correct schema."""
    table = benchmark_result["table"]
    assert table["schema"] == "tensorfem.grillage_ice_rule_benchmark/v1"
    assert "geometry" in table
    assert "rule" in table


def test_write_rule_benchmark(tmp_path, benchmark_result):
    """The benchmark can be written to JSON and read back."""
    path = write_rule_benchmark(benchmark_result, tmp_path / "rule_benchmark.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_checks"] == benchmark_result["n_checks"]
    assert payload["passed"] == benchmark_result["passed"]


# --------------------------------------------------------------------------- #
# additional physical consistency checks
# --------------------------------------------------------------------------- #
def test_rule_plate_thickness_less_than_actual():
    """At the rule design pressure, the required thickness equals actual net."""
    geom = case_study_geometry()
    rule = IACSPolarClassRule()
    p_rule = rule_design_pressure(geom, rule)
    t_req = rule_plate_thickness(geom.bay_spacing, p_rule, geom.yield_stress, rule)
    t_net = geom.plate_thickness - rule.t_corr
    assert abs(t_req / t_net - 1.0) < 1e-12


def test_margin_reasonable_range():
    """The safety margin lies in a physically reasonable range (0.5–2.0)."""
    geom = case_study_geometry()
    rule = IACSPolarClassRule()
    p_rule = rule_design_pressure(geom, rule)
    case = CombinedLoadCase(
        axial_stress=0.0, ice_pressure=1.0e6,
        patch_length=0.70, patch_width=0.35, hydrostatic_pressure=0.0)
    res = ultimate_ice_pressure(geom, case)
    margin = res["ice_pressure_ultimate"] / p_rule
    assert 0.5 < margin < 2.0, f"margin = {margin}"


def test_margin_decreases_with_axial_compression():
    """More axial compression → lower margin."""
    geom = case_study_geometry()
    rule = IACSPolarClassRule()
    p_rule = rule_design_pressure(geom, rule)
    margins = []
    for eta in (0.0, 0.15, 0.30, 0.45):
        case = CombinedLoadCase(
            axial_stress=eta * geom.yield_stress, ice_pressure=1.0e6,
            patch_length=0.70, patch_width=0.35, hydrostatic_pressure=0.0)
        res = ultimate_ice_pressure(geom, case)
        margins.append(res["ice_pressure_ultimate"] / p_rule)
    assert all(margins[i] >= margins[i + 1] for i in range(len(margins) - 1))


def test_different_gamma_m_gives_different_stiffener_modulus():
    """Different gamma_m gives different stiffener modulus requirement."""
    r1 = IACSPolarClassRule(polar_class="PC6", gamma_m=1.15)
    r2 = IACSPolarClassRule(polar_class="PC6", gamma_m=1.30)
    z1 = rule_stiffener_modulus(2.0e6, 0.35, 2.1, 0.70, 355.0e6, r1)
    z2 = rule_stiffener_modulus(2.0e6, 0.35, 2.1, 0.70, 355.0e6, r2)
    assert z2 > z1  # higher gamma_m → higher required modulus
