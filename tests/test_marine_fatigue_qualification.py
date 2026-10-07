import math

import pytest

from tensorfem.marine_fatigue_qualification import (
    SNLogLogCurve, StressBlock, hot_spot_stress_pa, miner_damage,
    run_fatigue_qualification, thickness_corrected_stress_pa,
)


def test_linear_and_quadratic_hot_spot_extrapolation():
    linear = hot_spot_stress_pa((132e6, 105e6), (0.004, 0.010), method="linear")
    quadratic = hot_spot_stress_pa(
        (121.2e6, 99.2e6, 87.2e6), (0.004, 0.009, 0.014), method="quadratic"
    )
    assert linear == pytest.approx(150e6)
    assert quadratic == pytest.approx(146e6)


def test_sn_curve_and_multi_block_miner_oracles():
    curve = SNLogLogCurve(100e6, 2e6, 3)
    assert curve.cycles_to_failure(125e6) == pytest.approx(1_024_000)
    result = miner_damage((StressBlock(100e6, 500_000), StressBlock(125e6, 256_000)), curve)
    assert result.block_damage == pytest.approx((0.25, 0.25))
    assert result.damage == pytest.approx(0.5)
    assert result.passed
    assert not miner_damage((StressBlock(100e6, 2_000_001),), curve).passed


def test_optional_thickness_correction_is_explicit_and_one_sided():
    corrected = thickness_corrected_stress_pa(
        100e6, 0.040, reference_thickness_m=0.025, exponent=0.25
    )
    uncorrected = thickness_corrected_stress_pa(
        100e6, 0.020, reference_thickness_m=0.025, exponent=0.25
    )
    assert corrected == pytest.approx(100e6 * 1.6**0.25)
    assert uncorrected == 100e6


@pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan, True, "1"])
def test_curve_inputs_fail_closed(bad):
    with pytest.raises((TypeError, ValueError)):
        SNLogLogCurve(bad, 2e6, 3)


def test_hot_spot_and_blocks_fail_closed():
    with pytest.raises(ValueError, match="requires 2"):
        hot_spot_stress_pa((1,), (0.1,), method="linear")
    with pytest.raises(ValueError, match="distinct"):
        hot_spot_stress_pa((1, 2), (0.1, 0.1), method="linear")
    with pytest.raises(ValueError, match="method"):
        hot_spot_stress_pa((1, 2), (0.1, 0.2), method="cubic")
    with pytest.raises(ValueError, match="at least one"):
        miner_damage((), SNLogLogCurve(1, 1, 1))


def test_qualification_gate_scope_units_and_oracles():
    report = run_fatigue_qualification()
    assert report["passed"]
    assert "no TensorLBM" in report["scope"]
    assert "not class-society certification" in report["certification"]
    assert report["units"] == {"stress": "Pa", "distance_and_thickness": "m", "cycles": "1"}
    assert len(report["evidence"]) == 4
    assert all(row["relative_error"] < row["tolerance"] <= 0.03 for row in report["evidence"])
