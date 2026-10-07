import math

import pytest

from tensorfem.marine_fatigue_advanced import (
    CrackGrowthBlock,
    ParisLaw,
    PathStressSample,
    RainflowCycle,
    hot_spot_stress_from_path_pa,
    paris_crack_growth,
    rainflow_cycles,
    run_advanced_fatigue_qualification,
    spectrum_miner_damage,
)
from tensorfem.marine_fatigue_qualification import SNLogLogCurve


def test_path_nodes_are_interpolated_before_linear_and_quadratic_extrapolation():
    # sigma(x) = 150 MPa - 4 GPa/m*x; nodes deliberately miss rule locations.
    path = tuple(
        PathStressSample(x, 150e6 - 4e9 * x)
        for x in (0.0, 0.003, 0.007, 0.012, 0.016)
    )
    assert hot_spot_stress_from_path_pa(path, 0.01, method="linear") == pytest.approx(150e6)
    assert hot_spot_stress_from_path_pa(path, 0.01, method="quadratic") == pytest.approx(150e6)


def test_rainflow_triangle_has_two_half_cycles_and_one_equivalent_cycle():
    cycles = rainflow_cycles((0.0, 100e6, 0.0))
    assert [(row.range_pa, row.mean_pa, row.count) for row in cycles] == [
        (100e6, 50e6, 0.5),
        (100e6, 50e6, 0.5),
    ]
    assert sum(row.count for row in cycles) == 1.0


def test_astm_four_point_closes_inner_cycle_and_retains_residue():
    cycles = rainflow_cycles((0.0, 100e6, 0.0, 200e6, 0.0))
    by_range = {}
    for row in cycles:
        by_range[row.range_pa] = by_range.get(row.range_pa, 0.0) + row.count
    assert by_range == pytest.approx({100e6: 1.0, 200e6: 1.0})


def test_variable_amplitude_miner_matches_hand_sum():
    cycles = (
        RainflowCycle(100e6, 0.0, 1.0),
        RainflowCycle(125e6, 10e6, 0.5),
    )
    curve = SNLogLogCurve(100e6, 2e6, 3.0)
    result = spectrum_miner_damage(cycles, curve)
    expected = 1.0 / 2e6 + 0.5 / 1_024_000.0
    assert result.damage == pytest.approx(expected)
    assert result.cycle_damage == pytest.approx((0.5e-6, 0.5 / 1_024_000.0))


def test_paris_m2_matches_closed_form_and_processes_blocks_sequentially():
    law = ParisLaw(coefficient_c=1e-24, exponent_m=2.0, geometry_factor=1.0)
    blocks = (CrackGrowthBlock(100e6, 400.0), CrackGrowthBlock(100e6, 600.0))
    result = paris_crack_growth(0.01, blocks, law)
    oracle = 0.01 * math.exp(1e-24 * math.pi * (100e6) ** 2 * 1000.0)
    assert result.final_crack_m == pytest.approx(oracle)
    assert len(result.crack_history_m) == 3
    assert not result.reached_critical


def test_paris_stops_after_crossing_critical_crack():
    law = ParisLaw(1e-24, 2.0)
    result = paris_crack_growth(
        0.01,
        (CrackGrowthBlock(100e6, 1000), CrackGrowthBlock(100e6, 1000)),
        law,
        # The exact first-block result is 0.010000314... m, so this threshold
        # is crossed by the first block and proves early-stop semantics.
        critical_crack_m=0.0100001,
    )
    assert result.reached_critical
    assert len(result.crack_history_m) == 2


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, True, "1"])
def test_path_and_paris_inputs_fail_closed(bad):
    with pytest.raises((TypeError, ValueError)):
        PathStressSample(bad, 1.0)
    with pytest.raises((TypeError, ValueError)):
        ParisLaw(bad, 2.0)


def test_strict_path_history_cycle_and_crack_validation():
    with pytest.raises(ValueError, match="strictly increasing"):
        hot_spot_stress_from_path_pa(
            (PathStressSample(0.01, 1.0), PathStressSample(0.005, 2.0)), 0.01
        )
    with pytest.raises(ValueError, match="bracket"):
        hot_spot_stress_from_path_pa(
            (PathStressSample(0.0, 1.0), PathStressSample(0.005, 2.0)), 0.01
        )
    with pytest.raises(ValueError, match="distinct"):
        rainflow_cycles((1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="count"):
        RainflowCycle(1.0, 0.0, 0.25)
    with pytest.raises(ValueError, match="exceed"):
        paris_crack_growth(0.01, (CrackGrowthBlock(1.0, 1.0),), ParisLaw(1.0, 1.0),
                           critical_crack_m=0.01)


def test_qualification_oracles_have_strict_three_percent_gate_and_scope():
    report = run_advanced_fatigue_qualification()
    assert report["passed"]
    assert "no TensorLBM" in report["scope"]
    assert "not class" in report["certification"]
    assert len(report["evidence"]) == 4
    assert all(row["relative_error"] < row["tolerance"] <= 0.03 for row in report["evidence"])
