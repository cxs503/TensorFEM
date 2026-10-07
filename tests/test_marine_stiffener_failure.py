import pytest

from tensorfem.marine_stiffener_failure import (
    circular_pit_bending_screen, lateral_torsional_buckling_screen,
    run_stiffener_failure_qualification,
)


def test_ltb_and_local_pit_qualification_passes_three_percent():
    report = run_stiffener_failure_qualification()
    assert report["passed"] is True
    ltb = report["lateral_torsional_buckling"]["results"]
    assert ltb[-1]["relative_error"] < 0.03
    assert all(a["relative_error"] > b["relative_error"] for a, b in zip(ltb, ltb[1:]))
    pits = report["local_circular_pit"]["results"]
    assert max(row["pit_area_relative_error"] for row in pits[-2:]) < 0.03
    assert "not plate-stiffener interaction" in report["scope"]


def test_warping_increases_lateral_torsional_resistance():
    without = lateral_torsional_buckling_screen(segments=64, warping_constant=1e-20)
    with_warping = lateral_torsional_buckling_screen(segments=64)
    assert with_warping["critical_moment"] > without["critical_moment"]


def test_local_pit_reduces_effective_bending_rigidity():
    row = circular_pit_bending_screen(cells=128)
    assert row["effective_bending_rigidity"] < 1.0
    assert row["pit_thickness_ratio"] < 1.0
    assert row["pit_area_relative_error"] < 0.03


@pytest.mark.parametrize("call", [
    lambda: lateral_torsional_buckling_screen(segments=True),
    lambda: lateral_torsional_buckling_screen(segments=1),
    lambda: circular_pit_bending_screen(cells=1),
    lambda: circular_pit_bending_screen(cells=8, rigidity_loss=1.0),
    lambda: circular_pit_bending_screen(cells=8, pit_center_x=0.1),
])
def test_failure_screens_reject_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
