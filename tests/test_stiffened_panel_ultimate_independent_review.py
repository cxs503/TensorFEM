"""Regression tests for the independent review of ``stiffened_panel_ultimate``.

Every reference value in this file was produced by
``scripts/verify_stiffened_panel_ultimate_independent.py``, which solves the
plate field with a finite-difference Kirchhoff solver and integrates the
yield-line virtual work numerically -- code paths that share nothing with the
module under test.  The two defects the review found are pinned here:

* ``simply_supported_plate_series`` used to return a deflection ``E`` times too
  large in SI units (it carried ``D/E`` into the sum and never multiplied back);
* ``flange_local_buckling_stress`` used the *total* flange width as the
  one-edge-free outstand, which under-predicted ``sigma_cr`` by a factor of
  about four and could silently halve the panel ultimate for wide flanges.

The single remaining step discontinuity of the yield-line selector is asserted
as a *documented limitation* (with its measured size) so that any change to it
is deliberate; see ``docs/stiffened-panel-ultimate-independent-review.md``.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.stiffened_panel_ultimate import (
    SQUARE_PLATE_PRESSURE_COEFFICIENT,
    StiffenedPanel,
    plate_pressure_bending_stress,
    simply_supported_plate_series,
    yield_line_collapse_pressure,
)

E = 2.06e11
NU = 0.3
T = 0.02
P = 1.0
B = 1.0
D_PLATE = E * T**3 / (12.0 * (1.0 - NU**2))


def panel(**kwargs) -> StiffenedPanel:
    base = dict(span=2.4, spacing=0.8, plate_thickness=0.020, web_height=0.20,
                web_thickness=0.010, flange_width=0.08, flange_thickness=0.012,
                young=E, yield_stress=235e6)
    base.update(kwargs)
    return StiffenedPanel(**base)


# ---------------------------------------------------------------------------
# 1. plate field: units and independent finite-difference reference
# ---------------------------------------------------------------------------
def test_deflection_is_a_physical_length_not_a_unit_up_to_E() -> None:
    """alpha = 3, b = 1 m, t = 20 mm, p = 1 Pa -> w ~ 8.1e-8 m (FD: 8.10572e-8)."""
    series = plate_pressure_bending_stress(3.0 * B, B, T, P, young=E, terms=201)
    assert series["centre_deflection"] == pytest.approx(8.10572e-08, rel=1e-2)
    # deflection must scale exactly as 1/E, i.e. D must be the physical rigidity
    unit_young = plate_pressure_bending_stress(3.0 * B, B, T, P, young=1.0, terms=201)
    assert series["centre_deflection"] * E == pytest.approx(
        unit_young["centre_deflection"], rel=1e-9)
    # the reported rigidity is D, not D/E
    assert simply_supported_plate_series(3.0 * B, B, T, P, young=E)["plate_rigidity"] == pytest.approx(D_PLATE, rel=1e-12)


def test_navier_moments_match_the_independent_fd_solution() -> None:
    """Centre moments from the finite-difference solve, three aspect ratios."""
    references = {
        1.0: {"my": 0.047886, "mx": 0.047886},
        2.0: {"my": 0.101680, "mx": 0.046353},
        3.0: {"my": 0.118861, "mx": 0.040630},
    }
    for alpha, ref in references.items():
        case = plate_pressure_bending_stress(alpha * B, B, T, P, young=E, terms=201)
        assert case["coefficient_m_over_pb2"] == pytest.approx(max(ref["my"], ref["mx"]), rel=2e-3)
        series = simply_supported_plate_series(alpha * B, B, T, P, young=E, terms=201)
        assert series["centre_moment_y"] / (P * B**2) == pytest.approx(ref["my"], rel=2e-3)
        assert series["centre_moment_x"] / (P * B**2) == pytest.approx(ref["mx"], rel=2e-3)


def test_square_plate_constant_is_the_converged_series() -> None:
    square = simply_supported_plate_series(B, B, T, P, young=E, terms=201)
    assert square["centre_moment"] == pytest.approx(SQUARE_PLATE_PRESSURE_COEFFICIENT, rel=1e-3)


def test_navier_series_reaches_the_one_way_strip_limits() -> None:
    """alpha -> inf: M = p b^2/8 and w = 5 p b^4/(384 D) (elementary beam result)."""
    series = simply_supported_plate_series(10.0 * B, B, T, P, young=E, terms=201)
    assert series["centre_moment_y"] == pytest.approx(P * B**2 / 8.0, rel=1e-3)
    assert series["centre_deflection"] == pytest.approx(5.0 * P * B**4 / (384.0 * D_PLATE), rel=1e-3)
    # and the two-way flexure is genuinely smaller for a square field
    square = simply_supported_plate_series(B, B, T, P, young=E, terms=201)
    assert square["centre_deflection"] < series["centre_deflection"] / 3.0


# ---------------------------------------------------------------------------
# 2. flange outstand convention
# ---------------------------------------------------------------------------
def test_flange_local_buckling_uses_the_outstand_not_the_total_width() -> None:
    p = panel(flange_width=0.080, flange_thickness=0.010, web_thickness=0.010)
    assert p.flange_outstand == pytest.approx((0.080 - 0.010) / 2.0, rel=1e-12)
    reference = 0.425 * math.pi**2 * E / (12.0 * (1.0 - NU**2)) / (p.flange_outstand / p.flange_thickness) ** 2
    assert p.flange_local_buckling_stress == pytest.approx(reference, rel=1e-12)
    # the pre-fix convention (total width) under-predicted by (b_f/outstand)^2
    total_width_value = 0.425 * math.pi**2 * E / (12.0 * (1.0 - NU**2)) / (p.flange_width / p.flange_thickness) ** 2
    assert p.flange_local_buckling_stress / total_width_value == pytest.approx(
        (p.flange_width / p.flange_outstand) ** 2, rel=1e-12)
    assert p.flange_local_buckling_stress > 4.0 * total_width_value


def test_wide_flange_panel_no_longer_loses_a_fifth_of_its_strength() -> None:
    """b_f/t_f = 30: the pre-fix screen returned 0.800 sigma_y, the review 0.941."""
    p = panel(flange_width=0.30, flange_thickness=0.010)
    assert p.panel_yield_ratio == pytest.approx(0.941, abs=2e-3)
    assert p.governing_stiffener_mode == "column"


def test_documented_ice_belt_panel_is_unchanged_by_the_fix() -> None:
    """The 20 mm / 800 mm benchmark panel is column governed; the fix must not move it."""
    p = panel()
    assert p.panel_yield_ratio == pytest.approx(0.93447, abs=1e-4)
    assert p.governing_stiffener_mode == "column"
    assert p.flange_local_buckling_stress > p.yield_stress


# ---------------------------------------------------------------------------
# 3. yield-line capacity: classical limits and the documented step
# ---------------------------------------------------------------------------
def test_pyramid_mechanism_matches_numerically_integrated_virtual_work() -> None:
    """12 (a^2+b^2)/(a^2 b^2) M_p equals the numeric virtual work to 1e-3."""
    sy, t = 235e6, 0.020
    mp = sy * t**2 / 4.0
    for alpha in (1.0, 1.25, 1.5):
        result = yield_line_collapse_pressure(alpha, 1.0, t, sy)
        classical = 12.0 * mp * (alpha**2 + 1.0) / (alpha**2 * 1.0)
        assert result["fan_mechanism_pressure"] == pytest.approx(classical, rel=1e-12)
        assert result["collapse_pressure"] == pytest.approx(classical, rel=1e-12)


def test_selector_step_at_aspect_1p5_is_a_documented_limitation() -> None:
    """Measured 53.8 % step: fan 17.33 -> strip 8.0 in p_u b^2/M_p units.

    The strip mechanism is only admissible for long fields, so the step is
    conservative rather than unsafe, but it is a real discontinuity of the
    screening curve and is recorded as a gap; see
    ``docs/stiffened-panel-ultimate-independent-review.md`` section 5.
    """
    sy, t = 235e6, 0.020
    mp = sy * t**2 / 4.0
    below = yield_line_collapse_pressure(1.5 - 1e-6, 1.0, t, sy)
    above = yield_line_collapse_pressure(1.5 + 1e-6, 1.0, t, sy)
    assert below["governing_mechanism"] == "fan"
    assert above["governing_mechanism"] == "transition"
    assert below["collapse_pressure"] * 1.0 / mp == pytest.approx(17.333, abs=1e-2)
    assert above["collapse_pressure"] * 1.0 / mp == pytest.approx(8.0, abs=1e-9)
    jump = 1.0 - above["collapse_pressure"] / below["collapse_pressure"]
    assert 0.5 < jump < 0.56
    # long fields are unaffected: one-way strip for a/b >= 2
    long_field = yield_line_collapse_pressure(4.0, 1.0, t, sy)
    assert long_field["collapse_pressure"] == pytest.approx(8.0 * mp, rel=1e-12)