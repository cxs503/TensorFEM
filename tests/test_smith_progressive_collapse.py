"""Regression tests for the Smith progressive-collapse panel surface.

Acceptance criteria (a)-(h) of campaign node P70-N01 are asserted here; each
test names the criterion it covers.  Criterion (c) (three-specimen experimental
comparison) can only be exercised through the harness with a synthetic fixture,
because the primary experimental table is WITHHELD in this environment -- see
``benchmarks/paik_thayamballi_2003_specimens.json``.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.smith_progressive_collapse import (
    EXPERIMENT_SCHEMA,
    IACSPanelGeometry,
    PlateStrip,
    SmithPanel,
    TeeStiffener,
    compare_experimental_panels,
    csr_format_panel_ultimate,
    equal_biaxial_capacity,
    panel_interaction,
    round_sig,
    sheet_shear_ultimate,
)

STEEL_E = 2.06e11
STEEL_NU = 0.3
STEEL_SY = 2.35e8


def _plate_for_beta(beta: float, *, aspect: float = 3.0, spacing: float = 0.8,
                    **kwargs) -> PlateStrip:
    """Plate field with a prescribed slenderness beta and preset a/b ratio."""
    thickness = spacing * math.sqrt(STEEL_SY / STEEL_E) / beta
    return PlateStrip(length=aspect * spacing, width=spacing,
                      thickness=thickness, **kwargs).validate()


@pytest.fixture(scope="module")
def iacs_geometry() -> IACSPanelGeometry:
    return IACSPanelGeometry()


@pytest.fixture(scope="module")
def iacs_panel(iacs_geometry: IACSPanelGeometry) -> SmithPanel:
    return iacs_geometry.smith_panel()


# --------------------------------------------------------------------------
# geometry and slenderness
# --------------------------------------------------------------------------
def test_buckling_coefficient_returns_four_for_integer_aspect() -> None:
    for aspect in (2.0, 3.0, 4.0, 5.0):
        plate = _plate_for_beta(2.0, aspect=aspect)
        assert plate.buckling_coefficient == pytest.approx(4.0, rel=1e-12)


def test_buckling_coefficient_rises_above_four_for_non_integer_aspect() -> None:
    # a/b = 1.5: m = 1 gives 4.694, m = 2 gives 4.340, so the exact minimum is
    # 4.340 (Timoshenko & Woinowsky-Krieger, simply supported uniaxial field).
    plate = _plate_for_beta(2.0, aspect=1.5)
    assert plate.buckling_coefficient == pytest.approx(4.3403, rel=1e-4)
    assert plate.buckling_coefficient > 4.0


def test_plate_slenderness_matches_definition() -> None:
    plate = _plate_for_beta(2.0)
    assert plate.beta == pytest.approx(2.0, rel=1e-12)
    # sigma_cr / sigma_y = k pi^2 /(12 (1-nu^2)) / beta^2 with k = 4
    expected = 4.0 * math.pi ** 2 / (12.0 * (1.0 - STEEL_NU ** 2)) / 4.0
    assert plate.buckling_stress / STEEL_SY == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# criterion (a): plate ultimate versus Faulkner (<= 10 %)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("beta", [1.0, 1.2, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
def test_criterion_a_plate_ultimate_agrees_with_faulkner(beta: float) -> None:
    plate = _plate_for_beta(beta)
    faulkner = min(1.0, 2.25 / plate.beta - 1.25 / plate.beta ** 2)
    assert plate.ultimate_ratio == pytest.approx(faulkner, rel=0.10)
    # at the reference imperfection / residual state the model IS the anchor
    assert plate.ultimate_ratio == pytest.approx(faulkner, rel=1e-12)


def test_plate_ultimate_decreases_monotonically_with_slenderness() -> None:
    values = [_plate_for_beta(beta).ultimate_ratio for beta in (1.0, 1.5, 2.0, 2.5, 3.0)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert all(0.0 < value <= 1.0 for value in values)


def test_plate_ultimate_never_exceeds_yield() -> None:
    for beta in (0.6, 0.8, 1.0, 1.5, 2.0):
        plate = _plate_for_beta(beta)
        assert plate.ultimate_stress <= STEEL_SY * (1.0 + 1e-12)


# --------------------------------------------------------------------------
# criterion (d): initial imperfection sensitivity (>= 5 % drop)
# --------------------------------------------------------------------------
def test_criterion_d_imperfection_sensitivity_direction_and_magnitude(
        iacs_geometry: IACSPanelGeometry) -> None:
    reference = iacs_geometry.plate_strip()
    ref_ratio = reference.imperfection_over_thickness  # 0.1 beta^2
    low = iacs_geometry.plate_strip(imperfection_ratio=ref_ratio).ultimate_ratio
    high = iacs_geometry.plate_strip(imperfection_ratio=3.0 * ref_ratio).ultimate_ratio
    drop = low / high - 1.0
    assert drop >= 0.05, f"imperfection 0.1beta^2 -> 0.3beta^2 dropped only {drop:.4f}"
    # documented calibration: ~5 % per doubling of the standard imperfection
    assert 0.05 <= drop <= 0.30
    doubled = iacs_geometry.plate_strip(imperfection_ratio=2.0 * ref_ratio).ultimate_ratio
    assert low / doubled - 1.0 == pytest.approx(0.05, abs=0.01)


def test_imperfection_factor_is_unity_at_reference_state() -> None:
    plate = _plate_for_beta(2.0)
    assert plate.imperfection_factor == pytest.approx(1.0, rel=1e-12)
    assert plate.imperfection_over_thickness == pytest.approx(
        0.1 * plate.beta ** 2, rel=1e-12)


# --------------------------------------------------------------------------
# criterion (e): residual stress effect (>= 3 % reduction)
# --------------------------------------------------------------------------
def test_criterion_e_residual_stress_reduces_plate_ultimate(
        iacs_geometry: IACSPanelGeometry) -> None:
    with_residual = iacs_geometry.plate_strip(residual_ratio=0.2)
    without = iacs_geometry.plate_strip(residual_ratio=0.0)
    reduction = 1.0 - with_residual.ultimate_ratio / without.ultimate_ratio
    assert reduction >= 0.03, f"residual stress reduced the ultimate by {reduction:.4f}"
    # the reference residual stress sigma_rc = 0.2 sigma_y defines factor 1.0
    assert with_residual.residual_factor == pytest.approx(1.0, rel=1e-12)
    assert without.residual_factor > 1.0
    stronger = iacs_geometry.plate_strip(residual_ratio=0.3)
    assert stronger.residual_factor < 1.0


def test_residual_effect_vanishes_for_stocky_fields() -> None:
    # beta = 0.8: sigma_cr > sigma_y, the field squashes before it buckles, so
    # the post-buckling fraction (and hence the residual-stress sensitivity) is 0.
    plate = _plate_for_beta(0.8)
    assert plate.post_buckling_fraction == 0.0
    assert plate.residual_factor == pytest.approx(1.0, rel=1e-12)


def test_residual_effect_grows_with_slenderness() -> None:
    # sigma_cr/sigma_y > 1 (beta <= 1.9) => the field squashes before it buckles
    assert _plate_for_beta(1.2).post_buckling_fraction == 0.0
    assert _plate_for_beta(1.8).post_buckling_fraction == 0.0
    stocky = _plate_for_beta(2.0).post_buckling_fraction
    slender = _plate_for_beta(3.0).post_buckling_fraction
    assert 0.0 < stocky < slender < 1.0


# --------------------------------------------------------------------------
# load-end shortening curve
# --------------------------------------------------------------------------
def test_plate_curve_is_continuous_at_phase_boundaries() -> None:
    plate = _plate_for_beta(2.25)
    eps_u = plate.peak_strain
    eps_1 = min(plate.buckling_stress, plate.ultimate_stress) / STEEL_E
    for eps in (eps_1, eps_u):
        left = plate.stress(eps * (1.0 - 1e-9))
        right = plate.stress(eps * (1.0 + 1e-9))
        assert left == pytest.approx(right, rel=1e-6)
    assert plate.stress(eps_u) == pytest.approx(plate.ultimate_stress, rel=1e-12)


def test_plate_curve_starts_elastic_with_unity_slope() -> None:
    plate = _plate_for_beta(2.25)
    eps = 1e-5
    assert plate.stress(eps) / STEEL_SY == pytest.approx(
        eps / plate.yield_strain, rel=1e-12)
    assert plate.peak_strain <= plate.yield_strain * (1.0 + 1e-12)


def test_tension_branch_is_elastic_and_capped_at_yield() -> None:
    plate = _plate_for_beta(2.25)
    assert plate.stress(-1e-5) == pytest.approx(-STEEL_E * 1e-5, rel=1e-12)
    assert plate.stress(-1e-2) == pytest.approx(-STEEL_SY, rel=1e-12)


def test_criterion_g_post_peak_branch_has_negative_stiffness(iacs_panel: SmithPanel) -> None:
    rows = iacs_panel.curve()["rows"]
    values = [row["panel_stress_over_yield"] for row in rows]
    peak = max(range(len(values)), key=values.__getitem__)
    assert peak > 0 and peak < len(values) - 1
    assert all(a <= b + 1e-15 for a, b in zip(values[:peak + 1], values[1: peak + 1]))
    assert all(a >= b - 1e-15 for a, b in zip(values[peak:], values[peak + 1:]))
    # strictly decreasing directly after the peak: the curve must not plateau at once
    assert values[peak] - values[peak + 5] > 1e-3
    # and no un-physical re-hardening anywhere in the tail
    assert values[-1] < 0.6 * values[peak]


def test_panel_ultimate_comes_from_curve_peak(iacs_panel: SmithPanel) -> None:
    curve = iacs_panel.curve()
    peak = max(row["panel_stress_over_yield"] for row in curve["rows"])
    assert curve["ultimate_ratio"] == pytest.approx(peak, rel=1e-12)
    assert iacs_panel.ultimate()["ultimate_ratio"] == pytest.approx(peak, rel=1e-12)


# --------------------------------------------------------------------------
# stiffener
# --------------------------------------------------------------------------
def test_stiffener_limiting_stress_is_min_of_mechanisms(iacs_geometry: IACSPanelGeometry) -> None:
    stiff = iacs_geometry.stiffener()
    mechanisms = {"column": stiff.column_stress,
                  "web_local": stiff.web_buckling_stress,
                  "flange_local": stiff.flange_buckling_stress}
    assert stiff.ultimate_stress == pytest.approx(min(min(mechanisms.values()), STEEL_SY),
                                                 rel=1e-12)
    assert stiff.governing_mode == min(mechanisms, key=mechanisms.get)
    assert stiff.ultimate_stress <= stiff.squash_stress * (1.0 + 1e-12)


def test_stiffener_euler_stress_falls_with_span_squared(iacs_geometry: IACSPanelGeometry) -> None:
    base = iacs_geometry.stiffener()
    longer = TeeStiffener(**{**base.__dict__, "panel_length": 2.0 * base.panel_length}
                          ).validate()
    assert longer.euler_stress == pytest.approx(base.euler_stress / 4.0, rel=1e-12)


def test_stiffener_perry_curve_between_euler_and_squash(iacs_geometry: IACSPanelGeometry) -> None:
    slender = TeeStiffener(**{**iacs_geometry.stiffener().__dict__,
                              "panel_length": 40.0}).validate()
    assert slender.column_stress <= slender.euler_stress * (1.0 + 1e-12)
    stocky = iacs_geometry.stiffener()
    assert stocky.column_stress <= stocky.squash_stress * (1.0 + 1e-12)


def test_stiffener_local_buckling_falls_with_thinner_web(iacs_geometry: IACSPanelGeometry) -> None:
    base = iacs_geometry.stiffener()
    thin = TeeStiffener(**{**base.__dict__, "web_thickness": base.web_thickness / 2.0}
                        ).validate()
    assert thin.web_buckling_stress == pytest.approx(base.web_buckling_stress / 4.0, rel=1e-12)


# --------------------------------------------------------------------------
# Smith combination
# --------------------------------------------------------------------------
def test_panel_stress_is_area_weighted(iacs_panel: SmithPanel) -> None:
    eps = 0.8 * iacs_panel.plate.yield_strain
    row = iacs_panel.stress(eps)
    expected = (iacs_panel.plate_area * row["plate_stress"]
                + iacs_panel.stiffener_area * row["stiffener_stress"]) / iacs_panel.total_area
    assert row["panel_stress"] == pytest.approx(expected, rel=1e-12)


def test_criterion_b_panel_ultimate_agrees_with_csr_format(iacs_panel: SmithPanel) -> None:
    result = csr_format_panel_ultimate(iacs_panel)
    assert abs(float(result["relative_error"])) <= 0.10, result
    assert result["governing_mechanism"] in {
        "plate_field", "panel_column", "web_local", "flange_local", "yield"}
    # both estimates must sit between the weakest component and yield
    assert 0.4 <= float(result["csr_format_ratio"]) <= 1.0
    assert 0.4 <= float(result["smith_ratio"]) <= 1.0


def test_panel_ultimate_between_component_ultimates(iacs_panel: SmithPanel) -> None:
    ultimate = iacs_panel.ultimate()
    low = min(iacs_panel.plate.ultimate_ratio, iacs_panel.stiffener.ultimate_ratio)
    high = max(iacs_panel.plate.ultimate_ratio, iacs_panel.stiffener.ultimate_ratio)
    assert low <= ultimate["ultimate_ratio"] <= high


# --------------------------------------------------------------------------
# criterion (f): biaxial interaction
# --------------------------------------------------------------------------
def test_criterion_f_equal_biaxial_ratio_is_one_over_sqrt_two(iacs_panel: SmithPanel) -> None:
    result = equal_biaxial_capacity(iacs_panel)
    assert result["ratio_to_uniaxial"] == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-12)
    assert abs(result["relative_error"]) <= 0.10


def test_equal_biaxial_with_jockeying_stays_within_ten_percent(iacs_panel: SmithPanel) -> None:
    result = equal_biaxial_capacity(iacs_panel, jockeying=0.2)
    assert abs(result["relative_error"]) <= 0.10
    assert result["ratio_to_uniaxial"] > 1.0 / math.sqrt(2.0)


def test_interaction_surface_hits_unity_at_pure_limits(iacs_panel: SmithPanel) -> None:
    ultimate = float(iacs_panel.ultimate()["ultimate_stress"])
    axial = panel_interaction(iacs_panel, sigma_x=ultimate)
    assert axial["csr_quadratic_utilisation"] == pytest.approx(1.0, rel=1e-9)
    shear = panel_interaction(iacs_panel, sigma_x=0.0,
                              tau=float(axial["shear_ultimate"]))
    assert shear["csr_quadratic_utilisation"] == pytest.approx(1.0, rel=1e-9)
    half = panel_interaction(iacs_panel, sigma_x=0.5 * ultimate)
    assert half["csr_quadratic_utilisation"] == pytest.approx(0.5, rel=1e-9)


def test_equal_biaxial_state_lands_on_the_surface(iacs_panel: SmithPanel) -> None:
    capacity = equal_biaxial_capacity(iacs_panel)
    state = panel_interaction(iacs_panel, sigma_x=capacity["equal_biaxial_stress"],
                              sigma_y=capacity["equal_biaxial_stress"])
    assert state["csr_quadratic_utilisation"] == pytest.approx(1.0, rel=1e-9)


def test_von_mises_surface_under_equal_biaxial_is_yield() -> None:
    panel = IACSPanelGeometry().smith_panel()
    state = panel_interaction(panel, sigma_x=STEEL_SY, sigma_y=STEEL_SY)
    # plane-stress von Mises: s1^2 - s1 s2 + s2^2 = sigma_y^2 at s1 = s2 = sigma_y
    assert state["von_mises_utilisation"] == pytest.approx(1.0, rel=1e-12)


def test_shear_ultimate_is_bounded_by_yield_over_sqrt_three(iacs_geometry: IACSPanelGeometry) -> None:
    shear = sheet_shear_ultimate(iacs_geometry.plate_strip())
    assert shear["tau_ultimate"] <= shear["tau_yield"] * (1.0 + 1e-12)
    slender = sheet_shear_ultimate(_plate_for_beta(3.5))
    stocky = sheet_shear_ultimate(_plate_for_beta(1.2))
    assert slender["tau_ultimate"] < stocky["tau_ultimate"]


# --------------------------------------------------------------------------
# criterion (h): six significant digits on every reported quantity
# --------------------------------------------------------------------------
def _is_six_digit(value: float) -> bool:
    """True when ``value`` is unchanged by rounding to six significant digits."""
    return float(f"{value:.5e}") == value


def test_criterion_h_summaries_carry_six_significant_digits(iacs_geometry: IACSPanelGeometry) -> None:
    panel = iacs_geometry.smith_panel()
    payload = panel.summary()
    values: list[float] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            values.append(float(node))

    walk(payload)
    assert values
    for value in values:
        assert value == round_sig(value), f"{value!r} is not 6-significant-digit rounded"
        assert _is_six_digit(value), f"{value!r} carries more than 6 significant digits"
    assert round_sig(176886123.456789) == 176886000.0
    assert round_sig(0.7527061234) == 0.752706


# --------------------------------------------------------------------------
# criterion (c): experimental comparison harness
# --------------------------------------------------------------------------
def _synthetic_specimen(specimen: str, *, scale: float = 1.0,
                        verified: bool = False) -> dict:
    geometry = IACSPanelGeometry()
    panel = geometry.smith_panel()
    computed = float(panel.ultimate()["ultimate_stress"])
    payload = {
        "specimen": specimen,
        "span_m": geometry.span,
        "spacing_m": geometry.spacing,
        "plate_thickness_m": geometry.plate_thickness,
        "web_height_m": geometry.web_height,
        "web_thickness_m": geometry.web_thickness,
        "flange_width_m": geometry.flange_width,
        "flange_thickness_m": geometry.flange_thickness,
        "young_pa": geometry.young,
        "poisson": geometry.poisson,
        "yield_stress_pa": geometry.yield_stress,
        "measured_ultimate_stress_pa": computed * scale,
        "provenance": "SYNTHETIC FIXTURE (not experimental data)",
        "primary_source_verified": verified,
    }
    assert set(payload) == set(EXPERIMENT_SCHEMA)
    return payload


def test_criterion_c_harness_recovers_injected_values_exactly() -> None:
    result = compare_experimental_panels([_synthetic_specimen("SYN-1")])
    assert result["specimen_count"] == 1
    assert result["worst_relative_error"] == pytest.approx(0.0, abs=1e-9)
    assert result["rows"][0]["computed_over_measured"] == pytest.approx(1.0, rel=1e-9)


def test_criterion_c_harness_reports_deviation_and_propagates_grade() -> None:
    result = compare_experimental_panels([
        _synthetic_specimen("SYN-1", scale=1.0),
        _synthetic_specimen("SYN-2", scale=1.10),
        _synthetic_specimen("SYN-3", scale=0.90),
    ])
    assert result["specimen_count"] == 3
    assert result["worst_relative_error"] == pytest.approx(1.0 / 0.9 - 1.0, rel=1e-9)
    # unverified transcriptions must never be promoted to REAL evidence
    assert result["evidence_grade"] == "WITHHELD"
    assert result["all_primary_sources_verified"] is False


def test_harness_rejects_incomplete_specimens() -> None:
    with pytest.raises(ValueError):
        compare_experimental_panels([{"specimen": "X"}])
    with pytest.raises(ValueError):
        compare_experimental_panels([])


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------
def test_validation_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError):
        PlateStrip(length=2.4, width=0.8, thickness=0.0).validate()
    with pytest.raises(ValueError):
        PlateStrip(length=2.4, width=0.8, thickness=0.012,
                   residual_ratio=1.2).validate()
    with pytest.raises(ValueError):
        PlateStrip(length=2.4, width=0.8, thickness=0.012,
                   residual_strength=1.5).validate()


def test_panel_requires_matching_geometry(iacs_geometry: IACSPanelGeometry) -> None:
    plate = iacs_geometry.plate_strip()
    stiffener = TeeStiffener(**{**iacs_geometry.stiffener().__dict__,
                                "plate_width": 0.9}).validate()
    with pytest.raises(ValueError):
        SmithPanel(plate=plate, stiffener=stiffener).validate()


def test_interaction_rejects_out_of_range_jockeying(iacs_panel: SmithPanel) -> None:
    with pytest.raises(ValueError):
        equal_biaxial_capacity(iacs_panel, jockeying=0.8)
    with pytest.raises(ValueError):
        panel_interaction(iacs_panel, sigma_x=1.0e8, jockeying=-0.1)