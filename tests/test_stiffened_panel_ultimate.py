"""Closed-form and rule-check tests for the stiffened-panel ultimate module."""
from __future__ import annotations

import math

import pytest

from tensorfem.stiffened_panel_ultimate import (
    IcePressureModel,
    SQUARE_PLATE_PRESSURE_COEFFICIENT,
    StiffenedPanel,
    StructuralUnit,
    csr_hull_girder_check,
    hull_girder_progressive_collapse,
    ice_wave_combined_check,
    iso19906_ice_belt_check,
    johnson_ostenfeld_strength,
    load_end_shortening,
    panel_load_end_shortening_curve,
    plate_pressure_bending_stress,
    simply_supported_plate_series,
    square_plate_coefficient_error,
    yield_line_collapse_pressure,
)

ICE_BELT = StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=0.020,
                          web_height=0.20, web_thickness=0.010,
                          flange_width=0.08, flange_thickness=0.012)


def test_faulkner_plate_strength_matches_effective_width() -> None:
    beta_two = StiffenedPanel(span=1.0, spacing=1.0, plate_thickness=1.0 / 58.0,
                              web_height=0.2, web_thickness=0.01, flange_width=0.08,
                              flange_thickness=0.012, young=2.10e11, yield_stress=2.50e8)
    assert beta_two.plate_slenderness == pytest.approx(2.0, rel=0.02)
    beta = beta_two.plate_slenderness
    assert beta_two.faulkner_plate_strength / beta_two.yield_stress == pytest.approx(
        2.0 / beta - 1.0 / beta**2, rel=1e-12)
    stocky = StiffenedPanel(span=1.0, spacing=0.5, plate_thickness=0.05,
                            web_height=0.2, web_thickness=0.01, flange_width=0.08,
                            flange_thickness=0.012)
    assert stocky.plate_slenderness < 1.0
    assert stocky.faulkner_plate_strength == stocky.yield_stress


def test_faulkner_strength_decreases_with_plate_slenderness() -> None:
    values = []
    for thickness in (0.016, 0.012, 0.010, 0.008):
        panel = StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=thickness,
                               web_height=0.2, web_thickness=0.01, flange_width=0.08,
                               flange_thickness=0.012)
        values.append(panel.faulkner_plate_strength)
    assert all(a > b for a, b in zip(values, values[1:]))


def test_johnson_ostenfeld_is_continuous_at_half_yield() -> None:
    sigma_y = 235e6
    sigma_e = 0.5 * sigma_y
    assert johnson_ostenfeld_strength(sigma_y, sigma_e) == pytest.approx(sigma_e, rel=1e-12)
    below = johnson_ostenfeld_strength(sigma_y, 0.3 * sigma_y)
    assert below == pytest.approx(0.3 * sigma_y, rel=1e-12)
    above = johnson_ostenfeld_strength(sigma_y, 2.0 * sigma_y)
    assert above == pytest.approx(sigma_y * (1.0 - 1.0 / 8.0), rel=1e-12)
    with pytest.raises(ValueError):
        johnson_ostenfeld_strength(sigma_y, 0.0)


def test_column_euler_stress_equals_yield_over_slenderness_squared() -> None:
    panel = ICE_BELT.validate()
    assert panel.column_euler_stress == pytest.approx(
        panel.yield_stress / panel.column_slenderness**2, rel=1e-12)
    # A pinned column's Euler load must fall as 1/span^2.
    longer = StiffenedPanel(span=2.0 * panel.span, spacing=panel.spacing,
                            plate_thickness=panel.plate_thickness,
                            web_height=panel.web_height, web_thickness=panel.web_thickness,
                            flange_width=panel.flange_width,
                            flange_thickness=panel.flange_thickness,
                            young=panel.young, poisson=panel.poisson,
                            yield_stress=panel.yield_stress)
    assert longer.column_euler_stress == pytest.approx(panel.column_euler_stress / 4.0, rel=1e-12)


def test_section_inertia_matches_numerical_integration() -> None:
    panel = ICE_BELT
    # Independent composite-section integration on a fine grid.
    dz = 1e-5
    columns = [
        (-panel.web_height, 0.0, panel.web_thickness),
        (-panel.web_height - 0.5 * panel.flange_thickness,
         -panel.web_height + 0.5 * panel.flange_thickness, panel.flange_width),
        (-0.5 * panel.plate_thickness, 0.5 * panel.plate_thickness, panel.spacing),
    ]
    area = 0.0
    first = 0.0
    cells = []
    for low, high, width in columns:
        count = max(int(round((high - low) / dz)), 1)
        step = (high - low) / count
        for index in range(count):
            z = low + (index + 0.5) * step
            cells.append((width * step, z))
            area += width * step
            first += width * step * z
    centroid = first / area
    inertia = sum(a * (z - centroid) ** 2 for a, z in cells)
    assert centroid == pytest.approx(panel.centroid_offset, rel=2e-3)
    assert inertia == pytest.approx(panel.column_inertia, rel=2e-2)


def test_tripping_screen_is_optional_and_reports_itself() -> None:
    panel = ICE_BELT
    default_summary = panel.summary()
    assert default_summary["governing_stiffener_mode"] in ("column", "flange_local")
    tripping = StiffenedPanel(**{**panel.__dict__, "include_tripping": True})
    assert tripping.stiffener_limiting_stress <= tripping.tripping_stress
    assert tripping.tripping_stress > 0.0
    assert panel.tripping_stress == pytest.approx(tripping.tripping_stress, rel=1e-12)


def test_load_end_shortening_peak_equals_closed_form() -> None:
    curve = panel_load_end_shortening_curve(ICE_BELT)
    assert curve["peak_matches_closed_form"]
    panel = ICE_BELT
    elastic = load_end_shortening(panel, 0.25 * panel.yield_stress / panel.young)
    assert elastic["panel_stress"] == pytest.approx(
        0.25 * panel.yield_stress, rel=1e-9)
    strains = [row["strain"] for row in curve["rows"]]
    stresses = [row["panel_stress"] for row in curve["rows"]]
    assert all(b >= a for a, b in zip(stresses, stresses[1:]))
    assert strains[0] > 0.0
    tension = load_end_shortening(panel, -2.0 * panel.yield_stress / panel.young)
    assert tension["panel_stress"] == pytest.approx(-panel.yield_stress, rel=1e-12)


def test_navier_series_reproduces_textbook_square_plate_moment() -> None:
    error = square_plate_coefficient_error(terms=61)
    assert error["relative_error"] < 5e-3
    assert SQUARE_PLATE_PRESSURE_COEFFICIENT == pytest.approx(0.0479, rel=1e-12)
    square = simply_supported_plate_series(1.0, 1.0, 0.01, 1.0, terms=61)
    assert square["centre_deflection"] > 0.0
    assert square["centre_moment"] == pytest.approx(
        SQUARE_PLATE_PRESSURE_COEFFICIENT, rel=5e-3)
    rectangular = plate_pressure_bending_stress(2.4, 0.8, 0.02, 5.0e5)
    assert rectangular["maximum_bending_stress"] > 0.0
    # A long plate field is one-way: the governing moment approaches p b^2 / 8.
    assert 0.10 < rectangular["coefficient_m_over_pb2"] < 0.125


def test_navier_series_converges_with_term_count() -> None:
    coarse = simply_supported_plate_series(1.5, 1.0, 0.01, 1.0, terms=9)
    fine = simply_supported_plate_series(1.5, 1.0, 0.01, 1.0, terms=101)
    assert abs(coarse["centre_moment"] / fine["centre_moment"] - 1.0) < 1e-2


def test_yield_line_square_plate_recovers_classical_value() -> None:
    result = yield_line_collapse_pressure(1.0, 1.0, 0.01, 235e6)
    mp = 235e6 * 0.01**2 / 4.0
    assert result["fan_mechanism_pressure"] == pytest.approx(24.0 * mp / 1.0**2, rel=1e-12)
    assert result["governing_mechanism"] == "fan"
    assert result["collapse_pressure"] == pytest.approx(24.0 * mp, rel=1e-12)


def test_yield_line_long_plate_switches_to_one_way_mechanism() -> None:
    mp = 235e6 * 0.01**2 / 4.0
    result = yield_line_collapse_pressure(4.0, 1.0, 0.01, 235e6)
    assert result["governing_mechanism"] == "strip"
    assert result["collapse_pressure"] == pytest.approx(8.0 * mp, rel=1e-12)
    clamped = yield_line_collapse_pressure(4.0, 1.0, 0.01, 235e6, boundary="clamped")
    assert clamped["strip_mechanism_pressure"] == pytest.approx(16.0 * mp, rel=1e-12)


def test_yield_line_mechanism_transition_is_flagged() -> None:
    transitional = yield_line_collapse_pressure(1.6, 1.0, 0.01, 235e6)
    assert transitional["governing_mechanism"] in ("fan", "strip", "transition")
    assert transitional["collapse_pressure"] > 0.0
    with pytest.raises(ValueError):
        yield_line_collapse_pressure(1.0, 1.0, 0.01, 235e6, boundary="free")


def test_ice_pressure_model_scales_with_area() -> None:
    model = IcePressureModel(reference_pressure=2.0e6, reference_area=1.0, exponent=-0.3)
    model.validate()
    assert model.pressure(1.0) == pytest.approx(2.0e6, rel=1e-12)
    assert model.pressure(8.0) == pytest.approx(2.0e6 * 8.0**-0.3, rel=1e-12)
    assert model.pressure(8.0) < model.pressure(1.0)
    with pytest.raises(ValueError):
        IcePressureModel(reference_pressure=1.0e6, reference_area=1.0, exponent=0.2).validate()


def test_iso19906_ice_belt_check_quantities_are_consistent() -> None:
    model = IcePressureModel(reference_pressure=1.5e6, reference_area=0.5, exponent=-0.3)
    result = iso19906_ice_belt_check(ICE_BELT, model)
    assert result["load_area"] == pytest.approx(ICE_BELT.span * ICE_BELT.spacing, rel=1e-12)
    assert result["ice_pressure"] == pytest.approx(model.pressure(result["load_area"]), rel=1e-12)
    assert result["allowable_utilisation"] == pytest.approx(
        result["maximum_bending_stress"] / (ICE_BELT.yield_stress), rel=1e-12)
    assert result["uls_utilisation"] == pytest.approx(
        result["ice_pressure"] / result["plastic_collapse_pressure"], rel=1e-12)
    # Doubling the pressure doubles both utilisations.
    doubled = iso19906_ice_belt_check(
        ICE_BELT, IcePressureModel(3.0e6, 0.5, -0.3))
    assert doubled["uls_utilisation"] == pytest.approx(2.0 * result["uls_utilisation"], rel=1e-12)
    with pytest.raises(ValueError):
        iso19906_ice_belt_check(ICE_BELT, model, load_area=10.0)


def _box_units(panel: StiffenedPanel, *, ultimate_stress: float | None = None):
    depth = 8.0
    units = []
    for name, y, area in (("deck", depth, 0.02), ("bottom", 0.0, 0.02),
                          ("side_port", depth / 2, 0.012), ("side_starboard", depth / 2, 0.012)):
        units.append(StructuralUnit(name, area, y, panel if ultimate_stress is None else None,
                                    ultimate_stress))
    return tuple(units)


def test_progressive_collapse_reaches_fully_plastic_moment_for_squash_units() -> None:
    sigma_y = 235e6
    units = _box_units(ICE_BELT, ultimate_stress=sigma_y)
    collapse = hull_girder_progressive_collapse(units, yield_stress=sigma_y, young=2.06e11)
    assert collapse["ultimate_moment"] == pytest.approx(
        collapse["upper_bound_fully_plastic_moment"], rel=5e-3)
    assert collapse["neutral_axis"] == pytest.approx(4.0, rel=1e-9)


def test_progressive_collapse_is_below_plastic_moment_for_panels() -> None:
    sigma_y = ICE_BELT.yield_stress
    units = _box_units(ICE_BELT)
    collapse = hull_girder_progressive_collapse(units, yield_stress=sigma_y, young=2.06e11)
    ratio = collapse["ultimate_moment"] / collapse["upper_bound_fully_plastic_moment"]
    # The compression side is limited by the panel ultimate while the tension
    # side still carries sigma_y, so the ratio sits above sigma_u/sigma_y.
    assert ICE_BELT.panel_ultimate_stress / sigma_y < ratio < 1.0
    # Elastic branch sanity: the first point must follow E I kappa.
    first = collapse["rows"][0]
    assert first["moment"] == pytest.approx(
        2.06e11 * collapse["section_inertia"] * first["curvature"], rel=1e-9)


def test_progressive_collapse_detects_the_post_peak_branch() -> None:
    units = _box_units(ICE_BELT)
    collapse = hull_girder_progressive_collapse(units, yield_stress=ICE_BELT.yield_stress,
                                                young=2.06e11)
    # With ideal-plastic limiting stresses the moment saturates on a plateau
    # instead of softening: the peak is reached and then held.
    assert collapse["plateau_reached"]
    index = collapse["peak_index"]
    assert 0 < index < len(collapse["rows"]) - 1
    assert collapse["rows"][index]["units_at_limit"]


def test_csr_hull_girder_check_uses_resistance_factor() -> None:
    sigma_y, young = ICE_BELT.yield_stress, 2.06e11
    units = _box_units(ICE_BELT)
    reference = hull_girder_progressive_collapse(units, yield_stress=sigma_y, young=young)
    ultimate = reference["ultimate_moment"]
    passing = csr_hull_girder_check(units, yield_stress=sigma_y, young=young,
                                    vertical_bending_moment=0.5 * ultimate)
    assert passing["passed"]
    assert passing["utilisation"] == pytest.approx(0.5, rel=1e-6)
    failing = csr_hull_girder_check(units, yield_stress=sigma_y, young=young,
                                    vertical_bending_moment=1.5 * ultimate)
    assert not failing["passed"]
    factored = csr_hull_girder_check(units, yield_stress=sigma_y, young=young,
                                     vertical_bending_moment=0.5 * ultimate,
                                     resistance_factor=1.25)
    assert factored["utilisation"] == pytest.approx(0.5 * 1.25, rel=1e-6)


def test_ice_wave_combination_matches_its_interaction_surface() -> None:
    model = IcePressureModel(reference_pressure=1.5e6, reference_area=0.5, exponent=-0.3)
    inertia = 2.3  # m^4 of the reference box; section modulus at deck level
    section_modulus = inertia / 4.0
    combined = ice_wave_combined_check(ICE_BELT, model, wave_bending_moment=1.0e6,
                                       section_modulus=section_modulus)
    in_plane = 1.0e6 / section_modulus / ICE_BELT.panel_ultimate_stress
    bending = combined["bending_stress"] / ICE_BELT.yield_stress
    assert combined["in_plane_utilisation"] == pytest.approx(in_plane, rel=1e-12)
    assert combined["combined_utilisation"] == pytest.approx(
        math.hypot(in_plane, bending), rel=1e-12)
    linear = ice_wave_combined_check(ICE_BELT, model, wave_bending_moment=1.0e6,
                                     section_modulus=section_modulus, interaction="linear")
    assert linear["combined_utilisation"] == pytest.approx(in_plane + bending, rel=1e-12)
    assert linear["combined_utilisation"] > combined["combined_utilisation"]
    with pytest.raises(ValueError):
        ice_wave_combined_check(ICE_BELT, model, wave_bending_moment=1.0,
                                section_modulus=section_modulus, interaction="cubic")


def test_panel_validation_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError):
        StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=-0.02, web_height=0.2,
                       web_thickness=0.01, flange_width=0.08,
                       flange_thickness=0.012).validate()
    with pytest.raises(ValueError):
        StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=0.02, web_height=0.2,
                       web_thickness=0.01, flange_width=0.004,
                       flange_thickness=0.012).validate()
    with pytest.raises(ValueError):
        StructuralUnit("deck", 0.01, 8.0).validate(235e6)