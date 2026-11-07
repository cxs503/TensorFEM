"""Tests for the corrosion-ice time-dependent ultimate strength module.

The gates mirror the acceptance criteria of the W10-N08 node:

* ``test_paik_anchor_rows_within_ten_percent`` -- field solver versus the
  Paik-type anchor over five DOP levels (< 10 %);
* ``test_uniform_mass_loss_discrete_matches_closed_form`` -- annual wastage
  integration versus the closed form (< 5 %);
* ``test_degeneracy_against_sibling_plate_buckling_node`` -- DOP = 0 versus the
  existing plate-buckling node, same geometry and load (< 2 %);
* ``test_history_is_monotone_for_every_realisation`` -- U(t) non-decreasing for
  every random realisation;
* ``test_monte_carlo_convergence_of_p50_strength`` -- 40 versus 80 realisations
  (the study script runs the 500/1000 pair and stores it in artifacts).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from tensorfem.marine_corrosion_ice_ultimate import (
    ARCTIC_STEELS, DNV_G101_ARCTIC_ICE_BELT, MELCHERS_IMMERSION, CorrosionLaw,
    IceDesignLoad, PanelConfig, PittingSpec, anchor_sensitivity,
    allowable_pitting_depth, allowable_service_life, build_pit_catalogue,
    calibrate_ice_pressure, clamped_plate_membrane_stress, decadal_history,
    faulkner_effective_width, gaussian_correlated_field, generate_pitting_field,
    ice_demand, inspection_schedule, intact_reference_check,
    monte_carlo_convergence, nodal_thickness_field, paik_cross_check,
    paik_pitted_reduction, panel_realization, plate_k, plate_ultimate_strength,
    realized_correlation_length, render_summary_markdown,
    thickness_field_buckling, thickness_field_buckling_fd,
    uniform_wastage_mass_loss,
)

warnings.simplefilter("ignore")


def _spec(**overrides) -> PittingSpec:
    base = dict(
        length_m=3.2, width_m=0.8, thickness_m=0.015, pitting_intensity=0.15,
        depth_model="depth_ratio", depth_ratio=0.5, reference_years=25.0,
        grid_x=20, grid_y=10,
    )
    base.update(overrides)
    return PittingSpec(**base)


def _config(**overrides) -> PanelConfig:
    base = dict(
        samples=3,
        pitting=_spec(),
        years=(0.0, 10.0, 25.0),
    )
    base.update(overrides)
    return PanelConfig(**base)


def _strength(config: PanelConfig, years: float):
    result, field = panel_realization(config=config, years=years)
    return result, field


# --------------------------------------------------------------------------- #
# corrosion kinetics
# --------------------------------------------------------------------------- #
def test_melchers_depth_follows_power_law():
    law = MELCHERS_IMMERSION
    assert law.depth_mm(0.0) == 0.0
    assert law.depth_mm(25.0) == pytest.approx(law.coefficient * 5.0, rel=1e-12)
    assert law.depth_mm(100.0) > law.depth_mm(25.0)


def test_corrosion_law_rejects_negative_years():
    with pytest.raises(ValueError):
        MELCHERS_IMMERSION.depth_mm(-1.0)


def test_dnv_law_has_zero_depth_inside_coating_life():
    law = DNV_G101_ARCTIC_ICE_BELT
    assert law.depth_mm(law.initiation_years * 0.5) == 0.0
    expected = law.coefficient * (25.0 - law.initiation_years)
    assert law.depth_mm(25.0) == pytest.approx(expected, rel=1e-12)


def test_uniform_mass_loss_discrete_matches_closed_form():
    for law in (MELCHERS_IMMERSION, DNV_G101_ARCTIC_ICE_BELT):
        result = uniform_wastage_mass_loss(
            years=25.0, area_m2=10.0, density_kg_m3=7850.0, law=law,
            time_step_years=1.0,
        )
        assert result.relative_error < 0.05
        assert result.analytic_kg > 0.0
        assert result.discrete_kg > 0.0


def test_uniform_mass_loss_error_shrinks_with_time_step():
    coarse = uniform_wastage_mass_loss(
        years=25.0, area_m2=1.0, density_kg_m3=7850.0, law=MELCHERS_IMMERSION,
        time_step_years=4.0,
    )
    fine = uniform_wastage_mass_loss(
        years=25.0, area_m2=1.0, density_kg_m3=7850.0, law=MELCHERS_IMMERSION,
        time_step_years=0.25,
    )
    assert fine.relative_error < coarse.relative_error


def test_uniform_mass_loss_rejects_non_law_input():
    with pytest.raises(TypeError):
        uniform_wastage_mass_loss(years=1.0, area_m2=1.0, density_kg_m3=7850.0,
                                  law={"coefficient": 1.0, "exponent": 1.0})


# --------------------------------------------------------------------------- #
# stochastic pitting field
# --------------------------------------------------------------------------- #
def test_correlated_field_is_unit_variance_and_correlated():
    rng = np.random.default_rng(3)
    field = gaussian_correlated_field(
        grid_x=64, grid_y=32, length_m=3.2, width_m=1.6,
        correlation_length_m=0.4, rng=rng,
    )
    assert field.shape == (32, 64)
    assert np.std(field) == pytest.approx(1.0, rel=0.05)
    measured = realized_correlation_length(field, 3.2 / 64)
    assert 0.2 < measured < 0.8


def test_correlation_length_increases_with_target_length():
    short = gaussian_correlated_field(
        grid_x=64, grid_y=32, length_m=3.2, width_m=1.6,
        correlation_length_m=0.2, rng=np.random.default_rng(11),
    )
    long = gaussian_correlated_field(
        grid_x=64, grid_y=32, length_m=3.2, width_m=1.6,
        correlation_length_m=0.8, rng=np.random.default_rng(11),
    )
    assert (realized_correlation_length(long, 3.2 / 64)
            > realized_correlation_length(short, 3.2 / 64))


def test_correlated_field_rejects_bad_inputs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        gaussian_correlated_field(grid_x=8, grid_y=8, length_m=1.0, width_m=1.0,
                                  correlation_length_m=-0.1, rng=rng)
    with pytest.raises(ValueError):
        gaussian_correlated_field(grid_x=8, grid_y=8, length_m=1.0, width_m=-1.0,
                                  correlation_length_m=0.1, rng=rng)


def test_pit_census_moments_match_the_targets():
    spec = _spec(pitting_intensity=0.2)
    catalogue = build_pit_catalogue(spec, rng=np.random.default_rng(5))
    assert catalogue["diameter"].size > 100
    assert np.mean(catalogue["diameter"]) == pytest.approx(0.030, rel=0.10)
    assert np.mean(catalogue["depth_factor"]) == pytest.approx(1.0, rel=0.08)
    assert np.mean(catalogue["initiation_years"]) == pytest.approx(
        spec.initiation_window_years / 2.0, rel=0.15)


@pytest.mark.parametrize("intensity", [0.05, 0.15, 0.30])
def test_realised_dop_matches_requested_intensity(intensity):
    spec = _spec(pitting_intensity=intensity)
    field = generate_pitting_field(spec=spec, years=30.0, law=MELCHERS_IMMERSION)
    assert field.dop_raster == pytest.approx(intensity, abs=0.06)


def test_pit_depth_is_volume_consistent_with_the_corrosion_law():
    law = MELCHERS_IMMERSION
    spec = _spec(depth_model="corrosion_law_volume", pitting_intensity=0.2)
    field = generate_pitting_field(spec=spec, years=25.0, law=law)
    removed = 1.0 - field.net_section_ratio
    assert removed * spec.thickness_m * 1000.0 == pytest.approx(
        law.depth_mm(25.0), abs=0.25 * law.depth_mm(25.0))


def test_field_grows_and_strength_falls_with_age():
    config = _config(pitting=_spec(depth_model="corrosion_law_volume"),
                     years=(0.0, 10.0, 20.0, 30.0))
    previous_thickness = math.inf
    previous_strength = math.inf
    previous_dop = -1.0
    for years in config.years:
        result, field = _strength(config, years)
        thickness = float(np.mean(np.asarray(field.thickness_grid)))
        assert field.dop_raster >= previous_dop - 1e-12
        assert thickness <= previous_thickness + 1e-12
        assert result.sigma_ultimate_mpa <= previous_strength + 1e-9
        previous_dop = field.dop_raster
        previous_thickness = thickness
        previous_strength = result.sigma_ultimate_mpa


def test_pitting_spec_validates_its_inputs():
    with pytest.raises(ValueError):
        _spec(pitting_intensity=1.2).validated()
    with pytest.raises(ValueError):
        _spec(depth_ratio=1.0).validated()
    with pytest.raises(ValueError):
        _spec(depth_model="nonsense").validated()
    with pytest.raises(TypeError):
        PittingSpec(length_m=1.0, width_m=1.0, thickness_m=0.01,
                    seed=True).validated()
    with pytest.raises(ValueError):
        _spec(grid_x=2).validated()


# --------------------------------------------------------------------------- #
# field-resolved buckling and ultimate strength
# --------------------------------------------------------------------------- #
def test_uniform_field_buckling_equals_the_classical_value():
    young, poisson, thickness = 206.0e9, 0.30, 0.015
    nodal = np.full((21, 41), thickness)
    result = thickness_field_buckling(
        thickness=nodal, length_m=3.2, width_m=0.8, young=young, poisson=poisson,
    )
    classical = (4.0 * math.pi**2 * young * thickness**2
                 / (12.0 * (1.0 - poisson**2) * 0.8**2))
    assert result.critical_stress == pytest.approx(classical, rel=1e-8)
    assert result.mean_thickness_m == pytest.approx(thickness, rel=1e-12)


def test_buckling_basis_convergence_on_a_pitted_field():
    spec = _spec(pitting_intensity=0.25)
    field = generate_pitting_field(spec=spec, years=25.0, law=MELCHERS_IMMERSION)
    nodal = nodal_thickness_field(field.thickness_effective_grid)
    values = [thickness_field_buckling(
        thickness=nodal, length_m=3.2, width_m=0.8, young=206e9, poisson=0.3,
        half_waves_x=hx, half_waves_y=hy,
    ).critical_stress for hx, hy in ((4, 2), (8, 3), (12, 4))]
    assert all(earlier > later for earlier, later in zip(values, values[1:]))
    assert abs(values[-1] / values[0] - 1.0) < 0.10


def test_buckling_quadrature_convergence():
    spec = _spec(pitting_intensity=0.25)
    field = generate_pitting_field(spec=spec, years=25.0, law=MELCHERS_IMMERSION)
    nodal = nodal_thickness_field(field.thickness_effective_grid)
    coarse = thickness_field_buckling(
        thickness=nodal, length_m=3.2, width_m=0.8, young=206e9, poisson=0.3,
        quadrature=(33, 9),
    ).critical_stress
    fine = thickness_field_buckling(
        thickness=nodal, length_m=3.2, width_m=0.8, young=206e9, poisson=0.3,
        quadrature=(65, 17),
    ).critical_stress
    assert abs(coarse / fine - 1.0) < 0.02


def test_finite_difference_and_modal_routes_agree_on_a_smooth_field():
    rng = np.random.default_rng(7)
    base = 0.015 * (1.0 - 0.05 * rng.random((21, 41)))
    modal = thickness_field_buckling(
        thickness=base, length_m=3.2, width_m=0.8, young=206e9, poisson=0.3,
        half_waves_x=8, half_waves_y=3, quadrature=(41, 21),
    )
    discrete = thickness_field_buckling_fd(
        thickness=base, length_m=3.2, width_m=0.8, young=206e9, poisson=0.3,
    )
    assert abs(modal.critical_stress / discrete.critical_stress - 1.0) < 0.05


def test_buckling_rejects_nonphysical_fields():
    with pytest.raises(ValueError):
        thickness_field_buckling(thickness=np.zeros((5, 5)), length_m=1.0,
                                 width_m=1.0, young=206e9, poisson=0.3)
    with pytest.raises(ValueError):
        thickness_field_buckling(thickness=np.full((2, 2), 0.01), length_m=1.0,
                                 width_m=1.0, young=206e9, poisson=0.3)
    with pytest.raises(ValueError):
        thickness_field_buckling(thickness=np.full((6, 6), 0.01), length_m=1.0,
                                 width_m=1.0, young=206e9, poisson=0.6)


def test_faulkner_effective_width_limits():
    assert faulkner_effective_width(1.0) == 1.0
    assert faulkner_effective_width(0.5) == 1.0
    assert faulkner_effective_width(2.0) == pytest.approx(2.0 / 2.0 - 1.0 / 4.0)
    assert faulkner_effective_width(4.0) < faulkner_effective_width(2.0)
    with pytest.raises(ValueError):
        faulkner_effective_width(0.0)


def test_paik_reduction_is_monotone_and_bounded():
    values = [paik_pitted_reduction(level) for level in (0.0, 0.1, 0.2, 0.3)]
    assert values[0] == 1.0
    assert all(earlier > later for earlier, later in zip(values, values[1:]))
    assert paik_pitted_reduction(1.0) >= 0.0
    with pytest.raises(ValueError):
        paik_pitted_reduction(1.5)
    with pytest.raises(ValueError):
        paik_pitted_reduction(-0.1)


def test_anchor_sensitivity_spans_the_coefficient_uncertainty():
    rows = anchor_sensitivity([0.0, 0.3])
    assert len(rows) == 6
    assert max(row["reduction"][1] for row in rows) < 1.0
    assert min(row["reduction"][1] for row in rows) > 0.5


def test_degeneracy_against_sibling_plate_buckling_node():
    report = intact_reference_check(config=_config())
    assert report["relative_error_vs_sibling"] < 0.02
    assert report["relative_error_vs_navier"] < 0.02
    assert report["half_waves"][0] > 0


def test_paik_anchor_rows_within_ten_percent():
    rows = paik_cross_check(config=_config(), dop_levels=(0.0, 0.075, 0.15, 0.225, 0.30))
    assert len(rows) == 5
    assert rows[0]["reduction_computed"] == pytest.approx(1.0, abs=2e-3)
    assert max(row["relative_error"] for row in rows) < 0.10
    assert all(row["reduction_computed"] <= 1.0 + 1e-9 for row in rows)


def test_ultimate_strength_falls_with_pit_depth():
    shallow, _ = _strength(_config(pitting=_spec(depth_ratio=0.25)), 25.0)
    deep, _ = _strength(_config(pitting=_spec(depth_ratio=0.75)), 25.0)
    assert deep.sigma_ultimate_mpa < shallow.sigma_ultimate_mpa
    assert shallow.sigma_ultimate_mpa <= shallow.sigma_intact_mpa + 1e-9


def test_net_section_cap_engages_for_severe_wastage():
    config = _config(pitting=_spec(pitting_intensity=0.6, depth_ratio=0.9))
    result, _ = _strength(config, 25.0)
    assert result.sigma_ultimate_mpa <= min(result.sigma_global_mpa,
                                            result.sigma_net_section_mpa) + 1e-9
    assert result.net_section_ratio < 1.0


def test_plate_k_matches_the_classical_relation():
    factor = plate_k(0.3)
    assert factor == pytest.approx(4.0 * math.pi**2 / (12.0 * (1.0 - 0.09)), rel=1e-12)
    with pytest.raises(ValueError):
        plate_k(0.6)


def test_ultimate_strength_rejects_wrong_field_type():
    with pytest.raises(TypeError):
        plate_ultimate_strength(field_values={"thickness_grid": np.ones((2, 2))},
                                young=206e9, poisson=0.3, yield_stress_mpa=355.0,
                                length_m=3.2, width_m=0.8)


# --------------------------------------------------------------------------- #
# ice demand and membrane screening
# --------------------------------------------------------------------------- #
def test_ice_pressure_area_law_reduces_pressure_with_area():
    small = IceDesignLoad(contact_area_m2=0.5).pressure_mpa()
    reference = IceDesignLoad(contact_area_m2=1.0).pressure_mpa()
    large = IceDesignLoad(contact_area_m2=4.0).pressure_mpa()
    assert small > reference > large
    assert reference == pytest.approx(2.5, rel=1e-12)


def test_ice_demand_scales_inversely_with_thickness():
    config = _config()
    thick = ice_demand(config=config, thickness_m=0.015)
    thin = ice_demand(config=config, thickness_m=0.012)
    assert thin.membrane_stress_mpa > thick.membrane_stress_mpa
    assert thin.membrane_stress_mpa / thick.membrane_stress_mpa == pytest.approx(
        0.015 / 0.012, rel=1e-12)


def test_membrane_screening_converges_in_shape_count():
    single = clamped_plate_membrane_stress(
        pressure_pa=2.5e6, radius_m=0.4, thickness_m=0.015, young=206e9,
        poisson=0.3, shapes=1,
    )
    triple = clamped_plate_membrane_stress(
        pressure_pa=2.5e6, radius_m=0.4, thickness_m=0.015, young=206e9,
        poisson=0.3, shapes=3,
    )
    # the trial family is deliberately small, so the refined solution is compared
    # with a loose tolerance and the family spread is reported by the study
    assert triple.membrane_stress_mpa == pytest.approx(single.membrane_stress_mpa,
                                                       rel=1e-3)
    assert 50.0 < triple.membrane_stress_mpa < 400.0
    assert triple.deflection_to_thickness > 0.1


def test_membrane_screening_zero_pressure_is_zero():
    result = clamped_plate_membrane_stress(
        pressure_pa=0.0, radius_m=0.4, thickness_m=0.015, young=206e9, poisson=0.3,
    )
    assert result.membrane_stress_mpa == 0.0


def test_arctic_steel_temperature_correction():
    grade = ARCTIC_STEELS["EH36"]
    assert grade.yield_stress_mpa == pytest.approx(355.0)
    cold = grade.yield_stress_at(-40.0)
    warm = grade.yield_stress_at(20.0)
    assert cold > warm
    assert cold / warm == pytest.approx(1.0 + 60.0 * grade.temperature_slope_per_c,
                                        rel=1e-12)
    with pytest.raises(ValueError):
        grade.yield_stress_at(20000.0)


def test_calibrated_ice_pressure_reproduces_the_target_utilization():
    config = calibrate_ice_pressure(config=_config(), utilization=0.55)
    result, _ = _strength(config, 0.0)
    demand = ice_demand(config=config, thickness_m=config.thickness_m)
    assert demand.membrane_stress_mpa / result.sigma_ultimate_mpa == pytest.approx(
        0.55, rel=0.02)


# --------------------------------------------------------------------------- #
# history, inverse problem and report rendering
# --------------------------------------------------------------------------- #
def test_history_is_monotone_for_every_realisation():
    history = decadal_history(config=_config(samples=6), samples=6)
    assert history["monotonicity"]["violations"] == 0
    assert history["monotonicity"]["monotone_sample_fraction"] == 1.0
    utilization = [row["utilization_p50"] for row in history["rows"]]
    assert all(earlier <= later + 1e-12 for earlier, later in zip(utilization, utilization[1:]))
    for row in history["rows"]:
        assert row["sigma_ultimate_mpa_p05"] <= row["sigma_ultimate_mpa_p50"]
        assert row["sigma_ultimate_mpa_p50"] <= row["sigma_ultimate_mpa_p95"]


def test_history_utilization_equals_the_stress_ratio():
    # the p50 of a ratio is not the ratio of p50s, but for the small scatter of
    # this module they must stay within two percent of each other.
    history = decadal_history(config=_config(samples=3), samples=3)
    for row in history["rows"]:
        assert row["utilization_p50"] == pytest.approx(
            row["sigma_ice_mpa_p50"] / row["sigma_ultimate_mpa_p50"], rel=0.02)


def test_monte_carlo_convergence_of_p50_strength():
    report = monte_carlo_convergence(config=_config(samples=2), levels=(40, 80))
    assert report["p50_relative_difference"] < 0.01
    assert report["passed"] is True
    assert report["levels"][80]["std"] >= 0.0


def test_allowable_life_statuses():
    config = _config(samples=3)
    history = decadal_history(config=config, samples=3)
    beyond = allowable_service_life(history=history, config=config)
    assert beyond["status"] in {"beyond_horizon", "governed", "already_below_target"}
    strict = PanelConfig(**{**config.__dict__, "target_safety_factor": 40.0})
    blocked = allowable_service_life(history=history, config=strict)
    assert blocked["status"] == "already_below_target"
    assert blocked["allowable_years"] == 0.0


def test_allowable_pitting_depth_is_governed_and_consistent():
    config = _config(samples=2)
    report = allowable_pitting_depth(config=config, design_years=25.0,
                                     iterations=12)
    assert report["status"] in {"governed", "not_governed", "target_unreachable_even_intact"}
    if report["status"] == "governed":
        assert 0.0 < report["allowable_mean_depth_mm"] < 0.97 * 15.0
        assert report["allowable_mean_rate_mm_per_year"] > 0.0
        assert report["depth_margin_factor"] is not None


def test_inspection_schedule_columns_and_interval_cap():
    config = _config(samples=3)
    history = decadal_history(config=config, samples=3)
    schedule = inspection_schedule(history=history, config=config)
    assert len(schedule) == len(history["rows"])
    for row in schedule:
        assert 0.0 <= row["next_inspection_interval_years"] <= 5.0
        assert row["design_ice_demand_mpa"] > 0.0
        assert row["action"] in {"routine", "renew_or_dock", "increase_frequency",
                                 "close_out"}


def test_render_summary_markdown_contains_the_history():
    config = _config(samples=2)
    result, _ = _strength(config, 25.0)
    payload = {
        "panel": {"steel_grade": "EH36", "length_m": 3.2, "width_m": 0.8,
                  "thickness_mm": 15.0, "plate_slenderness_beta": 2.31,
                  "design_temperature_c": -40.0,
                  "yield_stress_mpa_design": result.sigma_intact_mpa},
        "ice": {"design_pressure_mpa": 2.5,
                "membrane_stress_intact_mpa": {"quadratic": 124.8}},
        "history": {"rows": [{"years": 0.0, "dop_p50": 0.0,
                              "sigma_ultimate_mpa_p50": 262.4,
                              "utilization_p50": 0.397}]},
        "criteria": {"a_paik_anchor": {"passed": True}},
    }
    text = render_summary_markdown(payload)
    assert "EH36" in text
    assert "a_paik_anchor" in text
    assert "| 0 |" in text


def test_panel_config_validates_its_inputs():
    with pytest.raises(ValueError):
        PanelConfig(steel_grade="A36").validated()
    with pytest.raises(ValueError):
        PanelConfig(years=(0.0, 5.0, 5.0)).validated()
    with pytest.raises(ValueError):
        PanelConfig(samples=0).validated()
    with pytest.raises(ValueError):
        PanelConfig(reliability_fractile=0.75).validated()
    with pytest.raises(TypeError):
        PanelConfig(samples=2.5).validated()


def test_corrosion_law_rejects_nonpositive_exponent():
    law = CorrosionLaw(name="bad", coefficient=0.1, exponent=0.0)
    with pytest.raises(ValueError):
        uniform_wastage_mass_loss(years=5.0, area_m2=1.0, density_kg_m3=7850.0,
                                  law=law)


def test_field_seed_reproducibility():
    first = generate_pitting_field(spec=_spec(seed=42), years=25.0,
                                   law=MELCHERS_IMMERSION)
    second = generate_pitting_field(spec=_spec(seed=42), years=25.0,
                                    law=MELCHERS_IMMERSION)
    assert first.dop_raster == pytest.approx(second.dop_raster, rel=1e-12)
    assert float(np.mean(np.asarray(first.thickness_grid))) == pytest.approx(
        float(np.mean(np.asarray(second.thickness_grid))), rel=1e-12)