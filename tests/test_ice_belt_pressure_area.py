"""Acceptance tests for ``tensorfem.ice_belt_pressure_area`` (H4 round 1).

Each test names the campaign criterion it guards:

* ``test_law_*``              -- pressure-area law and its input guards (criterion 1)
* ``test_route_*``            -- the independent evaluation routes (criterion 1)
* ``test_quadrature_*``       -- force/pressure integral identity (criterion 1)
* ``test_rule_table_*``       -- rule-table comparator, incl. the refuse-to-claim path
* ``test_navier_*``           -- Navier patch series vs the classical uniform series
* ``test_panel_*``            -- MITC4 / Q4-SRI / Navier cross-validation (criterion 2)
* ``test_zone_*``             -- hull-area partition, belt, ordering, class registry
                                 (criterion 3)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.ice_belt_pressure_area import (
    DEMO_CLASS_FACTORS,
    DEMO_ZONE_FACTORS,
    DEMO_ZONE_FRACTIONS,
    DesignPatch,
    HullZone,
    IceBelt,
    PressureAreaLaw,
    RulePressureTable,
    check_belt_containment,
    check_class_ordering,
    check_curve_routes,
    check_design_pressure_zones,
    check_force_quadrature,
    check_load_linearity,
    check_pressure_provenance,
    check_zone_partition,
    compare_with_rule_table,
    design_pressure_zones,
    elastic_load_factor,
    fit_loglog_exponent,
    ice_belt_extent,
    navier_patch_centre,
    panel_cross_validation,
    panel_patches,
    partition_hull_zones,
    pressure_area_curve_rows,
    radial_field_mean_pressure,
    run_h4_verification,
)
from tensorfem.ice_belt_collapse import E_STEEL, NU_STEEL, navier_ssss_centre

AREAS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def demo_law(**kw) -> PressureAreaLaw:
    base = dict(p_ref=2.0e6, provenance="DEMONSTRATION test anchor")
    base.update(kw)
    return PressureAreaLaw(**base)


# --------------------------------------------------------------------------- #
# the law itself
# --------------------------------------------------------------------------- #
def test_law_pressure_is_monotone_decreasing_and_scales_with_cd_cf():
    law = demo_law()
    p = law.pressure(AREAS)
    assert np.all(np.diff(p) < 0.0)
    assert float(law.pressure(1.0)) == pytest.approx(2.0e6, rel=1e-15)
    doubled = demo_law(c_d=2.0)
    assert doubled.pressure(AREAS) == pytest.approx(2.0 * p, rel=1e-15)
    assert demo_law(c_f=0.5).pressure(AREAS) == pytest.approx(0.5 * p, rel=1e-15)


def test_law_design_force_grows_with_area_and_inverse_round_trips():
    law = demo_law()
    force = law.design_force(AREAS)
    assert np.all(np.diff(force) > 0.0)
    back = law.area_for_force(force)
    assert back == pytest.approx(np.asarray(AREAS), rel=1e-12)


def test_law_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        PressureAreaLaw(p_ref=0.0)
    with pytest.raises(ValueError):
        PressureAreaLaw(p_ref=2.0e6, exponent=1.0)
    with pytest.raises(ValueError):
        PressureAreaLaw(p_ref=2.0e6, area_ref=-1.0)
    with pytest.raises(ValueError):
        demo_law().pressure(0.0)
    with pytest.raises(ValueError):
        demo_law().pressure(-1.0)


def test_pressure_provenance_gate_fails_closed():
    law = demo_law()
    check_pressure_provenance(law)                       # not gated by default
    with pytest.raises(ValueError, match="not rule data"):
        check_pressure_provenance(law, require_design_data=True)
    rule = PressureAreaLaw(p_ref=2.0e6, provenance="IACS UR I2 Rev.1 (2016) 3.2")
    check_pressure_provenance(rule, require_design_data=True)


# --------------------------------------------------------------------------- #
# independent routes
# --------------------------------------------------------------------------- #
def test_route_agreement_is_machine_precision():
    law = demo_law()
    rows = pressure_area_curve_rows(law, AREAS)
    assert len(rows) == len(AREAS)
    for key in ("rel_diff_logspace", "rel_diff_marginal", "rel_diff_radial_mean",
                "rel_diff_area_inverse", "invariant_rel_diff"):
        assert max(abs(r[key]) for r in rows) < 1e-9, key


def test_marginal_route_needs_the_one_minus_alpha_factor():
    # The trap: dF/dA = (1 - alpha) p, so using dF/dA directly is off by (1-alpha).
    law = demo_law()
    area = np.array([1.0])
    naive = float(law.force_derivative(area)[0] / law.pressure(area)[0])
    assert naive == pytest.approx(1.0 - law.exponent, rel=1e-8)
    assert float(law.marginal_pressure(area)[0]) == pytest.approx(
        float(law.pressure(area)[0]), rel=1e-9)


def test_route_fit_recovers_the_exponent():
    law = demo_law(exponent=0.3)
    fit = fit_loglog_exponent(law, AREAS)
    assert fit["fitted_exponent"] == pytest.approx(-0.3, abs=1e-12)
    assert fit["max_abs_log_residual"] < 1e-12
    other = demo_law(exponent=0.6)
    assert fit_loglog_exponent(other, AREAS)["fitted_exponent"] == pytest.approx(-0.6, abs=1e-12)


def test_radial_mean_route_matches_the_law():
    law = demo_law()
    got = radial_field_mean_pressure(np.asarray(AREAS), law)
    assert got == pytest.approx(law.pressure(np.asarray(AREAS)), rel=1e-10)


# --------------------------------------------------------------------------- #
# force / pressure integral identity
# --------------------------------------------------------------------------- #
def test_quadrature_matches_the_force_difference_and_refinement_helps():
    law = demo_law()
    res = check_force_quadrature(law, AREAS[0], AREAS[-1])
    assert res["passed"] and abs(res["rel_error"]) < 1e-10
    # a single panel over a 320:1 area ratio is visibly worse than 32 panels
    assert abs(res["single_panel_rel_error"]) > abs(res["rel_error"])


# --------------------------------------------------------------------------- #
# rule-table comparator
# --------------------------------------------------------------------------- #
def _table_from_law(law: PressureAreaLaw, is_rule_data: bool) -> RulePressureTable:
    areas = (0.1, 0.5, 1.0, 2.0)
    pressures = tuple(float(p) / 1.0e6 for p in law.pressure(np.asarray(areas)))
    return RulePressureTable(rows=tuple(zip(areas, pressures)),
                            provenance="DEMONSTRATION table generated from the law",
                            clause="mechanism check only",
                            is_rule_data=is_rule_data)


def test_rule_table_comparator_refuses_demonstration_data():
    law = demo_law()
    with pytest.raises(ValueError, match="not flagged as rule data"):
        compare_with_rule_table(law, _table_from_law(law, is_rule_data=False))
    out = compare_with_rule_table(law, _table_from_law(law, is_rule_data=False),
                                  require_rule_data=False)
    assert out["passed"] and out["max_abs_rel_error"] < 1e-12


def test_rule_table_comparator_reports_point_by_point_errors():
    law = demo_law()
    table = _table_from_law(law, is_rule_data=True)
    out = compare_with_rule_table(law, table, tolerance=1e-12)
    assert out["passed"]
    assert [r["area_m2"] for r in out["rows"]] == [0.1, 0.5, 1.0, 2.0]
    # a 6 % offset table must be caught at 5 % tolerance
    shifted = RulePressureTable(
        rows=tuple((a, p * 1.06) for a, p in table.rows),
        provenance="DEMONSTRATION shifted table", is_rule_data=True)
    out2 = compare_with_rule_table(law, shifted, tolerance=0.05)
    assert not out2["passed"]
    assert out2["max_abs_rel_error"] == pytest.approx(0.06 / 1.06, rel=1e-9)
    assert all(r["rel_error"] < 0.0 for r in out2["rows"])   # got is *below* ref


def test_rule_table_validation_rejects_bad_tables():
    with pytest.raises(ValueError):
        RulePressureTable(rows=((1.0, 2.0),), provenance="x")
    with pytest.raises(ValueError):
        RulePressureTable(rows=((1.0, 2.0), (0.5, 3.0)), provenance="x")
    with pytest.raises(ValueError):
        RulePressureTable(rows=((1.0, 2.0), (2.0, 1.0)), provenance="  ")


# --------------------------------------------------------------------------- #
# zone division
# --------------------------------------------------------------------------- #
def test_zone_partition_tiles_without_gaps_or_overlaps():
    zones = partition_hull_zones(120.0, DEMO_ZONE_FRACTIONS, DEMO_ZONE_FACTORS)
    assert [z.key for z in zones] == ["stern", "midbody", "bow_intermediate", "bow"]
    check = check_zone_partition(zones, 120.0)
    assert check["passed"]
    assert check["max_boundary_gap"] <= 1e-12
    assert sum(z.length for z in zones) == pytest.approx(120.0, rel=1e-15)


def test_zone_partition_rejects_bad_boundaries():
    with pytest.raises(ValueError):
        partition_hull_zones(120.0, (("a", 0.5), ("b", 0.9)))      # does not reach L
    with pytest.raises(ValueError):
        partition_hull_zones(120.0, (("a", 0.5), ("b", 0.5), ("c", 1.0)))  # empty zone


def test_zone_partition_detects_a_hand_built_overlap():
    zones = (HullZone("a", 0.0, 60.0, 1.0), HullZone("b", 55.0, 120.0, 1.0))
    check = check_zone_partition(zones, 120.0)
    assert not check["Z1_length_tiling"] and not check["Z2_contiguity"]
    assert not check["passed"]


def test_belt_extent_and_containment():
    belt = ice_belt_extent(7.0, 2.9, 2.6)
    assert (belt.z_lower, belt.z_upper) == pytest.approx((4.1, 9.6))
    assert check_belt_containment(belt, depth=12.0)["passed"]
    assert not check_belt_containment(belt, depth=9.0)["passed"]
    with pytest.raises(ValueError):
        ice_belt_extent(2.0, 2.9, 2.6)      # belt would start below the base line
    with pytest.raises(ValueError):
        IceBelt(5.0, 5.0)


def test_panel_patches_ordering_is_enforced():
    patches = panel_patches(0.35, 1.6, 3.2, 6.4)
    assert [p.label for p in patches] == ["plate_field", "stiffener", "primary_member"]
    assert [p.area for p in patches] == pytest.approx([0.56, 1.12, 2.24])
    with pytest.raises(ValueError):
        panel_patches(0.35, 3.2, 1.6, 6.4)
    with pytest.raises(ValueError):
        DesignPatch("x", 0.0)


def test_design_pressure_zones_ordering_and_factor_linearity():
    law = demo_law()
    zones = partition_hull_zones(120.0, DEMO_ZONE_FRACTIONS, DEMO_ZONE_FACTORS)
    patches = panel_patches(0.35, 1.6, 3.2, 6.4)
    table = design_pressure_zones(law, zones, patches)
    checks = check_design_pressure_zones(table, law, zones, patches)
    assert checks["passed"]
    # bow carries more pressure than stern by exactly the factor ratio
    bow = next(r for r in table["rows"] if r["zone"] == "bow" and r["patch"] == "plate_field")
    stern = next(r for r in table["rows"] if r["zone"] == "stern" and r["patch"] == "plate_field")
    assert bow["pressure_pa"] / stern["pressure_pa"] == pytest.approx(
        DEMO_ZONE_FACTORS["bow"] / DEMO_ZONE_FACTORS["stern"], rel=1e-14)


def test_design_pressure_zone_audit_catches_a_corrupted_table():
    law = demo_law()
    zones = partition_hull_zones(120.0, DEMO_ZONE_FRACTIONS, DEMO_ZONE_FACTORS)
    patches = panel_patches(0.35, 1.6, 3.2, 6.4)
    table = design_pressure_zones(law, zones, patches)
    table["rows"][0]["pressure_pa"] *= 1.5        # corrupt one cell
    checks = check_design_pressure_zones(table, law, zones, patches)
    assert not checks["passed"]
    assert not checks["Z8_force_accounting"]


def test_class_ordering_follows_the_injected_registry_and_catches_inversion():
    ranks = {"PC1": 12, "PC4": 9, "PC7": 6}
    good = {k: 2.0e6 * DEMO_CLASS_FACTORS[k] for k in ranks}
    assert check_class_ordering(good, ranks)["passed"]
    bad = dict(good)
    bad["PC1"], bad["PC7"] = good["PC7"], good["PC1"]     # inverted
    assert not check_class_ordering(bad, ranks)["passed"]
    with pytest.raises(ValueError):
        check_class_ordering({"PC1": 1.0}, ranks)


# --------------------------------------------------------------------------- #
# the Navier patch series
# --------------------------------------------------------------------------- #
def test_navier_patch_series_reduces_to_the_uniform_series():
    span, t = 1.0, 0.01
    uniform = navier_patch_centre(span, span, t, E_STEEL, NU_STEEL, 1.0e4)
    reference = navier_ssss_centre(span, span, t, E_STEEL, NU_STEEL, 1.0e4)
    assert uniform == pytest.approx(reference, rel=1e-10)


def test_navier_patch_series_scales_linearly_and_is_below_the_uniform_value():
    span, t, p = 1.0, 0.01, 1.0e4
    full = navier_patch_centre(span, span, t, E_STEEL, NU_STEEL, p)
    half = navier_patch_centre(span, span, t, E_STEEL, NU_STEEL, 2.0 * p)
    assert half == pytest.approx(2.0 * full, rel=1e-12)
    patch = navier_patch_centre(span, span, t, E_STEEL, NU_STEEL, p, patch_area=0.25)
    # a smaller loaded area at the same pressure deflects the centre less
    assert patch < full
    # convergence of the series
    coarse = navier_patch_centre(span, span, t, E_STEEL, NU_STEEL, p, patch_area=0.25,
                                 terms=41)
    assert patch == pytest.approx(coarse, rel=1e-3)


# --------------------------------------------------------------------------- #
# panel cross-validation (criterion 2)
# --------------------------------------------------------------------------- #
def test_elastic_load_factor_targets_the_linear_range():
    law = demo_law()
    lf = elastic_load_factor(law, span_x=1.0, span_y=1.0, thickness=0.01)
    assert 0.0 < lf["load_factor"] < 1.0
    assert lf["p_validation_pa"] < lf["p_design_pa"]


@pytest.mark.parametrize("patch_fraction", [None, 0.25])
def test_panel_cross_validation_three_routes_within_ten_percent(patch_fraction):
    law = demo_law()
    kwargs = dict(span_x=1.0, span_y=1.0, thickness=0.010)
    lf = elastic_load_factor(law, patch_fraction=patch_fraction, **kwargs)
    out = panel_cross_validation(law, patch_fraction=patch_fraction,
                                 load_factor=lf["load_factor"],
                                 n_mitc=11, n_sri=12, **kwargs)
    assert out["passed"], out["pairwise_rel_diff"]
    assert out["max_pairwise_rel_diff"] < 0.10
    # the analytic route must sit between the two FE routes to within a few %
    assert out["pairwise_rel_diff"]["q4sri_vs_navier"] < 0.05


def test_panel_cross_validation_requires_a_square_panel():
    law = demo_law()
    with pytest.raises(ValueError):
        panel_cross_validation(law, span_x=1.0, span_y=0.8)


def test_panel_response_is_linear_at_the_validation_load():
    law = demo_law()
    lf = elastic_load_factor(law, span_x=1.0, span_y=1.0, thickness=0.010)
    lin = check_load_linearity(law, span_x=1.0, span_y=1.0, thickness=0.010,
                               load_factor=lf["load_factor"], n_mitc=9)
    assert lin["passed"] and lin["ratio"] == pytest.approx(2.0, rel=1e-4)


# --------------------------------------------------------------------------- #
# the whole H4 bundle
# --------------------------------------------------------------------------- #
def test_run_h4_verification_reports_partial_withheld_honestly():
    # small meshes for test speed; the artifact run uses the defaults
    bundle = run_h4_verification()
    assert bundle["criterion_2_panel_cross_validation"]["status"] == "PASS"
    assert bundle["criterion_3_design_pressure_zones"]["status"] in ("PASS",
                                                                    "PARTIAL_WITHHELD")
    # the rule-table comparison must be WITHHELD, never silently "passed"
    rule = bundle["criterion_1_pressure_area_curve"]["rule_table_comparison"]
    assert rule["status"] == "WITHHELD"
    assert "failure_reason" in rule and rule["fail_stage"]
    assert bundle["criterion_1_pressure_area_curve"]["status"] == "PARTIAL_WITHHELD"
    assert bundle["overall_status"] == "PARTIAL_WITHHELD"
    assert bundle["withheld_items"]
    assert len(bundle["curve_table"]) == 9
    assert len(bundle["pressure_zone_table"]) == 12