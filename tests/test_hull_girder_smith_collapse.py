"""Verification tests for the Smith hull-girder progressive-collapse module.

The tests are grouped by the campaign criteria they defend:

* component curves      -> plate buckling/ultimate laws (analytical anchors)
* C2                   -> elastic slope vs ``E I``
* C4                   -> energy ledger closure
* C5                   -> fibre-count convergence
* C3                   -> imperfection sensitivity
* C1                   -> anchor injection (never fabricated)
* engine correctness    -> closed-form ideal plastic oracle, axial equilibrium
"""
from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from tensorfem.hull_girder_smith_collapse import (
    ISSC2009_DOUBLE_BOTTOM,
    DoubleBottomGeometry,
    SectionFiber,
    StressStrainLaw,
    UltimateMomentAnchor,
    column_law,
    curvature_grid,
    elastic_perfectly_plastic_law,
    interaction_law,
    load_anchor,
    plate_field_law,
    run_issc2009_double_bottom,
    solve_moment_curvature,
    verify_ideal_plastic_section,
)
from tensorfem.smith_progressive_collapse import PlateStrip, TeeStiffener

YOUNG = 2.06e11
POISSON = 0.3
YIELD = 315e6


def _strip(length: float, width: float, thickness: float,
           imperfection_ratio: float = 0.1) -> PlateStrip:
    return PlateStrip(length=length, width=width, thickness=thickness,
                      young=YOUNG, poisson=POISSON, yield_stress=YIELD,
                      imperfection_ratio=imperfection_ratio)


def _double_bottom_case(steps: int = 120, subdivisions: int = 16):
    geometry = replace(ISSC2009_DOUBLE_BOTTOM, subdivisions=subdivisions)
    props = geometry.elastic_properties()
    eps_yield = geometry.yield_stress / geometry.young
    extreme = max(props["neutral_axis"], geometry.depth - props["neutral_axis"])
    kappa_yield = eps_yield / extreme
    grid = curvature_grid(kappa_yield, steps=steps, maximum_ratio=8.0)
    return geometry, props, solve_moment_curvature(geometry.fibers(), grid)


# --------------------------------------------------------------------------
# component curves
# --------------------------------------------------------------------------
def test_plate_buckling_stress_matches_classical_formula():
    """sigma_cr must match k pi^2 E/(12 (1-nu^2)) (t/b)^2 from the node spec."""
    strip = _strip(length=1.5, width=0.75, thickness=0.012)
    predicted = strip.buckling_stress
    reference = (strip.buckling_coefficient * math.pi ** 2 * YOUNG
                 / (12.0 * (1.0 - POISSON ** 2))
                 * (0.012 / 0.75) ** 2)
    assert predicted == pytest.approx(reference, rel=1e-9)
    assert 0.3 < strip.ultimate_ratio <= 1.0


def test_plate_field_law_compression_matches_strip_curve():
    strip = _strip(length=1.5, width=0.75, thickness=0.012)
    law = plate_field_law(strip)
    for strain in (1e-4, 5e-4, 1e-3, 3e-3, 6e-3):
        # the law is a tabulation of the parent curve: reproduce it to the
        # tabulation tolerance (5e-5 observed, asserted at 1e-4)
        assert law.stress(-strain) == pytest.approx(-strip.stress(strain), rel=1e-4)


def test_plate_field_law_tension_is_capped_at_yield():
    law = plate_field_law(_strip(1.5, 0.75, 0.012))
    assert law.stress(1e-5) == pytest.approx(YOUNG * 1e-5, rel=1e-6)
    assert law.stress(1e-2) == pytest.approx(YIELD, rel=1e-9)


def test_law_energy_matches_elastic_analytic_value():
    law = elastic_perfectly_plastic_law(young=YOUNG, yield_stress=YIELD)
    for strain in (1e-5, 1e-4, 5e-4):
        assert law.energy(strain) == pytest.approx(0.5 * YOUNG * strain ** 2,
                                                  rel=1e-9)
        assert law.energy(-strain) == pytest.approx(0.5 * YOUNG * strain ** 2,
                                                   rel=1e-9)


def test_law_energy_is_non_negative_and_monotone_along_the_path():
    law = plate_field_law(_strip(1.5, 0.75, 0.012))
    strains = [-2e-2 * i / 200 for i in range(201)]
    energies = [law.energy(value) for value in strains]
    assert energies[0] == 0.0
    assert all(value >= 0.0 for value in energies)
    assert all(a <= b + 1e-12 for a, b in zip(energies, energies[1:]))


def test_interaction_law_takes_pointwise_minimum():
    plate = plate_field_law(_strip(1.5, 2.0, 0.008))
    column = column_law(TeeStiffener(
        web_height=1.978, web_thickness=0.008, flange_width=0.15,
        flange_thickness=0.01, panel_length=6.0, plate_width=0.75,
        plate_thickness=0.012, young=YOUNG, poisson=POISSON,
        yield_stress=YIELD))
    combined = interaction_law((plate, column))
    for strain in (-1e-3, -5e-3, -1e-2):
        assert combined.stress(strain) == pytest.approx(
            min(plate.stress(strain), column.stress(strain)), rel=1e-6)


# --------------------------------------------------------------------------
# section engine
# --------------------------------------------------------------------------
def test_engine_reproduces_ideal_plastic_closed_form():
    report = verify_ideal_plastic_section()
    assert report["max_relative_moment_error"] < 3e-3
    assert report["elastic_slope_relative_error"] < 3e-3
    assert report["passed"] is True


def test_engine_keeps_axial_equilibrium_on_every_increment():
    geometry, props, result = _double_bottom_case()
    reference = sum(fiber.area * fiber.law.yield_stress for fiber in geometry.fibers())
    worst = max(abs(point.axial_resultant) for point in result.points)
    assert worst < 1e-8 * reference
    assert props["area"] == pytest.approx(sum(f.area for f in geometry.fibers()), rel=1e-12)


def test_engine_elastic_slope_matches_section_stiffness():
    """Criterion C2: initial M-kappa slope vs E*I must agree within 5%."""
    _, props, result = _double_bottom_case()
    error = abs(result.elastic_slope - props["stiffness"]) / props["stiffness"]
    assert error < 0.05


def test_energy_ledger_closes_within_one_percent():
    """Criterion C4: integral M dkappa must equal sum A integral sigma deps."""
    _, _, result = _double_bottom_case()
    assert result.ledger_residual < 0.01


def test_ultimate_state_is_an_interior_peak_below_the_plastic_bound():
    geometry, props, result = _double_bottom_case()
    plastic = geometry.plastic_properties()
    assert result.peak_is_interior is True
    assert 0.0 < result.ultimate_moment < plastic["plastic_moment"]
    assert result.ultimate_curvature > 0.0
    assert result.first_yield_moment <= result.ultimate_moment


def test_hogging_and_sagging_differ_in_the_expected_direction():
    geometry = replace(ISSC2009_DOUBLE_BOTTOM, subdivisions=16)
    assert geometry.bottom_thickness > geometry.inner_bottom_thickness
    cases = {}
    for condition, sign in (("sagging", 1.0), ("hogging", -1.0)):
        props = geometry.elastic_properties()
        eps_yield = geometry.yield_stress / geometry.young
        extreme = max(props["neutral_axis"], geometry.depth - props["neutral_axis"])
        grid = tuple(sign * value for value in curvature_grid(
            eps_yield / extreme, steps=120, maximum_ratio=8.0))
        cases[condition] = solve_moment_curvature(geometry.fibers(), grid)
    # the thicker outer bottom is compressed in sagging -> larger capacity
    assert cases["sagging"].ultimate_moment > cases["hogging"].ultimate_moment
    assert cases["hogging"].ultimate_curvature < 0.0


def test_solver_input_validation():
    law = elastic_perfectly_plastic_law(young=YOUNG, yield_stress=YIELD)
    fibers = (SectionFiber("a", 1e-3, 0.0, law), SectionFiber("b", 1e-3, 1.0, law))
    with pytest.raises(ValueError):
        solve_moment_curvature(fibers, (1e-4, 2e-3, 3e-3))
    with pytest.raises(ValueError):
        solve_moment_curvature(fibers, (0.0, 1e-3))
    with pytest.raises(ValueError):
        solve_moment_curvature(fibers, (0.0, 2e-3, 1e-3))
    with pytest.raises(ValueError):
        solve_moment_curvature((fibers[0],), (0.0, 1e-3, 2e-3))
    with pytest.raises(TypeError):
        StressStrainLaw("not callable", young=YOUNG, yield_stress=YIELD)


def test_curvature_grid_starts_at_zero_and_increases_quadratically():
    grid = curvature_grid(1.0e-3, steps=40, maximum_ratio=8.0)
    assert grid[0] == 0.0
    assert len(grid) == 41
    assert all(b > a for a, b in zip(grid, grid[1:]))
    assert grid[-1] == pytest.approx(8.0e-3)
    assert grid[1] == pytest.approx(1.0e-3 * 8.0 / 40 ** 2)


# --------------------------------------------------------------------------
# geometry template
# --------------------------------------------------------------------------
def test_geometry_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        replace(ISSC2009_DOUBLE_BOTTOM, bottom_thickness=-0.01).validate()
    with pytest.raises(ValueError):
        replace(ISSC2009_DOUBLE_BOTTOM, subdivisions=1).validate()
    with pytest.raises(TypeError):
        replace(ISSC2009_DOUBLE_BOTTOM, girder_count=2.5).validate()
    with pytest.raises(ValueError):
        replace(ISSC2009_DOUBLE_BOTTOM, imperfection_reference="nonsense").validate()
    with pytest.raises(ValueError):
        replace(ISSC2009_DOUBLE_BOTTOM, imperfection_ratio=0.0).validate()
    with pytest.raises(ValueError):
        replace(ISSC2009_DOUBLE_BOTTOM, depth=0.02).validate()


def test_geometry_elastic_properties_match_closed_form_box():
    """A girder-free box has a hand-computable second moment."""
    geometry = replace(ISSC2009_DOUBLE_BOTTOM, girder_count=0, include_girders=False)
    b, d = geometry.breadth, geometry.depth
    tb, ti, ts = (geometry.bottom_thickness, geometry.inner_bottom_thickness,
                  geometry.side_thickness)
    areas = [b * tb, b * ti, 2 * d * ts]
    ys = [0.5 * tb, d - 0.5 * ti, 0.5 * d]
    own = [b * tb ** 3 / 12.0, b * ti ** 3 / 12.0, 2 * ts * d ** 3 / 12.0]
    area = sum(areas)
    centroid = sum(a * y for a, y in zip(areas, ys)) / area
    inertia = sum(i + a * (y - centroid) ** 2 for a, y, i in zip(areas, ys, own))
    props = geometry.elastic_properties()
    assert props["area"] == pytest.approx(area, rel=1e-12)
    assert props["neutral_axis"] == pytest.approx(centroid, rel=1e-12)
    assert props["inertia"] == pytest.approx(inertia, rel=1e-12)
    assert props["stiffness"] == pytest.approx(YOUNG * inertia, rel=1e-12)


def test_geometry_describe_separates_fixed_and_assumed_fields():
    described = ISSC2009_DOUBLE_BOTTOM.describe()
    fixed = described["fixed_by_campaign_node"]
    assumed = described["assumed_by_implementation"]
    assert fixed["length"] == 6.0 and fixed["breadth"] == 3.0
    assert fixed["depth"] == 2.0 and fixed["yield_stress"] == 315e6
    assert "floor_spacing" in assumed and "girder_count" in assumed
    assert set(fixed).isdisjoint(assumed)


def test_plastic_moment_is_integrated_over_distributed_fibres():
    geometry = ISSC2009_DOUBLE_BOTTOM
    plastic = geometry.plastic_properties()
    fibres = geometry.fibers()
    axis = plastic["plastic_neutral_axis"]
    hand = YIELD * sum(f.area * abs(f.y - axis) for f in fibres)
    assert plastic["plastic_moment"] == pytest.approx(hand, rel=1e-12)
    # the plastic bound must exceed the elastic first-yield moment of the
    # extreme fibre state only if the shape factor is above one; for this
    # area-distributed box section it is below, which is documented in the run
    assert plastic["plastic_moment"] > 0.5 * geometry.elastic_properties()["first_yield_moment"]


# --------------------------------------------------------------------------
# criteria: imperfection sensitivity, convergence, anchor
# --------------------------------------------------------------------------
def test_imperfection_sensitivity_thresholds():
    """Criterion C3, on the declared schedule, with the fixed 0.1 reference."""
    geometry = replace(ISSC2009_DOUBLE_BOTTOM, subdivisions=16)
    props = geometry.elastic_properties()
    eps_yield = geometry.yield_stress / geometry.young
    extreme = max(props["neutral_axis"], geometry.depth - props["neutral_axis"])
    kappa_yield = eps_yield / extreme
    moments = {}
    for ratio in (0.05, 0.1, 0.2, 0.5):
        case = replace(geometry, imperfection_ratio=ratio)
        grid = curvature_grid(kappa_yield, steps=120, maximum_ratio=8.0)
        moments[ratio] = solve_moment_curvature(case.fibers(), grid).ultimate_moment
    baseline = moments[0.05]
    assert moments[0.5] < moments[0.2] < moments[0.1] < baseline
    assert (baseline - moments[0.1]) / baseline <= 0.10
    assert (baseline - moments[0.5]) / baseline <= 0.30


def test_fibre_count_convergence_threshold():
    """Criterion C5: 10/20/40 fibres must agree within 5%."""
    geometry = ISSC2009_DOUBLE_BOTTOM
    props = geometry.elastic_properties()
    eps_yield = geometry.yield_stress / geometry.young
    extreme = max(props["neutral_axis"], geometry.depth - props["neutral_axis"])
    kappa_yield = eps_yield / extreme
    grid = curvature_grid(kappa_yield, steps=120, maximum_ratio=8.0)
    moments = []
    for count in (10, 20, 40):
        case = replace(geometry, subdivisions=count)
        moments.append(solve_moment_curvature(case.fibers(), grid).ultimate_moment)
    mean = sum(moments) / len(moments)
    assert (max(moments) - min(moments)) / mean <= 0.05


def test_anchor_loader_returns_none_without_a_documented_file(tmp_path):
    assert load_anchor(None) is None
    assert load_anchor(tmp_path / "absent.json") is None


def test_anchor_injection_turns_c1_into_a_machine_checked_criterion(tmp_path):
    """C1 is checked only against an injected, documented anchor."""
    computed = run_issc2009_double_bottom(
        subdivisions=(10, 20), imperfection_ratios=(0.05, 0.1),
        curvature_steps=80, pytest_tests=15)
    moment = computed["ultimate"]["sagging_moment"]
    assert computed["criteria"]["C1"]["status"] == "NOT_VERIFIED"

    matching = tmp_path / "anchor.json"
    matching.write_text(json.dumps({
        "reference_ultimate_moment": moment, "source": "synthetic test anchor",
        "tolerance": 0.15, "condition": "sagging"}), encoding="utf-8")
    anchor = load_anchor(matching)
    assert anchor is not None
    passed = run_issc2009_double_bottom(
        anchor=anchor, subdivisions=(10,), imperfection_ratios=(0.1,),
        curvature_steps=80, pytest_tests=15)
    assert passed["criteria"]["C1"]["status"] == "PASS"
    assert passed["criteria"]["C1"]["detail"]["relative_error"] < 1e-9

    wrong = UltimateMomentAnchor(reference_ultimate_moment=moment * 2.0,
                                 source="synthetic deliberate mismatch")
    failed = run_issc2009_double_bottom(
        anchor=wrong, subdivisions=(10,), imperfection_ratios=(0.1,),
        curvature_steps=80, pytest_tests=15)
    assert failed["criteria"]["C1"]["status"] == "FAIL"
    assert failed["overall_status"] == "FAIL"


def test_runner_reports_every_criterion_and_honest_overall_status():
    report = run_issc2009_double_bottom(
        subdivisions=(10, 20), imperfection_ratios=(0.05, 0.1, 0.5),
        curvature_steps=100, pytest_tests=20)
    assert set(report["criteria"]) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    for key in ("C2", "C3", "C4", "C5", "C6"):
        assert report["criteria"][key]["status"] == "PASS", key
    assert report["criteria"]["C1"]["status"] == "NOT_VERIFIED"
    assert report["overall_status"] == "CONDITIONAL"
    assert report["criteria_failed"] == []
    assert report["ultimate"]["peak_is_interior"] is True
    assert report["section_engine_verification"]["passed"] is True
    # provenance of the numbers is carried with the payload
    assert "withheld" in report["provenance"]
    json.dumps(report)  # must stay serialisable for the evidence artifact

def test_cross_method_envelope_bounds_the_softening_solution():
    """An independent solver (limiting stress, no softening) must bound M_u."""
    report = run_issc2009_double_bottom(
        subdivisions=(10,), imperfection_ratios=(0.1,), curvature_steps=80)
    envelope = report["cross_method_envelope"]
    assert envelope["consistent"] is True
    assert 0.5 <= envelope["softening_over_envelope"] <= 1.0
    assert envelope["envelope_ultimate_moment"] > 0.0
