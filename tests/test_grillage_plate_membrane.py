"""Gates for :mod:`tensorfem.grillage_plate_membrane`.

The three solution routes of the module are cross-checked against each other
and against closed forms, so that the membrane reserve reported for the
GAP-PP-5 grillage anchor is not a number produced by a single unchecked code
path.  Gate identifiers ``M1..M8`` match the module docstring.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.grillage_plate_membrane import (
    RESTRAINT_LEVELS,
    StripMembraneGeometry,
    case_strip_geometry,
    membrane_reserve_table,
    strip_capacity,
    strip_membrane_exact,
    strip_membrane_fe,
    strip_membrane_series,
)

SPAN = 0.35
THICKNESS = 0.020
YOUNG = 206.0e9
YIELD = 355.0e6


@pytest.fixture(scope="module")
def strip() -> StripMembraneGeometry:
    return StripMembraneGeometry(span=SPAN, thickness=THICKNESS, young=YOUNG,
                                 yield_stress=YIELD, name="bay-field")


def linear_w(pressure: float, geom: StripMembraneGeometry) -> float:
    return 5.0 * pressure * geom.span ** 4 / (384.0 * geom.bending_rigidity)


def linear_m(pressure: float, geom: StripMembraneGeometry) -> float:
    return pressure * geom.span ** 2 / 8.0


@pytest.mark.parametrize("pressure", [1.0e4, 1.0e5])
def test_m1_exact_route_recovers_linear_field(strip, pressure):
    """M1: the small-load limit of the exact route is the linear strip field."""
    sol = strip_membrane_exact(strip, pressure)
    assert sol.converged
    assert sol.max_deflection == pytest.approx(linear_w(pressure, strip), rel=0.01)
    assert sol.max_moment == pytest.approx(linear_m(pressure, strip), rel=0.01)


@pytest.mark.parametrize("pressure", [1.0e4, 1.0e5])
def test_m2_series_route_recovers_linear_field(strip, pressure):
    """M2: same limit on the independent Ritz route."""
    sol = strip_membrane_series(strip, pressure)
    assert sol.max_deflection == pytest.approx(linear_w(pressure, strip), rel=0.01)
    assert sol.max_moment == pytest.approx(linear_m(pressure, strip), rel=0.01)


def test_m2_series_matches_textbook_deflection_coefficient(strip):
    """M2 (closed form): 21 odd modes reproduce 5/384 to <0.05 %."""
    pressure = 1.0e5
    sol = strip_membrane_series(strip, pressure, terms=21)
    assert sol.max_deflection / linear_w(pressure, strip) == pytest.approx(1.0, rel=5e-4)


@pytest.mark.parametrize("pressure", [5.0e5, 1.0e6, 2.33e6, 4.0e6])
def test_m3_series_agrees_with_exact_over_the_nonlinear_range(strip, pressure):
    """M3: two independent routes on the nonlinear range (1 %)."""
    exact = strip_membrane_exact(strip, pressure)
    series = strip_membrane_series(strip, pressure)
    assert series.axial_force == pytest.approx(exact.axial_force, rel=0.01)
    assert series.max_moment == pytest.approx(exact.max_moment, rel=0.01)
    assert series.max_deflection == pytest.approx(exact.max_deflection, rel=0.01)


@pytest.mark.parametrize("pressure", [5.0e5, 1.0e6, 4.0e6])
def test_m4_corotational_fe_agrees_with_exact(strip, pressure):
    """M4: finite-element route (no compatibility relation) vs the exact route."""
    exact = strip_membrane_exact(strip, pressure)
    fe = strip_membrane_fe(strip, pressure, n_elements=80)
    assert fe.converged
    assert fe.axial_force == pytest.approx(exact.axial_force, rel=0.03)
    assert fe.max_moment == pytest.approx(exact.max_moment, rel=0.03)
    assert fe.max_deflection == pytest.approx(exact.max_deflection, rel=0.03)


def test_m4_fe_mesh_convergence_is_monotone(strip):
    """M4 (convergence): the FE axial-force error decreases under refinement."""
    pressure = 2.33e6
    exact = strip_membrane_exact(strip, pressure).axial_force
    errors = []
    for n in (20, 40, 80):
        fe = strip_membrane_fe(strip, pressure, n_elements=n)
        errors.append(abs(fe.axial_force / exact - 1.0))
    assert errors[-1] < errors[0]
    assert errors[-1] < 1.0e-3


def test_m5_membrane_force_obeys_compatibility(strip):
    """M5: the FE axial force satisfies the end-shortening relation (*)."""
    pressure = 1.0e6
    fe = strip_membrane_fe(strip, pressure, n_elements=80)
    elastic = fe.axial_force * SPAN / strip.axial_rigidity
    # geometric shortening from the FE deflection field, reconstructed with the
    # exact route's field at the same axial force
    from tensorfem.grillage_plate_membrane import _shortening

    geometric = _shortening(strip, pressure, fe.axial_force)
    assert geometric == pytest.approx(elastic, rel=0.02)


def test_m6_free_edge_gives_no_membrane_reserve():
    """M6: k_s = 0 must collapse onto the committed linear criterion exactly."""
    free = case_strip_geometry("design-ice + design head", 0.0)
    cap = strip_capacity(free)
    assert cap.reserve_membrane == pytest.approx(1.0, rel=1e-9)
    assert cap.axial_force_plastic == pytest.approx(0.0, abs=1e-6)


def test_m6_reserve_is_monotone_in_edge_restraint():
    """M6 (monotonicity): more edge restraint never lowers the capacity."""
    reserves = []
    for _label, level in RESTRAINT_LEVELS[::-1]:      # free .. immovable
        cap = strip_capacity(case_strip_geometry("severe-ice + tank head", level))
        reserves.append(cap.reserve_membrane)
    assert all(b >= a - 1e-9 for a, b in zip(reserves, reserves[1:]))
    assert reserves[-1] > reserves[0]


def test_m7_ritz_solution_minimises_the_total_potential(strip):
    """M7: the Ritz coefficients are a stationary point of the potential."""
    pressure = 1.0e6
    sol = strip_membrane_series(strip, pressure)
    # perturb the reported peak deflection by +-2 % and rebuild a one-mode field
    L, EI, EA = strip.span, strip.bending_rigidity, strip.axial_rigidity
    k = math.pi / L

    def potential(a):
        delta = 0.25 * a ** 2 * k ** 2 * L
        return (0.25 * EI * L * (a * k ** 2) ** 2 + 0.5 * EA / L * delta ** 2
                - pressure * a * 2.0 * L / math.pi)

    a0 = sol.max_deflection
    for perturb in (0.98, 1.02):
        assert potential(a0 * perturb) > potential(a0)


def test_m8_acceptance_cases_are_bounded_and_conservative():
    """M8: the committed criterion stays below the membrane capacities."""
    rows = membrane_reserve_table(solver="exact")
    assert rows
    for row in rows:
        # first yield is below the committed plate-field limit: the committed
        # criterion is a plastic limit state, not a yield check
        assert row["first_yield_over_committed"] < 1.0
        # membrane reserve: never below 1 (tension cannot reduce the capacity
        # when the axial interaction is written with eta^2), bounded for this
        # geometry, and small enough to be a bounded statement
        assert 1.0 - 1e-9 <= row["reserve_membrane"] <= 1.25
        # the strip is stiffer or softer than the 2D field but is not a
        # replacement for it: the reduction error is a reported limitation
        assert abs(row["strip_vs_2D_error_pct"]) < 40.0


def test_m8_partial_restraint_reserve_is_small():
    """M8: at realistic (partial) edge restraint the reserve stays under 10 %."""
    for scenario in ("light-ice + ballast head", "design-ice + design head",
                     "severe-ice + tank head"):
        for level in (0.5, 0.25):
            cap = strip_capacity(case_strip_geometry(scenario, level))
            assert 1.0 <= cap.reserve_membrane < 1.10


def test_full_restraint_reserve_matches_reported_magnitude():
    """M8: immovable edge gives the largest reserve, 10-20 % for these cases."""
    reserves = []
    for scenario in ("light-ice + ballast head", "design-ice + design head",
                     "severe-ice + tank head"):
        cap = strip_capacity(case_strip_geometry(scenario, math.inf))
        reserves.append(cap.reserve_membrane)
    assert all(1.05 <= r <= 1.25 for r in reserves)


def test_geometry_validation():
    with pytest.raises(ValueError):
        StripMembraneGeometry(span=-1.0, thickness=THICKNESS)
    with pytest.raises(ValueError):
        StripMembraneGeometry(span=SPAN, thickness=0.0)
    with pytest.raises(ValueError):
        StripMembraneGeometry(span=SPAN, thickness=THICKNESS,
                              axial_stress=YIELD)
    with pytest.raises(ValueError):
        StripMembraneGeometry(span=SPAN, thickness=THICKNESS, restraint=-1.0)
    with pytest.raises(ValueError):
        strip_membrane_exact(StripMembraneGeometry(span=SPAN, thickness=THICKNESS),
                             -1.0)


def test_strip_capacity_reports_the_three_routes_consistently():
    """The exact and Ritz routes give the same capacity to <1 %."""
    geom = case_strip_geometry("design-ice + design head", math.inf)
    a = strip_capacity(geom, solver="exact")
    b = strip_capacity(geom, solver="series")
    assert a.pressure_plastic == pytest.approx(b.pressure_plastic, rel=0.01)
    assert a.pressure_first_yield == pytest.approx(b.pressure_first_yield, rel=0.01)