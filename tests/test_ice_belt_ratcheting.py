"""Qualification of the ice-belt ratcheting / shakedown capability (P226W3-N05 / m14467).

Machine-checked criteria (the numbers the report quotes are produced by
``scripts/ice_belt_ratcheting_verification.py``):

* C1 **elastic limit** -- the incremental solver returns ``E`` (and exactly zero
  plastic strain) everywhere below ``x + y = 1`` and detects plastic flow
  immediately above it (detectability gate, so C1 cannot pass vacuously);
* C2 **Melan LP vs closed form** -- the LP shakedown boundary equals the
  closed-form ``x_sd = 1 - y/4`` within ``5e-3`` for 8 values of ``y``;
* C3 **incremental vs Melan** -- region agreement (shakedown vs no shakedown)
  on a 5 x 5 diagram grid ``>= 95 %``;
* C4 **ratcheting rate** -- in the ratcheting region the per-cycle growth is
  constant (relative scatter ``<= 1e-6`` over the settled cycles) and the
  accumulated growth is linear in the cycle number (``R^2 >= 0.999``);
* C5 **no ratcheting under pure secondary cycling** -- with ``x = 0`` the net
  growth is zero for ``y <= 1.8`` and alternating plasticity (``P``) appears
  for ``y >= 2.2``; no solution may ratchet from a purely self-equilibrated
  cyclic load (a physical sanity law);
* C6 **pressure-area law** -- the log-log slope of ``p(A)`` equals
  ``-alpha`` to ``1e-12``, pressure is monotone decreasing in area, the Polar
  Class ordering is monotone, and the provenance gate fails closed on
  illustrative data;
* C7 **ice impact spectrum** -- cycle count, amplitude bounds, descending
  ordering, block re-binning preserving the cycle count, and the calibration of
  the truncated Weibull mean;
* C8 **moment-curvature law** -- the closed-form EPP moment-curvature relation
  behind :func:`first_cycle_plastic_deflection` matches a numerical
  through-thickness section integration to ``1e-10``;
* C9 **deflection accumulation** -- no permanent set below first yield,
  monotone non-decreasing accumulation, saturation (bounded) in the strong
  overload case, and consistency of the deflection limit / service-life
  bookkeeping.
"""

import math

import numpy as np
import pytest

from tensorfem.ice_belt_ratcheting import (
    IceBeltPlate,
    IceCollisionSpectrum,
    IceCycleBin,
    IcePressureModel,
    analytic_elastic_x,
    analytic_shakedown_x,
    bending_stress_from_pressure,
    bree_cycles,
    bree_region,
    check_pressure_provenance,
    deflection_to_membrane_strain,
    first_cycle_plastic_deflection,
    ice_belt_deflection_history,
    ice_belt_service_life,
    ice_collision_spectrum,
    ice_pressure,
    illustrative_class_pressure,
    melan_boundary,
    melan_shakedown,
    membrane_deflection,
    _solve_increment,
    _thickness_rule,
)

PLATE = IceBeltPlate(span_m=0.35, thickness_m=0.020, young_pa=206.0e9,
                     yield_pa=355.0e6, edge_condition="clamped")


# --------------------------------------------------------------------------- #
# C1 elastic limit (exact, x + y = 1)
# --------------------------------------------------------------------------- #
def test_c1_elastic_limit_is_exact_and_detectable():
    below = [(0.10, 0.20), (0.30, 0.30), (0.50, 0.40), (0.00, 0.90),
             (0.70, 0.25), (0.85, 0.10)]
    above = [(0.20, 0.90), (0.45, 0.60), (0.60, 0.45), (0.00, 1.05),
             (0.75, 0.30), (0.90, 0.15)]
    for x, y in below:
        res = bree_cycles(x, y, cycles=4, steps=6, fibres=41)
        assert res.region == "E", (x, y, res.region)
        assert res.plastic_range == 0.0
        assert abs(res.growth_per_cycle) == 0.0
    for x, y in above:
        assert x + y > 1.0
        res = bree_cycles(x, y, cycles=4, steps=6, fibres=41)
        assert res.region != "E", (x, y, res.region)
        # detectability: yielding must really have occurred (non-zero plastic
        # strain), otherwise the criterion could pass on a silent no-op
        assert res.plastic_strain_peak > 1e-6
    # the closed-form helper agrees with both sides of the boundary
    assert analytic_elastic_x(0.4) == pytest.approx(0.6)
    assert analytic_elastic_x(1.4) == 0.0


# --------------------------------------------------------------------------- #
# C2 / C3 shakedown: LP vs closed form, LP vs incremental
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("y", [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.4])
def test_c2_melan_boundary_matches_closed_form(y):
    lp = melan_boundary(y, fibres=41)
    assert lp == pytest.approx(analytic_shakedown_x(y), abs=5e-3)


def test_c2_melan_boundary_is_not_the_linear_residual_line():
    """The closed form must differ from the pure-bending ``1 - y/2`` line.

    Otherwise the LP would just be re-deriving the textbook bending-residual
    result and the claimed extension of the shakedown region would be vacuous.
    """
    for y in (0.8, 1.2, 1.6):
        assert analytic_shakedown_x(y) > max(0.0, 1.0 - 0.5 * y) + 0.05


def test_c3_incremental_matches_melan_on_the_diagram():
    xs = (0.15, 0.35, 0.55, 0.75, 0.92)
    ys = (0.2, 0.6, 1.0, 1.5, 2.0)
    agree = 0
    total = 0
    mismatches = []
    for y in ys:
        for x in xs:
            region = bree_region(x, y, cycles=6, steps=6, fibres=41)
            shakes = melan_shakedown(x, y, fibres=41)
            total += 1
            if (region in ("E", "S")) == shakes:
                agree += 1
            else:
                mismatches.append((x, y, region, shakes))
    assert agree / total >= 0.95, mismatches


def test_c3_failure_of_shakedown_is_plastic_cycling_or_ratcheting():
    # well beyond the shakedown limit at zero primary load -> alternating plasticity
    assert bree_region(0.0, 2.4, cycles=6, steps=6, fibres=41) == "P"
    # heavy primary load -> ratcheting
    heavy = bree_cycles(0.85, 1.5, cycles=8, steps=6, fibres=41)
    assert heavy.region == "R"
    assert heavy.growth_per_cycle > 1e-3


# --------------------------------------------------------------------------- #
# C4 ratcheting rate is steady (linear accumulation)
# --------------------------------------------------------------------------- #
def test_c4_ratcheting_rate_is_steady():
    res = bree_cycles(0.85, 1.5, cycles=10, steps=6, fibres=41)
    assert res.region == "R"
    settled = np.array(res.growth_history[3:], dtype=float)
    rate = res.growth_per_cycle
    assert rate > 0.0
    assert np.max(np.abs(settled - rate)) <= 1e-6 * abs(rate)
    cycles = np.arange(1, settled.size + 1)
    cumulative = np.cumsum(settled)
    slope, intercept = np.polyfit(cycles, cumulative, 1)
    resid = cumulative - (slope * cycles + intercept)
    ss_tot = np.sum((cumulative - cumulative.mean()) ** 2)
    assert 1.0 - np.sum(resid ** 2) / ss_tot >= 0.999


# --------------------------------------------------------------------------- #
# C5 no ratcheting under a purely self-equilibrated cyclic load
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("y", [0.5, 1.0, 1.4, 1.8])
def test_c5_no_ratcheting_without_primary_load(y):
    res = bree_cycles(0.0, y, cycles=6, steps=6, fibres=41)
    assert res.region in ("E", "S")
    assert abs(res.growth_per_cycle) == 0.0
    assert res.plastic_range == 0.0


def test_c5_alternating_plasticity_beyond_the_lateral_limit():
    res = bree_cycles(0.0, 2.4, cycles=6, steps=6, fibres=41)
    assert res.region == "P"
    assert abs(res.growth_per_cycle) <= 1e-9
    assert res.plastic_range > 1e-2


# --------------------------------------------------------------------------- #
# equilibrium / yield contract of the incremental solver
# --------------------------------------------------------------------------- #
def test_increment_satisfies_equilibrium_and_yield():
    z, w = _thickness_rule(41)
    eps_p = np.zeros_like(z)
    for x, y, lam in [(0.4, 0.9, 0.5), (0.6, 1.2, 1.0), (0.2, 1.6, 0.25)]:
        e, kappa, eps_new, sigma = _solve_increment(x, lam, y, z, w, eps_p)
        assert 0.5 * float(np.sum(w * sigma)) == pytest.approx(x, abs=1e-12)
        assert np.max(np.abs(sigma)) <= 1.0 + 1e-12
        assert kappa == 0.0                       # bending is restrained
        trial = e - eps_new - lam * y * z
        assert np.max(np.abs(trial)) <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# C6 IACS-style pressure-area law
# --------------------------------------------------------------------------- #
def test_c6_pressure_area_power_law():
    areas = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    p = ice_pressure(areas, p_ref_mpa=1.6, area_exponent=0.3)
    assert p[0] == pytest.approx(1.6 * 0.1 ** -0.3)
    slope = np.polyfit(np.log(areas), np.log(p), 1)[0]
    assert slope == pytest.approx(-0.3, abs=1e-12)
    assert np.all(np.diff(p) < 0.0)
    # C_d / C_f scale the pressure multiplicatively
    scaled = ice_pressure(areas, p_ref_mpa=1.6, c_d=1.2, c_f=0.8)
    assert np.allclose(scaled, 1.2 * 0.8 * p)


def test_c6_polar_class_ordering_and_provenance_gate():
    table = [illustrative_class_pressure(pc, 2.0) for pc in range(1, 8)]
    assert all(a > b for a, b in zip(table[:-1], table[1:]))
    model = IcePressureModel(p_ref_mpa=2.0, provenance="ILLUSTRATIVE-default")
    check_pressure_provenance(model)                      # allowed by default
    with pytest.raises(ValueError):
        check_pressure_provenance(model, require_design_data=True)
    check_pressure_provenance(
        IcePressureModel(p_ref_mpa=2.0, provenance="IACS-UR-I2 2022"),
        require_design_data=True)


def test_c6_pressure_model_validation_is_fail_closed():
    for kwargs in ({"p_ref_mpa": 0.0}, {"p_ref_mpa": 1.0, "area_exponent": 1.0},
                   {"p_ref_mpa": 1.0, "c_d": -1.0},
                   {"p_ref_mpa": 1.0, "area_ref_m2": 0.0}):
        with pytest.raises(ValueError):
            IcePressureModel(**kwargs)


# --------------------------------------------------------------------------- #
# C7 ice impact spectrum
# --------------------------------------------------------------------------- #
def test_c7_spectrum_structure_and_counts():
    spec = ice_collision_spectrum(design_pressure_mpa=3.0, impacts_per_voyage=40,
                                  voyages=5, n_bins=8)
    assert spec.total_cycles == 200
    amps = spec.amplitudes()
    assert np.all(np.diff(amps) <= 0.0)                   # descending blocks
    assert amps.max() <= 3.0 + 1e-12
    assert amps.min() > 0.0
    # truncated-Weibull calibration: mean amplitude ~ half the design pressure
    weights = spec.counts() / spec.total_cycles
    assert float(np.sum(weights * amps)) == pytest.approx(1.5, rel=0.25)
    rebinned = spec.bin_by_pressure(4)
    assert rebinned.total_cycles == spec.total_cycles
    assert len(rebinned) == 4
    scaled = spec.scaled(0.5)
    assert np.allclose(scaled.amplitudes(), 0.5 * spec.amplitudes())


def test_c7_spectrum_validation_is_fail_closed():
    for kwargs in ({"design_pressure_mpa": 0.0, "impacts_per_voyage": 5},
                   {"design_pressure_mpa": 1.0, "impacts_per_voyage": 0.0},
                   {"design_pressure_mpa": 1.0, "impacts_per_voyage": 5,
                    "weibull_shape": 0.0},
                   {"design_pressure_mpa": 1.0, "impacts_per_voyage": 5,
                    "max_amplitude_ratio": 1.5}):
        with pytest.raises(ValueError):
            ice_collision_spectrum(**kwargs)


def test_c7_empty_spectrum_is_handled():
    spec = IceCollisionSpectrum(bins=(), design_pressure_mpa=2.0,
                                loaded_area_m2=1.0, weibull_shape=1.0)
    result = ice_belt_service_life(PLATE, spec)
    assert result.total_deflection_m == 0.0
    assert result.utilisation == 0.0
    assert result.cycles_to_limit == float("inf")


# --------------------------------------------------------------------------- #
# C8 moment-curvature law used by the first-cycle set
# --------------------------------------------------------------------------- #
def _section_moment(kappa: float, sigma_y: float, young: float, t: float,
                    n: int = 200001) -> float:
    """Numerical through-thickness moment of an EPP rectangular section."""
    z = np.linspace(-0.5, 0.5, n) * t
    eps = kappa * z
    sigma = np.clip(young * eps, -sigma_y, sigma_y)
    return float(np.trapezoid(sigma * z, z))


def test_c8_moment_curvature_closed_form_matches_numerical_section():
    sigma_y, young, t = 355.0e6, 206.0e9, 0.020
    m_p = sigma_y * t ** 2 / 4.0
    kappa_y = 2.0 * sigma_y / (young * t)
    for factor in (1.2, 1.5, 2.0, 3.0, 5.0):
        kappa = factor * kappa_y
        predicted = m_p * (1.0 - (1.0 / 3.0) * (kappa_y / kappa) ** 2)
        assert _section_moment(kappa, sigma_y, young, t) == pytest.approx(
            predicted, rel=1e-9)


def test_c8_first_cycle_set_starts_at_first_yield():
    p_y = PLATE.yield_pressure_pa()
    assert first_cycle_plastic_deflection(PLATE, 0.5 * p_y) == 0.0
    assert first_cycle_plastic_deflection(PLATE, 0.999 * p_y) == 0.0
    small = first_cycle_plastic_deflection(PLATE, 1.01 * p_y)
    assert 0.0 < small < 1e-3
    # beyond the plastic moment the closed form flags the bending mechanism
    p_limit = p_y * 1.5
    assert math.isinf(first_cycle_plastic_deflection(PLATE, p_limit))


def test_c8_membranedeflection_kinematics_round_trip():
    for w in (1e-4, 5e-4, 2e-3):
        eps = deflection_to_membrane_strain(w, PLATE.span_m)
        assert membrane_deflection(eps, PLATE.span_m) == pytest.approx(w)
    assert membrane_deflection(0.0, 1.0) == 0.0


# --------------------------------------------------------------------------- #
# C9 deflection accumulation and service life
# --------------------------------------------------------------------------- #
def test_c9_no_permanent_set_below_first_yield():
    p = 0.9 * PLATE.yield_pressure_pa() / 1.0e6
    hist = ice_belt_deflection_history(PLATE, p, cycles=20, fibres=41)
    assert hist.final_deflection() == 0.0
    assert hist.plateau_reached


def test_c9_deflection_is_monotone_and_saturates():
    hist = ice_belt_deflection_history(PLATE, 3.0, cycles=120, fibres=41)
    w = hist.deflection_m
    assert w[0] > 0.0                                   # first-cycle plastic set
    assert np.all(np.diff(w) >= -1e-15)
    assert hist.plateau_reached
    # membrane action must bound the deflection well below the span
    assert hist.final_deflection() < 0.1 * PLATE.span_m
    # the membrane ratio grows monotonically and stays below the cap
    assert np.all(np.diff(hist.primary_ratio) >= -1e-15)
    assert np.max(hist.primary_ratio) <= 0.980 + 1e-12


def test_c9_service_life_bookkeeping():
    spec = ice_collision_spectrum(design_pressure_mpa=3.0, impacts_per_voyage=60,
                                  voyages=5, n_bins=6)
    result = ice_belt_service_life(PLATE, spec, cycles_per_bin=40, fibres=41)
    assert result.limit_m == pytest.approx(PLATE.deflection_limit_ratio
                                          * PLATE.span_m)
    assert result.total_deflection_m > 0.0
    assert result.utilisation == pytest.approx(
        result.total_deflection_m / result.limit_m)
    assert result.acceptable == (result.utilisation <= 1.0)
    assert len(result.histories) == len(spec)
    # each bin is ranked by pressure: the heaviest bin drives the largest set
    heaviest = result.histories[0]
    assert heaviest.pressure_mpa == pytest.approx(spec.max_pressure())
    assert heaviest.final_deflection() == max(
        h.final_deflection() for h in result.histories)


def test_c9_plate_validation_is_fail_closed():
    with pytest.raises(ValueError):
        IceBeltPlate(span_m=0.0, thickness_m=0.02, young_pa=2e11, yield_pa=3e8)
    with pytest.raises(ValueError):
        IceBeltPlate(span_m=1.0, thickness_m=0.02, young_pa=2e11, yield_pa=3e8,
                     edge_condition="free")
    with pytest.raises(ValueError):
        IceBeltPlate(span_m=1.0, thickness_m=0.02, young_pa=2e11, yield_pa=3e8,
                     membrane_restraint=1.5)
    with pytest.raises(ValueError):
        IceCycleBin(pressure_mpa=-1.0, count=3)