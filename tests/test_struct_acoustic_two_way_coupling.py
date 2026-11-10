"""Verification of the structural-acoustic two-way coupling surface (node p348w3-n07).

Anchors, all closed-form and constant-free:

* pulsating sphere (monopole) pressure / power / surface impedance,
* baffled circular piston far field (Airy) and on-axis exact Rayleigh integral,
* the ``Lw = Lv + C`` level law with ``C`` derived from ``W = rho c S <v^2> sigma``,
* energy balance of a breathing shell loaded by the *domain-solved* acoustic
  impedance.
"""
import math

import pytest
import torch

from tensorfem.struct_acoustics.plate_far_field import (
    angle_sweep_error,
    baffled_piston_far_field,
    bessel_j1,
    far_field_power_from_directivity,
    node_cos_theta_far_field,
    on_axis_exact,
    piston_directivity,
    piston_power_check,
    piston_radiation_efficiency,
    rayleigh_far_field,
    wavenumber as plate_k,
)
from tensorfem.struct_acoustics.pulsating_sphere import (
    RadialMesh,
    convergence_study,
    plane_wave_form_relative_error,
    plane_wave_power,
    plane_wave_power_relative_error,
    plane_wave_pressure,
    pressure as monopole_pressure,
    radiated_power,
    radiation_efficiency,
    solve_pulsating_sphere,
    surface_impedance,
    wavenumber,
)
from tensorfem.struct_acoustics.radiation_scale_law import (
    V_REF,
    W_REF,
    coupled_breathing_sphere,
    energy_balance_sweep,
    mean_square_from_peak,
    monopole_surface_source,
    piston_surface_source,
    rms_from_peak,
    scale_law_offset_db,
    scale_law_residual_db,
    scaling_slope_db,
    sound_power_level,
    spherical_shell_breathing_parameters,
    velocity_level,
)

FREQUENCIES = (50.0, 100.0, 200.0, 500.0)
RADII = (1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 40.0)
ANGLES = (0.0, 15.0, 30.0, 45.0, 60.0)
A = 1.0
PISTON_A = 0.5


# ---------------------------------------------------------------------------
# C1: monopole pressure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_c1_monopole_pressure_matches_closed_form(frequency):
    """C1: FEM nodal pressure within 3 % of the exact monopole at 10 radii."""
    sol = solve_pulsating_sphere(frequency, A, 1.0, 10.0, 48)
    probes = [int(torch.argmin(torch.abs(sol.nodes - r))) for r in RADII]
    exact = monopole_pressure(torch.tensor([float(sol.nodes[i]) for i in probes]), A, frequency, 1.0)
    for i, ev in zip(probes, exact):
        assert abs(complex(sol.pressure[i]) - complex(ev)) / abs(complex(ev)) < 0.03


def test_c1_mesh_refinement_reduces_error():
    """h-refinement must reduce the nodal error monotonically (verification)."""
    rows = convergence_study(500.0, A, (8, 16, 32, 48))
    errors = [row["max_relative_error"] for row in rows]
    assert errors[0] > errors[1] > errors[2] > errors[3]
    assert errors[-1] < 1e-3


def test_c1_flux_conservation_error_shrinks_with_mesh():
    fine = solve_pulsating_sphere(200.0, A, 1.0, 10.0, 96)
    coarse = solve_pulsating_sphere(200.0, A, 1.0, 10.0, 16)
    assert fine.flux_spread < coarse.flux_spread
    assert fine.flux_spread < 0.01


def test_impedance_low_and_high_ka_limits():
    """Re Z -> rho c (ka)^2 for ka -> 0 and -> rho c for ka -> inf."""
    rho_c = 1000.0 * 1500.0
    small = surface_impedance(1.0, 1.0)
    ka = wavenumber(1.0) * 1.0
    assert small.real == pytest.approx(rho_c * ka * ka / (1.0 + ka * ka), rel=1e-12)
    large = surface_impedance(1.0, 50000.0)  # ka = 209
    assert large.real == pytest.approx(rho_c, rel=1e-4)
    assert large.imag < 0.0  # mass-like (added mass) with e^{-i omega t}


def test_plane_wave_form_is_the_high_ka_limit():
    """The node form equals the exact monopole times (1 + i/(ka))."""
    for ka in (0.1, 0.5, 2.0, 40.0):
        radius = 1.0
        frequency = ka * 1500.0 / (2.0 * math.pi * radius)
        exact = monopole_pressure(radius, radius, frequency, 1.0)
        node = plane_wave_pressure(radius, radius, frequency, 1.0)
        assert node / exact == pytest.approx(complex(1.0, 1.0 / ka), rel=1e-12)
        assert abs(node - exact) / abs(exact) == pytest.approx(
            plane_wave_form_relative_error(ka), rel=1e-12
        )
        assert abs(node) / abs(exact) == pytest.approx(math.sqrt(1.0 + 1.0 / ka**2), rel=1e-12)
    # in the asymptotic domain the node form is within 3 %
    assert plane_wave_form_relative_error(50.0) < 0.03
    assert plane_wave_power_relative_error(50.0) < 0.03


def test_monopole_input_validation():
    with pytest.raises(ValueError):
        solve_pulsating_sphere(100.0, -1.0)
    with pytest.raises(ValueError):
        solve_pulsating_sphere(100.0, 1.0, 1.0, 1.0)  # truncation ratio must exceed 1
    with pytest.raises(ValueError):
        RadialMesh.uniform(1.0, 0.5, 4)
    with pytest.raises(ValueError):
        radiation_efficiency(0.0, 100.0)


# ---------------------------------------------------------------------------
# C2: monopole power
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_c2_fem_power_matches_closed_form(frequency):
    sol = solve_pulsating_sphere(frequency, A, 1.0, 10.0, 48)
    exact = radiated_power(A, frequency, 1.0)
    assert abs(sol.power_inner_surface - exact) / exact < 0.03


@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_c2_inner_and_outer_flux_agree(frequency):
    sol = solve_pulsating_sphere(frequency, A, 1.0, 10.0, 48)
    assert abs(sol.power_outer_surface - sol.power_inner_surface) / sol.power_inner_surface < 0.03


def test_c2_power_equals_half_area_impedance():
    """W = 1/2 A Re(Z) |v0|^2 must reproduce the closed-form power."""
    domain = solve_pulsating_sphere(200.0, A, 1.0, 10.0, 48)
    w_from_z = 0.5 * 4.0 * math.pi * A * A * surface_impedance(A, 200.0).real
    assert w_from_z == pytest.approx(radiated_power(A, 200.0, 1.0), rel=1e-12)
    assert domain.power_inner_surface == pytest.approx(w_from_z, rel=0.03)


def test_c2_plane_wave_power_is_asymptotic():
    """The node's 2 pi a^2 rho c v0^2 is reached only in the ka >> 1 limit."""
    radius = 50.0
    for frequency in FREQUENCIES:
        lam = 1500.0 / frequency
        n = max(48, int(8 * 3.0 * radius / lam))
        fem = solve_pulsating_sphere(frequency, radius, 1.0, 4.0, n)
        node = plane_wave_power(radius, 1.0)
        assert abs(fem.power_inner_surface - node) / node < 0.03
    # and it is clearly wrong at low ka
    assert abs(plane_wave_power(A, 1.0) - radiated_power(A, 50.0, 1.0)) / radiated_power(
        A, 50.0, 1.0
    ) > 0.9


def test_radiation_efficiency_low_frequency_law():
    """sigma -> (ka)^2 for ka -> 0 and -> 1 for ka -> inf."""
    ka = wavenumber(1.0)
    assert radiation_efficiency(1.0, 1.0) == pytest.approx(ka * ka / (1.0 + ka * ka), rel=1e-12)
    ka_small = wavenumber(0.1)
    assert radiation_efficiency(1.0, 0.1) == pytest.approx(ka_small * ka_small, rel=1e-2)
    assert radiation_efficiency(1.0, 1.0e5) > 0.999


# ---------------------------------------------------------------------------
# C3: plate far field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_c3_rayleigh_on_axis_matches_exact(frequency):
    """The numerical Rayleigh quadrature is pinned by the analytic on-axis integral."""
    r = 200.0 * PISTON_A
    num = rayleigh_far_field(r, 0.0, PISTON_A, frequency, 1.0)
    exact = on_axis_exact(r, PISTON_A, frequency, 1.0)
    assert abs(num - exact) / abs(exact) < 1e-6


@pytest.mark.parametrize("frequency", FREQUENCIES)
def test_c3_far_field_matches_airy_pattern(frequency):
    """C3: numerical far field within 5 % of the closed-form Airy directivity."""
    for row in angle_sweep_error(PISTON_A, frequency, ANGLES, 200.0, 1.0):
        assert row["magnitude_relative_error"] < 0.05


def test_c3_node_cos_theta_specification_deviation_is_recorded():
    """The node's cos(theta) lobe is 0 % off on axis and large off axis."""
    on_axis = node_cos_theta_far_field(100.0, 0.0, PISTON_A, 500.0, 1.0)
    closed = baffled_piston_far_field(100.0, 0.0, PISTON_A, 500.0, 1.0)
    assert abs(abs(on_axis) - abs(closed)) / abs(closed) < 1e-12
    wide = angle_sweep_error(PISTON_A, 500.0, (60.0,), 200.0, 1.0)[0]
    assert wide["node_cos_theta_relative_error"] > 0.3
    assert wide["magnitude_relative_error"] < 0.05


def test_piston_directivity_on_axis_and_power_identity():
    assert piston_directivity(wavenumber(500.0), PISTON_A, 0.0) == pytest.approx(1.0, rel=1e-6)
    check = piston_power_check(PISTON_A, 500.0, 1.0)
    assert check.relative_error < 1e-6
    assert 0.0 < check.sigma_closed_form <= 1.0


def test_piston_radiation_efficiency_low_frequency_law():
    ka = 0.05
    assert piston_radiation_efficiency(ka) == pytest.approx(ka * ka / 2.0, rel=5e-3)
    assert piston_radiation_efficiency(50.0) == pytest.approx(1.0, abs=5e-3)


def test_bessel_j1_against_series_and_known_values():
    def series(x, n=40):
        total = 0.0
        for m in range(n):
            total += (-1) ** m * (x / 2) ** (2 * m + 1) / (
                math.factorial(m) * math.factorial(m + 1)
            )
        return total

    for x in (0.0, 0.01, 0.5, 1.0, 1.8411837813406593, 3.0, 6.0, 11.9):
        assert bessel_j1(x) == pytest.approx(series(x), rel=1e-11, abs=1e-13)
    assert bessel_j1(1.8411837813406593) == pytest.approx(0.581865, abs=1e-5)
    # asymptotic branch (|x| >= 12) must stay within the documented 1e-7
    for x in (12.5, 15.0, 20.0):
        assert bessel_j1(x) == pytest.approx(series(x), abs=1e-7)
    assert bessel_j1(-1.0) == pytest.approx(-bessel_j1(1.0), rel=1e-12)


def test_far_field_power_matches_closed_form_power():
    for frequency in (100.0, 500.0):
        power = far_field_power_from_directivity(PISTON_A, frequency, 1.0)
        check = piston_power_check(PISTON_A, frequency, 1.0)
        assert power == pytest.approx(check.power_closed_form, rel=1e-6)


# ---------------------------------------------------------------------------
# C4: level law
# ---------------------------------------------------------------------------

def test_level_references_are_iso_1683():
    assert sound_power_level(1.0e-12) == pytest.approx(0.0, abs=1e-12)
    assert velocity_level(1.0e-9) == pytest.approx(0.0, abs=1e-12)
    assert W_REF == 1.0e-12 and V_REF == 1.0e-9


def test_peak_versus_rms_trap_is_3_db():
    assert velocity_level(1.0e-3) - velocity_level(rms_from_peak(1.0e-3)) == pytest.approx(
        10.0 * math.log10(2.0), rel=1e-12
    )
    assert mean_square_from_peak(2.0) == pytest.approx(2.0)


@pytest.mark.parametrize("frequency", (50.0, 100.0, 200.0))
def test_c4_scale_law_residual_within_2db(frequency):
    sphere = monopole_surface_source(A, frequency, 1.0e-3)
    piston = piston_surface_source(PISTON_A, frequency, 1.0e-3)
    for src in (sphere, piston):
        offset = scale_law_offset_db(1000.0, 1500.0, src.area, src.sigma)
        residual = sound_power_level(src.power) - velocity_level(src.v_rms) - offset
        assert abs(residual) <= 2.0
        assert abs(residual) < 1e-9
        assert abs(scale_law_residual_db(src.power, src.v_rms, 1000.0, 1500.0,
                                        src.area, src.sigma)) < 1e-9


def test_c4_velocity_decade_slope_is_one():
    slope = scaling_slope_db(lambda v: monopole_surface_source(A, 200.0, v).power, 1.0e-3)
    assert slope == pytest.approx(1.0, rel=1e-12)
    slope_p = scaling_slope_db(lambda v: piston_surface_source(PISTON_A, 500.0, v).power, 1.0e-3)
    assert slope_p == pytest.approx(1.0, rel=1e-12)


def test_c4_area_doubling_adds_three_db():
    """At fixed sigma and fixed velocity, doubling S must add exactly 3.0103 dB.

    (Doubling the *radius* would also change ka and therefore sigma, so the
    check is done on the derived law constant, where the area term is isolated.)
    """
    c1 = scale_law_offset_db(1000.0, 1500.0, 1.0, 0.7)
    c2 = scale_law_offset_db(1000.0, 1500.0, 2.0, 0.7)
    assert c2 - c1 == pytest.approx(10.0 * math.log10(2.0), rel=1e-12)


def test_scale_law_offset_explicit_value():
    """C = 10log10(rho c S sigma) + 20log10 v_ref - 10log10 W_ref, checked by hand."""
    expected = (
        10.0 * math.log10(1000.0 * 1500.0 * 4.0 * math.pi * 0.5)
        + 20.0 * math.log10(1.0e-9)
        - 10.0 * math.log10(1.0e-12)
    )
    assert scale_law_offset_db(1000.0, 1500.0, 4.0 * math.pi, 0.5) == pytest.approx(
        expected, rel=1e-12
    )


# ---------------------------------------------------------------------------
# C5: coupled energy balance
# ---------------------------------------------------------------------------

def test_shell_modal_parameters():
    shell = spherical_shell_breathing_parameters()
    assert shell.mass == pytest.approx(7850.0 * 0.02 * 4.0 * math.pi, rel=1e-12)
    expected_f0 = math.sqrt(
        2.0 * 210.0e9 / (7850.0 * 0.7 * 1.0)
    ) / (2.0 * math.pi)
    assert shell.natural_frequency == pytest.approx(expected_f0, rel=1e-12)


@pytest.mark.parametrize("frequency", FREQUENCIES)
@pytest.mark.parametrize("zeta", (0.01, 0.05, 0.1))
def test_c5_energy_balance(frequency, zeta):
    shell = spherical_shell_breathing_parameters()
    res = coupled_breathing_sphere(frequency, shell, 1.0e4, zeta, 48)
    assert res.energy_residual < 0.03
    assert res.energy_residual_outer < 0.03
    assert res.energy_residual < 1e-12  # algebraic identity of the coupling


@pytest.mark.parametrize("frequency", (50.0, 200.0, 500.0))
def test_c5_coupled_velocity_matches_analytic_impedance(frequency):
    shell = spherical_shell_breathing_parameters()
    res = coupled_breathing_sphere(frequency, shell, 1.0e4, 0.05, 48)
    assert res.velocity_relative_error < 0.03


def test_c5_measured_efficiency_matches_analytic():
    shell = spherical_shell_breathing_parameters()
    res = coupled_breathing_sphere(200.0, shell, 1.0e4, 0.05, 48)
    assert res.radiation_efficiency == pytest.approx(
        radiation_efficiency(shell.radius, 200.0), rel=1e-2
    )


def test_c5_stored_energy_term_is_required():
    """Negative control: dropping i(k - omega^2 m)/omega breaks the power identity."""
    shell = spherical_shell_breathing_parameters()
    res = coupled_breathing_sphere(200.0, shell, 1.0e4, 0.05, 48)
    omega = 2.0 * math.pi * 200.0
    c_d = shell.damping(0.05)
    # wrong operator (displacement-form stiffness used with velocity unknown)
    wrong = complex(1.0e4) / (complex(shell.stiffness - omega**2 * shell.mass, -omega * c_d)
                              - shell.wetted_area * res.impedance)
    p_in = 0.5 * (1.0e4 * wrong.conjugate()).real
    p_out = 0.5 * shell.wetted_area * res.impedance.real * abs(wrong) ** 2 + 0.5 * c_d * abs(wrong) ** 2
    assert abs(p_in - p_out) / abs(p_in) > 0.5
    # the correct operator satisfies the identity
    assert res.energy_residual < 1e-12


def test_energy_balance_sweep_covers_grid():
    rows = energy_balance_sweep((50.0, 500.0), (0.01, 0.1), elements=32)
    assert len(rows) == 4
    for row in rows:
        assert row["energy_residual"] < 0.03
        assert abs(row["scale_law_residual_db_analytic_sigma"]) < 2.0
        assert row["sigma_measured"] == pytest.approx(row["sigma_analytic"], rel=1e-2)


# ---------------------------------------------------------------------------
# Input validation on the far-field side
# ---------------------------------------------------------------------------

def test_plate_input_validation():
    with pytest.raises(ValueError):
        rayleigh_far_field(-1.0, 0.0, PISTON_A, 100.0)
    with pytest.raises(ValueError):
        rayleigh_far_field(10.0, 2.0, PISTON_A, 100.0)  # theta outside [0, pi/2]
    with pytest.raises(ValueError):
        baffled_piston_far_field(10.0, 0.0, -0.5, 100.0, 1.0)
    assert plate_k(100.0) == pytest.approx(2.0 * math.pi * 100.0 / 1500.0)
