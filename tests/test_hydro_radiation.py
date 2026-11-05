"""Tests for the finite-``ka`` baffled radiation operator.

The module supplies what ``hydro_added_mass`` explicitly leaves out (radiation
damping and finite-frequency acoustics), so the tests are built around the one
*exact* case available in closed form -- the circular piston in an infinite
rigid baffle -- plus a strict consistency anchor against the in-phase operator.
"""

import math

import pytest
import torch

from tensorfem.hydro_added_mass import (
    dipole_added_mass_matrix,
    disc_panels,
    piston_radiation_mass,
    sine_mode_values,
)
from tensorfem.hydro_radiation import (
    baffled_kernel_matrix,
    bessel_j1,
    effective_added_mass,
    modal_radiation_impedance,
    piston_impedance_exact,
    piston_reactance_factor,
    piston_resistance_factor,
    planar_radiation_impedance,
    struve_h1,
)

RHO = 1025.0
C_SOUND = 1500.0
RADIUS = 0.5


def test_special_functions_match_scipy():
    special = pytest.importorskip("scipy.special")
    xs = (0.1, 0.5, 1.0, 2.0, 4.0, 6.0)
    for x in xs:
        j1 = float(bessel_j1(x))
        assert abs(j1 - special.j1(x)) <= 1e-12 * max(1.0, abs(special.j1(x)))
        h1 = float(struve_h1(x))
        assert abs(h1 - special.struve(1, x)) <= 1e-11 * max(1.0, abs(special.struve(1, x)))
        r1 = float(piston_resistance_factor(x))
        assert abs(r1 - (1.0 - 2.0 * special.j1(x) / x)) <= 1e-12 * max(1.0, abs(r1))
        x1 = float(piston_reactance_factor(x))
        assert abs(x1 - 2.0 * special.struve(1, x) / x) <= 1e-11 * max(1.0, abs(x1))


def test_small_argument_limits_reproduce_rayleigh_added_mass():
    tiny = 1e-6
    assert float(piston_resistance_factor(0.0)) == pytest.approx(0.0, abs=1e-18)
    assert float(piston_reactance_factor(0.0)) == pytest.approx(0.0, abs=1e-18)
    # R1 ~ x^2/8, X1 ~ 4x/(3 pi)
    assert float(piston_resistance_factor(tiny)) == pytest.approx(tiny**2 / 8.0, rel=1e-12)
    assert float(piston_reactance_factor(tiny)) == pytest.approx(
        4.0 * tiny / (3.0 * math.pi), rel=1e-12
    )


def test_exact_piston_impedance_limits():
    a = RADIUS
    k = 1e-7
    z = piston_impedance_exact(a, k, RHO, C_SOUND)
    # Im(Z)/w is the added mass -> Rayleigh (8/3) rho a^3
    assert z.imag / (k * C_SOUND) == pytest.approx(piston_radiation_mass(a, RHO), rel=1e-6)
    # Re(Z) ~ rho c pi a^2 (ka)^2 / 2
    scale = RHO * C_SOUND * math.pi * a * a
    assert z.real / scale == pytest.approx((k * a) ** 2 / 2.0, rel=1e-5)
    # High ka: the resistance approaches rho c A; R1(x) -> 1 oscillating and
    # overshoots by ~9 % (R1(4) = 1.033, R1(6) = 1.092) before settling.
    for ka in (2.0, 4.0, 5.0):
        z_big = piston_impedance_exact(a, ka / a, RHO, C_SOUND)
        assert 0.0 < z_big.real <= scale * 1.10
    # R1(10) = 1.0087: saturated on the characteristic impedance rho c A.
    assert piston_impedance_exact(a, 5.0 / a, RHO, C_SOUND).real == pytest.approx(scale, rel=2e-2)


@pytest.mark.parametrize("ka", [0.1, 1.0, 2.0])
def test_discretised_piston_matches_exact_impedance(ka):
    k = ka / RADIUS
    exact = piston_impedance_exact(RADIUS, k, RHO, C_SOUND)
    centres, areas = disc_panels(RADIUS, 64)
    z = planar_radiation_impedance(centres, areas, k, RHO, C_SOUND)
    assert abs(z.real - exact.real) / exact.real < 0.03
    assert abs(z.imag - exact.imag) / exact.imag < 0.03


def test_piston_error_decreases_with_mesh_refinement():
    k = 1.0 / RADIUS
    exact = piston_impedance_exact(RADIUS, k, RHO, C_SOUND)
    errors = []
    for n in (16, 32, 64):
        centres, areas = disc_panels(RADIUS, n)
        z = planar_radiation_impedance(centres, areas, k, RHO, C_SOUND)
        errors.append(abs(z.real - exact.real) / exact.real)
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.03


def test_finite_k_operator_collapses_onto_in_phase_operator():
    centres, areas = disc_panels(RADIUS, 32)
    in_phase = dipole_added_mass_matrix(centres, areas, RHO)
    finite = baffled_kernel_matrix(centres, areas, 1e-8, RHO, C_SOUND)
    deviation = float((finite.real - in_phase).abs().sum()) / float(in_phase.abs().sum())
    assert deviation < 1e-9
    # Reciprocity: the complex operator is symmetric, not Hermitian.
    assert float((finite - finite.transpose(0, 1)).abs().max()) == pytest.approx(0.0, abs=1e-30)
    assert float((finite - finite.conj().transpose(0, 1)).abs().max()) > 0.0


def test_complex_kernel_diagonal_is_the_exact_patch_integral():
    centres, areas = disc_panels(RADIUS, 16)
    k = 2.0
    matrix = baffled_kernel_matrix(centres, areas, k, RHO, C_SOUND)
    radii = torch.sqrt(areas / math.pi)
    expected = RHO * areas * (1.0 - torch.exp(-1j * k * radii)) / (1j * k)
    assert torch.allclose(torch.diagonal(matrix), expected, rtol=1e-12, atol=0.0)
    # k -> 0: the patch value reduces to rho A a (the in-phase self term).
    low = baffled_kernel_matrix(centres, areas, 1e-10, RHO, C_SOUND)
    assert torch.allclose(torch.diagonal(low).real, RHO * areas * radii, rtol=1e-8)


def _panel_mesh(lx=1.2, ly=0.8, nx=12, ny=8):
    dx, dy = lx / nx, ly / ny
    xs = dx * (torch.arange(nx, dtype=torch.float64) + 0.5)
    ys = dy * (torch.arange(ny, dtype=torch.float64) + 0.5)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    centres = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    areas = torch.full((centres.shape[0],), dx * dy, dtype=torch.float64)
    return centres, areas


def test_modal_impedance_reduces_to_piston_for_unit_velocity():
    centres, areas = _panel_mesh()
    ones = torch.ones(centres.shape[0], dtype=torch.float64)
    k = 0.4
    z_piston = planar_radiation_impedance(centres, areas, k, RHO, C_SOUND)
    z_mode = modal_radiation_impedance(centres, areas, ones, k, RHO, C_SOUND)
    assert abs(z_mode - z_piston) <= 1e-12 * abs(z_piston)


def test_modal_low_frequency_reactance_reproduces_in_phase_added_mass():
    centres, areas = _panel_mesh()
    phi = sine_mode_values(centres, 1.2, 0.8, 1, 1)
    matrix = dipole_added_mass_matrix(centres, areas, RHO)
    reference = float(phi @ (matrix @ phi))
    k = 1e-8
    z = modal_radiation_impedance(centres, areas, phi, k, RHO, C_SOUND)
    assert z.imag / (k * C_SOUND) == pytest.approx(reference, rel=1e-6)


def test_finite_frequency_added_mass_collapses_above_ka_one():
    low = piston_radiation_mass(RADIUS, RHO)
    centres, areas = disc_panels(RADIUS, 64)
    collapse = []
    for ka in (0.5, 2.0):
        k = ka / RADIUS
        z = planar_radiation_impedance(centres, areas, k, RHO, C_SOUND)
        collapse.append(effective_added_mass(z, k * C_SOUND) / low)
    assert collapse[0] > collapse[1]
    # ka = 0.5 still carries ~93 % of the in-phase mass ...
    assert collapse[0] == pytest.approx(0.935, abs=0.02)
    # ... and by ka = 2 only ~31 % is left: the in-phase mass is not usable there.
    assert collapse[1] == pytest.approx(0.315, abs=0.02)


def test_radiation_resistance_is_non_negative_and_bounded():
    centres, areas = disc_panels(RADIUS, 32)
    scale = RHO * C_SOUND * math.pi * RADIUS**2
    for ka in (0.05, 0.5, 1.0, 3.0):
        k = ka / RADIUS
        z = planar_radiation_impedance(centres, areas, k, RHO, C_SOUND)
        # R1 oscillates about 1 and overshoots, so the bound is 1.10 * rho c A.
        assert 0.0 <= z.real < 1.15 * scale


def test_effective_added_mass_rejects_invalid_frequency():
    with pytest.raises(ValueError):
        effective_added_mass(1.0 + 0.0j, 0.0)
    with pytest.raises(ValueError):
        effective_added_mass(1.0 + 0.0j, -1.0)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        piston_impedance_exact(0.0, 1.0)
    with pytest.raises(ValueError):
        piston_impedance_exact(RADIUS, -1.0)
    with pytest.raises(ValueError):
        piston_impedance_exact(RADIUS, 1.0, rho_water=-1.0)
    with pytest.raises(ValueError):
        piston_impedance_exact(RADIUS, 1.0, sound_speed=0.0)
    with pytest.raises(ValueError):
        struve_h1(50.0)
    centres, areas = disc_panels(RADIUS, 8)
    with pytest.raises(ValueError):
        baffled_kernel_matrix(centres[:, :1], areas, 0.5)
    with pytest.raises(ValueError):
        baffled_kernel_matrix(centres, areas[:-1], 0.5)
    with pytest.raises(ValueError):
        baffled_kernel_matrix(centres, -areas, 0.5)
    with pytest.raises(ValueError):
        modal_radiation_impedance(centres, areas, torch.ones(3), 0.5)
