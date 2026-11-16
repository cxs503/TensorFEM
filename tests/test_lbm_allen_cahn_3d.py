"""Tests for the Allen-Cahn LBM solver (``lbm_allen_cahn_3d``).

Covers:
1. double-well potential and derivative consistency,
2. tanh profile analytical properties,
3. diffusion decay (cosine mode) validates the LBM diffusion operator,
4. shrinking sphere follows the analytical dR²/dt = −4M law,
5. D3Q19 is at least as accurate as D3Q15 for the shrinking sphere,
6. equilibrium initialisation and macroscopic recovery,
7. mass conservation under pure diffusion,
8. edge cases and error paths.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.lbm_allen_cahn_3d import (  # noqa: E402
    AllenCahnConfig,
    AllenCahnLBM,
    compare_d3q15_d3q19,
    double_well,
    double_well_derivative,
    effective_radius,
    run_diffusion_decay,
    run_shrinking_sphere,
    tanh_profile_1d,
)
from tensorfem.lbm_lattice_tables import get_table  # noqa: E402


# ---------------------------------------------------------------------------
# 1. double-well potential and derivative
# ---------------------------------------------------------------------------
def test_double_well_zeros():
    """f(φ) = 0 at φ = 0 and φ = 1."""
    phi = np.array([0.0, 1.0])
    assert np.allclose(double_well(phi), 0.0, atol=1e-15)


def test_double_well_derivative_zeros():
    """f'(φ) = 0 at φ = 0, φ = ½, and φ = 1."""
    phi = np.array([0.0, 0.5, 1.0])
    assert np.allclose(double_well_derivative(phi), 0.0, atol=1e-15)


def test_double_well_derivative_consistency():
    """Numerical derivative of f matches f' at several interior points."""
    phi = np.linspace(0.05, 0.95, 20)
    h = 1e-7
    f_num = (double_well(phi + h) - double_well(phi - h)) / (2 * h)
    f_ana = double_well_derivative(phi)
    assert np.max(np.abs(f_num - f_ana)) < 1e-5


def test_double_well_positive():
    """f(φ) > 0 for 0 < φ < 1."""
    phi = np.linspace(0.01, 0.99, 50)
    assert np.all(double_well(phi) > 0)


# ---------------------------------------------------------------------------
# 2. tanh profile
# ---------------------------------------------------------------------------
def test_tanh_profile_bounds():
    """φ ∈ [0, 1] for the tanh profile."""
    x = np.linspace(-50, 50, 200)
    phi = tanh_profile_1d(x, epsilon=2.0)
    assert phi.min() >= -1e-15
    assert phi.max() <= 1.0 + 1e-15


def test_tanh_profile_symmetry():
    """φ(x₀ + δ) + φ(x₀ − δ) = 1 for the tanh profile."""
    x0 = 10.0
    eps = 3.0
    delta = np.linspace(0, 20, 50)
    phi_plus = tanh_profile_1d(x0 + delta, eps, x0)
    phi_minus = tanh_profile_1d(x0 - delta, eps, x0)
    assert np.allclose(phi_plus + phi_minus, 1.0, atol=1e-14)


def test_tanh_profile_is_steady_state():
    """The tanh profile satisfies the Allen-Cahn ODE: 0 = φ'' − f'(φ)/ε²."""
    eps = 4.0
    x = np.linspace(0, 64, 500)
    phi = tanh_profile_1d(x, eps, 32.0)
    dx = x[1] - x[0]
    phi_xx = np.gradient(np.gradient(phi, dx), dx)
    fp = double_well_derivative(phi)
    residual = phi_xx - fp / eps ** 2
    # the residual should be small (limited by numerical differentiation)
    assert np.max(np.abs(residual)) < 1e-4


# ---------------------------------------------------------------------------
# 3. equilibrium initialisation and macroscopic recovery
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lattice_name", ("D3Q15", "D3Q19"))
def test_equilibrium_init_recovers_phi(lattice_name):
    """g_i^eq = w_i φ  =>  Σ g_i = φ."""
    cfg = AllenCahnConfig(lattice_name=lattice_name, nx=8, ny=8, nz=8, tau=1.0, epsilon=2.0)
    solver = AllenCahnLBM(cfg)
    rng = np.random.default_rng(42)
    phi_in = rng.random((8, 8, 8))
    solver.init_field(phi_in)
    phi_out = solver.phi
    assert np.max(np.abs(phi_out - phi_in)) < 1e-14


@pytest.mark.parametrize("lattice_name", ("D3Q15", "D3Q19"))
def test_cosine_init_recovers_profile(lattice_name):
    """Initialising with a cosine mode and reading back φ gives the same."""
    n = 16
    cfg = AllenCahnConfig(lattice_name=lattice_name, nx=n, ny=n, nz=n, tau=1.0, epsilon=4.0)
    solver = AllenCahnLBM(cfg)
    solver.init_cosine_mode(amplitude=0.1, mode=1, direction=0)
    phi = solver.phi
    # check that the profile is a cosine
    mid_y = n // 2
    mid_z = n // 2
    profile = phi[mid_z, mid_y, :]
    x = np.arange(n, dtype=float) + 0.5
    expected = 0.5 + 0.1 * np.cos(2.0 * np.pi * x / n)
    assert np.max(np.abs(profile - expected)) < 1e-14


# ---------------------------------------------------------------------------
# 4. diffusion decay validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lattice_name", ("D3Q15", "D3Q19"))
def test_diffusion_decay_recovers_viscosity(lattice_name):
    """The fitted diffusivity from cosine-mode decay must match D = cs²(τ−½) within 5%."""
    result = run_diffusion_decay(
        lattice_name=lattice_name,
        n=32, tau=1.0, amplitude=0.1, mode=1,
        steps=200, probe_every=10,
    )
    assert result["D_rel_err"] < 0.05, (
        f"{lattice_name}: D_rel_err = {result['D_rel_err']:.4f}"
    )


@pytest.mark.parametrize("lattice_name", ("D3Q15", "D3Q19"))
def test_diffusion_decay_amplitude_decreases(lattice_name):
    """The amplitude of the cosine mode must decrease monotonically."""
    result = run_diffusion_decay(
        lattice_name=lattice_name,
        n=32, tau=1.0, amplitude=0.1, mode=1,
        steps=200, probe_every=10,
    )
    amps = result["amplitudes"]
    # amplitude should decrease
    assert amps[-1] < amps[0], f"{lattice_name}: amplitude did not decrease"


@pytest.mark.parametrize("tau", (0.6, 0.8, 1.0, 1.5, 2.0))
def test_diffusion_decay_various_tau(tau):
    """The diffusion decay test works for a range of relaxation times."""
    result = run_diffusion_decay(
        lattice_name="D3Q19",
        n=32, tau=tau, amplitude=0.05, mode=1,
        steps=200, probe_every=10,
    )
    assert result["D_rel_err"] < 0.05, (
        f"tau={tau}: D_rel_err = {result['D_rel_err']:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. shrinking sphere validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lattice_name", ("D3Q15", "D3Q19"))
def test_shrinking_sphere_radius_decreases(lattice_name):
    """The sphere must shrink (R_final < R_initial)."""
    result = run_shrinking_sphere(
        lattice_name=lattice_name,
        n=32, tau=1.0, epsilon=3.0,
        radius0=10.0, steps=500,
        report_every=100,
    )
    assert result["R_final_lbm"] < result["radii"][0], (
        f"{lattice_name}: sphere did not shrink"
    )


@pytest.mark.parametrize("lattice_name", ("D3Q15", "D3Q19"))
def test_shrinking_sphere_dR2_dt_error_below_15_percent(lattice_name):
    """The dR²/dt error vs the analytical −4M must be < 15%."""
    result = run_shrinking_sphere(
        lattice_name=lattice_name,
        n=40, tau=1.0, epsilon=3.0,
        radius0=12.0, steps=150,
        report_every=30,
    )
    assert result["dR2_dt_rel_err"] < 0.15, (
        f"{lattice_name}: dR²/dt error {result['dR2_dt_rel_err']:.4f} >= 15%"
    )


def test_d3q19_at_least_as_accurate_as_d3q15():
    """D3Q19 (rank-4 isotropy) should not be worse than D3Q15 (rank-3)."""
    result = compare_d3q15_d3q19(
        n=40, tau=1.0, epsilon=3.0,
        radius0=12.0, steps=150,
        report_every=30,
    )
    assert result["d3q19_more_accurate"], (
        f"D3Q15 err={result['d3q15_dR2_dt_rel_err']:.4f}, "
        f"D3Q19 err={result['d3q19_dR2_dt_rel_err']:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. effective_radius
# ---------------------------------------------------------------------------
def test_effective_radius_of_uniform_sphere():
    """A uniform sphere of radius R gives the correct effective radius."""
    n = 50
    center = (25.0, 25.0, 25.0)
    R = 15.0
    zz, yy, xx = np.meshgrid(
        np.arange(n, dtype=float),
        np.arange(n, dtype=float),
        np.arange(n, dtype=float),
        indexing="ij",
    )
    dist = np.sqrt((xx + 0.5 - center[0]) ** 2 + (yy + 0.5 - center[1]) ** 2 + (zz + 0.5 - center[2]) ** 2)
    phi = (dist < R).astype(float)
    R_eff = effective_radius(phi, dx=1.0)
    # discrete volume is approximate; allow 5% tolerance
    assert abs(R_eff - R) / R < 0.05


# ---------------------------------------------------------------------------
# 7. mass conservation (diffusion only)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lattice_name", ("D3Q15", "D3Q19"))
def test_mass_conservation_diffusion_only(lattice_name):
    """With no reaction, total φ must be conserved (periodic BCs)."""
    cfg = AllenCahnConfig(lattice_name=lattice_name, nx=16, ny=16, nz=16,
                          tau=1.0, epsilon=4.0, steps=100)
    solver = AllenCahnLBM(cfg)
    solver.init_cosine_mode(amplitude=0.1, mode=1, direction=0)
    mass0 = solver.phi.sum()
    solver.run(100, disable_reaction=True)
    mass1 = solver.phi.sum()
    assert abs(mass1 - mass0) / abs(mass0) < 1e-10, (
        f"{lattice_name}: mass not conserved, rel err = {abs(mass1-mass0)/abs(mass0):.2e}"
    )


# ---------------------------------------------------------------------------
# 8. error paths
# ---------------------------------------------------------------------------
def test_tau_leq_half_raises():
    with pytest.raises(ValueError, match="tau must exceed 0.5"):
        AllenCahnLBM(AllenCahnConfig(tau=0.4))


def test_unknown_lattice_raises():
    with pytest.raises(KeyError):
        AllenCahnLBM(AllenCahnConfig(lattice_name="D9Q99"))


# ---------------------------------------------------------------------------
# 9. source correction factor
# ---------------------------------------------------------------------------
def test_source_correction_factor():
    """The Chapman-Enskog correction factor must be 1/(1 − 1/(2τ))."""
    for tau in (0.6, 0.8, 1.0, 1.5, 2.0, 3.0):
        cfg = AllenCahnConfig(tau=tau)
        solver = AllenCahnLBM(cfg)
        expected = 1.0 / (1.0 - 1.0 / (2.0 * tau))
        assert abs(solver._source_correction - expected) < 1e-14, (
            f"tau={tau}: correction = {solver._source_correction}, expected = {expected}"
        )
