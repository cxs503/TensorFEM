"""Tests for the two-way coupled LBM / Euler-Bernoulli FSI solver.

The tests are deliberately cheap (short runs) and check physical invariants:
mass/velocity boundedness of the lattice solver, analytic beam response,
momentum-conserving interface transfer, and a smoke test of the partitioned
coupling with its sub-iteration diagnostics.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.fsi_lbm_strong_coupling import (
    CantileverBeamFE,
    FSI2Parameters,
    ModalCantileverBeam,
    StrongCouplingFSI2,
    _dominant_frequency,
)
from tensorfem.lbm_channel2d import (
    ChannelConfig,
    ChannelLBM,
    ImmersedBody,
    ibm_force,
    macroscopic,
    peskin_delta,
)


# ---------------------------------------------------------------------------
# Lattice-Boltzmann core
# ---------------------------------------------------------------------------

def test_equilibrium_recovers_moments():
    rho = np.full((6, 5), 1.3)
    ux = np.full((6, 5), 0.02)
    uy = np.full((6, 5), -0.01)
    from tensorfem.lbm_channel2d import equilibrium
    f = equilibrium(rho, ux, uy)
    r, vx, vy = macroscopic(f)
    assert np.allclose(r, rho, atol=1e-12)
    assert np.allclose(vx, ux, atol=1e-12)
    assert np.allclose(vy, uy, atol=1e-12)


def test_peskin_delta_partition_of_unity():
    r = np.linspace(-2.5, 2.5, 501)
    d = peskin_delta(r)
    assert np.trapezoid(d, r) == pytest.approx(1.0, abs=5e-3)
    # four-point support
    assert np.all(peskin_delta(np.array([2.0 + 1e-9])) == 0.0)


def test_channel_flow_is_bounded_and_mass_conserving():
    cfg = ChannelConfig(nx=60, ny=26, u_mean=0.08, nu=0.02)
    s = ChannelLBM(cfg)
    for _ in range(120):
        s.step()
    ux, uy = s.velocity()
    assert np.isfinite(ux).all() and np.isfinite(uy).all()
    assert np.abs(ux).max() < 0.3
    rho = s.density()
    fluid = ~s.solid
    assert rho[fluid].min() > 0.8 and rho[fluid].max() < 1.2


def test_cylinder_drag_is_positive_and_plausible():
    """Re = 100 cylinder in the channel: CD = 3.2 +- 25 % at coarse resolution."""
    p = FSI2Parameters()
    cells = 12
    dx = p.D / cells
    u_lat, nu_lat = 0.08, 0.08 * cells / 100.0
    cfg = ChannelConfig(nx=int(p.W / dx), ny=int(p.H / dx), u_mean=u_lat,
                        nu=nu_lat,
                        cylinder_center=(p.cx / dx, p.cy / dx),
                        cylinder_radius=0.5 * p.D / dx,
                        perturbation=0.02 * u_lat)
    s = ChannelLBM(cfg)
    hist = []
    for _ in range(1500):
        s.step()
        hist.append(s.momentum_exchange_force())
    hist = np.array(hist)[750:]
    cd = 2.0 * hist[:, 0].mean() / (u_lat ** 2 * cells)
    assert 2.0 < cd < 4.5, f"CD={cd}"


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_cantilever_static_tip_deflection_matches_analytic():
    p = FSI2Parameters()
    beam = CantileverBeamFE(p.L, 10, p.EI, p.mass_per_length)
    F = np.zeros_like(beam.q)
    F[2 * beam.n_elem] = 1.0                    # unit tip force
    q = np.zeros_like(beam.q)
    q[beam.free] = np.linalg.solve(beam.Kff, F[beam.free])
    analytic = p.L ** 3 / (3.0 * p.EI)
    assert q[2 * beam.n_elem] / analytic == pytest.approx(1.0, rel=1e-3)


def test_cantilever_first_frequency_matches_analytic():
    p = FSI2Parameters()
    mb = ModalCantileverBeam(p.L, 14, p.EI, p.mass_per_length, n_modes=3)
    assert mb.frequencies[0] == pytest.approx(p.dry_first_frequency, rel=2e-3)


def test_newmark_conserves_energy_for_first_mode():
    p = FSI2Parameters()
    dt = 2.5e-4
    mb = ModalCantileverBeam(p.L, 14, p.EI, p.mass_per_length, n_modes=3)
    mb.initialise_effective(dt)
    mb.eta = np.array([0.02, 0.0, 0.0])
    mb._sync_full_state()
    e0 = mb.kinetic_energy() + mb.strain_energy()
    for _ in range(4000):                       # ~1 s, two periods
        mb.step(np.zeros_like(mb.q))
    e1 = mb.kinetic_energy() + mb.strain_energy()
    assert abs(e1 - e0) / e0 < 1e-3


def test_modal_truncation_is_accurate_for_static_load():
    p = FSI2Parameters()
    beam = CantileverBeamFE(p.L, 14, p.EI, p.mass_per_length)
    F = np.zeros_like(beam.q)
    for e in range(beam.n_elem):
        F[2 * e] += beam.le / 2
        F[2 * e + 2] += beam.le / 2
    exact = np.linalg.solve(beam.Kff, F[beam.free])[-2]
    mb = ModalCantileverBeam(p.L, 14, p.EI, p.mass_per_length, n_modes=8)
    eta = np.linalg.solve(np.diag(mb.omega2), mb.Phi.T @ F[mb.free])
    approx = (mb.Phi @ eta)[-2]
    assert abs(approx - exact) / abs(exact) < 1e-3


# ---------------------------------------------------------------------------
# Interface and coupling
# ---------------------------------------------------------------------------

def test_ibm_kernel_conserves_momentum():
    body = ImmersedBody(np.array([10.0]), np.array([10.0]), 25, 25)
    values = np.array([[0.3, -0.2]])
    grid = body.spread(values)
    assert grid[0].sum() == pytest.approx(values[0, 0], rel=1e-12)
    assert grid[1].sum() == pytest.approx(values[0, 1], rel=1e-12)


def test_interface_load_projection_conserves_resultant():
    p = FSI2Parameters()
    fsi = StrongCouplingFSI2(p, cells_per_D=12, u_lat=0.1)
    itf = fsi.interface
    rng = np.random.default_rng(3)
    fmap = np.zeros((2, fsi.ny, fsi.nx))
    fmap[1][:, itf.cols] = rng.normal(size=(fsi.ny, itf.cols.size))
    b, C, n_links, full_sum = itf.exchange_terms(
        fsi.beam.q, np.zeros_like(fmap[1]), fmap, 1.0)
    assert full_sum == pytest.approx(fmap[1][:, itf.cols].sum(), rel=1e-9)
    assert n_links > 0


def test_strong_coupling_smoke_run():
    p = FSI2Parameters()
    fsi = StrongCouplingFSI2(p, cells_per_D=10, u_lat=0.08,
                             max_sub_iterations=8, tolerance=1e-4)
    h = fsi.run(400, record_every=10)
    assert np.isfinite(h.tip_displacement).all()
    assert np.isfinite(h.drag).all()
    # after the start-up transient the cylinder drag must be positive and of
    # the expected order (rho U^2 D / 2 * CD with CD ~ 3)
    assert np.mean(h.drag[-5:]) > 0.0
    assert abs(np.mean(h.drag[-5:])) < 1e4
    assert max(h.sub_iterations) <= 8
    assert np.mean(h.sub_iterations) < 8        # the fixed point does converge


def test_dominant_frequency_of_pure_tone():
    t = np.arange(0.0, 5.0, 1e-3)
    y = np.sin(2 * math.pi * 2.5 * t)
    assert _dominant_frequency(t, y) == pytest.approx(2.5, abs=0.01)