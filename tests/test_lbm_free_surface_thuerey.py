"""Verification of the Thürey free-surface VOF mass exchange.

Anchor: XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE.

The suite is one test per claim:

1. interface/interface flux is antisymmetric -> the exchange conserves to
   machine precision (passes);
2. the exchange re-places a surplus on neighbouring capacity instead of
   dropping it (passes);
3. the naive clipping baseline used by many implementations does drop it —
   that is the defect the exchange repairs (passes);
4. a quiescent pool keeps its liquid mass exactly (passes);
5. a resting column must reproduce the analytic hydrostatic pressure profile —
   *documented failure* (xfail), numbers in the report;
6. the dam-break front must match Ritter (1892) — *documented failure*
   (xfail), numbers in the report.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfem.lbm_free_surface_thuerey import (
    FLUID,
    GAS,
    INTERFACE,
    FreeSurfaceConfig,
    FreeSurfaceLBM,
    hydrostatic_density,
    ritter_front,
    ritter_surface,
)


def _interface_cluster(ny=9, nx=9, seed=3):
    """A block of pure interface cells surrounded by gas (no fluid cells)."""
    s = FreeSurfaceLBM(FreeSurfaceConfig(nx=nx, ny=ny))
    s.flags[1:-1, 1:-1] = INTERFACE
    s.mass[1:-1, 1:-1] = 0.5
    rng = np.random.default_rng(seed)
    s.f = rng.random((9, ny, nx)) * 0.05 + 0.02
    s.f[:, s.flags == GAS] = 0.0
    return s


def test_interface_pair_flux_is_antisymmetric():
    """sum(dm) over an all-interface cluster must vanish (pair cancellation)."""
    s = _interface_cluster()
    dm = s._mass_flux(s.f.copy())
    iface = s.flags == INTERFACE
    total = float(dm[iface].sum())
    scale = float(np.abs(dm[iface]).sum())
    assert scale > 0.0                      # the check is not vacuous
    assert abs(total) < 1e-14 * max(1.0, scale)


def test_exchange_moves_surplus_without_losing_it():
    """An overflowing cell's surplus lands on neighbour capacity, not the bin."""
    s = _interface_cluster()
    rho = np.full_like(s.mass, 1.0)
    s.mass[4, 4] = 1.0 + 0.37                # over full by 0.37
    before = s.mass.sum()
    s._exchange(rho)
    after = s.mass.sum()
    assert abs(after - before) < 1e-12
    assert s.unplaced_mass < 1e-12
    iface = s.flags == INTERFACE
    assert np.all(s.mass[iface] <= rho[iface] + 1e-12)   # capacity respected


def test_clipping_baseline_loses_that_same_surplus():
    """The clipping variant (mass_exchange=False) drops the surplus: the defect."""
    surplus = 0.37
    ok = _interface_cluster()
    ok.mass[4, 4] = 1.0 + surplus
    ok._exchange(np.full_like(ok.mass, 1.0))

    naive = _interface_cluster()
    naive.cfg.mass_exchange = False
    naive.mass[4, 4] = 1.0 + surplus
    naive._exchange(np.full_like(naive.mass, 1.0))

    assert ok.unplaced_mass < 1e-12                       # conservative
    assert naive.unplaced_mass > 0.9 * surplus            # the surplus is gone
    assert ok.total_mass() > naive.total_mass() + 0.9 * surplus


def test_quiescent_pool_holds_liquid_mass_exactly():
    """No gravity, no flow: the liquid mass must not drift at all."""
    cfg = FreeSurfaceConfig(nx=32, ny=24, gravity=0.0, nu=2e-3)
    s = FreeSurfaceLBM(cfg)
    s.solid_bottom(1)
    s.fill_block(2, 18, 1, 18)
    m0 = s.total_mass()
    for _ in range(200):
        s.step()
    assert m0 > 0
    assert abs(s.total_mass() - m0) < 1e-12 * m0
    assert s.unplaced_mass < 1e-12


@pytest.mark.xfail(reason="documented FAIL: density slope does not settle to 3g within budget")
def test_hydrostatic_pressure_profile():
    """A resting column must build rho(y) = 1 + 3 g (h - y)."""
    gravity = 1e-5
    cfg = FreeSurfaceConfig(nx=32, ny=56, gravity=gravity, nu=2e-3)
    s = FreeSurfaceLBM(cfg)
    s.solid_bottom(1)
    s.fill_block(1, 31, 1, 41)
    for _ in range(1500):
        s.step()
    rho, _, uy = s.macroscopic()
    ys = np.arange(4, 39)
    slope = float(np.polyfit(ys, rho[4:39, 16], 1)[0])
    err = abs(slope - 3.0 * gravity) / (3.0 * gravity)
    assert err < 0.35, f"hydrostatic slope error {err:.1%} (slope {slope:.3e})"
    assert np.abs(uy).max() < 0.05


@pytest.mark.xfail(reason="documented FAIL: front speed is ~4x too slow (see report)")
def test_dam_break_front_matches_ritter():
    """Ritter dry-bed front: x_f = 2 sqrt(g h0) t (error < 15% required)."""
    g, h0, t = 1e-5, 40.0, 1200.0
    cfg = FreeSurfaceConfig(nx=160, ny=60, gravity=g, nu=2e-3)
    s = FreeSurfaceLBM(cfg)
    s.solid_bottom(1)
    s.fill_block(2, 40, 1, 41)
    for _ in range(int(t)):
        s.step()
    front = s.front_position()
    ref = float(ritter_front(t, g, h0, dam_x=40.0))
    err = abs(front - ref) / (ref - 40.0)
    assert err < 0.15, f"front {front} vs Ritter {ref:.1f} -> {err:.0%}"


def test_ritter_reference_is_self_consistent():
    """Sanity: the analytic helpers satisfy the invariants used above."""
    g, h0, t = 1e-5, 40.0, 500.0
    xf = float(ritter_front(t, g, h0))
    assert abs(xf - 2.0 * np.sqrt(g * h0) * t) < 1e-12
    h = ritter_surface(np.array([xf - 1e-9, 0.0, xf + 1.0]), t, g, h0)
    assert h[2] == 0.0
    assert h[0] < 1e-6
    assert abs(float(hydrostatic_density(np.array([0.0]), 10.0, 1e-3)[0]) - 1.03) < 1e-12