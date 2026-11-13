"""Verification tests for the moving-wall half-way bounce-back closure.

Anchor XVAL-LBM-MOVING-WALL-HALFWAY-BB-FLOAT64.  The tests are cheap
(`ny = 10`, ~2k steps, < 2 s each) but assert *invariants*, not just that code
runs:

* the Couette profile equals the exact straight line built on the half-way
  wall faces (relative L-inf < 1e-9 in float64; measured ~1e-12),
* the effective wall position extracted by extrapolating the profile lands on
  the wall face, i.e. half a cell away from the adjacent fluid node,
* the momentum-exchange wall force equals the analytic Couette stress
  `rho nu U / H`,
* the total lattice mass is conserved (a tangential moving wall must not act
  as a mass source),
* float32 storage loses several orders of accuracy with respect to float64.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfem.lbm_channel2d import ChannelConfig, ChannelLBM


U = 0.05
NY = 10
NU = 0.1


def _converge(sim: ChannelLBM, body=None, tol: float = 1e-13,
              max_steps: int = 20000, every: int = 25):
    prev = None
    for n in range(1, max_steps + 1):
        sim.step(body)
        if n % every == 0:
            ux, _ = sim.velocity()
            u = ux[1:-1, 0].copy()
            if prev is not None and np.abs(u - prev).max() < tol:
                return sim, n
            prev = u
    return sim, max_steps


def _couette_sim(**kw) -> ChannelLBM:
    cfg = ChannelConfig(nx=4, ny=NY, u_mean=0.0, nu=NU, u_wall_top=U,
                        periodic_x=True, **kw)
    return ChannelLBM(cfg)


def test_moving_wall_profile_is_exact_couette_line():
    sim, steps = _converge(_couette_sim())
    ux, uy = sim.velocity()
    H = NY - 2.0
    s = np.arange(1, NY - 1) - 0.5           # distance from the bottom wall face
    u_exact = U * s / H
    rel = np.abs(ux[1:-1, 0] - u_exact).max() / U
    assert rel < 1e-9, f"rel L-inf={rel:.3e} after {steps} steps"
    assert np.abs(uy).max() < 1e-12
    # fully developed: velocity independent of the streamwise index
    assert np.ptp(ux[1:-1, :], axis=1).max() < 1e-14


def test_effective_wall_position_is_half_a_cell_from_the_node():
    sim, _ = _converge(_couette_sim())
    ux, _ = sim.velocity()
    y = np.arange(1, NY - 1) - 0.5
    profile = ux[1:-1, 0]
    lo = np.polyfit(y[:3], profile[:3], 1)
    hi = np.polyfit(y[-3:], profile[-3:], 1)
    s_bottom = -lo[1] / lo[0]
    s_top = (U - hi[1]) / hi[0]
    # s is measured from the bottom wall face: the extrapolated no-slip point
    # must sit on the face (s = 0) and the extrapolated u = U point on the top
    # face (s = H), i.e. half a cell away from the adjacent node centres.
    assert abs(s_bottom) < 1e-7, f"bottom wall at s={s_bottom:.2e}"
    assert abs(s_top - (NY - 2.0)) < 1e-7, f"top wall at s={s_top:.2e}"
    assert abs((s_bottom - 0.5) + 0.5) < 1e-7     # -0.5 dx from first node
    assert abs((s_top - (NY - 2.5)) - 0.5) < 1e-7  # +0.5 dx from last node


def test_wall_force_equals_analytic_couette_stress():
    sim, _ = _converge(_couette_sim())
    force = sim.wall_force()
    sigma = NU * U / (NY - 2.0)              # rho = 1, per unit depth
    f_num = abs(force[1, 0]) / sim.nx
    assert abs(f_num - sigma) / sigma < 1e-8
    # the bottom wall at rest carries the same stress, opposite sign
    assert force[0, 0] > 0.0 > force[1, 0]
    assert abs(abs(force[0, 0]) / sim.nx - sigma) / sigma < 1e-8
    # wall-normal component = pressure force rho cs^2 per node (rho = 1 here)
    assert abs(abs(force[1, 1]) / sim.nx - 1.0 / 3.0) < 1e-6
    assert abs(abs(force[0, 1]) / sim.nx - 1.0 / 3.0) < 1e-6


def test_tangential_moving_wall_conserves_mass():
    sim = _couette_sim()
    m0 = float(sim.f.sum())
    for _ in range(1500):
        sim.step()
    assert abs(float(sim.f.sum()) - m0) / m0 < 1e-11


def test_rest_wall_limits_are_untouched():
    """U = 0 in the periodic box must stay at rest (no spurious forcing)."""
    cfg = ChannelConfig(nx=4, ny=NY, u_mean=0.0, nu=NU, periodic_x=True)
    sim = ChannelLBM(cfg)
    for _ in range(200):
        sim.step()
    ux, uy = sim.velocity()
    assert np.abs(ux).max() < 1e-14 and np.abs(uy).max() < 1e-14
    force = sim.wall_force()
    assert np.all(force[:, 0] == 0.0)          # no tangential momentum flux
    # only the wall-normal (pressure) part survives: rho cs^2 per node
    assert abs(abs(force[0, 1]) / sim.nx - 1.0 / 3.0) < 1e-12


def test_couette_poiseuille_matches_analytic_profile_and_balance():
    g = 1e-5
    sim = _couette_sim()
    body = np.zeros((2, NY, sim.nx))
    body[0] = g
    sim, _ = _converge(sim, body=body, tol=1e-13, max_steps=40000, every=50)
    ux, _ = sim.velocity()
    H = NY - 2.0
    s = np.arange(1, NY - 1) - 0.5
    u_exact = U * s / H + g / (2.0 * NU) * s * (H - s)
    rel = np.abs(ux[1:-1, 0] - u_exact).max() / np.abs(u_exact).max()
    assert rel < 1e-3, f"rel L-inf={rel:.3e}"
    force = sim.wall_force()
    # steady momentum balance: the two walls take up the body force
    balance = force[0, 0] + force[1, 0] - g * sim.nx * (NY - 2)
    assert abs(balance) / (g * sim.nx * (NY - 2)) < 1e-3


def test_float64_beats_float32_storage():
    sim64, steps = _converge(_couette_sim())
    ux64, _ = sim64.velocity()
    sim32 = _couette_sim()
    for _ in range(steps):
        sim32.step()
        sim32.f = sim32.f.astype(np.float32).astype(np.float64)
    ux32, _ = sim32.velocity()
    H = NY - 2.0
    s = np.arange(1, NY - 1) - 0.5
    u_exact = U * s / H
    e64 = np.abs(ux64[1:-1, 0] - u_exact).max() / U
    e32 = np.abs(ux32[1:-1, 0] - u_exact).max() / U
    assert e64 < 1e-9
    assert e32 / e64 > 1e3, f"float64={e64:.2e} float32={e32:.2e}"


def test_default_channel_mode_has_no_wall_forcing():
    """Non-periodic channel runs keep wall velocities at zero by default."""
    cfg = ChannelConfig(nx=20, ny=12, u_mean=0.05, nu=0.1)
    assert cfg.u_wall_top == 0.0 and cfg.u_wall_bottom == 0.0
    assert cfg.periodic_x is False
    sim = ChannelLBM(cfg)
    for _ in range(100):
        sim.step()
    force = sim.wall_force()
    # no wall velocity is set: the streamwise momentum exchange is only the
    # (small, physical) shear of the developing channel flow, not a wall term
    assert abs(force[0, 0]) / sim.nx < 1e-2 and abs(force[1, 0]) / sim.nx < 1e-2
    # the wall-normal part is the pressure force rho cs^2 per node
    assert abs(abs(force[0, 1]) / sim.nx - 1.0 / 3.0) < 0.02
    ux, _ = sim.velocity()
    assert np.isfinite(ux).all() and np.abs(ux).max() < 0.2