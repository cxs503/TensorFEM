"""Unit / mathematics tests for the two-component Shan-Chen MCMP solver.

These are the L1 (implementation) checks for the anchor
``XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO``: conservation and equilibrium
invariants, the pseudo-potential, the analytic single-interface interaction
force, a regression against the live lbmpy reference case, and a replay of the
defect that made the first delivery diverge.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tensorfem.lbm_shan_chen_mcmp import (
    CX, CY, GOLDEN_32x32, MCMPConfig, REF_A, ShanChenMCMP, equilibrium_d2q9,
    psi_exponential, relative_l2, run_reference_case, shan_chen_mcmp_evidence,
)


def test_equilibrium_at_rest_is_a_fixed_point():
    """Two uniform components at rest must not move (no spurious force)."""
    sim = ShanChenMCMP(MCMPConfig(nx=16, ny=16, g_ab=6.0, g_ba=6.0))
    sim.init_uniform(0.5, 0.5)
    f0 = sim.f.copy()
    sim.run(50)
    assert np.abs(sim.f - f0).max() < 1e-12
    assert sim.spurious_current() < 1e-12


def test_mass_conservation_at_g6():
    """Collision + periodic streaming conserve both component masses."""
    sim = ShanChenMCMP(MCMPConfig(nx=32, ny=32, g_ab=6.0, g_ba=6.0))
    sim.init_step_x(0.1, 0.9)
    m0 = sim.total_mass()
    sim.run(300)
    m1 = sim.total_mass()
    assert np.isfinite(sim.f).all()
    assert np.abs(m1 - m0).max() / m0.max() < 1e-12
    assert m0[0] == pytest.approx(0.5 * 32 * 32)


def test_psi_is_monotone_and_vanishes_at_zero():
    rho = np.linspace(1e-6, 5.0, 200)
    psi = psi_exponential(rho)
    assert psi[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(psi) > 0.0)
    # psi -> rho0 as rho -> inf
    assert psi_exponential(np.array([200.0]))[0] == pytest.approx(1.0, abs=1e-80)


def test_interaction_force_matches_single_interface_analytic_value():
    """Hand-derived force on the two cells flanking a straight interface.

    With ``psi(rho) = 1 - exp(-rho)`` and a step ``rho_a = 0.1 | 0.9`` the
    weighted neighbour sum is ``(1/9 + 2/36) * dpsi_b`` along ``x``, so

        |F_a| = psi_a * g_ab * (1/6) * |psi_b(right) - psi_b(left)|

    on either side of the interface.  This value is independent of the solver
    loop, so it pins the sign and the normalisation of the force.
    """
    psi_lo = 1.0 - np.exp(-0.1)
    psi_hi = 1.0 - np.exp(-0.9)
    expected_dilute = psi_lo * 6.0 * (1.0 / 6.0) * (psi_hi - psi_lo)
    expected_dense = psi_hi * 6.0 * (1.0 / 6.0) * (psi_hi - psi_lo)

    sim = ShanChenMCMP(MCMPConfig(nx=64, ny=8, g_ab=6.0, g_ba=6.0))
    sim.init_step_x(0.1, 0.9)
    F = sim.interaction_force()
    # x = 31 is the last dilute column, x = 32 the first dense column
    assert F[0, 0, 0, 31] == pytest.approx(expected_dilute, rel=1e-10)
    assert F[0, 0, 0, 32] == pytest.approx(expected_dense, rel=1e-10)
    # b sees the mirrored gradient of psi_a, so its force has the opposite sign
    # (the two components are pushed apart across the interface)
    assert F[1, 0, 0, 31] == pytest.approx(-expected_dense, rel=1e-10)
    assert F[1, 0, 0, 32] == pytest.approx(-expected_dilute, rel=1e-10)


def test_force_is_zero_in_a_uniform_region():
    sim = ShanChenMCMP(MCMPConfig(nx=16, ny=16, g_ab=6.0, g_ba=6.0))
    sim.init_uniform(0.3, 0.7)
    assert np.abs(sim.interaction_force()).max() < 1e-14


def test_regression_against_live_lbmpy_reference_case():
    """32x32, g_ab = 6, 400 steps must reproduce the lbmpy-generated profile."""
    if not GOLDEN_32x32.exists():  # pragma: no cover
        pytest.skip(f"missing reference {GOLDEN_32x32}")
    ref = json.loads(GOLDEN_32x32.read_text())
    n = int(ref["params"]["n"])
    sim = ShanChenMCMP(MCMPConfig(nx=n, ny=n, g_ab=float(ref["params"]["g_ab"]),
                                  g_ba=float(ref["params"]["g_ab"])))
    sim.init_step_x(0.1, 0.9)
    sim.run(int(ref["params"]["steps"]))
    err = relative_l2(sim.density()[0, 0, :], np.asarray(ref["rho_a_profile"]))
    assert err < 1e-8, f"rel L2 {err:.3e} against live lbmpy reference"


def test_anchor_case_is_inside_the_15_percent_band():
    """The published 64^2 / 1000-step anchor case (7 s)."""
    res = run_reference_case(steps=1000)
    assert res["diverged_at"] is None
    assert res["rel_l2_a"] < 0.15
    assert res["rel_l2_b"] < 0.15


def test_first_delivery_defect_is_reproduced_by_the_switch():
    """The ``psi(x-c)`` neighbour convention must diverge (kept for the post-mortem)."""
    sim = ShanChenMCMP(MCMPConfig(nx=32, ny=32, g_ab=6.0, g_ba=6.0,
                                  neighbour_convention="minus"))
    sim.init_step_x(0.1, 0.9)
    diverged = False
    for _ in range(200):
        sim.step()
        if not np.isfinite(sim.f).all():
            diverged = True
            break
    assert diverged


def test_registry_evidence_passes():
    rows = shan_chen_mcmp_evidence()
    assert len(rows) == 3
    for row in rows:
        assert row.passed, f"{row.id} error {row.error:.3e} > {row.tolerance}"