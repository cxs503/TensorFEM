"""Guo (2002) forcing scheme: invariants + the analytic Poiseuille benchmark.

The JOUR-DIFFLBM reference [3] is only meaningful if the scheme it points at
is implemented and *quantitatively* right.  These tests assert

* the source term's moment invariants (mass conserving; momentum
  ``(1 - 1/(2 tau)) F`` for the corrected scheme, ``F`` for the uncorrected
  one) — the discrete-lattice correction in numbers, not prose;
* the corrected scheme reproduces the analytic force-driven Poiseuille
  profile to well below the 3 % acceptance threshold;
* the uncorrected scheme's steady state keeps the analytic *shape* with its
  amplitude scaled by ``1 + 1/(2 tau)`` (the discrete lattice effect),
  so the correction is measurably load-bearing;
* grid refinement is second order for the corrected scheme;
* the *shipping* solver (:class:`tensorfem.lbm_channel2d.ChannelLBM`) reaches
  the same profile through its ``body_force`` branch.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorfem import lbm_forcing as F

CX = np.array(F.CX, dtype=float).reshape(9, 1, 1)
CY = np.array(F.CY, dtype=float).reshape(9, 1, 1)
W9 = np.asarray(F.W9, dtype=float).reshape(9, 1, 1)

NY = 18
NU = 0.05
TAU = 0.5 + 3.0 * NU


def _field(n=3, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(scale=0.01, size=(2, n, n))
    force = rng.normal(scale=1e-4, size=(2, n, n))
    return u, force


def test_guo_source_conserves_mass() -> None:
    u, force = _field()
    src = F.guo_source(u, force, TAU)
    total = src.sum(axis=0)
    assert np.max(np.abs(total)) < 1e-18


def test_guo_source_injects_corrected_momentum() -> None:
    """``sum_i c_i F_i = (1 - 1/(2 tau)) F`` — the Guo correction, exactly."""
    u, force = _field()
    mx, my = F.source_momentum(F.guo_source(u, force, TAU))
    pref = 1.0 - 0.5 / TAU
    assert np.allclose(mx, pref * force[0], atol=1e-16)
    assert np.allclose(my, pref * force[1], atol=1e-16)


def test_naive_source_injects_full_force() -> None:
    u, force = _field()
    mx, my = F.source_momentum(F.naive_source(u, force, TAU))
    assert np.allclose(mx, force[0], atol=1e-18)
    assert np.allclose(my, force[1], atol=1e-18)


def test_guo_source_is_velocity_independent_in_momentum() -> None:
    """The momentum injected by the corrected source term does not depend on
    the local velocity (that is what removes the discrete lattice effect)."""
    force = np.zeros((2, 3, 3))
    force[0] = 1e-3
    u1, _ = _field(seed=1)
    u2, _ = _field(seed=2)
    m1 = F.source_momentum(F.guo_source(u1, force, TAU))[0]
    m2 = F.source_momentum(F.guo_source(u2, force, TAU))[0]
    assert np.allclose(m1, m2, atol=1e-16)


def test_equilibrium_moments_roundtrip() -> None:
    rho = np.full((3, 3), 1.0)
    ux = np.full((3, 3), 0.03)
    uy = np.full((3, 3), -0.02)
    f = F.equilibrium(rho, ux, uy)
    r2, ux2, uy2 = F.guo_macroscopic(f)
    assert np.allclose(r2, 1.0)
    assert np.allclose(ux2, ux)
    assert np.allclose(uy2, uy)


@pytest.fixture(scope="module")
def guo_run():
    cfg = F.PoiseuilleConfig(ny=NY, nu=NU, scheme="guo")
    sim = F.PoiseuilleLBM(cfg)
    stats = sim.run()
    return cfg, sim, stats


@pytest.fixture(scope="module")
def naive_run():
    cfg = F.PoiseuilleConfig(ny=NY, nu=NU, scheme="naive")
    sim = F.PoiseuilleLBM(cfg)
    stats = sim.run()
    return cfg, sim, stats


def test_guo_poiseuille_within_acceptance(guo_run) -> None:
    cfg, sim, stats = guo_run
    assert stats["converged"], stats
    rel = abs(float(sim.profile().max()) - cfg.u_max_analytic) / cfg.u_max_analytic
    assert rel < 0.01, rel                       # measured ~3.5e-3
    assert sim.profile_l2_error() < 0.01


def test_naive_poiseuille_shows_discrete_lattice_effect(naive_run) -> None:
    """The uncorrected scheme's profile is the analytic parabola scaled by
    ``1 + 1/(2 tau)`` — the effect the Guo correction removes."""
    cfg, sim, _ = naive_run
    ratio = float(sim.profile().max()) / cfg.u_max_analytic
    law = 1.0 + 0.5 / cfg.tau
    assert abs(ratio - law) / law < 0.01, (ratio, law)
    # and it must be far outside the 3 % acceptance band
    assert abs(ratio - 1.0) > 0.5


def test_reference_scheme_is_strictly_better(guo_run, naive_run) -> None:
    _, guo_sim, _ = guo_run
    _, naive_sim, _ = naive_run
    assert guo_sim.profile_l2_error() * 10 < naive_sim.profile_l2_error()


def test_grid_refinement_is_second_order() -> None:
    errs = {}
    for ny in (10, NY):
        cfg = F.PoiseuilleConfig(ny=ny, nu=NU, scheme="guo")
        sim = F.PoiseuilleLBM(cfg)
        sim.run()
        errs[ny] = sim.profile_l2_error()
    order = np.log(errs[10] / errs[NY]) / np.log((NY - 2) / (10 - 2))
    assert 1.85 < order < 2.15, (order, errs)


def test_production_channel_solver_matches_analytic() -> None:
    """The shipping solver's body_force branch (Guo scheme) is code-backed."""
    row = F._production_row(NY, NU)
    assert row.converged
    assert row.rel_err_u_max < 0.01, row.rel_err_u_max
    assert row.rel_l2_profile < 0.01, row.rel_l2_profile


def test_benchmark_gate_passes_on_a_small_ladder() -> None:
    result = F.poiseuille_benchmark(
        resolutions=(10, NY), guo_only_resolutions=(), tau_sweep=(),
        include_production=False)
    assert result["gate"]["guo_finest"]["rel_err_u_max"] < 0.03
    assert F.benchmark_passed(result)
    assert result["convergence_order"]["mean_order"] == pytest.approx(2.0, abs=0.2)


def test_unknown_scheme_is_rejected() -> None:
    with pytest.raises(ValueError):
        F.PoiseuilleLBM(F.PoiseuilleConfig(scheme="bogus"))


def test_zero_force_keeps_the_fluid_at_rest() -> None:
    cfg = F.PoiseuilleConfig(ny=12, nu=0.1, scheme="guo")
    cfg.force_x = 0.0
    sim = F.PoiseuilleLBM(cfg)
    for _ in range(50):
        sim.step()
    assert np.max(np.abs(sim.profile())) < 1e-14

def test_poiseuille_relerr_gate_ny18_ny34() -> None:
    """CASE-POISEUILLE-2D-RELERR: rel_err_u_max < 1% at ny>=18 (anchor gate)."""
    for ny in (18, 34):
        cfg = F.PoiseuilleConfig(ny=ny, nu=0.05, scheme="guo")
        sim = F.PoiseuilleLBM(cfg)
        stats = sim.run()
        assert stats["converged"], stats
        rel_u = abs(float(sim.profile().max()) - cfg.u_max_analytic) / cfg.u_max_analytic
        rel_l2 = sim.profile_l2_error()
        assert rel_u < 0.01, f"ny={ny} rel_err_u_max={rel_u*100:.4f}% >= 1%"
        assert rel_l2 < 0.01, f"ny={ny} rel_l2={rel_l2*100:.4f}% >= 1%"
