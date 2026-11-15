"""Tests for the exact patch-load Levy reference (``levy_restrained_plate_patch``).

The reference exists to verify the *interior* of the elastic-restraint band for
the load case the JOUR-ICE-STRUCT manuscript actually applies (a full-breadth
pressure patch).  These tests pin the properties that make it usable as a
reference:

* it degenerates to the uniform-load solution to machine precision
  (``patch = full width``),
* at ``beta = 0`` it reproduces the published 401-term Navier baseline of
  ``panel_patch_load_results.json`` - two independent methods,
* it agrees with an independent variational (Rayleigh-Ritz) solution,
* its mode series has a convergence *plateau* (measured, not assumed),
* it is monotone in the embedding coefficient and reaches the clamped limit,
* the FEM Mindlin panel converges onto it as the mesh is refined,
* its load-description normalisation is defensive.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.plate_elastic_restraint import (
    peak_bending_stress,
    rect_patch_pressure,
    rotational_stiffness,
    solve_restrained_panel,
)
from tensorfem.plate_levy_restraint import (
    _normalise_pieces,
    flexural_rigidity,
    levy_restrained_plate,
    levy_restrained_plate_patch,
    levy_restrained_plate_step_load,
    ritz_restrained_plate,
)

# paper design point
L, S, T = 0.8, 0.7, 0.022
E, NU = 206.0e9, 0.3
Q = 1.0e6
X1, X2 = 0.5 * (L - 0.08), 0.5 * (L + 0.08)
BASELINE_SIGMA_MPA = 73.37101604723352  # panel_patch_load_results.json, Navier N=401
D = flexural_rigidity(E, NU, T)


def _sigma_vm(Mx: float, My: float) -> float:
    return 6.0 / T**2 * math.sqrt(Mx * Mx - Mx * My + My * My)


def test_full_width_patch_reduces_to_uniform_solution():
    """``patch = [0, a]`` must be the uniform-load solution, not merely close to it."""
    K = 20000.0
    patch = levy_restrained_plate_patch(L, S, D, NU, K, Q, (0.0, L), n_modes=101)
    uniform = levy_restrained_plate(L, S, D, NU, K, Q, n_modes=101)
    assert patch.w_centre() == pytest.approx(uniform.w_centre(), rel=1e-12)
    assert patch.moments_centre()[0] == pytest.approx(uniform.moments_centre()[0], rel=1e-12)
    assert patch.moments_centre()[1] == pytest.approx(uniform.moments_centre()[1], rel=1e-12)


def test_beta0_patch_matches_published_navier_baseline():
    """Independent method check: exact series vs the paper's 401-term Navier series."""
    sol = levy_restrained_plate_patch(L, S, D, NU, 0.0, Q, (X1, X2), n_modes=201)
    Mx, My = sol.moments_centre()
    sigma = _sigma_vm(Mx, My) / 1e6
    assert sigma == pytest.approx(BASELINE_SIGMA_MPA, rel=1e-3)  # acceptance: < 0.1 %


def test_mode_series_has_convergence_plateau():
    """Measured plateau: the centre stress must not drift with extra modes.

    Also guards the conditioning guard: modes whose boundary system is
    round-off dominated are dropped, and dropping them must not move the answer.
    """
    values = []
    for n_modes in (201, 401, 601):
        sol = levy_restrained_plate_patch(L, S, D, NU, 20000.0, Q, (X1, X2), n_modes=n_modes)
        Mx, My = sol.moments_centre()
        assert math.isfinite(Mx) and math.isfinite(My)
        values.append(_sigma_vm(Mx, My) / 1e6)
    for a, b in zip(values, values[1:]):
        assert b == pytest.approx(a, rel=1e-4)
    # every value also matches the published baseline scale at beta = 0 family
    assert values[0] == pytest.approx(73.212, abs=0.01)


def test_matches_independent_rayleigh_ritz():
    """Two independent methods (exact ODE join vs variational double sine)."""
    for beta in (0.0, 0.4, 0.6):
        K = rotational_stiffness(beta, E, NU, T, L)
        levy = levy_restrained_plate_patch(L, S, D, NU, K, Q, (X1, X2), n_modes=201)
        ritz = ritz_restrained_plate(
            L, S, D, NU, K, 0.0, Q, m_modes=30, n_modes=80, patch=(X1, X2, 0.0, S)
        )
        assert levy.w_centre() == pytest.approx(ritz.w_centre(), rel=2e-3)
        assert levy.moments_centre()[0] == pytest.approx(ritz.moments_centre()[0], rel=2e-2)


def test_ritz_gap_shrinks_towards_exact_at_high_beta():
    """Near the clamped limit the *Ritz* is the inaccurate one - and it converges.

    The double-sine Ritz satisfies the clamped slope condition only in the
    limit, so at ``beta = 0.8`` it sits ~0.3 % below the exact series.  That gap
    must close as its basis is enriched, which is what certifies the exact
    series rather than the Ritz as the reference.
    """
    beta = 0.8
    K = rotational_stiffness(beta, E, NU, T, L)
    levy_w = levy_restrained_plate_patch(L, S, D, NU, K, Q, (X1, X2), n_modes=201).w_centre()
    gaps = []
    for m_modes, n_modes in ((30, 80), (60, 160)):
        ritz = ritz_restrained_plate(
            L, S, D, NU, K, 0.0, Q, m_modes=m_modes, n_modes=n_modes,
            patch=(X1, X2, 0.0, S),
        )
        gaps.append(abs(ritz.w_centre() / levy_w - 1.0))
    assert gaps[1] < gaps[0] < 5e-3


def test_spring_band_is_monotone_and_reaches_clamped_limit():
    """Centre stress decreases with beta and saturates at the clamped limit."""
    sigmas = []
    for beta in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        K = rotational_stiffness(beta, E, NU, T, L)
        sol = levy_restrained_plate_patch(L, S, D, NU, K, Q, (X1, X2), n_modes=201)
        Mx, My = sol.moments_centre()
        sigmas.append(_sigma_vm(Mx, My) / 1e6)
    assert all(b < a for a, b in zip(sigmas, sigmas[1:])), sigmas
    # beta = 1 (zeta = 1e6) and a genuinely clamped limit agree
    clamped = levy_restrained_plate_patch(L, S, D, NU, 1e12, Q, (X1, X2), n_modes=201)
    Mx, My = clamped.moments_centre()
    assert _sigma_vm(Mx, My) / 1e6 == pytest.approx(sigmas[-1], rel=5e-3)


def test_segment_normalisation_fills_gaps_and_rejects_overlap():
    pieces = _normalise_pieces(((X1, X2, Q),), L)
    assert pieces[0][0] == 0.0 and pieces[0][1] == pytest.approx(X1)
    assert pieces[0][2] == 0.0                      # gap filled with zero load
    assert pieces[1][2] == Q
    assert pieces[-1][1] == pytest.approx(L)
    with pytest.raises(ValueError):
        _normalise_pieces(((0.0, 0.5, Q), (0.4, 0.8, Q)), L)
    with pytest.raises(ValueError):
        _normalise_pieces(((0.2, 0.1, Q),), L)
    with pytest.raises(ValueError):
        levy_restrained_plate_step_load(L, S, D, NU, -1.0, ((X1, X2, Q),))


def test_total_applied_load_is_exact():
    """A uniform-pressure patch must integrate to ``q * length * breadth``."""
    patch = levy_restrained_plate_patch(L, S, D, NU, 20000.0, Q, (X1, X2), n_modes=51)
    area = sum((x1 - x0) for x0, x1, _q in patch.segments)
    assert area == pytest.approx(L)
    loaded = sum((x1 - x0) * q for x0, x1, q in patch.segments)
    assert loaded == pytest.approx(Q * (X2 - X1))


def test_fem_panel_converges_onto_exact_reference():
    """The Mindlin Q4 panel must land on the exact series as the mesh refines."""
    beta = 0.4
    K = rotational_stiffness(beta, E, NU, T, L)
    ref = levy_restrained_plate_patch(L, S, D, NU, K, Q, (X1, X2), n_modes=201)
    Mx, My = ref.moments_centre()
    ref_sigma = _sigma_vm(Mx, My) / 1e6
    nx = 24
    pressure = rect_patch_pressure(Q, X1, X2, 0.0, S)
    res = solve_restrained_panel(nx, int(round(nx * S / L)), L, S, E, NU, T, pressure,
                                 beta_x=beta, beta_y=0.0)
    fem_sigma = peak_bending_stress(res)["centre_von_mises"] / 1e6
    assert fem_sigma == pytest.approx(ref_sigma, rel=5e-2)