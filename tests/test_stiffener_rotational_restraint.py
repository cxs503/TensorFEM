"""Verification of the physical stiffener/frame -> ``beta`` calibration.

Anchor: ``GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT``.

The tests do three separate things:

* check the *mechanics* of the calibration against something independent — the
  torsion closure ``k_theta = 12 GJ / L^2`` is verified against a finite
  difference solution of the torsion ODE, not against its own derivation;
* check the *algebra* tying the calibration to the existing solver — the
  ``k_theta -> beta -> K_rot`` round trip must be exact for every ``zeta``,
  including the clamped-saturated end where a naive ``beta = zeta/(1+zeta)``
  loses everything;
* check the *end-to-end* claim — a beta calibrated from real scantlings, fed to
  :func:`solve_restrained_panel`, must reproduce the exact Levý reference
  evaluated at that same physical stiffness.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

# Small, fixed thread pool: these tests run on a shared box whose load average
# is routinely ~100, and letting torch fan out over 128 cores makes the panel
# solves thrash (measured: 1.6 s -> >60 s).  No effect on results.
torch.set_num_threads(min(4, torch.get_num_threads()))

from tensorfem.plate_elastic_restraint import (
    beta_to_zeta,
    moments_at_nodes,
    rotational_stiffness,
    solve_restrained_panel,
)
from tensorfem.plate_levy_restraint import (
    flexural_rigidity,
    kirchhoff_strip_end_moment,
    levy_restrained_plate,
)
from tensorfem.stiffener_rotational_restraint import (
    calibrate_panel_restraint,
    beta_from_rotational_stiffness,
    flat_bar_torsion_constant,
    moment_fixity_strip,
    sensitivity_sweep,
    shear_modulus,
    thin_walled_torsion_constant,
    torsional_distributed_stiffness,
)

E_STEEL = 206.0e9
NU = 0.3
T_PLATE = 0.022


# ---------------------------------------------------------------------------
# Torsion constant
# ---------------------------------------------------------------------------

def test_flat_bar_torsion_constant_matches_square_limit():
    """A square section must approach J = 0.1406 b t^3 (Timoshenko & Goodier)."""
    b = 0.05
    j = flat_bar_torsion_constant(b, b)
    assert j / (0.1406 * b**4) == pytest.approx(1.0, rel=5e-3)


def test_flat_bar_torsion_constant_tends_to_thin_wall():
    """For b/t = 100 the series must reduce to b t^3 / 3 (thin-wall limit)."""
    b, t = 0.100, 0.001
    j = flat_bar_torsion_constant(b, t)
    thin = b * t**3 / 3.0
    assert j / thin == pytest.approx(1.0, rel=1e-2)
    assert j < thin  # the series correction always reduces J


def test_thin_walled_sum_is_additive_and_order_independent():
    rects = [(0.18, 0.010), (0.090, 0.012), (0.090, 0.012)]
    j = thin_walled_torsion_constant(rects)
    assert j == pytest.approx(sum(b * t**3 / 3.0 for b, t in rects), rel=1e-15)
    assert j == pytest.approx(thin_walled_torsion_constant(list(reversed(rects))), rel=1e-15)


def test_shear_modulus_steel():
    assert shear_modulus(E_STEEL, NU) == pytest.approx(206.0e9 / (2.0 * 1.3), rel=1e-15)


# ---------------------------------------------------------------------------
# The torsion closure, verified independently (this is the core new physics)
# ---------------------------------------------------------------------------

def _mean_twist_fd(gj: float, span: float, moments: float, n: int = 4001) -> float:
    """Mean twist of ``GJ phi'' = -m`` with ``phi(0) = phi(span) = 0`` by FD."""
    x = np.linspace(0.0, span, n)
    h = x[1] - x[0]
    rhs = np.full(n - 2, -moments / gj * h**2)
    # Tridiagonal system: phi[i-1] - 2 phi[i] + phi[i+1] = rhs[i]
    lower = np.ones(n - 3)
    diag = -2.0 * np.ones(n - 2)
    upper = np.ones(n - 3)
    from scipy.linalg import solve_banded

    phi = np.zeros(n)
    phi[1:-1] = solve_banded((1, 1), np.vstack([np.concatenate(([0.0], upper)),
                                                diag,
                                                np.concatenate((lower, [0.0]))]), rhs)
    return float(np.trapezoid(phi, x) / span)


@pytest.mark.parametrize("span", [0.35, 0.7, 0.8, 2.0])
def test_torsional_stiffness_matches_independent_fd_solution(span):
    """``k_theta = 12 GJ / L^2`` vs an independent finite-difference solve.

    The FD model knows nothing about the closed form: it discretises the ODE,
    solves the tridiagonal system and averages the twist.  Agreement proves the
    closure is the actual response of a torsionally end-fixed shaft.
    """
    gj = 6.5e3
    m = 1.0e3
    k_analytic = torsional_distributed_stiffness(gj, span)
    k_fd = m / _mean_twist_fd(gj, span, m)
    assert k_fd / k_analytic == pytest.approx(1.0, rel=2e-5)


def test_torsional_stiffness_energy_equivalence():
    """Strain energy definition ``k = 2U / mean(phi)^2`` must agree as well."""
    gj, span, m = 4.0e3, 0.6, 500.0
    # Analytic: phi = m x (L - x) / (2 GJ);  U = 1/2 int GJ phi'^2 dx
    coef = m / (2.0 * gj)
    # phi' = coef (L - 2x) -> int_0^L phi'^2 dx = coef^2 L^3 / 3
    U = 0.5 * gj * coef**2 * span**3 / 3.0
    mean_phi = m * span**2 / (12.0 * gj)
    # Distributed, not lumped: the spring is spread over the length, so
    # U = 1/2 k mean(phi)^2 * L.  Dropping the L gives the *nodal* spring
    # 12 GJ / L^3 which is a different (3x here) quantity.
    k_energy = 2.0 * U / (mean_phi**2 * span)
    assert k_energy == pytest.approx(torsional_distributed_stiffness(gj, span), rel=1e-12)


def test_torsional_stiffness_scales_inverse_square_with_support_span():
    """Halving the support span must quadruple the stiffness (quadratic entry)."""
    gj = 8.0e3
    k1 = torsional_distributed_stiffness(gj, 0.8)
    k2 = torsional_distributed_stiffness(gj, 0.4)
    assert k2 / k1 == pytest.approx(4.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Exact coupling to the solver parameterisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "k_theta",
    [0.0, 1.0, 1.0e3, 1.0e5, 2.0e5, 1.0e6, 1.0e8, 2.0e10, 2.8e10],
)
def test_beta_round_trip_reproduces_the_physical_stiffness(k_theta):
    """``k_theta -> beta -> K_rot`` must return the input for every zeta.

    This is the tie between the new calibration and the already-verified
    solver: whatever ``beta`` the solver needs, the stiffness it assembles is
    exactly the one the structure provides.  Covers ``zeta`` from 0 up to just
    below the parameterisation limit ``1/eps``; beyond that the map *cannot* be
    exact and :func:`beta_from_rotational_stiffness` says so (next test).
    """
    D = flexural_rigidity(E_STEEL, NU, T_PLATE)
    span = 0.7
    beta = beta_from_rotational_stiffness(k_theta, D, span)
    assert 0.0 <= beta <= 1.0
    # The map is exact in exact arithmetic.  In float64 the recoverable
    # precision degrades as the clamped end is approached, because the solver
    # reconstructs zeta from ``1 - beta`` (~1e-5 here): tolerance 1e-9, measured
    # worst case 6e-12 at zeta = 7e4.
    assert beta_to_zeta(beta) == pytest.approx(k_theta * span / D, rel=1e-9)
    k_back = rotational_stiffness(beta, E_STEEL, NU, T_PLATE, span)
    assert k_back == pytest.approx(k_theta, rel=1e-9, abs=1e-9)


def test_beta_limits_and_monotonicity():
    D = flexural_rigidity(E_STEEL, NU, T_PLATE)
    span = 0.7
    assert beta_from_rotational_stiffness(0.0, D, span) == 0.0
    betas = [
        beta_from_rotational_stiffness(k, D, span)
        for k in (1e3, 1e4, 1e5, 1e6, 1e7)
    ]
    assert all(b > a for a, b in zip(betas, betas[1:]))
    assert betas[-1] > betas[0]
    assert beta_from_rotational_stiffness(1e15, D, span) == 1.0


def test_beta_saturates_at_the_parameterisation_limit():
    """Above ``zeta = 1/eps`` the stiffness is not representable: say so.

    The solver's ``zeta = beta/(1 - beta + eps)`` tops out at ``1/eps``, so a
    stiffer physical restraint necessarily maps to the clamped end.  The map
    pins ``beta = 1`` rather than returning a number that would round-trip to a
    much softer stiffness, and the calibration warns.
    """
    D = flexural_rigidity(E_STEEL, NU, T_PLATE)
    span = 0.7
    k_sat = 1.0e6 * D / span           # exactly zeta = 1/eps
    assert beta_from_rotational_stiffness(k_sat, D, span) == pytest.approx(1.0, abs=1e-6)
    assert beta_from_rotational_stiffness(2.0 * k_sat, D, span) == 1.0
    # Below the limit the map is still a faithful inverse.
    k_below = 0.99 * k_sat
    assert beta_to_zeta(beta_from_rotational_stiffness(k_below, D, span)) == pytest.approx(
        k_below * span / D, rel=1e-9
    )


# ---------------------------------------------------------------------------
# Moment fixity: the physically interpretable companion of beta
# ---------------------------------------------------------------------------

def test_moment_fixity_limits_and_consistency_with_the_repo_strip_formula():
    D = flexural_rigidity(E_STEEL, NU, T_PLATE)
    span, q = 0.7, 1.0e6
    assert moment_fixity_strip(0.0, D, span) == 0.0
    k = 5.0e5
    expect = kirchhoff_strip_end_moment(q, span, D, k) / (q * span**2 / 12.0)
    assert moment_fixity_strip(k, D, span, q) == pytest.approx(expect, rel=1e-15)
    fix = [moment_fixity_strip(kk, D, span) for kk in (1e4, 1e5, 1e6, 1e7, 1e9)]
    assert all(b > a for a, b in zip(fix, fix[1:]))
    assert 0.0 < fix[0] < fix[-1] < 1.0
    # A stiffness of D/a already mobilises a third of the clamped moment.
    assert moment_fixity_strip(D / span, D, span) == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_beta_is_not_the_moment_fixity_ratio():
    """Guard against the tempting but wrong reading of ``beta``.

    ``beta`` is a solver reparameterisation; the mobilised moment fraction is
    genuinely different, which is why both are reported.
    """
    D = flexural_rigidity(E_STEEL, NU, T_PLATE)
    span = 0.7
    k = 1.2e5
    beta = beta_from_rotational_stiffness(k, D, span)
    fix = moment_fixity_strip(k, D, span)
    assert abs(beta - fix) > 0.05


# ---------------------------------------------------------------------------
# Panel-level calibration
# ---------------------------------------------------------------------------

def _ice_belt_calibration(**kwargs):
    """Realistic ice-belt scantlings: FB 150x12 longitudinals, FB 200x12 frames."""
    defaults = dict(
        young=E_STEEL,
        poisson=NU,
        thickness=T_PLATE,
        side_x=0.7,   # longitudinal spacing
        side_y=0.8,   # frame spacing
        longitudinal_rectangles=[(0.15, 0.012)],
        frame_rectangles=[(0.20, 0.012)],
    )
    defaults.update(kwargs)
    return calibrate_panel_restraint(**defaults)


def test_calibration_lands_strictly_inside_the_band():
    """Real scantlings must give a finite restraint, not either idealisation.

    This is the manuscript-relevant statement: ``0 < beta < 1`` with real
    margin, i.e. the SSSS result is neither the answer nor trivially close to
    it, and the clamped result is not reachable either.
    """
    cal = _ice_belt_calibration()
    for edge in (cal.longitudinal, cal.frame):
        assert 0.05 < edge.beta < 0.95, (edge.member, edge.beta)
        assert 0.1 < edge.moment_fixity < 0.9, (edge.member, edge.moment_fixity)
        assert edge.k_theta > 0.0
    assert cal.warnings == ()


def test_calibration_support_span_uses_the_perpendicular_member():
    """Default support spans must be the *perpendicular* member spacing.

    The longitudinal is held by the frames (``side_y``) and the frame by the
    longitudinals (``side_x``).  Getting this backwards changes the stiffness
    quadratically, so it is asserted explicitly.
    """
    cal = _ice_belt_calibration()
    assert cal.longitudinal.support_span == pytest.approx(0.8)
    assert cal.frame.support_span == pytest.approx(0.7)
    # Overriding to a common (wrong) choice must change the result materially.
    swapped = _ice_belt_calibration(longitudinal_support_span=0.7, frame_support_span=0.8)
    assert swapped.longitudinal.beta > cal.longitudinal.beta
    assert swapped.frame.beta < cal.frame.beta


def test_calibration_is_monotone_in_stiffener_size_and_support_span():
    small = _ice_belt_calibration(longitudinal_rectangles=[(0.10, 0.010)])
    large = _ice_belt_calibration(longitudinal_rectangles=[(0.20, 0.014)])
    assert large.longitudinal.beta > small.longitudinal.beta
    dense = _ice_belt_calibration(longitudinal_support_span=0.4)
    assert dense.longitudinal.beta > _ice_belt_calibration().longitudinal.beta
    # Torsionally fixing the frames more closely does *not* change the
    # longitudinal restraint -- the two families are independent.
    frame_dense = _ice_belt_calibration(frame_support_span=0.4)
    assert frame_dense.longitudinal.beta == pytest.approx(
        _ice_belt_calibration().longitudinal.beta, rel=1e-15
    )


def test_calibration_warns_when_effectively_clamped_or_simply_supported():
    saturated = _ice_belt_calibration(longitudinal_rectangles=None, longitudinal_torsion_rigidity=1.0e12)
    assert any("saturates" in w for w in saturated.warnings), saturated.warnings
    assert saturated.longitudinal.beta == 1.0

    heavy = _ice_belt_calibration(longitudinal_rectangles=None, longitudinal_torsion_rigidity=1.0e8)
    assert any("effectively clamped" in w for w in heavy.warnings), heavy.warnings
    assert heavy.longitudinal.beta > 0.95

    light = _ice_belt_calibration(longitudinal_rectangles=[(0.02, 0.002)])
    assert any("simply supported" in w for w in light.warnings), light.warnings


def test_calibration_accepts_explicit_torsion_rigidity_and_rejects_ambiguity():
    gj = 6.5e3
    cal = calibrate_panel_restraint(
        young=E_STEEL, poisson=NU, thickness=T_PLATE, side_x=0.7, side_y=0.8,
        longitudinal_torsion_rigidity=gj, frame_rectangles=[(0.20, 0.012)],
    )
    assert cal.longitudinal.torsion_rigidity == pytest.approx(gj, rel=1e-15)
    with pytest.raises(ValueError):
        calibrate_panel_restraint(
            young=E_STEEL, poisson=NU, thickness=T_PLATE, side_x=0.7, side_y=0.8,
            longitudinal_torsion_rigidity=gj, longitudinal_rectangles=[(0.15, 0.012)],
            frame_rectangles=[(0.20, 0.012)],
        )
    with pytest.raises(ValueError):
        calibrate_panel_restraint(
            young=E_STEEL, poisson=NU, thickness=T_PLATE, side_x=0.7, side_y=0.8,
            frame_torsion_rigidity=1.0,
        )


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        torsional_distributed_stiffness(1.0e3, 0.0)
    with pytest.raises(ValueError):
        torsional_distributed_stiffness(-1.0, 0.5)
    with pytest.raises(ValueError):
        beta_from_rotational_stiffness(-1.0, 1.0e5, 0.7)
    with pytest.raises(ValueError):
        beta_from_rotational_stiffness(math.inf, 1.0e5, 0.7)
    with pytest.raises(ValueError):
        flat_bar_torsion_constant(0.0, 0.01)
    with pytest.raises(ValueError):
        thin_walled_torsion_constant([])
    with pytest.raises(ValueError):
        shear_modulus(206e9, 0.5)


# ---------------------------------------------------------------------------
# End-to-end: calibrated beta must reproduce the exact reference
# ---------------------------------------------------------------------------

def test_calibrated_beta_reproduces_the_exact_levy_reference():
    """Closed loop: scantlings -> beta -> Mindlin FE vs exact Kirchhoff Levý.

    The physical stiffness enters the FE solver only as ``beta``; if the map is
    right, the FE centre deflection and centre moment must match the exact
    solution evaluated at that same ``k_theta`` — not at the idealised ends.
    """
    cal = _ice_belt_calibration()
    D = cal.D
    q = 1.0e6
    kx = cal.longitudinal.k_theta

    panel = solve_restrained_panel(
        nx=48, ny=56,
        side_x=cal.side_x, side_y=cal.side_y,
        young=cal.young, poisson=cal.poisson, thickness=cal.thickness,
        pressure=q, beta_x=cal.longitudinal.beta, beta_y=0.0,
    )
    cx, cy = cal.side_x / 2.0, cal.side_y / 2.0
    i = int(np.argmin((panel.nodes[:, 0] - cx) ** 2 + (panel.nodes[:, 1] - cy) ** 2))
    w_panel = float(panel.displacement[3 * i])
    mx_panel = float(moments_at_nodes(panel)[i, 0])

    exact = levy_restrained_plate(
        a=cal.side_x, b=cal.side_y, D=D, nu=cal.poisson, K=kx, q=q, n_modes=201
    )
    w_exact = exact.w_centre()
    mx_exact = exact.moments_centre()[0]

    assert w_panel / w_exact == pytest.approx(1.0, rel=2e-2), (w_panel, w_exact)
    assert mx_panel / mx_exact == pytest.approx(1.0, rel=2e-2), (mx_panel, mx_exact)
    # And the calibrated state must be clearly away from both idealisations.
    assert D > 0.0 and 0.0 < cal.longitudinal.moment_fixity < 1.0


def test_sensitivity_sweep_is_machine_readable_and_monotone():
    rows = sensitivity_sweep(
        young=E_STEEL, poisson=NU, thickness=T_PLATE, side_x=0.7, side_y=0.8,
        longitudinal_variants=[
            ("FB 120x10", [(0.12, 0.010)]),
            ("FB 150x12", [(0.15, 0.012)]),
            ("HP 180x10", [(0.18, 0.010), (0.06, 0.014)]),
        ],
        frame_variants=[("FB 200x12", [(0.20, 0.012)])],
        support_spans=[0.4, 0.7, 0.8, 1.2],
    )
    assert len(rows) == 3 * 4
    keys = {"support_span", "longitudinal", "frame", "k_theta_x", "zeta_x", "beta_x",
            "moment_fixity_x", "k_theta_y", "zeta_y", "beta_y", "moment_fixity_y"}
    assert keys <= set(rows[0])
    for row in rows:
        assert 0.0 <= row["beta_x"] <= 1.0
        assert row["moment_fixity_x"] == pytest.approx(
            moment_fixity_strip(row["k_theta_x"], _ice_belt_calibration().D, 0.7), rel=1e-12
        )
    # Coarser support grid (larger span) -> softer restraint, for fixed member.
    same_member = [r for r in rows if r["longitudinal"] == "FB 150x12"]
    by_span = sorted(same_member, key=lambda r: r["support_span"])
    assert all(b["beta_x"] < a["beta_x"] for a, b in zip(by_span, by_span[1:]))