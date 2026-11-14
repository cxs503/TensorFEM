"""Verification tests for the exact/variational rotational-restraint references.

Anchor: ``GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT``.  These tests do two things:

* prove the *references themselves* are right (exact Levy series against
  Timoshenko's plate coefficients, against a beam closed form, against a
  Rayleigh-Ritz solution, and against its own spring boundary condition);
* prove that the Mindlin FE panel
  (:func:`tensorfem.plate_elastic_restraint.solve_restrained_panel`) reproduces
  the exact reference **in the interior of the beta band**, not only at the two
  end points that were verified previously.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.plate_elastic_restraint import beta_to_zeta, moments_at_nodes, solve_restrained_panel
from tensorfem.plate_levy_restraint import (
    flexural_rigidity,
    kirchhoff_strip_deflection,
    kirchhoff_strip_end_moment,
    levy_restrained_plate,
    ritz_restrained_plate,
)

# Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed.,
# Table 8 (nu = 0.3, uniform load).
TIMO_SSSS_W = 0.00406   # w D / (q a^4)
TIMO_SSSS_M = 0.0479    # M / (q a^2)
TIMO_CCCC_W = 0.00126   # w D / (q a^4)
TIMO_CCCC_M = 0.0231    # M / (q a^2)


# ---------------------------------------------------------------------------
# The references
# ---------------------------------------------------------------------------

def test_levy_simply_supported_matches_timoshenko():
    """``K = 0`` must reduce the Levy solution to the SSSS plate exactly."""
    sol = levy_restrained_plate(1.0, 1.0, 1.0, 0.3, 0.0, 1.0, n_modes=401)
    assert sol.w_centre() / TIMO_SSSS_W - 1.0 == pytest.approx(0.0, abs=0.005)
    assert sol.moments_centre()[0] / TIMO_SSSS_M - 1.0 == pytest.approx(0.0, abs=0.005)
    # a simple support carries no edge moment, and the slope is non-zero
    assert sol.moments(0.0, 0.5)[0] == pytest.approx(0.0, abs=1e-12)
    assert abs(sol.slope(0.0, 0.5)) > 0.0


def test_levy_spring_boundary_condition_identity():
    """The recovered moment must satisfy ``M_n = -+ K dw/dn`` on the sprung edge."""
    D, nu, q = 1.0, 0.3, 1.0
    for K in (1.0, 10.0, 1e3, 1e6):
        sol = levy_restrained_plate(1.0, 1.0, D, nu, K, q, n_modes=401)
        m_x0 = sol.moments(0.0, 0.5)[0]
        m_xa = sol.moments(1.0, 0.5)[0]
        assert m_x0 == pytest.approx(-K * sol.slope(0.0, 0.5), rel=1e-10, abs=1e-14)
        assert m_xa == pytest.approx(+K * sol.slope(1.0, 0.5), rel=1e-10, abs=1e-14)


def test_levy_clamped_limit_matches_ritz_and_bounds():
    """``K -> inf`` closes the band; the result must sit between SSSS and CCCC."""
    D, nu, q = 1.0, 0.3, 1.0
    levy_open = levy_restrained_plate(1.0, 1.0, D, nu, 0.0, q, n_modes=401)
    levy_shut = levy_restrained_plate(1.0, 1.0, D, nu, 1e8, q, n_modes=401)
    # two opposite edges clamped, two simply supported: stiffer than SSSS,
    # softer than all-edge clamped.
    assert TIMO_CCCC_W < levy_shut.w_centre() < levy_open.w_centre()
    # The Ritz solution is variational, so a truncated series is always on the
    # stiff side; near the clamped limit the sine basis converges slowly, which
    # is why this comparison - and only this one - is allowed 5 %.
    ritz = ritz_restrained_plate(1.0, 1.0, D, nu, 1e8, 0.0, q, 40, 40)
    assert ritz.w_centre() == pytest.approx(levy_shut.w_centre(), rel=0.05)
    assert ritz.w_centre() < levy_shut.w_centre()


def test_ritz_reproduces_timoshenko_cccc():
    """Rayleigh-Ritz with all four springs shut must give the clamped plate."""
    sol = ritz_restrained_plate(1.0, 1.0, 1.0, 0.3, 1e6, 1e6, 1.0, 40, 40)
    assert sol.w_centre() / TIMO_CCCC_W - 1.0 == pytest.approx(0.0, abs=0.05)
    assert sol.moments_centre()[0] / TIMO_CCCC_M - 1.0 == pytest.approx(0.0, abs=0.05)


def test_ritz_matches_levy_when_y_spring_absent():
    """With ``Ky = 0`` the two references solve the same BVP and must agree."""
    for K in (0.0, 10.0, 1e2):
        levy = levy_restrained_plate(1.0, 1.0, 1.0, 0.3, K, 1.0, n_modes=401)
        ritz = ritz_restrained_plate(1.0, 1.0, 1.0, 0.3, K, 0.0, 1.0, 40, 40)
        # Ritz is variational (stiffer) and the sine basis converges slowly as
        # the restraint closes: 40 x 40 modes give -0.0 % / -0.9 % / -2.7 % on
        # the deflection for K = 0 / 10 / 100, always from below.
        assert ritz.w_centre() == pytest.approx(levy.w_centre(), rel=0.03)
        assert ritz.w_centre() <= levy.w_centre()
        assert ritz.moments_centre()[0] == pytest.approx(levy.moments_centre()[0], rel=0.03)


def test_levy_strip_limit_matches_beam_closed_form():
    """``b >> a`` must reproduce the cylindrical-bending (beam) closed form.

    This is a fully independent check: the beam end-moment and mid-span
    deflection are analytic, not a plate-theory series.
    """
    L, D, nu, q = 1.0, 1.0, 0.3, 1.0
    for K in (5.0, 50.0):
        sol = levy_restrained_plate(L, 100.0 * L, D, nu, K, q, n_modes=6401)
        assert sol.w_centre() == pytest.approx(
            kirchhoff_strip_deflection(q, L, D, K), rel=1e-3
        )
        assert sol.moments(0.0, 50.0 * L)[0] == pytest.approx(
            -kirchhoff_strip_end_moment(q, L, D, K), rel=1e-3
        )


def test_reference_monotone_in_restraint():
    """Deflection and centre moment must decrease monotonically with ``K``."""
    D, nu, q = 1.0, 0.3, 1.0
    w = []
    m = []
    for K in (0.0, 1.0, 10.0, 100.0, 1e3, 1e6):
        sol = levy_restrained_plate(1.0, 1.0, D, nu, K, q, n_modes=401)
        w.append(sol.w_centre())
        m.append(sol.moments_centre()[0])
    assert all(b < a for a, b in zip(w, w[1:]))
    assert all(b < a for a, b in zip(m, m[1:]))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        levy_restrained_plate(-1.0, 1.0, 1.0, 0.3, 0.0, 1.0)
    with pytest.raises(ValueError):
        levy_restrained_plate(1.0, 1.0, 1.0, 0.3, -1.0, 1.0)
    with pytest.raises(ValueError):
        levy_restrained_plate(1.0, 1.0, 1.0, 0.3, 0.0, 1.0, n_modes=0)
    with pytest.raises(ValueError):
        ritz_restrained_plate(1.0, 1.0, 1.0, 0.3, -1.0, 0.0, 1.0, 4, 4)


# ---------------------------------------------------------------------------
# The FE panel against the exact reference, across the whole band
# ---------------------------------------------------------------------------

def _fem_centre(nx, ny, a, b, E, nu, t, q, beta, beta_y=0.0):
    res = solve_restrained_panel(nx, ny, a, b, E, nu, t, q, beta_x=beta, beta_y=beta_y)
    nodes = res.nodes.numpy()
    mom = moments_at_nodes(res).numpy()
    k = int(np.argmin(np.abs(nodes[:, 0] - a / 2.0) + np.abs(nodes[:, 1] - b / 2.0)))
    return float(res.w().numpy()[k]), float(mom[k, 0])


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_fem_matches_exact_levy_inside_the_band(beta):
    """Headline check: the Mindlin panel reproduces the exact Levy solution.

    The panel is a 1 m x 1 m x 10 mm steel plate, uniformly loaded, restrained
    on the ``x`` edges only (``beta_y = 0``) so that the boundary value problem
    is exactly the one the Levy series solves.  Agreement is required to 0.5 %
    on the centre deflection and the centre bending moment - the observed error
    at 20 x 20 is below 0.2 %, and the previous evidence only pinned the two
    end points.
    """
    a = b = 1.0
    E, nu, t, q = 200.0e9, 0.3, 0.01, 1000.0
    D = flexural_rigidity(E, nu, t)
    K = beta_to_zeta(beta) * D / a

    exact = levy_restrained_plate(a, b, D, nu, K, q, n_modes=401)
    w, mx = _fem_centre(20, 20, a, b, E, nu, t, q, beta)

    assert w == pytest.approx(exact.w_centre(), rel=5e-3)
    assert mx == pytest.approx(exact.moments_centre()[0], rel=5e-3)


def test_fem_band_endpoints_recover_classical_plates():
    """``beta = 0`` / ``beta = 1`` must reproduce the classical coefficients."""
    a = b = 1.0
    E, nu, t, q = 200.0e9, 0.3, 0.01, 1000.0
    D = flexural_rigidity(E, nu, t)

    # beta_y = beta_x here: *all four* edges restrained, which is the classical
    # SSSS / CCCC pair (the K = 0 / K -> inf limits of the Levy test above are
    # the two-opposite-edges pair instead).
    w0, _ = _fem_centre(40, 40, a, b, E, nu, t, q, 0.0, beta_y=0.0)
    w1, _ = _fem_centre(40, 40, a, b, E, nu, t, q, 1.0, beta_y=1.0)
    # beta = 0 -> SSSS on all four edges, beta = 1 -> clamped on all four
    assert w0 * D / (q * a**4) == pytest.approx(TIMO_SSSS_W, rel=0.02)
    assert w1 * D / (q * a**4) == pytest.approx(TIMO_CCCC_W, rel=0.05)
    assert w1 < w0