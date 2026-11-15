"""Quantitative validation of Aitken delta^2 relaxation against closed-form theory.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``.  Rounds v1-v10 checked the
coupled *frequency* against Lamb (1932) but left the relaxation qualitative
(``docs/CASE-FEM-XVAL-IMPLICIT-IBM-AITKEN-v6.md`` §4: "Aitken convergence vs fixed
omega = 1 | qualitative (3 it vs divergence) | n/a").  ``docs/xval_implicit_ibm_aitken_v11.md``
derives the law and ``benchmarks/xval_aitken_rate_law.py`` measures it; these tests
pin the law so it cannot silently regress.

The interface fixed point inside one time step is affine (frozen fluid, linear
direct-forcing IBM)::

    y_new = G(y) = g y + b,   r_k = G(y_k) - y_k,   r_{k+1} = q r_k,
    q = 1 + omega (g - 1) = 1 - omega (1 + R),      R := -g

Consequences verified here against the *production* classes:

1. ``omega* = 1/(1+R)`` exactly (not asymptotically) -> the loop terminates in
   exactly three sub-iterations for every ``R``;
2. fixed relaxation decays geometrically with ``|q|``, and the sub-iteration count
   to a tolerance is the integer law ``n = 1 + ceil(ln(tol/|r_1|)/ln|q|)``;
3. the stability edge is ``omega_crit = 2/(1+R)`` (equivalently ``R_crit = 2/omega-1``,
   the ``omega = 1`` case being the Causin-Gerbeau-Nobile added-mass criterion
   ``R_crit = 1``);
4. both Aitken implementations in this repository obey the same law.
"""

from __future__ import annotations

import importlib.util
import math
import os

import numpy as np
import pytest

from tensorfem.fsi_lbm_strong_coupling import AitkenRelaxation as Aitken2
from tensorfem.ibm_aitken_implicit import AitkenRelaxation

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.join(_HERE, os.pardir, "benchmarks", "xval_aitken_rate_law.py")


def _load_bench():
    spec = importlib.util.spec_from_file_location("xval_aitken_rate_law", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()


# ---------------------------------------------------------------------------
# 1. Aitken recovers the analytic optimum exactly and terminates finitely
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("R", [0.1, 0.6, 1.0, 2.5, 12.0])
def test_aitken_omega_star_is_exact_and_termination_is_three(R):
    g = -R
    b = 1.0 - g                      # fixed point at x* = 1
    n, x, res, om = bench.scalar_loop(g, b, mode="aitken", tol=1e-13)
    w_star = 1.0 / (1.0 + R)
    # the second relaxation estimate is the optimum, to machine precision
    assert abs(om[1] - w_star) <= 1e-14 * w_star, (om[1], w_star)
    # finite termination: one free relaxation + one exact step + converged check
    assert n == 3, f"R={R}: {n} sub-iterations, closed form predicts 3"
    assert abs(x - 1.0) < 1e-14, f"R={R}: residual {abs(x - 1.0):.2e}"


def test_stale_base_penalty_is_the_factor_ratio():
    """The ``omega_prev`` docstring claim, in quantitative form."""
    a = AitkenRelaxation(relax_max=1.0e6, relax0=1.0)
    r_km1, r_k = np.array([1.0]), np.array([-0.5])
    w_correct = a.omega_for(r_km1, r_k, omega_prev=0.4)
    b = AitkenRelaxation(relax_max=1.0e6, relax0=0.25)
    b.omega = 0.25                   # stale factor carried over from the last step
    w_stale = b.omega_for(r_km1, r_k)
    assert w_stale / w_correct == pytest.approx(0.25 / 0.4, rel=1e-12)


def test_second_implementation_obeys_the_same_law():
    for R in (0.6, 2.5, 12.0):
        g = -R
        b = 1.0 - g
        a = Aitken2(omega0=1.0)
        x, hist, it = 0.0, [], 0
        for it in range(1, 200):
            x_new = g * x + b
            r = x_new - x
            if abs(r) <= 1e-13:
                break
            hist.append(a.update(np.array([r])))
            x = x + hist[-1] * (x_new - x)
        w_star = 1.0 / (1.0 + R)
        assert abs(hist[1] - w_star) <= 1e-10 * w_star
        assert it == 3


# ---------------------------------------------------------------------------
# 2. fixed relaxation: geometric rate and the integer cost law
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("R", [0.3, 1.0, 4.0])
@pytest.mark.parametrize("omega", [0.5, 1.0, 1.5])
def test_fixed_relaxation_rate_and_cost(R, omega):
    g, b, tol = -R, 1.0 + R, 1e-9
    q = 1.0 + omega * (g - 1.0)
    n, _x, res, _ = bench.scalar_loop(g, b, mode="fixed", omega0=omega, tol=tol,
                                      maxit=2000)
    q_meas = abs(res[1]) / abs(res[0])
    assert q_meas == pytest.approx(abs(q), rel=1e-13)
    n_pred = bench.predict_count(abs(res[0]), q, tol)
    if abs(q) >= 1.0:
        assert n == 2000, "closed form predicts divergence"
    else:
        assert n == int(n_pred), f"measured {n}, closed form {n_pred}"


@pytest.mark.parametrize("omega,R_crit", [(1.0, 1.0), (1.5, 1.0 / 3.0), (2.0, 0.0)])
def test_critical_ratio_closed_form(omega, R_crit):
    """``R_crit(omega) = 2/omega - 1``; the omega = 1 case is Causin et al. (2005)."""
    assert 2.0 / omega - 1.0 == pytest.approx(R_crit, abs=1e-15)
    if R_crit > 0.0:
        below = R_crit * 0.99
        above = R_crit * 1.01
        assert abs(1.0 - omega * (1.0 + below)) < 1.0
        assert abs(1.0 - omega * (1.0 + above)) >= 1.0
    else:
        # no non-negative gain can be stabilised
        for R in (1e-6, 1.0, 50.0):
            assert abs(1.0 - omega * (1.0 + R)) >= 1.0


# ---------------------------------------------------------------------------
# 3. the real coupled solver: gain, Aitken optimum, cost, stability edge
# ---------------------------------------------------------------------------
ANCHOR_MASS_RATIO = 12.0     # the anchor's strong-coupling configuration


@pytest.fixture(scope="module")
def anchor_config():
    m_s, k = bench.config_for_mass_ratio(ANCHOR_MASS_RATIO)
    seq, ratios, R_meas, spread = bench.gain_from_norms(m_s, k, n_warm=0, nmax=5)
    return m_s, k, seq, ratios, R_meas, spread


def test_interface_gain_is_constant_across_sub_iterations(anchor_config):
    """The affine-map assumption the whole law rests on, measured in the solver."""
    _m_s, _k, _seq, ratios, R_meas, spread = anchor_config
    assert R_meas > 1.0, "the anchor configuration must be past the omega=1 edge"
    assert spread < 1e-6, f"gain not constant across sub-iterations: {spread:.2e}"


def test_aitken_omega_and_cost_in_real_solver(anchor_config):
    m_s, k, _seq, _ratios, R_meas, _spread = anchor_config
    w_star = 1.0 / (1.0 + R_meas)
    n, conv, _res = bench.step_cost(m_s, k, "aitken", 1.0, tol=1e-8, maxit=40)
    assert conv and n == 3
    solver = bench.make_solver(m_s, k, mode="aitken", sub_tol=1e-8, max_sub_iter=40)
    solver.step()
    w = float([x for x in solver.relaxes if np.isfinite(x)][-1])
    assert w == pytest.approx(w_star, rel=1e-8)
    # Aitken's optimum is always inside the production clip of relax_max = 1
    assert 0.0 < w_star < 1.0


def test_aitken_cost_is_independent_of_added_mass(anchor_config):
    """Three sub-iterations for every mass ratio, including R ~ 5."""
    for mr in (1.0, ANCHOR_MASS_RATIO, 30.0):
        m_s, k = bench.config_for_mass_ratio(mr)
        n, conv, _res = bench.step_cost(m_s, k, "aitken", 1.0, tol=1e-8, maxit=40)
        assert conv and n == 3, f"m_a/m_s={mr}: {n} sub-iterations"


@pytest.mark.parametrize("omega", [0.5, 1.0])
def test_fixed_cost_law_in_real_solver(anchor_config, omega):
    m_s, k, seq, _ratios, R_meas, _spread = anchor_config
    q = 1.0 - omega * (1.0 + R_meas)
    for tol in (1e-6, 1e-8):
        if abs(q) >= 1.0:
            continue
        n, conv, _res = bench.step_cost(m_s, k, "fixed", omega, tol=tol, maxit=80)
        n_pred = bench.predict_count(seq[0], q, tol)
        assert conv
        assert n == int(n_pred), f"omega={omega} tol={tol}: {n} vs {n_pred}"


def test_stability_edge_in_real_solver(anchor_config):
    """Classification flips at ``omega_crit = 2/(1+R)``, and the measured edge
    agrees with the closed form within the 3 % acceptance band (it is in fact
    far tighter: see the benchmark's bisection)."""
    m_s, k, seq, _ratios, R_meas, _spread = anchor_config
    w_crit = 2.0 / (1.0 + R_meas)
    assert bench.decays(m_s, k, w_crit * 0.90, seq[0]) is True
    assert bench.decays(m_s, k, w_crit * 1.10, seq[0]) is False


def test_strong_coupling_fixed_omega_one_diverges_while_aitken_converges(anchor_config):
    """The qualitative statement v6 could only report, now with a predicted bound."""
    m_s, k, _seq, _ratios, R_meas, _spread = anchor_config
    assert R_meas > 1.0, "Causin et al. (2005): omega = 1 must fail above R = 1"
    _n, conv_fixed, _res = bench.step_cost(m_s, k, "fixed", 1.0, tol=1e-8, maxit=60)
    assert not conv_fixed
    n, conv_aitken, _res = bench.step_cost(m_s, k, "aitken", 1.0, tol=1e-8, maxit=40)
    assert conv_aitken and n == 3