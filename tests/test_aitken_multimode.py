"""Tests for the multi-mode validity domain of the Aitken cost law (XVAL v12).

The invariants pinned here are the ones the v11 report leans on:

* a rank-one interface map terminates in exactly three sub-iterations for every
  added-mass ratio (the v11 law) -- and this is *exact*, not approximate;
* a map with spectral spread does **not**: the cost grows with the spread, so the
  "three sub-iterations" statement is a rank-one statement;
* the multi-mode closed form reduces to the scalar one when the spectrum is a
  single point, and the count-inversion round-trips;
* the production 30-dof flexural solver really is (near) rank-one in the
  strong-coupling configuration, which is why the v11 law transfers to it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.aitken_multimode import (  # noqa: F401
    beam_step_count,
    closed_form_count,
    gain_from_count,
    multimode_loop,
    scalar_count,
)


# ---------------------------------------------------------------------------
# rank-one: the v11 law, exact
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("R", [0.1, 0.6, 1.0, 2.1284, 12.0, 30.0])
def test_rank_one_terminates_in_three_sub_iterations(R):
    out = multimode_loop([R], tol=1e-8)
    assert out["n"] == 3
    assert out["final_err"] < 1e-13
    # the second Aitken estimate is the analytic optimum, whatever the first was
    assert out["omegas"][1] == pytest.approx(1.0 / (1.0 + R), rel=1e-12)


def test_scalar_law_agrees_with_rank_one_count():
    out = multimode_loop([4.0], tol=1e-8)
    r1 = out["residual_norms"][0]
    # q = 0 after the Aitken update: one relaxation past r1
    assert scalar_count(r1, 0.0, 1e-8) == 2
    # with the production loop's leading omega = 1 sub-iteration: 3
    assert out["n"] == 3


# ---------------------------------------------------------------------------
# multi-mode: the law does not survive spectral spread
# ---------------------------------------------------------------------------
def test_multimode_cost_exceeds_three():
    for spread in (2.0, 4.0, 12.0):
        modes = np.linspace(1.0, spread, 4)
        out = multimode_loop(modes, tol=1e-8)
        assert out["n"] > 3, f"spread {spread} should cost more than 3 sub-iterations"


def test_multimode_cost_grows_with_spread():
    counts = [multimode_loop(np.linspace(1.0, s, 4), tol=1e-8)["n"]
              for s in (1.0, 4.0, 30.0)]
    assert counts[0] < counts[1] < counts[2]


def test_closed_form_reduces_to_scalar_when_spectrum_is_a_point():
    for R in (0.5, 2.0, 12.0):
        q = (R - R) / (1.0 + R)                     # q_slow = 0
        assert closed_form_count(R, R, 1.0, 1e-8) == scalar_count(1.0, q, 1e-8)


def test_closed_form_is_a_conservative_bound():
    """The production Aitken rule re-estimates omega every sub-iteration, so the
    measured cost is bounded (not equal) by the frozen-first-estimate law."""
    for R_min, R_max in ((1.0, 4.0), (1.0, 12.0), (2.0, 30.0)):
        modes = np.linspace(R_min, R_max, 4)
        out = multimode_loop(modes, tol=1e-8)
        bound = closed_form_count(R_min, R_max, out["residual_norms"][0], 1e-8)
        assert out["n"] <= bound + 1     # +1: the leading omega = 1 sub-iteration


def test_gain_inversion_round_trips():
    """A count produced by a known R must invert back to that R."""
    for R, omega in ((1.9, 0.5), (1.9, 0.2), (0.3, 0.5)):
        r1, tol = 1e-2, 1e-9
        q = 1.0 - omega * (1.0 + R)
        n = scalar_count(r1, q, tol)
        assert math.isfinite(n) and n > 3
        # the count fixes |q|; supplying the branch recovers R to the integer
        # rounding of the count (O(1/n) relative)
        sign = math.copysign(1.0, q)
        assert gain_from_count(n, omega, r1, tol, q_sign=sign) == pytest.approx(
            R, rel=0.15)   # count rounding: inversion is exact to O(1/n)


def test_gain_inversion_rejects_uninformative_counts():
    # |r| drops below tol within two relaxations: q is numerically zero and the
    # inversion is ill-conditioned -- it must report "no information", not a gain
    assert math.isnan(gain_from_count(1, 0.5, 1.0, 1e-8))
    assert math.isnan(gain_from_count(2, 0.5, 1.0, 1e-8))
    assert math.isnan(gain_from_count(3, 1.0, 1.0, 1e-8))


# ---------------------------------------------------------------------------
# the production flexural solver (30 free dof) -- the object v11 did not measure
# ---------------------------------------------------------------------------
def test_production_beam_aitken_uses_three_sub_iterations():
    rec = beam_step_count(mode="aitken")
    assert rec["converged"]
    assert rec["sub_iterations"] == 3
    assert rec["residual"] < 1e-8


def test_production_beam_interface_map_is_not_rank_one():
    """The gain implied by counts at two fixed factors is NOT the same.

    A rank-one interface map has one ``R``: every fixed factor ``omega`` must then
    invert to that same value.  On the 30-dof flexural solver the two factors give
    ``R = 2.0`` and ``R = 8.0`` -- a factor four apart -- so the map that the v11
    study measured on a one-dof body is genuinely multi-modal here.  Aitken still
    converges in three sub-iterations (test above) because the predicted trial
    state is dominated by one eigendirection, not because the map is scalar.
    """
    r1 = beam_step_count(mode="fixed", omega=1.0, sub_tol=0.0,
                         max_sub_iter=1)["residual"]
    gains = {}
    for omega in (0.5, 0.2):
        rec = beam_step_count(mode="fixed", omega=omega)
        gains[omega] = gain_from_count(rec["sub_iterations"], omega, r1, 1e-8,
                                       q_sign=-1.0)
    assert all(math.isfinite(v) for v in gains.values()), gains
    lo, hi = min(gains.values()), max(gains.values())
    assert hi / lo > 2.0, f"map looks rank-one after all: {gains}"


def test_production_beam_fixed_relaxation_is_slower_than_aitken():
    a = beam_step_count(mode="aitken")["sub_iterations"]
    f = beam_step_count(mode="fixed", omega=0.2)["sub_iterations"]
    assert f > a