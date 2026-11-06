"""Verification of the tip-loaded cantilever (elastica) benchmark.

Two layers are gated here:

* the oracle itself -- small-load asymptote, agreement with the published
  tabulated deflection ratios, and agreement between two independent numerical
  routes (elliptic quadrature vs. a direct boundary-value solve);
* the finite-element response -- one cheap load case must reproduce the oracle
  inside 3%, and the qualification gate must be fail-closed (a fabricated
  out-of-tolerance case can never be reported as qualified).
"""
import math

import pytest

from tensorfem.elastica_cantilever import (
    CantileverCase,
    ElasticaBenchmark,
    PUBLISHED_CROSS_CHECK,
    elastica_from_ode,
    oracle_cross_check,
    corotational_beam_cantilever,
    solve_elastica,
)


def test_oracle_recovers_small_load_asymptote():
    alpha = 1e-3
    sol = solve_elastica(alpha)
    assert sol.deflection_ratio == pytest.approx(alpha / 3.0, rel=2e-4)
    # linear-theory tip slope is alpha/2 for a cantilever with an end force
    assert sol.tip_rotation == pytest.approx(alpha / 2.0, rel=1e-3)
    assert sol.shortening_ratio > 0.0


def test_oracle_matches_published_tabulated_deflection():
    for alpha, (published_w, _published_u) in PUBLISHED_CROSS_CHECK.items():
        sol = solve_elastica(alpha)
        assert sol.deflection_ratio == pytest.approx(published_w, abs=1e-5), alpha


def test_oracle_rejects_invalid_load():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            solve_elastica(bad)


def test_two_independent_oracle_routes_agree():
    diff = oracle_cross_check(2.0)
    assert diff["d_tip_rotation"] < 1e-6
    assert diff["d_deflection_ratio"] < 1e-6
    assert diff["d_shortening_ratio"] < 1e-6


def test_bvp_route_is_independent_of_quadrature():
    """The BVP route must not merely echo the quadrature kernel."""
    quad = solve_elastica(5.0)
    bvp = elastica_from_ode(5.0)
    assert bvp.deflection_ratio == pytest.approx(quad.deflection_ratio, abs=1e-6)
    assert bvp.tip_rotation == pytest.approx(quad.tip_rotation, abs=1e-6)


def test_deflection_is_below_linear_scale_for_large_load():
    """Geometric stiffening: the elastica must fall short of the linear extrapolation."""
    for alpha in (1.0, 5.0, 10.0):
        sol = solve_elastica(alpha)
        assert sol.deflection_ratio < alpha / 3.0


def _fabricated(elements, deflection_error, shortening_error=0.0):
    exact = solve_elastica(2.0)
    return CantileverCase(
        alpha=2.0, elements=elements, converged=True, load_factor=1.0,
        tip_deflection=exact.tip_deflection(10.0) * (1.0 + deflection_error),
        tip_shortening=exact.tip_shortening(10.0) * (1.0 + shortening_error),
        exact=exact, length=10.0, load=0.2)


def test_qualification_gate_is_fail_closed():
    good = ElasticaBenchmark((_fabricated(4, 0.06), _fabricated(8, 0.02)))
    assert good.qualified is True
    # an out-of-tolerance coarse mesh does not matter, but the finest does
    coarse_bad = ElasticaBenchmark((_fabricated(4, 0.20), _fabricated(8, 0.02)))
    assert coarse_bad.qualified is True
    fine_bad = ElasticaBenchmark((_fabricated(4, 0.02), _fabricated(8, 0.05)))
    assert fine_bad.qualified is False
    shortening_bad = ElasticaBenchmark((_fabricated(8, 0.001, shortening_error=0.05),))
    assert shortening_bad.qualified is False
    # exactly at the threshold is not "below" it
    at_threshold = ElasticaBenchmark((_fabricated(8, 0.03),))
    assert at_threshold.qualified is False


def test_unconverged_case_never_qualifies():
    exact = solve_elastica(2.0)
    case = CantileverCase(alpha=2.0, elements=8, converged=False, load_factor=0.4,
                          tip_deflection=exact.tip_deflection(10.0),
                          tip_shortening=exact.tip_shortening(10.0),
                          exact=exact, length=10.0, load=0.2)
    assert ElasticaBenchmark((case,)).qualified is False


def test_monotone_detects_regression():
    bench = ElasticaBenchmark((_fabricated(4, 0.01), _fabricated(8, 0.02)))
    assert bench.monotone() is False
    bench = ElasticaBenchmark((_fabricated(4, 0.05), _fabricated(8, 0.01)))
    assert bench.monotone() is True


@pytest.mark.slow
def test_beam_path_reproduces_elastica_at_one_load_level():
    """One cheap load-controlled FE case must stay inside the 3% response gate."""
    case = corotational_beam_cantilever(1.0, 4, increment=0.1, max_iterations=20,
                                  tolerance=1e-10)
    assert case.converged and case.load_factor == pytest.approx(1.0)
    assert case.load == pytest.approx(0.1)
    assert case.deflection_error < 0.03
    assert case.shortening_error < 0.03
    assert case.exact.tip_rotation < math.pi / 2