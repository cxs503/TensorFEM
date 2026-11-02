"""Verification of the observed-order / GCI evidence evaluator.

Every reference in this file is an *analytic* truth (an exact convergence rate or
an exact limit), never a value produced by the code under test.
"""
import math

import pytest

from tensorfem.convergence_verification import (
    ConvergenceLevel,
    evaluate_convergence_study,
    grid_convergence_index,
    observed_order_from_errors,
    refinement_ratio,
    richardson_extrapolate,
    richardson_observed_order,
)


def _levels(func, h0, ratio, count):
    return [ConvergenceLevel(h0 / ratio**k, func(h0 / ratio**k), f"h={h0 / ratio**k:g}")
            for k in range(count)]


@pytest.mark.parametrize("power", [1.0, 1.5, 2.0, 3.0, 4.0])
def test_observed_order_recovers_analytic_rate(power):
    """A family f(h) = A + C h**p must be reported with order p (exact rate)."""
    levels = _levels(lambda h: 3.0 - 7.0 * h**power, 0.4, 2.0, 4)
    report = evaluate_convergence_study(levels, expected_order=power, order_band=1e-6)
    assert report["status"] == "passed"
    assert report["finest_order"] == pytest.approx(power, rel=1e-9)
    assert report["richardson_value"] == pytest.approx(3.0, rel=1e-9)


def test_richardson_extrapolation_hits_the_exact_limit():
    """f(h) = 1 + 3h^2 - 5h^3: measured order, extrapolated limit 1."""
    levels = _levels(lambda h: 1.0 + 3.0 * h**2 - 5.0 * h**3, 0.2, 2.0, 5)
    order = richardson_observed_order(levels[-3], levels[-2], levels[-1])
    assert order == pytest.approx(2.0, abs=0.25)  # h^3 term perturbs the tail
    extrapolated = richardson_extrapolate(levels[-2], levels[-1], order)
    finest_error = abs(levels[-1].value - 1.0)
    # extrapolation removes at least an order of magnitude of the finest error
    assert abs(extrapolated - 1.0) < 0.1 * finest_error


def test_gci_band_covers_the_analytic_error():
    """Roache's contract: the fine GCI band contains the true (analytic) error."""
    levels = _levels(lambda h: 2.0 + 5.0 * h**2, 0.25, 2.0, 3)
    gci = grid_convergence_index(levels[-2], levels[-1], 2.0)
    change = abs(levels[-1].value - levels[-2].value)
    assert gci["absolute_gci"] == pytest.approx(1.25 * change / (2.0**2 - 1.0), rel=1e-12)
    assert gci["relative_gci"] == pytest.approx(gci["absolute_gci"] / levels[-1].value, rel=1e-12)
    assert gci["band_lower"] <= 2.0 <= gci["band_upper"]
    assert abs(levels[-1].value - 2.0) <= gci["absolute_gci"]


def test_gci_shrinks_with_refinement():
    levels = _levels(lambda h: 1.0 + h**2, 0.4, 2.0, 4)
    coarse_pair = grid_convergence_index(levels[0], levels[1], 2.0)
    fine_pair = grid_convergence_index(levels[-2], levels[-1], 2.0)
    assert fine_pair["absolute_gci"] < coarse_pair["absolute_gci"]
    # for an O(h^2) family the band shrinks by r^p = 4 per refinement
    assert fine_pair["absolute_gci"] == pytest.approx(
        coarse_pair["absolute_gci"] / 4.0**2, rel=1e-9
    )


def test_non_monotone_sequence_is_blocked_and_never_extrapolated():
    levels = [
        ConvergenceLevel(0.4, 10.0),
        ConvergenceLevel(0.2, 9.0),
        ConvergenceLevel(0.1, 9.4),  # oscillation: no order exists
        ConvergenceLevel(0.05, 9.35),
    ]
    report = evaluate_convergence_study(levels)
    assert report["status"] == "blocked"
    assert report["passed"] is False
    assert "richardson_value" not in report
    assert "non-monotone" in report["blocked_reason"]


def test_constant_refinement_ratio_is_enforced():
    levels = [
        ConvergenceLevel(0.4, 1.6),
        ConvergenceLevel(0.2, 1.15),
        ConvergenceLevel(0.1, 1.0375),
        ConvergenceLevel(0.075, 1.02),  # ratio 4/3, not constant
    ]
    with pytest.raises(ValueError, match="constant refinement ratio"):
        evaluate_convergence_study(levels)
    with pytest.raises(ValueError, match="constant refinement ratio"):
        richardson_observed_order(ConvergenceLevel(0.4, 1.0), ConvergenceLevel(0.2, 2.0),
                                  ConvergenceLevel(0.05, 2.9))


def test_identical_levels_report_machine_precision_without_fake_order():
    levels = [ConvergenceLevel(h, 7.25) for h in (0.4, 0.2, 0.1)]
    report = evaluate_convergence_study(levels, reference_value=7.25)
    assert report["status"] == "passed"
    assert report["mean_observed_order"] is None
    assert report["richardson_value"] == pytest.approx(7.25)
    assert "machine_precision_convergence" in [c["name"] for c in report["checks"]]


def test_converged_but_wrong_is_failed_by_the_reference_check():
    """Self-consistency must not mask a systematic offset."""
    levels = _levels(lambda h: 2.2 + 5.0 * h**2, 0.25, 2.0, 3)
    report = evaluate_convergence_study(
        levels, expected_order=2.0, reference_value=2.0, reference_tolerance=0.03
    )
    assert report["status"] == "failed"
    assert report["reference"]["within_tolerance"] is False
    assert report["reference"]["extrapolated_relative_error"] == pytest.approx(0.1, rel=1e-6)


def test_reference_error_outside_gci_band_is_flagged():
    """A too-small band (wrong observed order) must be reported, not hidden."""
    levels = [ConvergenceLevel(h, 1.0 + 1e-3 * h) for h in (0.4, 0.2, 0.1)]
    report = evaluate_convergence_study(
        levels, expected_order=1.0, order_band=0.5,
        reference_value=1.01, reference_tolerance=1.0,
    )
    assert report["reference"]["finest_inside_gci"] is False
    assert "finest_error_inside_gci_band" in [
        c["name"] for c in report["checks"] if not c["passed"]
    ]
    assert report["status"] == "failed"


def test_observed_order_from_errors_matches_analytic_rate():
    errors = [2.0 * h**3 for h in (0.4, 0.2, 0.1, 0.05)]
    orders = observed_order_from_errors(errors, [2.0, 2.0, 2.0])
    assert orders == pytest.approx([3.0, 3.0, 3.0], rel=1e-12)
    with pytest.raises(ValueError):
        observed_order_from_errors([1.0, 0.0], [2.0])


def test_refinement_ratio_and_input_validation():
    with pytest.raises(ValueError, match="decreasing mesh size"):
        refinement_ratio(ConvergenceLevel(0.1, 1.0), ConvergenceLevel(0.2, 1.0))
    with pytest.raises(ValueError, match="finite and positive"):
        ConvergenceLevel(0.0, 1.0)
    with pytest.raises(ValueError, match="at least three"):
        evaluate_convergence_study([ConvergenceLevel(0.4, 1.0), ConvergenceLevel(0.2, 1.1)])
    with pytest.raises(ValueError, match="safety_factor"):
        grid_convergence_index(ConvergenceLevel(0.4, 1.0), ConvergenceLevel(0.2, 1.5), 2.0,
                               safety_factor=0.0)


def test_report_is_json_serialisable_and_carries_levels():
    import json

    levels = _levels(lambda h: 4.0 + h**2, 0.2, 2.0, 3)
    report = evaluate_convergence_study(levels, expected_order=2.0, reference_value=4.0)
    payload = json.dumps(report)
    assert "tensorfem.convergence-verification/1.0" in payload
    assert len(report["levels"]) == 3
    assert all(math.isfinite(p) for p in report["observed_orders"])