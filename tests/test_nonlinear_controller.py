import copy

from tensorfem.nonlinear_controller import (
    recommend_nonlinear_controls, verify_decision_chain,
)


def _point(force, displacement, *, balance=1e-9, energy=True):
    return {
        "force_n": force,
        "edge_shortening_m": displacement,
        "equilibrium_relative_norm": balance,
        "energy_balance_gate": {"passed": energy},
    }


def _diagnostic(residuals, reason="accepted", attempt=1):
    return {
        "attempt": attempt,
        "reason": reason,
        "iterations": [
            {"residual_relative": value} for value in residuals
        ],
    }


def test_controller_is_advisory_and_recommends_growth_only_for_stable_path():
    history = [_point(10, .001), _point(20, .002), _point(30, .003)]
    decision = recommend_nonlinear_controls(
        history, [_diagnostic([1e-2, 1e-4, 1e-7])]
    )
    assert decision["classification"] == "stable"
    assert decision["recommendation"]["step_size_factor"] == 1.25
    assert decision["automatic_switching_enabled"] is False
    assert decision["recommendation"]["newton_method"] == "modified_newton_candidate"


def test_controller_reduces_step_and_retains_robust_algorithms_for_difficulty():
    history = [_point(10, .001), _point(20, .002), _point(20.1, .003)]
    diagnostics = [
        _diagnostic([1., .9], reason="maximum_iterations", attempt=1),
        _diagnostic([1., .8], reason="maximum_iterations", attempt=2),
        _diagnostic([1., .1, .01], attempt=3),
    ]
    decision = recommend_nonlinear_controls(history, diagnostics)
    assert decision["classification"] == "difficult"
    assert decision["recommendation"] == {
        "step_size_factor": .5,
        "newton_method": "full_newton",
        "line_search": "enable",
        "arc_metric": "dimensionally_scaled",
    }


def test_controller_fails_closed_on_missing_or_failed_qualification_evidence():
    missing = recommend_nonlinear_controls([], [])
    assert missing["classification"] == "blocked"
    assert missing["recommendation"]["step_size_factor"] == 1.0

    failed = recommend_nonlinear_controls(
        [_point(1., .1, energy=False)], [_diagnostic([1., 1e-8])]
    )
    assert failed["classification"] == "unsafe"
    assert failed["recommendation"]["step_size_factor"] == .5
    assert failed["recommendation"]["newton_method"] == "full_newton"


def test_decision_chain_is_deterministic_across_restart_and_tamper_evident():
    first_history = [_point(10, .001), _point(20, .002), _point(30, .003)]
    first_diagnostics = [_diagnostic([1e-2, 1e-4, 1e-7])]
    first = recommend_nonlinear_controls(first_history, first_diagnostics)
    second_history = first_history + [_point(39, .004)]
    second_diagnostics = [_diagnostic([1e-2, 2e-3, 1e-7])]
    uninterrupted = recommend_nonlinear_controls(
        second_history, second_diagnostics,
        prior_decision_sha256=first["decision_sha256"],
    )

    # JSON persistence/restart supplies values, not Python object identity.
    replayed_first = copy.deepcopy(first)
    restarted = recommend_nonlinear_controls(
        copy.deepcopy(second_history), copy.deepcopy(second_diagnostics),
        prior_decision_sha256=replayed_first["decision_sha256"],
    )
    assert restarted == uninterrupted
    assert verify_decision_chain([replayed_first, restarted])

    tampered = copy.deepcopy(restarted)
    tampered["recommendation"]["step_size_factor"] = 2.0
    assert not verify_decision_chain([replayed_first, tampered])
