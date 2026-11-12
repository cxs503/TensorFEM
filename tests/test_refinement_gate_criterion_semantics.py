"""Every graded ladder gate must be a re-checkable comparison.

Regression tests for the two certificate defects repaired in the Scordelis-Lo
anchor (m16947), both rooted in :mod:`tensorfem.refinement_gate`:

* ``monotone_qoi`` was emitted as ``PASS`` with ``actual=None`` and
  ``threshold=None`` -- a verdict that no third party can recompute
  (audit code ``E_PASS_WITHOUT_CRITERION``).
* ``asymptotic_range_consistency`` set ``threshold`` to the *same* GCI band it
  reported as ``actual``, so ``value <= threshold`` was trivially true: a
  self-certifying criterion is no criterion.

Both are checked structurally here (a graded verdict needs finite
``actual``/``threshold`` plus an ``operator``) and numerically (the monotone
metric reproduces the old strict-monotonicity semantics; the asymptotic metric
is ``|e_finest| / GCI`` against 1).
"""

from __future__ import annotations

import math

import pytest

from tensorfem.refinement_gate import (
    FAIL,
    PASS,
    WITHHELD,
    GateVerdict,
    evaluate_reference_ladder,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _gate(report, gate_id):
    for gate in report["gates"]:
        if gate["id"] == gate_id:
            return gate
    raise AssertionError(f"gate {gate_id!r} missing from {[g['id'] for g in report['gates']]}")


def _graded_gates(report):
    return [g for g in report["gates"] if g["status"] != WITHHELD]


def _converging_ladder(p: float = 2.0, limit: float = 1.0):
    """Synthetic ladder with a clean observed order ``p`` (no reference bias)."""
    values = [limit + 3.0 * (0.5 ** (p * k)) for k in range(4)]
    return [(0.5**k, values[k], f"h{k}") for k in range(4)]


# --------------------------------------------------------------------------- #
# structural invariants
# --------------------------------------------------------------------------- #
def test_graded_gate_cannot_be_declared_without_numeric_criterion():
    """PASS/FAIL with a null criterion is refused at construction time."""
    with pytest.raises(ValueError, match="withhold the gate"):
        GateVerdict(id="x", description="d", status=PASS, actual=None,
                    threshold=None, detail="")
    with pytest.raises(ValueError, match="withhold the gate"):
        GateVerdict(id="x", description="d", status=FAIL, actual=1.0,
                    threshold=None, detail="")
    with pytest.raises(ValueError, match="withhold the gate"):
        GateVerdict(id="x", description="d", status=PASS, actual=float("nan"),
                    threshold=1.0, detail="")
    # WITHHELD is allowed to carry no criterion: it asserts nothing.
    GateVerdict(id="x", description="d", status=WITHHELD, actual=None,
                threshold=None, detail="")


def test_every_gate_carries_a_machine_readable_operator():
    report = evaluate_reference_ladder(
        _converging_ladder(), reference_value=1.0, tolerance=0.03,
        min_order=1.0, coarsest_sanity=0.5,
    )
    for gate in report["gates"]:
        assert gate["operator"] in ("<=", ">="), gate["id"]


def test_graded_gates_are_recomputable_from_their_own_numbers():
    report = evaluate_reference_ladder(
        _converging_ladder(), reference_value=1.0, tolerance=0.03,
        min_order=1.0, coarsest_sanity=0.5,
    )
    graded = _graded_gates(report)
    assert graded, "ladder should grade at least one gate"
    for gate in graded:
        actual, threshold = gate["actual"], gate["threshold"]
        assert actual is not None and threshold is not None, gate["id"]
        assert math.isfinite(actual) and math.isfinite(threshold), gate["id"]
        recomputed = actual <= threshold if gate["operator"] == "<=" else actual >= threshold
        assert recomputed == gate["passed"], gate["id"]


# --------------------------------------------------------------------------- #
# monotonicity metric
# --------------------------------------------------------------------------- #
def _ladder_with_values(values):
    return [(0.5**k, v, f"h{k}") for k, v in enumerate(values)]


@pytest.mark.parametrize(
    "values, expected_status, expected_transitions",
    [
        ([1.10, 1.05, 1.02, 1.01], PASS, 0.0),      # strictly decreasing
        ([0.90, 0.95, 0.98, 0.99], PASS, 0.0),      # strictly increasing
        ([0.90, 1.05, 0.98, 1.01], FAIL, 2.0),      # two sign flips
        ([0.90, 0.99, 0.99, 1.00], FAIL, 2.0),      # plateau: two flat transitions
    ],
)
def test_monotone_gate_reports_transition_count(values, expected_status, expected_transitions):
    report = evaluate_reference_ladder(_ladder_with_values(values), reference_value=1.0)
    gate = _gate(report, "monotone_qoi")
    assert gate["status"] == expected_status
    assert gate["actual"] == expected_transitions
    assert gate["threshold"] == 0.0
    assert gate["operator"] == "<="
    assert gate["passed"] is (expected_status == PASS)


# --------------------------------------------------------------------------- #
# asymptotic-range metric
# --------------------------------------------------------------------------- #
def test_asymptotic_gate_threshold_is_not_its_own_value():
    """The GCI band may not be both the measurement and the criterion.

    The band depends only on the ladder; the reported quantity must scale with
    the *true* finest-mesh error.  Moving the reference therefore scales the
    gate value proportionally -- which is impossible for a threshold that is
    merely a copy of the band.
    """
    ladder = _converging_ladder()
    base = evaluate_reference_ladder(ladder, reference_value=1.0, tolerance=0.03)
    shifted = evaluate_reference_ladder(ladder, reference_value=1.2, tolerance=0.03)
    g_base = _gate(base, "asymptotic_range_consistency")
    g_shift = _gate(shifted, "asymptotic_range_consistency")
    assert g_base["threshold"] == 1.0 and g_shift["threshold"] == 1.0
    e_base = abs(ladder[-1][1] - 1.0) / 1.0
    e_shift = abs(ladder[-1][1] - 1.2) / 1.2
    assert g_shift["actual"] / g_base["actual"] == pytest.approx(e_shift / e_base, rel=1e-12)


def test_asymptotic_gate_withholds_when_ladder_is_pre_asymptotic():
    """A reference far outside the band must be reported WITHHELD, not FAIL."""
    report = evaluate_reference_ladder(_converging_ladder(), reference_value=1.35,
                                       tolerance=0.03)
    gate = _gate(report, "asymptotic_range_consistency")
    assert gate["status"] == WITHHELD
    assert gate["actual"] > 1.0          # measurable, and honestly out of band
    assert gate["threshold"] == 1.0
    assert gate["passed"] is False