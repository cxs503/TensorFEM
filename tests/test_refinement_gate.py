"""Tests for the reference-benchmark refinement gate.

These pin three properties that the two repaired anchors rely on:

1. the observed order is the *logarithmic* Richardson estimate, recovered exactly
   from synthetic data of known order;
2. a ladder that cannot support an estimate is reported as ``WITHHELD`` with a
   reason and never as a number or a pass;
3. the accuracy of an anchor is certified at the Richardson limit, so a
   finest-mesh reading that "passes" by cancellation cannot carry the anchor
   (the MacNeal-Harder pinched-cylinder regression below, from the stored
   ladder of that anchor).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfem.refinement_gate import (
    FAIL,
    PASS,
    WITHHELD,
    evaluate_reference_ladder,
    observed_orders,
    summarise,
)

REPO = Path(__file__).resolve().parents[1]
VERIFIED = REPO / "benchmarks" / "verified"


def synthetic_ladder(limit: float, order: float, constant: float, n_levels: int = 4,
                     first_mesh: int = 4):
    """QoI values from ``w_n = limit + constant * h**order`` with h = 1/n."""
    levels = []
    for i in range(n_levels):
        n = first_mesh * 2**i
        h = 1.0 / n
        levels.append((h, limit + constant * h**order, f"{n}x{n}"))
    return levels


class TestObservedOrder:
    def test_recovers_second_order(self):
        levels = synthetic_ladder(limit=1.0, order=2.0, constant=0.5)
        orders = observed_orders(levels)
        assert len(orders) == 2
        for entry in orders:
            assert entry["order"] == pytest.approx(2.0, rel=1e-9), entry

    def test_recovers_first_order(self):
        levels = synthetic_ladder(limit=-2.0, order=1.0, constant=-0.3)
        for entry in observed_orders(levels):
            assert entry["order"] == pytest.approx(1.0, rel=1e-9)

    def test_non_monotone_triple_has_no_order(self):
        # Oscillating ladder: the difference changes sign, so no order exists.
        levels = [(0.25, 1.0), (0.125, 1.2), (0.0625, 1.1), (0.03125, 1.05)]
        orders = observed_orders(levels)
        assert orders[0]["order"] is None
        assert "non-monotone" in orders[0]["reason"]
        assert math.isfinite(orders[1]["order"])


class TestEvaluateReferenceLadder:
    def test_clean_ladder_passes(self):
        levels = synthetic_ladder(limit=1.0, order=2.0, constant=0.5)
        report = evaluate_reference_ladder(
            levels, reference_value=1.0, tolerance=0.03, min_order=1.0,
            coarsest_sanity=0.5,
        )
        summary = summarise(report)
        assert summary["accuracy_verdict"] == PASS
        assert summary["study_verdict"] == PASS
        assert summary["overall_verdict"] == PASS
        assert report["certified_accuracy"] <= 0.03

    def test_wrong_limit_fails_even_when_finest_looks_good(self):
        """Cancellation: the finest level is near the reference, the limit is not.

        w(h) = 1.10 - 327.68 h^2, so the finest (h = 1/64) level reads 1.02 --
        a 2 % "pass" -- while the ladder converges to 1.10, i.e. 10 % off.
        """
        levels = synthetic_ladder(limit=1.10, order=2.0, constant=-327.68,
                                  n_levels=4, first_mesh=8)
        report = evaluate_reference_ladder(levels, reference_value=1.0, tolerance=0.03)
        gates = {g["id"]: g for g in report["gates"]}
        assert gates["accuracy_finest"]["status"] == PASS     # finest reading "passes"
        assert gates["accuracy_asymptotic"]["status"] == FAIL  # the ladder does not
        assert report["passed"] is False

    def test_unresolvable_ladder_is_withheld_not_passed(self):
        levels = [(0.25, 1.0), (0.125, 1.2), (0.0625, 1.1), (0.03125, 1.15)]
        report = evaluate_reference_ladder(levels, reference_value=1.1, tolerance=0.5,
                                           min_order=1.0)
        gates = {g["id"]: g for g in report["gates"]}
        assert gates["accuracy_asymptotic"]["status"] == WITHHELD
        assert gates["observed_order"]["status"] == WITHHELD
        # WITHHELD criteria never amount to a pass, whatever else fails.
        assert report["status"] in (WITHHELD, FAIL)
        assert report["passed"] is False
        assert report["certified_accuracy"] is None

    def test_requires_three_levels(self):
        with pytest.raises(ValueError):
            evaluate_reference_ladder([(0.5, 1.0), (0.25, 1.01)], reference_value=1.0)


class TestAnchorArtifacts:
    """Bind the stored anchor evidence to the gate semantics (no solves here)."""

    def _ladder(self, case: str, reference: float):
        result = json.loads((VERIFIED / case / "result.json").read_text())
        if case == "fem_pinched_cylinder":
            levels = [(1.0 / math.sqrt(lv["n_elements"]), lv["displacement"])
                      for lv in result["levels"]]
        else:
            levels = [(1.0 / lv["nx"], lv["probe_displacement"])
                      for lv in result["cylindrical_results"]]
        return evaluate_reference_ladder(levels, reference_value=reference, tolerance=0.03,
                                         min_order=1.0, coarsest_sanity=0.5)

    def test_pinched_cylinder_certified_accuracy_is_not_the_finest_reading(self):
        report = self._ladder("fem_pinched_cylinder", -1.8248e-05)
        gates = {g["id"]: g for g in report["gates"]}
        # The 32x32 reading passes 3%, but the ladder converges to 3.27%.
        assert gates["accuracy_finest"]["status"] == PASS
        assert report["reported_accuracy_finest"] == pytest.approx(0.018162, abs=1e-5)
        assert report["certified_accuracy"] == pytest.approx(0.032719, abs=1e-4)
        assert gates["accuracy_asymptotic"]["status"] == FAIL
        assert gates["monotone_qoi"]["status"] == FAIL
        assert report["passed"] is False
        # The old hand-rolled estimator produced 4.325 for this same data.
        assert gates["observed_order"]["actual"] == pytest.approx(1.09279, abs=1e-4)

    def test_scordelis_lo_cylindrical_accuracy_passes_but_study_is_withheld(self):
        report = self._ladder("fem_shell_scordelis_lo", -0.3024)
        summary = summarise(report)
        assert summary["accuracy_verdict"] == PASS
        assert report["certified_accuracy"] == pytest.approx(0.005167, abs=1e-5)
        assert summary["study_verdict"] == WITHHELD
        assert summary["overall_verdict"] == WITHHELD

    def test_stored_gate_documents_are_honest(self):
        for case in ("fem_pinched_cylinder", "fem_shell_scordelis_lo"):
            doc = json.loads((VERIFIED / case / "gates_verified.json").read_text())
            assert doc["schema"] == "tensorfem.verified-gates.v1"
            statuses = {g["status"] for g in doc["gates"]}
            assert statuses <= {"PASS", "FAIL", "WITHHELD"}, (case, statuses)
            # A document may not claim PASS while carrying a failed certifying gate.
            if doc["overall_verdict"] == PASS:
                assert all(g["status"] == PASS for g in doc["gates"] if g["id"].startswith(("G1", "G2", "cylindrical", "accuracy"))), case
            assert doc["passed"] == (doc["overall_verdict"] == PASS)