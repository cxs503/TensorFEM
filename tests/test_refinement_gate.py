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

from tensorfem.convergence_verification import evaluate_convergence_study
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


class TestPreAsymptoticLadder:
    """A ladder the estimator cannot extrapolate must be graded, not crash.

    Regression for the shape of the MacNeal-Harder and Scordelis-Lo ladders
    after the m18206 shell-element repair: the values still move towards the
    reference, but the *increments* grow, so the observed order on a triple is
    zero or negative and ``r**p - 1 <= 0``.  ``grid_convergence_index`` raises
    on that input, and the study used to let the exception escape -- every
    consumer (the audit, the anchors, the certificate tests) then crashed
    instead of recording an honest WITHHELD.
    """

    def test_negative_order_on_the_finest_triple_is_withheld_not_raised(self):
        # increments grow: 0.1 -> 0.2 -> 0.35 (order < 1, and r**p-1 -> 0)
        levels = [(0.25, 1.0), (0.125, 1.1), (0.0625, 1.3), (0.03125, 1.65)]
        report = evaluate_reference_ladder(levels, reference_value=1.0, tolerance=0.03)
        assert report["certified_accuracy"] is None
        gates = {g["id"]: g for g in report["gates"]}
        assert gates["accuracy_asymptotic"]["status"] == WITHHELD
        assert "non-positive observed order" in gates["accuracy_asymptotic"]["detail"]
        assert report["passed"] is False

    def test_negative_order_on_a_coarser_triple_is_not_compared(self):
        # First triple ascends slowly (p < 0), the finest triple is clean.
        levels = [(0.25, 1.0), (0.125, 1.01), (0.0625, 1.03), (0.03125, 1.04)]
        report = evaluate_reference_ladder(levels, reference_value=1.04, tolerance=0.03)
        gates = {g["id"]: g for g in report["gates"]}
        # The finest pair is still graded; the band contract is recorded as
        # unevaluable rather than compared against a degenerate order.
        assert gates["accuracy_asymptotic"]["status"] in (PASS, FAIL)
        study = evaluate_convergence_study(levels, reference_value=1.04)
        names = [c["name"] for c in study["unevaluable_checks"]]
        assert "gci_band_contracts" in names
        assert study["gci"] is not None


class TestAnchorArtifacts:
    """Bind the stored anchor evidence to the gate semantics (no solves here)."""

    REFERENCE = {"fem_pinched_cylinder": -1.8248e-05, "fem_shell_scordelis_lo": -0.3024}

    @staticmethod
    def _ladders(case: str):
        """The stored refinement ladders of one anchor, keyed by graded element."""
        result = json.loads((VERIFIED / case / "result.json").read_text())
        if case == "fem_pinched_cylinder":
            return {"G": [(1.0 / math.sqrt(lv["n_elements"]), lv["displacement"])
                          for lv in result["levels"]]}
        return {
            element: [(1.0 / lv["nx"], lv["probe_displacement"])
                      for lv in result[f"{element}_results"]]
            for element in ("flat_facet", "cylindrical")
        }

    @staticmethod
    def _element_of(case: str, gate_id: str) -> str:
        return "G" if case == "fem_pinched_cylinder" else gate_id.split(".", 1)[0]

    @staticmethod
    def _gate_key(case: str, gate_id: str) -> str:
        # "G1_accuracy_finest" -> "accuracy_finest"; "cylindrical.accuracy_finest"
        # -> "accuracy_finest".
        return gate_id.split("_", 1)[1] if case == "fem_pinched_cylinder" else gate_id.split(".", 1)[1]

    #: Criteria the two anchors are graded with (tolerance 3 %, order >= 1,
    #: coarsest mesh within 50 %).  Kept here as numbers because the stored
    #: documents state some of them in prose ("<= 50%").
    CRITERIA = {"tolerance": 0.03, "min_order": 1.0, "coarsest_sanity": 0.5}

    def _reports(self, case: str):
        return {
            element: evaluate_reference_ladder(
                levels, reference_value=self.REFERENCE[case], **self.CRITERIA,
            )
            for element, levels in self._ladders(case).items()
        }

    @pytest.mark.parametrize("case", ["fem_pinched_cylinder", "fem_shell_scordelis_lo"])
    def test_stored_gate_documents_reproduce_from_the_stored_ladder(self, case):
        """``gates_verified.json`` must be exactly the grading of ``result.json``.

        The *numbers* are deliberately not frozen: the element under test can
        change and the anchor then has to be re-emitted.  What is frozen is the
        binding between the two artifacts, so a re-run can never leave a stale
        certificate behind (which is how the pinched-cylinder document kept a
        pre-repair PASS next to a ladder that had moved to 16 %).
        """
        doc = json.loads((VERIFIED / case / "gates_verified.json").read_text())
        reports = self._reports(case)
        assert doc["gates"], case
        for gate in doc["gates"]:
            report = reports[self._element_of(case, gate["id"])]
            recomputed = {g["id"]: g for g in report["gates"]}[self._gate_key(case, gate["id"])]
            assert gate["status"] == recomputed["status"], (case, gate["id"])
            assert gate["operator"] == recomputed["operator"], (case, gate["id"])
            if recomputed["threshold"] is None:
                assert gate["threshold"] is None, (case, gate["id"])
            else:
                assert float(gate["threshold"]) == pytest.approx(float(recomputed["threshold"]))
            if recomputed["actual"] is None:
                assert gate["actual"] is None, (case, gate["id"])
            else:
                assert float(gate["actual"]) == pytest.approx(float(recomputed["actual"]),
                                                              rel=1e-9, abs=1e-12)
        # ... and the overall verdict is the roll-up of those same gates.
        for element, report in reports.items():
            summary = summarise(report)
            if case == "fem_pinched_cylinder":
                stored = doc["verdicts"]
                assert doc["overall_verdict"] == summary["overall_verdict"]
                assert doc["certified_accuracy"] == pytest.approx(report["certified_accuracy"]) \
                    if report["certified_accuracy"] is not None else doc["certified_accuracy"] is None
                assert doc["reported_accuracy_finest"] == pytest.approx(
                    report["reported_accuracy_finest"], rel=1e-9, abs=1e-12)
            else:
                stored = doc["verdicts"][element]
                reported = doc["reported_vs_certified"][element]
                assert reported["reported_finest_error"] == pytest.approx(
                    report["reported_accuracy_finest"], rel=1e-9, abs=1e-12)
                certified = report["certified_accuracy"]
                if certified is None:
                    assert reported["certified_error"] is None
                else:
                    assert reported["certified_error"] == pytest.approx(certified, rel=1e-9, abs=1e-12)
            for key in ("accuracy_verdict", "study_verdict", "overall_verdict", "passed"):
                assert stored[key] == summary[key], (case, element, key)

    @pytest.mark.parametrize("case", ["fem_pinched_cylinder", "fem_shell_scordelis_lo"])
    def test_certified_accuracy_is_the_ladder_limit_not_the_finest_reading(self, case):
        """A ladder without a Richardson limit must not be certified.

        The claim an anchor makes is the asymptotic accuracy, so a document is
        only allowed to carry a ``certified`` number when the grading produced
        one; a pre-asymptotic ladder (non-positive observed order) has none and
        must say so with ``None`` instead of falling back to the finest level.
        """
        doc = json.loads((VERIFIED / case / "gates_verified.json").read_text())
        reports = self._reports(case)
        for element, report in reports.items():
            gates = {g["id"]: g for g in report["gates"]}
            if report["certified_accuracy"] is None:
                assert gates["accuracy_asymptotic"]["status"] == WITHHELD, (case, element)
            else:
                assert gates["accuracy_asymptotic"]["status"] in (PASS, FAIL), (case, element)
                assert gates["accuracy_asymptotic"]["actual"] == pytest.approx(
                    report["certified_accuracy"], rel=1e-9)

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