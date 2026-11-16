"""GAP-PP-3: the first-yield criterion must be *decidable* before it is judged.

Findings pinned here
--------------------
F1  A linear Navier first-yield pressure may not judge a path integrated with
    geometric nonlinearity / restrained in-plane displacement: membrane action
    delays first yield by a model difference that mesh refinement cannot remove.
    ``first_yield_reference`` states that, and ``plate_capacity_summary`` now
    publishes ``first_yield_error_comparable`` so no consumer can turn the
    offset into an acceptance failure.

F2  A Richardson limit is only interpretable on an asymptotic sequence.  The
    anchored matrix produced a *plateau* (0.2969 -> 0.2066 -> 0.2044, observed
    order 0.026) whose extrapolated error is ~0; ``met_in_mesh_limit = true``
    was published from it (docs/HONEST-FAIL-GAP-PP-3-3GRID-LIMIT.md finding D2).

F3  On the small-deflection free-in-plane configuration the transition offset is
    *larger* (+38.4 % at 8x4) than on the restrained nonlinear one (+29.7 %).
    The plateau therefore is not fully explained by the model difference; the
    measurement is recorded verbatim and the criterion stays withheld.  This
    test exists so the number cannot be quietly "corrected" later.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from tensorfem.plate_patch_plastic_limit import first_yield_reference  # noqa: E402
from gap_pp3_3grid_study import _rules, finalize  # noqa: E402

BUNDLE = os.path.join(ROOT, "docs", "assets", "gap_pp3_3grid_results.json")
EVIDENCE = os.path.join(ROOT, "docs", "assets", "gap_pp3_transition_reference.json")


class TestReferenceComparability:
    def test_small_deflection_free_path_is_comparable(self):
        ref = first_yield_reference(False, False)
        assert ref["comparable"] is True
        assert "reason" not in ref

    def test_restraint_alone_breaks_comparability(self):
        assert first_yield_reference(False, True)["comparable"] is False
        assert first_yield_reference(True, False)["comparable"] is False
        assert "membrane" in first_yield_reference(True, True)["reason"]

    def test_flag_is_boolean_not_truthy(self):
        """A dict/number would let ``and`` short-circuit silently."""
        for geo in (False, True):
            for rs in (False, True):
                assert isinstance(first_yield_reference(geo, rs)["comparable"], bool)


class TestAsymptoticityGate:
    def test_plateau_is_not_extrapolated_into_a_pass(self):
        """errors: the anchored plateau -> no order, no limit, no mesh-limit claim."""
        rules = _rules([0.29695118264780457, 0.20664238310868205, 0.20443778872102603],
                       [(8, 4), (16, 8), (24, 12)])
        assert rules["observed_order"] < 0.2
        errors = rules["errors"]
        decreasing = all(b < a for a, b in zip(errors, errors[1:]))
        order = rules["observed_order"]
        assert not (0.2 <= order <= 4.0 and decreasing)

    def test_exact_power_law_is_admissible(self):
        rules = _rules([0.25 ** 1.5, 0.125 ** 1.5, (1 / 12) ** 1.5],
                       [(8, 4), (16, 8), (24, 12)])
        assert 0.2 <= rules["observed_order"] <= 4.0
        assert abs(rules["extrapolated_error"]) < 1e-9

    def test_finalize_withholds_a_flat_sequence(self):
        merged = {"analytic": {"yield_line_upper_pressure": 2.0528e6}, "meshes": {}}
        for (nx, ny), err in zip([(8, 4), (16, 8), (24, 12)],
                                 [0.2969, 0.2066, 0.2044]):
            merged["meshes"][f"{nx}x{ny}"] = {
                "nx": nx, "ny": ny, "elements": nx * ny,
                "first_yield_error": err, "first_yield_comparable": False,
                "limit_load_measured": None, "limit_load_error_vs_ub": None,
                "permanent_deflection": None, "permanent_set_valid": False}
        finalize(merged)
        block = merged["criteria"]["transition_lt_5pct"]
        assert block["order_admissible"] is False
        assert block["met_in_mesh_limit"] is False
        assert block["decidable"] is False
        conv = merged["criteria"]["raw_error_convergence"]
        assert "extrapolated_error" not in conv          # not published at all
        assert abs(conv["extrapolated_error_inadmissible"]) < 0.05  # the trap number


class TestCommittedEvidence:
    def test_bundle_withholds_the_transition_criterion(self):
        bundle = json.load(open(BUNDLE))
        block = bundle["criteria"]["transition_lt_5pct"]
        assert block["decidable"] is False
        assert block["met_in_mesh_limit"] is False
        assert block["met_at_finest_mesh"] is False
        assert bundle["case"]["configuration"]["geometric"] is True

    def test_measured_reference_offsets_are_recorded(self):
        """The refutation of 'the plateau is only the membrane model difference'."""
        ev = json.load(open(EVIDENCE))["measured"]
        assert ev["8x4"]["linear-free"]["first_yield_error"] > 0.30
        assert ev["8x4"]["nonlinear-fixed"]["first_yield_error"] > 0.25
        # membrane action moves the transition towards the reference, it does not
        # explain the whole offset
        assert (ev["8x4"]["linear-free"]["first_yield_error"]
                > ev["8x4"]["nonlinear-fixed"]["first_yield_error"])
        assert ev["8x4"]["linear-free"]["reference_comparable"] is True
        assert ev["8x4"]["nonlinear-fixed"]["reference_comparable"] is False
