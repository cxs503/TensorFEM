"""Evidence-chain guard for the CASE-FEM-HERTZ-CONTACT anchor.

The anchor is graded on two scalars (contact radius ``a`` and indentation
``delta`` from an ALM/load-controlled FE model).  Both are *estimates* extracted
from a computed field, so the certificate is only as good as the estimators --
and an estimator defect does not show up as a red gate, it shows up as a *good
looking* number for the wrong reason.

These tests make three properties machine-checkable, with no solver run:

1. every certificate gate whose value is a function of ``result.json`` is
   re-derived by ``tools/hertz_anchor_recompute.py`` (arithmetic independent of
   the driver) and must agree, and no gate may contradict its own
   ``value <op> threshold``;
2. no graded contact radius may be *degenerate*, i.e. equal to a node radius
   (that is what happened at a/R = 0.10, h/a = 0.0075: the primary estimate was
   0.9825 = the outermost member node, -1.75%, while the two form-assisted
   estimators said -0.05% / +1.5% -- a mesh-tied number graded as a measurement);
3. the case-pool acceptance (<3% on ``a`` and ``delta``, load prescribed) holds
   for both load levels, read from the artifact rather than from prose.

See docs/AUDIT-m17149-hertz-anchor-evidence.md.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ANCHOR = os.path.join(ROOT, "benchmarks", "verified",
                      "fem_hertz_contact_loadcontrol")


@pytest.fixture(scope="module")
def audit():
    import hertz_anchor_recompute as h

    return h.recompute(ANCHOR)


def _codes(audit, code):
    return [f for f in audit["findings"] if f["code"] == code]


def test_recompute_tool_sees_the_certificate(audit):
    assert audit["runs_checked"], "no runs found: the dump is missing"
    assert audit["gates_total"] >= 20
    # a certificate whose values cannot be re-derived at all would be prose
    assert audit["gates_value_recomputed"] >= 12


def test_no_gate_contradicts_its_numbers(audit):
    bad = _codes(audit, "VERDICT_CONTRADICTION") + _codes(audit, "GATE_VALUE_MISMATCH")
    assert bad == [], f"certificate contradicts its own dump: {bad}"


def test_oracle_and_error_columns_reproduce(audit):
    assert _codes(audit, "ORACLE_MISMATCH") == []
    assert _codes(audit, "ERROR_PCT_MISMATCH") == []


def test_no_degenerate_contact_radius(audit):
    """Regression guard for finding F1 (spurious round-off edge node)."""
    bad = _codes(audit, "DEGENERATE")
    assert bad == [], (
        "contact radius coincides with a node radius -- the crossing landed on "
        f"the node and the graded value is mesh-tied: {bad}")


def test_estimators_agree_on_the_graded_level(audit):
    assert _codes(audit, "ESTIMATOR_SCATTER") == []


def test_case_pool_acceptance_from_the_artifact():
    res = json.load(open(os.path.join(ANCHOR, "result.json")))
    graded = [r for r in res["runs"].values() if r["h_over_a"] <= 0.0075]
    assert {r["a_over_R"] for r in graded} >= {0.05, 0.10}
    for row in graded:
        assert abs(row["error_pct"]["contact_radius_linear"]) < 3.0, row["key"]
        assert abs(row["error_pct"]["indentation"]) < 3.0, row["key"]
    # the constraint problem must be *solved*, not regularised away
    for row in graded:
        assert row["penetration_over_delta"] <= 1e-5
        assert row["load_residual"] <= 1e-9


def test_membership_threshold_is_physical():
    """The member-set cut must be a pressure ratio, not a multiplier ratio."""
    src = open(os.path.join(ROOT, "src", "tensorfem",
                            "hertz_halfspace_loadcontrol.py")).read()
    assert "membership_ratio = 1e-6" in src
    assert "1e-10 * max(float(np.max(lam))" not in src
