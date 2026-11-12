"""Fail-closed gate over the ``benchmarks/verified/*`` certificate registry.

``benchmarks/verified/test_verification_ladder.py`` verifies a handful of
anchors and **skips** the rest when their artifacts are absent, so a registry
that loses its evidence still shows a green suite.  These tests close that hole
from the outside: every anchor is audited from the artifacts it ships, the
verdicts are recomputed from ``value``/``threshold``, and the set of anchors
that cannot be certified must equal a shrink-only debt registry.

Design contract
---------------
* A new anchor without a self-consistent ``gates_verified.json`` fails here.
* A repaired anchor must be removed from ``anchor_certificate_debt.json``
  (the ``stale`` check), so the debt number cannot drift upwards unnoticed.
* The audit never re-runs a solver, so these tests stay fast (< 2 s) and can run
  on every commit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "verified_anchor_audit.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("verified_anchor_audit", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # ``@dataclass`` resolves annotations through ``sys.modules``, so the module
    # must be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    assert TOOL.is_file(), "the certificate auditor is missing"
    return _load_tool()


@pytest.fixture(scope="module")
def audit(tool):
    return tool.audit_registry()


@pytest.fixture(scope="module")
def diff(tool, audit):
    debt_path = tool.DEBT_JSON
    assert debt_path.is_file(), "the debt registry is missing; run tools/verified_anchor_audit.py --write"
    return tool.debt_diff(audit, json.loads(debt_path.read_text()))


# --------------------------------------------------------------------------- #
# registry-level invariants
# --------------------------------------------------------------------------- #
def test_registry_is_discovered(tool, audit):
    """Every anchor directory is audited (guards a silent discovery regression)."""
    directories = tool.discover()
    assert len(directories) >= 30
    assert audit["anchors"] == len(directories)
    assert audit["gates_total"] > 150


def test_every_gate_verdict_is_recomputed_not_trusted(audit):
    """The audit must recompute almost every verdict, not echo the declared one."""
    assert audit["gates_recomputed"] >= 0.75 * audit["gates_total"]


def test_no_unregistered_certificate_defect(diff):
    """New debt is rejected: an anchor may not be added without a certificate."""
    assert diff["unregistered"] == [], (
        "anchors with certificate defects that the debt registry does not list: "
        f"{diff['unregistered']}"
    )
    assert diff["defect_drift"] == [], (
        "anchors whose defect list changed; regenerate the registry: " f"{diff['defect_drift']}"
    )


def test_debt_registry_is_stale_free(diff):
    """A repaired anchor must be deleted from the debt registry, not tolerated."""
    assert diff["stale"] == [], (
        "anchors are now certified but still listed as debt: " f"{diff['stale']}"
    )


def test_no_contradiction_under_declared_operator(audit):
    """A declared gate verdict must follow from its own value/threshold.

    This is the anti-fabrication invariant: a gate written with an explicit
    ``operator`` cannot claim PASS while its own numbers fail it.
    """
    offenders = []
    for record in audit["records"]:
        for gate in record["gate_detail"]:
            if gate["status"] == "contradicted" and gate["operator_source"] == "declared":
                offenders.append((record["anchor"], gate["id"], gate["value"], gate["threshold"]))
    assert offenders == []


def test_no_non_finite_gate_value_is_unregistered(audit, tool):
    """A NaN/inf ``value`` can never grade a gate; it must be booked as debt."""
    registered = json.loads(tool.DEBT_JSON.read_text())["anchors"]
    seen = 0
    for record in audit["records"]:
        for gate in record["gate_detail"]:
            if "E_GATE_VALUE_NOT_FINITE" in gate["defects"]:
                seen += 1
                assert record["anchor"] in registered
    assert seen >= 1, "the registry is expected to still carry at least one non-finite gate"


def test_certified_anchors_carry_tracked_evidence(audit):
    """Certification implies the declared artifacts exist and are committed."""
    for record in audit["records"]:
        if record["certified"]:
            assert record["has_gates"] and record["has_result"]
            assert "E_ARTIFACT_MISSING" not in record["defects"]
            assert "E_ARTIFACT_UNTRACKED" not in record["defects"]


# --------------------------------------------------------------------------- #
# normalisation contract (unit level, synthetic gates)
# --------------------------------------------------------------------------- #
def test_declared_operator_contradiction_is_detected(tool):
    gate = tool.audit_gate({"id": "g", "value": 0.2, "threshold": 0.1, "operator": "<=", "passed": True}, 0)
    assert gate.recomputed_passed is False
    assert gate.status == "contradicted"
    assert "E_GATE_CONTRADICTION" in gate.defects


def test_honest_fail_is_accepted(tool):
    gate = tool.audit_gate({"id": "g", "value": 0.2, "threshold": 0.1, "operator": "<=", "passed": False}, 0)
    assert gate.status == "agrees"
    assert gate.defects == []


def test_non_finite_value_is_not_a_measurement(tool):
    gate = tool.audit_gate({"id": "g", "value": float("nan"), "threshold": 1.0, "operator": "<=", "passed": True}, 0)
    assert "E_GATE_VALUE_NOT_FINITE" in gate.defects


def test_pass_without_criterion_is_rejected(tool):
    gate = tool.audit_gate({"id": "g", "value": 3.0, "threshold": None, "passed": True}, 0)
    assert "E_PASS_WITHOUT_CRITERION" in gate.defects


def test_conservative_fail_without_criterion_is_tolerated(tool):
    gate = tool.audit_gate({"id": "g", "value": None, "threshold": None, "passed": False}, 0)
    assert gate.defects == []
    assert "W_UNRECOMPUTABLE_FAIL" in gate.warnings


def test_prose_operator_is_inferred_and_flagged(tool):
    gate = tool.audit_gate({"id": "g", "quantity": "rel err < 3% at facets=8", "value": 0.001, "passed": True}, 0)
    assert gate.operator_source == "inferred"
    assert gate.recomputed_passed is True
    assert {"W_OPERATOR_INFERRED", "W_THRESHOLD_FROM_PROSE"} <= set(gate.warnings)


def test_trend_statement_is_not_read_as_a_bound(tool):
    """``e(4) > e(8) > e(32)`` carries no numeric threshold; it must not be invented."""
    gate = tool.audit_gate({"id": "g", "quantity": "e(4) > e(8) > e(32)", "passed": True}, 0)
    assert gate.threshold is None
    assert gate.recomputed_passed is None


def test_dict_shaped_gate_map_is_supported(tool):
    gates = tool._iter_gates({"G1": {"value": 0.0, "threshold": 1.0, "passed": True}})
    assert gates and gates[0][1] == "G1"


def test_audit_is_read_only_and_complete(tool):
    """The audit must not mutate the artifacts it inspects."""
    before = {path: path.stat().st_mtime_ns for path in tool.VERIFIED_DIR.glob("*/gates_verified.json")}
    tool.audit_registry()
    after = {path: path.stat().st_mtime_ns for path in before}
    assert before == after
