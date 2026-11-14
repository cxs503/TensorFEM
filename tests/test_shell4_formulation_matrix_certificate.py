"""Certificate gates for the ``fem_shell4_formulation_matrix`` anchor.

The formulation study (``benchmarks/verified/fem_shell4_formulation_matrix/study.py``)
existed for several missions with **no** ``gates_verified.json``, so the
fail-closed registry audit carried it in
``benchmarks/verified/anchor_certificate_debt.json`` (``E_NO_GATES``): a green
test suite could not tell whether its numbers were still true.  These tests pin
the repaired certificate:

* the anchor is certified by the registry auditor and absent from the debt list;
* every gate recomputes ``value <operator> threshold -> passed`` from its own
  numbers (so a hand-edited verdict fails);
* the gate values agree with ``result.json`` (so a hand-edited dump fails);
* the four *negative controls* really assert a defect, i.e. flipping the
  inequality would make them red — a negative control that passes under both
  polarities is not a check at all.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = REPO_ROOT / "benchmarks" / "verified" / "fem_shell4_formulation_matrix"
GATES_PATH = ANCHOR_DIR / "gates_verified.json"
RESULTS_PATH = ANCHOR_DIR / "result.json"
EMITTER = ANCHOR_DIR / "make_gates.py"
AUDITOR = REPO_ROOT / "tools" / "verified_anchor_audit.py"
ANCHOR = "fem_shell4_formulation_matrix"
NEGATIVE_CONTROLS = (
    "A.sri_spurious_present [negative control]",
    "B.assembled_sri_spurious [negative control]",
    "C.scordelis_lo.order_sri [negative control]",
    "C.scordelis_lo.sri_degradation [negative control]",
)


def _load(path: Path, name: str):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gates() -> dict:
    assert GATES_PATH.is_file(), "the anchor must publish gates_verified.json"
    return json.loads(GATES_PATH.read_text())


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def test_envelope_schema_is_canonical(gates):
    assert gates["schema"] == "tensorfem.verified-gates.v1"
    assert gates["anchor"] == ANCHOR
    assert gates["passed"] is True
    assert gates["failed_gates"] == []
    assert gates["gates"], "an empty certificate is not a certificate"
    assert gates["limitations"], "limitations must be stated, not omitted"


def test_every_gate_carries_a_numeric_criterion(gates):
    for gate in gates["gates"]:
        assert gate["operator"] in ("<=", ">="), gate["id"]
        assert isinstance(gate["value"], float), gate["id"]
        assert isinstance(gate["threshold"], float), gate["id"]
        assert gate["value"] == pytest.approx(gate["value"], rel=0, abs=0)  # finite
        assert gate["quantity"].strip(), gate["id"]
        assert gate["reference"].strip(), gate["id"]


def test_declared_verdicts_recompute(gates):
    """A gate's verdict must follow from its own numbers."""
    for gate in gates["gates"]:
        expected = (
            gate["value"] <= gate["threshold"]
            if gate["operator"] == "<="
            else gate["value"] >= gate["threshold"]
        )
        assert gate["passed"] is expected, f"{gate['id']}: verdict does not recompute"


def test_gate_values_track_results_json(gates, results):
    """Re-derive the headline quantities from ``result.json`` independently."""
    by_id = {g["id"]: g for g in gates["gates"]}

    rank = results["rank_accounting"]
    assert by_id["A.rigid_body_zero_energy"]["value"] == pytest.approx(
        max(r["rigid_residual"] for r in rank), rel=1e-12, abs=0
    )
    assert by_id["A.mitc_spurious_free"]["value"] == pytest.approx(
        max(r["spurious_modes"] for r in rank if r["plate_shear"] == "mitc")
    )
    assert by_id["A.sri_spurious_present [negative control]"]["value"] == pytest.approx(
        min(r["spurious_modes"] for r in rank if r["plate_shear"] == "sri")
    )

    census = results["mesh_census"]
    assert by_id["B.assembled_null_dim"]["value"] == pytest.approx(
        max(abs(r["null_dim"] - r["expected_rigid_dim"]) for r in census if r["plate_shear"] == "mitc")
    )

    sl = results["scordelis_lo"]
    finest = {
        (r["formulation"], r["plate_shear"]): r
        for r in sl
        if r["n"] == max(q["n"] for q in sl if q["formulation"] == r["formulation"]
                         and q["plate_shear"] == r["plate_shear"])
    }
    for formulation in ("shell4", "shell4_mitc"):
        gate = by_id[f"C.scordelis_lo.{formulation}+mitc"]
        assert gate["value"] == pytest.approx(
            abs(finest[(formulation, "mitc")]["relative_error"]), rel=1e-9, abs=0
        )

    mem = results["membrane"]
    fin = [r for r in mem if r["nt"] == max(q["nt"] for q in mem if
                                           (q["formulation"], q["plate_shear"]) ==
                                           (r["formulation"], r["plate_shear"]))]
    assert by_id["D.membrane_finest_error"]["value"] == pytest.approx(
        max(abs(r["radial_error"]) for r in fin), rel=1e-9, abs=0
    )
    nu = results["element"]["poisson"]
    assert by_id["D.poisson_coupling"]["value"] == pytest.approx(
        max(abs(abs(r["axial_displacement"] / r["radial_displacement"]) - nu / 2.0) for r in fin),
        rel=1e-9, abs=0,
    )


def test_negative_controls_are_one_sided(gates):
    """Each negative control must be sensitive to its polarity."""
    by_id = {g["id"]: g for g in gates["gates"]}
    for gid in NEGATIVE_CONTROLS:
        gate = by_id[gid]
        # the reversed inequality must be violated, otherwise the gate passes under
        # both polarities and therefore checks nothing
        reversed_verdict = (gate["value"] <= gate["threshold"]) if gate["operator"] == ">=" else (
            gate["value"] >= gate["threshold"]
        )
        assert reversed_verdict is False, f"{gid} also passes with the inequality reversed"
        assert gate["kind"] == "negative_control"


def test_sri_deficiency_is_recorded_not_hidden(gates, results):
    """The SRI ladder must be reported as diverging (negative order)."""
    orders = results["orders"]
    assert orders["scordelis_lo::shell4+sri"] < 0.0
    assert orders["scordelis_lo::shell4+mitc"] > 1.0
    sri_error = min(
        abs(r["relative_error"]) for r in results["scordelis_lo"]
        if r["plate_shear"] == "sri"
    )
    assert sri_error > 0.10, "SRI would no longer be materially wrong; re-baseline the anchor"


def test_anchor_is_certified_and_out_of_debt():
    auditor = _load(AUDITOR, "verified_anchor_audit_for_shell4_test")
    audit = auditor.audit_registry()
    record = next((r for r in audit["records"] if r["anchor"] == ANCHOR), None)
    assert record is not None, "the auditor does not see the anchor"
    assert record["certified"] is True, f"defects: {record['defects']}"
    assert record["gates"] == len(json.loads(GATES_PATH.read_text())["gates"])
    debt = json.loads((REPO_ROOT / "benchmarks" / "verified" / "anchor_certificate_debt.json").read_text())
    assert ANCHOR not in debt["anchors"], "a repaired anchor must be deleted from the debt registry"


def test_declared_artifacts_exist():
    envelope = json.loads(GATES_PATH.read_text())
    for name, rel in envelope["artifacts"].items():
        if name.endswith("sha256"):
            continue
        assert (REPO_ROOT / rel).is_file(), f"named artifact {name} missing: {rel}"


def test_emitter_reproduces_the_committed_gates(gates):
    """Re-emitting from the committed dump must reproduce gate ids and values."""
    emitter = _load(EMITTER, "shell4_make_gates_test")
    rebuilt = emitter.build_gates(json.loads(RESULTS_PATH.read_text()))
    committed = {g["id"]: g for g in gates["gates"]}
    assert {g["id"] for g in rebuilt} == set(committed)
    for gate in rebuilt:
        assert gate["value"] == pytest.approx(committed[gate["id"]]["value"], rel=1e-12, abs=0)
        assert gate["threshold"] == pytest.approx(committed[gate["id"]]["threshold"])
        assert gate["passed"] is committed[gate["id"]]["passed"]