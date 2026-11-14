"""Regression gate for the CASE-FEM-HERTZ-CONTACT independent audit.

The Hertz case-pool anchor publishes ``result.json`` + ``gates_verified.json``
and two prose documents quote those numbers.  ``tools/hertz_case_pool_
independent_audit.py`` re-derives the Hertz oracle from the primary sources and
grades the artifact and the documents against it.  This test pins the three
things that can silently rot:

* the oracle in the artifact still agrees with an independently derived one;
* every published error percentage is still reproducible from the stored FE
  numbers (so editing ``result.json`` cannot leave stale ``error_pct`` behind);
* the documents still quote the artifact (so a table row cannot drift back to a
  pre-fix value, which is exactly what happened with ``-1.750%``).

It does not re-run the solver and it is not a physics test; it is the contract
that the evidence bundle and its prose stay in sync.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "hertz_case_pool_independent_audit",
        ROOT / "tools" / "hertz_case_pool_independent_audit.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


@pytest.fixture(scope="module")
def report(tool):
    return tool.audit()


def test_oracle_matches_hand_computed_hertz_values(tool):
    """a, delta, p0 for the canonical case, computed by hand in the test."""
    load, radius, young, nu = 1831.5018315018315, 10.0, 100000.0, 0.3
    e_star = young / (1.0 - nu * nu)
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    got = tool.hertz_oracle(load, radius, young, nu)
    assert got["equivalent_modulus"] == pytest.approx(e_star, rel=0.0)
    assert got["contact_radius"] == pytest.approx(a, rel=1e-15)
    assert got["indentation"] == pytest.approx(a * a / radius, rel=1e-15)
    assert got["peak_pressure"] == pytest.approx(3.0 * load / (2.0 * math.pi * a * a), rel=1e-15)


def test_artifact_oracle_is_the_independent_one(report):
    """No gate may grade against a mis-derived reference."""
    hard = [f for f in report["findings"] if f["code"] == "E_ORACLE_MISMATCH"]
    assert hard == [], hard


def test_published_errors_are_reproducible_from_stored_fe_numbers(report):
    """``error_pct`` and ``gate_table`` must agree with a fresh recomputation."""
    drift = [f for f in report["findings"] if f["code"] in ("E_ARTIFACT_DRIFT", "E_GATE_TABLE_DRIFT")]
    assert drift == [], drift


def test_case_pool_acceptance_holds(report):
    """<3 % on a and delta over the whole acceptance ladder."""
    assert report["physics_verdict"]["passed"], report["physics_verdict"]
    assert report["worst_pct"]["contact_radius"] < report["tolerance_pct"]
    assert report["worst_pct"]["indentation"] < report["tolerance_pct"]


def test_documents_quote_the_artifact(report):
    """The prose tables must not drift from the evidence bundle."""
    assert report["docs_verdict"]["passed"], report["docs_verdict"]
    assert report["doc_claims"], "no document claims were parsed -- the parser silently stopped matching"


def test_certificate_gates_are_self_consistent(report):
    assert [f for f in report["findings"] if f["code"] == "E_GATE_SELFCONTRADICTION"] == []


def test_peak_pressure_claim_is_reported_with_its_estimator(report):
    """p0 passes the 5 % gate only under the L2-consistent field; the spread is disclosed."""
    spread = [f for f in report["findings"] if f["code"] == "W_PEAK_ESTIMATOR_SPREAD"]
    assert spread, "the lumped-vs-consistent peak-pressure spread must stay visible"
    assert spread[0]["lumped_max_abs_pct"] > 5.0


def test_committed_audit_artifact_is_current(tool, report):
    """The committed JSON must equal a fresh audit (rerun after editing the bundle)."""
    import json

    path = tool.AUDIT_OUT
    assert path.exists(), "run tools/hertz_case_pool_independent_audit.py --write"
    stored = json.loads(path.read_text())
    for key in ("rows", "findings", "physics_verdict", "docs_verdict", "worst_pct"):
        assert stored[key] == report[key], f"committed audit artifact is stale in {key!r}"

def test_repaired_document_claims_stay_repaired(report):
    """The pre-fix numbers recorded in FIXED_DEFECTS must not reappear.

    ``-1.750%`` (the pre-m17149 estimator value) and the mislabelled ``+0.493%``
    p0 row were published as current results.  Pinning them here means a future
    edit that reverts the table fails this test instead of quietly re-publishing
    a number the artifact contradicts.
    """
    assert report["fixed_defects"], "the repair record disappeared"
    for d in report["fixed_defects"]:
        assert d["status"] == "repaired", d
