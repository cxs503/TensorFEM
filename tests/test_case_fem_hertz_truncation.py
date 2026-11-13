"""CASE-FEM-HERTZ-CONTACT: truncation-error attribution and certificate reproduction.

The load-controlled Hertz anchor reports an indentation error that *saturates*
near -1.0 % under mesh refinement.  These tests fix the two claims the m17031
attribution rests on, and they are all cheap:

* the measured error follows ``err(D) = e_inf + c / D`` in the domain depth ``D``
  (in units of ``a``) to within 0.01 pp, and the law predicts hold-out depths;
* a *linear static* solve with the exact Hertz tractions — no contact algorithm
  at all — reproduces the same O(1/D) coefficient, which is what proves the
  plateau is a boundary/discretisation effect and not an ALM artefact;
* the committed raw dump plus the committed gate manifest are mutually
  consistent: re-grading the dump offline reproduces every gate value and every
  pass/fail flag in ``gates_verified.json`` (zero compute, fresh checkout).

The heavy sweep itself lives in ``benchmarks/verified/fem_hertz_contact_loadcontrol``
and is not re-run here.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHOR = os.path.join(HERE, "..", "benchmarks", "verified", "fem_hertz_contact_loadcontrol")
RESULT = os.path.join(ANCHOR, "result.json")
GATES = os.path.join(ANCHOR, "gates_verified.json")
DUMP = os.path.join(ANCHOR, "results_dump.json")
LAW_JSON = os.path.join(ANCHOR, "truncation_law.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(RESULT) and os.path.exists(GATES)),
    reason="Hertz anchor certificate is not in this checkout",
)


def _load_run_module():
    """Import ``run.py`` from the anchor directory (it is a script, not a package)."""
    path = os.path.join(ANCHOR, "run.py")
    spec = importlib.util.spec_from_file_location("fem_hertz_contact_loadcontrol_run", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "verified"))
    spec.loader.exec_module(module)
    return module


def test_inverse_depth_law_recovers_its_own_parameters():
    """The 1/D fit must return the generating parameters on synthetic data."""
    run = _load_run_module()
    e_inf, c = -0.383, -60.4
    depths = (24.0, 48.0, 96.0, 144.0, 192.0)
    points = [(d, e_inf + c / d) for d in depths]
    law = run.fit_inverse_depth_law(points, holdout_depths=(144.0, 192.0))
    assert law["e_inf_pct"] == pytest.approx(e_inf, abs=1e-9)
    assert law["c_pct_depth"] == pytest.approx(c, abs=1e-8)
    assert law["max_resid_pp"] < 1e-9
    # the hold-out fit (three depths) must extrapolate the other two
    assert law["holdout"]["max_abs_deviation_pp"] < 1e-9


def test_inverse_depth_law_flags_a_non_inverse_depth_signal():
    """A 1/D^2 signal must not be accepted as a 1/D law (falsifiability guard)."""
    run = _load_run_module()
    depths = (24.0, 48.0, 96.0, 144.0, 192.0)
    points = [(d, -60.4 / d**2) for d in depths]
    law = run.fit_inverse_depth_law(points, holdout_depths=(144.0, 192.0))
    assert law["max_resid_pp"] > 0.01  # above the certificate's own threshold


def test_committed_truncation_law_matches_the_documented_thresholds():
    """The committed law is inside the declared gate and holds out of sample."""
    if not os.path.exists(LAW_JSON):
        pytest.skip("truncation_law.json not present")
    with open(LAW_JSON) as fh:
        law = json.load(fh)
    with open(RESULT) as fh:
        result = json.load(fh)
    crit = result["criteria"]
    contact = result["truncation_law"]
    assert contact["max_resid_pp"] <= crit["truncation_law_residual_pp"]
    assert contact["holdout"]["max_abs_deviation_pp"] <= crit["truncation_law_holdout_pp"]
    assert abs(contact["e_inf_pct"]) <= crit["extrapolated_indentation_pct"]
    # independent code path: static solve must find the same O(1/D) coefficient
    static = law["comparison"]["static_truncation_coeff_pct_depth"]
    assert static == pytest.approx(contact["c_pct_depth"], abs=0.5)
    assert law["depth_law"]["max_resid_pp"] <= crit["truncation_law_residual_pp"]


def test_static_compliance_reproduces_the_boundary_term():
    """Exact Hertz tractions, one linear solve: the 1/D term reappears (~30 s)."""
    spec = importlib.util.spec_from_file_location(
        "fem_hertz_truncation_law", os.path.join(ANCHOR, "truncation_law.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shallow = module.static_compliance_case(0.03, 48.0)
    deep = module.static_compliance_case(0.03, 96.0)
    # err(D) = e_inf + c/D  =>  err(48) - err(96) = c/96
    c_measured = (shallow["indentation_error_pct"] - deep["indentation_error_pct"]) * 96.0
    assert c_measured == pytest.approx(-60.4, abs=1.5)
    # the FE body is stiffer than the half-space at both depths (deficit, not overshoot)
    assert shallow["indentation_error_pct"] < 0.0 and deep["indentation_error_pct"] < 0.0
    assert shallow["indentation_error_pct"] < deep["indentation_error_pct"]
    # the prescribed traction integrates to the required load; at h/a = 0.03 the
    # surface ring areas overestimate the patch edge, so allow 0.5 %
    assert deep["prescribed_force_over_P"] == pytest.approx(1.0, abs=5e-3)


def test_committed_dump_regrades_to_the_committed_certificate():
    """Offline re-grading of the raw dump must reproduce gates_verified.json."""
    if not os.path.exists(DUMP):
        pytest.skip("raw dump not committed")
    run = _load_run_module()
    with open(DUMP) as fh:
        dump = json.load(fh)
    with open(GATES) as fh:
        manifest = json.load(fh)
    gates = {g.id: g for g in run.build_gates(dump)}
    assert set(gates) == {g["id"] for g in manifest["gates"]}
    for row in manifest["gates"]:
        g = gates[row["id"]]
        assert float(row["value"]) == pytest.approx(float(g.value), rel=1e-12, abs=1e-15)
        assert bool(row["passed"]) == bool(g.passed), f"gate {row['id']} flips on re-grade"
    # ``passed`` is the boolean "all gates pass" flag of the manifest schema
    assert bool(manifest["passed"]) is all(g.passed for g in gates.values())
    assert bool(manifest["failed_gates"] == []) is bool(manifest["passed"])
    # the raw artifact declared by the certificate must exist in the checkout
    with open(RESULT) as fh:
        declared = json.load(fh)["artifacts"]["raw_dump"]
    assert os.path.exists(os.path.join(HERE, "..", declared)), declared


def test_certificate_reports_no_discretisation_heroics():
    """The extrapolated accuracy statement must stay inside the 3% pool mandate."""
    with open(RESULT) as fh:
        result = json.load(fh)
    law = result["truncation_law"]
    assert abs(law["e_inf_pct"]) < result["criteria"]["contact_radius_pct"]
    # the accepted model (D = 96a) loses 63 % of its indentation error to truncation
    trunc_d96 = law["c_pct_depth"] / 96.0
    measured = [r for r in result["truncation"] if r["domain_depth_over_a"] == 96.0][0]
    assert trunc_d96 / measured["error_pct"]["indentation"] == pytest.approx(0.62, abs=0.05)