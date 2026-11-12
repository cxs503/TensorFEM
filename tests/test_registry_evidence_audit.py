"""Fail-closed gates on the registry evidence-class audit.

These tests exist because "the registry passes 69/69 checks below 3 %" is an
aggregate that mixes three very different kinds of evidence.  The gate makes an
unclassified (or silently reclassified) check a hard error, keeps the number of
genuinely discretisation-verified checks from regressing, and proves that the
audit is not vacuous by feeding it a deliberately mislabelled report.
"""
from __future__ import annotations

import copy

import pytest

from tensorfem.benchmark_registry import verification_report
from tensorfem.registry_evidence_audit import (
    CLASS_ORDER,
    EVIDENCE_CLASSES,
    THIN_MARGIN_THRESHOLD,
    audit_evidence_classes,
    classification_basis,
    spheroid_equator_displacement_closed_form,
)

#: Measured on the merged tree (2026-11-12).  Deliberately a lower bound: adding
#: discretisation-verified checks is allowed, silently losing them is not.
MIN_DISCRETIZED = 40


@pytest.fixture(scope="module")
def report():
    return verification_report()


@pytest.fixture(scope="module")
def audit(report):
    return audit_evidence_classes(report)


def test_every_registered_check_is_classified(audit):
    assert audit["unclassified_ids"] == []
    assert audit["stale_ids"] == []
    assert audit["classified"] == audit["total_checks"]


def test_classes_are_in_the_declared_enumeration():
    assert set(CLASS_ORDER) == {"discretized", "trial_space", "closed_form", "identity"}
    for check_id, (cls, basis) in EVIDENCE_CLASSES.items():
        assert cls in CLASS_ORDER, check_id
        assert basis.strip(), check_id
    with pytest.raises(KeyError):
        classification_basis("no.such.check")


def test_discretized_coverage_does_not_regress(audit):
    assert audit["counts"]["discretized"] >= MIN_DISCRETIZED
    assert audit["discretized_share"] > 0.5


def test_algebra_only_checks_are_never_counted_as_fe_evidence(audit):
    """Every sub-1e-9 check must be algebra/identity, not a claimed FE solve."""
    for item in audit["thin_margin"]:
        assert item["class"] in CLASS_ORDER
    for check_id in audit["algebra_only_ids"]:
        assert classification_basis(check_id)[0] in ("closed_form", "identity")


def test_thin_margins_are_reported_and_bounded(audit):
    assert audit["thin_margin"], "expected at least one thin-margin check"
    worst = audit["thin_margin"][0]
    assert worst["margin"] == max(item["margin"] for item in audit["thin_margin"])
    for item in audit["thin_margin"]:
        assert item["margin"] >= THIN_MARGIN_THRESHOLD
        assert item["margin"] < 1.0, f"{item['id']} is outside tolerance"


def test_gate_is_not_vacuous(report):
    """A mislabelled / unknown check must be flagged, not silently accepted."""
    doctored = copy.deepcopy(report)
    doctored["results"].append(dict(report["results"][0], id="synthetic.unclassified"))
    flagged = audit_evidence_classes(doctored)
    assert flagged["unclassified_ids"] == ["synthetic.unclassified"]

    dropped = copy.deepcopy(report)
    dropped["results"] = [r for r in dropped["results"] if r["id"] != "truss.bar"]
    assert "truss.bar" in audit_evidence_classes(dropped)["stale_ids"]


def test_spheroid_reference_matches_independent_membrane_closed_form(report):
    """The Rayleigh-Ritz reference is re-derived from membrane theory alone."""
    row = next(r for r in report["results"] if r["id"] == "shell.spheroid.displacement_equator")
    case = dict(a=1.0, b=0.5, thickness=0.01, young=1.0, poisson=0.3, pressure=1.0)
    oracle = spheroid_equator_displacement_closed_form(**case)
    assert abs(row["reference"] - oracle) / abs(oracle) < 1e-9
    # sphere limit: a = b reduces to the classical w = p R^2 (1 - nu) / (2 E t)
    sphere = spheroid_equator_displacement_closed_form(
        a=2.0, b=2.0, thickness=0.05, young=7.0e10, poisson=0.33, pressure=1.0e5
    )
    exact = 1.0e5 * 2.0**2 * (1.0 - 0.33) / (2.0 * 7.0e10 * 0.05)
    assert abs(sphere - exact) / abs(exact) < 1e-12
    # the FE value is 2.31 % from the reference: the margin the registry spends
    assert 0.02 < abs(row["computed"] - oracle) / abs(oracle) < 0.03


def test_oracle_detects_a_doctored_reference():
    case = dict(a=1.0, b=0.5, thickness=0.01, young=1.0, poisson=0.3, pressure=1.0)
    oracle = spheroid_equator_displacement_closed_form(**case)
    assert abs(1.01 * oracle - oracle) / abs(oracle) > 1e-9