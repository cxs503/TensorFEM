"""Gates for the certification-margin audit of the MITC4 cylindrical-shell
buckling claim (:mod:`scripts.audit_cyl_buckling_certification_margin`).

The claim under audit is ``benchmarks/cylindrical_shell_buckling_mitc_evidence.json``
``audit.certified = True`` with reason "finest affordable mesh (nz=30, nt=32) sits
within 3 % of the Donnell finite-length reference".  A certification that rests on
one reading must at minimum survive the question *is the family converging to the
reference or merely crossing it?*  These gates recompute that answer from the
stored raw eigen records only:

* the axial family is monotone with no sign change (a crossing would make the
  single passing mesh meaningless);
* the three-parameter fit ``e(h) = e_inf + C h^p`` puts the **converged limit**
  inside the 3 % gate (this is the claim, stated correctly);
* the uncured full-integration Shell4 family stays outside the gate by a wide
  margin, so the gate is not vacuous;
* the source gate field ``within_3pct_any`` is an ``any(...)`` over the family,
  not the finest mesh -- recorded so nobody reads it as "the finest mesh passes".

No number is hand-copied: everything is recomputed from the evidence JSON.
"""
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "cylindrical_shell_buckling_mitc_evidence.json"
SCRIPT = ROOT / "scripts" / "audit_cyl_buckling_certification_margin.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cyl_cert_margin", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    module = _load_module()
    records = json.loads(EVIDENCE.read_text())["eigen"]
    families = {tag: module.analyse(records, tag,
                                   "ntheta" if tag == "mitc-axsym-nt" else "nz")
                for tag in module.FAMILIES}
    return module, families


def test_source_evidence_still_claims_certification():
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["audit"]["certified"] is True
    # The field the reason text leans on is an `any` over the family.
    assert "within_3pct_any" in evidence["audit"]["mitc_axsym"]


def test_axial_family_is_monotone_and_does_not_change_sign(audit):
    _, families = audit
    fam = families["mitc-axsym"]
    errors = [level["error"] for level in fam["levels"]]
    assert errors == sorted(errors, reverse=True), errors
    assert all(e > 0.0 for e in errors), "a sign change would void a single passing mesh"
    assert fam["monotone"] is True


def test_fitted_limit_not_the_finest_reading_is_the_certification(audit):
    _, families = audit
    fam = families["mitc-axsym"]
    # The finest raw reading passes the 3 % gate ...
    assert abs(fam["finest_error"]) < 0.03
    # ... and the family converges to the reference, not past it.
    assert fam["fit_converged"] is True
    assert fam["extrapolated_error"] is not None
    assert abs(fam["extrapolated_error"]) < 0.03
    assert 1.0 < fam["fit_order"] < 3.0
    # Only one level passes in raw readings: the claim cannot be "any mesh".
    assert sum(fam["within_gate"]) == 1


def test_circumferential_family_also_converges_below_the_gate(audit):
    _, families = audit
    nt = families["mitc-axsym-nt"]
    errors = [level["error"] for level in nt["levels"]]
    assert errors == sorted(errors, reverse=True), errors
    assert abs(nt["extrapolated_error"]) < 0.03


def test_gate_is_not_vacuous_uncured_shell4_stays_outside(audit):
    _, families = audit
    shell4 = families["shell4-axsym"]
    mitc = families["mitc-axsym"]
    assert abs(shell4["finest_error"]) > 0.03
    # The assumed-shear cure buys more than a factor of 5 at the same mesh.
    assert abs(shell4["finest_error"]) / abs(mitc["finest_error"]) > 5.0


def test_documented_numbers_match_the_artifacts():
    """The doc must not drift from the machine-produced JSON (anti hand-copy)."""
    doc = ROOT / "docs" / "cylindrical-shell-buckling-certification-margin.md"
    artifact = ROOT / "artifacts" / "cyl_buckling_certification_margin.json"
    if not doc.exists() or not artifact.exists():  # pragma: no cover
        pytest.skip("artifact or document not built in this checkout")
    data = json.loads(artifact.read_text())
    text = doc.read_text()
    assert f"{data['fitted_limit_error'] * 100:.3f}" in text
    assert f"{data['fitted_order']:.3f}" in text
    assert f"{data['shell4_finest_error'] * 100:.2f}" in text