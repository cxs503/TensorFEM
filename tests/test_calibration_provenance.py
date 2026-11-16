"""Tests for the calibration-factor provenance layer (JOUR-ICE-STRUCT-CALIB-PROV).

The tests assert *invariants*, not just that functions run:

* fail-closed auditing: an incomplete or unknown provenance never comes out
  ``AUDITABLE``;
* the weakest-link rule: a derived ratio is never more auditable than its weaker
  operand, and uncertainties combine in quadrature;
* ``cf`` reproduces its own definition, or construction fails;
* ``calibrated_total`` refuses an unauditable factor on request and otherwise
  reports the unaudited bias;
* the fleet statistics reproduce the ledger numbers (34.5 % spread, the -16.44 %
  / +12.38 % member error of the shipped constant) from the raw min/max only.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.calibration_provenance import (  # noqa: E402
    AUDITABLE,
    MIN_REGRESSION_SAMPLES,
    PARTIAL,
    UNAUDITABLE,
    CalibrationFactor,
    ProvenanceRecord,
    SourceKind,
    UnauditableCalibration,
    best_single_constant,
    calibrated_total,
    combine_provenance,
    constant_penalty,
    fleet_dispersion,
)

SHA = "a" * 64
SHA2 = "b" * 64


def _code_rule(**kw) -> ProvenanceRecord:
    base = dict(
        quantity="design ice load",
        source_kind=SourceKind.CODE_RULE,
        citation="IACS UR I2 Rev.5, I2.3.2.1 (iii)",
        document_sha256=SHA,
        clause="I2.3.2.1 (iii)",
        validity={"displacement_kt": [5.0, 1000.0]},
    )
    base.update(kw)
    return ProvenanceRecord(**base)


def _judgement(**kw) -> ProvenanceRecord:
    base = dict(
        quantity="declared coefficient",
        source_kind=SourceKind.JUDGEMENT,
        citation="declared normalisation",
        validity={"area_m2": [0.1, 10.0]},
    )
    base.update(kw)
    return ProvenanceRecord(**base)


# --------------------------------------------------------------------------- #
# records and auditing
# --------------------------------------------------------------------------- #
def test_code_rule_complete_is_auditable():
    v = _code_rule().audit()
    assert v.verdict == AUDITABLE and v.auditable and not v.missing


@pytest.mark.parametrize(
    "field, value",
    [("document_sha256", ""), ("document_sha256", "not-a-hash"),
     ("clause", ""), ("citation", "")],
)
def test_code_rule_missing_evidence_is_unauditable(field, value):
    v = _code_rule(**{field: value}).audit()
    assert v.verdict == UNAUDITABLE
    assert v.missing, "the audit must name what is missing"
    assert field in v.missing[0]


def test_code_rule_without_validity_envelope_is_only_partial():
    v = _code_rule(validity={}).audit()
    assert v.verdict == PARTIAL
    assert not v.auditable


def test_judgement_is_never_auditable():
    v = _judgement().audit()
    assert v.verdict == UNAUDITABLE
    assert "judgement" in v.reasons[0]


def test_unknown_source_kind_fails_closed():
    v = _judgement(source_kind="vibes").audit()
    assert v.verdict == UNAUDITABLE
    assert "unknown source_kind" in v.reasons[0]


def test_experiment_requires_declared_uncertainty():
    partial = ProvenanceRecord(
        quantity="trial ratio", source_kind=SourceKind.EXPERIMENT,
        citation="ice trials 2024", validity={"a": 1},
    ).audit()
    assert partial.verdict == UNAUDITABLE
    ok = ProvenanceRecord(
        quantity="trial ratio", source_kind=SourceKind.EXPERIMENT,
        citation="ice trials 2024", relative_uncertainty=0.06,
        uncertainty_basis="six trials, sample sd", validity={"a": 1},
    ).audit()
    assert ok.verdict == AUDITABLE


def test_regression_requires_dataset_estimator_samples_and_uncertainty():
    def verdict(**kw):
        base = dict(
            quantity="cf", source_kind=SourceKind.REGRESSION_DATASET,
            citation="dataset A", dataset_sha256=SHA, estimator="OLS",
            sample_size=MIN_REGRESSION_SAMPLES, relative_uncertainty=0.05,
            validity={"area_m2": [1, 10]},
        )
        base.update(kw)
        return ProvenanceRecord(**base).audit()

    assert verdict().verdict == AUDITABLE
    assert verdict(sample_size=MIN_REGRESSION_SAMPLES - 1).verdict == UNAUDITABLE
    assert verdict(dataset_sha256="").verdict == UNAUDITABLE
    assert verdict(estimator="").verdict == UNAUDITABLE
    assert verdict(relative_uncertainty=None).verdict == UNAUDITABLE


def test_record_hash_is_order_insensitive_and_content_sensitive():
    a = _judgement(validity={"x": [1, 2], "y": [3, 4]})
    b = _judgement(validity={"y": [3, 4], "x": [1, 2]})
    assert a.sha256() == b.sha256()
    assert a.sha256() != _judgement(validity={"x": [1, 3], "y": [3, 4]}).sha256()


def test_negative_uncertainty_is_rejected():
    with pytest.raises(ValueError):
        _judgement(relative_uncertainty=-0.1)


# --------------------------------------------------------------------------- #
# weakest-link combination
# --------------------------------------------------------------------------- #
def test_ratio_of_auditable_and_judgement_is_unauditable():
    derived = combine_provenance(_code_rule(), _judgement(), quantity="ratio")
    assert derived.audit().verdict == UNAUDITABLE
    assert "weakest" in derived.note
    assert _code_rule().citation in derived.citation
    assert _judgement().citation in derived.citation


def test_ratio_of_two_partial_is_partial_not_auditable():
    derived = combine_provenance(
        _code_rule(validity={}), _code_rule(quantity="other", validity={}),
        quantity="ratio",
    )
    assert derived.audit().verdict == PARTIAL


def test_ratio_of_two_auditable_is_auditable_and_uncertainties_add_in_quadrature():
    left = _code_rule().with_uncertainty(0.03, "tolerance stack")
    right = _code_rule(quantity="reference").with_uncertainty(0.04, "trial scatter")
    derived = combine_provenance(left, right, quantity="ratio")
    assert derived.audit().verdict == AUDITABLE
    assert derived.relative_uncertainty == pytest.approx(math.hypot(0.03, 0.04))


def test_combine_provenance_rejects_non_records():
    with pytest.raises(TypeError):
        combine_provenance(_code_rule(), 1.0, quantity="ratio")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# calibration factor
# --------------------------------------------------------------------------- #
def test_factor_must_reproduce_its_own_definition():
    CalibrationFactor(value=0.5, raw_quantity="raw", reference_quantity="ref",
                      provenance=_code_rule(), raw_value=4.0, reference_value=2.0)
    with pytest.raises(ValueError, match="does not reproduce"):
        CalibrationFactor(value=0.6, raw_quantity="raw", reference_quantity="ref",
                          provenance=_code_rule(), raw_value=4.0, reference_value=2.0)


def test_non_positive_factor_is_rejected():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            CalibrationFactor(value=bad, raw_quantity="r", reference_quantity="q",
                              provenance=_judgement())


def test_calibrated_total_carries_bias_and_can_refuse():
    factor = CalibrationFactor(
        value=0.75, raw_quantity="raw", reference_quantity="ref",
        provenance=_judgement(relative_uncertainty=0.2, uncertainty_basis="assumed"),
    )
    with pytest.raises(UnauditableCalibration) as excinfo:
        calibrated_total(100.0, factor, require_auditable=True)
    assert "refusing" in str(excinfo.value)

    out = calibrated_total(100.0, factor)
    assert out.value == pytest.approx(75.0)
    assert not out.auditable
    assert out.unaudited_bias_rel == pytest.approx(0.2)
    assert out.as_dict()["provenance_sha256"] == factor.provenance.sha256()


def test_auditable_factor_reports_zero_unaudited_bias():
    factor = CalibrationFactor(value=2.0, raw_quantity="raw", reference_quantity="ref",
                               provenance=_code_rule())
    out = factor.apply(5.0, require_auditable=True)
    assert out.value == pytest.approx(10.0) and out.auditable
    assert out.unaudited_bias_rel == 0.0


def test_scaling_identity_error_of_total_equals_error_of_factor():
    cf_used, cf_true = 0.753028, 0.901199
    for raw in (0.5, 17.0, 1.0e4):
        assert (raw * cf_used) / (raw * cf_true) - 1.0 == pytest.approx(cf_used / cf_true - 1.0)


# --------------------------------------------------------------------------- #
# fleet statistics (ledger numbers)
# --------------------------------------------------------------------------- #
LEDGER_MIN, LEDGER_MAX, LEDGER_CF = 0.670077, 0.901199, 0.753028


def test_fleet_dispersion_reproduces_the_ledger_spread():
    disp = fleet_dispersion({"min": LEDGER_MIN, "max": LEDGER_MAX})
    assert disp["n"] == 2
    assert disp["dispersion_rel"] == pytest.approx(0.34492, abs=5e-6)
    assert disp["dispersion_rel"] == pytest.approx((LEDGER_MAX - LEDGER_MIN) / LEDGER_MIN)
    assert disp["min"] == LEDGER_MIN and disp["max"] == LEDGER_MAX


def test_constant_penalty_fails_the_15_percent_standard():
    pen = constant_penalty([LEDGER_MIN, LEDGER_MAX], LEDGER_CF, tolerance=0.15)
    # error of the calibrated value relative to the member's own value: cf/v - 1
    assert pen["error_rel_max"] == pytest.approx(LEDGER_CF / LEDGER_MIN - 1.0)
    assert pen["error_rel_min"] == pytest.approx(LEDGER_CF / LEDGER_MAX - 1.0)
    assert pen["worst_abs_error_rel"] == pytest.approx(abs(LEDGER_CF / LEDGER_MAX - 1.0))
    assert pen["worst_abs_error_rel"] > 0.15
    assert pen["verdict"] == "FAIL"


def test_best_single_constant_is_the_harmonic_midpoint():
    out = best_single_constant([LEDGER_MIN, LEDGER_MAX])
    assert out["constant"] == pytest.approx(2.0 / (1.0 / LEDGER_MIN + 1.0 / LEDGER_MAX))
    assert out["minimax_abs_error_rel"] == pytest.approx(
        out["constant"] / LEDGER_MIN - 1.0, rel=1e-12)
    # the floor of any single constant on this fleet spread is just under 15 %
    assert 0.14 < out["minimax_abs_error_rel"] < 0.15
    assert best_single_constant([0.8, 0.8])["uniform"] is True


def test_dispersion_and_penalty_input_validation():
    with pytest.raises(ValueError):
        fleet_dispersion([])
    with pytest.raises(ValueError):
        fleet_dispersion([0.5, 0.0])
    with pytest.raises(ValueError):
        constant_penalty([0.5], 0.0)
    with pytest.raises(ValueError):
        constant_penalty([0.5], 0.6, tolerance=0.0)


# --------------------------------------------------------------------------- #
# driver end-to-end (no figures): the audit numbers must be reproduced
# --------------------------------------------------------------------------- #
def _load_driver():
    path = ROOT / "benchmarks" / "jour_ice_struct_calib_prov" / "run_calib_prov.py"
    spec = importlib.util.spec_from_file_location("run_calib_prov", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driver_reproduces_audit_and_refuses_the_documented_ratio():
    driver = _load_driver()
    data, by_id = driver.load_anchors()
    live = json.loads((ROOT / "benchmarks" / "jour_ice_struct_calib_prov"
                       / "live_state.json").read_text())

    audit = driver.audit_live_state(live)
    assert audit["fleet_dispersion"]["dispersion_rel"] == pytest.approx(0.34492, abs=5e-6)
    assert audit["constant_penalty"]["verdict"] == "FAIL"
    assert audit["scaling_identity"]["identity_holds"] is True
    assert audit["best_constant_meets_tolerance"] is True  # the floor is 14.7 %

    ratio = driver.model_ratio_table(data, by_id)
    assert len(ratio["rows"]) == len(driver.SHIP["displacements_kt"]) * len(driver.ISO_EXPONENTS)
    assert ratio["C_R_invariance"]["dispersion_invariant"] is True
    # the ratio is parameter dependent: its spread exceeds nothing it can hide
    assert ratio["ratio_range"]["max_over_min"] > 1.5
    assert ratio["code_implied_pressure_area_exponent"]["dlnp_dlnA"] > 0.0

    prov = driver.provenance_gate(data, ratio, live)
    assert prov["gate"]["iacs_verdict"]["verdict"] == AUDITABLE
    assert prov["gate"]["iso_verdict"]["verdict"] == UNAUDITABLE
    assert prov["gate"]["derived_verdict"]["verdict"] == UNAUDITABLE
    assert prov["gate"]["refusal"]["raised"] is True
    assert prov["gate"]["refusal"]["exception"] == "UnauditableCalibration"
    stages = {s["stage"]: s["record"]["audit"]["verdict"]
              for s in prov["remediation_ladder"]}
    assert stages["as_shipped"] == UNAUDITABLE
    assert stages["regression_n4"] == UNAUDITABLE  # too few samples
    assert stages["regression_n8"] == AUDITABLE    # full evidence set supplied