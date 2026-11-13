"""CASE-CANTILEVER-TIP -- guards for the measured-shear-coefficient evidence bundle.

The numbers live in ``benchmarks/verified/fem_cantilever_shear/kappa_recovery.json``;
these tests re-derive every published quantity from the stored sweep instead of
trusting the summary (recomputing the fits costs nothing, the solves are not repeated).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "verified" / "fem_cantilever_shear"
EVIDENCE = BENCH / "kappa_recovery.json"
LADDER = BENCH / "result.json"
DOC = REPO / "docs" / "CASE-CANTILEVER-TIP-kappa-recovery.md"


@pytest.fixture(scope="module")
def evidence() -> dict:
    if not EVIDENCE.exists():
        pytest.skip("kappa_recovery.json not generated")
    return json.loads(EVIDENCE.read_text())


@pytest.fixture(scope="module")
def module():
    import sys

    sys.path.insert(0, str(BENCH))
    import kappa_recovery

    return kappa_recovery


def test_artifacts_present(evidence):
    assert DOC.exists()
    for path in evidence["figures"].values():
        assert (REPO / path).exists(), path
    assert evidence["anchor"] == "fem_cantilever_shear"
    assert evidence["mission"] == "CASE-CANTILEVER-TIP"


def test_symbolic_oracle_rederives_both_shear_coefficients(module):
    """The 6/5 (energy) and 3/2 (section-centre) coefficients must be derived, not quoted."""
    oracle = module.exact_plane_stress_oracle()
    assert oracle["all_symbolic_checks_zero"] is True
    assert oracle["biharmonic_residual"] == "0"
    assert oracle["shear_resultant"] in {"-P", "-1.0*P"}
    # int tau^2 dA = (6/5) (P/A)^2 A  ->  b_energy = 6/5, kappa_energy = 5/6
    assert abs(oracle["b_energy_float"] - 1.2) < 1e-12
    assert abs(oracle["kappa_energy_float"] - 5.0 / 6.0) < 1e-12
    # section-centre translation of the same field -> b = 3/2, kappa = 2/3
    assert abs(oracle["b_centroid_float"] - 1.5) < 1e-12
    assert abs(oracle["kappa_centroid_float"] - 2.0 / 3.0) < 1e-12


def test_fit_is_rederivable_from_the_stored_sweep(evidence, module):
    rows = evidence["sweep"]["fit"]
    fit = module.fit_coefficients(rows)
    recorded = evidence["fit"]
    for key in ("A", "B", "C", "a_bending_ratio", "b_shear_ratio", "kappa_eff",
                "c_end_offset_m", "r2", "sigma_b", "sigma_kappa"):
        assert fit[key] == pytest.approx(recorded[key], rel=1e-9, abs=1e-12), key
    assert recorded["r2"] > 0.99
    assert recorded["sigma_kappa"] > 0.0


def test_model_reproduces_every_measured_level(evidence):
    fit = evidence["fit"]
    for row in evidence["sweep"]["fit"]:
        x = 1.0 / row["slenderness_L_over_h"]
        pred = fit["A"] + fit["B"] * x**2 + fit["C"] * x**3
        assert abs(row["error_vs_eb"] - pred) <= 1e-3
    assert fit["max_abs_residual"] <= 1e-3


def test_gate_verdicts_are_rederivable(evidence, module):
    """Every gate's boolean must follow from its own value and criterion."""
    gates = {g["name"]: g for g in evidence["gates"]}
    assert gates["symbolic_oracle_checks"]["value"] == 1.0
    for gate in evidence["gates"]:
        assert gate["verdict"] == module.ladder_verdict(gate), gate["name"]
    # the honest-FAIL set is part of the evidence, not something to be quietly dropped
    assert gates["kappa_eff_within_10pct_of_energy_5_6"]["verdict"] is False
    assert gates["kappa_eff_not_centroid_value"]["verdict"] is False
    assert gates["fit_reproduces_sweep"]["verdict"] is True
    assert gates["kappa_eff_uncertainty"]["verdict"] is True


def test_measured_kappa_is_closest_to_the_derived_centroid_value(evidence, module):
    oracle = module.exact_plane_stress_oracle()
    k_e = evidence["fit"]["kappa_eff"]
    d_centroid = abs(k_e - oracle["kappa_centroid_float"])
    d_energy = abs(k_e - oracle["kappa_energy_float"])
    assert d_centroid < d_energy
    # and the falsified hypothesis is recorded as such in the generated document
    text = DOC.read_text()
    assert "Honest FAIL" in text
    assert f"{k_e:.4f}" in text


def test_crosscheck_paths_are_recorded(evidence):
    rows = evidence["ladder_crosscheck"]
    assert len(rows) >= 5
    ladder = {r["slenderness_L_over_h"]: r for r in json.loads(LADDER.read_text())["ladder"]}
    for row in rows:
        published = ladder[row["slenderness_L_over_h"]]["solid_hex20_levels"][-1]
        assert row["measured_delta_m"] == pytest.approx(published["tip_deflection_m"], rel=1e-12)
        assert row["ladder_shear_recovery"] == pytest.approx(published["shear_recovery"], rel=1e-12)


def test_recovered_bending_ratio_is_close_to_one(evidence):
    """Bending flexibility is recovered to ~0.5 %: it cannot explain the shear residual."""
    assert abs(evidence["fit"]["a_bending_ratio"] - 1.0) < 0.01
    assert np.isfinite(evidence["fit"]["c_end_offset_m"])