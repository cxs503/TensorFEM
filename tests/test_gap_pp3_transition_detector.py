"""Transition-detector regression tests for GAP-PP-3-ICE-STRUCT-PLASTIC-SET.

The anchor's first acceptance criterion is "elastic->plastic transition point vs
the analytical yield load < 5 %".  ``docs/assets/gap_pp3_transition_detector.json``
is the programme-generated evidence that decides it; these tests assert the
*invariants* of that evidence and of the two estimators it compares, instead of
merely executing the code.

Invariants
----------
1. the elastic-field transition error decreases monotonically with h and is a
   second-order sequence whose Richardson limit is < 5 %;
2. ``< 5 %`` is met from 16x8 on and *not* at 8x4 (no silent pass by mesh luck);
3. the dense-ladder path crossing reproduces the elastic-field value (the
   ``PatchPath.first_yield_factor`` detector is not biased when the ladder stays
   elastic up to the transition);
4. the anchored 3-grid matrix's +29.7 %/+20.7 %/+20.4 % plateau is *larger* than
   the elastic-field error on the same meshes -- i.e. the plateau is a ladder
   artefact, not an element error (finding of docs/CASE-GAP-PP-3-TRANSITION-DETECTOR.md);
5. ``plate_capacity_summary`` exposes the ladder-independent estimator and labels
   its basis.
"""
from __future__ import annotations

import json
import math
import os

import pytest
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    low_temperature_flow_stress,
    plate_capacity_summary,
)

ASSETS = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
EVIDENCE = os.path.join(ASSETS, "gap_pp3_transition_detector.json")
MATRIX = os.path.join(ASSETS, "gap_pp3_3grid_results.json")

A, B, T = 2.0, 1.0, 0.020
PATCH = (0.5, 0.5)
E_MOD, NU, HARDENING = 206e9, 0.3, 1e9
GRADE, TEMPERATURE_C = "DH36", -20.0


@pytest.fixture(scope="module")
def evidence() -> dict:
    if not os.path.exists(EVIDENCE):
        pytest.skip("transition-detector evidence not generated")
    with open(EVIDENCE) as handle:
        return json.load(handle)


def _by_elements(evidence: dict) -> list[dict]:
    return [entry for _, entry in sorted(evidence["meshes"].items(),
                                         key=lambda kv: kv[1]["elements"])]


class TestMeshConvergence:
    def test_errors_are_positive_and_decreasing(self, evidence):
        """A coarse simply-supported FE plate is too stiff, so it yields late
        (positive error); refinement must reduce that error monotonically."""
        errors = [entry["error_elastic"] for entry in _by_elements(evidence)]
        assert len(errors) >= 3
        assert all(error > 0.0 for error in errors)
        assert all(b < a for a, b in zip(errors, errors[1:]))

    def test_observed_order_is_second_order(self, evidence):
        conv = evidence["criteria"]["convergence"]
        assert conv["asymptotic"] is True
        assert 1.5 <= conv["observed_order"] <= 3.0

    def test_richardson_limit_is_below_criterion(self, evidence):
        conv = evidence["criteria"]["convergence"]
        assert abs(conv["extrapolated_error"]) < 0.05

    def test_criterion_met_from_16x8_not_at_8x4(self, evidence):
        criteria = evidence["criteria"]["transition_lt_5pct_elastic_field"]
        assert criteria["met_from_mesh"] == "16x8", criteria
        assert criteria["met_at_finest_mesh"] is True
        by_name = {n: e["error_elastic"] for n, e in evidence["meshes"].items()}
        assert abs(by_name["8x4"]) > 0.05, "8x4 must not pass the 5 % criterion"
        assert abs(by_name["16x8"]) < 0.05
        assert abs(by_name["24x12"]) < 0.05


class TestDetectorAgreement:
    def test_dense_ladder_reproduces_elastic_field(self, evidence):
        for name, verdict in evidence["criteria"]["dense_ladder_reproduces_elastic_field"].items():
            assert verdict is True, f"{name}: path crossing deviates from the elastic field"

    def test_dense_ladder_crossing_value(self, evidence):
        entry = evidence["dense_ladder"]["8x4"]
        assert entry["first_yield_factor"] == pytest.approx(843884.605638, rel=1e-9)

    def test_low_load_probe_is_configuration_independent(self, evidence):
        """At 0.20 p_y the anchored (geometric + restrained) and the linear/free
        solves are the same problem, so the probe is usable for both cases."""
        for name, entry in evidence["meshes"].items():
            delta = entry.get("config_delta")
            if delta is None:
                continue
            assert abs(delta) < 1e-9, f"{name}: probe depends on the configuration"


class TestLadderArtefact:
    def test_anchored_plateau_exceeds_elastic_error(self, evidence):
        """The 3-grid matrix's plateau is a ladder artefact, not an element bias."""
        if not os.path.exists(MATRIX):
            pytest.skip("3-grid matrix evidence not present")
        with open(MATRIX) as handle:
            matrix = json.load(handle)
        for name, entry in evidence["meshes"].items():
            path_error = matrix["meshes"].get(name, {}).get("first_yield_error")
            if path_error is None:
                continue
            assert path_error > entry["error_elastic"] + 0.03, (
                f"{name}: path crossing {path_error:.4f} is not clearly above the "
                f"elastic-field error {entry['error_elastic']:.4f}")


class TestSummaryApi:
    def test_elastic_estimator_matches_probe(self):
        flow = low_temperature_flow_stress(GRADE, TEMPERATURE_C)
        py = analytic_first_yield_pressure(A, B, T, E_MOD, NU, flow.yield_stress, PATCH)
        summary = plate_capacity_summary(
            A, B, T, young=E_MOD, poisson=NU, grade=GRADE, temperature_c=TEMPERATURE_C,
            patch=PATCH, hardening=HARDENING, load_factors=[0.9 * py["pressure"]],
            nx=8, ny=4, layers=9)
        fe = summary["fe"]
        assert fe["first_yield_basis"] == "elastic-field"
        assert fe["first_yield_factor_elastic"] == pytest.approx(
            flow.yield_stress / fe["fe_surface_mises_per_pa"], rel=1e-12)
        assert fe["first_yield_error_elastic"] == pytest.approx(
            (fe["first_yield_factor_elastic"] - py["pressure"]) / py["pressure"], rel=1e-12)
        # coarse mesh: the elastic-field transition error is ~18 %, and the
        # summary must say so rather than hide behind a normalised number
        assert abs(fe["first_yield_error_elastic"]) > 0.05
        # ladder too short to bracket the crossing -> the path estimator is None
        assert fe["first_yield_error_path"] is None