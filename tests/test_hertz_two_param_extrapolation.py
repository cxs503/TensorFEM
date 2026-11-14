#!/usr/bin/env python3
"""Regression tests for the joint (dz, H) limit study of the Hertz anchor.

They assert *invariants and decision logic*, not merely "the code ran":

* the pre-declared decision rule is a pure function of (e0, band) and behaves
  exactly as documented for the three verdict classes;
* the two-parameter grid is complete and converges monotonically in both
  parameters (refining either one must not increase |e_delta|);
* the nested limit is bracketed by the data it came from (a sanity bound that
  catches sign/order bugs in the extrapolation);
* the two independent channels (nested Richardson vs global two-term
  least-squares) agree to within the reported uncertainty band -- otherwise the
  band is not credible;
* the case-pool criterion (<3% on a and delta) holds on the accepted model;
* the generated document carries the artifact's numbers verbatim (no hand
  copying, no drift between doc and JSON).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "benchmarks", "verified",
                     "fem_hertz_contact_loadcontrol")
sys.path.insert(0, os.path.abspath(BENCH))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks", "verified"))

import two_param_extrapolation as tp  # noqa: E402
import make_two_param_doc as mkd  # noqa: E402
from _harness import richardson  # noqa: E402

ART = json.load(open(os.path.join(BENCH, "two_param_extrapolation.json")))


# --------------------------------------------------------------------------- #
# decision rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("e0,band,expect", [
    (0.00, 0.02, "H_disc"),      # small, band contains 0
    (-0.03, 0.05, "H_disc"),     # small negative, band contains 0
    (0.25, 0.03, "H_bias"),      # large, band excludes 0
    (-0.20, 0.05, "H_bias"),     # large negative, band excludes 0
    (0.10, 0.05, "inconclusive"),   # neither small enough nor convincingly large
    (0.20, 0.40, "inconclusive"),   # large but band swallows 0
    (float("nan"), 0.05, "inconclusive (grid incomplete)"),
])
def test_decide_rule(e0, band, expect):
    assert tp.decide(e0, band) == expect


def test_decide_thresholds_match_artifact():
    """The thresholds used by the artifact are the ones declared in CRITERIA."""
    assert ART["criteria"]["disc_abs_pct"] == tp.CRITERIA["disc_abs_pct"]
    assert ART["criteria"]["bias_abs_pct"] == tp.CRITERIA["bias_abs_pct"]
    assert ART["criteria"]["case_pool_pct"] == 3.0


# --------------------------------------------------------------------------- #
# harness arithmetic used by the study
# --------------------------------------------------------------------------- #
def test_richardson_arithmetic_reference_sequence():
    """A clean first-order sequence must be extrapolated to (almost) zero."""
    r = richardson(0.4, 0.2, 0.1, refinement_ratio=2.0)
    assert abs(r["observed_order"] - 1.0) < 1e-9
    assert abs(r["richardson_limit"]) < 1e-9


# --------------------------------------------------------------------------- #
# grid completeness + monotone convergence
# --------------------------------------------------------------------------- #
def test_grid_status_is_recorded():
    """Cells that did not land must be *declared*, not silently absent."""
    labels = {c[0] for c in tp.CELLS}
    assert set(ART["cells_missing"]) <= labels
    have = {p["label"] for p in ART["points"] if p["source"] == "cell"}
    assert have | set(ART["cells_missing"]) == labels
    assert ART["n_points"] == len(have) + sum(
        1 for p in ART["points"] if p["source"] != "cell")


def test_convergence_in_vertical_spacing():
    """At fixed depth, refining dz must not increase |e_delta|."""
    for depth in sorted({p["depth_over_a"] for p in ART["points"]}):
        sub = sorted([p for p in ART["points"]
                      if abs(p["depth_over_a"] - depth) < 1e-9],
                     key=lambda p: -p["dz_over_a"])
        if len(sub) < 2:
            continue
        mags = [abs(p["e_delta_pct"]) for p in sub]
        for coarse, fine in zip(mags, mags[1:]):
            assert fine <= coarse + 1e-3, (
                f"dz refinement increased |e_delta| at H={depth}a: {mags}")


def test_convergence_in_depth():
    """At fixed dz, deepening the domain must not increase |e_delta|."""
    for dz in sorted({round(p["dz_over_a"], 6) for p in ART["points"]}):
        sub = sorted([p for p in ART["points"] if round(p["dz_over_a"], 6) == dz],
                     key=lambda p: p["depth_over_a"])
        if len(sub) < 2:
            continue
        mags = [abs(p["e_delta_pct"]) for p in sub]
        for shallow, deep in zip(mags, mags[1:]):
            assert deep <= shallow + 1e-3, (
                f"depth refinement increased |e_delta| at dz={dz}a: {mags}")


# --------------------------------------------------------------------------- #
# the limit itself
# --------------------------------------------------------------------------- #
INCOMPLETE = ART["e0_pct"] is None
needs_limit = pytest.mark.skipif(
    INCOMPLETE, reason="joint limit not closed: finest-dz depth ladder pending")


@needs_limit
def test_limit_is_bracketed_by_data():
    """A first-order extrapolation cannot leave the data range by more than the
    data's own spread -- catches sign/ordering bugs."""
    e0 = ART["e0_pct"]
    assert e0 is not None and math.isfinite(e0)
    vals = [p["e_delta_pct"] for p in ART["points"]]
    lo, hi = min(vals), max(vals)
    spread = hi - lo
    assert lo - abs(spread) <= e0 <= hi + abs(spread), (e0, lo, hi)


@needs_limit
def test_two_channels_agree_within_band():
    """Nested Richardson vs global two-term fit must agree inside the band."""
    e0 = ART["e0_pct"]
    fit = ART["global_fit"]["e0_pct"]
    band = ART["band_pct"]
    assert band is not None and band > 0.0
    assert abs(e0 - fit) <= max(band, 0.05), (
        f"nested {e0} vs global fit {fit} differ by more than the band {band}")


@needs_limit
def test_dz_order_is_physical():
    """Richardson must return a *positive* observed order for the outer
    extrapolation; a NaN order means the sequence is not monotone and the study
    must not report a limit (honest-FAIL guard)."""
    assert math.isfinite(ART["nested"]["dz_observed_order"])
    assert ART["nested"]["dz_observed_order"] > 0.0


def test_interim_estimate_is_reported_while_incomplete():
    """An incomplete grid must ship an explicitly-labelled interim number."""
    it = ART.get("interim_estimate") or {}
    if INCOMPLETE:
        assert it.get("status", "").startswith("interim")
        assert math.isfinite(it["e0_interim_pct"])
        assert it["dz_limit_pct"] is not None and it["depth_limit_pct"] is not None


def test_pool_criterion_holds_on_accepted_model():
    pool = ART["pool"]
    assert abs(pool["e_delta_pct"]) < ART["criteria"]["case_pool_pct"]
    assert abs(pool["e_a_linear_pct"]) < ART["criteria"]["case_pool_pct"]
    assert pool["within_3pct"] is True


def test_verdict_matches_rule():
    assert ART["verdict"] == tp.decide(ART["e0_pct"], ART["band_pct"])


# --------------------------------------------------------------------------- #
# artifact -> document consistency (zero hand copying)
# --------------------------------------------------------------------------- #
@needs_limit
def test_doc_numbers_match_artifact(tmp_path):
    doc = mkd.render(ART)
    assert ART["verdict"] in doc or ART["verdict"].startswith("inconclusive")
    # every grid cell's e_delta appears in the rendered table
    for p in ART["points"]:
        assert f"{p['e_delta_pct']:+.4f}" in doc, p["label"]
    # the joint limit and its band are injected, not typed
    assert f"{ART['e0_pct']:+.4f}" in doc
    assert f"{ART['band_pct']:.4f}" in doc
    # the reference and the pool tolerance must be visible
    assert "Hertz 1882" in doc
    assert "<3%" in doc