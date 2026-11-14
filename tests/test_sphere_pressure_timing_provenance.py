"""Determinism boundary of the CASE-SPHERE-PRESSURE record (``result.json``).

Why this module exists
----------------------
The committed record used to carry a single hand-editable wall-clock scalar
(``flat_facet_control/wall_clock_s``).  Two things went wrong with it:

* the frozen sample read ``13.23 s`` while independent runs measured
  ``0.44-1.64 s`` for the same leaf -- a 8-30x stale outlier inside a record
  that is otherwise reproducible bit-for-bit;
* the closed PR #860 "fixed" it by typing ``0.54`` into the JSON without a run
  behind it.  Nothing in the tree could tell that apart from a measurement.

The record now declares its determinism boundary (``determinism`` block) and
reports the wall clock as a *repeated* measurement (``timing`` block, median of
``TIMING_REPEATS`` samples).  These tests lock three properties:

1. the declaration covers every non-reproducible leaf -- an undeclared
   non-deterministic leaf is a failure, not a surprise;
2. the timing record is internally consistent, so a lone edit of the scalar
   (the #860 pattern) is rejected, and an order-of-magnitude stale record is
   rejected by the plausibility bound while legitimate run-to-run spread is not;
3. a fresh solve reproduces every declared-deterministic leaf (leaf-by-leaf,
   1-ulp tolerance), i.e. the committed JSON is still what the code produces.

Runtime: ~2.5 s for the timing block, ~14 s for the full re-run.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASE_DIR = os.path.join(ROOT, "benchmarks", "verified", "fem_sphere_pressure_membrane")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, CASE_DIR)

import repro_check  # noqa: E402
from tensorfem.sphere_pressure_case import (  # noqa: E402
    LSB_TOL,
    TIMING_LEAVES,
    TIMING_PLAUSIBILITY_FACTOR,
    TIMING_REPEATS,
    flat_facet_shell4_control,
    run_case,
)


def _committed() -> dict:
    with open(os.path.join(CASE_DIR, "result.json")) as fh:
        return json.load(fh)


def _timing_record(median_s: float, *, samples=None, scalar=None) -> dict:
    """A minimal record whose timing block is internally consistent by construction."""
    if samples is None:  # bracket the median symmetrically
        samples = (round(median_s * 0.95, 3), round(median_s, 3), round(median_s * 1.05, 3))
    ordered = sorted(samples)
    median_s = ordered[len(ordered) // 2]
    return {
        "flat_facet_control": {
            "N_mean_rel_error": 133.16588851674985,
            "nonradial_residual_rel": 0.8397114365930362,
            "wall_clock_s": round(median_s if scalar is None else scalar, 2),
            "timing": {
                "clock": "time.perf_counter",
                "repeats": len(ordered),
                "samples_s": list(samples),
                "median_s": round(median_s, 3),
                "min_s": round(ordered[0], 3),
                "max_s": round(ordered[-1], 3),
                "spread_ratio": round(ordered[-1] / ordered[0], 2),
                "gated": False,
                "plausibility_factor": TIMING_PLAUSIBILITY_FACTOR,
            },
        },
        "figures": ["out/figures/convergence.png"],
        "reference": {"values": {"sigma_pa": 125.0}},
    }


# ---------------------------------------------------------------- declaration
def test_committed_record_declares_its_determinism_boundary():
    """Every leaf a re-run cannot reproduce must be declared before it is relied on."""
    record = _committed()
    declaration = record["determinism"]
    assert declaration["declared_nondeterministic_leaves"] == list(TIMING_LEAVES)
    assert declaration["lsb_tolerance_rel"] == LSB_TOL == 1e-12
    assert declaration["timing_plausibility_factor"] == TIMING_PLAUSIBILITY_FACTOR
    assert declaration["checker"].endswith("repro_check.py")

    # the declaration is exhaustive: no wall-clock-shaped leaf sits outside it
    undeclared = [
        path
        for path in repro_check.flatten(record)
        if ("clock" in path or path.startswith("/flat_facet_control/timing"))
        and not repro_check.is_timing_leaf(path)
        and not path.startswith("/determinism")
    ]
    assert undeclared == []

    # the timing record itself must be a measurement, not a number
    assert repro_check.timing_consistency(record) == []
    timing = record["flat_facet_control"]["timing"]
    assert timing["repeats"] == TIMING_REPEATS >= 3
    assert timing["gated"] is False
    assert timing["clock"] == "time.perf_counter"


# ------------------------------------------------- the closed PR #860 counterfactual
def test_a_lone_edit_of_the_wall_clock_scalar_is_rejected():
    """The #860 pattern -- type 0.54 into the scalar, measure nothing."""
    record = _timing_record(0.74, samples=(0.70, 0.72, 0.80), scalar=0.54)
    problems = repro_check.timing_consistency(record)
    assert any("wall_clock_s" in p and "median_s" in p for p in problems)

    # ... and a record with no timing block at all cannot pass either: a bare
    # scalar is exactly what #860 edited
    bare = _timing_record(0.74)
    del bare["flat_facet_control"]["timing"]
    problems = repro_check.timing_consistency(bare)
    assert problems and "missing" in problems[0]


def test_a_stale_frozen_timing_sample_fails_the_plausibility_bound():
    """The record this PR replaces carried 13.23 s; a re-run measures ~0.7 s."""
    stored = _timing_record(13.23, samples=(13.1, 13.23, 13.4))
    fresh = _timing_record(0.74, samples=(0.70, 0.74, 0.80))
    verdict = repro_check.compare(stored, fresh)
    assert verdict["passed"] is False
    assert verdict["timing_ratio"] > TIMING_PLAUSIBILITY_FACTOR
    assert any("plausibility" in p or "stale" in p for p in verdict["timing_problems"])


def test_legitimate_run_to_run_spread_is_not_flagged():
    """A 1.5x move is normal host noise; the gate must stay green on it."""
    stored = _timing_record(0.74, samples=(0.70, 0.74, 0.80))
    fresh = _timing_record(1.10, samples=(1.05, 1.10, 1.30))
    verdict = repro_check.compare(stored, fresh)
    assert verdict["timing_ratio"] < TIMING_PLAUSIBILITY_FACTOR
    assert verdict["passed"] is True, verdict["timing_problems"]
    # the tolerated leaves are reported, never silently dropped
    assert {row["leaf"] for row in verdict["timing_leaf_moves"]} == {
        "/flat_facet_control/wall_clock_s",
        "/flat_facet_control/timing/median_s",
        "/flat_facet_control/timing/min_s",
        "/flat_facet_control/timing/max_s",
        "/flat_facet_control/timing/samples_s[0]",
        "/flat_facet_control/timing/samples_s[1]",
        "/flat_facet_control/timing/samples_s[2]",
        "/flat_facet_control/timing/spread_ratio",
    }


# ---------------------------------------------------- drift vs. 1-ulp wobble
def test_drift_is_separated_from_last_bit_noise():
    """The flat-facet control has a measured 1 ulp wobble; a real move is not that."""
    stored = _timing_record(0.74)
    last_bit = _timing_record(0.74)
    last_bit["flat_facet_control"]["nonradial_residual_rel"] = 0.839711436593036
    verdict = repro_check.compare(stored, last_bit)
    assert verdict["drift"] == [] and verdict["passed"] is True

    moved = _timing_record(0.74)
    moved["flat_facet_control"]["nonradial_residual_rel"] = 0.8397  # 1e-5, a real change
    verdict = repro_check.compare(stored, moved)
    assert [row["leaf"] for row in verdict["drift"]] == [
        "/flat_facet_control/nonradial_residual_rel"
    ]
    assert verdict["passed"] is False

    # a gate verdict flip is drift too, never a tolerated leaf
    flipped = _timing_record(0.74)
    flipped["reference"]["values"]["sigma_pa"] = 250.0
    assert repro_check.compare(stored, flipped)["passed"] is False


def test_figure_panels_are_compared_by_basename_not_by_directory():
    stored = _timing_record(0.74)
    fresh = _timing_record(0.74)
    fresh["figures"] = ["/tmp/xyz/figures/convergence.png"]
    assert repro_check.compare(stored, fresh)["passed"] is True
    fresh["figures"] = ["/tmp/xyz/figures/sensitivity.png"]  # different panel
    assert repro_check.compare(stored, fresh)["passed"] is False


# ----------------------------------------------------------- live measurements
def test_fresh_control_reports_a_repeated_measurement():
    """The producer must emit the median of ``TIMING_REPEATS`` samples."""
    flat = flat_facet_shell4_control()
    timing = flat["timing"]
    assert timing["repeats"] == TIMING_REPEATS
    assert len(timing["samples_s"]) == TIMING_REPEATS
    assert timing["min_s"] <= timing["median_s"] <= timing["max_s"]
    assert timing["median_s"] == pytest.approx(
        sorted(timing["samples_s"])[TIMING_REPEATS // 2], abs=5e-4
    )
    # the legacy scalar is the median published at 2 dp, the block at 3 dp, so the
    # two may legitimately differ by half a unit in the last published digit
    assert flat["wall_clock_s"] == pytest.approx(timing["median_s"], abs=0.0051)
    # spread_ratio is published rounded to 2 dp
    assert timing["spread_ratio"] == pytest.approx(timing["max_s"] / timing["min_s"], abs=0.01)
    # a single sample is not allowed to define the record: the block is what the
    # consistency check reads, and it must be self-consistent
    record = {"flat_facet_control": flat}
    assert repro_check.timing_consistency(record) == []


def test_committed_record_reproduces_from_a_fresh_solve(tmp_path):
    """The whole record, leaf by leaf: no drift outside the declared boundary."""
    stored = _committed()
    fresh = run_case(out_dir=str(tmp_path))
    fresh["figures"] = [
        os.path.join(CASE_DIR, os.path.basename(p)) for p in fresh.get("figures") or []
    ]
    verdict = repro_check.compare(stored, fresh)
    assert verdict["leaves"]["union"] == len(repro_check.flatten(stored)) >= 1400
    assert verdict["drift"] == [], verdict["drift"][:5]
    assert verdict["timing_problems"] == []
    assert verdict["timing_ratio"] <= TIMING_PLAUSIBILITY_FACTOR
    assert verdict["passed"] is True
