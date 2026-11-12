"""Guard for the rainflow fix spec (docs/rainflow-counting-defect.md §5).

The specification asks a repaired ``rainflow_cycles`` to satisfy three criteria
at once.  ``benchmarks/verified/rainflow_fix_spec/run.py`` grades three
candidate counters against them; these tests lock the *result* so the finding
cannot silently drift, and so a future repair cannot claim compliance by
picking whichever criterion is convenient.

They do not assert that the defect is fixed (it is not): they assert which
criteria are mutually exclusive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "verified" / "rainflow_fix_spec"
sys.path.insert(0, str(BENCH))

import run as spec  # noqa: E402


@pytest.fixture(scope="module")
def graded():
    return spec.grade_anchors(), spec.grade_gate()


def test_no_candidate_satisfies_all_three_criteria(graded):
    anchors, gate = graded
    for cname in spec.CANDIDATES:
        c1 = all(
            row["counters"][cname].get("c1_pass", False)
            for name, row in anchors.items()
            if name in spec.EXPECT
        )
        c3 = anchors["sine_40_periods"]["counters"][cname]["c3_pass"]
        c2 = gate[cname]["c2_pass"]
        assert not (c1 and c2 and c3), cname


def test_nested_spec_and_sine_spec_are_mutually_exclusive(graded):
    """The only counter that reproduces {1,1,2,2,3,3} halves the sine range."""
    anchors, _ = graded
    nested = anchors["nested_zigzag"]["counters"]
    sine = anchors["sine_40_periods"]["counters"]
    spec_satisfiers = [c for c in spec.CANDIDATES if nested[c]["c1_pass"]]
    assert spec_satisfiers == ["remainder"], spec_satisfiers
    for cname in spec_satisfiers:
        assert not sine[cname]["c3_pass"], cname
        # every sine cycle closed at the interpolated level: range halved,
        # count doubled
        assert sine[cname]["sum_count"] == pytest.approx(80.0, rel=0.05)
        assert sine[cname]["max_range"] == pytest.approx(0.5 * 2.0 * spec.SINE_A, rel=0.02)


def test_the_recommended_four_point_rule_also_misses_the_nested_anchor(graded):
    anchors, _ = graded
    st = anchors["nested_zigzag"]["counters"]["four_point"]
    assert st["full_ranges"] != [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    # and it loses arcs: bookkeeping residual is not negligible
    assert abs(st["arc_conservation_residual"]) > 1.0


def test_arc_conservation_alone_does_not_separate_good_from_bad(graded):
    """Production and remainder both conserve arcs yet disagree on structure."""
    anchors, _ = graded
    for cname in ("production", "remainder"):
        for name in ("const_amplitude_3", "nested_zigzag", "two_level_block"):
            st = anchors[name]["counters"][cname]
            assert abs(st["arc_conservation_residual"]) < 1e-6, (cname, name)


def test_engineering_gate_penalises_the_spec_satisfying_counter(graded):
    """C2 (Rayleigh, 3%) is the criterion that protects production damage."""
    _, gate = graded
    assert gate["production"]["c2_pass"]
    assert gate["four_point"]["c2_pass"]
    assert not gate["remainder"]["c2_pass"]
    # the mechanism: closing at interpolated levels doubles the cycle rate
    assert gate["remainder"]["count_rate_error"] > 0.5
    assert gate["remainder"]["damage_rate_ratio"] < 0.5