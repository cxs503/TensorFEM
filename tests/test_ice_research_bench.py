"""Tests for the ICE-SHOW-RESEARCH-BENCH ice research bench.

The bench is a *curation/showcase* surface: it does not re-derive any solver,
so the tests assert (a) the problem library is self-consistent (≥12 problems,
every one with a criterion and a reference), (b) both one-click demos run a
real solver path and emit a figure + a machine-readable verdict, and (c) the
ICE-* task board view is consistent with the library.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.ice_research_bench import (
    ICE_PROBLEMS,
    check_problem_library,
    demo_ice_pressure_area,
    demo_ice_plate_flexural_limit,
    ice_task_board,
)


# ---------------------------------------------------------------------------
# 1. Problem library: ≥12 problems, every one with a criterion and a reference
# ---------------------------------------------------------------------------

def test_library_has_at_least_twelve_problems():
    assert len(ICE_PROBLEMS) >= 12


def test_every_problem_has_a_criterion():
    """The 'all problems carry a 判据' acceptance is enforced by construction."""
    for p in ICE_PROBLEMS:
        assert p.criterion and p.criterion.strip(), f"{p.id} lacks a criterion"


def test_every_problem_has_a_reference():
    for p in ICE_PROBLEMS:
        assert p.reference and p.reference.strip(), f"{p.id} lacks a reference"


def test_problem_ids_are_unique():
    ids = [p.id for p in ICE_PROBLEMS]
    assert len(ids) == len(set(ids))


def test_library_self_check_passes():
    check = check_problem_library()
    assert check["n_problems"] >= 12
    assert check["n_with_criterion"] == check["n_problems"]
    assert check["n_with_reference"] == check["n_problems"]
    assert check["passed"] is True


def test_library_self_check_rejects_missing_criterion():
    """A problem without a criterion must fail the self-check (not pass)."""
    from tensorfem.ice_research_bench import IceProblem
    bad = [IceProblem(id="X", title="t", background="b",
                      criterion="", reference="r", module="m")]
    check = check_problem_library(bad)
    assert check["passed"] is False
    assert check["n_with_criterion"] == 0


# ---------------------------------------------------------------------------
# 2. One-click demos: real solver path, figure + verdict
# ---------------------------------------------------------------------------

def test_demo_ice_pressure_area_runs_and_passes(tmp_path):
    """Pressure-area demo: three routes agree to machine precision."""
    out = demo_ice_pressure_area(out_png=str(tmp_path / "pa.png"))
    assert out["demo"] == "ice_pressure_area"
    assert out["n_points"] >= 5
    assert (tmp_path / "pa.png").exists()          # figure emitted
    assert out["all_passed"] is True               # route agreement < 1e-9
    for v in out["verdicts"]:
        assert v["passed"] is True


def test_demo_ice_plate_flexural_limit_runs_and_passes(tmp_path):
    """Flexural-limit demo: FE first-crack load vs Kelvin/Nevel both < 3 %."""
    out = demo_ice_plate_flexural_limit(out_png=str(tmp_path / "fl.png"))
    assert out["demo"] == "ice_plate_flexural_limit"
    assert (tmp_path / "fl.png").exists()          # figure emitted
    assert out["all_passed"] is True               # both < 3 %
    for v in out["verdicts"]:
        assert v["rel_error"] < 0.03
        assert v["passed"] is True


# ---------------------------------------------------------------------------
# 3. ICE-* task board view
# ---------------------------------------------------------------------------

def test_task_board_is_consistent_with_library():
    board = ice_task_board()
    assert board["check"]["passed"] is True
    assert len(board["rows"]) == len(ICE_PROBLEMS)
    # every row carries the criterion / reference / module columns
    for row in board["rows"]:
        assert row["criterion"]
        assert row["reference"]
        assert row["module"]
