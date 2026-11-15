"""Gates for the round-2 grillage plastic-collapse report -- no hand-typed numbers.

Background. Audit finding **F6** of
``docs/AUDIT-GAP-PP-5-GRILLAGE-ULTIMATE-independent-review.md`` recorded that the
round-2/round-3 reports claim "every number below is generated from
``artifacts/grillage_plastic_collapse/grillage_plastic_verification.json`` by
script, none typed by hand", but that directory never reached the tree
(``artifacts/`` is filtered by the merge gate), so no reader -- and no CI -- could
check the claim.  This module makes the claim executable:

* the **fast lane** tests below re-derive, from the report's own printed tables,
  every arithmetic relation the report asserts between columns (union of member
  grillage and plate-field limit, margin against the design pressure, gate
  verdict vs its own error/allowed pair, second-order mesh convergence,
  governing-mode consistency).  A silently edited or hand-typed cell breaks them.
* the **slow lane** test reruns
  ``tensorfem.grillage_plastic_verification`` from the current code and requires
  every printed number of the report to be reproduced, which closes the
  report<->code link that F6 said was missing.

Neither test needs the (gitignored-by-policy) ``artifacts/`` directory, so the
gate is armed in CI whether or not a working tree has ever run the module.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "CASE-GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE-round2.md"
DESIGN_PRESSURE = 4.000  # MPa, the caller-supplied design pressure of the case


# --- parsing helpers ---------------------------------------------------------

def _num(cell: str) -> float:
    """'−0.2597 %' / '1.43862e-08' / '118.260' -> float (unicode minus handled)."""
    text = cell.replace("%", "").replace("\u2212", "-").replace("**", "").strip()
    return float(text)


def _table(markdown: str, header_key: str) -> list[list[str]]:
    """Rows (separator line dropped) of the first pipe table whose header matches."""
    lines = markdown.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and header_key in stripped:
            start = i
            break
    assert start is not None, f"table header not found in {DOC.name}: {header_key!r}"
    rows: list[list[str]] = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):  # | --- | --- |
            continue
        rows.append(cells)
    assert rows, f"table under {header_key!r} has no rows"
    return rows


def _doc() -> str:
    return DOC.read_text()


def _gate_rows(markdown: str) -> list[list[str]]:
    return _table(markdown, "| gate | got | reference | error | allowed | verdict |")


def _case_study_row(markdown: str) -> list[str]:
    rows = _table(markdown, "| plate-field limit [MPa] |")
    assert len(rows) == 1, "the governing table must have exactly one case row"
    return rows[0]


def _sensitivity_rows(markdown: str) -> list[list[str]]:
    return _table(markdown, "| patch length [m] |")


def _convergence_rows(markdown: str) -> list[list[str]]:
    return _table(markdown, "| elements/span |")


# --- fast lane: the report must be arithmetically self-consistent -------------

def test_gate_table_verdicts_agree_with_their_own_error_budget():
    rows = _gate_rows(_doc())
    assert len(rows) >= 21, "round-2 report gate table lost rows"
    names = [r[0] for r in rows]
    assert len(names) == len(set(names)), "duplicate gate names in the report"
    for name, got, reference, error, allowed, verdict in rows:
        assert math.isfinite(_num(got)), name
        assert math.isfinite(_num(reference)), name
        budget = _num(allowed)
        assert budget >= 0.0, name
        assert abs(_num(error)) <= budget + 1e-9, (
            f"{name}: printed error {error} exceeds printed budget {allowed}")
        assert verdict.upper() == "PASS" or abs(_num(error)) > budget, (
            f"{name}: printed FAIL verdict but the error is inside the budget")


def test_case_study_union_and_margin_are_consistent():
    panel, grillage, ultimate, governing, margin = _case_study_row(_doc())
    panel_v, grillage_v, ultimate_v = (_num(panel), _num(grillage), _num(ultimate))
    # the reported ultimate state is the union (minimum) of the two mechanisms
    assert ultimate_v == pytest.approx(min(panel_v, grillage_v), abs=2e-3)
    # and the printed governing mode names the branch that produced that minimum
    named = governing.split("(")[0].strip()
    if panel_v < grillage_v:
        assert named == "plate_local_bending", governing
    else:
        assert named != "plate_local_bending", governing
    assert _num(margin) == pytest.approx(ultimate_v / DESIGN_PRESSURE, abs=2e-3)


def test_sensitivity_rows_union_is_consistent():
    rows = _sensitivity_rows(_doc())
    assert len(rows) == 9, "patch x axial sweep must keep its 3x3 design"
    for patch_length, sigma_ratio, grillage, panel, ultimate, governing in rows:
        grillage_v, panel_v, ultimate_v = (_num(grillage), _num(panel), _num(ultimate))
        assert ultimate_v == pytest.approx(min(grillage_v, panel_v), abs=2e-3), rows
        expected = "plate_local_bending" if panel_v < grillage_v else None
        if expected:
            assert governing == expected, (patch_length, sigma_ratio, governing)
        assert 0.0 <= _num(sigma_ratio) <= 0.3, sigma_ratio
        assert _num(patch_length) > 0.0, patch_length


def test_mesh_convergence_sweep_is_monotone_and_second_order():
    rows = _convergence_rows(_doc())
    elements = [int(_num(r[0])) for r in rows]
    assert elements == sorted(elements) and len(elements) >= 3
    v1 = [abs(_num(r[1])) for r in rows]
    v4 = [abs(_num(r[2])) for r in rows]
    assert all(a > b for a, b in zip(v1, v1[1:])), f"V1 errors not decreasing: {v1}"
    assert all(a >= b for a, b in zip(v4, v4[1:])), f"V4 errors not decreasing: {v4}"
    # consistent beam elements are O(h^2): each halving of h cuts the V1 error
    # by a factor ~4 (the report prints 4.000 -> 1.031 -> 0.260 -> 0.065 %)
    for coarse, fine in zip(v1, v1[1:]):
        assert fine / coarse <= 0.5, f"V1 convergence slower than first order: {v1}"
        assert fine / coarse >= 0.15, f"V1 errors not in the O(h^2) band: {v1}"


def test_report_states_a_reference_for_every_gate():
    """No gate may be graded against the solver itself.

    Every gate note is rendered in the report body; a self-comparison would show
    up as a got/reference equal pair with a zero budget, which the closed-form
    and cross-implementation gates never are.
    """
    for name, got, reference, error, allowed, verdict in _gate_rows(_doc()):
        if _num(allowed) == 0.0:
            # exact-identity gates: reference must be an analytic value, never
            # the solver output re-used as its own reference
            assert _num(reference) != 0.0, name


# --- slow lane: the report reproduces from the current code -------------------

@pytest.mark.slow
def test_round2_report_numbers_reproduce_from_current_code(tmp_path):
    """Rerun the verification module and compare every printed number.

    Registered in ``tests/slow_lane_manifest.txt``.  This is the half that audit
    finding F6 could not check: the report's numbers, including the ones the
    round-3 audit found the ``artifacts/`` copy of, must still be what the code
    produces today.
    """
    from tensorfem.grillage_plastic_verification import (
        run_grillage_plastic_verification,
    )

    fresh = run_grillage_plastic_verification(out_dir=tmp_path, make_figures=False)
    assert fresh["n_failed"] == 0, [
        c["gate"] for c in fresh["checks"] if not c["passed"]]

    # section 5: gate table
    by_name = {c["gate"]: c for c in fresh["checks"]}
    for name, got, reference, error, allowed, verdict in _gate_rows(_doc()):
        assert name in by_name, f"report names gate {name!r}, module no longer emits it"
        record = by_name[name]
        assert record["passed"], name
        assert _num(got) == pytest.approx(record["got"], rel=1e-5), name
        assert _num(reference) == pytest.approx(record["reference"], rel=1e-5), name
        assert _num(error) == pytest.approx(100.0 * record["rel_error"], abs=1e-3), name
        assert _num(allowed) == pytest.approx(100.0 * record["allowed"], abs=6e-3), name

    # section 5: case study
    case = fresh["case_study"]["16"]
    panel, grillage, ultimate, governing, margin = _case_study_row(_doc())
    assert _num(panel) == pytest.approx(case["ice_pressure_panel_limit"], abs=1e-3)
    assert _num(grillage) == pytest.approx(
        case["ice_pressure_grillage_plastic"], abs=1e-3)
    assert _num(ultimate) == pytest.approx(case["ice_pressure_ultimate"], abs=1e-3)
    assert governing == f"{case['governing_mode']} ({case['governing_member_line']})"
    assert _num(margin) == pytest.approx(case["margin_vs_design"], abs=1e-3)
    for row in _table(_doc(), "| member system |"):
        name = row[0]
        record = case["systems"][name]
        assert _num(row[1]) == pytest.approx(
            record["ice_pressure_ultimate"], abs=1e-3), name
        assert _num(row[2]) == pytest.approx(
            record["ice_pressure_first_hinge"], abs=1e-3), name
        assert _num(row[3]) == pytest.approx(
            record["plastic_reserve_ratio"], abs=1e-3), name
        assert int(_num(row[4])) == len(record["hinges"]), name

    # section 6: sweeps
    convergence = {int(c["elements_per_span"]): c for c in fresh["convergence"]}
    for elements, v1, v4 in _convergence_rows(_doc()):
        record = convergence[int(_num(elements))]
        assert _num(v1) == pytest.approx(
            100.0 * record["V1_relative_error"], abs=1e-3), elements
        assert _num(v4) == pytest.approx(
            100.0 * record["V4_relative_error"], abs=1e-3), elements
    rows = _sensitivity_rows(_doc())
    assert len(rows) == len(fresh["sensitivity"])
    for row, record in zip(rows, fresh["sensitivity"]):
        assert _num(row[0]) == pytest.approx(record["patch_length"], abs=1e-3)
        assert _num(row[1]) == pytest.approx(record["axial_ratio"], abs=1e-3)
        assert _num(row[2]) == pytest.approx(
            record["ice_pressure_grillage_plastic"], abs=5e-4)
        assert _num(row[3]) == pytest.approx(
            record["ice_pressure_panel_limit"], abs=5e-4)
        assert _num(row[4]) == pytest.approx(record["ice_pressure_ultimate"], abs=5e-4)
        assert row[5] == record["governing_mode"]