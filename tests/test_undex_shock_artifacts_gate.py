"""Cross-artifact gate for the UNDEX shock evidence (``benchmarks/undex_shock``).

The UNDEX chain publishes five tables written by three drivers at three
different times, and a reader is entitled to assume they describe one and the
same ensemble.  They did not: the case matrix and the SRS comparison were
generated while ``cole_anchor.json`` still declared four cases, so two of the six
cases the anchor defines were missing from the published tables, and the
``gci_*`` columns had been produced with the refinement levels handed to
:func:`tensorfem.undex_chain.roache_gci` in the wrong order (which flips the
sign of the observed order, clips it to the lower bound, and inflates every
index) with quantities that the three levels do not demonstrate written as ``0``
instead of ``n/a``.

This module freezes the repaired state so it cannot regress silently:

* ``test_audit_is_green`` -- the reconciliation audit
  (:func:`tensorfem.undex_artifacts.audit_undex_shock_artifacts`) passes on the
  committed artifacts;
* the definition checks are re-derived independently here (band maxima from the
  published SRS curves, wall peak against ``fem_vs_cole.csv``, GCI from the
  recorded three-level values) rather than trusting the artifact under test;
* six negative controls corrupt a private copy of the artifacts and require the
  audit to *fail* -- a gate that cannot fail is not a gate.

Nothing in this module re-runs the finite-element solves; the physics is frozen
by :mod:`tests.test_undex_shock`, :mod:`tests.test_undex_chain` and
:mod:`tests.test_undex_cole_chain3`.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from tensorfem.undex_artifacts import (GCI_LIMIT, NOT_DEMONSTRATED,
                                       QUANTITY_SOURCES,
                                       audit_undex_shock_artifacts,
                                       gci_level_values, load_artifacts,
                                       recompute_gci_columns)
from tensorfem.undex_chain import SRS_BANDS_HZ, roache_gci

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "undex_shock"
FIGS = ROOT / "docs" / "assets" / "w23_undex_shock"
EXT_SCRIPT = ROOT / "scripts" / "w23_undex_srs_extend.py"
REL_TOL = 1.0e-9


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def art():
    return load_artifacts(ROOT)


@pytest.fixture(scope="module")
def audit():
    return audit_undex_shock_artifacts(ROOT)


@pytest.fixture
def scratch_root(tmp_path):
    """A private copy of the artifacts, safe to corrupt."""
    (tmp_path / "benchmarks").mkdir()
    shutil.copytree(BENCH, tmp_path / "benchmarks" / "undex_shock")
    (tmp_path / "docs" / "assets").mkdir(parents=True)
    shutil.copytree(FIGS, tmp_path / "docs" / "assets" / "w23_undex_shock")
    (tmp_path / "scripts").mkdir()
    shutil.copy(EXT_SCRIPT, tmp_path / "scripts" / EXT_SCRIPT.name)
    return tmp_path


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _mutate_json(path: Path, mutate) -> None:
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc, indent=2) + "\n")


# ---------------------------------------------------------------------------
# 1. the committed artifacts reconcile
# ---------------------------------------------------------------------------
def test_audit_is_green(audit):
    assert audit["failed"] == [], audit["failed"]
    assert audit["ok"] is True


def test_every_check_is_reported_with_a_status(audit):
    names = {"case_coverage", "extension_provenance", "chain2_self_consistency",
             "matrix_vs_chain3", "gci_column_semantics", "srs_internal",
             "matrix_srs_link", "figures", "chain_metrics_record"}
    assert set(audit["checks"]) == names
    for name, check in audit["checks"].items():
        assert check["status"] in ("PASS", "FAIL"), name


def test_all_anchor_cases_reach_every_artifact(audit, art):
    cov = audit["checks"]["case_coverage"]
    assert cov["n_cases"] == len(art["anchor"]["cases"]) == 6
    assert cov["missing"] == {}
    assert "C5" in cov["per_artifact"]["case_matrix"]
    assert "C6" in cov["per_artifact"]["srs_comparison"]


def test_extension_states_how_the_new_rows_were_obtained(art):
    ext = art["srs"]["extension"]
    assert ext["added_cases"] == ["C5", "C6"]
    assert Path(ext["generated_by"]).name == "w23_undex_srs_extend.py"
    repro = ext["reproduction_check"]
    assert repro["pass"] is True
    assert repro["tolerance"] <= 1.0e-8
    for cid, case in repro["per_case"].items():
        # a published case recomputed through the same code path
        assert case["max_rel_dev"] <= repro["tolerance"], (cid, case)
        assert case["fields_compared"] >= 30
    xcheck = ext["gci_correction"]["cross_check_vs_own_solves"]
    assert xcheck["pass"] is True
    assert {"C1", "C5", "C6"} <= set(xcheck["per_case"])


# ---------------------------------------------------------------------------
# 2. the GCI columns mean what they say
# ---------------------------------------------------------------------------
def test_gci_columns_follow_the_documented_argument_order(art):
    expected = recompute_gci_columns(art["grid"])
    for cid, row in art["matrix"].items():
        for column in QUANTITY_SOURCES:
            want = expected[cid][column]
            got = row[column]
            if want == NOT_DEMONSTRATED:
                assert got == NOT_DEMONSTRATED, (cid, column, got)
            else:
                assert abs(float(got) - want) / want < REL_TOL, (cid, column, got)


def test_non_demonstrated_quantities_are_not_scored_zero(art):
    """The peak wall pressure is mesh independent (``p_wall(0) = 2 p_m``).

    Its three levels are identical, so no refinement study can demonstrate a
    convergence order for it: the column must say so instead of writing ``0``,
    which reads as "zero discretisation error".
    """
    for cid, row in art["matrix"].items():
        levels = gci_level_values(art["grid"]["cases"][cid], "gci_wall_pressure")
        assert len(set(levels)) == 1, cid
        assert row["gci_wall_pressure"] == NOT_DEMONSTRATED, cid
        assert roache_gci(*levels)["available"] is False
    # and no case may claim convergence while a quantity is undemonstrated
    for cid, row in art["matrix"].items():
        cells = [row[c] for c in QUANTITY_SOURCES]
        demonstrated = all(c != NOT_DEMONSTRATED for c in cells) and \
            all(float(c) < GCI_LIMIT for c in cells if c != NOT_DEMONSTRATED)
        assert (row["gci_pass"] == "True") is bool(demonstrated), cid
        assert row["gci_pass"] == "False", (
            "%s: some quantity is undemonstrated, so gci_pass cannot be True" % cid)


def test_gci_correction_is_recorded_with_before_and_after(art):
    corr = art["srs"]["extension"]["gci_correction"]
    assert corr["authority"].endswith("grid_convergence.json")
    assert corr["sentinel"] == NOT_DEMONSTRATED
    for cid, log in corr["per_case"].items():
        assert set(log["after"]) == set(QUANTITY_SOURCES) | {"gci_pass"}
        if cid in ("C1", "C2", "C3", "C4"):
            assert set(log["before"]) == set(QUANTITY_SOURCES) | {"gci_pass"}
    # the correction removed convergence claims; it never added one
    flips = corr["verdict_flips"]
    assert set(flips) <= {"C1", "C2", "C3", "C4"}
    for cid in flips:
        assert corr["per_case"][cid]["before"]["gci_pass"] == "True"
        assert corr["per_case"][cid]["after"]["gci_pass"] is False


def test_criterion4_verdict_agrees_with_the_corrected_columns(art):
    c4 = art["grid"]["criterion4"]
    verdict = art["criteria"]["criteria"]["4_grid_convergence_10pct"]["verdict"]
    assert verdict == ("PASS" if not (c4["above_limit"] or c4["not_demonstrated"])
                       else "FAIL")
    assert verdict == "FAIL", "chain-3 criterion 4 fails; the matrix must not " \
                             "claim gci_pass for any case"


# ---------------------------------------------------------------------------
# 3. independent re-derivation of the published definitions
# ---------------------------------------------------------------------------
def test_srs_band_table_is_the_curve_maximum(art):
    bands = [tuple(b) for b in art["srs"]["bands_definition"]]
    assert bands == [tuple(b) for b in SRS_BANDS_HZ]
    for cid, case in art["srs"]["cases"].items():
        f = case["curves"]["frequencies_hz"]
        for zeta, key in ((0.02, "damping_0.02"), (0.05, "damping_0.05")):
            fem = case["curves"]["fem_sa_ms2_zeta%.2f" % zeta]
            ref = case["curves"]["reference_sa_ms2_zeta%.2f" % zeta]
            table = case[key]
            assert len(table) == len(bands) >= 6
            for row, (lo, hi) in zip(table, bands):
                idx = [i for i, fq in enumerate(f) if lo <= fq <= hi]
                want_fem = max(fem[i] for i in idx)
                want_ref = max(ref[i] for i in idx)
                assert abs(row["fem_peak_sa_ms2"] - want_fem) / want_fem < REL_TOL
                assert abs(row["reference_peak_sa_ms2"] - want_ref) / want_ref < REL_TOL
                assert abs(row["relative_deviation"]
                           - abs(want_fem - want_ref) / want_ref) < REL_TOL
                assert row["samples"] == len(idx)


def test_matrix_srs_summary_is_the_band_table(art):
    for cid, row in art["matrix"].items():
        table = art["srs"]["cases"][cid]["damping_0.02"]
        assert int(row["srs_bands"]) == len(table)
        want = max(b["relative_deviation"] for b in table)
        assert abs(float(row["srs_band_max_deviation"]) - want) / want < REL_TOL


def test_matrix_wall_peak_agrees_with_fem_vs_cole(art):
    worst = 0.0
    for cid, row in art["matrix"].items():
        a = float(row["wall_peak_pressure_mpa"])
        b = float(art["fem_vs_cole"][cid]["fem_wall_peak_mpa"])
        worst = max(worst, abs(a - b) / b)
    assert worst < REL_TOL


def test_rigid_wall_doubling_and_impulse_ratio_hold(art):
    for cid, row in art["matrix"].items():
        assert abs(float(row["wall_peak_over_cole"]) - 2.0) < 1.0e-12, cid
        assert abs(float(row["wall_peak_rel_dev_vs_cole"]) - 1.0) < 1.0e-12, cid
        precav = float(row["wall_impulse_precav_pa_s"])
        cole = float(row["cole_impulse_pa_s"])
        assert abs(float(row["wall_impulse_precav_over_cole"]) - precav / cole) \
            / (precav / cole) < 1.0e-12, cid
        assert abs(float(row["wall_impulse_precav_rel_dev_vs_cole"])
                   - abs(precav - cole) / cole) < 1.0e-12, cid


def test_chain_metrics_still_describes_its_own_rows(art):
    cm = art["chain_metrics"]
    assert cm["mission_id"] == 15095
    assert [r["case_id"] for r in cm["rows"]] == ["C1", "C2", "C3", "C4"]
    for row in cm["rows"]:
        mine = art["matrix"][row["case_id"]]
        for col in ("cole_peak_pressure_mpa", "cole_impulse_pa_s",
                    "recompute_rel_dev_peak_pressure"):
            a, b = float(row[col]), float(mine[col])
            assert abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1e-30), col


# ---------------------------------------------------------------------------
# 4. negative controls: the audit must be able to fail
# ---------------------------------------------------------------------------
def test_negative_control_missing_case(scratch_root):
    path = scratch_root / "benchmarks" / "undex_shock" / "case_matrix.csv"
    rows = [r for r in _rows(path) if r["case_id"] != "C6"]
    _write_rows(path, rows)
    rep = audit_undex_shock_artifacts(scratch_root)
    assert rep["ok"] is False
    assert "case_coverage" in rep["failed"]
    assert rep["checks"]["case_coverage"]["missing"]["case_matrix"] == ["C6"]


def test_negative_control_corrupted_srs_band(scratch_root):
    path = scratch_root / "benchmarks" / "undex_shock" / "srs_comparison.json"

    def corrupt(doc):
        doc["cases"]["C5"]["damping_0.02"][0]["fem_peak_sa_ms2"] *= 1.05

    _mutate_json(path, corrupt)
    rep = audit_undex_shock_artifacts(scratch_root)
    assert rep["ok"] is False
    assert "srs_internal" in rep["failed"]


def test_negative_control_zero_sentinel_for_undemonstrated_gci(scratch_root):
    path = scratch_root / "benchmarks" / "undex_shock" / "case_matrix.csv"
    rows = _rows(path)
    for row in rows:
        if row["case_id"] == "C1":
            row["gci_deflection"] = "0.0"
    _write_rows(path, rows)
    rep = audit_undex_shock_artifacts(scratch_root)
    assert rep["ok"] is False
    assert "gci_column_semantics" in rep["failed"]


def test_negative_control_broken_doubling_factor(scratch_root):
    path = scratch_root / "benchmarks" / "undex_shock" / "case_matrix.csv"
    rows = _rows(path)
    for row in rows:
        if row["case_id"] == "C2":
            row["wall_peak_over_cole"] = "1.5"
    _write_rows(path, rows)
    rep = audit_undex_shock_artifacts(scratch_root)
    assert rep["ok"] is False
    assert "chain2_self_consistency" in rep["failed"]


def test_negative_control_missing_figure(scratch_root):
    (scratch_root / "docs" / "assets" / "w23_undex_shock" / "fig7_srs_six_case.png").unlink()
    rep = audit_undex_shock_artifacts(scratch_root)
    assert rep["ok"] is False
    assert "figures" in rep["failed"]


def test_negative_control_undeclared_stale_chain_metrics(scratch_root):
    path = scratch_root / "benchmarks" / "undex_shock" / "srs_comparison.json"

    def drop_note(doc):
        doc["extension"].pop("chain_metrics_note", None)

    _mutate_json(path, drop_note)
    rep = audit_undex_shock_artifacts(scratch_root)
    assert rep["ok"] is False
    assert "chain_metrics_record" in rep["failed"]


def test_negative_control_unproven_extension(scratch_root):
    path = scratch_root / "benchmarks" / "undex_shock" / "srs_comparison.json"

    def break_proof(doc):
        doc["extension"]["reproduction_check"]["pass"] = False

    _mutate_json(path, break_proof)
    rep = audit_undex_shock_artifacts(scratch_root)
    assert rep["ok"] is False
    assert "extension_provenance" in rep["failed"]

# ---------------------------------------------------------------------------
# 5. the human-readable companion cannot drift from the audit
# ---------------------------------------------------------------------------
def test_artifacts_doc_reports_the_current_audit(audit):
    text = (BENCH / "ARTIFACTS.md").read_text()
    assert "Generated by `scripts/w23_undex_srs_extend.py`" in text
    for name, check in audit["checks"].items():
        assert "| `%s` | %s |" % (name, check["status"]) in text, name
    for cid in audit["cases"]:
        assert cid in text
