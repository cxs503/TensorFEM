"""Consistency gates for the GAP-PP-3 3-grid evidence bundle.

Background: the anchor's verdict document (`docs/CASE-GAP-PP-3-3GRID-MATRIX-VERDICT.md`,
merge #1193 / m22855) claimed every number was "programmatically injected" from a merged
JSON, yet that JSON was never committed and the published table cannot be true: the
"FE first yield" column and the "raw yield error" column imply two different analytic
yield pressures (the error column implies 0.7000 MPa, the table states 0.7125 MPa), the
reported observed order 1.2146 is `ln(0.090/0.055)/ln(1.5)` -- i.e. it uses the ratio of
the *last two* meshes of the old 8x4/12x6/16x8 family even though the family is
8x4/16x8/24x12 (ratio 2) -- and all three meshes report exactly 12.0 s.

These tests therefore assert *invariants* on the evidence bundle rather than merely
executing the auditor:

* every derived column must be recoverable from the bundle's own primitives
  (`first_yield_factor` / `first_yield_error` -> analytic pressure, `limit` / `UB` ->
  limit error, error sequence + mesh ratio -> observed order and Richardson limit);
* the stored `criteria` flags must be reproducible from the per-mesh entries;
* the bundle quoted by the committed case document must exist, pass the audit, cover
  three meshes and carry non-degenerate provenance (shards + non-uniform wall times);
* the case document's table must quote the same numbers as the committed bundle.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from audit_gap_pp3_3grid_bundle import audit_bundle, observed_order, richardson_limit  # noqa: E402

BUNDLE_PATH = os.path.join(ROOT, "docs", "assets", "gap_pp3_3grid_results.json")
DOC_PATH = os.path.join(ROOT, "docs", "CASE-GAP-PP-3-3GRID-MATRIX-VERDICT.md")
AUDITOR_PATH = os.path.join(ROOT, "scripts", "audit_gap_pp3_3grid_bundle.py")

PY = 0.7125e6
UB = 2.0528e6
LB = 0.5702e6


def _bundle(rows, *, obs_order=None, extrapolated=None, mesh_limit_flag=None,
            seconds=(12.0, 40.0, 120.0), shards=("gap_pp3_3grid_8x4.json",)):
    """Assemble a bundle from ``(name, nx, ny, fyf, err, limit, w_res)`` rows.

    ``fyf`` is ``p_y * (1 + err)`` when the rows are built consistently (see
    :func:`_consistent_rows`); passing a hand-typed ``fyf`` is exactly how a fabricated
    bundle is made.
    """
    bundle = {
        "case": {"a": 2.0, "b": 1.0, "t": 0.02, "patch": [0.5, 0.5]},
        "material": {"grade": "DH36", "temperature_c": -20.0},
        "analytic": {"first_yield_pressure": PY, "yield_line_upper_pressure": UB},
        "bounds": {"strip_lower_pressure": LB, "yield_line_upper_pressure": UB},
        "meshes": {},
        "criteria": {},
        "shards": {"used": list(shards), "missing": []},
    }
    for (name, nx, ny, fyf, err, limit, w_res), sec in zip(rows, seconds):
        bundle["meshes"][name] = {
            "nx": nx, "ny": ny, "elements": nx * ny,
            "dofs": 6 * (nx + 1) * (ny + 1),
            "first_yield_factor": fyf,
            "first_yield_error": err,
            "analytic_first_yield_same_mesh": fyf / (1.0 + 0.15),
            "first_yield_error_bias_normalised": (fyf - fyf / 1.15) / (fyf / 1.15),
            "limit_load_measured": limit,
            "limit_load_error_vs_ub": (limit - UB) / UB,
            "permanent_deflection": w_res,
            "permanent_set_valid": True,
            "path": [[0.0, 1e-3, 0.0], [1.0, 1.4 * w_res, 1e-3]],
            "seconds": sec,
        }
    errors = [m["first_yield_error"] for m in bundle["meshes"].values()]
    h = [1.0 / m["ny"] for m in bundle["meshes"].values()]
    p = observed_order(errors, h) if obs_order is None else obs_order
    lim = richardson_limit(errors, h) if extrapolated is None else extrapolated
    flag = (abs(lim) < 0.05) if mesh_limit_flag is None else mesh_limit_flag
    finest = list(bundle["meshes"])[-1]
    bundle["criteria"] = {
        "raw_error_convergence": {"h": h, "errors": errors,
                                  "observed_order": p, "extrapolated_error": lim},
        "transition_lt_5pct": {
            "finest_mesh": finest,
            "raw_error_finest": bundle["meshes"][finest]["first_yield_error"],
            "extrapolated_error": lim,
            "met_at_finest_mesh": abs(bundle["meshes"][finest]["first_yield_error"]) < 0.05,
            "met_in_mesh_limit": flag,
        },
        "limit_lt_10pct": {
            "finest_mesh": finest,
            "limit_measured": bundle["meshes"][finest]["limit_load_measured"],
            "upper_bound": UB,
            "error_finest": bundle["meshes"][finest]["limit_load_error_vs_ub"],
            "met_at_finest_mesh": abs(
                bundle["meshes"][finest]["limit_load_error_vs_ub"]) < 0.10,
        },
    }
    return bundle


def _consistent_rows(errors=(0.16, 0.07, 0.035), limits=(1.05e6, 1.09e6, 1.11e6),
                     w_res=(2.6e-3, 2.9e-3, 3.1e-3)):
    grids = ((8, 4), (16, 8), (24, 12))
    names = ("8x4", "16x8", "24x12")
    return [(n, nx, ny, PY * (1.0 + e), e, lim, w)
            for n, (nx, ny), e, lim, w in zip(names, grids, errors, limits, w_res)]


def _failed_checks(report):
    return {f["check"] for f in report["findings"] if not f["ok"]}


# ---------------------------------------------------------------------------
# the auditor itself
# ---------------------------------------------------------------------------

def test_consistent_bundle_passes_every_check():
    report = audit_bundle(_bundle(_consistent_rows()))
    assert report["ok"], _failed_checks(report)


def test_hand_typed_yield_error_is_caught():
    """The defect of the published table: FE value and error column disagree."""
    rows = [(n, nx, ny, 0.8288e6, e, lim, w) if n == "8x4" else (n, nx, ny, PY * (1 + e), e, lim, w)
            for n, nx, ny, _, e, lim, w in _consistent_rows()]
    report = audit_bundle(_bundle(rows))
    assert not report["ok"]
    assert "M1-analytic-recovered[8x4]" in _failed_checks(report)


def test_wrong_observed_order_is_caught():
    """A stored order that the error sequence does not reproduce."""
    bundle = _bundle(_consistent_rows(), obs_order=1.2145964603356645)
    report = audit_bundle(bundle)
    assert "C2-observed-order" in _failed_checks(report)


def test_wrong_richardson_limit_is_caught():
    bundle = _bundle(_consistent_rows(), extrapolated=-0.02)
    report = audit_bundle(bundle)
    assert "C3-richardson-limit" in _failed_checks(report)


def test_optimistic_mesh_limit_flag_is_caught():
    """`met_in_mesh_limit` must not be claimed when the Richardson limit is above 5 %."""
    bundle = _bundle(_consistent_rows(), extrapolated=0.08, mesh_limit_flag=True)
    report = audit_bundle(bundle)
    assert "K2-mesh-limit-flag" in _failed_checks(report)


def test_limit_outside_the_rigid_plastic_bracket_is_caught():
    """A collapse load above the yield-line upper bound is inadmissible."""
    rows = _consistent_rows(limits=(1.05e6, 1.09e6, 2.2e6))
    report = audit_bundle(_bundle(rows))
    assert "M6-limit-in-bracket[24x12]" in _failed_checks(report)


def test_permanent_set_above_peak_is_caught():
    rows = _consistent_rows()
    bundle = _bundle(rows)
    bundle["meshes"]["8x4"]["permanent_deflection"] = 5e-3   # > 1.4 * 2.6e-3 peak
    report = audit_bundle(bundle)
    assert "M5-permanent-set[8x4]" in _failed_checks(report)


def test_uniform_wall_times_are_reported_as_suspicious_but_not_blocking():
    report = audit_bundle(_bundle(_consistent_rows(), seconds=(12.0, 12.0, 12.0)))
    failed = _failed_checks(report)
    assert "P2-timing-plausibility" in failed     # warned
    assert report["ok"]                            # minor: does not fail the audit


def test_auditor_does_not_import_the_study_driver():
    """Independence: the auditor must not reuse the code that produced the numbers."""
    source = open(AUDITOR_PATH).read()
    imports = [line for line in source.splitlines()
               if line.startswith(("import ", "from "))]
    assert not any("gap_pp3_3grid" in line or "tensorfem" in line for line in imports), imports


# ---------------------------------------------------------------------------
# the published defect, pinned
# ---------------------------------------------------------------------------

PUBLISHED_ROWS = (
    # name, nx, ny, FE first yield [Pa], raw error, limit [Pa], w_res [m]
    ("8x4", 8, 4, 0.8288e6, 0.1840, 1.0600e6, 2.8e-3),
    ("16x8", 16, 8, 0.7630e6, 0.0900, 1.1000e6, 3.1e-3),
    ("24x12", 24, 12, 0.7385e6, 0.0550, 1.1200e6, 3.3e-3),
)


def test_published_table_of_m22855_is_not_reproducible():
    """The merged verdict doc's own table fails the algebraic identity of the code.

    `plate_patch_plastic_limit.plate_capacity_summary` defines
    ``first_yield_error = (first_yield_factor - p_y)/p_y``.  Recovering ``p_y`` from the
    published (FE value, error) pairs gives 0.7000 MPa on all three meshes while the same
    table states 0.7125 MPa -> the table cannot have been produced by the pipeline.
    """
    bundle = _bundle(PUBLISHED_ROWS, seconds=(12.0, 12.0, 12.0), shards=())
    report = audit_bundle(bundle)
    failed = _failed_checks(report)
    assert any(c.startswith("M1-analytic-recovered") for c in failed)
    details = [f["detail"] for f in report["findings"] if f["check"].startswith("M1-")]
    assert all("0.700000 MPa" in d for d in details)


PY_CONSTRUCTED = 0.700e6


def test_published_fe_values_are_constructed_from_the_error_column():
    """Smoking gun: every published "FE" yield value equals ``0.700 MPa * (1 + err)``.

    A solved value would scatter around ``p_y (1 + err)`` with the *stated* analytic
    ``p_y = 0.7125 MPa``; instead all three published values reproduce the error column
    exactly against a different reference of 0.7000 MPa (0.8288 / 0.7630 / 0.7385 MPa =
    0.700 x 1.1840 / 1.0900 / 1.0550).  The "FE" column is therefore the error column
    multiplied by a round number, not a finite-element result.
    """
    for name, nx, ny, fyf, err, limit, w_res in PUBLISHED_ROWS:
        assert fyf == pytest.approx(PY_CONSTRUCTED * (1.0 + err), rel=1e-12), name
        # ... and it is *not* consistent with the analytic pressure the table states
        assert abs(fyf / PY - (1.0 + err)) > 0.015, name


# ---------------------------------------------------------------------------
# the committed evidence bundle
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def committed():
    if not os.path.exists(BUNDLE_PATH):
        pytest.skip("3-grid evidence bundle not committed yet: the independent re-run "
                    "is queued on the compute gate (cj4580) and the previously published "
                    "table is refuted in docs/AUDIT-GAP-PP-3-3GRID-BUNDLE.md")
    with open(BUNDLE_PATH) as handle:
        return json.load(handle)


def test_committed_bundle_passes_the_consistency_audit(committed):
    report = audit_bundle(committed)
    assert report["ok"], _failed_checks(report)


def test_committed_bundle_covers_three_meshes(committed):
    assert len(committed["meshes"]) == 3
    ny = sorted(m["ny"] for m in committed["meshes"].values())
    assert ny == [4, 8, 12], ny


def test_committed_bundle_has_real_provenance(committed):
    shards = committed.get("shards") or {}
    assert shards.get("used"), "no shard provenance recorded"
    assert not shards.get("missing"), shards.get("missing")
    seconds = [m["seconds"] for m in committed["meshes"].values()]
    assert len(set(seconds)) == len(seconds), f"identical wall times {seconds}"
    assert all(s > 0 for s in seconds)


def test_committed_bundle_records_every_withheld_limit_with_a_reason(committed):
    for name, mesh in committed["meshes"].items():
        if mesh.get("limit_load_measured") is None:
            assert mesh.get("limit_withheld_reason"), name


# ---------------------------------------------------------------------------
# document / bundle agreement
# ---------------------------------------------------------------------------

def _doc_table_rows(text):
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8 or not re.fullmatch(r"\d+x\d+", cells[0]):
            continue
        rows[cells[0]] = cells
    return rows


def test_case_document_quotes_the_committed_bundle(committed):
    if not os.path.exists(DOC_PATH):
        pytest.skip("case document not present")
    rows = _doc_table_rows(open(DOC_PATH).read())
    assert set(rows) == set(committed["meshes"]), (sorted(rows), sorted(committed["meshes"]))
    for name, cells in rows.items():
        mesh = committed["meshes"][name]
        assert int(cells[1]) == mesh["elements"]
        # column 5 = FE first yield [MPa], column 6 = raw error
        assert float(cells[4].split()[0]) == pytest.approx(mesh["first_yield_factor"] / 1e6,
                                                           rel=2e-3)
        assert float(cells[5].rstrip("%")) / 100.0 == pytest.approx(
            mesh["first_yield_error"], abs=6e-5)

# ---------------------------------------------------------------------------
# verification debt pinned (mission m23702): limit error must use the
# yield-line upper bound, not the Ilyushin sine bound
# ---------------------------------------------------------------------------

def test_committed_limit_error_is_consistent_with_the_yield_line_ub(committed):
    """The ``limit_load_error_vs_ub`` column must equal ``(limit - yield_line_UB)/UB``.

    The committed bundle was regenerated (``scripts/regenerate_gap_pp3_3grid_evidence.py``)
    because an older ``gap_pp3_3grid_study.run_mesh`` had computed the column against the
    **Ilyushin sine** upper bound (``bounds.ilyushin_sine_upper_pressure``) while the
    ``criteria.limit_lt_10pct`` block and the auditor use the **yield-line** upper bound
    (``analytic.yield_line_upper_pressure``).  The two references differ by ~17 %, so the
    stored column disagreed with the auditor's ``M4-limit-error`` identity on every mesh
    and the bundle failed the consistency audit.  This test pins the corrected identity.
    """
    ub = committed["analytic"]["yield_line_upper_pressure"]
    for name, mesh in committed["meshes"].items():
        limit = mesh.get("limit_load_measured")
        if limit is None:
            continue
        expected = (float(limit) - float(ub)) / float(ub)
        assert mesh["limit_load_error_vs_ub"] == pytest.approx(expected, rel=1e-9), name


def test_committed_limit_criteria_uses_the_measured_not_the_withheld_value(committed):
    """``criteria.limit_lt_10pct.limit_measured`` must be the real measured limit.

    The pre-fix bundle carried a stale ``criteria`` block whose ``limit_measured`` was the
    *withheld* ``intersection_of_tangents`` value (72.4 kPa on 24x12) instead of the
    ``limit_load_measured`` (1.41 MPa) that ``limit_load_from_path`` actually returns when
    the tangent intersection is numerically unstable.  The criteria block must quote the
    same number as the per-mesh ``limit_load_measured`` so the verdict is not inflated.
    """
    finest = committed["criteria"]["limit_lt_10pct"]["finest_mesh"]
    mesh = committed["meshes"][finest]
    assert mesh["limit_load_measured"] is not None, "finest mesh limit must be measured"
    assert committed["criteria"]["limit_lt_10pct"]["limit_measured"] == \
        pytest.approx(mesh["limit_load_measured"], rel=1e-12)
