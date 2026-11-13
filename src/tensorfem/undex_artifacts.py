"""Cross-artifact reconciliation for the UNDEX shock benchmark chain.

The UNDEX shock evidence lives in several files that were produced by different
scripts at different times
(:mod:`tensorfem.undex_shock` -> :mod:`tensorfem.undex_chain` -> the W23/W24
drivers), so a published number can go stale without any single script looking
wrong.  This module loads the artifacts and checks that they agree with each
other and with their own definitions.  It is the machine-readable form of the
question "may these tables be read as one consistent record?".

Checks (each reported with ``PASS``/``FAIL`` plus the worst deviation found):

``case_coverage``
    the anchor's case set, the case matrix, the SRS comparison and the chain-3
    ``fem_vs_cole.csv`` describe the *same* ensemble.  A table that covers four
    of six cases is stale, however self-consistent its rows are.
``extension_provenance``
    the cases added after the fact carry provenance: which script, which code
    path, which anchor hash, and the recomputation proof that the rows merged
    into the original table come from the same code path.
``chain2_self_consistency``
    the matrix columns that are *definitions* (Cole recomputation, the rigid-wall
    doubling factor) hold to round-off, and the derived verdicts follow from the
    numbers they are derived from.
``matrix_vs_chain3``
    quantities that both the chain-2 and the chain-3 scripts report for the same
    case agree -- two independent drivers over one solver.
``gci_column_semantics``
    the ``gci_*`` columns are the Roache index of the recorded three-level
    values in the documented argument order, a quantity that is *not*
    demonstrated is marked :data:`NOT_DEMONSTRATED` rather than scored ``0``, and
    ``gci_pass`` follows from that.
``srs_internal``
    the SRS comparison is internally consistent: band definitions, damping
    curves, and the band table recomputed from the published curves.
``matrix_srs_link``
    the matrix's SRS columns are exactly the summary of the SRS comparison.
``figures``
    every published figure exists, is a real PNG, and is not a placeholder.
``chain_metrics_record``
    the mission-15095 provenance record still describes the rows it claims, and
    its (older) anchor hash is *declared* rather than silently inconsistent.

Only a ``FAIL`` blocks: the point of the module is to make staleness visible,
not to make it impossible to publish an honest partial record.
"""
from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

from .undex_chain import SRS_BANDS_HZ, roache_gci

__all__ = ["NOT_DEMONSTRATED", "QUANTITY_SOURCES", "GCI_LIMIT", "DEFAULT_FIGURES",
           "gci_level_values", "recompute_gci_columns", "load_artifacts",
           "audit_undex_shock_artifacts"]

#: written instead of a number when the GCI is not demonstrated
NOT_DEMONSTRATED = "n/a"
#: the chain-2/chain-3 convergence limit on the fine-grid GCI
GCI_LIMIT = 0.10
#: matrix column -> (grid_convergence section, quantity, recorded level order)
QUANTITY_SOURCES = {
    "gci_deflection": ("gci", "peak_center_deflection_m", "fine_first"),
    "gci_acceleration": ("gci", "peak_center_acceleration_ms2", "fine_first"),
    "gci_impulse_precav": ("supplementary", "wall_impulse_precav_pa_s", "coarse_first"),
    "gci_wall_pressure": ("supplementary", "peak_wall_pressure_pa", "coarse_first"),
}
DEFAULT_FIGURES = ("fig1_cole_environment.png", "fig2_wall_pressure.png",
                   "fig3_panel_response.png", "fig4_srs.png",
                   "fig5_energy_ledger.png", "fig6_grid_convergence.png",
                   "fig7_srs_six_case.png")
REL_TOL = 1.0e-9
STRICT_TOL = 1.0e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0.0 else abs(a - b) / scale


def _num(x) -> float | None:
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def gci_level_values(case_gc: dict, column: str) -> tuple[float, float, float]:
    """``(fine, medium, coarse)`` three-level values behind a matrix column.

    ``grid_convergence.json`` stores the critical quantities with the levels in
    ``fine, medium, coarse`` order (``roache_gci``'s own signature) and the
    supplementary ones in the run order ``coarse, medium, fine``; the mapping is
    declared per quantity in :data:`QUANTITY_SOURCES` instead of guessed here.
    """
    section, quantity, order = QUANTITY_SOURCES[column]
    values = case_gc[section][quantity]["values"]
    if order == "fine_first":
        return float(values[0]), float(values[1]), float(values[2])
    return float(values[2]), float(values[1]), float(values[0])


def recompute_gci_columns(grid_convergence: dict) -> dict[str, dict]:
    """The ``gci_*``/``gci_pass`` columns the recorded three-level values imply.

    A quantity whose three levels are non-monotone (or identical, as the peak
    wall pressure is by construction: ``p_wall(0) = 2 p_m`` independent of the
    mesh) cannot demonstrate convergence; it is reported as
    :data:`NOT_DEMONSTRATED` and never scored as ``0 %``.
    """
    out = {}
    for cid, case in grid_convergence["cases"].items():
        cols, demonstrated = {}, True
        for column in QUANTITY_SOURCES:
            fine, medium, coarse = gci_level_values(case, column)
            g = roache_gci(fine, medium, coarse, ratio=1.5)
            if g["available"]:
                cols[column] = g["gci_fine"]
                demonstrated = demonstrated and g["gci_fine"] < GCI_LIMIT
            else:
                cols[column] = NOT_DEMONSTRATED
                demonstrated = False
        cols["gci_pass"] = bool(demonstrated)
        out[cid] = cols
    return out


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("%s is not a PNG" % path.name)
    return struct.unpack(">II", head[16:24])


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_artifacts(root: str | Path | None = None) -> dict:
    """Read every artifact the audit needs (paths relative to the repo root)."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    bench = root / "benchmarks" / "undex_shock"
    art = {"root": root, "bench": bench}
    art["anchor"] = json.loads((bench / "cole_anchor.json").read_text())
    with (bench / "case_matrix.csv").open(newline="") as fh:
        rd = csv.DictReader(fh)
        art["matrix_header"] = list(rd.fieldnames)
        art["matrix"] = {r["case_id"]: r for r in rd}
    art["srs"] = json.loads((bench / "srs_comparison.json").read_text())
    art["chain_metrics"] = json.loads((bench / "chain_metrics.json").read_text())
    with (bench / "fem_vs_cole.csv").open(newline="") as fh:
        art["fem_vs_cole"] = {r["case_id"]: r for r in csv.DictReader(fh)}
    art["grid"] = json.loads((bench / "grid_convergence.json").read_text())
    art["criteria"] = json.loads((bench / "criteria_verdicts.json").read_text())
    art["figures_dir"] = root / "docs" / "assets" / "w23_undex_shock"
    return art


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def _check_coverage(art: dict) -> dict:
    anchor_cases = [c["case_id"] for c in art["anchor"]["cases"]]
    sets = {"anchor": set(anchor_cases),
            "case_matrix": set(art["matrix"]),
            "srs_comparison": set(art["srs"]["cases"]),
            "fem_vs_cole": set(art["fem_vs_cole"]),
            "grid_convergence": set(art["grid"]["cases"])}
    worst = max(sets.values(), key=len)
    ok = all(s == worst for s in sets.values())
    return {"status": "PASS" if ok else "FAIL",
            "n_cases": len(worst),
            "per_artifact": {k: sorted(v) for k, v in sets.items()},
            "missing": {k: sorted(worst - v) for k, v in sets.items() if v != worst}}


def _check_extension(art: dict, root: Path) -> dict:
    ext = art["srs"].get("extension")
    if not ext:
        return {"status": "FAIL", "detail": "srs_comparison.json has no extension block"}
    script = root / ext["generated_by"]
    repro = ext.get("reproduction_check", {})
    per = repro.get("per_case", {})
    worst = max((c["max_rel_dev"] for c in per.values()), default=float("inf"))
    covered = set(art["matrix"]) - set(ext["added_cases"])
    ok = (script.exists()
          and bool(repro.get("pass"))
          and worst <= max(float(repro.get("tolerance", 0.0)), REL_TOL)
          and float(repro.get("tolerance", 1.0)) <= 1.0e-8
          and set(ext["added_cases"]) <= set(art["matrix"])
          and len(per) >= 1)
    return {"status": "PASS" if ok else "FAIL",
            "script": ext["generated_by"], "script_exists": script.exists(),
            "added_cases": sorted(ext["added_cases"]),
            "cases_published_before_extension": sorted(covered),
            "reproduction_worst_rel_dev": worst,
            "reproduction_tolerance": repro.get("tolerance"),
            "reproduction_cases": sorted(per)}


def _check_chain2_self_consistency(art: dict) -> dict:
    worst, where, problems = 0.0, "", []
    for cid, row in art["matrix"].items():
        for col in ("recompute_rel_dev_peak_pressure", "recompute_rel_dev_decay_time",
                    "recompute_rel_dev_bubble_period", "recompute_rel_dev_impulse"):
            d = abs(_num(row[col]))
            if d > worst:
                worst, where = d, "%s:%s" % (cid, col)
        if d > STRICT_TOL:
            problems.append("%s:%s=%.3e" % (cid, col, d))
        if row["criterion1_pass"] not in ("True", "False"):
            problems.append("%s:criterion1_pass=%r" % (cid, row["criterion1_pass"]))
        elif (row["criterion1_pass"] == "True") != (worst <= 0.01):
            problems.append("%s:criterion1_pass disagrees with the deviations" % cid)
        dev_col = _num(row["wall_peak_rel_dev_vs_cole"])
        ratio = _num(row["wall_peak_over_cole"])
        if abs(ratio - 2.0) > STRICT_TOL:
            problems.append("%s:wall_peak_over_cole=%.15g (rigid-wall doubling is 2)"
                            % (cid, ratio))
        if abs(dev_col - 1.0) > STRICT_TOL:
            problems.append("%s:wall_peak_rel_dev_vs_cole=%.15g (doubling implies 1)"
                            % (cid, dev_col))
        precav = _num(row["wall_impulse_precav_pa_s"])
        cole_imp = _num(row["cole_impulse_pa_s"])
        if _rel(_num(row["wall_impulse_precav_over_cole"]), precav / cole_imp) > STRICT_TOL:
            problems.append("%s:wall_impulse_precav_over_cole is not the ratio" % cid)
        if _rel(_num(row["wall_impulse_precav_rel_dev_vs_cole"]),
                abs(precav - cole_imp) / cole_imp) > STRICT_TOL:
            problems.append("%s:wall_impulse_precav_rel_dev_vs_cole is not |dev|"
                            % cid)
        if _num(row["energy_residual_worst_level"]) < _num(row["energy_residual_max"]):
            problems.append("%s:worst-level energy residual below the fine one" % cid)
    return {"status": "PASS" if not problems else "FAIL",
            "worst_recompute_rel_dev": worst, "worst_field": where,
            "problems": problems[:12], "n_problems": len(problems)}


def _check_matrix_vs_chain3(art: dict) -> dict:
    worst, where, problems = 0.0, "", []
    pairs = (
        ("wall_peak_pressure_mpa", 1.0e6,
         lambda gc, cid: gci_level_values(gc["cases"][cid], "gci_wall_pressure")[0]),
        ("center_deflection_mm", 1.0e-3,
         lambda gc, cid: gci_level_values(gc["cases"][cid], "gci_deflection")[0]),
        ("center_acceleration_g", 9.80665,
         lambda gc, cid: gci_level_values(gc["cases"][cid], "gci_acceleration")[0]),
        ("wall_impulse_precav_pa_s", 1.0,
         lambda gc, cid: gci_level_values(gc["cases"][cid], "gci_impulse_precav")[0]),
    )
    for cid, row in art["matrix"].items():
        for col, scale, ref in pairs:
            dev = _rel(_num(row[col]) * scale, ref(art["grid"], cid))
            if dev > worst:
                worst, where = dev, "%s:%s" % (cid, col)
            if dev > REL_TOL:
                problems.append("%s:%s rel dev %.3e" % (cid, col, dev))
        dev = _rel(_num(row["energy_residual_worst_level"]),
                   art["grid"]["cases"][cid]["energy_ledger"]
                   ["residual_full_record_worst_level"])
        if dev > worst:
            worst, where = dev, "%s:energy_residual_worst_level" % cid
        if dev > REL_TOL:
            problems.append("%s:energy_residual_worst_level rel dev %.3e" % (cid, dev))
        fem_peak = _rel(_num(row["wall_peak_pressure_mpa"]),
                        _num(art["fem_vs_cole"][cid]["fem_wall_peak_mpa"]))
        if fem_peak > REL_TOL:
            problems.append("%s:wall peak disagrees with fem_vs_cole (%.3e)"
                            % (cid, fem_peak))
    return {"status": "PASS" if not problems else "FAIL",
            "worst_rel_dev": worst, "worst_field": where,
            "problems": problems[:12], "n_problems": len(problems)}


def _check_gci_columns(art: dict) -> dict:
    expected = recompute_gci_columns(art["grid"])
    worst, where, problems = 0.0, "", []
    for cid, row in art["matrix"].items():
        exp = expected[cid]
        for column, want in exp.items():
            got = row[column]
            if column == "gci_pass":
                if (got == "True") != bool(want):
                    problems.append("%s:gci_pass=%s but the recorded levels give %s"
                                    % (cid, got, want))
                continue
            if want == NOT_DEMONSTRATED:
                if got != NOT_DEMONSTRATED:
                    problems.append("%s:%s=%s but the levels do not demonstrate "
                                    "convergence" % (cid, column, got))
                continue
            g = _num(got)
            if g is None:
                problems.append("%s:%s=%r is not a number" % (cid, column, got))
                continue
            dev = _rel(g, float(want))
            if dev > worst:
                worst, where = dev, "%s:%s" % (cid, column)
            if dev > REL_TOL:
                problems.append("%s:%s=%.6g but the recorded levels give %.6g"
                                % (cid, column, g, float(want)))
    verdict = art["criteria"]["criteria"]["4_grid_convergence_10pct"]["verdict"]
    c4 = art["grid"]["criterion4"]
    c4_expect = "PASS" if not (c4["above_limit"] or c4["not_demonstrated"]) else "FAIL"
    if c4_expect != verdict:
        problems.append("criteria_verdicts criterion 4 (%s) disagrees with "
                        "grid_convergence (%s)" % (verdict, c4_expect))
    return {"status": "PASS" if not problems else "FAIL",
            "worst_rel_dev": worst, "worst_field": where,
            "limit": GCI_LIMIT, "criterion4_verdict": verdict,
            "problems": problems[:12], "n_problems": len(problems)}


def _check_srs_internal(art: dict) -> dict:
    doc = art["srs"]
    problems = []
    bands = [tuple(b) for b in doc["bands_definition"]]
    if bands != [tuple(b) for b in SRS_BANDS_HZ]:
        problems.append("band definition %s != library %s"
                        % (bands, [tuple(b) for b in SRS_BANDS_HZ]))
    if int(doc.get("band_coverage", 0)) != len(bands):
        problems.append("band_coverage does not count the bands")
    worst, where = 0.0, ""
    schemas = {cid: tuple(sorted(case)) for cid, case in doc["cases"].items()}
    if len(set(schemas.values())) > 1:
        problems.append("SRS case schemas differ: %s"
                        % {k: v for k, v in schemas.items() if
                           v != max(schemas.values(), key=len)})
    for cid, case in doc["cases"].items():
        curves = case["curves"]
        f = curves["frequencies_hz"]
        if len(f) < 2 or any(b <= a for a, b in zip(f, f[1:])):
            problems.append("%s:frequencies are not strictly increasing" % cid)
        if f[0] < SRS_BANDS_HZ[0][0] or f[-1] > SRS_BANDS_HZ[-1][1]:
            problems.append("%s:frequency grid leaves the band definition" % cid)
        for key in ("fem_sa_ms2_zeta0.02", "reference_sa_ms2_zeta0.02",
                    "fem_sa_ms2_zeta0.05", "reference_sa_ms2_zeta0.05"):
            if len(curves[key]) != len(f):
                problems.append("%s:%s length %d != %d grid points"
                                % (cid, key, len(curves[key]), len(f)))
        for zeta, key in ((0.02, "damping_0.02"), (0.05, "damping_0.05")):
            table = case[key]
            if len(table) != len(bands):
                problems.append("%s:%s has %d bands" % (cid, key, len(table)))
            for row, (lo, hi) in zip(table, bands):
                fem = row["fem_peak_sa_ms2"]
                ref = row["reference_peak_sa_ms2"]
                band = [i for i, fq in enumerate(f) if lo <= fq <= hi]
                dev = _rel(fem, max(curves["fem_sa_ms2_zeta%.2f" % zeta][i]
                                    for i in band))
                if dev > worst:
                    worst, where = dev, "%s:%s:%g-%g" % (cid, key, lo, hi)
                if dev > REL_TOL:
                    problems.append("%s:%s %g-%g fem peak %.6g != curve max"
                                    % (cid, key, lo, hi, fem))
                if _rel(row["relative_deviation"], abs(fem - ref) / ref) > REL_TOL:
                    problems.append("%s:%s %g-%g deviation is not |fem-ref|/ref"
                                    % (cid, key, lo, hi))
                if _rel(row["peak_ratio_fem_over_reference"], fem / ref) > REL_TOL:
                    problems.append("%s:%s %g-%g ratio is not fem/ref"
                                    % (cid, key, lo, hi))
        if not (_num(case["first_frequency_hz"]) > 0.0):
            problems.append("%s:first_frequency_hz is not positive" % cid)
    return {"status": "PASS" if not problems else "FAIL",
            "worst_peak_recompute_rel_dev": worst, "worst_field": where,
            "bands": ["%g-%g" % b for b in bands],
            "problems": problems[:12], "n_problems": len(problems)}


def _check_matrix_srs_link(art: dict) -> dict:
    problems, worst, where = [], 0.0, ""
    for cid, row in art["matrix"].items():
        case = art["srs"]["cases"][cid]
        if int(row["srs_bands"]) != len(art["srs"]["bands_definition"]):
            problems.append("%s:srs_bands=%s" % (cid, row["srs_bands"]))
        want = max(b["relative_deviation"] for b in case["damping_0.02"])
        dev = _rel(_num(row["srs_band_max_deviation"]), want)
        if dev > worst:
            worst, where = dev, "%s:srs_band_max_deviation" % cid
        if dev > REL_TOL:
            problems.append("%s:srs_band_max_deviation=%.6g != %.6g"
                            % (cid, _num(row["srs_band_max_deviation"]), want))
    return {"status": "PASS" if not problems else "FAIL",
            "worst_rel_dev": worst, "worst_field": where,
            "problems": problems[:12], "n_problems": len(problems)}


def _check_figures(art: dict, figures=DEFAULT_FIGURES) -> dict:
    problems, sizes = [], {}
    for name in figures:
        path = art["figures_dir"] / name
        if not path.exists():
            problems.append("%s missing" % name)
            continue
        if path.stat().st_size < 5000:
            problems.append("%s is %d bytes (placeholder?)"
                            % (name, path.stat().st_size))
            continue
        try:
            w, h = _png_size(path)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        sizes[name] = [w, h]
        if min(w, h) < 300:
            problems.append("%s is %dx%d px" % (name, w, h))
    return {"status": "PASS" if not problems else "FAIL", "sizes_px": sizes,
            "problems": problems, "n_problems": len(problems)}


def _check_chain_metrics(art: dict) -> dict:
    cm = art["chain_metrics"]
    problems, worst, where = [], 0.0, ""
    if int(cm["mission_id"]) != 15095:
        problems.append("mission_id=%s is not the chain-2 mission" % cm["mission_id"])
    if not cm["criterion1_recompute"]["pass"]:
        problems.append("chain-2 criterion 1 did not pass in its own record")
    for row in cm["rows"]:
        cid = row["case_id"]
        if cid not in art["matrix"]:
            problems.append("%s is in chain_metrics but not in the matrix" % cid)
            continue
        for col in ("cole_peak_pressure_mpa", "cole_impulse_pa_s",
                    "recompute_rel_dev_peak_pressure"):
            dev = _rel(_num(row[col]), _num(art["matrix"][cid][col]))
            if dev > worst:
                worst, where = dev, "%s:%s" % (cid, col)
            if dev > REL_TOL:
                problems.append("%s:%s disagrees with the matrix (%.3e)"
                                % (cid, col, dev))
    anchor_now = art["bench"] / "cole_anchor.json"
    import hashlib
    current = hashlib.sha256(anchor_now.read_bytes()).hexdigest()
    stale = cm["anchor_sha256"] != current
    note = art["srs"].get("extension", {}).get("chain_metrics_note")
    if stale and not note:
        problems.append("chain_metrics was recorded against another anchor "
                        "revision and no artifact declares it")
    return {"status": "PASS" if not problems else "FAIL",
            "recorded_cases": [r["case_id"] for r in cm["rows"]],
            "matrix_cases": sorted(art["matrix"]),
            "anchor_hash_matches_current": not stale,
            "declared_in": "srs_comparison.json -> extension.chain_metrics_note"
            if note else None,
            "worst_rel_dev": worst, "worst_field": where,
            "problems": problems[:12], "n_problems": len(problems)}


def audit_undex_shock_artifacts(root: str | Path | None = None) -> dict:
    """Run every cross-artifact check and summarise the outcome."""
    art = load_artifacts(root)
    checks = {
        "case_coverage": _check_coverage(art),
        "extension_provenance": _check_extension(art, art["root"]),
        "chain2_self_consistency": _check_chain2_self_consistency(art),
        "matrix_vs_chain3": _check_matrix_vs_chain3(art),
        "gci_column_semantics": _check_gci_columns(art),
        "srs_internal": _check_srs_internal(art),
        "matrix_srs_link": _check_matrix_srs_link(art),
        "figures": _check_figures(art),
        "chain_metrics_record": _check_chain_metrics(art),
    }
    failed = [k for k, v in checks.items() if v["status"] != "PASS"]
    return {"ok": not failed, "failed": failed, "checks": checks,
            "cases": sorted(art["matrix"]),
            "note": ("a PASS means the artifacts may be read as one record; it "
                     "does not upgrade any physical verdict -- those live in "
                     "criteria_verdicts.json")}