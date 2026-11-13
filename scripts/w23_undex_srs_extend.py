"""Extend the W23 UNDEX chain artifacts to the full anchor case ensemble.

Why this script exists
----------------------
``scripts/w23_undex_shock_chain_run.py`` (mission 15095) generated
``benchmarks/undex_shock/{case_matrix.csv,srs_comparison.json,chain_metrics.json}``
at a time when ``cole_anchor.json`` declared **four** cases (C1-C4).  The chain-3
study (mission 15973) later grew the same anchor to **six** cases (adding C5,
W=2 kg/R=25 m and C6, W=100 kg/R=20 m) and re-ran every chain-3 artifact
(``fem_vs_cole.csv``, ``grid_convergence.json``, ``criteria_verdicts.json``) over
all six.  The chain-2 artifacts were never regenerated, so the published case
matrix and SRS comparison silently covered only two thirds of the ensemble the
anchor actually defines -- an artifact-staleness defect, not a physics result.

This script closes that gap **without touching the C1-C4 numbers**: it recomputes
the chain-2 row and the chain-2 SRS block for the missing cases through the same
code path and the same parameters, and merges them into the two artifacts.  It
then proves that the code path is identical by recomputing one already-published
case (default C1) and comparing every numeric field against the committed row
(tolerance ``1e-9`` relative); a mismatch aborts the merge instead of publishing
quietly different digits.

``chain_metrics.json`` is deliberately *not* rewritten: it is the mission-15095
provenance record (script/package hashes, criterion verdicts of the chain-2
protocol).  Its coverage gap is declared in the merged artifact's
``extension.chain_metrics_note`` and re-checked by
``tensorfem.undex_artifacts.audit_undex_shock_artifacts``.

Usage (staged so each call fits a short shell budget)
-----------------------------------------------------
::

    PYTHONPATH=src python scripts/w23_undex_srs_extend.py --stage solve --cases C5   # ~90 s
    PYTHONPATH=src python scripts/w23_undex_srs_extend.py --stage post  --cases C5   # ~2 s
    PYTHONPATH=src python scripts/w23_undex_srs_extend.py --stage solve --cases C1   # reproof case
    PYTHONPATH=src python scripts/w23_undex_srs_extend.py --stage post  --cases C1
    PYTHONPATH=src python scripts/w23_undex_srs_extend.py --stage merge              # writes artifacts

Intermediate results are cached under ``.scratch/undex_ext_cache`` (git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.undex_chain import (SRS_BANDS_HZ, assemble_panel,  # noqa: E402
                                   cole_case_environment,
                                   cole_recompute_vs_anchor,
                                   compare_panel_to_taylor, panel_fsi_shock,
                                   pressure_closure, roache_gci,
                                   srs_band_table, srs_pair)
from tensorfem.undex_artifacts import (NOT_DEMONSTRATED,  # noqa: E402
                                       QUANTITY_SOURCES, gci_level_values,
                                       recompute_gci_columns)
from tensorfem.undex_shock import (compare_to_taylor_analytic,  # noqa: E402
                                   taylor_analytic, taylor_fsi_fem)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "benchmarks" / "undex_shock"
ANCHOR = OUT / "cole_anchor.json"
MATRIX = OUT / "case_matrix.csv"
SRS = OUT / "srs_comparison.json"
FIGS = REPO / "docs" / "assets" / "w23_undex_shock"
CACHE = REPO / ".scratch" / "undex_ext_cache"

#: identical to scripts/w23_undex_shock_chain_run.py
LEVELS = ((12, 8, 40), (18, 12, 60), (27, 18, 90))
PANEL = {"length": 1.5, "width": 1.0, "thickness": 0.010, "young": 206.0e9,
         "poisson": 0.3, "density": 7850.0}
GCI_LIMIT = 0.10
GCIQ = {"peak_center_deflection_m": "peak centre deflection [m]",
        "peak_center_acceleration_ms2": "peak centre acceleration [m/s^2]",
        "wall_impulse_precav_pa_s": "pre-cavitation wall impulse [Pa s]",
        "peak_wall_pressure_pa": "peak wall pressure [Pa]"}
REPRO_TOL = 1.0e-9
#: ``gci_*``/``gci_pass`` exactly as published by scripts/w23_undex_shock_chain_run.py
#: (mission 15095, merge d582f272).  Kept as a literal because it is a *record*
#: of what was published -- the correction below must not be circular, and a
#: second run of this script must reproduce the same before/after log.
CHAIN2_GCI_BEFORE = {
    "C1": {"gci_deflection": "0.0", "gci_acceleration": "0.0041169977139499615",
           "gci_impulse_precav": "0.09898941100579425",
           "gci_wall_pressure": "0.0", "gci_pass": "True"},
    "C2": {"gci_deflection": "0.009779199576900193",
           "gci_acceleration": "0.004116997713954329",
           "gci_impulse_precav": "0.0", "gci_wall_pressure": "0.0",
           "gci_pass": "True"},
    "C3": {"gci_deflection": "0.012800847202985874",
           "gci_acceleration": "0.004116997713954939",
           "gci_impulse_precav": "0.0", "gci_wall_pressure": "0.0",
           "gci_pass": "True"},
    "C4": {"gci_deflection": "0.018169347533292145",
           "gci_acceleration": "0.004116997713953343",
           "gci_impulse_precav": "0.0", "gci_wall_pressure": "0.0",
           "gci_pass": "True"},
}
#: fields that are compared when reproving the code path on a published case;
#: the ``gci_*`` columns are excluded because they are deliberately corrected
#: (argument order + sentinel) -- comparing them would ask the reproof to
#: reproduce the defect
REPRO_SKIP = ("case_id", "gci_deflection", "gci_acceleration",
              "gci_impulse_precav", "gci_wall_pressure", "gci_pass")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _cache(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


def anchor_cases() -> list[dict]:
    return json.loads(ANCHOR.read_text())["cases"]


def case_environment(case: dict) -> dict:
    return cole_case_environment(case["weight_kg"], case["range_m"],
                                 case["depth_m"])


def stage_solve(cid: str) -> None:
    """Three-level panel solves for one case; cache the full result objects."""
    case = next(c for c in anchor_cases() if c["case_id"] == cid)
    env = case_environment(case)
    models = {k: assemble_panel(nx=lv[0], ny=lv[1], **PANEL)
              for k, lv in enumerate(LEVELS)}
    levels = [panel_fsi_shock(peak_pressure=env["peak_pressure_pa"],
                              decay_time=env["decay_time_s"], nx=LEVELS[k][0],
                              ny=LEVELS[k][1], steps_per_decay=LEVELS[k][2],
                              model=models[k]) for k in range(len(LEVELS))]
    torch.save({"case_id": cid, "env": env, "levels": levels,
                "levels_meta": [lv.meta for lv in levels]},
               _cache("%s_levels.pt" % cid))
    print("solved %s: peak wall %.6e Pa, dt %.6e s"
          % (cid, levels[-1].meta["peak_wall_pressure_pa"],
             levels[-1].meta["dt_s"]))


def stage_post(cid: str) -> None:
    """Everything downstream of the solves; cache the matrix row + SRS block."""
    d = torch.load(_cache("%s_levels.pt" % cid), weights_only=False)
    case = next(c for c in anchor_cases() if c["case_id"] == cid)
    anchor = json.loads(ANCHOR.read_text())
    env = d["env"]
    levels = d["levels"]
    fine = levels[-1]
    rec = next(r for r in anchor["recorded_values"] if r["case_id"] == cid)
    rc = next(r for r in cole_recompute_vs_anchor(anchor)["rows"]
              if r["case_id"] == cid)
    gci = {q: roache_gci(*[lv.meta[q] for lv in levels], ratio=1.5) for q in GCIQ}
    tc = compare_panel_to_taylor(fine)
    ref = taylor_analytic(fine.time, peak_pressure=env["peak_pressure_pa"],
                          decay_time=env["decay_time_s"],
                          areal_mass=fine.meta["areal_mass_kg_m2"])
    pair = srs_pair(fine.center_acceleration, ref.acceleration,
                    dt=fine.meta["dt_s"], damping_ratios=(0.02, 0.05))
    bands = srs_band_table(pair, damping=0.02)
    clo = pressure_closure(fine, cole_peak_pa=env["peak_pressure_pa"],
                           cole_impulse_pa_s=env["specific_impulse_pa_s"])
    fe = taylor_fsi_fem(peak_pressure=env["peak_pressure_pa"],
                        decay_time=env["decay_time_s"],
                        areal_mass=fine.meta["areal_mass_kg_m2"], length=1.0,
                        elements=120, dt=env["decay_time_s"] / 60.0,
                        duration=8.0 * env["decay_time_s"])
    fe_ref = taylor_analytic(fe.time, peak_pressure=env["peak_pressure_pa"],
                             decay_time=env["decay_time_s"],
                             areal_mass=fine.meta["areal_mass_kg_m2"])
    fe_cmp = compare_to_taylor_analytic(fe, fe_ref)
    t_fe = fe.cavitation_time_s or float(fe.time[-1])
    msk = fe.time <= t_fe
    fe_imp = float(torch.trapezoid(fe.interface_pressure[msk], fe.time[msk]))
    gv = {q: (gci[q]["gci_fine"] if gci[q]["available"] else 0.0) for q in GCIQ}
    row = {"case_id": cid, "weight_kg": case["weight_kg"],
           "range_m": case["range_m"], "depth_m": case["depth_m"],
           "reduced_range": rec["reduced_range"],
           "cole_peak_pressure_mpa": rec["peak_pressure_pa"] / 1e6,
           "cole_decay_time_us": rec["decay_time_s"] * 1e6,
           "cole_bubble_period_s": rec["bubble_period_s"],
           "cole_impulse_pa_s": rec["specific_impulse_pa_s"],
           "recompute_peak_pressure_mpa": rc["recomputed"]["peak_pressure_pa"] / 1e6,
           "recompute_rel_dev_peak_pressure": rc["relative_deviation"]["peak_pressure"],
           "recompute_rel_dev_decay_time": rc["relative_deviation"]["decay_time"],
           "recompute_rel_dev_bubble_period": rc["relative_deviation"]["bubble_period"],
           "recompute_rel_dev_impulse": rc["relative_deviation"]["specific_impulse"],
           "criterion1_pass": bool(max(rc["relative_deviation"].values()) < 0.01),
           "wall_peak_pressure_mpa": fine.meta["peak_wall_pressure_pa"] / 1e6,
           "wall_peak_over_cole": clo["reflection_factor_peak"],
           "wall_peak_rel_dev_vs_cole": clo["peak_rel_dev_vs_cole_free_field"],
           "wall_impulse_precav_pa_s": fine.meta["wall_impulse_precav_pa_s"],
           "wall_impulse_precav_over_cole":
               fine.meta["wall_impulse_precav_pa_s"] / env["specific_impulse_pa_s"],
           "wall_impulse_precav_rel_dev_vs_cole":
               abs(fine.meta["wall_impulse_precav_pa_s"]
                   - env["specific_impulse_pa_s"]) / env["specific_impulse_pa_s"],
           "cavitation_onset_us": (fine.meta["cavitation_onset_s"] or 0.0) * 1e6,
           "min_wall_pressure_mpa": fine.meta["min_wall_pressure_pa"] / 1e6,
           "center_deflection_mm": fine.meta["peak_center_deflection_m"] * 1e3,
           "center_acceleration_g": fine.meta["peak_center_acceleration_ms2"] / 9.80665,
           "deflection_ratio_vs_rigid": tc["deflection_ratio"],
           "taylor_velocity_rel_err": tc["center_velocity"]["relative_max_error"],
           "taylor_pressure_rel_err": tc["center_wall_pressure"]["relative_max_error"],
           "energy_residual_max": fine.meta["energy_residual_max"],
           "energy_residual_worst_level":
               max(lv.meta["energy_residual_max"] for lv in levels),
           "gci_deflection": gv["peak_center_deflection_m"],
           "gci_acceleration": gv["peak_center_acceleration_ms2"],
           "gci_impulse_precav": gv["wall_impulse_precav_pa_s"],
           "gci_wall_pressure": gv["peak_wall_pressure_pa"],
           "fe_fluid_pressure_rel_err": fe_cmp["interface_pressure"]["relative_max_error"],
           "fe_fluid_velocity_rel_err": fe_cmp["velocity"]["relative_max_error"],
           "fe_fluid_impulse_precav_pa_s": fe_imp,
           "fe_fluid_impulse_rel_dev_vs_cole":
               abs(fe_imp - env["specific_impulse_pa_s"]) / env["specific_impulse_pa_s"],
           "srs_bands": len(bands),
           "srs_band_max_deviation": max(b["relative_deviation"] for b in bands)}
    row["gci_pass"] = bool(all(row[k] < GCI_LIMIT for k in
                               ("gci_deflection", "gci_acceleration",
                                "gci_impulse_precav", "gci_wall_pressure")))
    srs_case = {
        "weight_kg": case["weight_kg"], "range_m": case["range_m"],
        "reference": ("Taylor exact air-backed areal-mass plate at the panel areal "
                      "mass; FEM curve = flexible SS panel, so deviations are the "
                      "panel-flexibility effect, not solver error"),
        "areal_mass_kg_m2": fine.meta["areal_mass_kg_m2"],
        "first_frequency_hz": fine.meta["first_frequency_hz"],
        "damping_0.02": bands,
        "damping_0.05": srs_band_table(pair, damping=0.05),
        "curves": {"frequencies_hz": pair["frequencies_hz"].tolist(),
                   "fem_sa_ms2_zeta0.02": pair["fem"].sa_ms2[0.02].tolist(),
                   "reference_sa_ms2_zeta0.02":
                       pair["reference"].sa_ms2[0.02].tolist(),
                   "fem_sa_ms2_zeta0.05": pair["fem"].sa_ms2[0.05].tolist(),
                   "reference_sa_ms2_zeta0.05":
                       pair["reference"].sa_ms2[0.05].tolist()},
        # same keys as the chain-2 entries, so every case is readable the same way
        "panel_fem_curves": {
            "time_s": fine.time.tolist(),
            "center_acceleration_ms2": fine.center_acceleration.tolist(),
            "center_wall_pressure_pa": fine.center_wall_pressure.tolist(),
            "center_deflection_m": fine.center_deflection.tolist(),
            "incident_pressure_pa": fine.incident_pressure.tolist(),
            "energy_incident_j": fine.energy_incident.tolist(),
            "energy_reflected_j": fine.energy_reflected.tolist(),
            "energy_plate_j": fine.energy_plate.tolist(),
            "energy_residual": fine.energy_residual.tolist()},
    }
    _cache("%s_post.json" % cid).write_text(
        json.dumps({"case_id": cid, "row": row, "srs_case": srs_case}, indent=2))
    print("post %s: srs_band_max_deviation %.6f, gci_pass %s"
          % (cid, row["srs_band_max_deviation"], row["gci_pass"]))


def _csv_value(v):
    """CSV text for a corrected GCI cell (``n/a`` stays a sentinel)."""
    if v is None or v == NOT_DEMONSTRATED:
        return NOT_DEMONSTRATED if v is not None else ""
    if isinstance(v, bool):
        return "True" if v else "False"
    return repr(float(v))


def _as_float(v):
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def reproduction_check(committed: dict, recomputed: dict) -> dict:
    """Compare a recomputed row with the published one field by field."""
    worst, worst_field, compared = 0.0, "", 0
    for k, old in committed.items():
        if k in REPRO_SKIP:
            continue
        new = recomputed[k]
        a, b = _as_float(old), _as_float(new)
        if a is None or b is None:
            if str(old) != str(new):
                worst, worst_field = float("inf"), k
            continue
        compared += 1
        scale = max(abs(a), abs(b), 1.0e-30)
        dev = abs(a - b) / scale
        if dev > worst:
            worst, worst_field = dev, k
    return {"max_rel_dev": worst, "worst_field": worst_field,
            "fields_compared": compared, "fields_excluded": list(REPRO_SKIP),
            "tolerance": REPRO_TOL, "pass": bool(worst <= REPRO_TOL)}


def gci_cross_check(cases=("C1", "C5", "C6")) -> dict:
    """Do this script's own three-level solves match the recorded level values?

    The corrected ``gci_*`` columns are recomputed from
    ``grid_convergence.json``, so the level values in that file are the input.
    For the cases this script solved itself the same three levels exist
    independently, and they must agree before the recorded values are used as a
    source.
    """
    grid = json.loads((OUT / "grid_convergence.json").read_text())
    per_case = {}
    for cid in cases:
        path = _cache("%s_levels.pt" % cid)
        if not path.exists():
            continue
        own = torch.load(path, weights_only=False)["levels_meta"]  # coarse -> fine
        worst = 0.0
        for column, (_, quantity, _) in QUANTITY_SOURCES.items():
            recorded = gci_level_values(grid["cases"][cid], column)
            measured = tuple(float(m[quantity]) for m in reversed(own))
            for a, b in zip(measured, recorded):
                scale = max(abs(a), abs(b))
                worst = max(worst, 0.0 if scale == 0.0 else abs(a - b) / scale)
        per_case[cid] = {"max_rel_dev": worst,
                         "quantities": list(QUANTITY_SOURCES),
                         "pass": bool(worst <= 1.0e-9)}
    return {"per_case": per_case,
            "pass": bool(per_case) and all(c["pass"] for c in per_case.values()),
            "tolerance": 1.0e-9,
            "note": ("the gci columns are recomputed from the three-level values "
                     "recorded by the chain-3 driver; for the cases solved here "
                     "the same levels are reproduced from an independent run")}


def stage_merge(add=("C5", "C6"), verify=("C1",)) -> None:
    header, committed_rows = [], {}
    with MATRIX.open(newline="") as fh:
        rd = csv.DictReader(fh)
        header = list(rd.fieldnames)
        for r in rd:
            committed_rows[r["case_id"]] = r
    posts = {}
    for cid in list(add) + list(verify):
        p = _cache("%s_post.json" % cid)
        if not p.exists():
            raise SystemExit("missing cache for %s -- run --stage solve/post" % cid)
        posts[cid] = json.loads(p.read_text())["row"]

    checks = {cid: reproduction_check(committed_rows[cid], posts[cid])
              for cid in verify}
    worst = max(c["max_rel_dev"] for c in checks.values())
    for cid, c in checks.items():
        print("reproof %s: max rel dev %.3e on %s (%d fields) -> %s"
              % (cid, c["max_rel_dev"], c["worst_field"], c["fields_compared"],
                 "PASS" if c["pass"] else "FAIL"))
    if not all(c["pass"] for c in checks.values()):
        raise SystemExit("reproof failed (worst %.3e > %.1e): refusing to merge "
                         "digits that differ from the published chain-2 run"
                         % (worst, REPRO_TOL))

    # --- GCI columns: correct argument order + honest sentinel ---------------
    grid = json.loads((OUT / "grid_convergence.json").read_text())
    implied = recompute_gci_columns(grid)
    xcheck = gci_cross_check()
    if not xcheck["pass"]:
        raise SystemExit("three-level cross-check failed: %s" % xcheck)
    gci_log, flipped = {}, []
    for cid in [c["case_id"] for c in anchor_cases()]:
        before = dict(CHAIN2_GCI_BEFORE[cid]) if cid in CHAIN2_GCI_BEFORE else None
        for column, want in implied[cid].items():
            value = want if column == "gci_pass" else (
                NOT_DEMONSTRATED if want == NOT_DEMONSTRATED else want)
            if cid in committed_rows:
                committed_rows[cid][column] = _csv_value(value)
            else:
                posts[cid][column] = value
        after = {c: implied[cid][c] for c in implied[cid]}
        if before and before["gci_pass"] == "True" and not implied[cid]["gci_pass"]:
            flipped.append(cid)
        gci_log[cid] = {"published_by_chain2": before is not None,
                        "before": before, "after": after}

    # --- case matrix: append the missing cases, physics rows untouched ------
    new_rows = [{k: posts[cid][k] for k in header}
                for cid in add if cid not in committed_rows]
    all_rows = list(committed_rows.values()) + new_rows
    order = [c["case_id"] for c in anchor_cases()]
    all_rows.sort(key=lambda r: order.index(r["case_id"])
                  if r["case_id"] in order else len(order))
    with MATRIX.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(all_rows)

    # --- SRS comparison: add the missing cases + the provenance block -------
    doc = json.loads(SRS.read_text())
    for cid in add:
        doc["cases"][cid] = json.loads(_cache("%s_post.json" % cid).read_text())["srs_case"]
    doc["cases"] = {cid: doc["cases"][cid] for cid in order if cid in doc["cases"]}
    doc["extension"] = {
        "generated_by": "scripts/w23_undex_srs_extend.py",
        "script_sha256": sha(REPO / "scripts" / "w23_undex_srs_extend.py"),
        "git_commit_at_run": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
            text=True).stdout.strip(),
        "anchor_sha256": sha(ANCHOR),
        "reason": ("cole_anchor.json grew to 6 cases under the chain-3 study, "
                   "while the chain-2 artifacts were generated for 4; the "
                   "missing cases were recomputed through the chain-2 code path "
                   "instead of leaving the published artifacts short of the "
                   "anchor ensemble"),
        "added_cases": [c for c in add],
        "code_path": {"levels_nx_ny_steps_per_decay": [list(v) for v in LEVELS],
                      "panel": PANEL, "gci_limit": GCI_LIMIT,
                      "srs_bands_hz": [list(b) for b in SRS_BANDS_HZ],
                      "srs_n_frequencies": 120, "srs_f_min_hz": 1.0,
                      "srs_f_max_hz": 10000.0, "damping_ratios": [0.02, 0.05],
                      "duration_theta": 16.0, "shear_treatment": "mitc",
                      "bc": "ss"},
        "reproduction_check": {"checked_cases": list(verify), "per_case": checks,
                              "tolerance": REPRO_TOL,
                              "pass": all(c["pass"] for c in checks.values()),
                              "note": ("a published case is recomputed through "
                                       "the same code path; agreement to 1e-9 "
                                       "relative is what licenses merging the "
                                       "new cases into the same table")},
        "chain_metrics_note": ("chain_metrics.json stays the mission-15095 (C1-C4) "
                               "provenance record and is not rewritten here; its "
                               "recorded anchor_sha256 predates the 6-case anchor"),
        "gci_correction": {
            "what": ("the gci_* and gci_pass columns of every row were recomputed "
                     "as roache_gci(fine, medium, coarse) from the three-level "
                     "values recorded in grid_convergence.json"),
            "why": ("the chain-2 driver passed the levels coarse->fine into a "
                    "signature that expects the fine level first; that flips the "
                    "sign of the observed order, which is then clipped to the "
                    "lower bound 0.5, so every published GCI was inflated; a "
                    "quantity whose three levels do not demonstrate convergence "
                    "was written as 0 instead of " + NOT_DEMONSTRATED),
            "sentinel": NOT_DEMONSTRATED,
            "limit": GCI_LIMIT,
            "verdict_flips": flipped,
            "per_case": gci_log,
            "log_is_a_fixed_record": ("before values are the literal chain-2 "
                                      "publication, not re-read from the file, so "
                                      "a second run logs the same correction"),
            "authority": "benchmarks/undex_shock/grid_convergence.json",
            "cross_check_vs_own_solves": xcheck,
            "not_changed": ("no physics column moves: peaks, impulses, "
                            "deflections, energy residuals and SRS entries are "
                            "the published values (see reproduction_check)"),
        },
    }
    SRS.write_text(json.dumps(doc, indent=2) + "\n")
    print("merged rows: %s ; srs cases: %s" % ([r["case_id"] for r in all_rows],
                                               list(doc["cases"])))
    render_figure(doc)
    render_artifacts_doc(gci_log, flipped, xcheck)


def render_figure(doc: dict) -> None:
    """fig7: band-peak SRS deviation, all cases (the extension's own picture)."""
    FIGS.mkdir(parents=True, exist_ok=True)
    cids = list(doc["cases"])
    bands = [tuple(b) for b in doc["bands_definition"]]
    labels = ["%g-%g" % b for b in bands]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    width = 0.8 / len(cids)
    x = np.arange(len(bands))
    for i, cid in enumerate(cids):
        devs = [b["relative_deviation"] for b in doc["cases"][cid]["damping_0.02"]]
        ax.bar(x + i * width, np.array(devs) * 100.0, width,
               label=cid, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x + width * (len(cids) - 1) / 2.0)
    ax.set_xticklabels(labels)
    ax.set_xlabel("SRS band [Hz]")
    ax.set_ylabel("band peak deviation FEM vs Taylor [%]")
    ax.set_title("UNDEX panel SRS, band-peak deviation (zeta = 0.02), %d cases"
                 % len(cids))
    ax.axhline(15.0, color="red", ls="--", lw=0.9, label="15 % gate")
    ax.legend(ncol=4, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_srs_six_case.png", dpi=150)
    plt.close(fig)
    print("wrote", FIGS / "fig7_srs_six_case.png")


def render_artifacts_doc(gci_log: dict, flipped: list, xcheck: dict) -> Path:
    """Write benchmarks/undex_shock/ARTIFACTS.md from the artifacts themselves."""
    from tensorfem.undex_artifacts import (DEFAULT_FIGURES,
                                            audit_undex_shock_artifacts)
    audit = audit_undex_shock_artifacts(REPO)
    srs = json.loads(SRS.read_text())
    rows = {r["case_id"]: r for r in csv.DictReader(MATRIX.open(newline=""))}
    cases = [c["case_id"] for c in anchor_cases()]
    L = []
    A = L.append
    A("# UNDEX shock artifacts -- how the tables reconcile")
    A("")
    A("Generated by `scripts/w23_undex_srs_extend.py` (do not edit by hand); the "
      "numbers below are read from the artifacts, and the check table is the "
      "output of `tensorfem.undex_artifacts.audit_undex_shock_artifacts`.")
    A("")
    A("The UNDEX evidence is written by three drivers at three times, so it is "
      "only readable if the tables are reconciled against each other. This file "
      "states what each artifact covers and what the audit says about it.")
    A("")
    A("## 1. What each artifact covers")
    A("")
    A("| artifact | cases | driver | written for |")
    A("|---|---|---|---|")
    A("| `cole_anchor.json` | %s | `scripts/w24_cole_anchor_citation.py` "
      "| anchor + citation ledger |" % ", ".join(cases))
    A("| `case_matrix.csv` | %s | `scripts/w23_undex_shock_chain_run.py` + this "
      "extension | chain-2 ensemble |" % ", ".join(rows))
    A("| `srs_comparison.json` | %s | same | chain-2 ensemble |"
      % ", ".join(srs["cases"]))
    A("| `chain_metrics.json` | %s | `scripts/w23_undex_shock_chain_run.py` "
      "| mission 15095 provenance record |"
      % ", ".join(r["case_id"] for r in
                  json.loads((OUT / "chain_metrics.json").read_text())["rows"]))
    A("| `fem_vs_cole.csv` | %s | `scripts/w24_undex_fem_vs_cole.py` "
      "| chain-3 criteria |" % ", ".join(
          r["case_id"] for r in csv.DictReader((OUT / "fem_vs_cole.csv").open())))
    A("| `grid_convergence.json` | %s | same | chain-3 convergence |"
      % ", ".join(json.loads((OUT / "grid_convergence.json").read_text())["cases"]))
    A("| `docs/assets/w23_undex_shock/%s` | 4-case figures (fig7: %d cases) "
      "| chain-2 driver + this extension | see caption in each figure |"
      % (", ".join(DEFAULT_FIGURES), len(srs["cases"])))
    A("")
    A("`chain_metrics.json` is deliberately *not* rewritten by the extension: it "
      "is the mission-15095 record (its own script/package hashes and the chain-2 "
      "criterion verdicts), and its `anchor_sha256` predates the six-case anchor. "
      "That staleness is declared in `srs_comparison.json` -> "
      "`extension.chain_metrics_note` instead of being papered over.")
    A("")
    A("## 2. What was stale and how it was repaired")
    A("")
    A("* The matrix and the SRS comparison were generated while the anchor "
      "declared four cases. C5 (W = 2 kg, R = 25 m) and C6 (W = 100 kg, R = 20 m) "
      "were missing from both tables while `fem_vs_cole.csv` and "
      "`grid_convergence.json` already covered all six. Both cases were "
      "recomputed through the chain-2 code path and merged; the physics columns "
      "of the published rows were not touched.")
    A("* Proof that the merged rows come from the same code path: case C1 was "
      "recomputed independently and every physics field agrees with the published "
      "row to %.1e (relative), %d fields compared."
      % (srs["extension"]["reproduction_check"]["per_case"]["C1"]["max_rel_dev"],
         srs["extension"]["reproduction_check"]["per_case"]["C1"]["fields_compared"]))
    A("* The three-level values used to correct the GCI columns were themselves "
      "cross-checked for %s: worst relative deviation %.1e."
      % (", ".join(sorted(xcheck["per_case"])),
         max(c["max_rel_dev"] for c in xcheck["per_case"].values())))
    A("")
    A("## 3. The GCI columns: argument order and the `n/a` sentinel")
    A("")
    A("`roache_gci` takes `(fine, medium, coarse)`. The chain-2 driver passed the "
      "three levels `coarse -> fine`, which flips the sign of the observed order; "
      "the order is then clipped to its lower bound 0.5, so every published index "
      "was inflated, and a quantity whose levels do not demonstrate convergence "
      "was written as `0` -- which reads as *zero discretisation error*. Both are "
      "corrected here: the columns are recomputed from the recorded three-level "
      "values in the documented order, and an undemonstrated quantity is marked "
      "`%s`." % NOT_DEMONSTRATED)
    A("")
    A("| case | quantity | published (chain-2) | corrected |")
    A("|---|---|---|---|")
    for cid in cases:
        log = gci_log.get(cid, {})
        before = log.get("before") or {}
        for column in QUANTITY_SOURCES:
            old = before.get(column, "(not published)")
            new = log["after"].get(column, "")
            if isinstance(new, float):
                new = "%.5f" % new
            A("| %s | `%s` | %s | %s |" % (cid, column, old, new))
        A("| %s | `gci_pass` | %s | %s |"
          % (cid, before.get("gci_pass", "(not published)"),
             log["after"].get("gci_pass")))
    A("")
    A("Consequence: `gci_pass` flips from `True` to `False` for %s, which is what "
      "the chain-3 record already said -- `criteria_verdicts.json` scores "
      "criterion 4 as %s, with the undemonstrated quantities listed by name. The "
      "peak wall pressure is undemonstrated by construction: `p_wall(0) = 2 p_m` "
      "independent of the mesh, so its three levels are identical and no "
      "refinement study can attribute an order to it."
      % (", ".join(flipped) or "no case",
         json.loads((OUT / "criteria_verdicts.json").read_text())
         ["criteria"]["4_grid_convergence_10pct"]["verdict"]))
    A("")
    A("## 4. The reconciliation gate")
    A("")
    A("`python -m pytest tests/test_undex_shock_artifacts_gate.py -q` (fast, no "
      "solves). Current outcome: **%s**."
      % ("PASS" if audit["ok"] else "FAIL: " + ", ".join(audit["failed"])))
    A("")
    A("| check | status | what it would catch |")
    A("|---|---|---|")
    for name, check in audit["checks"].items():
        A("| `%s` | %s | see `tensorfem.undex_artifacts` |"
          % (name, check["status"]))
    A("")
    A("The gate is not decorative: seven negative controls corrupt a private "
      "copy of the artifacts (missing case row, corrupted SRS band peak, `0.0` in "
      "place of `n/a`, broken rigid-wall doubling, missing figure, undeclared "
      "stale `chain_metrics`, unproven extension) and require the audit to fail.")
    A("")
    A("## 5. Limits of what this record establishes")
    A("")
    A("* The Cole constants are WITHHELD (see `README.md` section 2): every "
      "deviation here is measured against the repository's own transcription, "
      "not against a deposited copy of Cole (1948).")
    A("* The SRS reference is the exact Taylor air-backed *areal-mass* plate at "
      "the panel areal mass, so the band deviations measure the panel-flexibility "
      "effect (a stated physical difference), not solver error.")
    A("* Subtracting the two views never rescues a verdict: the impulse-like "
      "comparison and criterion 4 stay FAIL, and this file does not re-score "
      "them.")
    A("* `fig1`-`fig6` remain the four-case chain-2 figures; only `fig7` covers "
      "the six-case ensemble. Regenerating them needs the full six-case driver "
      "run (~8 min) and is not part of this change.")
    A("")
    path = OUT / "ARTIFACTS.md"
    path.write_text("\n".join(L) + "\n")
    print("wrote", path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=("solve", "post", "merge"))
    ap.add_argument("--cases", default="C5,C6",
                    help="comma separated case ids for the solve/post stages")
    ap.add_argument("--verify", default="C1",
                    help="comma separated published cases to reprove at merge")
    args = ap.parse_args()
    if args.stage == "solve":
        for cid in [c.strip() for c in args.cases.split(",") if c.strip()]:
            stage_solve(cid)
    elif args.stage == "post":
        for cid in [c.strip() for c in args.cases.split(",") if c.strip()]:
            stage_post(cid)
    else:
        stage_merge(add=tuple(c.strip() for c in args.cases.split(",") if c.strip()),
                    verify=tuple(c.strip() for c in args.verify.split(",") if c.strip()))


if __name__ == "__main__":
    main()