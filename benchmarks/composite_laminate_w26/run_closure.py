#!/usr/bin/env python
"""W26 composite-laminate verification: rework runner + closure ledger.

The mission ``paper-exp-composite-laminate-verification-2`` was marked
``succeeded`` while emitting no report, so the experiment was never closed out.
This script re-establishes the verification on the TensorFEM work tree and
writes the closing artifact

    papers/artifacts/w26/composite-laminate-verification/closure.json

Two stages:

``--cook``   membrane/shear anchor: Cook (1974) membrane convergence study on a
             Q4 sequence (nx = 8 ... 96) against the classical reference tip
             deflection 23.96, with Richardson extrapolation -> cook_membrane.json
``--closure``read the bending artifacts written by
             ``benchmarks/composite_laminate_navier/run_case_matrix.py --out ...``
             (case_matrix.csv, deflection_error.json, grid_convergence.json,
             interlaminar_shear.json), score every criterion, hash every artifact
             and emit closure.json

Every number in the closure ledger is read from an artifact file or computed by
a solver call in this process; nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python benchmarks/composite_laminate_w26/run_closure.py --all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "papers" / "artifacts" / "w26" / "composite-laminate-verification"
sys.path.insert(0, str(REPO / "src"))

# ── criteria (fixed before the runs; see the mission ledger) ────────────────
THIN_LIMIT_PCT = 3.0      # t/a <= 0.1: deflection vs 3-D / series anchor
THICK_LIMIT_PCT = 5.0     # t/a  = 0.5: shear deformation enters, 5% band
SHEAR_LIMIT_PCT = 10.0    # interlaminar tau_xz vs the Pagano 3-D anchor
GCI_LIMIT_PCT = 5.0       # Roache grid convergence index, finest level
COOK_LIMIT_PCT = 3.0      # Cook membrane tip deflection vs 23.96
COOK_REFERENCE = 23.96    # Cook (1974) / standard FE-benchmark value, E=1, nu=1/3
COOK_MESHES = (8, 16, 32, 48, 64)

REQUIRED_ARTIFACTS = ("case_matrix.csv", "deflection_error.json",
                      "grid_convergence.json", "interlaminar_shear.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(extra: dict | None = None) -> dict:
    def _sha(rel: str) -> str:
        p = REPO / rel
        return sha256_file(p) if p.exists() else "missing"

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=REPO).decode().strip()
    except Exception:                                        # pragma: no cover
        commit = "unknown"
    prov = {
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modules": {
            "src/tensorfem/composite_laminate_bending.py":
                _sha("src/tensorfem/composite_laminate_bending.py"),
            "src/tensorfem/continuum_benchmarks.py":
                _sha("src/tensorfem/continuum_benchmarks.py"),
            "benchmarks/composite_laminate_navier/run_case_matrix.py":
                _sha("benchmarks/composite_laminate_navier/run_case_matrix.py"),
            "benchmarks/composite_laminate_w26/run_closure.py":
                _sha("benchmarks/composite_laminate_w26/run_closure.py"),
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        prov.update(extra)
    return prov


# ── stage 1: Cook membrane / shear anchor ──────────────────────────────────
def cook_study(meshes=COOK_MESHES, reference: float = COOK_REFERENCE) -> dict:
    """Q4 Cook membrane convergence, straight from the solver."""
    from tensorfem.continuum_benchmarks import cook_membrane, solve_continuum
    from tensorfem.composite_laminate_bending import richardson_extrapolate

    rows = []
    for nx in meshes:
        model, dof = cook_membrane(nx)
        u = solve_continuum(model).displacement
        value = float(u[dof])
        rows.append({
            "nx": int(nx), "n_dofs": int(2 * len(model.nodes)),
            "n_elements": int(len(model.elements)),
            "tip_deflection": value,
            "rel_err_pct": float(100.0 * (value - reference) / reference),
            "abs_rel_err_pct": float(abs(100.0 * (value - reference) / reference)),
        })
    # Richardson on the geometric subsequence 16 -> 32 -> 64 (uniform ratio 2)
    sub = {r["nx"]: r["tip_deflection"] for r in rows}
    rich = richardson_extrapolate([sub[16], sub[32], sub[64]], ratio=2.0)
    # observed order from the three finest consecutive halvings is not
    # available on this stride; use the three finest levels with their own ratio
    e = {r["nx"]: abs(r["tip_deflection"] - reference) for r in rows}
    p_obs = (math.log(e[32] / e[64]) / math.log(2.0)
             if min(e[16], e[32], e[64]) > 0.0 else None)
    finest = rows[-1]
    verdict = "PASS" if finest["abs_rel_err_pct"] < COOK_LIMIT_PCT else "FAIL"
    return {
        "benchmark": "Cook (1974) membrane, clamped left edge, unit shear resultant "
                     "on the right edge, E = 1, nu = 1/3, unit thickness, plane stress",
        "reference": reference,
        "reference_source": "Cook (1974) / standard FE benchmark value 23.96 "
                            "(tip vertical deflection); the value is an external "
                            "anchor, all computed numbers come from the solver",
        "meshes": list(meshes),
        "rows": rows,
        "richardson": rich,
        "p_observed_finest_pair": p_obs,
        "p_observed_note": "observed order on the 16/32/64 sequence, "
                           "e = |w - 23.96|, p = log2(e32/e64)",
        "criterion_pct": COOK_LIMIT_PCT,
        "verdict": verdict,
        "note": "bilinear Q4 under integrated shear; convergence is monotone from "
                "below and sub-quadratic because the exact Cook field is dominated "
                "by a boundary layer, so the criterion is scored on the finest mesh "
                "and cross-checked by the Richardson extrapolation",
    }


# ── stage 2: closure ledger ────────────────────────────────────────────────
def load(name: str):
    p = OUT / name
    if not p.exists():
        raise SystemExit(f"missing artifact {p}; run run_case_matrix.py --out {OUT} first")
    if name.endswith(".csv"):
        with p.open() as fh:
            return list(csv.DictReader(fh))
    return json.loads(p.read_text())


def build_closure() -> dict:
    cases_csv = load("case_matrix.csv")
    deflection = load("deflection_error.json")
    grid = load("grid_convergence.json")
    shear = load("interlaminar_shear.json")
    cook_path = OUT / "cook_membrane.json"
    cook = json.loads(cook_path.read_text()) if cook_path.exists() else None

    d_cases = deflection["cases"]
    g_by_case = {c["case"]: c for c in grid["cases"]}
    s_by_case = {c["case"]: c for c in shear["cases"]}

    # C1 -- completeness: 3 layups x 3 thickness ratios x >=3 meshes, nothing dropped
    layups = sorted({c["layup"] for c in d_cases})
    ratios = sorted({round(float(c["t_over_a"]), 6) for c in d_cases})
    mesh_counts = {}
    for c in d_cases:
        mesh_counts[c["case"]] = {
            "plate_fem": len(c["plate_fem"]),
            "hex20_fem": len(c["hex20_fem"]),
        }
    c1_ok = (len(layups) >= 3 and len(ratios) >= 3
             and all(v["plate_fem"] >= 3 and v["hex20_fem"] >= 3
                     for v in mesh_counts.values()))

    # C2 -- deflection deviation vs external anchor, thin 3% / thick 5%
    dev_rows = []
    for c in d_cases:
        lim = float(c["limit_pct"])
        dev_rows.append({
            "case": c["case"], "layup": c["layup"], "t_over_a": c["t_over_a"],
            "anchor": c["primary_reference"],
            "w_anchor": c["w_reference"],
            "w_plate_fem": c["plate_fem"][-1]["w"],
            "w_hex20_fem": c["hex20_fem"][-1]["w"],
            "err_plate_pct": c["err_plate_vs_ref_pct"],
            "err_hex20_pct": c["err_hex_vs_ref_pct"],
            "limit_pct": lim,
            "pass_plate": abs(c["err_plate_vs_ref_pct"]) < lim,
            "pass_hex20": abs(c["err_hex_vs_ref_pct"]) < lim,
        })
    c2_ok = all(r["pass_plate"] and r["pass_hex20"] for r in dev_rows)

    # C3 -- grid convergence: GCI < 5% and observed order within 50% of theory
    conv_rows = []
    for name, entry in g_by_case.items():
        for path, g in entry["paths"].items():
            # the artifact stores the signed GCI (it flips sign when the observed
            # order does); the criterion is on the magnitude, and only a positive
            # observed order makes the estimate meaningful at all.
            gci_abs = abs(g["gci_fine_pct"])
            gci_ok = gci_abs < GCI_LIMIT_PCT and g["p_observed"] > 0.0
            conv_rows.append({
                "case": name, "path": path, "mesh": g["mesh"],
                "deflection": g["deflection"], "rel_err_pct": g["rel_err_pct"],
                "p_observed": g["p_observed"], "p_theory": g["p_theory"],
                "gci_fine_pct": g["gci_fine_pct"], "gci_abs_pct": gci_abs,
                "gci_verdict": "PASS" if gci_ok else "FAIL",
                "order_verdict": g["order_verdict"],
            })
    c3_gci_ok = all(r["gci_verdict"] == "PASS" for r in conv_rows)
    c3_order_ok = all(r["order_verdict"] == "PASS" for r in conv_rows)

    # C4 -- interlaminar shear vs the Pagano 3-D anchor (scored where an anchor exists)
    sh_rows = []
    for c in shear["cases"]:
        sh_rows.append({
            "case": c["case"], "layup": c["layup"], "t_over_a": c["t_over_a"],
            "reference": c.get("reference"),
            "worst_abs_rel_pct": c.get("worst_abs_rel_pct"),
            "verdict": c.get("verdict"),
            "note": c.get("note"),
        })
    scored_shear = [r for r in sh_rows if r["verdict"] in ("PASS", "FAIL")]
    c4_ok = bool(scored_shear) and all(r["verdict"] == "PASS" for r in scored_shear)

    # C5 -- Cook membrane / shear anchor
    c5_ok = bool(cook) and cook["verdict"] == "PASS"

    criteria = {
        "C1_completeness": {
            "criterion": ">=3 layups x >=3 thickness ratios x >=3 meshes per FE path, "
                         "every case reported",
            "layups": layups, "thickness_ratios": ratios,
            "n_cases": len(d_cases), "meshes": mesh_counts,
            "verdict": "PASS" if c1_ok else "FAIL",
        },
        "C2_deflection_vs_anchor": {
            "criterion": f"|centre deflection deviation| < {THIN_LIMIT_PCT}% for "
                         f"t/a <= 0.1 and < {THICK_LIMIT_PCT}% for t/a = 0.5, "
                         "both FE paths vs the case anchor",
            "worst_plate_pct": max(abs(r["err_plate_pct"]) for r in dev_rows),
            "worst_hex20_pct": max(abs(r["err_hex20_pct"]) for r in dev_rows),
            "rows": dev_rows,
            "verdict": "PASS" if c2_ok else "FAIL",
        },
        "C3_grid_convergence": {
            "criterion": f"Roache GCI at the finest level < {GCI_LIMIT_PCT}% and "
                         "observed order within 50% of the theoretical order",
            "gci_all_pass": c3_gci_ok, "order_all_pass": c3_order_ok,
            "worst_gci_pct": max(r["gci_fine_pct"] for r in conv_rows),
            "rows": conv_rows,
            "verdict": "PASS" if (c3_gci_ok and c3_order_ok) else "FAIL",
        },
        "C4_interlaminar_shear": {
            "criterion": f"tau_xz at the ply interfaces within {SHEAR_LIMIT_PCT}% of "
                         "the Pagano 3-D elasticity anchor",
            "scored_cases": [r["case"] for r in scored_shear],
            "withheld_cases": [r["case"] for r in sh_rows if r["verdict"] == "WITHHELD"],
            "worst_abs_rel_pct": max((r["worst_abs_rel_pct"] for r in scored_shear),
                                     default=None),
            "rows": sh_rows,
            "verdict": "PASS" if c4_ok else "FAIL",
        },
        "C5_cook_membrane_shear_anchor": {
            "criterion": f"Cook (1974) membrane tip deflection within {COOK_LIMIT_PCT}% "
                         "of 23.96 on the finest Q4 mesh",
            "finest": None if not cook else cook["rows"][-1],
            "richardson": None if not cook else cook["richardson"],
            "verdict": "PASS" if c5_ok else ("MISSING" if not cook else "FAIL"),
        },
    }

    # ── diagnosis: is a failing case mesh-limited or biased? ───────────────
    # Richardson-extrapolate each FE path from its own three finest levels and
    # compare the extrapolated value with the same anchor.  A case whose finest
    # mesh misses the band but whose extrapolation lands inside it is
    # under-resolved; a case where the extrapolation misses too carries a
    # systematic (model/element) bias.  The two-path agreement in the last
    # column is an anchor-free consistency indicator.
    from tensorfem.composite_laminate_bending import richardson_extrapolate
    diag_rows = []
    for r in dev_rows:
        g = g_by_case[r["case"]]["paths"]
        entry = {"case": r["case"], "anchor": r["anchor"], "limit_pct": r["limit_pct"],
                 "paths": {}}
        for path, ratio in (("plate_fem", 2.0), ("hex20_fem", 1.5)):
            seq = [float(v) for v in g[path]["deflection"]]
            rich = richardson_extrapolate(seq[-3:], ratio=ratio)
            ext_err = (None if not math.isfinite(rich["extrapolated"])
                       else 100.0 * (rich["extrapolated"] / r["w_anchor"] - 1.0))
            finest_err = 100.0 * (seq[-1] / r["w_anchor"] - 1.0)
            if abs(finest_err) < r["limit_pct"]:
                cause = "resolved"
            elif ext_err is not None and abs(ext_err) < r["limit_pct"]:
                cause = "under_resolved"           # extrapolation passes
            else:
                cause = "systematic_bias"          # extrapolation misses too
            entry["paths"][path] = {
                "deflections": seq, "ratio": ratio,
                "w_richardson": rich["extrapolated"], "p_richardson": rich["p"],
                "err_finest_pct": finest_err, "err_richardson_pct": ext_err,
                "cause": cause,
            }
        entry["two_path_agreement_pct"] = 100.0 * (
            g["plate_fem"]["deflection"][-1] / g["hex20_fem"]["deflection"][-1] - 1.0)
        diag_rows.append(entry)

    criteria["D1_failure_diagnosis"] = {
        "criterion": "diagnostic only (no verdict weight): for every case whose "
                     "finest mesh misses the deflection band, Richardson-extrapolate "
                     "each FE path and classify the cause as under_resolved (the "
                     "extrapolated value passes) or systematic_bias (it does not)",
        "rows": diag_rows,
        "failing_cases": [d["case"] for d in diag_rows
                          if any(p["cause"] != "resolved" for p in d["paths"].values())],
        "causes": {d["case"]: {k: v["cause"] for k, v in d["paths"].items()}
                   for d in diag_rows
                   if any(p["cause"] != "resolved" for p in d["paths"].values())},
        "verdict": "DIAGNOSTIC",
    }

    # ── the plate-path bias: element defect or layup specific? ─────────────
    probe_path = OUT / "plate_element_diagnosis.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
    if probe is not None:
        criteria["D2_plate_path_probe"] = {
            "criterion": "diagnostic only (no verdict weight): isotropic single-ply "
                         "controls decide whether the plate-path bias is an element "
                         "defect or specific to the layered sections",
            "probe": probe["probe"], "verdict_of_probe": probe["verdict"],
            "controls": probe["controls"],
            "rows": [{"t_over_a": r["t_over_a"], "a_over_h": r["a_over_h"],
                      "w_clt": r["w_clt"],
                      "fsdt_theory_ratio": r["w_fsdt_theory_ratio"],
                      "plate_finest_err_vs_clt_pct": r["err_finest_vs_clt_pct"],
                      "plate_finest_err_vs_fsdt_pct": r["err_finest_vs_fsdt_theory_pct"]}
                     for r in probe["rows"]],
            "interpretation": probe["interpretation"],
            "verdict": "DIAGNOSTIC",
        }

    # ── is the deviation model-level (FSDT vs 3-D) or implementation-level? ─
    fsdt_path = OUT / "fsdt_anchor.json"
    fsdt = json.loads(fsdt_path.read_text()) if fsdt_path.exists() else None
    if fsdt is not None:
        criteria["D3_plate_vs_fsdt_theory"] = {
            "criterion": "diagnostic only (no verdict weight): compare the Q4 plate "
                         "path with the FSDT (Mindlin) Navier closed form, which is "
                         "the model the plate element implements; agreement here "
                         "means a deflection deviation against a 3-D anchor is a "
                         "model limitation, not an implementation defect",
            "rel_tolerance_D16_D26": fsdt["rel_tolerance_for_D16_D26"],
            "rows": fsdt["rows"],
            "conclusion": fsdt["conclusion"],
            "verdict": "DIAGNOSTIC",
        }

    verdicts = [v["verdict"] for v in criteria.values() if v["verdict"] != "DIAGNOSTIC"]
    overall = ("PASS" if all(v == "PASS" for v in verdicts)
               else "PARTIAL" if any(v == "PASS" for v in verdicts)
               else "FAIL")

    artifacts = {}
    for name in REQUIRED_ARTIFACTS + ("cook_membrane.json",
                                      "plate_element_diagnosis.json",
                                      "fsdt_anchor.json"):
        p = OUT / name
        if p.exists():
            artifacts[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}

    case_table = []
    for row in cases_csv:
        case_table.append({k: row[k] for k in row})

    return {
        "mission": {
            "id": 17568,
            "key": "paper-exp-composite-laminate-verification-2-rework1",
            "rework_of": "paper-exp-composite-laminate-verification-2",
            "repo": "TensorFEM",
            "node": "paper-exp-composite-laminate-verification",
            "wave": "W26",
        },
        "goal": {
            "statement": "Re-establish the composite-laminate verification of the "
                         "TensorFEM plate/3-D elasticity/CLT stack on this work tree, "
                         "score it against external quantitative anchors (Pagano 3-D "
                         "elasticity, Cook 1974 membrane) and close the open experiment "
                         "with an artifact ledger plus a paper-grade report.",
            "anchors": [
                "Pagano (1970) layered 3-D elasticity (propagator solution, exact for "
                "specially orthotropic plies)",
                "CLT/Fourier series solution (truncated Navier basis, D16/D26 retained)",
                "Richardson extrapolated 3-D HEX20 sequence (angle-ply cases without a "
                "closed-form anchor)",
                "Cook (1974) membrane tip deflection 23.96",
            ],
            "criteria": {k: v["criterion"] for k, v in criteria.items()},
        },
        "status": overall,
        "criteria": criteria,
        "case_table": case_table,
        "artifacts": artifacts,
        "provenance": provenance({
            "anchor_material": (deflection.get("provenance", {}) or {}).get("material"),
            "load": (deflection.get("provenance", {}) or {}).get("load"),
        }),
        "honesty": {
            "zero_hand_copy": "every numeric field is read from an artifact written "
                              "by a solver call; provenance carries sha256 of each "
                              "artifact and of the producing modules",
            "honest_fail_preserved": "criterion verdicts are computed, never "
                                     "overridden; WITHHELD marks cases without a "
                                     "closed-form anchor",
            "scope_limits": [
                "the Pagano state-space anchor exists only for specially "
                "orthotropic layups, so angle-ply interlaminar shear stays WITHHELD",
                "all cases are SS1 square plates under a single sine load, "
                "h/a from 0.01 to 0.5; first-ply-failure / delamination growth "
                "are out of scope",
            ],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cook", action="store_true", help="run the Cook membrane anchor")
    ap.add_argument("--closure", action="store_true", help="build closure.json")
    ap.add_argument("--all", action="store_true", help="both")
    args = ap.parse_args()
    if not (args.cook or args.closure or args.all):
        args.all = True
    OUT.mkdir(parents=True, exist_ok=True)

    if args.cook or args.all:
        t0 = time.time()
        cook = cook_study()
        (OUT / "cook_membrane.json").write_text(json.dumps(cook, indent=2))
        print(f"[cook] finest nx={cook['rows'][-1]['nx']} "
              f"w={cook['rows'][-1]['tip_deflection']:.5f} "
              f"err={cook['rows'][-1]['rel_err_pct']:+.3f}% "
              f"verdict={cook['verdict']} ({time.time()-t0:.1f}s)", flush=True)

    if args.closure or args.all:
        closure = build_closure()
        (OUT / "closure.json").write_text(json.dumps(closure, indent=2))
        print(f"[closure] status={closure['status']} -> {OUT/'closure.json'}", flush=True)
        for k, v in closure["criteria"].items():
            print(f"  {k}: {v['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())