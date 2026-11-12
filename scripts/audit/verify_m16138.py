#!/usr/bin/env python3
"""Board-audit verification of mission m16138 (CASE-FEM-MORLEY-SKEW-PLATE).

Everything is measured from the delivered artifacts / archived snapshot, nothing
is typed by hand: the nine delivered gates are recomputed from ``result.json``,
``summary.md`` is re-derived from ``result.json`` and string-compared, artifact
hashes are computed, and the delivered anchor is compared with an audit-side
independent biharmonic solve.

    python scripts/audit/verify_m16138.py   -> runs/m16196_audit/audit_summary.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAP = REPO / "runs/m16196_audit/_source_m16138"
CASE = SNAP / "benchmarks/verified/fem_morley_skew_plate"
SUBJECT = Path("/root/tp-agent/work/m16138")
OUT = REPO / "runs/m16196_audit"
sys.path.insert(0, str(REPO / "scripts/audit"))

MORLEY = {"wc_N": 0.408, "M1_N": 1.91, "M2_N": 1.08}
ACC = {1000.0: (0.4134, 1.922, 1.106), 100.0: (0.4248, 1.956, 1.143), 10.0: (0.5177, 2.064, 1.205)}
POOL_TOL = 0.05          # tensor_case_pool #101 tol_policy: <5% on w*D/(q a^4) vs Morley


def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def rel(x, ref):
    return abs(x - ref) / abs(ref)


def recompute_gates(d: dict) -> list[dict]:
    """Rebuild the delivered gate table from the raw numbers in result.json."""
    thin = d["thin_cases"]
    fin = {float(k): v[-1] for k, v in thin.items()}
    sq = d["square_control"]["alpha"]
    thick = d["thick_cases"]
    thick_n = {c["n"]: c["wc_N"] for c in thick}
    sri40 = next(c for c in thin["100.0"] if c["n"] == 40)
    mitc40 = next(c for c in d["mitc_cases"] if c["n"] == 40)
    sweep = max(abs(r["parasitic_shear_energy_ratio"]) for r in d["mitc_skew_sweep"])
    steps = sum(1 for a, b in zip(thin["1000.0"], thin["1000.0"][1:]) if b["wc_N"] <= a["wc_N"])
    m1, m2 = fin[1000.0]["M1_N"], fin[1000.0]["M2_N"]
    return [
        {"id": "G1_square_pipeline_control", "value": rel(sq, 0.0040624), "threshold": 0.01},
        {"id": "G2_thick_cross_check_Lh10", "value": rel(thick_n[max(thick_n)], 0.5177), "threshold": 0.02},
        {"id": "G3_morley_thin_Lh1000", "value": rel(fin[1000.0]["wc_N"], MORLEY["wc_N"]), "threshold": 0.05},
        {"id": "G4_morley_thin_Lh100", "value": rel(fin[100.0]["wc_N"], MORLEY["wc_N"]), "threshold": 0.05},
        {"id": "G5_accurate_Lh1000", "value": rel(fin[1000.0]["wc_N"], ACC[1000.0][0]), "threshold": 0.03},
        {"id": "G6_monotone_convergence", "value": float(steps), "threshold": 0.5},
        {"id": "G7_shear_field_independence",
         "value": abs(mitc40["wc_N"] - sri40["wc_N"]) / sri40["wc_N"], "threshold": 0.03},
        {"id": "G8_moments_Lh1000",
         "value": max(rel(m1, ACC[1000.0][1]), rel(m2, ACC[1000.0][2])), "threshold": 0.10},
        {"id": "G9_mitc_skew_invariance", "value": sweep, "threshold": 1e-9},
    ]


def main() -> dict:
    res = json.loads((CASE / "result.json").read_text())
    delivered = json.loads((CASE / "gates_verified.json").read_text())
    delivered_rows = {g["id"]: g for g in delivered["gates"]}
    mine = recompute_gates(res)
    for row in mine:
        drow = delivered_rows[row["id"]]
        row["passed_recomputed"] = row["value"] <= row["threshold"]
        row["delivered_value"] = drow["value"]
        row["delivered_passed"] = drow["passed"]
        row["value_matches_delivered"] = abs(row["value"] - drow["value"]) <= 1e-9 * max(1.0, abs(drow["value"]))
        row["passed_matches_delivered"] = row["passed_recomputed"] == drow["passed"]

    # summary.md must be re-derivable from result.json (no hand-typed numbers)
    md = (CASE / "summary.md").read_text()
    table_rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "---" not in ln][1:]
    md_ok = 0
    for ln in table_rows:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        ratio, n = float(cells[0]), int(cells[1])
        row = next(c for c in res["thin_cases"][str(ratio)] if c["n"] == n)
        if (abs(float(cells[4]) - row["wc_N"]) < 5e-6 and abs(float(cells[5]) - row["M1_N"]) < 5e-6
                and abs(float(cells[6]) - row["M2_N"]) < 5e-6):
            md_ok += 1

    # audit-side independent solve (validated on the exact square coefficient)
    from independent_biharmonic import solve
    fd = []
    L, E, nu, q = 100.0, 10.92, 0.3, 1.0
    for lh in (1000.0, 100.0):
        h = L / lh
        D = E * h ** 3 / (12 * (1 - nu ** 2))
        for ang in (90.0, 30.0):
            wc, _ = solve(128, ang, L, D, q)
            fd.append({"skew_deg": ang, "L_over_h": lh, "wc_N": wc / (1e-3 * q * L ** 4 / D)})
    fd30 = next(r["wc_N"] for r in fd if r["skew_deg"] == 30.0 and r["L_over_h"] == 1000.0)
    fd90 = next(r["wc_N"] for r in fd if r["skew_deg"] == 90.0 and r["L_over_h"] == 1000.0)

    # the delivered Rayleigh-Ritz "independent oracle": dropped terms in row 0 of G
    alpha = math.radians(30.0)
    p1, p2, p3 = 1.0 / L, -math.cos(alpha) / (L * math.sin(alpha)), 1.0 / (L * math.sin(alpha))
    ritz = {
        "delivered_G_row0": [p1 ** 2, 0.0, 0.0],
        "correct_G_row0": [p1 ** 2, 2 * p1 * p2, p2 ** 2],
        "dropped_cross_over_row0_norm": abs(2 * p1 * p2) / (p1 ** 2),
        "dropped_eta_eta_over_row0_norm": abs(p2 ** 2) / (p1 ** 2),
        "delivered_wc_N_ladder": [r["wc_N"] for r in res["ritz_diagnostic"]],
        "delivered_ritz_terms": [r["n_terms"] for r in res["ritz_diagnostic"]],
    }

    files = sorted(p for p in SNAP.rglob("*") if p.is_file())
    arts = {str(p.relative_to(REPO)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}

    commit = None
    try:
        import subprocess
        commit = subprocess.run(["git", "-C", str(SUBJECT), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(SUBJECT), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip().splitlines()
    except Exception:
        dirty = []
    summary = {
        "anchor": "CASE-FEM-MORLEY-SKEW-PLATE",
        "subject_mission": 16138,
        "audit_mission": 16196,
        "subject_worktree": str(SUBJECT),
        "subject_head_commit": commit,
        "subject_dirty_paths": dirty,
        "gates_recomputed": mine,
        "gates_all_value_match": all(r["value_matches_delivered"] for r in mine),
        "gates_all_verdict_match": all(r["passed_matches_delivered"] for r in mine),
        "delivered_overall_passed": delivered["passed"],
        "delivered_failed_gates": delivered["failed_gates"],
        "n_failed": sum(1 for r in mine if not r["passed_recomputed"]),
        "summary_md_rows": len(table_rows),
        "summary_md_rows_consistent": md_ok,
        "case_pool_tolerance": POOL_TOL,
        "pool_metric": "wc*D/(q*a^4) vs Morley 0.408e-3; finest L/h=1000 mesh -> %.6f" % (
            rel(next(c for c in res["thin_cases"]["1000.0"] if c["n"] == 80)["wc_N"], MORLEY["wc_N"])),
        "pool_metric_pass": rel(next(c for c in res["thin_cases"]["1000.0"] if c["n"] == 80)["wc_N"],
                                MORLEY["wc_N"]) < POOL_TOL,
        "errors_delivered": res["errors"],
        "independent_fd": fd,
        "independent_fd_square_vs_exact_pct": 100 * (fd90 / 4.0624 - 1),
        "independent_fd_vs_delivered_finest_pct": 100 * (
            next(c for c in res["thin_cases"]["1000.0"] if c["n"] == 80)["wc_N"] / fd30 - 1),
        "independent_fd_vs_morley_pct": 100 * (fd30 / MORLEY["wc_N"] - 1),
        "ritz_diagnostic_defect": ritz,
        "artifacts": arts,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "artifacts"}, indent=1,
                     ensure_ascii=False)[:6000])
    return summary


if __name__ == "__main__":
    main()