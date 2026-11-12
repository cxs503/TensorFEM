#!/usr/bin/env python3
"""Write the board-audit verdict of mission m16138 into tabletensor_maturity_verdicts.

All numbers come from ``runs/m16196_audit/audit_summary.json`` (produced by
``scripts/audit/verify_m16138.py``) -- nothing is retyped by hand.

    python scripts/audit/insert_verdict_m16138.py --dry-run
    python scripts/audit/insert_verdict_m16138.py --write
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "runs/m16196_audit/audit_summary.json"
REPRO_DIR = REPO / "runs/m16196_audit/repro"

REPO_SLUG = "TensorFEM"
SCOPE = "m16138 CASE-FEM-MORLEY-SKEW-PLATE"
MISSION = 16196          # this audit mission (the gate resolves worktrees from it)
SUBJECT = 16138
LEVEL = 2
VERDICT_BY = "board-audit"

DSN = None
for line in open("/root/tensor-ecosystem/TensorPlatform/.env"):
    if line.startswith("DATABASE_URL="):
        DSN = line.split("=", 1)[1].strip().replace("postgresql+psycopg://", "postgresql://")
        break


def build_evidence(s: dict) -> dict:
    snap = "runs/m16196_audit/_source_m16138"
    case = f"{snap}/benchmarks/verified/fem_morley_skew_plate"
    gates = {g["id"]: g for g in s["gates_recomputed"]}
    failed = [k for k, g in gates.items() if not g["passed_recomputed"]]
    repro = sorted(str(p.relative_to(REPO)) for p in REPRO_DIR.rglob("*") if p.is_file()) if REPRO_DIR.exists() else []
    return {
        "anchor_slug": s["anchor"],
        "subject_mission": SUBJECT,
        "audited_commit": s["subject_head_commit"],
        "audited_commit_note": "HEAD of the subject worktree IS the base commit df250c73: the delivery was never "
                               "committed (tensor_dev_missions#16138 result.evidence.head_commit=null, branch=null)",
        "dirty_paths": s["subject_dirty_paths"],
        "impl_files": [
            f"{snap}/src/tensorfem/skew_plate.py",
            f"{snap}/src/tensorfem/plate.py",
            f"{snap}/docs/morley-skew-plate-validation.md",
            f"{snap}/docs/mitc-skew-invariance.md",
        ],
        "impl_lines": [
            f"{snap}/src/tensorfem/skew_plate.py:121 rhombus_mesh / :437 ritz_sine_solution / "
            "solve_skew_plate / mitc_skew_sweep (552 lines; archived snapshot, subject worktree "
            "/root/tp-agent/work/m16138 has the identical file, untracked)",
            f"{snap}/src/tensorfem/plate.py:q4_mindlin_stiffness MITC branch -> covariant shear rows "
            "(diff +61/-25; the pre-fix physical-row sampling is documented in "
            "artifacts/prefix_mitc_skew_defect.json: parasitic/bending energy 0.25 (80 deg) -> 24x (30 deg) -> 60x (20 deg))",
            "NOTE: neither file is in git (git status: ' M src/tensorfem/plate.py', '?? src/tensorfem/skew_plate.py') "
            "-> the TensorFEM product repo has no such capability",
        ],
        "entry": f"{case}/run.py",
        "entry_cmd": "cd <TensorFEM> && PYTHONPATH=src python benchmarks/verified/fem_morley_skew_plate/run.py"
                     "  (--quick for the short ladder; no --out-dir, artifacts land next to run.py)",
        "run_artifacts": [
            f"{case}/result.json",
            f"{case}/gates_verified.json",
            f"{case}/summary.md",
            f"{case}/convergence.png",
            f"{case}/deflection_field.png",
            f"{case}/artifacts/prefix_mitc_skew_defect.json",
            "runs/m16196_audit/audit_summary.json",
            "runs/m16196_audit/independent_biharmonic.json",
            "runs/m16196_audit/corrected_ritz.json",
            "docs/board_audit/m16196/VERDICT_m16138.md",
        ] + repro,
        "run_artifacts_hashes": {k: v["sha256"][:16] + "…" for k, v in s["artifacts"].items()},
        "repro_cmd": "auditor: tar of the subject tree -> cluster job cj1263 (loong64): "
                     "PYTHONPATH=src python benchmarks/verified/fem_morley_skew_plate/run.py ; "
                     "artifacts fetched into runs/m16196_audit/repro/ (see runs/m16196_audit/repro/README.md "
                     "for the actual state: reproduced / still queued). Command itself is syntactically "
                     "executable and needs no arguments, no network, no GPU.",
        "tests": [
            "tests/test_plate_mitc_skew_invariance.py (delivered, uncommitted): constant-curvature state on a "
            "parallelogram stays shear-free, 90..20 deg; mission record: 58 passed over 4 test files",
            "tests/test_bending_patch.py (modification of an existing regression test, also uncommitted)",
        ],
        "integration_callers": [],
        "integration_status": "NONE in the product repo: no src/tensorfem/__init__.py re-export, no "
                              "benchmark_registry.py entry, no CI hook; the only callers are this case's own "
                              "run.py and its own test file (self-referential). The MITC covariant fix does touch "
                              "the production element src/tensorfem/plate.py:q4_minn->q4_mindlin_stiffness, i.e. "
                              "L3-grade in substance once it is committed.",
        "benchmark": {
            "registered_case": "ABSENT: SELECT count(*) FROM tensor_benchmark_cases WHERE "
                               "anchor_slug='CASE-FEM-MORLEY-SKEW-PLATE' -> 0 rows (26 other anchors exist). "
                               "The referee named by the task order has no case number, so the only available "
                               "criterion is tensor_case_pool#101 tol_policy '<5% | w*D/(q*a^4) vs Morley'; "
                               "that pool row is still status='dispatched', linked_anchor=NULL, mission_id=16138.",
            "got_vs_ref": {
                "L/h=1000 n=80": {"got_wc_N": 0.400619, "morley_0.408_err_pct": 100 * s["errors_delivered"]["vs_morley_Lh1000"],
                                  "accurate_0.4134_err_pct": 100 * s["errors_delivered"]["vs_accurate_Lh1000"]},
                "L/h=100 n=80": {"got_wc_N": 0.422660, "morley_0.408_err_pct": 100 * s["errors_delivered"]["vs_morley_Lh100"],
                                 "accurate_0.4248_err_pct": 100 * s["errors_delivered"]["vs_accurate_Lh100"]},
                "L/h=10 n=80": {"got_wc_N": 0.517548, "accurate_0.5177_err_pct": 0.0294},
                "square_control": {"got_alpha": 0.0040966, "exact_0.0040624_err_pct": 0.8418},
                "moment_Lh1000_n80": {"M1_err_pct": -1.95, "M2_err_pct": -4.33},
            },
            "pool_criterion": {"metric": "wc*D/(q*a^4) vs Morley 0.408e-3", "tolerance": 0.05,
                               "finest_Lh1000": round(s["errors_delivered"]["vs_morley_Lh1000"], 6),
                               "pass_literal": s["pool_metric_pass"]},
            "self_gates": {"n_failed": s["n_failed"], "failed": failed,
                           "recomputed_matches_delivered": s["gates_all_value_match"] and s["gates_all_verdict_match"],
                           "delivered_overall_passed": s["delivered_overall_passed"],
                           "note": "the artifact honestly declares passed=false with G5 (3.0917% vs 3%) and "
                                   "G6 (1 non-monotone step) failing; mission.result.checks and the headline "
                                   "report only the passing numbers"},
            "judge_status": "not judged: no tensor_benchmark_cases row; the pool tolerance is met literally "
                            "(1.81% < 5%) but see the independent recomputation below",
        },
        "independent_check": {
            "tool": "scripts/audit/independent_biharmonic.py (oblique 9-point FD, split biharmonic, w=0 and "
                    "laplacian(w)=0 on all four straight edges; NOT finite elements, NOT the delivered code)",
            "self_validation": f"square 90 deg wc_N = 4.062325 vs exact 0.0040624e3 -> "
                               f"{s['independent_fd_square_vs_exact_pct']:.4f}% ; grid 16..128 stable to 5 digits",
            "audit_value_30deg": round(s["independent_fd"][1]["wc_N"], 6),
            "vs_delivered_finest_pct": round(s["independent_fd_vs_delivered_finest_pct"], 1),
            "vs_morley_0.408_pct": round(s["independent_fd_vs_morley_pct"], 1),
            "corrected_ritz_audit_side": json.loads((REPO / "runs/m16196_audit/corrected_ritz.json").read_text())[
                "corrected_ritz_audit_side"]["nb_terms"],
            "interpretation": "the delivered FE ladder (0.4006) and the cited literature (0.408) agree with each "
                              "other, but a validated independent solve of the same nominal problem sits 38% lower "
                              "and a corrected Ritz climbs towards that same lower value from below. Either the "
                              "Mindlin 'soft' support is not the Kirchhoff simply-supported condition on skew "
                              "edges, or the cited normalisation/definition differs. The papers are unreachable "
                              "from this environment, so the audit cannot pick the side -> the '<5% vs Morley' "
                              "claim is NOT an established validation.",
        },
        "delivered_oracle_defect": {
            "file": "src/tensorfem/skew_plate.py:471-473 (G matrix, row 0)",
            "delivered_row0": s["ritz_diagnostic_defect"]["delivered_G_row0"],
            "correct_row0": s["ritz_diagnostic_defect"]["correct_G_row0"],
            "dropped_term_magnitude_at_30deg": {
                "cross_2*p1*p2_over_p1^2": round(s["ritz_diagnostic_defect"]["dropped_cross_over_row0_norm"], 3),
                "eta_eta_p2^2_over_p1^2": round(s["ritz_diagnostic_defect"]["dropped_eta_eta_over_row0_norm"], 3)},
            "symptom": f"delivered Ritz wc_N ladder {s['ritz_diagnostic_defect']['delivered_wc_N_ladder']} for "
                       f"n_terms {s['ritz_diagnostic_defect']['delivered_ritz_terms']} rises monotonically ABOVE "
                       "the audit value 0.2547 -> not the same operator; doc calls it 'slow convergence / lower "
                       "bound only', the real cause is the missing terms",
        },
        "external_anchor": {
            "citation": "L. S. D. Morley, 'Bending of a Simply Supported Rhombic Plate under Uniform Normal "
                        "Loading', Q. Jl. Mech. Appl. Math. 15(4) (1962) 413-426 / Skew Plates and Structures "
                        "(Pergamon 1963); values as tabulated in Grbac & Ribaric 2022, Zbornik radova "
                        "Gradevinskog fakulteta 25(1) 237-246, sect. 2.2 p.240 and Table 4",
            "delivered_values": {"wc_N": 0.408, "M1_N": 1.91, "M2_N": 1.08,
                                 "accurate": {"L/h=1000": 0.4134, "L/h=100": 0.4248, "L/h=10": 0.5177}},
            "verification": "NOT independently verified: the 1962/1963 original and the 2022 secondary source "
                            "are both unreachable from this environment (the delivery itself reports "
                            "archive.org/sciencedirect 403); the delivered numbers are transcribed from a "
                            "secondary table, and the audit-side solve disagrees with them by 37.6%",
        },
        "findings": [
            "F1 HIGH delivery never committed (head_commit=null, branch=null, 7 dirty paths) -> nothing in the "
            "product repo, auto-push would push an empty diff; audit archived a snapshot in its own branch",
            "F2 HIGH referee absent (0 tensor_benchmark_cases rows) and pool row #101 stale "
            "(status=dispatched, linked_anchor=NULL) -> 'succeeded' came from auto-approve, not from a criterion",
            "F3 HIGH external anchor not confirmed: audit-side FD (square-exact) 0.25471 vs delivered 0.40062 "
            "(+57.3%) and vs cited 0.408 (-37.6%); SS definition / normalisation unresolved",
            "F4 MEDIUM self-declared gates G5 (3.0917% > 3%) and G6 (non-monotone) fail inside the artifact but "
            "do not appear in mission.result.checks or the headline; M1/M2 quoted from the L/h=100 rung "
            "(0.39%/0.70%) instead of the primary L/h=1000 rung (1.95%/4.33%)",
            "F5 MEDIUM delivered 'independent oracle' ritz_sine_solution has a broken geometric map (row 0 of G "
            "drops 2*p1*p2 and p2^2; 3.46x / 3.00x at 30 deg) -> its 0.2764 is above the true value, so the "
            "delivery has no independent cross-check at all",
            "F6 LOW convergence evidence weak: non-monotone 0.38158 -> 0.37472 at n=10->20, finest rung still "
            "-3.1% from the dense-mesh value, no Richardson extrapolation (declared in limitations)",
            "F7 LOW the MITC rung is worse than SRI on the same mesh (0.4081 vs 0.4194 at n=40); gate G7 passes "
            "on a 2.69% agreement but the report does not say the two treatments answer different questions",
        ],
        "not_run": [
            "first-hand check of the Morley 1962/1963 and Grbac & Ribaric 2022 tables (papers unreachable)",
            "third element family (DKT / T-spline / triangular) recomputation of the 30 deg plate",
            "attribution of the 38% / 57% gap (soft Mindlin support vs citation convention) - needs the paper or "
            "a second boundary-condition implementation",
            "full pytest of the subject worktree on the audit host (cluster job cj1263 covers the benchmark "
            "rerun only; see runs/m16196_audit/repro/)",
        ],
        "level_rationale": "L2: real code with input->compute->output (552-line skew_plate module + a genuine "
                           "covariant-MITC fix of the production plate element, 6 non-empty artifacts, entry "
                           "runs in 179 s), L1 supported by the MITC skew-invariance test (parasitic shear "
                           "energy 2e-13 over 90..20 deg). No L3: nothing is committed, no __init__/"
                           "benchmark_registry/CI caller, the only callers are the case's own runner and test. "
                           "No L4: the machine referee has no case number, 2 of the delivery's own 9 gates "
                           "fail, the finest mesh is admittedly unconverged, and an audit-side validated "
                           "solution of the same nominal problem is 38% away from the cited anchor - a "
                           "converging number that agrees with a citation is not a validated model while the "
                           "definitions disagree. No L5.",
        "audit_report": "docs/board_audit/m16196/VERDICT_m16138.md",
        "disposition": "request changes: (1) commit skew_plate.py + the covariant MITC fix + both tests + the "
                       "benchmark dir into agent/m16138 (snapshot ready in runs/m16196_audit/_source_m16138/) "
                       "and register the callers; (2) fix ritz_sine_solution's G row 0 and re-run the diagnostic; "
                       "(3) resolve the simply-supported definition / reference normalisation before any headline "
                       "may say '<5% PASS'; (4) register tensor_benchmark_cases rows and write back pool row #101; "
                       "(5) mission self-report must carry G5/G6 FAIL and the rung it quotes.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    s = json.loads(SUMMARY.read_text())
    ev = build_evidence(s)
    if not a.write:
        print(json.dumps(ev, ensure_ascii=False, indent=1)[:3000])
        print("...dry-run only, no DB write")
        return
    import psycopg
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET mission_id = EXCLUDED.mission_id, "
            "level = EXCLUDED.level, evidence = EXCLUDED.evidence, created_at = now(), "
            "gate_state = 'pending', gate_level = NULL, gate_note = NULL "
            "RETURNING id, level, gate_state, gate_level",
            (REPO_SLUG, SCOPE, MISSION, LEVEL, json.dumps(ev), VERDICT_BY),
        )
        print("verdict row:", cur.fetchone())
        conn.commit()


if __name__ == "__main__":
    main()