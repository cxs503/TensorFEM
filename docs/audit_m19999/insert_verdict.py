#!/usr/bin/env python3
"""Board-audit verdict for m19999 (CASE-FEM-CANTILEVER-TIP, TensorFEM).

Independence: every number in `evidence` is read from an audited artifact copy,
produced by an audit-side re-solve on main's src/tensorfem, re-derived by the
auditor from first principles, or read straight out of the platform database.
Pointers are relative to this worktree (/root/tp-agent/work/m20061) so the gate
can re-open them; the audited mission's originals are recorded with the commit
that introduced them (5504271e on agent/m19999).

Level 3 (系统集成). L4 withheld: tensor_benchmark_cases has no referee row for
this anchor (0 of 121 rows) and the delivered evidence JSON carries the parent
revision, so "platform-registered referee + reproducible from the recorded
revision" does not hold. L5 refused: textbook single-anchor verification.

Idempotent: one row per (repo, scope, verdict_by) -> ON CONFLICT update.
Run: python3 docs/audit_m19999/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
REPO = "TensorFEM"
SCOPE = "m19999 CASE-FEM-CANTILEVER-TIP"
MISSION = 20061
BY = "board-audit"
LEVEL = 3

EVIDENCE = {
    "impl_files": (
        "benchmarks/case_fem_cantilever_tip.py:141 section_properties, :155 analytic_reference, "
        ":171 castigliano_integral, :203 beam_eb_level, :233 beam_timoshenko_level, "
        ":258 solid_level, :320 run_study, :786 main  (audited copy: "
        "docs/audit_m19999/audited_case/case_fem_cantilever_tip.py, sha256 2ce2eea9…; "
        "introduced by commit 5504271e on agent/m19999). Solvers reused from main: "
        "src/tensorfem/frame2d.py FrameModel/solve_frame_static, "
        "src/tensorfem/hex20.py:278 solve_hex20, :322 solve_hex20_sparse, :364 structured_hex20_mesh"
    ),
    "entry": (
        "python benchmarks/case_fem_cantilever_tip.py --json (TensorBench workload, doc:786) / "
        "--write-docs (docs page + 3 figures); audited copy staged at "
        "artifacts/m19999_audit/repro/benchmarks/case_fem_cantilever_tip.py"
    ),
    "run_artifacts": (
        "docs/audit_m19999/audited_case/case_fem_cantilever_tip_evidence.json "
        "(sha256 68816c133af52b06229ffd4dfd507faa04df90b61ea77bddbfa15f6da1090a14; run_id "
        "CASE-FEM-CANTILEVER-TIP-20261114T075225Z; original at "
        "/root/tp-agent/work/m19999 commit 5504271e); figures "
        "docs/audit_m19999/audited_case/assets/{convergence,levels,field_uz}.png "
        "(sha256 in docs/audit_m19999/audited_case/SHA256SUMS.txt); audit-side re-solve "
        "artifacts/m19999_audit/verify_reference_mesh.json + verify_remaining_levels.json; "
        "audit figures docs/audit_m19999/figs/fig1_convergence.png, fig2_got_vs_ref.png"
    ),
    "tests": (
        "tests/test_case_fem_cantilever_tip.py (9 tests, audited copy "
        "docs/audit_m19999/audited_case/test_case_fem_cantilever_tip.py sha256 29fff9bf…); "
        "auditor ran it on main's solvers: "
        "PYTHONPATH=src python3 -m pytest -q artifacts/m19999_audit/repro/tests/"
        "test_case_fem_cantilever_tip.py -> 9 passed, 1 warning in 16.83s"
    ),
    "integration_callers": (
        "platform pipeline: tensorbench_results rows tbr-cf7b79515e584afa91ab709a350447af + "
        "tbr-c7f81b3c76cc4f208580757f8b3d1f9f (mission_id=19999, workload_id "
        "benchmarks/case_fem_cantilever_tip.py, cluster_job_id=2435, run_id "
        "CASE-FEM-CANTILEVER-TIP-20261114T081205Z, spec/result/stdout sha256 recorded); "
        "in-repo caller: tests/test_case_fem_cantilever_tip.py:33 importlib loads the workload "
        "module; no other module imports it (grep -rn case_fem_cantilever_tip over repo)"
    ),
    "benchmark": {
        "declared_referee_table": (
            "ABSENT: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
            "anchor_slug='CASE-FEM-CANTILEVER-TIP' -> 0 rows (table has 121 rows; TensorFEM owns 7: "
            "CASE-FEM-CLAMPED-PLATE-UNIFORM, CASE-FEM-JINTEGRAL-EDGE-CRACK, GAP-FEM-3D-COHESIVE, "
            "GAP-FEM-ELASTOPLASTIC, GAP-FEM-WINKLER, H3-ICE-PLATE-FEM, XVAL-FEM-TPB1976) -> "
            "no platform-registered tolerance to judge against; L4 withheld"
        ),
        "substitute_machine_judgement": (
            "tensorbench_results: metric tip_deflection_solid_reference_m ref=1.000000000e-02 "
            "got=9.964047018716e-03 rel=+0.359530% tolerance_rel=0.02 passed=t (job 2435); "
            "metric tip_deflection_beam_eb_m rel=+1.3e-12 tolerance_rel=0.02 passed=t"
        ),
        "tolerance_provenance": (
            "tensor_case_pool.case_code='FEM-CANTILEVER-TIP': ref_source='解析 PL³/3EI', "
            "tol_policy='<2%', tier='core', status='dispatched'; matches module constant "
            "TOLERANCE=0.02 at benchmarks/case_fem_cantilever_tip.py:113"
        ),
        "got_vs_ref": "audited artifact -0.359547% (auditor re-solve reproduces got bit-identically); <2%? YES; <3%(ecosystem line)? YES",
        "h_ladder_percent": "8x1x1 -2.5139 / 16x1x1 -1.1609 / 16x2x2 -0.8633 / 32x2x2 -0.4015 / 32x4x4(ref) -0.3595 / 64x4x4 -0.1810 (monotone)",
    },
    "repro_cmd": (
        "auditor, on main's src/tensorfem in this worktree: (1) "
        "PYTHONPATH=src OMP_NUM_THREADS=4 python3 artifacts/m19999_audit/"
        "verify_reference_mesh.py -> solid 32x4x4 = 9.964045336499231e-03 m, rel diff vs artifact "
        "0.000e+00; (2) verify_remaining_levels.py -> 16x1 / 32x2 / 64x4 ladder rows, all three "
        "slenderness points (L/h=100/50/20) and the 2P line reproduced with rel diff 0.000e+00 "
        "(ratio tip(2P)/tip(P)=2.000000000000); (3) --quick full path reproduces 8x1x1 and 16x2x2 "
        "to 13 significant digits. NOT run: single full run_study invocation and --write-docs "
        "(>120 s local budget; queued job cj2457 killed once per-level coverage made it redundant)"
    ),
    "external_anchor": (
        "auditor re-derivation from first principles, no import of the audited module "
        "(artifacts/m19999_audit/recheck_analytic.py, .json): I=bh^3/12=8.333333333333335e-10 m^4, "
        "EI=166.6667 N·m^2, delta_EB=PL^3/3EI=9.999999999999998e-03 m, kappa=5/6, "
        "G=E/(2(1+nu))=7.692307692307692e+10 Pa, kappa*A*G=6.410256e+06 N, "
        "shear=PL/(kappa*A*G)=7.799999999999999e-07 m (0.0078% of bending), "
        "delta_T=1.000078000000000e-02 m; independent 20001-point composite Simpson on the moment "
        "and shear fields agrees to +3.47e-16 / -7.87e-15. Literature legs: Timoshenko, Strength of "
        "Materials Part I 3rd ed. (1955) ch.4; Blevins (1979) Table 8-1 DOI 10.1115/1.3153712 "
        "(article-level, no digits transcribed)"
    ),
    "findings": {
        "F1_medium_high": (
            "no tensor_benchmark_cases row for the anchor -> the case is machine-judged only through "
            "tensorbench_results with a tolerance taken from tensor_case_pool, not by a registered "
            "benchmark case. Remediation: INSERT the referee row (tolerance '<2%', metric "
            "tip_deflection_solid_reference_m)"
        ),
        "F2_medium_platform_side": (
            "tensor_reports id=18246 (mission 19999) has all six required sections but figures carry "
            "only name+caption, no png_base64. CONTROL EXPERIMENT: the auditor posted its own report "
            "(id=18284) carrying 99700 + 74684 B of base64 in figures, and the stored row still has "
            "no png_base64 (figures->0 ? 'png_base64' = false) while the older id=17486 does hold "
            "224768 B -> the current /tensor/reports ingest drops the image payload, so this is a "
            "platform-side defect, NOT a defect of m19999; remediation moves to the platform service"
        ),
        "F3_low_medium": (
            "evidence JSON implementation.revision=9ad824f70f8356444ac6e7ae90cade14e41dfcfb (= parent "
            "of 5504271e) does not contain the workload file, and tensorbench_results."
            "implementation_revision='unknown' -> 'reproduce from the recorded revision' fails"
        ),
        "F4_low": (
            "artifact got=9.964045336499231e-03 vs ingested row got=9.964047018716e-03, rel 1.7e-9 "
            "drift across runs, undocumented (both pass; auditor re-solve matches the artifact bit "
            "for bit, so the drift is in the cj2435 environment)"
        ),
        "F5_low_legacy": (
            "benchmarks/cantilever_tip_artifacts/results.json (m15330/m17033 vintage) points at "
            "convergence.png/comparison.png that do not exist in the repo, and its repro_cmd.sh runs "
            "the older benchmarks/cantilever_tip_deflection.py, not the anchor workload"
        ),
    },
    "checks_not_run": [
        "single full run_study invocation (all 11 ladder levels in one command) -- not run: >120 s "
        "local short-test budget; queued cluster job cj2457 killed after per-level bit-identical "
        "coverage made it redundant",
        "--write-docs path (make_figures + render_docs regeneration) -- not run",
        "pixel-level inspection of the three delivered PNGs -- not done",
        "large-deflection regime (elastica) -- outside the declared linear case, not run",
        "comparison against an independent FE code (Abaqus/CalculiX) -- not run, no second solid "
        "solver in the estate",
    ],
    "level_rationale": (
        "L3 confirmed: real input->compute->output path on the repo's frame2d/hex20 solvers, 9/9 "
        "pytest gate green on an independent checkout, every numeric row of the evidence payload "
        "reproduced bit-identically on main's solvers, and the workload is integrated into the "
        "platform TensorBench pipeline (2 ingested judgement rows, cluster job 2435, spec/result/"
        "stdout sha256). L4 withheld: the designated referee table has no row for this anchor (F1) "
        "and the delivered provenance revision predates the workload (F3), so the L4 requirements "
        "'platform-registered referee' and 'reproducible from the recorded revision' are unmet -- "
        "consistent with precedent m19997 (0 referee rows -> L4 refused). L5 refused: textbook "
        "single-anchor beam verification, no experimental or research-engineering substrate."
    ),
    "disposition": (
        "request changes (small remediation round, reopened): (1) register the "
        "tensor_benchmark_cases referee row for CASE-FEM-CANTILEVER-TIP; (2) re-publish report "
        "18246 with embedded png_base64 figures; (3) write implementation.revision=5504271e into "
        "the evidence artifact and carry the real revision into tensorbench_results; F4/F5 advisory."
    ),
    "verdict_doc": "docs/audit_m19999/AUDIT-m19999-CASE-FEM-CANTILEVER-TIP.md",
    "audit_artifacts": [
        "docs/audit_m19999/AUDIT-m19999-CASE-FEM-CANTILEVER-TIP.md",
        "docs/audit_m19999/audited_case/ (SHA256SUMS.txt + 4 text files + 3 png)",
        "docs/audit_m19999/make_figures.py",
        "docs/audit_m19999/figs/fig1_convergence.png",
        "docs/audit_m19999/figs/fig2_got_vs_ref.png",
        "artifacts/m19999_audit/recheck_analytic.py",
        "artifacts/m19999_audit/recheck_analytic.json",
        "artifacts/m19999_audit/verify_reference_mesh.py",
        "artifacts/m19999_audit/verify_reference_mesh.json",
        "artifacts/m19999_audit/verify_remaining_levels.py",
        "artifacts/m19999_audit/verify_remaining_levels.json",
        "artifacts/m19999_audit/repro/benchmarks/case_fem_cantilever_tip.py",
        "artifacts/m19999_audit/repro/tests/test_case_fem_cantilever_tip.py",
        "docs/audit_m19999/insert_verdict.py",
        "docs/audit_m19999/post_report.py",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    payload = json.dumps(EVIDENCE, ensure_ascii=False)
    if args.dry_run:
        print(json.dumps(EVIDENCE, ensure_ascii=False, indent=1))
        return 0
    sql = f"""
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('{REPO}', '{SCOPE}', {MISSION}, {LEVEL}, $json${payload}$json$::jsonb, '{BY}')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id,
      level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      created_at = now()
RETURNING id, gate_state, gate_level;
SELECT count(*) AS referee_rows FROM tensor_benchmark_cases
 WHERE anchor_slug='CASE-FEM-CANTILEVER-TIP';
"""
    res = subprocess.run(["psql", DSN, "-At", "-c", sql], capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr[-800:])
        return res.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())