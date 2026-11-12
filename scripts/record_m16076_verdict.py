#!/usr/bin/env python3
"""Record the board-audit verdict for m16076 (CASE-FEM-PINCHED-CYLINDER) into
tensor_maturity_verdicts.

Every pointer below exists in this audit worktree (TensorFEM @ agent/m16175) or in
the audited delivery (commit 5dbf8c83, merged 43745cdb).  Idempotent:
ON CONFLICT (repo, scope, verdict_by) DO UPDATE, gate_state reset to 'pending'.

Usage: python3 scripts/record_m16076_verdict.py [--dry-run]
"""
from __future__ import annotations

import json
import subprocess
import sys

REPO = "TensorFEM"
SCOPE = "m16076 CASE-FEM-PINCHED-CYLINDER"
AUDIT_MID = 16175
LEVEL = 4
VERDICT_BY = "board-audit"

EVIDENCE = {
    "impl_files": [
        "src/tensorfem/pinched_cylinder.py:27 pinched_cylinder_model (octant, x in [0,300], theta in [0,pi/2], P/4 corner load)",
        "src/tensorfem/pinched_cylinder.py:77 solve_pinched_cylinder_linear (dense production path)",
        "src/tensorfem/pinched_cylinder.py:92 pinched_cylinder_sparse_system (new CSR assembly)",
        "src/tensorfem/pinched_cylinder.py:117 solve_pinched_cylinder_sparse (SuperLU, reaches N=64 / 25350 DOF)",
        "src/tensorfem/cylindrical_shell4.py (reused production Reissner-Mindlin cylindrical Shell4 element)",
    ],
    "entry": "benchmarks/verified/fem_pinched_cylinder/run.py",
    "run_artifacts": [
        "benchmarks/verified/fem_pinched_cylinder/case_result.json",
        "benchmarks/verified/fem_pinched_cylinder/case_gates.json",
        "docs/board_audit/m16175/artifacts_repro/repro.log (audit re-run, 540.6 s wall, 8/8 rungs bit-identical)",
        "docs/board_audit/m16175/artifacts_repro/case_result_rerun.json",
        "docs/board_audit/m16175/artifacts_repro/case_gates_rerun.json",
        "docs/board_audit/m16175/artifacts_repro/figures/case_convergence.png",
        "docs/board_audit/m16175/artifacts_repro/figures/case_field.png",
        "docs/board_audit/m16175/artifacts_repro/figures/case_sensitivity.png",
    ],
    "tests": [
        "tests/test_pinched_cylinder.py (auditor ran: PYTHONPATH=src pytest tests/test_pinched_cylinder.py -q -> 6 passed in 55.23 s)",
        "tests/test_pinched_cylinder.py::test_case_pool_pinched_cylinder_16x16_within_five_percent",
        "tests/test_pinched_cylinder.py::test_sparse_path_reproduces_the_production_dense_solution",
        "tests/test_pinched_cylinder.py::test_axial_end_clamp_stiffens_the_diaphragm_model",
    ],
    "integration_callers": [
        "src/tensorfem/__init__.py:107-109 (public re-export of pinched_cylinder_model / solve_pinched_cylinder_linear / solve_pinched_cylinder_nonlinear)",
        "src/tensorfem/__init__.py:324-325 (__all__)",
        "src/tensorfem/benchmark_cases.py:107-109 (case runner dispatch 'pinched_cylinder' -> solve_pinched_cylinder_linear)",
        "src/tensorfem/cylindrical_shell4_diagnostics.py:147 pinched_cylinder_ladder (diagnostics API consumer)",
    ],
    "benchmark": {
        "registered_row_for_anchor": 0,
        "note": "tensor_benchmark_cases has 0 rows matching CASE-FEM-PINCHED-CYLINDER (checked anchor_slug/case_name/metric/ref_source, ILIKE '%pinch%'); "
                "tensor_case_pool holds case_code FEM-PINCHED-CYLINDER repo TensorFEM mission_id 16076 with status 'dispatched' (never flipped to a verified/passed state) "
                "-> the board machine judge has no registered row; the numbers are therefore carried here explicitly.",
        "criterion": "load-point radial deflection at the 16x16 octant mesh within <5% of MacNeal-Harder 1.8248e-5 in (mission m16076 goal + tensor_case_pool tol_policy)",
        "ref_vs_got": [
            {"mesh": "4x4", "got_in": -1.772659884661117e-05, "ref_in": -1.8248e-05, "err_pct": 2.857, "tol_pct": 5.0, "pass": True},
            {"mesh": "8x8", "got_in": -1.7636652834588284e-05, "ref_in": -1.8248e-05, "err_pct": 3.350, "tol_pct": 5.0, "pass": True},
            {"mesh": "12x12", "got_in": -1.806824e-05, "ref_in": -1.8248e-05, "err_pct": 0.985, "tol_pct": 5.0, "pass": True},
            {"mesh": "16x16 (judge point)", "got_in": -1.827849365234856e-05, "ref_in": -1.8248e-05, "err_pct": 0.167, "tol_pct": 5.0, "pass": True},
            {"mesh": "24x24", "got_in": -1.848067e-05, "ref_in": -1.8248e-05, "err_pct": 1.275, "tol_pct": 5.0, "pass": True},
            {"mesh": "32x32", "got_in": -1.857942e-05, "ref_in": -1.8248e-05, "err_pct": 1.816, "tol_pct": 5.0, "pass": True},
            {"mesh": "48x48", "got_in": -1.868019e-05, "ref_in": -1.8248e-05, "err_pct": 2.368, "tol_pct": 5.0, "pass": True},
            {"mesh": "64x64", "got_in": -1.873469e-05, "ref_in": -1.8248e-05, "err_pct": 2.667, "tol_pct": 5.0, "pass": True},
            {"metric": "extrapolated discrete limit (power law, p=0.97)", "got_in": -1.88937106450656e-05, "ref_in": -1.8248e-05, "err_pct": 3.539, "tol_pct": 5.0, "pass": True},
            {"metric": "extrapolated discrete limit (log model, not selected)", "got_in": -1.7424948517366607e-05, "ref_in": -1.8248e-05, "err_pct": 4.510, "tol_pct": 5.0, "pass": True},
            {"metric": "dense vs sparse operator relative difference", "got_rel": 1.2922996006636822e-13, "tol_rel": 1e-10, "pass": True},
            {"metric": "shear factor k in {0.5, 5/6, 1.0} spread at 16x16", "got_pct": 1.3079334291806743, "tol_pct": 1.0, "pass": False},
            {"metric": "drilling factor 1e-10..1e-4 spread at 16x16", "got_pct": 1.028066520802895e-11, "tol_pct": 0.1, "pass": True},
            {"metric": "end-ring axial clamp stiffening (diagnostic, not a criterion)", "got_pct": 16.6, "tol_pct": None, "pass": None},
        ],
        "criteria_summary": "anchor criterion (<5% at 16x16, and at every rung including the finest) PASS; delivery self-gates 5/6 PASS with G5 (shear insensitivity, predeclared 1%) FAIL - disclosed, not retuned",
    },
    "repro_cmd": "PYTHONPATH=src python benchmarks/verified/fem_pinched_cylinder/run.py",
    "repro_result": {
        "auditor_rerun": True,
        "wall_seconds": 540.6,
        "rungs_bit_identical": "8/8 (only solve_seconds and generated_at differ from benchmarks/verified/fem_pinched_cylinder/case_result.json)",
        "gates_reproduced": "G1..G4,G6 PASS, G5 FAIL (identical to delivered case_gates.json)",
        "environment": "python 3.11.6 / torch 2.7.1 / scipy 1.16.0 / numpy 2.4.6 / loongarch64 / float64",
    },
    "external_anchor": {
        "paper": "MacNeal, R. L. & Harder, R. L. (1985), 'A proposed standard set of problems to test finite element accuracy', Finite Elements in Analysis and Design 1(1), 3-20, DOI 10.1016/0168-874X(85)90003-4",
        "value": "-1.8248e-5 in load-point radial deflection (converged reference quoted with the problem statement; the case has no closed form)",
        "limitation": "1985 text not reachable from this environment -> value is the one carried in the problem statement, not re-transcribed from the table/page the mission asked for; no second element family was used to re-derive it",
    },
    "findings": [
        "F1 MEDIUM (disclosed by delivery, reproduced by audit): the 16x16 0.167% is error cancellation - 3.256% stiffer-than-limit discretisation cancelled against a discrete limit 3.539% more flexible than the reference; per-rung error vs the driver's own limit falls monotonically 6.18% -> 0.84%. The 0.167% must not be read as element accuracy.",
        "F2 MEDIUM report hygiene: tensor_reports id 15225 section 6 marks the FAILED shear gate G5 (1.308% vs predeclared 1.0%, case_gates.json passed=false) with a check mark, while its own summary says 5/6 pass / overall FAIL.",
        "F3 LOW bookkeeping (same pattern as m16071 F3): tensor_benchmark_cases has 0 rows for the anchor and tensor_case_pool row FEM-PINCHED-CYLINDER is still 'dispatched' after the mission succeeded -> no registered row for the machine judge.",
        "F4 LOW: the 3 PNGs did not survive the merge to main (binary filter); they regenerate deterministically from the repro command - audit copies in docs/board_audit/m16175/artifacts_repro/figures/ with byte-identical sizes 79635/60098/109523 B.",
        "F5 LOW test coverage: test_transverse_shear_correction_is_not_a_fit_parameter checks k=1 vs 5/6 at N=8 (<1% passes) while the predeclared gate checks it at N=16 (1.308% FAIL) -> the unit test under-covers the gate; test_case_pool_pinched_cylinder_16x16_within_five_percent freezes a cancelling value and would trip on a genuinely improved element.",
    ],
    "not_run": [
        "independent re-derivation of the reference with a second shell element family (3.5% model bias is attributed to the point-load singularity, not proven)",
        "literature transcription check of the 1985 table/page (paper inaccessible from this environment)",
    ],
    "audit_report": "docs/board_audit/m16175/VERDICT_m16076.md",
    "level_rationale": "L1 tests assert physical invariants and pass (6 passed); L2 real input->compute->output library code; L3 public re-export + benchmark_cases runner dispatch + diagnostics consumer; L4 complete case (8-rung ladder, all rungs inside the declared 5% tolerance), real error analysis (two extrapolation models + 4 sensitivity sweeps + explicit cancellation forensics) and one-command bit-identical reproduction. Not L5: the delivery runs an existing element on a standard case and adds a storage path, no new research/engineering result.",
}

SQL = """INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('{repo}', '{scope}', {mid}, {level}, '{ev}'::jsonb, '{by}')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level, evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL
RETURNING id, repo, scope, mission_id, level, verdict_by, gate_state;"""


def run(sql):
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-F", "|", "-c", sql],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("psql failed: " + r.stderr[:400])
    return r.stdout.strip()


if __name__ == "__main__":
    ev = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
    sql = SQL.format(repo=REPO, scope=SCOPE, mid=AUDIT_MID, level=LEVEL, ev=ev, by=VERDICT_BY)
    if "--dry-run" in sys.argv:
        print(sql[:1200])
    else:
        print("row:", run(sql))
        print("gate-relevant evidence keys:", sorted(EVIDENCE.keys()))
        print("parsed:", run(f"SELECT level||'|'||gate_state FROM tensor_maturity_verdicts WHERE repo='{REPO}' "
                             f"AND scope='{SCOPE}' AND verdict_by='{VERDICT_BY}'"))