#!/usr/bin/env python3
"""Insert the m18651 board-audit maturity verdict into tensor_maturity_verdicts.

Idempotent on (repo, scope, verdict_by): re-running replaces the evidence of the
existing row for this scope instead of duplicating it.
"""
from __future__ import annotations

import json
import subprocess
import sys

REPO = "TensorFEM"
SCOPE = "m18651 XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO"
MISSION_ID = 18747
LEVEL = 2
VERDICT_BY = "board-audit"

W = "/root/tp-agent/work/m18651"
A = "/root/tp-agent/work/m18747/artifacts/xval_m18651_audit"

EVIDENCE = {
    "impl_files": [
        f"{W}/src/tensorfem/lbm_shan_chen_mcmp.py:1-332 (class ShanChenMCMP:132, MCMPConfig:101, "
        "interaction_force:213, barycentric_velocity:241, step:270; D2Q9 NumPy float64, periodic)",
        f"{W}/examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py:1-179 (OSS_REF:43-63, run_reference_arm:67, "
        "stability_boundary:98, laplace arm:~115, main:157)",
    ],
    "entry": [
        f"{W}/examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py:157 main() -- auditor executed it "
        "end-to-end in /tmp/audit18651 (exit 0, ~4 min, produced results.json)",
    ],
    "run_artifacts": [
        f"{W}/docs/assets/lbm_shan_chen_mcmp_barycentric_guo/results.json (2734 B: oss_reference, "
        "reference_arms[3], stability_boundary[7], laplace{points[3], fit})",
        f"{W}/docs/XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO.md (121 lines, six sections ①-⑥ present)",
        f"{A}/audit.json (auditor probes: provenance / doc-number traceability / "
        "independent divergence+mass-conservation runs)",
        f"{A}/m18651_board_audit.png (4-panel evidence figure)",
        f"{A}/audit_m18651.py + {A}/gen_figure.py (re-runnable audit harness)",
    ],
    "tests": [
        "NONE — grep -rn lbm_shan_chen_mcmp tests/ -> 0 hits; delivered doc §6 admits '未做: tests file'; "
        "auditor consequence: no target for pytest, L1 sub-criterion unproven",
        "auditor-side substitute in artifacts/xval_m18651_audit/audit_m18651.py: probe_solver() "
        "(g_ab=0/1 500 steps: rel mass drift 2.8e-14; g_ab=6: first non-finite step 30 == delivered)",
    ],
    "integration_callers": [
        "NONE — repo-wide grep for 'lbm_shan_chen_mcmp' outside examples/ and the module itself -> 0 hits; "
        "no main-solver / platform call path (L3 not met)",
    ],
    "benchmark": {
        "tensor_benchmark_cases": "anchor_slug='XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO' -> 0 rows "
                                  "(psql -d tensorplatform, 2026-11-13 by auditor): no L4 judge row exists",
        "ref_vs_got": "ABSENT — all three convention arms diverged before any profile comparison "
                      "(eq_shift=half/forced: step 30, none/bary: 26, tau/bary: 14; "
                      "results.json:reference_arms), so rel_l2_a / rel_linf_a were never computed "
                      "(examples/...xval.py:81-90 guards them behind 'diverged_at_step is None')",
        "secondary_arm": "Laplace law g_ab=1, 96^2, 3000 steps: sigma=0.006251, c=0.004754, R^2=0.2396, "
                         "per-radius rel dev [5.91, 9.9, 11.48] >> 0.15 -> FAIL (self-declared withheld/FAIL)",
        "verdict": "FAIL on the anchor; honest (no number was invented)",
    },
    "repro_cmd": f"cd {W} && PYTHONPATH=src python examples/lbm_shan_chen_mcmp_barycentric_guo_xval.py "
                 "(auditor re-run in /tmp/audit18651: exit 0; regenerated results.json is field-by-field "
                 "identical to the committed one, 0 diffs incl. float equality)",
    "external_anchor": (
        "lssfau/lbmpy tests/shan_chen/test_shan_chen_two_component.py (its own ref array is quoted from "
        "lbm-principles-practice/code/chapter9/shanchen.cpp, nsteps=1000, nfluids=2, gA=0). "
        "Auditor: embedded OSS_REF['ref_a'] is BIT-EXACT with the OSS test literal (max|dev|=0.0); "
        "sha256(embedded ref_a)=11456c100b457f82569d0d831866a5bf994333794dc156726a9b52b0586e7c53, "
        "sha256(OSS test file)=7e88e89825a34a56e79723fa670a1114926d6282766bf98b918849577fa63415. "
        "BUT the delivered claim 'the OSS test itself was executed on this machine and passes' is NOT "
        "supported: the mission's own log /tmp/m18651_oss/lbmpy_two_component.log only contains "
        "\"can't open file run_lbmpy_ref.py\"; under the installed lbmpy 2.0 / pystencils 2.0 the test "
        "fails at line 30 with AttributeError 'LBStencil' object has no attribute 'weights' "
        "(auditor rerun /tmp/lbtest_clean/out.log; same trace in /tmp/lbmpy_direct.log). "
        "So the anchor is a faithful copied literal, never executed; no sha256 was recorded despite the "
        "driver comment claiming one."),
    "auditor_rerun": {
        "date": "2026-11-13",
        "artifact_diff": "0 / 0 fields",
        "claim_crosscheck": {"g_ab=6 first non-finite step": "delivered 30 vs auditor 30",
                             "g_ab=1 stability": "delivered stable vs auditor 500 steps stable"},
        "mass_conservation": "rel drift 2.8e-14 (g_ab=0 and g_ab=1, 500 steps)",
        "env": "python 3.11.6, numpy (see audit.json)",
    },
    "overstated_claims": [
        "HIGH: 'The OSS test itself was executed on this machine (lbmpy 2.0 + pystencils 2.0) and passes, "
        "so the reference is live and not a stale text' — contradicted by the mission's own failing log and "
        "by an auditor rerun (API drift at test line 30)",
        "MEDIUM: comment 'file sha256 recorded below' — no sha256 key exists in OSS_REF",
        "LOW: dead ref_b fallback (ref_b not carried although the OSS test has it)",
        "LOW: root-cause narrative ('over-strong equilibrium velocity shift', 'supersonic lean phase') is "
        "not supported by its own data (eq_shift=none arm also diverges at step 26; half-arm visible max "
        "density 0.79 < IC 0.9)",
    ],
    "level_rationale": (
        "L2: real 332-line D2Q9 MCMP Shan-Chen kernel (input->compute->output), bit-identical reproducible "
        "artifacts, mass conserved to 1e-14, stable and productive for g_ab<=3. NOT L3 (no caller outside "
        "examples/), NOT L4 (benchmark table 0 rows + zero ref-vs-got numbers because every arm diverged), "
        "NOT L5. Caveat: L1 (independent unit/math verification) is unproven (0 tests), so a strict ladder "
        "reading caps at L1; L2 is granted for implementation substance, not correctness."),
    "repo_boundary_note": "anchor belongs to the TensorLBM domain (lssfau/lbmpy); this delivery landed in the "
                          "TensorFEM tree and is scoped as the fluid-side enabler, as declared in report §1",
}

SQL = """
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (%s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT (repo, scope, verdict_by)
DO UPDATE SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
              evidence = EXCLUDED.evidence, gate_state = 'pending', gate_level = NULL,
              gate_note = NULL
RETURNING id, gate_state, level;
"""


def main() -> int:
    payload = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
    stmt = SQL % (f"'{REPO}'", f"'{SCOPE}'", str(MISSION_ID), str(LEVEL),
                  f"'{payload}'", f"'{VERDICT_BY}'")
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tAc", stmt],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())