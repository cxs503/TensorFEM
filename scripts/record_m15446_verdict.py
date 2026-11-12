#!/usr/bin/env python3
"""Record the board-audit verdict for m15446 into tensor_maturity_verdicts.

Evidence keys follow the maturity_gate JUDGE_SCHEME contract; every pointer is a
path/line that exists in this audit worktree (or the audited worktree).
Idempotent: ON CONFLICT (repo, scope, verdict_by) DO UPDATE.

Usage: python3 scripts/record_m15446_verdict.py [--dry-run]
"""
from __future__ import annotations

import json
import subprocess
import sys

REPO = "TensorFEM"
SCOPE = "m15446 REPRO-FEM-HILLERBORG1976"
AUDIT_MID = 15645
LEVEL = 2
VERDICT_BY = "board-audit"

EVIDENCE = {
    "impl_files": [
        "src/tensorfem/fictitious_crack.py",
        "tests/test_fictitious_crack.py",
    ],
    "entry": "benchmarks/hillerborg1976_repro.py",
    "run_artifacts": [
        "docs/board_audit/m15645/artifacts_m15446/repro.json",
        "docs/board_audit/m15645/artifacts_m15446/fig_size_effect.png",
        "docs/board_audit/m15645/artifacts_m15446/fig_load_cmod.png",
    ],
    "tests": [
        "tests/test_fictitious_crack.py (15 passed, 15.62 s local)",
        "tests/test_board_audit_m15446.py (5 passed, audit guards)",
    ],
    "integration_callers": [],
    "benchmark": {
        "registered_row_for_anchor": 0,
        "note": "tensor_benchmark_cases has no row for REPRO-FEM-HILLERBORG1976; "
                "XVAL-FEM-TPB1976 is the same paper but a different (LEFM) criterion",
        "ref_vs_got": [
            {"metric": "K_I from model compliance vs Srawley-Gross handbook",
             "got_pct": 11.80, "ref_pct": 0.0, "tol_pct": 5.0, "pass": False},
            {"metric": "P_peak/P_LEFM at W/l_ch=8", "got": 0.4638, "ref": 1.0,
             "err_pct": 53.62, "tol_pct": 5.0, "pass": False},
            {"metric": "G_F back-calculated vs input", "got_J_m2": 19.73, "ref_J_m2": 100.0,
             "err_pct": 80.27, "tol_pct": 5.0, "pass": False},
            {"metric": "energy balance |Wext-Us-Uif-D|/Wext", "got": 2.926e33,
             "ref": 0.0, "tol": 1e-2, "pass": False},
            {"metric": "P_peak drift nodes_y 48->96", "got_pct": 1.32, "tol_pct": 2.0, "pass": True},
            {"metric": "P_peak drift k0 1e2..1e4", "got_pct": 0.50, "tol_pct": 1.0, "pass": True},
        ],
        "criteria_summary": "4 PASS / 6 FAIL of 10 predeclared criteria (recomputed, 0 flag disagreements)",
    },
    "repro_cmd": "python3 benchmarks/hillerborg1976_repro.py --out artifacts/hillerborg1976",
    "external_anchor": {
        "paper": "Hillerborg, Modeer & Petersson, Cem. Concr. Res. 6 (1976) 773-781",
        "anchors_used": ["Srawley-Gross SENB K_I handbook", "closed-form softening envelope area = G_F",
                         "Maxwell-Betti reciprocity", "net-section modulus-of-rupture load"],
        "limitation": "1976 figures not digitised; no experimental curve comparison",
    },
    "audit_notes": {
        "audit_report": "docs/board_audit/m15645/VERDICT_m15446.md",
        "verify_output": "docs/board_audit/m15645/verify_output.json",
        "artifacts_not_on_main": "docs/repro-hillerborg1976.md landed in af10148d without artifacts/hillerborg1976 "
                                 "(mission commit message: 4 files filtered by the gate); artifacts survive on "
                                 "branch agent/m15446 commit bc5d61a8 and in the mission worktree",
        "quick_run": "python3 benchmarks/hillerborg1976_repro.py --quick -> 60 s, criteria: K2/K5/K8 pass, rest fail",
        "full_rerun": "cluster job cj1165 submitted (repo=TensorFEM ref=agent/m15446), still queued at audit time",
        "l3_blocker": "no callers of tensorfem.fictitious_crack outside driver/tests; not exported in __init__",
        "l4_blocker": "no registered benchmark case for the anchor + 6/10 predeclared criteria fail",
    },
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
        print(sql[:1500])
    else:
        print("row:", run(sql))
        print("gate-relevant evidence keys:",
              sorted(json.loads(json.dumps(EVIDENCE)).keys()))
        print("parsed level:", run(f"SELECT level FROM tensor_maturity_verdicts WHERE repo='{REPO}' "
                                   f"AND scope='{SCOPE}' AND verdict_by='{VERDICT_BY}'"))