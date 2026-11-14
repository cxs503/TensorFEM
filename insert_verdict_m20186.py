#!/usr/bin/env python3
"""Insert the m20186 board-audit maturity verdict for m20143 / CASE-FEM-HEAT-L2.

Every evidence value is a pointer established by the audit in
docs/board_audit/m20186/audit_m20186.md; absent facts are stated as absent with the
command that proves it, never as a fake path.
"""
from __future__ import annotations

import json
import subprocess
import sys

MISSION_ID = 20186
REPO = "TensorFEM"
SCOPE = "m20143 CASE-FEM-HEAT-L2"
LEVEL = 0

EVIDENCE = {
    "impl_files": (
        "ABSENT from every git ref: src/tensorfem/heat_conduction_l2.py exists only as an untracked file in the "
        "m20143 work tree (/root/tp-agent/work/m20143/src/tensorfem/heat_conduction_l2.py, 176 lines, git status '??'); "
        "`git log --all --oneline -- src/tensorfem/heat_conduction_l2.py` -> empty; agent/m20143 tip f5e3ca29 == "
        "main@f634c7c5, `git log main..agent/m20143` -> empty; `git ls-tree -r --name-only HEAD | grep -ci heat` -> 0. "
        "Work-tree substance delegates to the shipped kernel src/tensorfem/thermal.py "
        "(ThermalModel/assemble_thermal/solve_steady_thermal, Q4)"
    ),
    "entry": (
        "work-tree only: /root/tp-agent/work/m20143/src/tensorfem/__init__.py:439-445 exports "
        "run_heat_conduction_l2/HeatConductionL2Result (git status 'M', uncommitted) with defaults "
        "order_threshold=1.8, l2_threshold=0.03; no such export on any ref -> not importable from the repository; "
        "no CLI subcommand, no platform API caller"
    ),
    "run_artifacts": (
        "work-tree only: /root/tp-agent/work/m20143/benchmarks/verified/fem_heat_conduction_l2/"
        "{result.json 2252B, gates_verified.json 1501B, run.py 15692B, README.md, fig_convergence.png 73515B, "
        "fig_field.png 117492B} - all untracked; `git log --all -- benchmarks/verified/fem_heat_conduction_l2` -> empty"
    ),
    "tests": (
        "work-tree only: tests/test_heat_conduction_l2.py (8 tests). Independent rerun by m20186: "
        "`cd /root/tp-agent/work/m20143 && OMP_NUM_THREADS=4 PYTHONPATH=src pytest tests/test_heat_conduction_l2.py -q` "
        "-> 8 passed in 6.39s. Test content is weak: convergence test asserts order>=1.8 (anchor asks 2), and "
        "test_heat_conduction_l2_heat_balance asserts max|K_ff T_f - f_f| which is a solve residual (tautologically ~1e-15), "
        "not an independent conservation invariant"
    ),
    "integration_callers": (
        "NONE: the only integration is the uncommitted __init__.py re-export; no module on any ref imports "
        "heat_conduction_l2 / run_heat_conduction_l2; no platform API route or main-solver path calls it -> not L3"
    ),
    "benchmark": (
        "NO JUDGE ROW: `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
        "anchor_slug='CASE-FEM-HEAT-L2'` -> 0 rows (table holds 125 rows; sibling FEM anchors exist, e.g. "
        "CASE-FEM-CLAMPED-PLATE-UNIFORM '<3%'). The board contract makes the judge row the sole referee, so L4 is "
        "structurally impossible for this mission; the tolerances (finest L2<3%, mean order>=1.8) exist only in the "
        "mission's own gates_verified.json, i.e. author-declared and self-lowered from the anchor's 2nd order"
    ),
    "repro_cmd": (
        "Executable only inside the m20143 work tree (files absent from main): "
        "`cd /root/tp-agent/work/m20143 && OMP_NUM_THREADS=4 PYTHONPATH=src pytest tests/test_heat_conduction_l2.py -q` "
        "-> 8 passed / 6.39s; driver `PYTHONPATH=src python benchmarks/verified/fem_heat_conduction_l2/run.py`. "
        "Rejected-from-main check: `git ls-tree -r --name-only HEAD | grep -c benchmarks/verified/fem_heat_conduction_l2` -> 0"
    ),
    "external_anchor": (
        "Manufactured harmonic field T=sin(pi x)*sinh(pi y)/sinh(pi) (Zienkiewicz & Taylor, FEM vol.1, ch.3-4) - analytic "
        "reference, not a DB benchmark row. Independent re-derivation (docs/board_audit/m20186/verify_heat_l2.py): the four "
        "declared levels reproduce bit-exactly (max rel diff 1.2e-16 on 32x32, n_dof 9/49/225/961 identical), so the numbers "
        "are genuine solver output, not hard-coded; BUT observed orders 1.810/1.884/1.935, mean 1.876 < 2 - the headline "
        "'>=2nd order' is not supported by the mission's own numbers. Root cause is metric, not kernel: the mission uses a "
        "node-RMS over ALL nodes including the exactly imposed Dirichlet boundary nodes; restricting to free dofs gives "
        "2.392/2.173/2.080/2.038, and an added 64x64 level gives 1.966 (all nodes) / 2.038 (free dofs). Same anchor already "
        "delivered by m20054: commit 8867c2a4 on refs/heads/agent/m20054, src/tensorfem/heat_l2_convergence.py:112-113,233 "
        "Gauss-integrated ||e||_0 gives 1.998-2.000, audited as L4 in verdict id=828 (mission 20108)"
    ),
    "db_rows": (
        "tensor_reports WHERE mission_id=20143 -> 0 rows (report discipline unmet: docs/case_fem_heat_l2_report.json with "
        "six sections + 2 embedded figures exists on disk but was never published; the CASE-FEM-HEAT-L2 report rows in "
        "tensor_reports belong to missions 20054 (id=18279) and 20108 (id=18309)); "
        "tensor_maturity_verdicts WHERE mission_id=20143 -> 0 rows before this audit"
    ),
    "verdict_level": LEVEL,
    "level_basis": (
        "L0: the in-repo substance attributable to m20143 is zero - no ref contains the implementation, tests, artifacts or "
        "docs, the branch tip equals main, no report row, and no judge row. A genuinely L2-grade implementation (real solve -> "
        "real error -> real order, reproducible) does exist, but only as uncommitted work-tree files that never entered the "
        "repository, so it cannot be booked as repository maturity. Even if committed it would cap at L2 (no judge row -> no L4; "
        "no caller -> no L3), and its '>=2 order' headline contradicts its own data (mean 1.876, all pairs < 2, threshold "
        "self-lowered to 1.8, metric inflated by boundary-node normalisation). The anchor's L4 belongs to m20054 "
        "(verdict id=828); m20143 is a parallel duplicate of the same anchor and should be de-duplicated, not double-booked"
    ),
    "disposition": "reopened",
    "checks_run": [
        "psql tensor_benchmark_cases anchor_slug='CASE-FEM-HEAT-L2' -> 0 rows",
        "psql tensor_reports WHERE mission_id=20143 / scenario_key ILIKE 'CASE-FEM-HEAT-L2' -> 0 row for 20143",
        "psql tensor_maturity_verdicts WHERE mission_id=20143 -> 0 rows (pre-audit)",
        "git for-each-ref --contains 8867c2a4 -> refs/heads/agent/m20054 only (anchor delivered elsewhere, not in main)",
        "git log --all -- src/tensorfem/heat_conduction_l2.py / benchmarks/verified/fem_heat_conduction_l2 -> empty",
        "git status / git log main..agent/m20143 in the m20143 work tree -> uncommitted only",
        "PYTHONPATH=<m20143>/src pytest tests/test_heat_conduction_l2.py -q -> 8 passed in 6.39s",
        "independent driver docs/board_audit/m20186/verify_heat_l2.py: 4 declared levels reproduced bit-exactly; "
        "free-dof L2 norm and 64x64 level added (local, LADDER=4x4..64x64)",
        "compute gate job cj2563 (64x64,128x128 extension) submitted; 128x128 not obtained in window -> asymptotics "
        "argued from 64x64 level",
    ],
    "inflated_claims": [
        "title/README/exports claim 'L2 convergence >= 2nd order'; delivered orders are 1.810/1.884/1.935 (mean 1.876) "
        "and every pair is < 2 - the gate was predeclared at 1.8, not 2",
        "report section 2 says the observed order comes from 'Richardson extrapolation'; the code computes a plain "
        "log-log slope (heat_conduction_l2.py:157-163) - no extrapolation",
        "gate 3 'heat balance residual' is presented as global conservation evidence but is the algebraic residual of the "
        "linear solve, true by construction",
        "delivery implied by the 'succeeded' mission status: nothing was committed, the report was never published to "
        "tensor_reports, and no benchmark row was registered for the anchor",
    ],
}


def main() -> int:
    ev = json.dumps(EVIDENCE, ensure_ascii=False)
    sql_literal = ev.replace("'", "''")
    sql = f"""
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('{REPO}', '{SCOPE}', {MISSION_ID}, {LEVEL}, '{sql_literal}', 'board-audit');
SELECT id, repo, scope, mission_id, level, verdict_by, created_at
  FROM tensor_maturity_verdicts WHERE mission_id = {MISSION_ID} AND verdict_by = 'board-audit';
"""
    with open("/tmp/verdict_m20186.sql", "w") as f:
        f.write(sql)
    r = subprocess.run(
        ["psql", "-U", "postgres", "-d", "tensorplatform", "-f", "/tmp/verdict_m20186.sql"],
        capture_output=True,
        text=True,
    )
    print(r.stdout)
    print(r.stderr[-800:], file=sys.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())