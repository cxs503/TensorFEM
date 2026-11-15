#!/usr/bin/env python3
"""Insert the m22176 board-audit maturity verdict for m22156 / GAP-ICE-GATE-POWER.

Every evidence value is a pointer established by the audit in
docs/board_audit/m22176/audit_m22176.md; absent facts are stated as absent with the
command that proves it, never as a fake path.  Numbers are injected from
docs/board_audit/m22176/verify_output.json, not transcribed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

MISSION_ID = 22176
REPO = "TensorFEM"
SCOPE = "m22156 GAP-ICE-GATE-POWER"
LEVEL = 1

OUT = Path("docs/board_audit/m22176/verify_output.json")
V = json.loads(OUT.read_text()) if OUT.exists() else {}

piston = V.get("piston", {})
levels = {f"n{r['n_cells']}": round(r["rel_error"] * 100, 5) for r in piston.get("levels", [])}
barge = V.get("barge", {})
bem = barge.get("vs_verified_bem_2d", {})
doc_code = V.get("doc_vs_code", {})

EVIDENCE = {
    "impl_files": (
        "commit 7b947ff3da0b5461651d4c156eb5c6db83841791 adds exactly two files (+338 lines): "
        "tests/test_hydro_ice_anchor.py (242 lines) and docs/GAP-ICE-GATE-POWER-hydro-anchor.md (96 lines). "
        "The only new computational function is test-local: heave_added_mass_rectangular_barge at "
        "tests/test_hydro_ice_anchor.py:36-89 (not importable as library code; labelled "
        "method='strip_theory_newman_1977' but the C_a log-linear blend and f_3d=1/(1+2T/L) have no traceable "
        "equation number in Newman 1977). The production module it exercises, src/tensorfem/hydro_added_mass.py "
        "(303 lines, piston_radiation_mass/disc_panels/dipole_added_mass_matrix), is byte-identical to main and "
        "predates this mission (git log -1 -- src/tensorfem/hydro_added_mass.py -> a6c555ae, mission m2134)"
    ),
    "entry": (
        "pytest only: `PYTHONPATH=src pytest tests/test_hydro_ice_anchor.py`. No CLI subcommand, no platform API "
        "route, no library export (heave_added_mass_rectangular_barge lives inside the test module). The doc's "
        "own tool-chain section lists only tensorfem.hydro_added_mass + the test-local reference"
    ),
    "run_artifacts": (
        "ABSENT: no result JSON, no figures, no run manifest in the commit - `git show --stat 7b947ff3` lists only "
        "the two source files; no benchmarks/<case>/ directory, no out/*.json. The acceptance record "
        "(tensor_dev_missions id=22156, result.checks) is a pytest -k invocation string, not an artifact pointer. "
        "Board audit m22176 therefore re-derived every number with docs/board_audit/m22176/verify_ice_gate_power.py "
        "-> docs/board_audit/m22176/verify_output.json"
    ),
    "tests": (
        "tests/test_hydro_ice_anchor.py, 17 pytest items, zero skip markers (`grep -c skip` -> 0). Auditor rerun on "
        "main src: full-file run does NOT finish within 120 s (hangs in test_piston_anchor_passes_3pct_gate, "
        "n_cells=512); `-k 'not passes_3pct_gate'` -> 16 passed, 1 deselected, 34.73 s. The mission's own "
        "acceptance command was `pytest ... -k 'not piston_numerical and not piston_anchor'` -> 15 passed, "
        "2 deselected, i.e. it deselected exactly the two genuine numerical-convergence items and left only "
        "tautologies (formula==formula, scaling laws, ref*1.10 -> 10 % < 15 %, ref*2.65 -> 165 % > 15 %, "
        "arithmetic negative controls). The doc claims '15/17 tests pass (2 skipped)' - false: 0 skips, they were "
        "deselected. Anchor audit pins live in tests/test_hydro_anchor_reference_audit.py (4 tests, 5.01 s, 4 passed)"
    ),
    "integration_callers": (
        "NONE. Nothing imports the mission module or its reference: `grep -rn heave_added_mass_rectangular_barge` "
        "hits only the test file; benchmarks/ice_coupled_6dof/*.py (cummins_ice_6dof.py, run_w23/run_w30) contains no "
        "reference to it; no API/CLI path. Not L3"
    ),
    "benchmark": (
        "NO JUDGE ROW: `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
        "anchor_slug='GAP-ICE-GATE-POWER'` -> 0 rows (table has 144 rows; anchor_slug ILIKE '%GATE%' only matches "
        "the unrelated H6-MATURITY-GATE). The board contract makes the judge row the sole referee, so L4 is "
        "structurally impossible for this mission. The only pre-existing acceptance numbers (15 % / 3 %) are "
        "author-declared inside the mission's own doc"
    ),
    "repro_cmd": (
        "`cd <m22156 work tree> && OMP_NUM_THREADS=4 PYTHONPATH=src pytest tests/test_hydro_ice_anchor.py -q` -> "
        "does not terminate in 120 s (n_cells=512 item builds a 205887^2 float64 dense operator = "
        f"{piston.get('n_cells_512_projection', {}).get('dense_GiB_est')} GiB); "
        "`... -k 'not passes_3pct_gate'` -> 16 passed / 34.73 s. Auditor reproduction of the numbers: "
        "`PYTHONPATH=src python docs/board_audit/m22176/verify_ice_gate_power.py` (runs in 37 s, writes verify_output.json)"
    ),
    "external_anchor": (
        "Rayleigh (1945) baffled-piston M_rad=(8/3)rho*a^3 is a genuine exact anchor and the dipole operator does "
        f"converge to it: auditor rerun rel error {levels.get('n16')}% / {levels.get('n64')}% / {levels.get('n256')}% "
        "at n_cells=16/64/256 (observed order 0.976 then 0.585) - real, but it exercises main's pre-existing operator. "
        "The barge reference is NOT anchored: at the chain's own section (L=100,B=20,T=6) it gives "
        f"{bem.get('mission_ref_2d_kg_per_m', 0)/1e3:.2f} t/m against the ecosystem's verified independent 2D anchor "
        f"228.6 t/m (board report tensor_reports id=19234 Q5: BEM mirror-Green + FD Laplace, mutually 3.8 %, deep-circle "
        f"self-proof <=2.9 %) -> {bem.get('rel_error_vs_bem', 0):.4%}, i.e. 6x outside its own 15 % band. The audited "
        "quantity is 2D A_inf/L [kg/m] while the mission returns 3D kg (factor L=100 mismatch), and its geometry "
        "(T=8 m) differs from the chain (T=6 m, Cb=0.70). The claimed '165 % reproduction' is synthetic: "
        "test_declared_a33_165pct_fails sets declared = ref*2.65, and the doc's own operands (declared 9.5e6 kg vs "
        "reference 3.6e6 kg) match neither the code (4.06058e7 kg) nor the chain (8.610e6 kg) and have the OPPOSITE "
        "deviation sign to report 19234 (declared 86.1 t/m BELOW the independent 228.6 t/m)"
    ),
    "verdict_level": LEVEL,
    "level_basis": (
        "L1: a genuine unit-level mathematical verification exists (dipole operator -> Rayleigh piston, auditor rerun "
        f"{levels.get('n16')}%/{levels.get('n64')}%/{levels.get('n256')}%, monotone), which is L1-grade real verification - "
        "but it targets main's pre-existing operator (m2134) and the mission's own acceptance command deselected it. "
        "L2 denied: the only new computational function computes the wrong quantity (3D kg vs the audited 2D kg/m), "
        "for the wrong geometry (T=8 vs T=6, Cb ignored), and its output is "
        f"{bem.get('rel_error_vs_bem', 0):.2%} off the ecosystem's verified independent anchor - it is not a functioning "
        "independent implementation of the audited capability (physical plausibility fails too: A33/displaced = "
        f"{barge.get('chain_geometry_L100_B20_T6', {}).get('A33_over_displaced', 0):.2f} vs the usual 0.5-1.0 band, "
        "while the shipped sanity test accepts 0.1-5.0). L3 denied: no caller. L4 denied: judge row absent. L5 denied. "
        "REAL-criterion information content claimed by the mission is not delivered: reference source = invented "
        "(+92.5% off verified anchor); observed order = never reported; detection rate = 0 (declared values are "
        "synthesised from the reference inside the test); rejected negative controls = arithmetic identities with no "
        "reject logic to be rejected. Docs vs shipped code disagree: "
        f"A33_3D doc 3.6e6 kg vs code {doc_code.get('code', {}).get('A33_3D_kg', 0):.5g} kg "
        f"({doc_code.get('rel_diff', {}).get('A33_3D_kg', 0):.2%})"
    ),
    "findings": [
        {
            "id": "F1",
            "claim": "the anchor reproduces the 165 % declared-vs-independent added-mass discrepancy of report 19234",
            "evidence": "tests/test_hydro_ice_anchor.py:190-200 constructs declared = a33_ref*2.65 and asserts >15 %; the real "
                        "operands (86.1 t/m declared vs 228.6 t/m independent, per report 19234 Q5) are never read, and the "
                        "deviation direction in the mission doc (declared above reference) is the opposite of the audited case",
            "severity": "blocking (correctness)",
            "remediation": "feed the chain's real declaration from benchmarks/ice_coupled_6dof (A_inf = 1.00*mass = 8.610e6 kg) "
                           "and the verified 2D anchor into the gate; a check that cannot fail detects nothing",
        },
        {
            "id": "F2",
            "claim": "the reference provides an independent 15 % referee for A33",
            "evidence": f"same section (T=6): reference {bem.get('mission_ref_2d_kg_per_m', 0)/1e3:.2f} t/m vs verified BEM "
                        f"228.6 t/m -> {bem.get('rel_error_vs_bem', 0):+.2%}; 3D kg vs 2D kg/m unit mismatch; geometry T=8 vs T=6",
            "severity": "blocking (correctness)",
            "remediation": "re-base on the 2D per-metre basis with the chain geometry (L=100,B=20,T=6,Cb=0.70) and cross-check "
                           "the formula itself against 228.6 t/m before using it as a referee",
        },
        {
            "id": "F3",
            "claim": "the delivered report numbers are program-injected mission output",
            "evidence": f"A33_3D doc 3.6e6 kg vs shipped code {doc_code.get('code', {}).get('A33_3D_kg', 0):.5g} kg "
                        f"({doc_code.get('rel_diff', {}).get('A33_3D_kg', 0):+.2%}); convergence table doc 12/5/2/1 % vs measured "
                        f"{levels.get('n16')}/{levels.get('n64')}/{levels.get('n256')} % + n512 not_run; '2 skipped' vs 0 skip "
                        "markers (they were deselected); '~2 s at N=512' vs 315.8 GiB dense operator",
            "severity": "blocking (evidence integrity)",
            "remediation": "regenerate the doc from verify_output.json; state n512 as not_run with the cost projection",
        },
        {
            "id": "F4",
            "claim": "the anchor has a registered referee band",
            "evidence": "tensor_benchmark_cases WHERE anchor_slug='GAP-ICE-GATE-POWER' -> 0 rows (144 rows in table)",
            "severity": "blocking-for-L4 (platform-side)",
            "remediation": "register the case row (case_name ice_added_mass_2d, metric |A33_ref-A33_decl|/A33_decl, "
                           "tolerance 15 %, ref_source report 19234 Q5 BEM+FD double route, status pending)",
        },
        {
            "id": "F5",
            "claim": "the anchor gives REAL-criterion information content (baseline source, observed order, detection rate, rejected negative controls)",
            "evidence": "only the observed order is real (0.976/0.585, unreported by the mission); baseline source invented; "
                        "detection rate 0 by construction; negative controls are arithmetic identities",
            "severity": "major (information content)",
            "remediation": "predeclare a gate function that takes (reference, declared) and returns PASS/FAIL, then prove it "
                           "rejects ref*1.16..ref*100 and accepts ref*1.10 with values taken from the chain, not synthesised",
        },
        {
            "id": "F6",
            "claim": "15/17 tests pass, 2 skipped",
            "evidence": "tensor_dev_missions id=22156 result.checks: pytest ... -k 'not piston_numerical and not piston_anchor' "
                        "-> 15 passed, 2 deselected; grep -c skip -> 0",
            "severity": "major (process)",
            "remediation": "run the deselected items in the acceptance command (both pass: auditor rerun 16 passed/34.73 s "
                           "with only the infeasible n512 item excluded) and report deselection honestly",
        },
        {
            "id": "F7",
            "claim": "the reference module is reusable library code",
            "evidence": "heave_added_mass_rectangular_barge is defined inside the test module; modal_added_mass_ratio imported "
                        "but never used",
            "severity": "minor (hygiene)",
            "remediation": "either promote a validated reference into src/tensorfem with a cited equation number, or keep it in "
                           "tests and drop the 'Newman 1977 Sec. 4.3' attribution",
        },
    ],
    "checks_run": [
        "psql tensor_benchmark_cases WHERE anchor_slug='GAP-ICE-GATE-POWER' -> 0 rows (144 total)",
        "psql tensor_reports WHERE mission_id=22156 -> 0 rows (report never published to the platform)",
        "psql tensor_maturity_verdicts WHERE mission_id=22156 -> 0 rows (pre-audit)",
        "psql tensor_dev_missions id=22156 result::jsonb -> acceptance command + enforced deselection of both piston items",
        "psql tensor_reports id=19234 markdown -> declared A_inf/L=86.1 t/m, independent BEM 228.6 t/m, 165.5 %, T=6, Cb=0.70",
        "git show --stat 7b947ff3 -> 2 files, +338 lines, no artifacts",
        "git show HEAD:src/tensorfem/hydro_added_mass.py | diff - <m22156>/src/... -> identical",
        "pytest audit_tmp/test_hydro_ice_anchor.py (full) -> no completion in 120 s",
        "pytest audit_tmp/test_hydro_ice_anchor.py -k 'not passes_3pct_gate' -> 16 passed, 1 deselected, 34.73 s",
        "grep -c skip audit_tmp/test_hydro_ice_anchor.py -> 0",
        "python docs/board_audit/m22176/verify_ice_gate_power.py -> verify_output.json (piston 4.6033/0.31646/0.062547 %, barge +92.52 % vs BEM, doc-vs-code -91.13 %)",
        "pytest tests/test_hydro_anchor_reference_audit.py -> 4 passed, 5.01 s",
        "NOT RUN: n_cells=512 checkpoint (205887 panels, 315.8 GiB dense) and any >=1024 level - marked not_run, not passed",
    ],
    "inflated_claims": [
        "'复现对抗审查 #19234 发现的 165% 附加质量偏差' - synthesized as ref*2.65 with the opposite deviation sign to the audited case",
        "'Newman 1977 Sec. 4.3' strip-theory reference - no traceable equation; +92.5 % off the verified independent anchor",
        "'15/17 tests pass (2 skipped due to O(N^2) cost)' - 0 skip markers; the 2 items were deselected by the mission's own acceptance command and both actually pass",
        "'N=512 需要约 2 秒' - the operator is dense; n512 needs ~315.8 GiB and was never run (3 % is already met at n=64)",
        "doc convergence table 12/5/2/1 % - measured 4.6033/0.31646/0.062547 %",
        "doc A33_3D 3.6e6 kg / C_a 1.047 / f_3d 0.833 - shipped code gives 4.06058e7 kg / 1.14885 / 0.86207",
        "'REAL criterion anchor with baseline source / detection rate / rejected negative controls' - three of the four are not delivered",
    ],
    "disposition": "reopened",
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
    Path("/tmp/verdict_m22176.sql").write_text(sql)
    r = subprocess.run(
        ["psql", "-U", "postgres", "-d", "tensorplatform", "-f", "/tmp/verdict_m22176.sql"],
        capture_output=True,
        text=True,
    )
    print(r.stdout)
    if r.stderr:
        print(r.stderr[-800:], file=sys.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())