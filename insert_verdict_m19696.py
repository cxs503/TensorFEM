#!/usr/bin/env python3
"""Insert the m19696 board-audit maturity verdict for m19614 / GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.

Evidence values are real pointers established by the audit (see audit_report_m19696.md):
absent-in-repo facts are stated as absent with the exact command that proves it, never as a fake path.
"""
from __future__ import annotations

import json
import subprocess
import sys

MISSION_ID = 19696
REPO = "TensorFEM"
SCOPE = "m19614 GAP-PP-1-ICE-STRUCT-CONTACT-PATCH"
LEVEL = 0

EVIDENCE = {
    "impl_files": (
        "ABSENT for m19614: claimed commit 826e860a is not in the object db "
        "(`git cat-file -t 826e860a` -> fatal: Not a valid object name); branch agent/m19614 has no ref "
        "(200 local agent/* refs + mirror /root/tensor-ecosystem/TensorFEM checked); main@9ad824f7 has no "
        "src/tensorfem/contact_patch_load.py. Sibling implementation of the same anchor exists on branch "
        "agent/m18930: src/tensorfem/contact_patch_load.py:1-999 (patch_from_contact -> MITC4/Mindlin -> "
        "principal_moments), attributed to m18657/m18930, NOT to m19614 and NOT merged to main"
    ),
    "entry": (
        "ABSENT: `python -m tensorfem.cli contact-patch-load` does not exist in main@9ad824f7 "
        "(no such subcommand in cli.py); no caller imports contact_patch_load anywhere on main"
    ),
    "run_artifacts": (
        "ABSENT: `git log --all -- artifacts/contact_patch_load/stress_fields.json "
        "benchmarks/run_contact_patch_study.py docs/CASE-FEM-CONTACT-PATCH-LOAD-COUPLING.md` -> empty; "
        "filesystem-wide find for stress_fields.json/peak_stress.png/stress_contour.png -> none; "
        "m19614 worktree /root/tp-agent/work/m19614 reclaimed. Only agent/m18930 holds "
        "artifacts/contact_patch_load/{study.json,concentration_sweep.json}"
    ),
    "tests": (
        "ABSENT for m19614 (tests/test_contact_patch_load.py not in main nor any ref). Independent rerun of the "
        "sibling implementation: agent/m18930:tests/test_contact_patch_load.py -> 14 passed in 88.81s "
        "(PYTHONPATH=src pytest -q), i.e. the capability substance exists but belongs to m18657/m18930"
    ),
    "integration_callers": (
        "NONE on main: grep/import scan finds no module referencing contact_patch_load; the module is unmerged "
        "(branch agent/m18930 only) -> not L3"
    ),
    "benchmark": (
        "NO JUDGE ROW: `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
        "anchor_slug='GAP-PP-1-ICE-STRUCT-CONTACT-PATCH'` -> (0 rows) on postgres tensorplatform "
        "(118 rows / 83 anchors total). Nearest parent row: anchor_slug=H3-ICE-PLATE-FEM, metric='Hetenyi ice-plate "
        "deflection/moment/failure location vs series', tolerance='<5%', status=pending. Hence no tolerance exists "
        "for the anchor's own metric (non-uniform vs uniform peak-stress amplification) -> L4 structurally impossible"
    ),
    "repro_cmd": (
        "NOT EXECUTABLE on main@9ad824f7: benchmarks/run_contact_patch_study.py missing; "
        "`python -m tensorfem.cli contact-patch-load` missing; tests/test_contact_patch_load.py missing. "
        "Only reproducible command is the sibling's: "
        "`git archive agent/m18930 | tar -x -C <dir> && PYTHONPATH=src pytest tests/test_contact_patch_load.py -q` -> 14 passed/88.81s"
    ),
    "external_anchor": (
        "tensor_benchmark_cases anchor_slug=H3-ICE-PLATE-FEM (Hetenyi/Westergaard ice-plate series solution, tol <5%, "
        "status=pending) is the parent external anchor; the non-uniform-patch amplification claim of m19482 "
        "(report id=17792) self-declares 'no analytic reference' and substitutes force/moment conservation + mesh "
        "independence -> measured amplification is configuration dependent: 1.009-1.025 (centre peak) and "
        "1.090-1.191 (one-sided ramp) in agent/m18930 artifacts/contact_patch_load/concentration_sweep.json"
    ),
    "verdict_level": LEVEL,
    "level_basis": (
        "L0: m19614 has zero verifiable in-repo functional substance (no impl, no tests, no artifacts, no docs, "
        "no CLI entry, no report row in tensor_reports for mission_id=19614). The anchor's capability substance is "
        "L2-grade but belongs to m18657/m18930 (unmerged branch agent/m18930); existing verdict id=794 "
        "(scope 'm19482 GAP-PP-1-ICE-STRUCT-CONTACT-PATCH', level 2, mission_id=19562) is not inherited by m19614"
    ),
    "disposition": "reopened",
    "checks_run": [
        "psql tensor_benchmark_cases anchor query -> 0 rows",
        "git cat-file/branch/log reachability for commit 826e860a and branch agent/m19614 -> absent",
        "git log --all -- <claimed artifact paths> -> empty",
        "filesystem find for claimed artifacts -> none",
        "psql tensor_reports WHERE mission_id=19614 -> 0 rows",
        "PYTHONPATH=src pytest tests/test_contact_patch_load.py (agent/m18930 export) -> 14 passed in 88.81s",
        "read agent/m18930 artifacts/contact_patch_load/concentration_sweep.json + study.json mesh_stability",
    ],
    "inflated_claims": [
        "headline 'commit 826e860a (branch agent/m19614)' - commit and branch do not exist",
        "checks 'pytest tests/test_contact_patch_load.py ... -> 42 passed' - file absent from every ref",
        "checks 'run_contact_patch_study.py -> study.json/stress_fields.json/3 PNG' - none present in any ref",
        "on_disk_changes lists 6 src/doc/artifact paths, 0 of which exist in the repository",
        "report not published although mission rules require it (tensor_reports has no mission_id=19614 row)",
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
    with open("/tmp/verdict_m19696.sql", "w") as f:
        f.write(sql)
    r = subprocess.run(
        ["psql", "-U", "postgres", "-d", "tensorplatform", "-f", "/tmp/verdict_m19696.sql"],
        capture_output=True,
        text=True,
    )
    print(r.stdout)
    print(r.stderr[-800:], file=sys.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())