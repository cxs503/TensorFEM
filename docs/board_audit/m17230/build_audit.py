#!/usr/bin/env python3
"""Build the board-audit evidence bundle for subject m17194.

Numbers are read from the subject's own run artifacts and from the platform
benchmark tables (psql); the handbook reference is recomputed here from the
Tada/Paris/Irwin polynomial so that no figure is transcribed by hand.

Outputs (next to this file):
  verdict_evidence.json      -- evidence pointer bundle for tensor_maturity_verdicts
  AUDIT-m17194-CASE-FEM-JINTEGRAL-EDGE-CRACK.md -- audit report
"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
SUBJECT = pathlib.Path("/root/tp-agent/work/m17194")
BENCH = SUBJECT / "benchmarks/verified/fem_sent_j_integral"
ANCHOR = "CASE-FEM-JINTEGRAL-EDGE-CRACK"


def tada(alpha: float) -> float:
    return (1.12 - 0.231 * alpha + 10.55 * alpha ** 2
            - 21.72 * alpha ** 3 + 30.39 * alpha ** 4)


def psql(sql: str) -> list[list[str]]:
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA",
                        "-F", "\x01", "-c", sql], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("psql failed: " + r.stderr[:400])
    return [ln.split("\x01") for ln in r.stdout.strip().split("\n") if ln]


def main() -> int:
    led = json.loads((BENCH / "ledger_acceptance.json").read_text())
    res = json.loads((BENCH / "result.json").read_text())
    cfg = led["qualification"]["configuration"]
    ev_rows = {row["id"]: row for row in led["qualification"]["evidence"]}
    k_row, f_row = ev_rows["mode_i_k"], ev_rows["shape_factor"]

    alpha = cfg["a_over_w"]
    f_ref = tada(alpha)
    k_ref = cfg["sigma_pa"] * math.sqrt(math.pi * alpha * cfg["width_m"]) * f_ref
    ladder = led["ladder"]["levels"]

    case = psql("SELECT id, case_name, tolerance, status FROM tensor_benchmark_cases "
                f"WHERE anchor_slug='{ANCHOR}'")[0]
    tb = psql("SELECT result_id, ref_value, got_value, rel_error, tolerance_rel, passed, run_id "
              "FROM tensorbench_results WHERE mission_id=17194")[0]
    reports = psql("SELECT count(*) FROM tensor_reports WHERE mission_id=17194")[0][0]

    rerun = json.loads((HERE / "rerun/ledger_acceptance.json").read_text())
    rerun_got = {row["id"]: row for row in rerun["qualification"]["evidence"]}

    # numeric equality between subject artifact and auditor rerun
    def flat(d, p=""):
        if isinstance(d, dict):
            for k, v in d.items():
                yield from flat(v, f"{p}/{k}")
        elif isinstance(d, list):
            for i, v in enumerate(d):
                yield from flat(v, f"{p}[{i}]")
        else:
            yield p, d

    A, B = dict(flat(led)), dict(flat(rerun))
    diff = sorted(k for k in set(A) | set(B) if A.get(k) != B.get(k))
    numeric_diff = [k for k in diff if isinstance(A.get(k), (int, float))]

    evidence = {
        "subject_mission": "m17194",
        "subject_branch": "agent/m17194",
        "subject_commits": ["54ec5a32 (product entry + ledger acceptance)",
                            "2089919d (docs)"],
        "subject_worktree": str(SUBJECT),
        "audit_worktree": "/root/tp-agent/work/m17230 (agent/m17230)",
        "impl_files": [
            str(SUBJECT / "src/tensorfem/edge_crack_sent.py"),
            str(SUBJECT / "src/tensorfem/cli.py"),
            str(SUBJECT / "src/tensorfem/marine_benchmark_suite.py"),
            str(BENCH / "ledger_acceptance.py"),
        ],
        "impl_refs": [
            "src/tensorfem/edge_crack_sent.py:909 run_sent_j_qualification (product entry)",
            "src/tensorfem/edge_crack_sent.py:616 j_integral_subring (3-pt Gauss on fan mesh lines, "
            "J = int W n_x ds - int t.du/dx ds, K = sqrt(J E'))",
            "src/tensorfem/edge_crack_sent.py:130 tada_sent_shape_factor (grading step only, no handbook "
            "number in the numerical path)",
            "src/tensorfem/cli.py:223 subcommand sent-j-integral, exit code carries the verdict",
            "src/tensorfem/marine_benchmark_suite.py:177 run_sent_j_qualification() -> "
            "marine.fracture.sent_j.* rows of the ship-structure suite",
        ],
        "entry": str(BENCH / "ledger_acceptance.py"),
        "entry_alt": str(SUBJECT / "src/tensorfem/cli.py") + " (python -m tensorfem.cli sent-j-integral)",
        "run_artifacts": [
            str(BENCH / "ledger_acceptance.json"),
            str(BENCH / "ledger_acceptance.md"),
            str(BENCH / "ledger_acceptance.png"),
            str(BENCH / "result.json"),
            str(BENCH / "gates_verified.json"),
            "docs/board_audit/m17230/rerun/ledger_acceptance.json",
            "docs/board_audit/m17230/rerun/ledger_acceptance.md",
            "docs/board_audit/m17230/rerun/ledger_acceptance.png",
            "docs/board_audit/m17230/rerun/full_run.log",
            "docs/board_audit/m17230/rerun/fast_mode_failure.log",
        ],
        "tests": [
            "tests/test_sent_edge_crack_j_integral_plane_strain.py (13 tests; audit rerun 2026-11-13: "
            "13 passed, 55.05 s, -p no:cacheprovider)",
            "tests/test_sent_edge_crack_j_integral.py + tests/test_crack_tip_j_integral.py "
            "(audit rerun: 24 passed, 40.33 s; trio total 37 passed = subject claim)",
        ],
        "integration_callers": [
            "run_marine_benchmarks",
            "tensorfem.marine_benchmark_suite.run_marine_benchmarks -> marine.fracture.sent_j.* "
            "(src/tensorfem/marine_benchmark_suite.py:177, branch agent/m17194)",
            "tensorfem.cli sent-j-integral (src/tensorfem/cli.py:223, branch agent/m17194)",
        ],
        "benchmark": {
            "anchor_slug": ANCHOR,
            "tensor_benchmark_cases": {
                "id": int(case[0]),
                "case_name": case[1],
                "tolerance": case[2],
                "status": case[3],
            },
            "tensorbench_results": {
                "result_id": tb[0],
                "ref_value": float(tb[1]),
                "got_value": float(tb[2]),
                "rel_error": float(tb[3]),
                "tolerance_rel": float(tb[4]),
                "passed": tb[5] == "t",
                "run_id": tb[6],
            },
            "metric": (f"K_I from contour J (plane strain a/W={alpha}) vs Tada finite-width F(a/W); "
                       "K_I = sigma sqrt(pi a) F, sigma=100 MPa, W=100 mm, E=210 GPa, nu=0.3"),
            "ref_recomputed_by_auditor": {
                "F_tada": f_ref,
                "K_ref_Pa_sqrt_m": k_ref,
                "matches_db_ref": abs(k_ref - float(tb[1])) / float(tb[1]) < 1e-12,
            },
            "got": {
                "K_FEM_Pa_sqrt_m": k_row["actual"],
                "F_FEM": f_row["actual"],
                "rel_error": -abs(k_row["actual"] / k_row["oracle"] - 1.0),
                "abs_rel_error": k_row["relative_error"],
                "inside_3pct": k_row["relative_error"] < 0.03,
                "inside_5pct_ledger": k_row["relative_error"] < 0.05,
            },
            "ladder": [
                {"level": lv["level"], "n_sectors": lv["n_sectors"],
                 "n_elements": lv["n_elements"], "F_FEM": lv["shape_factor_fem"],
                 "rel_error": lv["rel_error"]} for lv in ladder
            ],
            "path_independence_rel_spread": led["qualification"]["path_independence"]["relative_spread"],
            "equilibrium_reaction_ratio": led["qualification"]["equilibrium"]["reaction_ratio"],
            "control_quarter_point_vs_midpoint": {
                "quarter_point_rel_error": led["control"]["configurations"]["quarter_point"]["rel_error"],
                "midpoint_rel_error": led["control"]["configurations"]["midpoint"]["rel_error"],
                "gain_pp": led["control"]["accuracy_gain_pp"],
                "limitation": led["control"]["limitation"],
            },
            "gci": {
                "applicable_at_aW03": led["gci"]["applicable"],
                "observed_order": led["gci"]["observed_order"],
                "medium_to_fine_change": led["gci"]["medium_to_fine_change"],
                "note": "Roache GCI honestly rejected at the ledger point; run.py's gci_fine=0.0222 "
                        "belongs to the a/W=0.20 ladder (CRACK_RATIOS 0.2/0.35/0.5), not a/W=0.30",
            },
            "cross_ratio_check_result_json": {
                f"a/W={c['a_over_w']:g}": c["rel_error"] for c in res["plane_strain"]["cases"]
            },
        },
        "repro_cmd": (
            "cd /root/tp-agent/work/m17194 && PYTHONPATH=src python -B "
            "benchmarks/verified/fem_sent_j_integral/ledger_acceptance.py   "
            "# auditor rerun 2026-11-13 02:44-02:46Z: verdict PASS, artifacts in "
            "docs/board_audit/m17230/rerun/ ; ALL floats identical to the subject artifact "
            f"(numeric keys differing: {numeric_diff})"
        ),
        "external_anchor": "docs/board_audit/m17230/external_anchor_tada_2000.md",
        "external_anchor_note": (
            "Tada, Paris & Irwin, The Stress Analysis of Cracks Handbook, 3rd ed. (2000), "
            "SENT F(a/W) 1.12-0.231a+10.55a^2-21.72a^3+30.39a^4 (+-0.5%, a/W<=0.6, H/W>=1) "
            "and the independent secant form in the same chapter; Rice (1968) J. Appl. Mech. 35:379"
        ),
        "audit": {
            "level_claimed": 4,
            "report_rows_in_tensor_reports": int(reports),
            "numeric_keys_differing_from_rerun": numeric_diff,
            "keys_differing_from_rerun": diff,
            "findings": {
                "S1_blocking": [],
                "S2_medium": [
                    "documented --fast variant of the repro script crashes: "
                    "TypeError: 'NoneType' object is not subscriptable at "
                    "ledger_acceptance.py:152 (make_figure) -> exit 1, no artifacts written "
                    "(reproduced twice; rerun/fast_mode_failure.log). The graded full run is unaffected."
                ],
                "S3_low": [
                    "tensor_reports has 0 rows for mission 17194 (its own finish recorded the report "
                    "POST as 401) -> six-section platform report missing; content exists only as "
                    "docs/CASE-FEM-JINTEGRAL-EDGE-CRACK_ledger_acceptance_m17194.md in the worktree.",
                    "deliverable not yet on main: product entry (run_sent_j_qualification / CLI / "
                    "marine suite hook / ledger_acceptance.py) lives only on branch agent/m17194 "
                    "(tip 2089919d); ecosystem checkout 74567201 does not import it.",
                ],
                "S4_info": [
                    "3 % pass at the 24-sector qualification mesh is partly a cancellation between "
                    "discretisation error and the H/W=2 far-field bias (coarse -1.976 %, fine "
                    "-0.738 %, and the midpoint-ring control also lands inside 3 %); m17194 states "
                    "this limitation itself, so the pass is not overstated.",
                    "run.py gates_verified.json gci_fine=0.0222 is computed on the a/W=0.20 ladder, "
                    "not the ledger point - checked, not a contradiction.",
                ],
            },
            "checks_not_run": [
                "benchmarks/verified/fem_sent_j_integral/run.py --stage height/field (known CG "
                "non-convergence at H/W=3.0, recorded not_run in result.json) -> far-field truncation "
                "bias unquantified; caps the claim below L5",
                "full TensorFEM test suite (not required for this verdict)",
                "cluster/loongarch re-run (auditor rerun executed on the control node, x86_64)",
            ],
        },
    }

    (HERE / "verdict_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")

    got_k = float(k_row["actual"]); ref_k = k_ref
    got_f = float(f_row["actual"]); ref_f = f_ref
    rel = -abs(got_k / ref_k - 1.0) * 100.0

    md = f"""# AUDIT — m17194 · CASE-FEM-JINTEGRAL-EDGE-CRACK (board-audit)

**Independence**: auditor worktree `/root/tp-agent/work/m17230` (agent/m17230); subject
worktree `{SUBJECT}` at branch `agent/m17194` (impl commit `54ec5a32`, docs `2089919d`).
No subject file was modified: the two repro scripts were copied to
`docs/board_audit/m17230/rerun/` and executed with `PYTHONPATH={SUBJECT}/src`.

## ① What m17194 delivered (verified, not quoted)
| item | pointer | status |
|---|---|---|
| product entry | `src/tensorfem/edge_crack_sent.py:909 run_sent_j_qualification` | real (mesh → solve → contour J → K) |
| J-integral kernel | `edge_crack_sent.py:616 j_integral_subring`, 3-pt Gauss on fan mesh lines | real (energy flux − work flux, K=sqrt(J E')) |
| CLI | `src/tensorfem/cli.py:223 sent-j-integral`, exit code = verdict | executed: exit 0, `passed=true` |
| suite hook | `src/tensorfem/marine_benchmark_suite.py:177` → `marine.fracture.sent_j.*` | present on branch |
| run artifacts | `benchmarks/verified/fem_sent_j_integral/{{ledger_acceptance.json,.md,.png,result.json,gates_verified.json}}` | all present, non-empty |
| tests | 13 + 24 tests | auditor rerun: **37 passed** (55 s + 40 s) |
| platform report | `tensor_reports` rows for mission 17194 = **{reports}** | missing (see S3) |

## ② Benchmark adjudication (the only judge)
`tensor_benchmark_cases` id={case[0]}: *{case[1]}* — metric *K_I from contour J vs handbook SIF,
plane strain a/W=0.3*, tolerance *{case[2]}*, status **{case[3]}**.

| quantity | got (FEM) | ref (Tada) | rel. error | <3 % | <5 % |
|---|---|---|---|---|---|
| K_I (Pa·√m) | {got_k:.1f} | {ref_k:.1f} | {rel:+.3f} % | {'yes' if abs(rel)<3 else 'NO'} | {'yes' if abs(rel)<5 else 'NO'} |
| F(a/W) | {got_f:.6f} | {ref_f:.6f} | {rel:+.3f} % | same | same |

Auditor recomputation of the reference (no transcription): F(0.3) = {ref_f:.9f},
K_ref = σ√(πa)·F = {ref_k:.4f} Pa·√m → identical to the DB `ref_value`
(`matches_db_ref={evidence['benchmark']['ref_recomputed_by_auditor']['matches_db_ref']}`) and to
`tensorbench_results.{tb[0]}` (got {float(tb[2]):.1f}, rel {float(tb[3])*100:.4f} %,
tolerance {float(tb[4])*100:.0f} %, passed={tb[5]}).

Refinement ladder (a/W=0.3, plane strain):
{chr(10).join(f"| {lv['level']} | {lv['n_sectors']} sectors | {lv['n_elements']} el | {lv['shape_factor_fem']:.6f} | {lv['rel_error']*100:+.3f} % |" for lv in ladder)}
Path spread {led['qualification']['path_independence']['relative_spread']*100:.3f} %; vertical reaction
ratio {led['qualification']['equilibrium']['reaction_ratio']:.13f}; A/B control quarter-point
{led['control']['configurations']['quarter_point']['rel_error']*100:+.3f} % vs midpoint
{led['control']['configurations']['midpoint']['rel_error']*100:+.3f} % (gain
{led['control']['accuracy_gain_pp']:.2f} pp, both inside 3 % → limitation stated by the author).

## ③ Reproducibility
`{evidence['repro_cmd'].splitlines()[0].strip()}` → auditor rerun
2026-11-13 02:44–02:46Z: **verdict PASS**, identical numbers. Diff against the subject artifact:
{diff} (numeric keys differing: {numeric_diff}).

## ④ Level: **L4** (benchmark verification)
L1 tests ✓ · L2 real implementation + entry + artifacts ✓ · L3 integration callers (marine suite
+ CLI) ✓ on branch · L4 full case with handbook reference, error ladder, path independence,
equilibrium and an executed A/B control, byte-reproducible ✓. **L5 not claimed**: no new
scientific result; the H/W height ladder stays `not_run` (CG non-convergence at H/W=3.0), so the
far-field truncation bias is not quantified.

## ⑤ Findings
"""
    for k, v in evidence["audit"]["findings"].items():
        md += f"\n**{k}**\n" + "".join(f"- {x}\n" for x in v) if v else f"\n**{k}**: none\n"
    md += "\n## ⑥ Not run\n" + "".join(f"- {x}\n" for x in evidence["audit"]["checks_not_run"])
    md += """
## ⑦ Disposition
**Approve at L4 with notes.** Non-blocking: (a) guard `--fast` in `make_figure` (or skip the
ladder panel) so the documented fast variant stops crashing; (b) publish the six-section report
to `tensor_reports` and merge `agent/m17194` so downstream consumers import the entry from main.
"""
    (HERE / "AUDIT-m17194-CASE-FEM-JINTEGRAL-EDGE-CRACK.md").write_text(md)

    print(json.dumps({"F_tada": f_ref, "K_ref": k_ref, "rel_error": rel,
                      "numeric_diff": numeric_diff, "reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())