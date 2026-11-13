# AUDIT — m17194 · CASE-FEM-JINTEGRAL-EDGE-CRACK (board-audit)

**Independence**: auditor worktree `/root/tp-agent/work/m17230` (agent/m17230); subject
worktree `/root/tp-agent/work/m17194` at branch `agent/m17194` (impl commit `54ec5a32`, docs `2089919d`).
No subject file was modified: the two repro scripts were copied to
`docs/board_audit/m17230/rerun/` and executed with `PYTHONPATH=/root/tp-agent/work/m17194/src`.

## ① What m17194 delivered (verified, not quoted)
| item | pointer | status |
|---|---|---|
| product entry | `src/tensorfem/edge_crack_sent.py:909 run_sent_j_qualification` | real (mesh → solve → contour J → K) |
| J-integral kernel | `edge_crack_sent.py:616 j_integral_subring`, 3-pt Gauss on fan mesh lines | real (energy flux − work flux, K=sqrt(J E')) |
| CLI | `src/tensorfem/cli.py:223 sent-j-integral`, exit code = verdict | executed: exit 0, `passed=true` |
| suite hook | `src/tensorfem/marine_benchmark_suite.py:177` → `marine.fracture.sent_j.*` | present on branch |
| run artifacts | `benchmarks/verified/fem_sent_j_integral/{ledger_acceptance.json,.md,.png,result.json,gates_verified.json}` | all present, non-empty |
| tests | 13 + 24 tests | auditor rerun: **37 passed** (55 s + 40 s) |
| platform report | `tensor_reports` rows for mission 17194 = **0** | missing (see S3) |

## ② Benchmark adjudication (the only judge)
`tensor_benchmark_cases` id=37: *SENT edge crack, plane strain, a/W=0.3* — metric *K_I from contour J vs handbook SIF,
plane strain a/W=0.3*, tolerance *<5% (ledger) / <3% (platform)*, status **passed**.

| quantity | got (FEM) | ref (Tada) | rel. error | <3 % | <5 % |
|---|---|---|---|---|---|
| K_I (Pa·√m) | 49951986.6 | 50959183.4 | -1.976 % | yes | yes |
| F(a/W) | 1.627111 | 1.659919 | -1.976 % | same | same |

Auditor recomputation of the reference (no transcription): F(0.3) = 1.659919000,
K_ref = σ√(πa)·F = 50959183.3718 Pa·√m → identical to the DB `ref_value`
(`matches_db_ref=True`) and to
`tensorbench_results.m17194-sent-j-a03-ps` (got 49951986.6, rel 1.9765 %,
tolerance 5 %, passed=t).

Refinement ladder (a/W=0.3, plane strain):
| coarse | 24 sectors | 390 el | 1.627111 | -1.976 % |
| medium | 48 sectors | 1050 el | 1.637590 | -1.345 % |
| fine | 96 sectors | 2842 el | 1.647677 | -0.738 % |
Path spread 0.170 %; vertical reaction
ratio 1.0000000000001; A/B control quarter-point
-1.976 % vs midpoint
-2.550 % (gain
0.57 pp, both inside 3 % → limitation stated by the author).

## ③ Reproducibility
`cd /root/tp-agent/work/m17194 && PYTHONPATH=src python -B benchmarks/verified/fem_sent_j_integral/ledger_acceptance.py   # auditor rerun 2026-11-13 02:44-02:46Z: verdict PASS, artifacts in docs/board_audit/m17230/rerun/ ; ALL floats identical to the subject artifact (numeric keys differing: [])` → auditor rerun
2026-11-13 02:44–02:46Z: **verdict PASS**, identical numbers. Diff against the subject artifact:
['/gci/note', '/generated_at'] (numeric keys differing: []).

## ④ Level: **L4** (benchmark verification)
L1 tests ✓ · L2 real implementation + entry + artifacts ✓ · L3 integration callers (marine suite
+ CLI) ✓ on branch · L4 full case with handbook reference, error ladder, path independence,
equilibrium and an executed A/B control, byte-reproducible ✓. **L5 not claimed**: no new
scientific result; the H/W height ladder stays `not_run` (CG non-convergence at H/W=3.0), so the
far-field truncation bias is not quantified.

## ⑤ Findings

**S1_blocking**: none

**S2_medium**
- documented --fast variant of the repro script crashes: TypeError: 'NoneType' object is not subscriptable at ledger_acceptance.py:152 (make_figure) -> exit 1, no artifacts written (reproduced twice; rerun/fast_mode_failure.log). The graded full run is unaffected.

**S3_low**
- tensor_reports has 0 rows for mission 17194 (its own finish recorded the report POST as 401) -> six-section platform report missing; content exists only as docs/CASE-FEM-JINTEGRAL-EDGE-CRACK_ledger_acceptance_m17194.md in the worktree.
- deliverable not yet on main: product entry (run_sent_j_qualification / CLI / marine suite hook / ledger_acceptance.py) lives only on branch agent/m17194 (tip 2089919d); ecosystem checkout 74567201 does not import it.

**S4_info**
- 3 % pass at the 24-sector qualification mesh is partly a cancellation between discretisation error and the H/W=2 far-field bias (coarse -1.976 %, fine -0.738 %, and the midpoint-ring control also lands inside 3 %); m17194 states this limitation itself, so the pass is not overstated.
- run.py gates_verified.json gci_fine=0.0222 is computed on the a/W=0.20 ladder, not the ledger point - checked, not a contradiction.

## ⑥ Not run
- benchmarks/verified/fem_sent_j_integral/run.py --stage height/field (known CG non-convergence at H/W=3.0, recorded not_run in result.json) -> far-field truncation bias unquantified; caps the claim below L5
- full TensorFEM test suite (not required for this verdict)
- cluster/loongarch re-run (auditor rerun executed on the control node, x86_64)

## ⑦ Disposition
**Approve at L4 with notes.** Non-blocking: (a) guard `--fast` in `make_figure` (or skip the
ladder panel) so the documented fast variant stops crashing; (b) publish the six-section report
to `tensor_reports` and merge `agent/m17194` so downstream consumers import the entry from main.
