# Board audit m17408 — m17361 `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`

**Audit mission**: m17408 (board-audit, 交差核验 — verification, not re-do)
**Audited mission**: m17361 `【锚点=XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION】【第1轮·任务榜·开源对标】` (state: succeeded)
**Repo**: TensorFEM · audited branch `agent/m17361` · audit branch `agent/m17408`
**Referee criterion**: mission goal — `<15 %` vs the OSS/reference implementation (owner precision standard); `tensor_benchmark_cases` has **0 registered rows** for this anchor (F4).
**Audit inputs (byte copies, `artifacts/audit_m17361/raw/`, sha256 in `artifacts/audit_m17361/sha256_manifest.txt`)**: `run_xval.py`, `gen_figures.py`, `xval_aitken_ibm_report.md`, `xval_results.json`, `verdict.json`, `aitken_comparison.json`, `algorithmic_comparison.json` from `/root/tp-agent/work/m17361`.
**Independence limits**: no network in the sandbox → the LIFE repository was **not** fetched; the delivered "LIFE (typical)" parameters are therefore an unverified transcription (`WITHHELD`, F6). Nothing in `src/`, `tests/` was modified by this audit; measurements were re-run in the audit worktree only.

---

## 1. Deliverable inventory (the three-piece check)

| piece | delivered? | evidence |
|---|---|---|
| implementation | **pre-existing**, not delivered by m17361 | `src/tensorfem/fsi_lbm_strong_coupling.py` (767 lines) landed in merge **#585 (m8136)**; `git log --oneline -1 -- src/tensorfem/fsi_lbm_strong_coupling.py` → `1164e2d4 Merge #585 (m8136)` |
| entry / harness | yes, **untracked** | `benchmarks/xval_aitken_ibm/run_xval.py` (sha256 `e12cd0de…`), `gen_figures.py`; `git status` in m17361 → `?? artifacts/xval_aitken_ibm/`, `?? benchmarks/xval_aitken_ibm/`, `?? docs/xval_aitken_ibm_report.md` |
| run artifacts | yes, **untracked** | `artifacts/xval_aitken_ibm/xval_results.json` (`0aef28f1…`), `verdict.json` (`f1b1b18b…`), `aitken_comparison.json` (`359e0645…`), `algorithmic_comparison.json` (`905417a9…`), `figures.json` (3 base64 PNGs) |
| report | markdown only, **never published** | `docs/xval_aitken_ibm_report.md` (`db46a346…`) exists on disk; `SELECT … FROM tensor_reports WHERE mission_id=17361` → **0 rows** |
| commit | **none** | `agent/m17361` tip == `a1572ffb Merge #868 (m17230)`, `git diff --stat <tip>` empty → m17361 produced no commit of its own (F3) |

Repro (`repro_cmd`, re-run by the audit in a fresh worktree):
`cd benchmarks/audit_m17361/xval_aitken_ibm && python3 run_xval.py` → **15.8 s**, and every JSON leaf compared: **117/117 identical** to the delivered `xval_results.json` (`artifacts/audit_m17361/repro_stdout.log`). The delivered artifacts are therefore reproducible and deterministic.

## 2. What the registered referee says

`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION'` → **0 rows** (0 rows for every `XVAL-FEM%` anchor as well). The `<15 %` gate exists only in the mission goal text and is hard-coded as `0.15` in `run_xval.py:442`. The audit therefore judges against the mission goal plus the external Turek-Hron values the script itself declares (`CYLINDER_2D2_REF` at `run_xval.py:52-56`).

## 3. Findings (severity-ranked)

### F1 — **blocking** · drag coefficient double-counted (measurement defect)
`run_xval.py:256-257` defines `q_dyn = 0.5*u_lat**2*cells_per_D` — i.e. `0.5·ρU²D` — and then computes `CD_mean = 2.0*hist[:,0].mean()/q_dyn`. `ChannelLBM.momentum_exchange_force()` **already** returns the doubled momentum transfer (`src/tensorfem/lbm_channel2d.py:427-428`: `F[0] += 2.0*int_mom*cxj`), and the repository's own test uses the consistent normalisation `cd = 2.0*F/(u_lat**2*cells)` = `F/q_dyn` (`tests/test_fsi_lbm_strong_coupling.py:84`). The delivered script is therefore **exactly 2× too large**.
Evidence: delivered `CD = 6.022`; restated `CD = 3.011` = `6.022/2` (audit `recheck_metrics.json`, check `P1_CD_fixed_matches_half_delivered = true`).
Consequence: the declared `fluid_solver: FAIL — CD error 86.4 %` (`artifacts/xval_aitken_ibm/verdict.json`) is **not a solver result**; restated, `CD` is **−6.8 %** vs 3.23 → **passes <15 %**.

### F2 — **blocking** · Strouhal number uses a wrong lattice→physical time step
`run_xval.py:261`: `dt_phys = dx/(u_lat*U_mean/D)`. The solver itself uses `self.dt = self.u_lat*self.dx/p.U_mean` (`src/tensorfem/fsi_lbm_strong_coupling.py:578`). Ratio = `D/u_lat²` = **15.625**, so the delivered frequency — and hence `St` — is 15.625× too small.
Evidence: delivered `St = 0.01616`; restated `St = 0.2526` (check `P2_dt_ratio_15.625 = true`); the patched script prints `St: 0.253 (ref 0.295, err 14.4 %)`.
Consequence: the declared `St error 94.5 %` is a **unit bug**, not a discretisation error. Restated error **−14.3 %** → passes <15 % only barely (margin 0.7 pp).

### F3 — **blocking** · the whole delivery is untracked and the report was never published
`git status` in the m17361 worktree shows the three deliverable paths as `??`; `agent/m17361` has no commit. Nothing m17361 produced is in git history, and `tensor_reports` has no row for mission 17361. If the worktree is recycled the evidence is gone. (Byte copies are preserved here: `artifacts/audit_m17361/raw/` + `sha256_manifest.txt`.)

### F4 — **major** · no machine referee for the anchor
0 rows in `tensor_benchmark_cases` for this anchor (and for every `XVAL-FEM%`). The mission's acceptance line exists only as prose in the mission goal and as a local constant, so no gate can machine-check future revisions. Remediation: register `CD/CL_amp/St` rows with `tolerance = 0.15` **and** the restated formulas (F1/F2) before the anchor is re-run.

### F5 — **major** · the FSI half of the anchor is a smoke test, and the report's claim about it is overstated
`run_short_fsi_test()` (`run_xval.py:335-374`) runs 300 steps at `cells_per_D=10`, `tolerance=1e-3`, and records `n_sub_iterations[-5:]`; `converged` is computed as `all(s < 10 …)` over the last 10 entries — i.e. the "converged" predicate is *sub-iteration count < 10*, **not** the residual tolerance it was run with. `FSI2_REF = {frequency 2.0 Hz, amplitude 0.034 m}` is defined at `run_xval.py:46-49` and then **never used**: there is no tip-amplitude/frequency comparison against Turek-Hron FSI-2 anywhere in the delivery. The report's "all sub-iterations converged within tolerance" (§4.4) is supported by 5 recorded snapshots out of 30 recording windows and by the wrong predicate.

### F6 — **major** · the OSS side of the "cross-validation" was never executed
No LIFE checkout exists in the m17361 worktree; `aitken_comparison.json`'s "life_aitken" trace is produced by re-running **our own** `AitkenRelaxation` with LIFE-ish bounds (`omega0=0.5, min=0.1, max=1.0`, labelled "typical" in `run_xval.py:59-65`). Both traces are therefore the same code (the delivered `omegas` lists are point-wise proportional, 2.0/1.0 clamps). What is actually established is agreement of the `Δ²` formula with Küttler & Wall (2008) — a **unit-level** result, not an implementation-to-implementation comparison. The external anchor is `WITHHELD` (no network at audit time).

### F7 — **minor** · unsupported grid-convergence table
Report §5.2 attributes `D/dx 12 → ~15 % CD error, 20 → ~11 %` to "existing FSI validation P153W2-N03". No such artifact exists in the repo (`docs/P153W2-N03_fsi_lbm.json` absent; the only tracer is `benchmarks/fsi_strong_coupling/make_acceptance.py`, which *would* write it). The table also contradicts the delivery's own 86 % at `D/dx=12`.
The audit measured the missing data itself (`recheck_metrics.json`): restated `CD` −6.8 % → **−9.5 %**, `St` −14.3 % → **−8.6 %** for `D/dx 12 → 20`. Predeclared expectation `P4` ("CD and St both move towards the reference") is **partially falsified**: `St` improves, `CD` does **not** (slightly worse). Recorded as-is.

### F8 — **minor** · `CL_amp` fails for a prosaic reason: the averaging window is too short
`CL_amp` uses a consistent normalisation, so F1/F2 do not affect it: delivered −62.6 %. Re-running with a 2000-step average gives **−29.2 %**, and at `D/dx=20` **−41.2 %** (checks `P3_*`): the failure is dominated by an under-converged 1000-step window and secondarily by the reference mismatch (see F9), not by the relaxation. The delivered script has no periodicity/convergence guard before averaging.
Audit did **not** establish a passing `CL_amp`; it stays an honest FAIL for this capability at these grids.

### F9 — **minor** · reference provenance mismatch
The run is a **bare rigid cylinder** at `u_lat=0.08, Re=100`, while `CYLINDER_2D2_REF` (`CD 3.23, CL_amp 1.0, St 0.295`) is quoted from the Turek-Hron "2D-2"/FSI-2 configuration that carries the elastic beam. The label "cylinder only" in `run_xval.py:51` is not what the numbers are. Also, the validation of record for the structure side is analytic-vs-FE with 10 cubic Hermite elements, where the static case is *exactly* representable (`error 4.2e-13`), so that assertion is weak by construction; and `static_deflection.pass` is stored as the **string** `"True"` (`xval_results.json`), a type defect that would break any machine gate.

### F10 — **minor** · orphan artifact
`artifacts/xval_aitken_ibm/aitken_comparison.json` has **no producer script** in the tree (`grep -rn "aitken_comparison" --include=*.py` → only the audit's own report mentions it): its 20-iteration traces cannot be regenerated from the delivery.

### Proposed repair (not applied to another mission's delivery)
`artifacts/audit_m17361/fix/run_xval_measurement_fix.patch` — a 2-line patch (F1, F2). Applied to a copy and re-run by the audit: `CD 3.011 (−6.8 %)`, `St 0.253 (−14.4 %)`, `CL_amp 0.374 (−62.6 %)`, i.e. **2 of 3 fluid metrics pass the <15 % gate** once measured consistently. Repair of the delivered file itself belongs to the reopening mission.

## 4. Level judgement (L0–L5)

**Verdict: L2.**

* **L0 (docs)** — exceeded: real code, tests and artifacts exist.
* **L1 (math/unit verification)** — **met**: `Δ²` Aitken update matches Küttler & Wall (2008) with the signed clamp (`fsi_lbm_strong_coupling.py:503-531`); Hermite FE structure matches the analytic cantilever to 8.6e-7 (frequency) and the IBM/Peskin kernel conserves momentum; `pytest tests/test_fsi_lbm_strong_coupling.py` → **12 passed in 14.76 s** (audit run).
* **L2 (independent functional implementation)** — **met**: `src/tensorfem/fsi_lbm_strong_coupling.py` is a genuine implicit (sub-iteration) IBM Dirichlet-Neumann scheme with Aitken relaxation, `ChannelLBM` fluid, Hermite/modal structure: input → compute → output, exercised end-to-end by the 300-step coupled run (final tip displacement 5.451e-3 m, finite, drag positive).
* **L3 (system integration)** — **not met**: callers are `benchmarks/fsi_strong_coupling/run_validation.py:39,180`, `run_chunk.py:22-44` and the module's own `__main__` (`:758`). Nothing in `api/`, not exported from `src/tensorfem/__init__.py` — the capability is reachable only from benchmark harnesses and tests (same rule applied in the m15396 audit).
* **L4 (benchmark validation)** — **not met**: (a) no registered referee row (F4); (b) the delivered ref-vs-got table is measured with two formula defects that flip 2 of its 3 "FAIL" verdicts (F1, F2); (c) the anchor's FSI-2 quantitative comparison is absent (F5); (d) the OSS side was never executed (F6). A benchmark whose error column is wrong cannot certify L4 even when the underlying solver turns out to be fine.

## 5. Checks performed / not performed

Performed: three-piece existence + sha256; `git status`/history of `agent/m17361`; referee query on `tensor_benchmark_cases`; published-report query on `tensor_reports`; full re-run of the delivered harness (117/117 JSON leaves identical); restatement of every fluid metric from the same force history with the repository's own formulas (`recheck_metrics.py`, predeclared P1–P4); grid/window sensitivity (`D/dx` 12 with 1000- and 2000-step averages, `D/dx` 20); `pytest tests/test_fsi_lbm_strong_coupling.py` (12 passed); formula/unit derivations at `lbm_channel2d.py:427`, `fsi_lbm_strong_coupling.py:578`, `tests/test_fsi_lbm_strong_coupling.py:84`.

Not performed (`unknown / not_run`, **not** passed): fetching or running joconnor22/LIFE (no network → external anchor `WITHHELD`); any 3-D LBM path; a momentum-budget (control-volume) independent check of the drag coefficient — the F1 correction rests on the repository's own normalisation plus the delivered-`CD`/2 identity, not on a second independent force estimator; establishing a passing `CL_amp` (still FAIL at every grid/window tried); long-time periodicity proof for the lift history.

## 6. Claims of m17361 that the evidence does not support

1. "fluid_solver: FAIL — CD error 86.4 % / St error 94.5 %" → measurement artefacts (F1, F2); restated −6.8 % / −14.3 %.
2. "All sub-iterations converged within tolerance" → predicate is `sub_iterations < 10`, and only the last 5 of 30 recorded windows are in the artifact (F5).
3. Grid-convergence table in §5.2 (D/dx=20 → ~11 %/~10 %) attributed to P153W2-N03 → no such artifact is committed; audit's own numbers are −9.5 % (CD) and −8.6 % (St), with CD *not* monotone (F7).
4. "Cross-validation against joconnor22/LIFE" → no LIFE code was run; the "LIFE" trace is our own class with different bounds (F6). The honest content is "formula-level agreement with Küttler & Wall (2008)".

## 7. Disposition

**Request changes (reopen).** The capability itself is L2-worthy and the Aitken/IBM machinery is real and tested, but the delivering mission's acceptance evidence is defective at the measurement layer and untracked at the VCS layer, and the anchor's OSS side was never executed. Reopen for: apply the F1/F2 patch, re-run and re-table (with a periodicity guard and a matched reference for `CL_amp`), commit the harness + artifacts, register the three `tensor_benchmark_cases` rows at `tolerance=0.15`, and (for L3) expose the coupling through a non-benchmark caller.

Artifacts: `artifacts/audit_m17361/{raw/,sha256_manifest.txt,repro_stdout.log,recheck_metrics.json,fixed_run_results.json,fix/,figures.json,audit_m17361_metrics.png}`; scripts `benchmarks/audit_m17361/{recheck_metrics.py,make_figure.py}`.