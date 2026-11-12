# Board audit — m16076 `CASE-FEM-PINCHED-CYLINDER` (TensorFEM)

**Audit mission:** m16175 (cross-check / 交差核验, not a redo) · **Verdict by:** `board-audit` · **Target mission:** m16076 (status `succeeded`, PR #756 merged as `43745cdb`)
**Audited commit:** `5dbf8c83` ("CASE-FEM-PINCHED-CYLINDER: sparse path + case-pool driver …"), merged into the audited worktree at `43745cdb`
**Audit worktree:** `/root/tp-agent/work/m16175` (TensorFEM, branch `agent/m16175`)

---

## ⓪ Disposition

**Level 4 — benchmark validation (complete case + error analysis + reproducible).**
Core criterion of the anchor (load-point deflection at the 16×16 octant mesh within 5 % of MacNeal–Harder `-1.8248e-5 in`) **passes at every one of the 8 rungs**, the driver re-runs bit-identically from one command, and the error analysis is real (ladder + extrapolation + 4 sensitivity sweeps). Three findings are attached, none of which changes the level:

* **F1 MEDIUM (scientific, disclosed by the delivery):** the 0.167 % at 16×16 is an *error cancellation*, not accuracy. Against the driver's own extrapolated discrete limit the 16×16 rung is 3.256 % away, and that limit is 3.539 % off the reference. The delivery states this itself; the audit reproduces it independently (§④).
* **F2 MEDIUM (report hygiene):** the published report (id 15225) marks the **failed** shear-factor gate G5 (`1.308 %` vs a predeclared `1.0 %`; `case_gates.json` → `passed: false`) with a ✅ in section ⑥, while its own summary line correctly says "5/6 项通过，总体 FAIL". One bullet contradicts the machine gate.
* **F3 LOW (bookkeeping, same pattern as m16071/F3):** `tensor_benchmark_cases` has **0 rows** for `CASE-FEM-PINCHED-CYLINDER`, so the board's machine judge has no registered row to adjudicate; `tensor_case_pool` does hold `FEM-PINCHED-CYLINDER` (repo TensorFEM, mission 16076) but its status is still `dispatched` — never flipped to a verified/passed state after the mission succeeded.
* **F4 LOW (delivery hygiene):** the three PNGs did not survive the merge into main (binary filter). They are regenerated deterministically by the repro command; copies are in `artifacts_repro/figures/` with byte-identical sizes to the originals (79 635 / 60 098 / 109 523 B).

---

## ① Case (physics / geometry / parameters)

MacNeal–Harder "pinched cylinder" (MacNeal & Harder 1985, *Finite Elem. Anal. Des.* 1(1):3–20, DOI 10.1016/0168-874X(85)90003-4): thin cylindrical shell, length `L = 600 in`, radius `R = 300 in`, thickness `t = 3 in`, `E = 3e6 psi`, `ν = 0.3`, rigid end diaphragms (cross-section translation restrained, axial translation free), two diametrically opposed **unit** radial forces at mid-span; measured quantity = radial deflection of the shell **at the load point**. Reference value quoted with the problem statement: `w_ref = -1.8248e-5 in` (no closed form).

| dimensionless | value |
|---|---|
| `R/t` | 100 |
| `L/R` | 2.0 |
| `ν` | 0.3 |
| `P/(E t²)` | 3.704e-8 |

## ② Method / implementation (what the delivery actually built)

1. **Reuse, not a new solver**: the pre-existing production element `tensorfem.cylindrical_shell4.cylindrical_shell4_stiffness` (Reissner–Mindlin Shell4, exact cylindrical metric, 2×2 membrane+bending, 1-point selective transverse shear).
2. `src/tensorfem/pinched_cylinder.py:27` `pinched_cylinder_model()` — symmetry **octant** `x∈[0,300]`, `θ∈[0,π/2]`, corner load `P/4`, mid-span plane `u=0`+in-plane rotations, circumferential reflection planes, end-ring diaphragm `v=w=0` (`u` free; `axial_u_clamped` only as a diagnostic).
3. `:77` `solve_pinched_cylinder_linear()` — dense elimination (the pre-existing production path).
4. `:92/:117` `pinched_cylinder_sparse_system()` / `solve_pinched_cylinder_sparse()` — **new** CSR assembly + SuperLU, so the ladder reaches `N=64` (25 350 DOF) where the dense operator would need > 5 GB.
5. Entry point: `benchmarks/verified/fem_pinched_cylinder/run.py` — 8-rung ladder (4…64), dense/sparse equivalence, two-model extrapolation, shear / drilling / end-condition / anisotropic sweeps, 3 figures + `case_result.json` + `case_gates.json`.

## ③ Results — programmatic, zero hand-copy (recomputed by the audit)

`PYTHONPATH=src python benchmarks/verified/fem_pinched_cylinder/run.py` → **540.6 s wall** on this host (loongarch64, py 3.11.6 / torch 2.7.1 / scipy 1.16.0 / numpy 2.4.6 — identical stack to the recorded `environment` block).

| mesh | DOF | got (in) | ref (in) | err vs ref | `<5 %` | err vs own discrete limit |
|---|---|---|---|---|---|---|
| 4×4 | 150 | -1.772660e-05 | -1.8248e-05 | 2.857 % | ✅ | 6.177 % |
| 8×8 | 486 | -1.763665e-05 | -1.8248e-05 | 3.350 % | ✅ | 6.653 % |
| 12×12 | 1014 | -1.806824e-05 | -1.8248e-05 | 0.985 % | ✅ | 4.369 % |
| **16×16 (judge point)** | 1734 | **-1.827849e-05** | -1.8248e-05 | **0.167 %** | ✅ | 3.256 % |
| 24×24 | 3750 | -1.848067e-05 | -1.8248e-05 | 1.275 % | ✅ | 2.186 % |
| 32×32 | 6534 | -1.857942e-05 | -1.8248e-05 | 1.816 % | ✅ | 1.663 % |
| 48×48 | 14406 | -1.868019e-05 | -1.8248e-05 | 2.368 % | ✅ | 1.130 % |
| 64×64 | 25350 | -1.873469e-05 | -1.8248e-05 | 2.667 % | ✅ | 0.842 % |

Reproduction fidelity: all **8/8 rung displacements are bit-identical** to the delivered `case_result.json` (the only differences in the file are `solve_seconds`, this host being 1.3–2.7× slower: 204.7 s vs 76.9 s at N=64, and the re-run timestamp).

Self-consistency: dense vs sparse `1.292e-13` rel (threshold 1e-10); drilling factor `1e-10…1e-4` → `1.03e-11 %`; shear factor `k∈{5/6,1.0,0.5}` → max `1.308 %` (**fails the predeclared 1 % gate**).

## ④ Independent error forensics (audit's own arithmetic)

* **Cancellation, quantified.** Errors relative to the driver's own limit (`u_lim = -1.889371e-05 in`) fall monotonically from 6.18 % (4×4) to 0.84 % (64×64), while errors relative to the published reference go 2.86 → 3.35 → 0.99 → **0.17** → 1.28 → 1.82 → 2.37 → 2.67 %. The 16×16 "0.167 %" is `3.26 %` of stiffer-than-limit discretisation cancelled against `3.54 %` of more-flexible-than-reference model bias. **Reading 0.167 % as element accuracy is wrong**; reading it as "the declared 5 % acceptance criterion is met" is correct.
* **Extrapolation is model-dependent, and one alternative is near the edge.** Power law `u_inf + C N^-p` → `p = 0.97`, `u_lim = -1.889371e-05` (3.539 % off ref, residual 8.1e-18 in² ≈ 7e-5 relative); pure log model `a + b ln N` → `-1.742495e-05` (**4.510 %** off ref, residual 5.0e-15 in²). Both are inside 5 %, the log one only just; the delivery records both and does not hide the loser.
* **The `p ≈ 0.97` exponent is physically consistent** with a point-load singularity (algebraic, near-first-order), i.e. the non-monotone ladder is not an implementation accident but the expected signature of a point-loaded thin shell.
* **Not a "loose constraints to fake small error" case**: clamping the end-ring axial translation stiffens the load-point response by ~16.6 % (the audit's `test_axial_end_clamp_stiffens_the_diaphragm_model` re-passed), so the delivered small error does not come from an over-relaxed boundary condition.

## ⑤ Pitfalls / limits

1. The single number the anchor asks for is the least trustworthy point on the ladder (error cancellation). Any future "finest mesh = most accurate" automation would also mislead here in the opposite direction (N=64 is 2.67 %).
2. `tests/test_pinched_cylinder.py::test_case_pool_pinched_cylinder_16x16_within_five_percent` freezes that cancelling value under a 5 % assertion — a genuinely better element would move N=16 *up* toward its limit and could trip this test. It is a fragile guard, worth knowing before anyone "fixes" the element.
3. `test_transverse_shear_correction_is_not_a_fit_parameter` checks `k=1 vs 5/6` at **N=8** (passes < 1 %), while the case driver's predeclared gate checks it at **N=16** and fails (1.308 %). The unit test therefore under-covers the gate it appears to stand for.
4. Shear-factor sensitivity (1.3 %) means the answer is not fully modelling-parameter-free at the 1 % level; it is inside the 5 % case tolerance, but a future "1 % accurate" claim would need a shear treatment that is insensitive by construction.
5. Registered-row gap (F3): with no `tensor_benchmark_cases` row, `maturity_gate` cannot re-derive `got vs tol` for this anchor; my verdict therefore carries the numbers explicitly in the evidence `benchmark` block.

## ⑥ Error judgement — explicit answers

| quantity | got | ref / criterion | error | `< 3 %` ? |
|---|---|---|---|---|
| load-point deflection, 16×16 (declared judge point) | -1.827849e-05 in | -1.8248e-05 in, tol 5 % | **0.167 %** | **yes (0.167 % < 3 %)** |
| finest mesh 64×64 | -1.873469e-05 in | same | 2.667 % | yes |
| extrapolated discrete limit (power) | -1.889371e-05 in | same | 3.539 % | **no (honest: > 3 %, still < 5 %)** |
| extrapolated discrete limit (log, not selected) | -1.742495e-05 in | same | 4.510 % | no |
| dense vs sparse operator | 1.292e-13 rel | ≤ 1e-10 | — | yes |
| shear-factor spread `k∈{0.5,5/6,1}` | 1.308 % | predeclared ≤ 1 % | — | **fail (G5, disclosed)** |
| drilling-factor spread `1e-10…1e-4` | 1.03e-11 % | predeclared ≤ 0.1 % | — | yes |

**Answer to "是否 < 3 %": at the declared judge point yes (0.167 %); at the model's own converged limit no (3.54 %).** Both statements are in the delivery; the report only needs the second one flagged as ❌ rather than ✅ where it applies to G5 (F2).

## Checks performed / not performed

* Performed: read the audited commit and diff; re-ran the full `repro_cmd` (540.6 s, 8/8 bit-identical rungs, gates reproduced); ran `PYTHONPATH=src pytest tests/test_pinched_cylinder.py -q` → **6 passed in 55.23 s**; recomputed the ladder, both extrapolation models and all sensitivity numbers from the regenerated JSON; queried `tensor_benchmark_cases`, `tensor_case_pool`, `tensor_dev_missions`, `tensor_reports`; grepped integration callers.
* **Not performed**: independent re-derivation of `1.8248e-5` with a different element family (would need a second shell discretisation — the 3.5 % model bias is therefore *attributed* to the point-load singularity, not *proven*); access to the 1985 paper text from this environment (reference value is the one carried in the problem statement, not re-transcribed from the table/page as the mission asked for).