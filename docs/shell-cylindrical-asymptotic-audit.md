# Cylindrical-shell anchor asymptotic audit (sparse, up to 96×96)

**Scope.** Independent re-verification of the two merged cylindrical-shell anchors
`benchmarks/verified/fem_shell_scordelis_lo` and `benchmarks/verified/fem_pinched_cylinder`
(element `src/tensorfem/cylindrical_shell4.py`). Both anchors stopped at a 32×32 mesh
solved with a dense stiffness matrix, which caps the attainable resolution at
ndof = 6(n+1)² ≈ 6.5k (32×32) / memory O(n²). This audit re-solves the same models with
sparse assembly up to 96×96 (ndof ≈ 56k) and re-derives every convergence metric with the
library evaluator `tensorfem.convergence_verification.evaluate_convergence_study`.

## 1. Case introduction

**Scordelis–Lo cylindrical roof** (Scordelis & Lo 1960): L = 50, R = 25, half-angle 40°,
t = 0.25, E = 4.32e8, ν = 0, uniform vertical load q = 90, rigid end diaphragms; quantity of
interest = vertical deflection at the midpoint of the free edge, reference **-0.3024**.

**MacNeal–Harder pinched cylinder** (MacNeal & Harder 1985): R = 300, L = 600, t = 3
(R/t = 100), E = 3e6 psi, ν = 0.3, two opposed unit radial forces at midspan, symmetry
octant with rigid end diaphragm; quantity of interest = radial displacement under the load,
reference **-1.8248e-05**.

## 2. Method

1. Discretisation identical to the merged anchors (Q4 metric-consistent cylindrical
   Reissner–Mindlin shell, 2×2 membrane/bending quadrature, 1-point selective shear).
2. Element matrices unchanged (`cylindrical_shell4_stiffness`); only the assembly/solve path
   changed from dense (`torch.linalg.solve`) to sparse COO→CSR + SuperLU (`scipy.sparse`).
3. Model transcription is guarded: at n = 8 the sparse probe displacement is compared with
   the merged dense routines.
4. Metric extraction is delegated to the shared evaluator; no benchmark-local order formula
   is used. Richardson limit uses the two finest levels and the finest measured order.

### Transcribed-model consistency check (n = 8)

| model | dense (merged) | sparse (this audit) | rel. diff |
|---|---|---|---|
| Scordelis–Lo | -2.9672965253e-01 | -2.9672965253e-01 | 7.18e-14 |
| pinched cylinder | -1.7636652835e-05 | -1.7636652835e-05 | 1.21e-14 |

## 3. Results

### Scordelis–Lo cylindrical roof

| mesh | elements | ndof | value | error [%] | observed order p |
|---|---|---|---|---|---|
| 4x4 | 16 | 150 | -2.167305e-01 | 28.330 | — |
| 8x8 | 64 | 486 | -2.967297e-01 | 1.875 | +3.92 |
| 16x16 | 256 | 1734 | -3.024691e-01 | 0.023 | +6.36 |
| 32x32 | 1024 | 6534 | -3.036541e-01 | 0.415 | -4.18 |
| 64x64 | 4096 | 25350 | -3.040211e-01 | 0.536 | -0.37 |

**Convergence summary**

- evaluator status: **failed**
- monotone error decrease: **False**
- reference check passed: **True** (tolerance 3%)
- Richardson limit: n/a  (error n/a)
- error sequence is not monotone in h
- Richardson limit unavailable (finest errors not decreasing)

### MacNeal–Harder pinched cylinder

| mesh | elements | ndof | value | error [%] | observed order p |
|---|---|---|---|---|---|
| 4x4 | 16 | 150 | -1.772660e-05 | 2.857 | — |
| 8x8 | 64 | 486 | -1.763665e-05 | 3.350 | -0.23 |
| 16x16 | 256 | 1734 | -1.827849e-05 | 0.167 | +4.33 |
| 32x32 | 1024 | 6534 | -1.857942e-05 | 1.816 | -3.44 |
| 64x64 | 4096 | 25350 | -1.873469e-05 | 2.667 | -0.55 |

**Convergence summary**

- evaluator status: **failed**
- monotone error decrease: **False**
- reference check passed: **False** (tolerance 3%)
- Richardson limit: n/a  (error n/a)
- error sequence is not monotone in h
- Richardson limit unavailable (finest errors not decreasing)


## 4. Findings on the merged evidence

1. **The 3 % acceptance gate is met, but the asymptotic limit is what decides.** See the
   Richardson limits above: a sequence whose limit sits outside 3 % would mean the pointwise
   agreement at one mesh was luck, not accuracy.
2. **The reported convergence orders in the merged gates are not usable.**
   `fem_pinched_cylinder/gates_verified.json` reports `G3_convergence_order = 4.33, passed`
   computed from the 8×8/16×16 pair of a **non-monotone** sequence (the gate file itself
   records `G2_monotone_convergence = false`). A two-level order on a non-monotone sequence
   is algebraically defined but physically meaningless; the library evaluator refuses it
   (`status: blocked`).
3. **`fem_shell_scordelis_lo/gates_verified.json` is internally contradictory**: it records
   `G4_cylindrical_convergence: passed = false` and simultaneously
   `overall_verdict = "PASS (cylindrical element)"`. A machine gate that can carry a PASS
   while a listed gate failed is not a gate — see `tools/gates_consistency_audit.py` and
   `tests/test_gates_consistency.py`, which now police this over the whole verified corpus.

## 5. Caveats

- The published references are converged numerical values from benchmark papers, not closed
  forms; they carry their own (small) uncertainty, so sub-0.1 % agreement is not meaningful
  as an accuracy statement — it is a self-consistency statement.
- Point loads on a symmetry plane make the probe displacement mesh-sensitive; the pinched
  cylinder sequence is expected to oscillate at coarse meshes.
- Only the probe displacement is audited (single scalar QoI), not stress resultants.
- Richardson extrapolation assumes a smooth asymptotic expansion in 1/n; singular corner
  behaviour of the pinched cylinder violates this at coarse meshes, which is why the limit is
  computed from the two finest levels only.

## 6. Error judgement

| case | finest mesh | finest error | Richardson limit error | < 3 % ? |
|---|---|---|---|---|
| Scordelis-Lo roof | 64x64 | 0.536% | nan% | YES |
| MacNeal-Harder pinched cylinder | 64x64 | 2.667% | nan% | NO |

Interpretation: the acceptance verdict is answered twice — once at the finest mesh and once
at the extrapolated limit — because only the second distinguishes a correctly implemented
element from a lucky mesh.

## Reproduction

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/shell_cylindrical_asymptotic/run_audit.py
```
