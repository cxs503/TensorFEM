# XVAL — implicit IBM + Aitken relaxation (TensorFEM vs LIFE)

Anchor `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` (m17428, repair round of m17361).
Reference: **joconnor22/LIFE** (Lattice Boltzmann–Immersed boundary–Finite Element, GPLv3).
Line-cited port (no code copied): `src/Objects.cpp:26-60` (sub-iteration loop),
`:150-213` (predictor + Aitken), `:235-321` (epsilon conservation),
`src/IBMNode.cpp:26-136` (interpolate / direct force / spread), `:144-204` (support + ds),
`inc/Utils.h:220-233` (3-point delta), `src/FEMElement.cpp:99-116` (corotational force),
`src/FEMBody.cpp:244-289` (residual + Aitken numerator/denominator). Index:
TensorPlatform `docs/oss_inventory/FEAT-OSS-INVENTORY-LBM_LIFE.md`.

## Deliverables

| item | path |
|---|---|
| implementation | `src/tensorfem/ibm_aitken_implicit.py` |
| entry | `python -m tensorfem.ibm_aitken_implicit submerged n_steps=20` (`solve`, `cantilever_free_vibration`) |
| tests | `tests/test_ibm_aitken_implicit.py` — 13 tests, 12 pass / 1 fail |
| benchmark / repro | `python benchmarks/xval_implicit_ibm_aitken.py --quick` |
| run artifacts | `benchmarks/xval_implicit_ibm_aitken_artifacts/{evidence.json,xval_summary.md,xval_tip_history.png,xval_aitken_relaxation.png,runs/submerged_aitken_tip.tsv}` |

## Cross-validation table (ref vs got, tolerance 15 %)

| item | ref (source) | got | verdict |
|---|---|---|---|
| 3-point kernel partition of unity | 2.2e-16 (inventory §七.2) | 2.22e-16 | PASS |
| force transmission, eps = 1 | 11–37 % (inventory §七.2) | 61.1 % | FAIL (band mismatch) |
| force transmission, ε corrected | ≤ 9e-16 | 2.22e-16 | PASS |
| ε system residual |A ε − 1| | 0 | 2.22e-16 | PASS |
| total marker-force conservation | ~1e-16 | 6.2e-16 | PASS |
| beam static tip PL³/3EI | analytic | 0.99997× analytic | PASS |
| beam f1 (linearised eigen) | 1.0217 Hz (Euler–Bernoulli) | 0.1445 Hz | FAIL |
| beam vacuum free-vibration f | 1.0217 Hz | 43.04 Hz | FAIL (×42) |
| Aitken vs fixed ω=1 (scalar g=2.5) | optimal 1/(1−g) | not converged in 200 it | FAIL |
| submerged f_fsi | added-mass 38.87 Hz | NaN (interface diverged) | FAIL |

**Honest verdict: L2 delivery achieved (real implementation + entry + run artifacts + ref/got
benchmark + repro command); accuracy target NOT met.** The IBM operator chain is verified to
round-off against the LIFE-documented numbers; the structural transient and the coupled
behaviour are open failures, reported as such.

## Open defects / hypotheses

1. Beam transient frequency ×42 off while the static tip deflection is exact → stiffness law
   right, suspect the consistent mass / rotation-DOF mapping on the Newmark path.
   `scipy.linalg.eigh` segfaults (exit 139) on this loong64 image; the modal path was
   re-implemented with `numpy.linalg.eigvalsh` after Cholesky whitening.
2. Aitken does not reach the analytic optimum on the scalar probe and the submerged interface
   never reaches `sub_tol = 1e-8` (20/20 MAXIT hits for both Aitken and fixed ω=1) → suspect the
   stage order of `R_k` vs `U_km1` (predictor then solve) or the `relax_max` clamp.
3. Submerged run diverges once markers leave the lattice; re-test after (2).

## Friction

* LIFE executable **not run**: upstream needs Boost (absent) plus `using namespace std`
  compatibility fixes; a POSIX `boost::filesystem` shim got to the header parse of
  `inc/FEMNode.h`/`inc/FEMElement.h`. Hence every "ref" is analytic/inventory-documented, not a
  paired LIFE run — the documented next step.
* `scipy.linalg.eigh` → SIGSEGV (139) on loong64; prefer `numpy.linalg.eigvalsh`.

## Deviations from LIFE

1. Markers = beam centreline nodes (LIFE offsets to the surface + shape-function load projection).
2. Fluid = periodic D2Q9 BGK box with Guo half-split forcing; no regularised walls / convective
   outlet → only the coupling algorithm is in scope.
3. Structural tangent by central differencing (LIFE hand-codes `K_NL`).
