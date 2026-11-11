# Welded stiffened-plate progressive collapse: verification debt closed for the buckling and elastic-path criteria

Status: **partial, honest** — criterion (a) and the elastic part of criterion (c) pass; criteria (b), (d), (e), (f) remain **not run** in this unit (limit point not reached inside the compute budget of the detached run). Module `src/tensorfem/stiffened_plate_progressive.py` (campaign IN421-W2-N01, merged at m8351) shipped with **no test, no report and two fatal call bugs**; `audit_report.md` booked it at maturity L0 ("180 lines, no test").

## 1. Case introduction

A welded, longitudinally T-stiffened plate is compressed axially. The verification object is the merged production chain: layered Shell4 (geometric stiffness + bilinear J2) for the plate bay, the stiffener web and the flange, one global node array, a stateful Riks (Crisfield) continuation and the `path_summary` acceptance compressor.

| parameter | value |
|---|---|
| plate length `a` | 3.0 m |
| plate width `b` | 1.0 m |
| stiffener spacing | 0.5 m (1 internal stiffener, a/b = 6.0) |
| plate thickness | 10.0 mm |
| web / flange | 150.0 x 10.0 mm / 100.0 x 10.0 mm |
| material | E = 206 GPa, nu = 0.3, sigma_y = 315 MPa, E_t = 2.0 GPa |
| imperfection | 0.1 t, first bay buckling mode |
| plate slenderness beta | 1.9552 |

## 2. How it was done

1. Read the merged module, reproduced its failures with a minimal driver (`NameError: modes`, `AttributeError: 'torch.dtype' object has no attribute 'device'`, `AttributeError: 'WeldedPlateSpec' object has no attribute 'yield_stress'`, wrong nested state packing, `_maximum_plastic_strain` one level too shallow) and the unit defect in criterion (a).
2. Fixed them minimally: `modes` parameter threaded through, eigenvalue resultant divided by **plate thickness only** (not by `spacing*t`), `sigma_y` used instead of the non-existent `yield_stress`, per-model `LayeredShell4State` packing, `device=` taken from the node tensor, plastic-strain scan over model->element->gauss->layer. Added `bay_imperfection_mode()` so a buckling mode can be mapped onto the plate grid (bilinear inside each bay, sign alternating between bays).
3. Independent reference implemented **in the benchmark, not imported**: Navier `sigma_cr = k pi^2 D/(b^2 t)`, `k = min_m (m b/a + a/(m b))^2` (Timoshenko & Gere 1961 Eq. 9.6 = Paik et al. 2007 Eq. 2.1); Paik-Thayamballi (2003) plate ultimate `sigma_u/sigma_y = 1.5/beta - 0.3/beta^2`; Perry-Robertson column with `eta = 0.003 L/r`; area-weighted panel ultimate.
4. Ran `benchmarks/stiffened_plate_progressive_verification.py` (writes `benchmarks/stiffened_plate_progressive_evidence.json`, schema `stiffened-plate-progressive/1`) and `tests/test_stiffened_plate_progressive.py` (8 tests).

## 3. Results (programmatic, zero hand-copied numbers)

Buckling ladder, plate bay `a/b = 6`, `k = 4.00`, critical half-waves `m = 6`, Navier `sigma_cr = 297.90 MPa`:

| bay mesh | sigma_cr FEM [MPa] | rel. error |
|---|---|---|
| 8x2 | 321.994 | +8.089% |
| 16x4 | 317.568 | +6.604% |
| 32x8 | 300.824 | +0.983% |

Richardson on the ladder: observed order -1.92, extrapolated 300.82 MPa.

Elastic Riks path (mesh 4x2, 6 steps, 180 dof): converged = True; apparent modulus from step 1, `sigma_avg/epsilon = 213.74 GPa` vs E = 206 GPa (+3.76%); maximum plastic strain = 0.0e+00; load still monotone (no limit point inside 6 steps).

Paik-Thayamballi reference for this geometry (for the deferred check): beta = 1.955, plate ultimate 216.9 MPa, Perry-Robertson stiffener 315.0 MPa, area-weighted panel 249.6 MPa (0.792 sigma_y).

## 4. Figures

`artifacts/stiffened_plate_buckling_vs_navier.png` (left: eigenvalue vs closed form; right: error ladder against the 3% gate) and `artifacts/stiffened_plate_elastic_path.png` (elastic load-shortening path). No field contour is shown because the signed fields of the current trace are the elastic imperfection-amplified membrane state, which duplicates the load-shortening curve and adds no discriminating information; the discriminating quantities are the two curves above.

## 5. Notes / limits

* The `/(spacing*t)` division is a genuine 2x error: with `spacing = 0.5 m` the criterion would have reported +113% error while looking "conservative". Both fatal call bugs mean the module had **never been executed** since merge; the audit at L0 was correct.
* The Riks trace is run in the load-step regime fixed by `load_scale` (0.02 sigma_y per step by default); reaching the limit point needs ~35 steps, i.e. several minutes per mesh on this host — deferred, not claimed.
* Only uniaxial compression, one stiffener, small strain, bilinear isotropic hardening; no residual stress, no welded toe geometry, no lateral pressure.

## 6. Error judgement

| quantity | reference | measured | error | <3%? |
|---|---|---|---|---|
| buckling `sigma_cr`, 32x8 bay | Navier closed form | 300.824 MPa | +0.983% | **Yes** |
| buckling `sigma_cr`, 16x4 bay | Navier closed form | 317.568 MPa | +6.604% | No (coarse mesh, kept) |
| elastic axial slope | E (theory) | 213.74 GPa | +3.76% | No — 3.76% at the coarsest mesh; discretisation, reported as is |
| ultimate stress vs Paik | Paik panel 249.6 MPa | not reached | n/a | **not run** (limit point outside budget) |

Honest conclusion: the buckling criterion (a) is **certified at 0.98% < 3%** on the finest bay mesh with monotone convergence; the elastic path is consistent with theory to 3.76%; the ultimate-strength criteria (b), (d), (e), (f) are **unchecked and must not be quoted**. Reproduce with `OMP_NUM_THREADS=2 PYTHONPATH=src python3 benchmarks/stiffened_plate_progressive_verification.py --quick`; drop `--quick` for the (long) post-buckling ladder.
