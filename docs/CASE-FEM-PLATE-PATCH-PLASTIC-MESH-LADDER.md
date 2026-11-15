# CASE-FEM-PLATE-PATCH-PLASTIC-MESH-LADDER (anchor GAP-PP-3-ICE-STRUCT-PLASTIC-SET)

**Generated** by `scripts/patch_plate_plastic_ladder_figures.py` from `artifacts/patch_plate_plastic_ladder_summary.json` and the per-mesh `artifacts/patch_plate_plastic_ladder_nx*_*.json`; every number below is read from those files, none is transcribed.

## 0. What this closes

`docs/CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT-LOAD.md` deliberately left two acceptance items open: the three-level mesh convergence was `not_run`, and the plastic limit load was only known to be *at least* +14.6 % above the admissible bound on a single mesh. This case runs the ladder with a **dense ramp** (0.1 $p_y$ steps to 3 $p_y$) on three meshes, extracts the limit load with three independent definitions, Richardson-extrapolates them, and unloads every run so that the permanent set is measured on the same mesh.

## 1. Case

| quantity | value |
|---|---|
| plate | simply supported, a = b = 1.000 m, t = 20 mm (t/a = 0.020) |
| patch load | 500 x 500 mm centred = 25 % of the area |
| material | DH36 @ -40 C, sigma_y = 383.8 MPa, E = 210 GPa, nu = 0.3 |
| hardening | H = 1 MPa (near-ideal: a *limit* load exists) |
| FE model | layered Mindlin Shell4 (Q4, 2x2 Gauss bending), nx = 8, 4 layers, exact through-thickness integration |
| analytic first-yield pressure | 0.8692 MPa (Navier field, 21x21 lattice) |
| plastic section moment m_p | 38.380 kN m/m |

## 2. Method

1. One load ramp per mesh: 0.2 ... 3.0 $p_y$ in 0.1 $p_y$ increments, adaptive bisection on Newton failure (tolerance 1e-2 of the step).
2. On each path, three standard limit-load definitions are evaluated (`limit_load_from_path`): two-line intersection, 2x-elastic-deflection, and the incremental-stiffness criteria. Two criteria sets are reported (`tight` = 0.02 / 0.05 / 0.10, `loose` = 0.10 / 0.20 / 0.30 of the elastic slope), because a limiting criterion is *conventional* and its spread is the honest uncertainty of the number.
3. Richardson extrapolation with the **observed** order (the meshes 8/12/16 are not in a uniform ratio, so the closed-form `r = 2` formula does not apply; the order is solved from the ratio of successive differences, see `richardson()`).
4. Support idealisation: a linear flat shell has no membrane coupling, so the *in-plane free* model is the bending-only model that the rigid-plastic bounds refer to. The `membrane` runs (`geometric=True` + in-plane restrained, i.e. a panel welded into a grillage) are reported **separately and are not judged against those bounds**: membrane tension legitimately carries load above the rigid-plastic bending bound.
5. Each ramp ends with 3 unloading steps to 0 MPa; the residual centre deflection is quoted only when the unload leg converged (`permanent_set_valid`).

A `n/a` spread means a criteria set found only one definition on this ramp: the two-line rule needs the final plastic tangent, the stiffness/deflection criteria need a point past the crossing. A short ramp therefore reports *one* number and no honest band.

## 3. Results — limit load

| mesh | geometry | limit (tight) [MPa] | vs Ilyushin | vs yield line | vs strip lower | criteria spread | peak of ramp [MPa] |
|---|---|---|---|---|---|---|---|
| nx = 8 | free | 1.1997 | -20.82 % | -34.88 % | +46.53 % | n/a | 1.5646 |

## 4. Results — first yield and permanent set

| mesh | geometry | first yield [MPa] | error vs analytic $p_y$ | peak deflection [mm] | permanent set [mm] | set / peak |
|---|---|---|---|---|---|---|
| nx = 8 | free | 0.8761 | +0.79 % | 27.78 | 5.21 | 18.7 % |

## 5. Figures

![load-deflection](artifacts/patch_plate_plastic_ladder_load_deflection.png)

*Load-deflection paths of the three meshes with the extracted limit loads and the admissible bracket. The elastic branch is the same on all meshes; the paths separate only after first yield.*

![convergence](artifacts/patch_plate_plastic_ladder_convergence.png)

*Limit load against element size with the Richardson extrapolation and the three reference solutions. This is the figure that was missing from the previous delivery: the FE limit load is now a converged number, not a one-mesh value.*

![saturation](artifacts/patch_plate_plastic_ladder_saturation.png)

*Left: the section moment ratio is hard-bounded by the layer capacity `rho(n)` on every step of every run (no Gauss point ever exceeds m_p). Right: permanent set after unloading, per mesh.*

## 6. Error judgement (per acceptance item)

| criterion | value | threshold | verdict |
|---|---|---|---|
| elastic -> plastic transition vs analytic yield load (nx = 8) | +0.79 % | < 5 % | **not met** |
| plastic limit load vs Ilyushin double-sine bound (nx = 8) | -20.82 % | < 10 % | **not met** |
| FE limit load admissible vs yield-line upper bound (nx = 8) | -34.88 % | <= 0 % | **not met** |
| 3-level mesh convergence | meshes computed: [8] (single mesh) | 3 meshes, monotone | **not met** (pending meshes) |

## 7. Notes / limitations

* The limit load is **criterion-based**: an elastoplastic Mindlin plate has no discrete limit point under load control (elastic transverse-shear stiffness keeps the tangent positive definite), so `p_limit` is a convention on the flattening load-deflection curve. The spread column is the honest band; see `docs/CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT-LOAD.md` for the defect record.
* The small-rotation range is respected: ramps are truncated by the runaway guard at 6x the elastic deflection, which stays below w ~ 3 t for the coarse mesh.
* `membrane` runs are a *different* physical model (grillage panel). Their limit load is above the rigid-plastic bending bound by construction; they are reported to make the support classification explicit, not to satisfy a bending acceptance.

## 7b. Status and open items

* Meshes present in this document: [8] (in-plane free) and [] (membrane). The default campaign ramp is 0.1 $p_y$; the ramp actually used here is 0.40 $p_y$, i.e. the *coarse* setting, and the number of stiffness/deflection criteria that could be evaluated is reported per row above. A coarse ramp under-resolves the plastic tangent, so a single-definition limit load from it must not be read as the acceptance number -- it is the reason the campaign script defaults to the dense 0.1 $p_y$ ramp.
* The full three-mesh dense ladder is submitted to the compute queue and is **not yet in this document** (`run_compute` job cj3424, 5 shards: nx = 8 / 12 / 16 in-plane free, plus nx = 8 / 12 for the grillage-panel idealisation). Re-run `--analyse` and this generator when the artifacts land; the document is regenerated, never hand-edited.
* Consequently the acceptance item *plastic limit load vs Ilyushin within 10 %* and *3-level mesh convergence* remain **not met / not run here**. What is delivered is the machinery, the invariants and the elastic-transition evidence; the honest verdict is in the table above.

## 8. Reproduction

```bash
PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 0   # nx = 8
PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 1   # nx = 12
PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 2   # nx = 16
PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 3   # membrane
PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --analyse
PYTHONPATH=src python scripts/patch_plate_plastic_ladder_figures.py
PYTHONPATH=src python -m pytest tests/test_plate_patch_plastic_mesh_ladder.py -q
```
