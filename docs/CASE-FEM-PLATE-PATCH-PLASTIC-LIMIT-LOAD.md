# CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT-LOAD

Anchor `GAP-PP-3-ICE-STRUCT-PLASTIC-SET` — **second-generation correction of the
plastic limit-load claim**. The previous certificate
(`docs/CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT.md`) reported the *end of a load ramp*
as the plastic collapse load and compared it with the yield-line upper bound
(2.6077 MPa, **+41.55 %**, an inadmissible over-strength for a bending mechanism),
then attributed the mismatch to membrane action and to a "1.50x layering
correction". Both statements are **retracted** here with measurements.

Everything below is emitted by
`scripts/patch_plate_plastic_limit_load.py` into
`artifacts/patch_plate_plastic_limit_load_nx8_n4_s1.json`; nothing is transcribed.

## 1. Case

| quantity | value |
|---|---|
| plate | simply supported, a = b = 1.000 m, t = 20 mm (t/a = 0.020) |
| patch load | 500 × 500 mm centred = 25 % of the area |
| material | DH36 @ −40 °C, sigma_y = 383.8 MPa, E = 210 GPa, nu = 0.3 |
| hardening | H = 1 MPa (near-ideal plasticity: the limit load is a *limit* load) |
| FE model | layered Shell4 (Q4, 2×2 Gauss bending, 1-point shear), 8×8, **4 layers**, exact through-thickness integration |
| analytic first-yield pressure | 0.8692 MPa |

## 2. What was wrong with the old number

| defect | evidence |
|---|---|
| the reported "collapse" was the last prescribed load factor | previous ladder ran to 3.0 × P_y = 2.6076 MPa and **every step converged**; `PatchPath.collapse_is_limit_point` was already `False` there |
| the "membrane explanation" is impossible in this model | the Shell4 is a **linear flat shell**: membrane strain `B_m·(u,v)` does not involve `w`, and the measured in-plane resultants are **N = 0.000 kN/m** at every load level |
| the "1.50x layering correction" was a coincidence | the section capacity of a layered section is `m_p (1 − 1/n²)` = 8/9 = **0.889** at n = 3 — exactly the measured `|M|/m_p` plateau, and it is a *deficit*, not an over-strength |

## 3. New closed-form result: layered section plastic moment

```
m_p,n = sum_k sigma_y |z_k| (t/n) = m_p * rho(n),   rho(n) = 1 - 1/n^2 (n odd),  1 (n even)
```

implemented as `section_moment_capacity` and verified three ways:

| n | measured FE `|M|/m_p` plateau | closed form | note |
|---|---|---|---|
| 3 | 0.8889 | 8/9 = 0.8889 | deficit (the old "correction") |
| 4 | 1.0000 | 1.0000 | **even layer counts are exact** — used for certification |

The new invariant `max_section_moment_ratio` = `sqrt(Mx² − MxMy + My² + 3Mxy²)/m_p`
is recorded at every path point and is a **hard bound**: no Gauss point may ever
exceed `rho(n)` (asserted in
`tests/test_plate_patch_plastic_limit_load.py::test_elastic_section_capacity_is_never_exceeded_by_the_fe_section`).

## 4. Load–deflection path (nx = 8, n = 4, near-ideal plasticity)

`P_y` = analytic first-yield pressure; `k0` = 12.2 mm per `P_y` (elastic slope).

| p/P_y | p [MPa] | w centre [mm] | w/w_elastic | \|M\|/m_p |
|---|---|---|---|---|
| 0.4 | 0.348 | 4.88 | 1.00 | 0.248 |
| 0.8 | 0.695 | 9.77 | 1.00 | 0.497 |
| 1.1 | 0.956 | 13.43 | 1.00 | 0.683 |
| 1.4 | 1.217 | 17.36 | 1.02 | 0.841 |
| 1.7 | 1.478 | 24.32 | 1.17 | 0.897 |
| 2.0 | 1.738 | 46.29 | 1.90 | **1.000** |

## 5. Diagnosis (defect found by this case)

**The layered Mindlin Shell4 plate has no discrete limit point under load
control.** Once the section moments saturate, the *elastic* transverse shear
stiffness keeps the global tangent positive definite, so Newton keeps converging
while the deflection runs away — `collapse_factor` then only measures how far the
ramp was pushed. Consequences:

1. `collapse = last ramp factor` is inadmissible (retired here; the new extractor
   `limit_load_from_path` returns three standard definitions — two-line
   intersection, 2×-elastic deflection, incremental-stiffness criteria — plus
   their **spread**, and a regression test asserts `limit_load < 0.6 × peak`).
2. At 2.0 `P_y` = 1.738 MPa the FE still carries load with `w = 1.90 × w_elastic`
   and `|M|/m_p = 1.000`, i.e. the **discrete** model is *over-stiff* relative to
   the admissible upper bound (1.516 MPa, §6) by at least **+14.6 %** before any
   limit point is even reached. The remaining cause is the Q4 bending-plasticity
   discretisation (a hinge line must localise inside a single 125 mm element),
   which is the classic "displacement limit analysis converges from above" error,
   not membrane action.
3. A runaway guard (`max_deflection_factor`, `bisection_tolerance`) was added:
   the near-ideal ladder used to die in a bisection storm past saturation, which
   is why the previous run stopped at a prescribed factor.

## 6. Bounds (all closed form, independent of the FE)

| bound | formula | value |
|---|---|---|
| Ilyushin double-sine **upper** (tightest) | `8 m_p / (0.20267 a²)` | **1.5158 MPa** |
| yield-line pyramid upper | `8 m_p / (a²/6)` | 1.8422 MPa |
| strip statically admissible **lower** | `p_x + p_y` | 0.8188 MPa |

The double-sine mechanism wastes far less work than the corner pyramid when the
load covers only the central quarter, so it — not the pyramid — is the bound the
FE must be compared with. Bracket `[0.819, 1.516] MPa` (width ratio 1.85).

## 7. Error judgement (honest, per acceptance item)

| criterion | value | threshold | verdict |
|---|---|---|---|
| elastic → plastic transition vs analytic yield load (nx = 8, exact z) | +0.66 % | < 5 % | **met** |
| layered section capacity `m_p,n/m_p` vs closed form (n = 3, 4) | exact (0.8889, 1.0000) | exact | **met** |
| plastic limit load vs Ilyushin upper bound | **not reachable on nx = 8**: the discrete model carries 1.738 MPa (2.0 P_y) at 1.90 × elastic deflection, i.e. **+14.6 % above** the 1.516 MPa bound, and has no limit point there | < 10 % | **NOT met** |
| 3-level mesh/layer convergence (nx = 8/12/16, n = 4/8, shear probes) | ladder submitted as cluster job **cj3211** (7 shards); not returned before this delivery | monotone | **not_run** |
| pytest | 40 passed (`tests/test_plate_patch_plastic_limit*.py`, `test_layered_shell4_plasticity.py`) | >= 10 | **met** |
| permanent deflection post-processing | unload leg from 1.7 P_y converged (see JSON `unloading`) | residual > 0 | **met** |

The anchor's *limit-load* acceptance is therefore **not satisfied**, and the
number that was previously reported (+41.55 %) is replaced by the honest
statement above: the discrete over-strength is at least +14.6 % and the FE limit
load cannot be read off a load ramp at all.

## 8. Notes / limitations

* **No figure in this delivery.** The field plots and the load–deflection
  comparison need the mesh ladder (cj3211) that is still queued; producing a
  curve from a single mesh would repeat the very error this case fixes (one
  curve is not a convergence study). The numeric tables are the evidence.
* **The shear probes were not run** (`shear_scale` 0.25 / 4.0 in cj3211). They
  are the planned discriminator between "Q4 plastic-hinge discretisation" and
  "elastic-shear load path" as the dominant over-strength mechanism.
* Deflections beyond `w ≈ 3 t` leave the small-rotation validity range; the
  runaway guard now stops there.

## 9. Reproduction

```bash
PYTHONPATH=src python scripts/patch_plate_plastic_limit_load.py --configs 8:4:1:0
PYTHONPATH=src python -m pytest tests/test_plate_patch_plastic_limit_load.py -q
```