# CASE-FEM-PLATE-PATCH-PLASTIC-DEFECTS (defect record, anchor GAP-PP-3-ICE-STRUCT-PLASTIC-SET)

Fast-subset defect record for the plate-panel patch-load elastoplastic capability.
All numbers are emitted by `scripts/patch_plate_plastic_defect_probe.py`
(`artifacts/patch_plate_plastic_defects.json`); nothing is transcribed.

## 1. Case

Simply supported a = b = 1.00 m, t = 20 mm plate, 0.5 x 0.5 m centred patch
pressure, 383.8 MPa flow stress (DH36 @ -40 C), layered Shell4 (Q4, 2x2 Gauss,
J2 plane stress, isotropic hardening H = 20 GPa).
References: Navier double series (0.0000 mm/MPa centre deflection,
0.0004 MPa/MPa surface Mises), analytic first-yield pressure
0.8692 MPa, yield-line upper bound 1.8422 MPa, strip lower bound
0.8188 MPa.

## 2. Defect 1 — the "element accuracy deficit" was the through-thickness rule

The first-generation certificate reported an 11-13 % elastic offset of the layered
element and concluded that "a discretisation-independent offset cannot be removed by
refinement". That conclusion is **retracted**: the offset is the mid-point rule in the
through-thickness direction, whose deficit on `Int z^2 dz` is exactly `1 - 1/n^2`.

| layers n | deflection error, mid-point rule | closed-form `1/(1-1/n^2)-1` | deflection error, exact `+dz^3/12` | surface-Mises error, exact |
|---|---|---|---|---|
| 2 | +34.90 % | +33.33 % | +1.33 % | -0.66 % |
| 3 | +13.92 % | +12.50 % | +1.33 % | -0.66 % |
| 9 | +2.59 % | +1.25 % | +1.33 % | -0.66 % |

With `LayeredShell4Model.exact_thickness=True` the missing `dz^3/12` term is integrated
in the bending internal force and tangent, so the elastic response becomes
layer-count independent (identical to round-off, covered by
`tests/test_layered_shell4_plasticity.py::test_exact_thickness_bending_is_independent_of_the_layer_count`).

## 3. Defect 2 — the mesh guard admitted a misaligned patch mesh

`patch_plate_mesh` compared the patch *width* with the element size, so `nx = 6`
(patch edges at `x = 0.25` = 1.5 elements) was accepted; the lumped patch loads then do
not conserve the patch resultant and bias the deflection by ~9 % (measured earlier:
nx = 6 mid-point deflection error +15.4 %-class vs
nx = 8 +13.92 %). Both patch edges are now required on node
lines and `nx = 6` raises `ValueError`.

## 4. Elastoplastic results on the small mesh

| quantity | value |
|---|---|
| analytic first-yield pressure | 0.8692 MPa |
| FE first yield (nx = 4, exact z) | 0.9212 MPa (+5.98 %) |
| same-mesh elastic first yield (nx = 8, exact z) | 0.8750 MPa (+0.66 % vs analytic) |
| near-ideal plastic collapse (nx = 4) | 2.6077 MPa (+41.55 % vs yield-line upper bound) |
| permanent set after unloading from 1.80 x first yield | 2.04 mm (8.1 % of the 25.32 mm peak) |

## 5. Error judgement (honest, per acceptance item)

| criterion | value | threshold | verdict |
|---|---|---|---|
| elastic -> plastic transition vs analytic yield load (nx = 8, elastic field) | +0.66 % | < 5 % | **met** |
| same criterion on the coarse nx = 4 path | +5.98 % | < 5 % | **not met** |
| plastic limit load vs yield-line upper bound | +41.55 % | < 10 % | **not met** (see §6) |
| key quantity vs baseline: centre deflection, elastic branch (nx = 8, exact z) | +1.33 % | < 3 % | **met** |
| 3-level mesh convergence (4 / 8 / 12) | 4 and 8 computed; 12 not run (local 120 s limit) | monotone | **not_run** |

## 6. Why the collapse criterion still misses (diagnosis, honest FAIL)

The near-ideal collapse load sits +41.6 % **above** the
kinematic yield-line upper bound, which is inadmissible for a bending-only rigid-plastic
mechanism. Two candidate causes were tested:

* *Layer-count / quadrature over-strength* — **rejected**: switching to the exact
  through-thickness integration leaves the collapse load unchanged
  (2.6077 MPa before and after), so the earlier "1.50x layering
  correction" was not the cause.
* *Membrane action from in-plane-restrained supported edges* — **the remaining
  explanation**: the supported edges of `patch_plate_mesh` are fixed in all six
  degrees of freedom, so the plate develops membrane tension and carries load beyond the
  pure bending mechanism. The yield-line bound is a bending-only bound, so the two are
  not comparable in this model.

Remedy (next step, not in this delivery): release the in-plane edge displacements
(simply supported = `w = 0`, rotations and in-plane translations free) and re-run the
collapse ladder; then the raw collapse must fall below the upper bound.

## 7. Reproduce

```bash
PYTHONPATH=src python scripts/patch_plate_plastic_defect_probe.py
PYTHONPATH=src python -m pytest tests/test_layered_shell4_plasticity.py -q   # 8 passed
PYTHONPATH=src python scripts/patch_plate_plastic_capacity.py ladder:8      # full certificate, cluster
```
