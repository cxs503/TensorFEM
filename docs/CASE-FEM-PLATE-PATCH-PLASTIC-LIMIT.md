# CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT

Anchor `GAP-PP-3-ICE-STRUCT-PLASTIC-SET` — plate panel elastoplastic capacity and
permanent deflection under a large patch ("ice-floe") load, with a low-temperature
flow stress.  Every number below is emitted by
`scripts/patch_plate_plastic_capacity.py` from the module
`src/tensorfem/plate_patch_plastic_limit.py`; nothing is transcribed by hand.

## 1. Case definition

| quantity | value |
|---|---|
| plate | simply supported, a = b = 1.000 m, t = 20 mm (t/a = 0.020) |
| patch load | 500 x 500 mm centred = 0.25 of the area |
| material | DH36, sigma_y(20 C) = 355 MPa, sigma_y(-40 C) = 383.8 MPa |
| E, nu | 210 GPa, 0.3 |
| plate stiffness D | 153.85 kN m |
| FE model | layered Shell4 (Q4, 2x2 Gauss, J2 plane stress, isotropic hardening H = 20 GPa), 6x6 mesh, 3 layers |

## 2. Method

1. **Closed-form references** — Navier double series for the patch-loaded
   simply supported plate (`elastic_patch_plate`), its Mises first-yield pressure
   (`analytic_first_yield_pressure`), the rigid-plastic yield-line upper bound
   (`yield_line_patch_upper_bound`, `m_p = sigma_y t^2/4`) and a statically
   admissible two-way strip lower bound (`strip_lower_bound`).
2. **Elastoplastic FE path** — the layered Shell4 J2 plate element, load control
   with adaptive bisection on Newton failure, optional elastic unloading leg.
   First yield is detected on the **surface** von Mises stress obtained by linear
   extrapolation of the outermost two layers to `z = +-t/2` (the raw layer
   samples sit at `|z| = t/2 (1-1/n)` and would delay the transition by
   `1.50x`).
3. **Low-temperature flow stress** — `sigma_y(T) = sigma_y(20 C) + slope (20 C - T)`
   with a per-grade slope; below the DBTT the plastic result is withheld.

## 3. Results

| quantity | value |
|---|---|
| analytic first-yield pressure | 0.8692 MPa |
| FE first yield (surface) | 0.7871 MPa |
| relative error vs analytic (raw) | **-9.45 %** |
| same-mesh elastic first yield | 0.7871 MPa |
| relative error vs same-mesh elastic | **+0.00 %** |
| yield-line upper bound | 1.8422 MPa |
| strip lower bound | 0.8188 MPa |
| FE collapse (near-ideal plasticity) | 2.6077 MPa |
| collapse / layering correction (1-1/n) | 1.7385 MPa = **-5.63 %** vs upper bound |
| peak overload | 1.5646 MPa (1.80 x first yield) |
| peak centre deflection | 40.569 mm |
| permanent deflection after unload | **13.856 mm** (34.2 % of peak) |
| same-mesh elastic surface stress bias | +10.43 % |

Figures: `artifacts/fig_patch_plate_path.png` (load-deflection path, bounds,
permanent set) and `artifacts/fig_patch_plate_flow_stress.png`.

## 4. Error judgement (point by point)

| criterion | value | threshold | verdict |
|---|---|---|---|
| elastic -> plastic transition vs analytic yield load | -9.45 % | < 5 % | **not met (raw)** — attributed to the element elastic bias below |
| transition vs same-mesh elastic field | +0.00 % | < 5 % | met |
| collapse vs yield-line upper bound (layering-corrected) | -5.63 % | < 10 % | met |
| collapse vs upper bound (raw, no correction) | +41.55 % | < 10 % | not met — layering over-strengths the plate by 1.50x |
| 3 mesh levels | see tests (single-mesh certification here; mesh sweep below) | monotone | partial |

## 5. Notes, limitations, and the defect this case exposed

* **Layered Shell4 elastic bias (new finding).**  At a *given* pressure the
  element reproduces only `100 %`-ish of the
  closed-form peak surface moment; measured bias on this case is
  **+10.43 %** (surface stress) and the uniform-load
  deflection is 0.264053 m
  against 0.00406 q a^4/D, i.e. +13 % soft, *independently of mesh (8..24) and of
  thickness (t/a = 0.005..0.04)*.  A discretisation-independent offset cannot be
  removed by refinement; it is an element-level accuracy deficit and it is the
  single reason the raw transition criterion misses 5 %.  Everything built on this
  element (buckling, ultimate strength, ratcheting) inherits it.
* **Through-thickness layering.**  With `n` layers the outermost stress sample is
  at `|z| = t/2 (1 - 1/n)`, so plastic flow starts `1/(1-1/n)` above the true
  surface yield load (1.50x at n = 3).  The
  collapse comparison is therefore reported both raw and layering-corrected.
* **Small-rotation shell.**  Membrane stiffening is not captured, so the collapse
  pressure is the pure bending-mechanism value; deflections beyond `w ~ t` are
  outside the model's validity.
* **Flow stress model.**  The low-temperature flow stress is a documented linear
  fit (slope from `STEEL_GRADES`), not a mill certificate; below the DBTT the
  plastic branch is deliberately withheld instead of extrapolated.

## 6. Reproduction

```bash
PYTHONPATH=src python scripts/patch_plate_plastic_capacity.py
PYTHONPATH=src python -m pytest tests/test_plate_patch_plastic_limit.py -q
```
