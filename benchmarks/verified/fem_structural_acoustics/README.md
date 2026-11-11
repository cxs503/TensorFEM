# Verified benchmark: FEM structure-acoustic face (plate + cylinder)

* node `p472w2-N09`, repository TensorFEM, driver `run.py`
* gates: **15 / 17 PASS** — failing: C3_cylinder_sigma_vs_jf_L2, C4_cylinder_sigma_vs_jf_L8
* artefacts: `result.json` (all numbers), `gates_verified.json` (gate table),
  `figures/` (4 figures), full report `docs/reports/fem-structural-acoustics.md`

## What is verified

1. **Plate modal** — Q4 Mindlin FEM vs Kirchhoff closed form, first six modes,
   max error 2.34 % (criterion 3 %).
2. **Plate radiation efficiency** — 4-D Rayleigh integral of the FEM mode velocity
   vs the closed-form angular spectrum: 0.40 %
   worst case above ka = 1, 0.35 % below.
3. **Cylinder radiation** — equivalent-source exterior Helmholtz solve vs the
   Junger & Feit infinite-cylinder window closure for four modes:
   41.0 % worst case at
   R = 0.5 m, L = 2 m (honest FAIL of the 15 % criterion in the compact regime) and
   30.3 % at L = 8 m.
4. **Exterior-solve anchor** — pulsating sphere power reproduced to
   0.0017 % (2 % criterion).
5. **Negative control** — rigid baffled surface radiates
   0.0e+00 W (machine precision).

## Run

```
python benchmarks/verified/fem_structural_acoustics/run.py
python benchmarks/verified/fem_structural_acoustics/make_report.py
```
