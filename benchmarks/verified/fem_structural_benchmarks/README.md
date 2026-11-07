# `fem_structural_benchmarks` — beam / plate / axisymmetric verification face (IN41-N01)

Three element families, three independent closed-form oracles, three graded
ladders. One directory per campaign node; this one is the *structural face* that
complements the single-rung anchors (`fem_cantilever`, `fem_plate_static`, …)
with an axisymmetric (shell-of-revolution) family and with load cases whose
reference values are not reused from any earlier anchor.

| case | element (library kernel) | oracle | levels | worst error at the finest level |
|---|---|---|---|---:|
| A 悬臂梁端部集中力 | 2-node cubic Hermite frame (`frame2d`) | `w = PL³/(3EI)`, `σ = Mc/I` | `n = 4, 8, 16` | 2.7e-12 (round-off: exact cubic reproduction) |
| B 四边简支薄板均布载荷 | 4-node Mindlin/MITC4 plate (`plate.q4_mindlin_stiffness`) | Navier series α = 0.00406235; Timoshenko α = 0.00406 | `n = 8, 16, 32` | 3.0e-03 vs series, 3.6e-03 vs table |
| C 厚壁圆筒内压 | 4-node axisymmetric ring (`hertz_axisymmetric.assemble_axisymmetric_stiffness`) | Lamé `σ_r = A - B/r²`, `σ_θ = A + B/r²` | `nr = 8, 16, 32` | 2.4e-03 (`σ_θ`), 1.7e-03 (`σ_r`) at 5 radii |

Observed orders (h halving, finest pair): `2.03` (beam, uniform-load interior
field), `3.84` (plate, L2 deflection norm), `1.90` (cylinder, recovered `σ_θ`).
All twelve gates in `gates_verified.json` pass; thresholds are the campaign
numbers (1 % beam, 2 % plate, 1 % cylinder/5 radii, order ≥ 1.5, three levels).

## Files

```
result.json           every number per level (generated, never hand-edited)
gates_verified.json   schema tensorfem.verified-gates.v1: per gate the reference
                      formula, the value, the threshold and the verdict, plus
                      artifacts and limitations
case_a_beam.png       deflected shape + bending stress vs the closed form
case_b_plate.png      deflection field (contour) + refinement history
case_c_cylinder.png   σ_r / σ_θ vs r/a against Lamé + pointwise error
run.py                the driver; its module docstring states the physics
render_doc.py         regenerates docs/fem-structural-benchmarks-in41.md from the JSON
```

## Reproduce

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/verified/fem_structural_benchmarks/run.py   # ~40 s
PYTHONPATH=src python benchmarks/verified/fem_structural_benchmarks/render_doc.py
PYTHONPATH=src pytest tests/test_fem_structural_benchmarks.py                                    # ~9 s
```

## Reading notes (do not over-claim)

* Case A's tip-load errors are a **round-off floor**, not a converged ladder: the
  cubic Hermite element spans the exact cubic field. The measurable rate comes
  from the uniform-load sub-case (quartic field, element interiors).
* Case B's *centre* deflection error is not monotone in the mesh
  (`3.9e-3 → 7.3e-4 → 3.0e-3`), so the plate rate is graded on the L2 deflection
  norm. The 2 % gate is satisfied at all three levels either way.
* Case C reports stresses after standard Gauss→nodal recovery. The raw
  Gauss-point errors are also in `result.json` (`gauss_vs_recovered`): at
  `nr = 32` the un-recovered `σ_θ` error is 1.3e-2, i.e. **the 1 % gate is reached
  by the recovery, not by the raw element values**. `σ_r` vanishes at the free
  outer surface, where a relative error is undefined, so that point is graded by
  the load-normalised absolute error (gate `C2b`) and reported as `null`
  relative error instead of being dropped silently.
* Everything here is linear elastic, small displacement, single physics. It
  certifies element kernels, consistent load vectors, boundary conditions and
  stress recovery — not curved-shell meshing, plasticity, contact, buckling or
  dynamics.