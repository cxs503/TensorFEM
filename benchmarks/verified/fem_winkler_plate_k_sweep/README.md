# GAP-FEM-WINKLER — Winkler foundation `k` sweep vs Hetényi

Anchor for the **foundation-coefficient coupling** of TensorFEM's plate and beam
foundation elements. The companion anchor `fem_winkler_plate/` fixes one `k`
(buoyancy) and sweeps the mesh; this one fixes the mesh family and sweeps `k`
across decades, which is the only way a units / quadrature-weight / factor-of-`l`
error in `K_w = k ∫ Nᵀ N dΩ` can be exposed (because `k l⁴ = D` binds every
length scale together).

| family | model | load | reference | `k` values | span |
|---|---|---|---|---|---|
| A | infinite plate, quarter-symmetry Q4 Mindlin (MITC) | central point load | Hetényi kei(r/l), `w(0)=P/(8√(Dk))` | 1e3, 1e4, 10055.25 (ρ_w g), 1e5, 1e6 N/m³ | 3 decades |
| B | finite simply supported EB beam, L = 20 m | central point / uniform | exact Fourier-sine solution | 1e4, 1e5, 1e6 N/m² | 2 decades |
| C | finite simply supported plate, 10 m × 10 m | uniform pressure | Navier double series | 1e4, 1e5, 1e6 N/m³ | 2 decades |

## Gates (all machine-checked, see `gates_verified.json`)

| gate | quantity | threshold | worst value |
|---|---|---|---|
| A | centre deflection vs Hetényi | ≤ 5 % | **0.2124 %** (k = 1e6) |
| A | profile `w(r)` at r/l = 0.5, 1, 2, 3 (normalised by `w(0)`) | ≤ 5 % | **0.0785 %** |
| A | foundation reaction equilibrium (free outer edge) | ≤ 1 % | **1.37e-08 %** |
| B | beam sup-norm deflection vs sine series | ≤ 5 % | **1.94e-04 %** |
| C | finite plate centre deflection vs Navier | ≤ 5 % | **0.1533 %** |
| D | sweep span | ≥ 2 decades, ≥ 3 values | 3.0 / 2.0 / 2.0 decades |

27/27 gates pass. All three chains also satisfy the stricter 3 % reading.

## Is the gate inert?

No. `w ∝ k^(-1/2)`, so the 5 % deflection gate is a ~10 % gate on `k` itself. Mutation
probe (element built with `f·k`, reference with `k`, n = 20, worst over k = 1e3..1e6):

| f | worst deflection error | caught by the 5 % gate |
|---|---|---|
| 1.00 | 0.395 % | no (clean) |
| 1.01 | 0.889 % | no (sensitivity limit) |
| 1.25 | 10.910 % | **yes** |
| 4.00 | 50.193 % | **yes** |
| 0.25 | 99.468 % | **yes** |

Frozen as regression test `test_gate_detects_a_mutated_foundation_stiffness`.

## Reproduce

```bash
python benchmarks/fem_winkler_plate_k_sweep.py             # full sweep, rewrites these artifacts
python benchmarks/fem_winkler_plate_k_sweep.py --quick     # coarse meshes
python -m pytest tests/test_fem_winkler_plate_k_sweep.py -q
python scripts/winkler_k_sweep_doc.py                      # regenerate docs/...k-sweep.md
```

## Artifacts

| file | contents |
|---|---|
| `result.json` | every number: per-`k` plate rows (with profile stations), beam rows (full curves), finite-plate rows, thick-plate regime diagnostic, gate list |
| `gates_verified.json` | machine-readable verdicts (`schema: tensorfem.verified-gates.v1`) |
| `winkler_k_sweep_profile.png` | non-dimensional `w(r)/w(0)` vs `r/l`, FEM lines vs kei dashes |
| `winkler_k_sweep_error.png` | centre/profile/equilibrium error vs `k`, 5 % acceptance line |
| `winkler_k_sweep_beam.png` | beam deflection curves, FEM lines vs series markers, 3 `k_b` |

## Notes / limits

- `kei(r/l)` changes sign at `r/l ≈ 2.4`, so the profile gate is normalised by `w(0)`;
  a pointwise *relative* error near the root is meaningless (it reads ~3 % spuriously).
- A supported plate's foundation reaction is **not** an equilibrium residual — the edge
  supports carry part of the load; that case reports the foundation load share instead.
- Thick-plate regime: for `t/l > 0.1` the Mindlin answer is physically above the
  Kirchhoff kei solution by O((t/l)²). `regime_diagnostic` records this (150 mm ice plate,
  `k = 1e6`: 3.54 % → 4.43 % → 4.84 % for n = 20/40/60, error *growing* towards the
  converged Mindlin value). Family A therefore uses a 50 mm steel plate, `t/l ≤ 0.04`.
- One-parameter Winkler only (no shear-layer coupling); family A is central-load only.

## References

1. Hetényi, M. (1946). *Beams on Elastic Foundation*. University of Michigan Press.
2. Timoshenko & Woinowsky-Krieger (1959). *Theory of Plates and Shells*, 2nd ed., §62.