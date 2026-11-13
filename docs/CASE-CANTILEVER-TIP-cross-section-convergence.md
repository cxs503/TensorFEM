# CASE-CANTILEVER-TIP — cross-section convergence of the HEX20 shear recovery

**Anchor**: `CASE-CANTILEVER-TIP` (算例池 · 悬臂梁端挠度) · **Bundle**:
`benchmarks/verified/fem_cantilever_shear/cross_section_convergence.py` ·
**Generated**: 2026-11-13T10:28:53+0000 · **Infrastructure gates**: PASS · **Scientific claim**: honest FAIL

> Regenerate with `PYTHONPATH=src python benchmarks/verified/fem_cantilever_shear/cross_section_convergence.py` — this document is rendered from the JSON artifact, no number is typed by hand.

## ① Case introduction

The case-pool anchor asks for the tip deflection of a slender cantilever against the
Euler-Bernoulli / Timoshenko closed forms, tolerance `< 3 %`. The published slenderness
ladder (`docs/CASE-CANTILEVER-TIP-shear-validity.md`) certifies that tolerance on the
whole validity ladder, but records this debt verbatim:

> The 2x2 / 3x3 quadratic cross-section meshes cannot represent the parabolic shear
> distribution exactly; the refinement trend (shear recovery rising with mesh density)
> is reported instead of a converged value, and the shear-recovery gate is deliberately
> loose (>= 0.5) for that reason.

The cause was mechanical: `tensorfem.hex20.solve_hex20` assembles a dense
`n_dofs x n_dofs` matrix, so the ladder could not exceed ~5e3 dof and the section mesh had
to stay at `2x2 / 3x3`. This document removes the cap and answers the question the ladder
could only pose: **does the continuum recover the full Timoshenko shear deflection as the
section converges?**

| parameter | value |
|---|---|
| `E` | 200 GPa |
| `nu` | 0.3 |
| tip load `P` | 5 N (spread over the end face, zero net moment) |
| section `B x H` | 10 x 10 mm |
| `kappa` | 0.8333 (5/6, average-strain) |
| slenderness rungs `L/h` | 20, 10, 5, 4 |
| cross-section subdivisions `n` | 2, 3, 4, 6 |
| mesh family | axial = round(slenderness*n), cross = n x n (isotropic h) |

Successful run cost: 4 rungs x 4 levels = 16 HEX20 solves, largest model 65919 dof, 58.4 s of solver time.

## ② Method

1. **Sparse path** (`tensorfem.hex20.solve_hex20_sparse` +
   `assemble_hex20_stiffness_sparse`): COO triplets summed on conversion, SuperLU on CSC,
   Dirichlet elimination identical to the dense path. The dense operator is deliberately
   *not* materialised, which is what lifts the dof cap.
2. **Certify the new path against existing evidence, not against itself**:
   (a) dense vs sparse on the published meshes; (b) sparse vs the tip deflections already
   stored in `result.json`; (c) force/moment reaction equilibrium on every model.
3. **Isotropic refinement family**: level `n` uses `cross = n x n` and
   `axial = round(L/h * n)`, so a level changes only `h` and the section and the span are
   refined together (refining the section alone would hide the axial error in the
   comparison).
4. **Metric**: `shear_recovery = (d_solid - d_EB) / (d_Ti - d_EB)` — the fraction of the
   Euler-Bernoulli -> Timoshenko gap the continuum recovers; `1` would mean the continuum
   confirms `kappa = 5/6`, `0` that it shows no shear flexibility at all.
5. **Extrapolation**: Richardson on the two finest levels assuming `O(h^2)`, with the
   order-1 variant reported as a model-form spread; the order implied by the two finest
   increments is computed from the data rather than assumed.

## ③ Results

### 3.1 The sparse path is the same discretisation

| `L/h` | mesh | dofs | dense tip (m) | sparse tip (m) | rel. diff | vs published |
|---:|---|---:|---:|---:|---:|---:|
| 4 | (40, 2, 2) | 3663 | 6.597265366450e-07 | 6.597265366449e-07 | 2.9e-13 | 1.7e-13 |
| 5 | (30, 3, 3) | 5160 | 1.269834436877e-06 | 1.269834436881e-06 | 3.1e-12 | 3.0e-12 |

Infrastructure gates: sparse-vs-dense `3.1e-12`,
sparse-vs-published `3.0e-12`, force equilibrium `2.5e-09`, moment equilibrium `1.9e-09`, heaviest model `65919` dof — all PASS.

### 3.2 Shear recovery vs cross-section refinement

| `L/h` | oracle gap | recovery at n = 2, 3, 4, 6 | finest vs EB | order |
|---:|---:|---|---:|---:|
| 20 | 0.1950% | -1.6828, -1.2428, -1.0693, -0.9243 | -0.180 % | 1.52 |
| 10 | 0.7800% | -0.3458, -0.1223, -0.0354, +0.0369 | +0.029 % | 1.57 |
| 5 | 3.1200% | +0.3315, +0.4446, +0.4869, +0.5212 | +1.626 % | 1.78 |
| 4 | 4.8750% | +0.4704, +0.5606, +0.5935, +0.6196 | +3.020 % | 1.95 |

Three things are visible and none of them says `kappa = 5/6`:

1. **Not converged at the thick rung.** At `L/h = 4` the recovery is still climbing
   (+0.5935 -> +0.6196); the last cross-section increment is 2.61 percentage points, above the 1 pp gate. The order
   implied by the two finest increments is 1.95, so the
   `O(h^2)` extrapolation used below is defensible *at that rung*.
2. **The extrapolated limit is not 1.** `shear_recovery -> 0.640 +- 0.031` (spread = order 2 vs order 1), which excludes
   `1.0` by a wide margin: the continuum recovers about two thirds of the Timoshenko
   shear deflection at `L/h = 4`, not all of it.
3. **The metric is not rung-independent**, so it is not a measurement of `kappa`: at the
   finest mesh it runs from -0.924 (`L/h = 20`) to +0.620 (`L/h = 4`), a spread of 1.544. A shear coefficient is a property of the section and cannot depend on
   the span. The reason is visible in the last column: on the slender rungs the
   residual against Euler-Bernoulli (23.11 % of the gap at `L/h = 10`) is the
   same size as the whole shear gap (`0.7800%`), so the
   ratio of two comparably small numbers is being read as a physical fraction.

### 3.3 The anchor's own verdict, re-graded at convergence

| quantity | value | basis |
|---|---:|---|
| converged tip deflection, `L/h = 4` | `1.031222 x d_EB` | Richardson |
| converged error vs Timoshenko | **-1.671 %** | same |
| case-pool tolerance | 3.000 % | anchor |
| verdict | **PASS** | `tolerance check: True` |

The coarsest level of this family (n = 2) gives -2.462 % against Timoshenko, and the published `(40,2,2)` mesh gives -1.709 %. Refining by 3x in
the section moves the converged value to `-1.671 %`: the residual is **a model
difference, not a mesh artefact**, and the `< 3 %` verdict is mesh-robust with the whole
margin intact.

### 3.4 The published attribution of the slender-rung residual is falsified

The published §5 attributes the `-0.40 % / -0.38 %` residual at `L/h >= 20` to the
clamped root being stiffer than the idealised built-in. That residual does not survive
section refinement:

| `L/h` | published residual vs EB | published mesh / dofs | refined residual vs EB | refined dofs |
|---:|---:|---|---:|---:|
| 20 | -0.380 % | (32, 2, 2) / 2943 | -0.180 % | 65919 |
| 10 | -0.146 % | (24, 3, 3) / 4152 | +0.029 % | 33159 |

At `L/h = 20` the residual falls by more than half and at `L/h = 10` it changes sign
(`-0.146 %` -> `+0.029 %`). A clamped-root
boundary layer is a physical field and cannot vanish when the section mesh is refined,
so the published attribution is **falsified**: that residual is dominated by
discretisation error, and the root clamp is not the constant `-0.4 %` bias the
limitation assumes.

## ④ Figure

`docs/assets/cantilever_shear/cross_section_convergence.png` — shear recovery vs
cross-section subdivisions for all four rungs, with the `kappa = 5/6` line at `1.0`, the
Euler-Bernoulli line at `0.0`, and the Richardson limit of each rung drawn dotted. The
curves rise with the section mesh (not converged), and their limits stay far below `1.0`;
the vertical distance between the rungs at fixed `n` is the rung dependence that
disqualifies the metric as a `kappa` measurement.

## ⑤ Notes, limitations, and what the honest FAIL means

* **This is an honest FAIL on the scientific claim, and it is recorded as such**
  (`claim_supported = false` in the artifact, three claim gates failed, and
  `tests/test_cantilever_cross_section_convergence.py` fails if anyone upgrades them
  silently). The infrastructure is not in doubt: it is certified against the previously
  published numbers to `3e-12`.
* **The `kappa` debt is not closed** — it is now *better localized*. Before, the loose
  `>= 0.5` gate was read as 'the solid recovers shear'. Now: the section mesh alone does
  not resolve it; the metric is contaminated by a non-shear residual of the same order as
  the shear gap on slender rungs, and it is rung-dependent by construction of the
  clamped 3-D model.
* **Scope of the claim**: one element family (HEX20, 3x3x3 integration), one load case
  (uniform end-face load, zero net moment), one root BC (all three displacement
  components clamped over the root face), `nu = 0.3`, sections `n <= 6`. No
  extrapolation is claimed beyond `L/h = 4`.
* **The next experiment is now well posed**, and it is not another mesh study: the
  remaining candidate that scales like the shear term is the clamped-root boundary layer
  (it suppresses shear deformation over a region of extent ~`h`, which is 25 % of the
  span at `L/h = 4`). It can only be separated from the theory error by an A/B on the root
  condition — e.g. the root face clamped vs the root section restrained against
  translation and rotation but free to warp — which a mesh study of the present kind
  structurally cannot do.
* **Cost note**: element-level tensor ops on a 128-core box oversubscribe; the bundle pins
  `torch.set_num_threads(8)`, which took the assembly of
  the `L/h = 20`, `n = 6` model from 50 s to 0.9 s. Timings in the artifact are from a
  loaded machine and are not performance claims.

## ⑥ Error verdict

| quantity | reference | got | error | `< 3 %`? |
|---|---:|---:|---:|:--:|
| tip deflection, `L/h = 4`, converged | Timoshenko | `1.031222 d_EB` | -1.671 % | **是** |
| tip deflection, `L/h = 4`, converged | Euler-Bernoulli | `1.031222 d_EB` | +3.122 % | 否 — **by design**: at this rung the shear signal itself is 4.875% of `d_EB` (> the 3 % tolerance), so EB is not a valid oracle here; the published ladder grades this rung against Timoshenko |
| tip deflection, `L/h = 4`, finest mesh (n = 6) | Timoshenko | `1.030204 d_EB` | -1.768 % | 是 |
| sparse vs dense solver | dense LAPACK | identical | 3.1e-12 | 是 |
| sparse vs published artifact | `result.json` | identical | 3.0e-12 | 是 |
| shear recovery at `L/h = 4`, extrapolated | `1.0` (Timoshenko `kappa = 5/6`) | 0.640 +- 0.031 | **-35.954 %** | **否 — honest FAIL** |

**Conclusion.** The anchor's `< 3 %` verdict at `L/h = 4` survives convergence and is in
fact strengthened: the converged residual against Timoshenko is `-1.671 %`, a
model difference (Timoshenko over-predicts for a fully clamped 3-D cantilever) rather
than discretisation. The `kappa = 5/6` convention is **not** confirmed by the continuum
(`shear recovery -> 0.64`, not `1`), the shear-recovery metric is shown to be rung-dependent and
therefore not a `kappa` measurement, and the published attribution of the slender-rung
`-0.4 %` residual to the clamped root is falsified. The `kappa` verification debt stays
open (未闭合) and is now localized to the root boundary layer, which needs a root-BC A/B
experiment rather than a finer mesh.

