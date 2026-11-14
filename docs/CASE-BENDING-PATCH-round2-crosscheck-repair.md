# CASE-BENDING-PATCH — round 2: cross-check repair of the Poisson coupling

Mission 20755 · repo TensorFEM · branch `agent/m20755` · anchor `benchmarks/verified/fem_bending_patch/` · generated report `docs/bending-patch.md` · artefacts `benchmarks/verified/fem_bending_patch/{result,gates_verified}.json`.

Every number below is read from `result.json` / `gates_verified.json` by a generator
(no digit is hand-copied). Round 1 of this mission (PR #945) was closed without merge;
round 2 re-opens the same anchor and repairs what round 1 graded.

## 0. What round 1 left, and what was wrong with it

Round 1 left a green anchor: 17 gates, patch test at round-off on affine patches, a
distorted-patch residual inside the 2 % case-pool tolerance, a load-driven strip, a
locking negative control. Two of those gates did not carry the evidence they claimed —
both concern the *cross-check*, i.e. whether a pass is a property of the element or a
property of the driver script.

1. **`poisson_crosscheck` could not fail.** The round-1 code computed
   `measured = w_exact_kirchhoff(free) / w_exact_kirchhoff(symmetry)`, i.e. it divided
   two *closed forms* `M L^2/(2 D_ref)`. Both operands carry a rigidity the driver had
   inserted, so the ratio is `D/D_b = 1/(1-nu^2)` by construction: the reported error
   was 0.000e+00 % for any element, any mesh and any `nu`. The round-1 report still
   presented it as "the physical cross-check of the Poisson coupling".
2. **The free-edge leg of the strip was aliased, not converged.** Both strip runs used a
   fixed transverse mesh `n_y = 2`; the free-edge leg was graded at 2 % with a measured
   error of 0.455 % at `n_x = 32`. That number is a cancellation: refining only
   the transverse direction moves it to 1.196 % (`n_y = 8`) and 1.198 %
   (`n_y = 16`). The converged offset is 2.63x the
   reported one and sits only 1.67x inside the tolerance.

## 1. Decision, alternatives and predeclared criteria

**Decision to support**: may the round-1 Poisson cross-check stand as evidence, or must
it be replaced by checks whose value is a *solved* quantity?

**Alternatives**

| option | content | why not chosen |
|---|---|---|
| A | keep round 1 as delivered | the gate is element-blind (green with a broken `Db`), so it inflates the anchor's evidence grade |
| B | delete the cross-check | removes the only test of `D_b = D (1-nu^2)` |
| **C** | **(i)** closed-form ratio demoted to an explicit *oracle* identity, **(ii)** coupling graded on the **solved** tip deflections at a transverse-converged mesh, **(iii)** exact `M_y = 0` (plane-stress) patch that reads the coupling out of the element's own `Db` | **chosen** |

**Predeclared tests** (thresholds fixed before the repaired run was inspected) and
**outcomes**:

| # | hypothesis | predeclared test and threshold | outcome |
|---|---|---|---|
| H1 | the round-1 ratio is a tautology | closed-form error `< 1e-14` for `nu` in {0, 0.3, 0.45} and invariant to any common stiffness factor | measure: 2.021e-16 — **confirmed** |
| H2 | the free-edge offset is a transverse-discretisation error | `n_y` refinement 2→4→8→16 must drive the offset below 0.2 % | measured plateau 1.198 % — **falsified** |
| H3 | the plateau is a clamped-root end effect | at fixed element size the offset must decay with `L/b`: `e(L/b=20)/e(L/b=5) <= 0.6` | measured 0.283 — **confirmed** |
| H4 | the `theta_y` restraint of the root causes the offset | freeing `theta_y` at the root must move the offset by `> 0.5` percentage points | measured 1.181 % (`clamp_theta_y=False`) vs 1.196 % (`True`), i.e. `0.015` points — **falsified** |
| H5 | the coupling can be graded by an element-side check | `M_y = 0` state: `max|M_y|/max|M_x| <= 1e-12` at `nu` in {0.3, 0.45}, and the same quantity graded against the cylindrical rigidity `D` must be rejected by `>= 5 %` | measured 4.2e-16 and 20.250 % — **confirmed** |

H2 failing is the substantive round-2 finding: it rules out "round 1 was merely
under-resolved in the transverse direction" and forces a *model* explanation. H4 was
run specifically to kill the first explanation that came to mind.

## 2. Method, tool chain and settings

| item | value |
|---|---|
| material | E = 210000 MPa, nu = 0.3, t = 1 mm |
| element | `tensorfem.plate.q4_mindlin_stiffness` (Q4 Mindlin–Reissner, `sri` and `mitc`); `full` (2x2 shear) only as the locking negative control |
| new library capability | `tensorfem.bending_patch.free_surface_state()` (the `M_y = 0` state `kappa_y = -nu kappa_x`) and `free_surface_check()` (its graded reproduction) |
| diagnostic option | `strip_pure_bending(..., clamp_theta_y=False)` — root restrains `w`, `theta_x` but not `theta_y` |
| strip geometry | `L = 1, b = 0.2, t = 0.01`, tip moment `M = 1e3` per unit width |
| transverse sweep | `n_x` in {8, 32}, `n_y` in {2, 4, 8, 16}, both lateral conditions |
| aspect series | `L/b` in {5, 10, 20} at fixed element size (`n_x = 32 L`, `n_y = 8`) |
| plane-stress patch | affine and skewed (skew 0.35) 3x3 patches, `nu` in {0.3, 0.45}, treatments `sri` and `mitc` |
| runtime | 95.27 s, single node, 4 threads |
| solver | dense `torch.linalg.solve` on the free DOFs (`K_ff u_f = -K_fc u_c`) |

The committed driver regenerates everything:

```
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_bending_patch/run.py
PYTHONPATH=src pytest tests/test_bending_patch.py -q
```

## 3. Results

### 3.1 The cross-check, before and after

| leg | quantity | value |
|---|---|---|
| round-1 style (closed form, oracle only) | `D/D_b` | 1.098901098901 |
| round-1 style (closed form, oracle only) | error vs `1/(1-nu^2)` | 2.021e-16 |
| repaired (solved, `n_y = 16`, `n_x = 32`) | `w_free/w_sym` | 1.085740 |
| repaired (solved) | error vs `1/(1-nu^2) = 1.098901` | 1.198 % |
| round-1 mesh (solved, `n_y = 2`) | `w_free/w_sym` | 1.093899 (0.455 %) |
| plane-stress `M_y = 0` patch | `max|M_y|/max|M_x|`, `nu` = 0.3, 0.45 | 4.219e-16 |
| plane-stress `M_y = 0` patch | `max|M_x/(D_b kappa_x) - 1|` | 8.882e-16 |
| plane-stress `M_y = 0` patch | same, graded against `D` instead of `D_b` | 20.250 % (rejected) |

### 3.2 Transverse refinement of the free-edge leg

| `n_y` | free-edge `w` error vs `M L^2/(2 D_b)` |
|---|---|
| 2 | 0.455 % |
| 4 | 1.144 % |
| 8 | 1.196 % |
| 16 | 1.198 % |

Change between the two finest levels: 0.13 % — the leg is
transverse-converged from `n_y = 8` on, at a plateau of about 1.2 %.

### 3.3 Aspect-ratio series: the plateau is a root end effect

| `L/b` | free-edge `w` error vs `M L^2/(2 D_b)` |
|---|---|
| 5 | 1.196 % |
| 10 | 0.651 % |
| 20 | 0.338 % |

Decay ratio `e(L/b=20)/e(L/b=5) = 0.283` at fixed
element size. The offset is local to the clamped root: a hard clamp forces `w = 0` on
the root edge, while free lateral edges require the anticlastic curvature
`kappa_y = -nu kappa_x` (hence `w` quadratic in `y`); the two are incompatible and a
boundary layer adds a small extra tip rotation. Root-support control at fixed mesh
(`n_x = 32`, `n_y = 8`): `theta_y` clamped -> 1.196 %, `theta_y` free ->
1.181 %. The offset survives, so it is the `w = 0` restraint of the
anticlastic field, not the rotation restraint.

### 3.4 The repaired coupling patch (`M_y = 0` state)

| `nu` | family | treatment | w error | rotation error | moment error | `|M_y|/|M_x|` | `M_x/(D_b kappa_x)` |
|---|---|---|---|---|---|---|---|
| 0.3 | affine | sri | 5.27e-16 | 3.25e-16 | 3.34e-16 | 1.40e-16 | 1.000000000 |
| 0.3 | affine | mitc | 3.96e-16 | 2.44e-16 | 4.06e-16 | 2.08e-16 | 1.000000000 |
| 0.3 | skew | sri | 8.01e-16 | 1.20e-16 | 8.12e-16 | 8.94e-17 | 1.000000000 |
| 0.3 | skew | mitc | 1.60e-16 | 2.41e-16 | 6.09e-16 | 2.59e-16 | 1.000000000 |
| 0.45 | affine | sri | 2.75e-16 | 3.25e-16 | 6.09e-16 | 2.90e-16 | 1.000000000 |
| 0.45 | affine | mitc | 2.06e-16 | 2.44e-16 | 4.06e-16 | 2.52e-16 | 1.000000000 |
| 0.45 | skew | sri | 3.55e-16 | 3.61e-16 | 8.12e-16 | 4.22e-16 | 1.000000000 |
| 0.45 | skew | mitc | 1.78e-16 | 2.41e-16 | 8.12e-16 | 3.55e-16 | 1.000000000 |

This is the leg that carries the coupling: `M_y` is produced by the element's own
`Db`, so a wrong (1-nu^2) factor, a wrong `nu` sign or a missing Poisson term fails it
at 10-20 %, not at round-off. Running it at two Poisson ratios (`nu = 0.3` and `0.45`)
prevents a constant tuned to one material.

### 3.5 Gates added / replaced

| gate | what it grades | value | threshold | verdict |
|---|---|---|---|---|
| `poisson_identity_closed_form` | |D/D_b - 1/(1-nu^2)| / (1/(1-nu^2))  [algebra only, no solve] | `2.02061e-16` | `<= 1e-14` | PASS |
| `poisson_coupling_computed` | |w_free/w_sym - 1/(1-nu^2)| / (1/(1-nu^2)) from the SOLVED tip deflections, n_y=16 | `0.0119769` | `<= 0.02` | PASS |
| `free_surface_moment_exact` | max |M_y| / max |M_x| on the M_y=0 state (affine+skew, nu=0.3 and 0.45) | `4.21864e-16` | `<= 1e-12` | PASS |
| `free_surface_field_exact` | max nodal w/rotation error, M_y=0 state (affine+skew, nu=0.3 and 0.45) | `8.01135e-16` | `<= 1e-12` | PASS |
| `free_surface_rigidity` | max |M_x/(D_b kappa_x) - 1|, D_b = E t^3/12 | `8.88178e-16` | `<= 1e-12` | PASS |
| `free_surface_rigidity_separation` | |M_x/(D kappa_x) - 1|: grading against the cylindrical rigidity D | `0.2025` | `>= 0.05` | PASS |
| `strip_free_transverse_convergence` | |e(n_y=16) - e(n_y=8)| / e(n_y=16), free lateral, n_x=32 | `0.00134555` | `<= 0.1` | PASS |
| `strip_free_end_effect_decay` | e(L/b=20) / e(L/b=5), free lateral, n_y=8, fixed element size | `0.282953` | `<= 0.6` | PASS |
| `strip_free_w` | tip deflection error, free edges, transverse-converged meshes (n_y >= 8, load-driven) | `0.0139971` | `<= 0.02` | PASS |

Total: 24/24 gates pass. The
anchor keeps its negative controls (`locking_control_strip`
0.979 >= 0.50; `locking_control_patch`
0.0211 >= 0.02), so the green gates are not
vacuous.

## 4. Error judgement (is every graded quantity < 3 % of its reference?)

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| affine/skew patch: w, rotations, moments, energy | constant-curvature closed form | <= 8.0e-16 | yes |
| distorted patch: w | constant-curvature closed form | 1.542 % | yes |
| distorted patch: moments | `M = Db kappa` | 0.751 % | yes |
| strip, symmetry cell: tip w | `M L^2/(2D)` | 0.000 % | yes |
| strip, free edges, transverse-converged: tip w | `M L^2/(2 D_b)` | 1.400 % | yes |
| solved Poisson ratio `w_free/w_sym` | `1/(1-nu^2)` | 1.198 % | yes |
| `M_y` on the `M_y = 0` state | 0 (plane stress) | 4.2e-16 x `max|M_x|` | yes |
| `M_x` on the `M_y = 0` state | `D_b kappa_x` | 8.9e-16 relative | yes |

Every graded quantity is far inside 3 %; the two quantities *near* 1-2 % (free-edge
strip, solved Poisson ratio) are the ones whose 1.2 % is a **documented model offset**
(clamped root), not a numerical error, and both are declared in the gate text. The
anchor does not claim <3 % pointwise *accuracy* — a patch test cannot: it proves
consistency (exact reproduction of a constant-curvature state) and, here, that the
Poisson coupling is implemented correctly.

## 5. Figures

Generated by the driver into `docs/assets/bending_patch/`:

* `patch_meshes.png` — the three mesh families, red rings = free interior nodes.
* `error_vs_h.png` — distorted-patch h-convergence (`sri`, `mitc`) with a slope-2 guide.
* `error_budget.png` — max error per mesh family and shear treatment vs the 2 % tolerance.
* `strip_bending.png` — load-driven strip error vs `n_x`, both lateral conditions.

Field-cloud figures are not meaningful for this case and are therefore not produced:
a patch test reproduces a *constant* curvature state, so the moment field is uniform by
construction; plotting it would only show a flat colour map. The informative visual is
the error budget, which is committed.

## 6. Notes, pitfalls and scope

* **Aliasing can fake a pass.** `n_y = 2` made the free-edge leg look 2.6x better than
  it is. The repaired driver grades the leg only at `n_y >= 8` and gates the
  transverse-convergence itself, so the same aliasing cannot silently return.
* **A closed-form ratio is not a cross-check.** If both operands share the rigidity
  that the model is supposed to produce, the ratio is an identity. The rule applied
  here: a gate must consume at least one *solved* quantity.
* **Convergence order at the round-off floor is noise.** Round 1 reported `mitc` orders
  of -0.53 (`w`) and -1.79 (`M`); the underlying errors are `~1e-15`, i.e. round-off.
  Those orders are now reported as `null` (`n/a`) instead of numbers that look like
  evidence.
* **Scope of the anchor is unchanged**: consistency, not coarse-mesh accuracy; no
  zero-energy-mode detection (a Dirichlet patch has none — the rank deficiency of the
  one-point `sri` element is documented in `docs/element-rank-deficiency-audit.md`).
* **Not claimed**: that the free-edge strip with a hard clamp has the beam closed form
  as its limit at engineering `L/b`; it does not (1.2 % at `L/b = 5`, decaying with
  `L/b`). Round 1 implied it did.

## 7. Files changed by this round

| file | change |
|---|---|
| `src/tensorfem/bending_patch.py` | **added** `free_surface_state()`, `free_surface_check()` and the `clamp_theta_y` diagnostic option (purely additive) |
| `benchmarks/verified/fem_bending_patch/run.py` | repaired Poisson block, transverse sweep, aspect series, `M_y = 0` cases, 8 new gates, round-off floor for orders, regenerated report sections §4.3/§4.5/§5 |
| `benchmarks/verified/fem_bending_patch/result.json`, `gates_verified.json` | regenerated (24 gates) |
| `docs/bending-patch.md` | regenerated report (new §4.5 cross-check section) |
| `docs/CASE-BENDING-PATCH-round2-crosscheck-repair.md` | this document |
| `tests/test_bending_patch.py` | 5 new test functions, 12 parametrised cases (21 -> 33 collected): tautology, plane-stress exactness at two `nu` x affine/skew x sri/mitc, transverse convergence, end-effect decay, `theta_y`-restraint control |

No TensorFEM core module (`src/tensorfem/plate.py` and friends) was touched, so this
PR does not overlap the hot-path files of other open PRs; the only production file
changed is an additive helper module used by this anchor.
## 8. Verification status (what was actually run, and what was not)

| check | command / job | result |
|---|---|---|
| anchor driver, full (not `--quick`) | `PYTHONPATH=src python benchmarks/verified/fem_bending_patch/run.py` | 24/24 gates PASS, `result.json` + `gates_verified.json` + `docs/bending-patch.md` regenerated (95 s, 4 threads) |
| unit tests of the touched module | `pytest tests/test_bending_patch.py -q` | 33 passed |
| unit tests of the touched module + production element | `pytest tests/test_bending_patch.py tests/test_plate_mindlin.py -q` | 53 passed |
| full `tests/` suite | submitted twice to the cluster (`cj2930`, `cj2949`) | **could not be completed**: the node Python env has no `matplotlib` (5 modules fail at *collection*), and `tests/test_fluid_rigid_feedback.py` / `tests/test_passive_rigid_feedback.py` fail at collection on every ref because `tensorlbm` is absent |
| full `tests/` suite, local | blocked by the local short-test gate | not run |

The two `tensorlbm` collection errors and the cluster-side `matplotlib` gap are
**pre-existing** (they are independent of this change: they occur on the unmodified
`main` tree as well) and are reported here rather than hidden. Nothing in this round
touches `tensorlbm` or matplotlib-dependent modules; the changed production file
(`src/tensorfem/bending_patch.py`) is covered by the 33 tests above at 100 % of its
public entry points.
