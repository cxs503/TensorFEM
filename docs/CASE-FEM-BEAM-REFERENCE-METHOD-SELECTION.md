# CASE-FEM-BEAM-REFERENCE-METHOD-SELECTION

**Anchor**: `XVAL-LBM-BENCHMARK-METHOD-SELECTION` — practice "a numerical record must
state *how* its benchmark reference was selected, and that selection must be
recomputable from raw measurements", applied inside TensorFEM (worktree
`agent/m17935`, repo TensorFEM; the LBM row of the ledger stays with TensorLBM).

**Verdict**: the selection is now a *value*, not a paragraph. `tensorfem.method_selection`
applies six predeclared rules and emits a record that `certify_selection` re-derives from
the stored raw probe errors; applied to the cantilever beam it yields a quantitative
answer to the question every beam benchmark silently assumes:

> For a rectangular cantilever, the Euler-Bernoulli closed form `q L^4 / (8 E I)` may serve
> as the benchmark reference only for `L/h >= 8.44` at the 3 % certification line, or
> `L/h >= 3.53` at the owner 15 % unacceptable line. Below those, the shear-flexible
> (Timoshenko) reference must be quoted.

Two independent routes (closed-form algebra vs FE assembly+solve) agree on that threshold to
`5.1e-15` relative, and the reference closed form itself is reproduced by mesh-refined FEM to
`1.3e-13` — so the selection is not tuned to the mesh, and the mesh is not tuned to the selection.

---

## ① Case introduction

A prismatic rectangular cantilever under uniform load — the canonical beam benchmark. What
changes across the sweep is only the *slenderness*, because that is the single parameter the
reference-model choice depends on.

| quantity | value |
|---|---|
| span `L` | 1.0 m |
| section | rectangle, width `b = 0.1 m`, depth `h = L / (L/h)` |
| slenderness sweep | `L/h ∈ {2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 40}` |
| material | `E = 210 GPa`, `ν = 0.3`, `G = E / 2(1+ν) = 80.77 GPa` |
| shear coefficient | `κ = 5/6` (the `advanced_beam` default) |
| load | `q = 1000 N/m`, uniform, work-equivalent nodal load |
| boundary | clamped at `x = 0` (`w = θ = 0`), free at `x = L` |
| tolerances tested | `3 %` (repo certification line), `15 %` (owner unacceptable line) |

Reference ("truth") is the Timoshenko closed field

```
w_T(x) = q x²(6L² − 4Lx + x²)/(24 E I) + q(2Lx − x²)/(2 κ G A)
ψ_T(x) = q x(3L² − 3Lx + x²)/(6 E I)          (shear-free: ψ' = M/EI)
```

and it is **verified inside the run** against mesh-refined FEM of the shear-flexible element
(`advanced_beam.timoshenko_stiffness`), so quoting it as truth is not circular.

Probes (relative errors, all vs the verified truth):

| probe | meaning | why it is in the set |
|---|---|---|
| `w_tip` | tip deflection | the quantity beam benchmarks actually quote |
| `psi_tip` | tip section rotation | shear-insensitive **control**: `ψ` is identical in both models, so any signal here is a bug, not physics |
| `w_mid` | mid-span deflection | largest shear share (`K_mid/K_tip = 2.12`), hence the binding probe |

## ② Method

1. **Selection is a procedure, not prose** (`src/tensorfem/method_selection.py`). Input:
   candidates (name, cost, per-probe relative errors). Output: `SelectionRecord` with
   admissible set, winner, decision, signed per-probe margins, Pareto set and the rule text.
2. **Six predeclared rules, applied in order** — R1 comparability (shared probe set, finite
   non-negative errors, fail closed: an unmeasured probe is not "0 % error"); R2 admissibility
   `max_p error <= tolerance` (inclusive); R3 winner = lowest cost, ties by accuracy then name
   (deterministic, independent of dict ordering); R4 the winner must be Pareto-non-dominated
   on `(cost, worst error)` or the inputs are refused; R5 no candidate admissible ⇒
   `decision = "no_admissible_method"`, `selected = None` (no silent least-bad); R6 margins
   `error/tolerance − 1` recorded per probe, so "PASS at 99.9 % of the line" is visible.
3. **Cost is a deterministic proxy** (number of inputs/dofs a reference needs), never wall
   time — wall time is not reproducible across hosts and would make the record host-dependent.
4. **Candidates** (all three are already-shipped TensorFEM code paths, no new numerics):

   | candidate | cost | route |
   |---|---|---|
   | `EB-closed` | 1.0 | `q L⁴/(8EI)` — no `G`, no `A`, no `κ` |
   | `TIMO-closed` | 3.0 | closed field above — adds `G`, `A`, `κ` |
   | `TIMO-fem-n2` | 6.0 | 6 dofs, `advanced_beam.timoshenko_stiffness` assembled and solved |

5. **Two measurement protocols**, both recorded, to separate model error from interpolation:

   * *nodal* (`NODAL_MESH = 2`, mid-span and tip are nodes) — both 2-node elements are
     **nodal-exact** for a uniform load at any mesh, so a nodal probe measures **model error
     only**;
   * *field* (`FIELD_MESH = 1`, probe interpolated inside the single element) — what a
     careless comparison does, and it confounds interpolation with model error.

6. **Two independent threshold routes.** Route A: closed-form algebra —
   `(h/L)² = tol / ((1 − tol) K)` with `K_tip = 2(1+ν)/(3κ) = 1.04` and
   `K_mid = 1.41176(1+ν)/κ = 2.2024`. Route B: bisection on the ratio of two
   **FE-assembled-and-solved** nodal values (`EB` vs `TIMO`), sharing no algebra with route A.
   Agreement is a gate, not a remark.
7. **Artifacts**: `benchmarks/beam_reference_method_selection.py` (record generator),
   `benchmarks/beam_reference_method_selection_record.json` (raw probes + selections +
   break-evens), `benchmarks/beam_reference_method_selection_figures.py`,
   `tests/test_method_selection.py` (32 tests).
8. **Mutation control** (assertions are load-bearing, not decorative): flipping one
   `rows[].selected` in the record JSON turns `test_every_selection_block_recertifies` red;
   changing R2's `<=` to `<` turns `test_boundary_error_equal_to_tolerance_is_admissible` red;
   restoring the files returns 32/32 green.
9. **No new numerics were introduced** — the study reuses `beam_foundation.eb_beam_stiffness`
   and `advanced_beam.timoshenko_stiffness`; the only new code is the selection procedure.

## ③ Results

**Reference verification** (nested meshes `n = 1, 2, 4, 8`, all 12 slenderness states):

| check | worst relative error over the whole sweep |
|---|---|
| Timoshenko closed field vs shear-flexible FEM (`n >= 2`) | `1.32e-13` |
| EB closed form vs EB-FEM nodal values | `2.53e-15` |

**EB model deficit vs the verified Timoshenko truth** (nodal protocol):

| `L/h` | `w_tip` | `w_mid` | analytic shear share | `w_mid` field-sampled (1 element) | inflation |
|---|---|---|---|---|---|
| 2 | 20.63 % | 35.51 % | 20.63 % | 39.30 % | 1.11× |
| 3 | 10.36 % | 19.66 % | 10.36 % | 24.39 % | 1.24× |
| 4 | 6.10 % | 12.10 % | 6.10 % | 17.27 % | 1.43× |
| 5 | 3.99 % | 8.10 % | 3.99 % | 13.50 % | 1.67× |
| 6 | 2.81 % | 5.77 % | 2.81 % | 11.31 % | 1.96× |
| 8 | 1.60 % | 3.33 % | 1.60 % | 9.01 % | 2.71× |
| 10 | 1.03 % | 2.16 % | 1.03 % | 7.91 % | 3.67× |
| 15 | 0.46 % | 0.97 % | 0.46 % | 6.80 % | 7.01× |
| 20 | 0.26 % | 0.55 % | 0.26 % | 6.40 % | 11.68× |
| 40 | 0.065 % | 0.14 % | 0.065 % | 6.01 % | 43.74× |

The `w_tip` deficit equals the analytic shear share `qL²/(2κGA) / w_T` to `1e-9` — an identity,
not a curve fit; the test suite asserts it.

**Break-even slenderness — two routes:**

| tolerance | route A `w_mid` (closed algebra) | route B `w_mid` (FE bisection) | rel. diff | route A `w_tip` only |
|---|---|---|---|---|
| 3 % | **8.4386** | **8.4386** | `5.05e-15` | 5.7989 |
| 15 % | **3.5327** | **3.5327** | `1.51e-15` | 2.4276 |

**Selection tables** (`selected` recomputed by the rule, then re-certified):

| tolerance | probes | first `L/h` where `EB-closed` is selected |
|---|---|---|
| 3 % | `w_tip, psi_tip, w_mid` | 10.0 (`L/h = 8` → 3.33 % > 3 %) |
| 3 % | `w_tip` only | 6.0 |
| 15 % | `w_tip, psi_tip, w_mid` | 4.0 (`L/h = 3` → 19.66 % > 15 %) |
| 15 % | `w_tip` only | 3.0 |

**Protocol bias** (same model pair, different measurement discipline, `L/h = 10`):

| quantity | value |
|---|---|
| EB deficit, nodal protocol | 2.155 % |
| EB deficit, 1-element field sample | 7.910 % |
| inflation | 3.67× |
| Timoshenko `w_mid` error, 1-element field sample | 6.47 % |
| field-probe convergence (EB, `L/h = 10`) | `n=1: 7.91 %` → `n≥2: 2.1549 %` (flat to `1e-11`) |

So a *correct* shear-flexible code measured with a 1-element field probe reports a 6.47 %
"code error" at `L/h = 10` — 3× the true model deficit and more than twice the 3 % line,
i.e. the sloppy protocol would reject a right answer and accept (or reject) a model on
evidence that is mostly interpolation.

## ④ Figures

`docs/figures/beam_reference_method_selection.png` (generated by
`benchmarks/beam_reference_method_selection_figures.py` from the record JSON, no hand-entered
numbers):

* (a) EB model deficit vs `L/h` on both probes, with the 3 %/15 % lines and the break-even
  stars — shows the `w_mid` curve as the binding one;
* (b) protocol bias: nodal vs 1-element field inference of the same model deficit, the two
  curves diverging as the beam gets slender (the sloppier the mesh, the more "model error"
  you invent);
* (c) selection map: which reference the rule selects per `L/h`, for both tolerances and both
  probe sets — the blue/purple split is the deliverable;
* (d) field-probe convergence in `n`, with the `n >= 2` plateau annotated as the pure model
  deficit.

## ⑤ Notes / pitfalls / limits

* **Model error is not discretisation error.** Both shipped 2-node beam elements are
  nodal-exact for a uniform load (`n = 1` suffices at the tip, `n = 2` at mid-span). That is
  why the whole table above is *model* error; a study that refines the mesh cannot reduce it,
  and a study that reports "error vs EB" is measuring a modelling choice, not accuracy.
* **Probe choice moves the verdict by 1.46× in slenderness** (`w_mid` 8.44 vs `w_tip` 5.80 at
  3 %). A benchmark record must therefore declare the functional, not just "deflection".
  This is the same conclusion the hollow-sphere `M1/M2` cross-validation reached
  (`CASE-HOLLOW-SPHERE-3D`), now with a slenderness axis attached.
* **`κ` and `ν` are inputs to the threshold, not constants of nature.** `K ∝ (1+ν)/κ`: with
  `κ = 10(1+ν)/(12+11ν) = 0.8497` instead of `5/6`, the 3 % threshold moves 8.4386 → 8.3570
  (`∝ sqrt(1/κ)`, so −0.97 %) and the 15 % threshold 3.5327 → 3.4986. The record stores `κ`
  so this stays auditable.
* **Scope**: prismatic, isotropic, uniform load, static, one end clamped. Point loads make
  both elements nodal-exact at the tip and would hide the discretisation axis entirely;
  shear-deformable layered/composite sections need `κ` from the layup and are out of scope.
* **Cost is a proxy.** `EB-closed:1 / TIMO-closed:3 / TIMO-fem-n2:6` encodes input burden,
  not FLOPs; a different cost model would change the winner only in the narrow band where
  both are admissible, never the admissibility verdict itself.
* **No experimental data is involved.** These are two mathematical models compared with each
  other; nothing here validates either against a measured beam.

## ⑥ Error judgement

| key quantity | baseline | error | < 3 % ? |
|---|---|---|---|
| Timoshenko closed field (reference) at 12 slenderness states | shear-flexible FEM, `n >= 2` | `1.32e-13` worst | **yes** (12 orders below) |
| EB closed form at 12 slenderness states | EB-FEM nodal values | `2.53e-15` worst | **yes** |
| break-even `L/h`, 3 % | closed-form route A vs FE route B | `5.05e-15` | **yes** |
| break-even `L/h`, 15 % | closed-form route A vs FE route B | `1.51e-15` | **yes** |
| EB model deficit `w_tip` | analytic shear share | `< 1e-9` on a per-state identity asserted by `test_model_deficit_matches_the_analytic_shear_share` | **yes** |
| EB model deficit `w_mid` at `L/h = 10` (nodal vs field protocol) | each other | **3.67× (267 %)** | **no — by design**: that discrepancy is the finding, and it is attributed to interpolation, not to either model |

Physical validation against experiment: **not performed and not claimed** — the reference here
is a verified mathematical model, so the only error statement that can be made is the internal
consistency above. The 15 %-unacceptable / 3 %-certification standards are applied to the
*model-to-model* deficit, which is exactly the quantity that decides whether an EB reference may
be quoted for a given geometry.