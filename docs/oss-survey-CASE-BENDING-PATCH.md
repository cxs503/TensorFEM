# CASE-BENDING-PATCH — round-2 method survey and selection (open source + literature)

Mission m15427 (anchor `CASE-BENDING-PATCH`, TensorFEM). Round 2 of the owner's
two-round protocol: *survey first, then implement*. Round 1 (m15409) ended in an
honest FAIL with no recorded `fail_stage` / `failure_reason`; this document plus
`benchmarks/verified/fem_bending_patch/` is the deliberate de-risking of that
failure mode: define the criterion and the target numbers **before** writing the
element-level code.

## 0. Reproducibility of this survey (honest access log)

| source | status in this sandbox | consequence |
|---|---|---|
| `~/oss-cache` on the control host | **read**: only LBM/DEM checkouts (`Maarten-vd-Sande_lbm`, `hschwane`, `pylbm`) — no FEM/plate-bending code | no open-source patch-test implementation could be read locally |
| `~/oss-cache` on the cluster nodes | **submitted** (`run_cluster` job `cj1115`, grep for `patch test`/`constant curvature` across `~/oss-cache`), still **queued** at finish time | result not available for this report; declared unknown, not claimed |
| `oss_inventory` / `oss_solutions` tables | **permission denied** for the agent role (`psql` on `tensorplatform`) | the SQL catalogue could not be queried; selection records are therefore delivered as an *insert script* artifact for the platform owner to apply (same convention as `m15187/artifacts/audit-GAP-DEM-CLUMP-m15028/oss_solutions_insert.sql`) |
| textbooks / journal papers | no outbound network | the **criterion** and the constitutive form are cited; every target number is re-derived in code from that form, so the evidence chain never depends on an unavailable table (the same honest convention as `docs/patch-test-robinson.md`) |

No claim in the report rests on a source that could not be read.

## 1. What the existing repository already had (and why it is not this task)

| existing artefact | what it tests | gap for `CASE-BENDING-PATCH` |
|---|---|---|
| `src/tensorfem/patch_test.py`, `benchmarks/patch_test_robinson.py`, `docs/patch-test-robinson.md` | Robinson A/B/C patch test for **plane-stress continuum** elements (Q4/Q8/TET10), including a "pure bending" *stress* state | a continuum patch test says nothing about a **plate/shell bending** element: those have an independent rotation field and a transverse-shear energy that can lock |
| `src/tensorfem/plate.py`, `plate_mindlin.py` | Q4 Mindlin-Reissner element (SRI and MITC shear treatments) + Winkler foundation + Hetényi reference | element exists, but **no element-level consistency proof for bending** was recorded anywhere |
| `src/tensorfem/element_rank_audit.py`, `docs/element-rank-deficiency-audit.md` | zero-energy modes of the one-point element | orthogonality: rank deficiency of the *free* element; a patch test cannot see it (documented as a limitation) |
| `docs/rectangular-bending-validation.md`, `plate-bending-buckling-validation.md`, `foundation-plate-bending-validation.md` | *system*-level plate bending benchmarks (deflection/moments vs series or literature) | system benchmarks mix model error with element error; they cannot separate "the element is consistent" from "the mesh is fine enough" |

⇒ The missing capability is exactly the one the anchor names: a **constant-curvature
(bending) patch test** for the plate element, graded against a literature criterion.

## 2. Method genealogy (what standard practice says)

| # | source | method | what we take |
|---|---|---|---|
| G1 | Irons & Ahmad, *Techniques of Finite Elements* (Ellis Horwood, 1980), ch. 8 | the patch test as a **consistency** requirement: an assembly of elements with free interior nodes, driven only by boundary data belonging to the element's span, must reproduce that state exactly | the criterion itself, in its pass/fail (exactness) form |
| G2 | Zienkiewicz & Taylor, *The Finite Element Method*, Vol. 2 (5th ed., 2000), §10.4 | restatement for plates/shells: the constant-curvature state is the "span" a plate element must reproduce; failure signals inconsistency or parasitic shear | the constant-curvature state and the requirement that it be shear-free |
| G3 | Batoz, Bathe & Ho, *Int. J. Numer. Methods Eng.* **15** (1980) 1771–1812, §4 | constant-curvature / constant-moment patch tests applied to plate elements; distinguishes "reproduces the state" from "converges to it" | the split between **affine** patches (must be exact) and **distorted** patches (converge with h) adopted here |
| G4 | Bathe & Dvorkin, *Int. J. Numer. Methods Eng.* **21** (1985) 367–383 | MITC4: assumed natural transverse shear; removes shear locking / spurious modes while retaining the bending span | used as an *independent element family* (the repo already ships `shear_treatment="mitc"`) so the result is not an artefact of one shear treatment |
| G5 | Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., Art. 10 (and Timoshenko, *Strength of Materials* II, ch. 3) | closed forms for the load-driven controls: cylindrical bending of a plate strip (`D = E t³/(12(1−ν²))`) and pure bending of a narrow strip with free edges (`D_b = E t³/12`) | the two **load-driven** cases, whose ratio `1/(1−ν²)` cross-checks the Poisson coupling |

Selection rule applied: *exactness where the discrete span allows it, measured
convergence where it does not, and a negative control that must fail.* Sources G1–G3
supply the criterion, G4 an independent discretisation, G5 a load-driven oracle —
so the case is not graded on a state that the driver itself prescribed.

## 3. Selected method (what was implemented)

`src/tensorfem/bending_patch.py` (new, ~450 lines) + `benchmarks/verified/fem_bending_patch/run.py`:

1. **Exact oracle** `CurvatureState`: `w = −(κ_x x² + κ_xy x y + κ_y y²)/2`,
   `θ_x = κ_x x + κ_xy y/2`, `θ_y = κ_xy x/2 + κ_y y` — zero transverse shear and
   constant curvatures in the *engineering* measure `θ_x,y + θ_y,x` that
   `tensorfem.plate` integrates. Target moments `M = Db κ`.
   `oracle_selfcheck()` re-verifies shear-free-ness / constancy before assembly.
2. **Patch families**: affine (`structure` + `skew`, parallelograms → the pass/fail
   form) and distorted (non-affine; the measured, h-convergent residual).
3. **Solve**: production element `tensorfem.plate.q4_mindlin_stiffness`
   (`sri` and `mitc`), exact field on all boundary nodes, free interior solved from
   `K_ff u_f = −K_fc u_c`; moments recovered with the element's own `Bb` at 2×2 Gauss.
4. **Negative control**: the same driver with 2×2 (full) shear integration
   (`shear_treatment="full"`), which locks; it must fail by a wide margin.
5. **Load-driven strip**: cantilever under a consistent tip moment, two lateral
   conditions (`symmetry` → cylindrical bending, `free` → beam rigidity `D_b`).

## 4. Transfer record (for `oss_solutions`, `feature_slug='CASE-BENDING-PATCH'`)

| repo / source | tag | method | convention | transfer | evidence |
|---|---|---|---|---|---|
| `literature/Irons-Ahmad-1980` | `patch-test-as-consistency` | the patch test is a *necessary* consistency condition; grade it as exactness on the element's own affine span | never present a system benchmark as a patch test | adapt | `docs/bending-patch.md` §4.1 (error ≤ 8.0e-16) |
| `literature/Zienkiewicz-Taylor-FEM-v2-§10.4` | `constant-curvature-state` | the plate span to be reproduced is a shear-free constant-curvature state | the twist must be split half-half between θ_x,y and θ_y,x because the library integrates the *engineering* twist | adapt | `CurvatureState.field`, verified by `oracle_selfcheck` |
| `literature/Batoz-Bathe-Ho-1980` | `affine-vs-distorted-patches` | affine meshes must be exact; distorted meshes converge | report both, never merge them into one "error" | adapt | §4.1 vs §4.2 of `docs/bending-patch.md` |
| `literature/Bathe-Dvorkin-1985` | `mitc4-independent-family` | run the same patch with an independent shear treatment | a pass with one treatment is not evidence | adapt | `mitc` columns of the result table |
| `literature/Timoshenko-Woinowsky-Krieger-Art10` | `load-driven-strip-oracle` | close the "the answer was prescribed" hole with a load-driven case | strip narrow → `D_b = Et³/12`; symmetry cell → `D = Et³/(12(1−ν²))` | adapt | §4.3 and the `1/(1−ν²)` cross-check (measured ratio matches to 0.02%) |
| `oss-cache(+cluster cj1115)` | `no-oss-implementation-found` | no open-source patch-test implementation was readable in this sandbox (local cache has LBM/DEM only; cluster grep still queued) | the case is built from the criterion, not from a borrowed implementation | reference-not-adopted | this document §0 |

## 5. Result of the transfer

All 17 machine gates pass (`benchmarks/verified/fem_bending_patch/gates_verified.json`):
affine/skew constant-curvature reproduction at 8.0e-16 (displacement), 9.4e-16
(moment), 1.6e-15 (energy); distorted patch 1.54 % (displacement) / 0.75 %
(moment) / 0.17 % (energy) against the 2 % case-pool tolerance; load-driven strip
2.2e-9 (symmetry cell) and 1.76 % (free edges, coarsest mesh); the locking control
fails at 97.9 %, so the passing gates are not vacuous.

## 6. What this survey did **not** settle

* The cluster grep (`cj1115`) had not completed — if it returns an open-source
  patch-test implementation, a follow-up should compare the mesh family and the
  grading tolerance against §3 (the criterion itself would not change).
* `oss_inventory` / `oss_solutions` could not be written from this tenant
  (permission denied); the records above are delivered as
  `artifacts/oss_solutions/CASE-BENDING-PATCH_oss_solutions_insert.sql`.