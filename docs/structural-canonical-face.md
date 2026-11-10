# Canonical structural verification face: tip-loaded beam, rectangular plate, modal, `K_t`

*Module*: `tensorfem.structural_canonical_face`
*Evidence*: `benchmarks/structural_canonical_face.json`
*Figures*: `docs/assets/structural_canonical/`
*Audit*: `PYTHONPATH=src python scripts/audit_structural_canonical_face.py` (do **not** pin `OMP_NUM_THREADS`; see §5)
*Tests*: `python -m pytest tests/test_structural_canonical_face.py -q`

## 1. Case introduction (physical background, geometry, parameters)

`tensorfem.structural_verification` opened the structural ladder with a
*uniformly* loaded cantilever, a **square** Kirchhoff plate and cantilever modes;
`tensorfem.kirsch_hole` established the `K_t = 3` reference with **Q4**
continuum elements; and `benchmarks/verified/` already carries several of the
classical anchors. The honest split, checked against the gate files in
`benchmarks/verified/*/gates_verified.json`, is:

| # | quantity | prior evidence in this tree | what is new here |
|---|---|---|---|
| C1 | cantilever with a **concentrated tip load**, `w_tip = PL³/(3EI)` | `fem_beam_static` L1: tip deflection and tip rotation, Hermite beam, ≤0.5 % | clamp reactions, Clapeyron energy identity, and the explicit statement that the tip number is *nodally exact* for this load case |
| C2 | **rectangular** simply supported plate, `w_c = α q b⁴/D`, `α(a/b)` | square only: `fem_plate`, `fem_plate_static` (`α = 0.00406`), `structural_verification` | the aspect-ratio sweep `a/b ∈ {1, 1.5, 2, 3, 60}` vs Timoshenko table 8, and the **rectangular** Navier series (the repository only carried the square series) |
| C3 | cantilever `f₁` | `fem_modal` L3 (modes 1–5) | only re-run as a campaign gate |
| C4 | rectangular plate `f₁₁` | `fem_modal` L3, `fem_plate_modal` (1.2 × 0.8 m) | only re-run as a campaign gate |
| C5 | `K_t = 3.0 ± 0.1` | `kirsch_hole` with **Q4** (module docs + tests) | the same limit on an independent element family (**Q8**, exactly circular edges) |
| C6 | observed `L₂` order `p ≥ 1.9` for **linear** plate elements | Hermite (≈4, `fem_beam_static`), BFS (1.94, `fem_plate_static`), Q8 (`fem_cantilever`) | the bilinear Q4 rate (`p = 2.03`), i.e. the first *linear*-element rate in the tree |

So four of the six gates (C1, C3, C4 and the square part of C2) repeat quantities
that are anchored elsewhere; they are included because the campaign needs one
machine-checkable gate file, and the module docstring says so. The new numerical
content is the aspect-ratio face, the Q8 `K_t` family and the linear-element
rate. Two by-products of building the face are also recorded: the `grid()`
metadata defect fixed in `structural_verification` (see §5) and the rectangular
Navier series, which any later rectangular-plate work can reuse.

### Parameter tables (exactly the values used by the audit script)

Beam (C1, C3):

| symbol | value | note |
|---|---|---|
| `L` | 1.0 m | cantilever length |
| `EI` | 2000 N·m² | bending stiffness (product, not `E` and `I` separately) |
| `P` | 10 N | downward concentrated tip load |
| `ρA` | 8 kg/m | mass per unit length (modal arm) |
| mesh | 8 Hermite elements (1–8 in the exactness test) | production `modal.uniform_beam_matrices` |

Plate bending (C2): `b = 1 m` (short side), `a/b ∈ {1, 1.5, 2, 3}`,
`E = 210 GPa`, `ν = 0.3`, `t = 10 mm` (`b/t = 100`), `q = 1 Pa`, BFS mesh
`16·(a/b) × 16` elements (1024–3072 free dofs — see the JSON for each level).

Plate vibration (C4) and the linear-element rate arm (C6): `a = 2 m`,
`b = 1 m`, `t = 10 mm` (`a/t = 200`), steel `E = 200 GPa` (C4) / `E = 210 GPa`
(C6, matching the bending arm), `ν = 0.3`, `ρ = 7850 kg/m³`, Q4 Mindlin element
with selective reduced shear integration (SRI), hard simple support.

Kirsch (C5): hole radius `a = 1`, annulus to `r = 4`, uniaxial far-field
`T = 1`, plane stress, `E = 1e5`, `ν = 0.3`, Q8 polar mesh `n_θ = 24`,
`n_radial = 8`.

### Independent references

* Euler–Bernoulli cantilever: `w(x) = Px²(3L−x)/(6EI)`, `θ(L) = PL²/(2EI)`,
  `f₁ = 1.875104²/(2πL²)√(EI/ρA)` (Blevins, *Formulas for Natural Frequency and
  Mode Shape*, table 8-1).
* Navier double-sine series for the simply supported rectangular plate
  (Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed.,
  §30), `α` values from table 8 (three digits) and the cylindrical-bending limit
  `α → 5/384`.
* Kirsch (1898) field, `σ_θθ(a, π/2) = 3T` (Timoshenko & Goodier, *Theory of
  Elasticity*, 3rd ed., §39).
* Mode-1 cantilever shape `cosh z − cos z − σ(sinh z − sin z)`,
  `σ = (cosh z + cos z)/(sinh z + sin z)`, `z = 1.8751`.

## 2. How it was done (methods, settings, toolchain)

Every arm drives a **production** path; the module adds only the reference
series, the rectangular-geometry wrapper and the gate bookkeeping.

| arm | production path (in this repository) | added here |
|---|---|---|
| C1 | `modal.uniform_beam_matrices` (4-dof Hermite beam, consistent mass) | tip force vector, reactions `Ku − f`, strain energy `½uᵀKu` |
| C2 | `plate_buckling_hermite.bfs_plate_element_matrices` (Bogner–Fox–Schmit) | general `length × width` geometry, consistent uniform-pressure load |
| C3 | `modal.cantilever_beam_modes` (`Kφ = ω²Mφ`, Cholesky reduction) | — |
| C4 | `plate_vibration.solve_plate_vibration` (Q4 Mindlin, consistent mass) | — |
| C5 | `quadratic_plane.solve_q8_plane` + `polar_q8_mesh` (exactly circular edges) | exact Kirsch displacement on the outer rim, hole-edge recovery |
| C6 | `plate.q4_mindlin_stiffness` (SRI Q4) | rectangular assembly, uniform load, L2 norm against Navier |

Boundary conditions:

* C1/C3: clamp `w = 0`, `w_,x = 0` at `x = 0`.
* C2: hard simple support — `w = 0` plus the **tangential** slope
  (`w_,x` on the `y`-edges, `w_,y` on the `x`-edges), normal moment natural.
* C4/C6: Mindlin hard simple support — `w = 0` on all edges, `θ_x` restrained on
  the `y`-edges and `θ_y` on the `x`-edges, normal rotation free.
* C5: Dirichlet on the outer rim with the exact Kirsch displacement; the hole
  edge is traction free, i.e. satisfied **naturally** by the variational
  statement, so the only error is discretisation.

Toolchain: pure Python/torch (C1–C4), numpy dense algebra (C2, C6), sparse CG
(C5). The artifact records host (`loongarch64`), python, `tensorfem` version and
the exact command, so the numbers are reproducible with one command.

## 3. Results (numbers straight from `benchmarks/structural_canonical_face.json`)

### Gate summary

| gate | quantity | computed | reference | measure | gate | pass |
|---|---|---|---|---|---|---|
| C1 | cantilever tip deflection | 1.6666666667e-3 m | 1.6666666667e-3 m | rel. 2.6e-13 | < 1 % | yes |
| C2 | rectangular plate centre deflection | see table | Navier | worst rel. 5.89e-3 | < 2 % | yes |
| C3 | cantilever `f₁` | 8.8479323 Hz | 8.8479139 Hz | rel. 2.1e-6 | < 3 % | yes |
| C4 | plate `f₁₁` (a/b = 2) | 30.4575 Hz | 29.9915 Hz | rel. 1.55e-2 | < 3 % | yes |
| C5 | `K_t` (Q8, 24×8) | 3.0273 | 3.0 | abs. 2.73e-2 | ±0.1 | yes |
| C6 | observed `L₂` order (bilinear plate) | 2.028 | ≥ 1.9 | least-squares fit over 5 meshes | ≥ 1.9 | yes |

### C1 — tip-loaded cantilever invariants (8 elements)

| quantity | FE | exact | relative error |
|---|---|---|---|
| tip deflection `w(L)` | 1.6666666667e-3 m | 1.6666666667e-3 m | 2.6e-13 |
| tip rotation `θ(L)` | 2.5000000000e-3 | 2.5000000000e-3 | 2.4e-13 |
| base reaction | 10.0000000000 N | `P = 10 N` | 4.7e-13 |
| base moment | 10.0000000000 N·m | `PL = 10 N·m` | 3.2e-13 |
| strain energy | 8.3333333333e-3 J | `P²L³/(6EI)` = 8.3333333333e-3 J | 7.6e-13 |
| nodal field (all nodes) | — | exact cubic | max abs. 4.4e-16 m |

### C2 — aspect-ratio coefficient of the simply supported plate

`α = w_c D/(q b⁴)`, `b` the short side; `α_T` is Timoshenko table 8.

| `a/b` | mesh | free dofs | `α` (BFS FE) | `α` (Navier series) | FE error | `α_T` | table vs series |
|---|---|---|---|---|---|---|---|
| 1.0 | 16×16 | 1024 | 0.00403842 | 0.00406235 | −0.589 % | 0.00406 | −0.058 % |
| 1.5 | 24×16 | 1536 | 0.00769126 | 0.00772402 | −0.424 % | 0.00772 | −0.052 % |
| 2.0 | 32×16 | 2048 | 0.01009162 | 0.01012866 | −0.366 % | 0.01013 | +0.013 % |
| 3.0 | 48×16 | 3072 | 0.01219288 | 0.01223325 | −0.326 % | 0.01223 | −0.023 % |
| 60 | — | — | — | 0.01302108 | — | 5/384 = 0.01302083 | +1.9e-5 |

Mesh ladder at `a/b = 1` (same element size in both directions):
16×16 → −0.589 %, 24×24 → −0.262 %, 32×32 → −0.148 %; at `a/b = 2`:
32×16 → −0.366 %, 48×24 → −0.163 %, 64×32 → −0.091 %. The FE value approaches
the exact one from **below** (a conforming displacement element overestimates the
stiffness), which is the expected sign and is asserted in the tests.

### C3/C4 — modal

| case | mesh | computed | exact | relative error |
|---|---|---|---|---|
| cantilever `f₁` | 8 Hermite elements | 8.8479323 Hz | 8.8479139 Hz | +2.1e-6 |
| plate `f₁₁`, `a/b = 2` | 16×8 Q4 | 30.4575 Hz | 29.9915 Hz | +1.55 % |
| plate `f₁₁`, `a/b = 2` | 32×16 Q4 | 30.1017 Hz | 29.9915 Hz | +0.37 % |

The Mindlin `f₁₁` converges from above, and the residual at 16×8 is the
shear-deformation + discretisation bias of a thin (`a/t = 200`) plate.

### C5 — Kirsch stress concentration (Q8 family)

| `n_θ` | `n_radial` | nodes | `K_t` | \|error\| | displacement rel. `L₂` |
|---|---|---|---|---|---|
| 12 | 2 | 96 | 2.6927 | 0.307 | 1.7e-2 |
| 16 | 2 | 128 | 2.7272 | 0.273 | 1.5e-2 |
| 16 | 4 | 224 | 2.9642 | 0.036 | 3.3e-3 |
| 24 | 4 | 336 | 2.9764 | 0.024 | 2.9e-3 |
| 24 | 8 | 624 | 3.0273 | 0.027 | 3.8e-4 |

The hole-edge hoop-stress distribution (48 samples at `n_θ = 24`) matches
`T(1 − 2cos 2θ)` with a maximum absolute deviation of **0.0273 T**, i.e. 0.9 % of
the `3T` peak: `3.0273` at `θ = 90°` and `−1.0104` at `θ = 0°` (exact `−1`) are
both reproduced.

### C6 — linear-element `L₂` rate of the deflection field

`a/b = 2`, `t/b = 0.01`, Q4 Mindlin, `h = b/level` on both axes.

| level (short side) | elements | dofs | relative `L₂` error of `w` | pairwise order |
|---|---|---|---|---|
| 4 | 8×4 | 135 | 1.2230e-1 | — |
| 8 | 16×8 | 459 | 3.0818e-2 | 1.989 |
| 12 | 24×12 | 975 | 1.3540e-2 | 2.028 |
| 16 | 32×16 | 1683 | 7.4688e-3 | 2.068 |
| 20 | 40×20 | 2583 | 4.6545e-3 | 2.119 |

Least-squares fit over the whole ladder: **p = 2.028** (gate 1.9). All four
pairwise orders are ≥ 1.98, so the conclusion does not depend on the fit
strategy.

## 4. Figures

| figure | caption |
|---|---|
| `beam_static.png` | Cantilever deflection along the span, FE nodes vs the exact cubic, with the nodal relative error on a log scale (round-off, because the load is nodal). |
| `plate_aspect.png` | `α(a/b)` for the simply supported uniform-pressure plate: Navier series (line), Timoshenko table 8 (open squares), BFS FE (filled circles) and the strip limit `5/384`; right panel: C2 gate against the 2 % line. |
| `plate_field.png` | Deflection field of the `a/b = 2` plate (32×16 BFS): FE, Navier, and the pointwise difference normalised by the exact peak (inside ±3 % on a 19-level palette). |
| `modal.png` | Cantilever first mode (Hermite FE nodes vs the exact Bernoulli mode) and the SSSS plate mode (1,1) from the Q4 eigen-path on a `a/b = 2` mesh. |
| `kirsch_edge.png` | Hole-edge hoop stress around the circle (Q8 nodal recovery vs `T(1−2cos2θ)`) and the `K_t` ladder against the `3.0 ± 0.1` band. |
| `convergence.png` | Log–log `L₂` error of the bilinear plate deflection vs element size, with the fitted slope and a `p = 2` reference. |

## 5. Notes, pitfalls, limitations

* **C1 is exact, not convergent.** With the load applied at a node, the cubic
  Hermite solution reproduces a cubic exact field at every node, so the tip error
  is round-off on *any* mesh (1–8 elements are all identical in the tests). The
  informative content of the arm is therefore the rotation, reactions and
  Clapeyron energy identities, and the docstring says so rather than dressing a
  1e-13 as a discretisation result.
* **Kirchhoff reference vs Mindlin element.** The bending references are thin-plate
  (Kirchhoff). The Q4 arms are Mindlin with shear correction 5/6, so their
  residual carries an `O((t/b)²)` model bias; at `t/b = 0.01` it is below the
  gates and is deliberately not removed (removing it would hide the element
  error behind a model correction).
* **`K_t` needs radial resolution.** `n_radial = 2` gives only `K_t = 2.69–2.73`;
  the hole-edge recovery needs at least 4 radial layers to enter the ±0.1 band.
  `K_t` approaches 3 from above (3.027 at 24×8) — extrapolation to zero `h` would
  reduce, not increase, the reported number.
* **SRI Q4 has two spurious mechanisms** (`docs/element-rank-deficiency-audit.md`).
  On the structured SSSS meshes used here they are suppressed by the boundary
  conditions; the MITC shear treatment is available and gives the same square-plate
  answer to 3 digits (checked during development, not part of the committed
  gates).
* **Defect found and fixed while building this face.**
  `structural_verification.KirchhoffPlateSolution.grid()` reported a `y` extent of
  `side·n_y/n_x` while the assembled elements have size `side/n_y`, so for
  `n_y ≠ n_x` the plotting coordinates described a plate the stiffness matrix
  never modelled. `grid()` now spans `[0, side]` on both axes and a regression test
  (`test_square_plate_helper_grid_spans_the_modelled_plate`) pins it.
* **Pinned thread counts corrupt the eigensolver on this platform.** The audit was
  originally recorded as `OMP_NUM_THREADS=1 ...`; re-running that exact command on
  loongarch64 fails. flexiblas 3.3.0 returns a spectrum that does **not** satisfy
  `S v = λ v` for the reduced plate matrix: at 32×16 it reports `λ₁ = −3.9e10`
  (true value `+3.58e4`), i.e. a relative eigen-residual of `1.4` on the full spectrum and
  `1.1e-4`–`1.8e-4` on the retained mode instead of `~1e-16`. The same assembly with the default (or ≥3) thread count reproduces
  the committed `30.10165 Hz` exactly, and a genuine spurious mechanism would show
  up as `λ/λ_max ~ 1e-16`, not as a large negative eigenvalue. Because such a
  spectrum is *plausible looking*, `tensorfem.plate_vibration` now verifies
  `max ‖S v − λ v‖ / λ_max` on the retained modes and raises
  (`EIGEN_RESIDUAL_TOLERANCE = 1e-8`) rather than returning wrong frequencies;
  the audit command no longer pins the thread count and the evidence JSON records
  the thread count actually used. Healthy solves sit at `~5e-16`
  (measured `4.8e-16`–`8.7e-16` across SSSS/clamped/free, thick, Winkler and
  non-square plates), the corrupt spectrum at `1.1e-4`/`6.7e-2`: the `1e-8` bound sits
  seven orders above a healthy solve and at least four below a corrupt one. The guard is covered by
  `test_inconsistent_eigensolver_output_is_rejected`, which scrambles the
  eigenvectors of a real solve and requires the rejection.
* **Not covered here.** Thick/shear-deformable plates beyond the `t/b ≤ 0.1`
  regime, clamped and point-supported plates, orthotropic or laminated plates,
  post-buckling/imperfect plates, plates on elastic foundations (covered by
  `plate_vibration`), pressure-loaded shells and any 3-D effects. The face
  verifies *element and reference correctness*, not ship structural adequacy.
* The repository's three `benchmarks/verified/fem_{beam,plate,modal}` artifact tests
  fail in this checkout because their `convergence.png` files are absent from the
  tree (pre-existing, unrelated to this change; the JSON artifacts themselves
  parse and pass their assertions).

## 6. Error judgement against the baselines

| key quantity | baseline | error | is it < 3 %? |
|---|---|---|---|
| C1 tip deflection | `PL³/(3EI)` | 2.6e-13 | **yes** (machine precision; gate 1 %) |
| C1 tip rotation | `PL²/(2EI)` | 2.4e-13 | **yes** |
| C1 base reaction / moment | `P`, `PL` | 3.2e-13 / 4.7e-13 | **yes** |
| C1 strain energy | `P²L³/(6EI)` | 7.6e-13 | **yes** |
| C2 `α`, `a/b = 1` | Navier series | −0.589 % | **yes** (gate 2 %) |
| C2 `α`, `a/b = 1.5` | Navier series | −0.424 % | **yes** |
| C2 `α`, `a/b = 2` | Navier series | −0.366 % | **yes** |
| C2 `α`, `a/b = 3` | Navier series | −0.326 % | **yes** |
| C2 `α` vs Timoshenko table | table 8 (3 digits) | ≤ 0.058 % | **yes** |
| C3 `f₁` | Euler–Bernoulli | 2.1e-6 | **yes** |
| C4 `f₁₁` (16×8) | Navier | +1.55 % | **yes** |
| C4 `f₁₁` (32×16) | Navier | +0.37 % | **yes** |
| C5 `K_t` | 3.0 | +0.027 (0.9 %) | **yes** (absolute gate ±0.1) |
| C6 observed order | `p ≥ 1.9` | p = 2.028 | **yes** |

No gate in this face is above 3 %; two of them (C2, C5) are stricter (2 % and an
absolute ±0.1) and are met with margin. The residual errors are discretisation
and, for the Mindlin arms, the thin-plate reference bias; none of them is hidden
by a fitted correction factor.
