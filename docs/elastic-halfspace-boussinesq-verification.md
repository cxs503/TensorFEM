# Elastic half-space under a uniformly loaded patch — Boussinesq verification of the truncated HEX8 path

Verification-debt item: every 3-D solid benchmark in this repository loads a
**closed** body (torsion bar, cylinder, plate, sphere, Hertzian pair). Nothing
checked what happens when the reference solution lives on an **unbounded**
domain and the mesh has to truncate it. This note closes that gap for the merged
`solid3d` path (`hex8_stiffness` + `sparse_core`), with the closed forms in
`src/tensorfem/elastic_halfspace.py` and the executable evidence in
`examples/halfspace_verification.py` / `tests/test_elastic_halfspace.py`.

## 1. Case, geometry and parameters

| item | value |
| --- | --- |
| half-space | isotropic, `E = 1`, `nu = 0.3` (all results scale exactly: `w ∝ p(1-nu²)/E`) |
| loaded patch | uniform pressure `p = 1` on a `2a x 2a` square, `a = 1` |
| quantity 1 | surface settlement at the patch centre and at the patch corner |
| quantity 2 | `sigma_zz` on the load axis, from the surface to `20a` |
| exact reference 1 | Boussinesq/Love settlement `w = (p(1-nu²)/πE) ∫_A dA/r` |
| exact reference 2 | Newmark corner factor, `σ_z/p` closed form |
| FE model | quarter box `[0,L]x[0,L]x[0,D]`, `L = D = 20a`, graded HEX8 |
| boundary conditions | `u_x = 0` on `x = 0, L`; `u_y = 0` on `y = 0, L`; `u_z = 0` on `z = D` |
| loads | consistent nodal loads of the uniform pressure on `[0,a]²` |

The three planes `x = 0`, `y = 0` are symmetry planes; the outer faces and the
bottom carry the *weakest* admissible truncation constraints (rollers and a
vertical-only bottom), because clamping the base would stiffen the model without
modelling anything physical and would hide the truncation error behind a second,
larger error.

### The two closed forms

With `I(u,v) = u·asinh(v/u) + v·asinh(u/v) = ∫₀ᵘ∫₀ᵛ dr ds/√(r²+s²)` (the
antiderivative, `∂²I/∂u∂v = 1/√(u²+v²)`), the settlement of a uniformly loaded
rectangle is a 2×2 corner inclusion–exclusion:

```
w(x,y) = C [ I~(x,y) − I~(x−lx,y) − I~(x,y−ly) + I~(x−lx,y−ly) ],  C = p(1−nu²)/(πE)
I~(u,v) = sign(u)·sign(v)·I(|u|,|v|)
```

and the vertical stress uses the same inclusion–exclusion on the Newmark corner
factor, written with `atan2` so that the branch survives the wide-patch case
`m²n² > m²+n²+1` where the textbook `arctan` form loses `π`.

## 2. Method

1. **Grade the oracle first.** Both closed forms are checked against a composite
   10-point Gauss quadrature of the *point-load* kernels
   (`3p z³/2π (r²+z²)^{-5/2}` and `C/r`), which shares no algebra with them.
   Agreement: `1e-12` relative for settlement, `1e-9` for stress, plus the
   antiderivative identity `∂²I/∂u∂v = 1/√(u²+v²)` by Richardson-extrapolated
   central differences (`1e-7`, roundoff-limited).
   Literature anchors: a square footing of equal area settles within 1% of
   Love's circular-footing `2(1−nu²)pR/E`.
2. **Refine the quarter model jointly** in the near field and the far field
   (the first far-field cell is held at the near-field spacing `h₀`, growth ratio
   solved by bisection to land exactly on `20a`), so the comparison is unaided by
   a grading ratio tuned for one level.
3. **Separate the truncation question** from the mesh question: fix the grading
   ratio at 1.3 and vary only the domain extent `L = 4a, 8a, 20a`.
4. Grade the settlement against the closed form and `σ_z` at element centroids
   against the closed form **at the same point** (no axis averaging, which would
   flatter the answer).

## 3. Results

All numbers are machine-produced by `examples/halfspace_verification.py`
(`docs/halfspace_evidence.json`); nothing is transcribed by hand.

**Exact reference values** (`p = E = a = 1`): centre `w = 2.042403`,
corner `w = 1.021202`; in dimensionless form `wE/(pa) = 2.0424` and `1.0212`.

**Quarter model vs the half-space**

| level | nodes | DOF | `h₀/a` | `w_centre` FE | error | `w_corner` FE | error | `σ_z` worst err (axis) | reaction balance | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coarse | 1859 | 5577 | 0.50 | 1.90827 | **6.57%** | 0.91693 | **10.21%** | 6.9% | 2.0e-11 | 7 s |
| medium | 4335 | 13005 | 0.25 | 1.94236 | **4.90%** | 0.92620 | **9.31%** | **1.7%** | 2.8e-11 | 46 s |
| fine | 7497 | 22491 | 0.125 | 1.94720 | **4.66%** | 0.92727 | 9.20% | — | — | 172 s |

**Domain-truncation study** (grading ratio fixed at 1.3, `h₀ = 0.5a`, `D = 20a`)

| `L/a` | `w_centre` FE | error vs exact | reaction balance |
| --- | --- | --- | --- |
| 4 | 2.37934 | **+16.50%** (high) | 1e-11 |
| 8 | 1.96377 | −3.85% (low) | 1e-11 |
| 20 | 1.93234 | −5.39% (low) | 1e-11 |

Figures: `docs/figures/halfspace_settlement_convergence.png` (log–log error vs
element size), `halfspace_stress_profile.png` (`|σ_z|/p` vs depth, FE vs
Newmark), `halfspace_settlement_profile.png` (surface settlement along the
symmetry plane, FE vs Boussinesq), `halfspace_sigma_zz_field.png` (`σ_z/p`
contour on the `x–z` slice).

## 4. Error judgement — is it below 3%?

| quantity | reference | best error | below 3%? |
| --- | --- | --- | --- |
| settlement oracle vs point-load quadrature | independent quadrature | `1e-12` | **yes** |
| Newmark stress vs point-load quadrature | independent quadrature | `1e-9` | **yes** |
| `∂²I/∂u∂v = 1/√(u²+v²)` | analytic | `1e-7` | **yes** |
| **`σ_z` on the load axis** (medium mesh) | Newmark closed form | **1.7%** | **yes** |
| **patch-centre settlement** (fine mesh) | Boussinesq/Love | **4.66%** | **no** |
| patch-corner settlement (fine mesh) | Boussinesq/Love | 9.20% | **no** |
| load balance (reaction sum) | applied load | `2.8e-11` | **yes** |

Honest FAIL on the settlement criterion at affordable cost, with the mechanism
identified:

* the settlement integrand is `1/r`-singular at the patch corner and very slowly
  decaying at infinity, so the value is dominated by material several patch
  widths away. The observed convergence is ~`O(h^1)` and still 4.66% at 22 491
  DOF; extrapolating the last two levels leaves roughly 3–4% — the mesh, not the
  truncation, is the binding constraint at `L = 20a`.
* the corner settlement sits on the traction discontinuity at the patch edge
  (`σ_z` jumps from `p` to `0`), which no displacement interpolation resolves at
  `O(h²)`; it stays near 9%.

## 5. Notes, traps and scope

* **Trap 1 — the interior-point branch of the Newmark chart.** The 2×2 corner
  inclusion–exclusion must use offsets *corner minus field point*, with the
  quadrant sign carried by the antiderivative. Writing the offsets as
  *field minus corner* (the natural thing to code) flips two of the four signs
  and collapses the stress to ≈0 for field points **inside** the loaded patch —
  silently, with every external point still correct. The two formulations agree
  to 4 digits at exterior points, so only an interior quadrature check catches it.
* **Trap 2 — patch origin.** The quarter model loads `[0,a]²`, so the open
  interval of the full patch is `[-a,a]²` and the closed form, anchored at the
  origin, must be asked at `(x+a, y+a)`. Omitting the shift compares the FE
  *centre* with the closed-form *corner* and reports a fictitious 30–120%
  stress error (measured before the fix: 54% coarse, 122% medium).
  After the fix the same runs give 6.9% and 1.7%.
* **Trap 3 — domain truncation is not conservative.** A 4-patch-wide box is
  16.5% **high** and a 20-patch-wide box 5.4% **low**, both with a reaction
  balance of `1e-11`. Confinement amplifies the stress under the load faster than
  it suppresses the far field, so "my model is stiffer, therefore conservative"
  is wrong here. No residual, energy or reaction check can see this; only the
  closed form can, which is why the oracle had to come first.
* **Trap 4 — bottom constraint.** Clamping `u_z` *and* the horizontal components
  at `z = D` (the usual reflex) removes the truncation signal entirely at these
  depths; the base must be free to slide.
* Scope: linear elasticity, uniform pressure, square patch, quarter symmetry.
  The stratified/rigid-base footing problem and the *rigid* punch (Cattaneo /
  Boussinesq rigid-footing `π(1−nu²)pa/2E`) are not covered — the rigid punch
  needs a contact iteration, not a prescribed traction.
* Cost: the fast part of `tests/test_elastic_halfspace.py` runs 17 checks in
  ~105 s; the refinement ladder is behind `TENSORFEM_RUN_SLOW_HALFSPACE=1`.

## 6. Follow-ups

1. Rigid-footing (Boussinesq punch) settlement `π(1−nu²)pa/(2E)` — needs the
   existing contact machinery; would also re-use this oracle.
2. Two-layer half-space (Timoshenko's stratified solution) — the marine
   relevance: keel-block and dock-block compliance on layered bedding.
3. Richardson/GCI on the settlement ladder with a *second* domain size to
   separate the `h` and `L` error contributions (the two are entangled at
   `L = 20a` in the present study).