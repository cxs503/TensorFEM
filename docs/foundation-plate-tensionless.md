# One-sided (tensionless) Winkler foundation on Mindlin plates — qualification

**Module** `src/tensorfem/foundation_plate_tensionless.py`
**Benchmark** `benchmarks/verify_foundation_plate_tensionless.py` → `artifacts/foundation_plate_tensionless/`
**Tests** `tests/test_foundation_plate_tensionless.py` (28 tests)
**Status** all seven gates pass (`all_passed=true` in `artifacts/foundation_plate_tensionless/evidence.json`)

---

## 1. Why this case exists

`docs/beam-foundation-unilateral.md` qualifies the **beam** on a tensionless Winkler
foundation (Hetényi anchor 1.8e-9, lift-off length via a semi-analytic reattachment
condition, `k→∞` / `k→0` limits, active-set residuals ≤ 2.4e-11).  Its own §4.6 then
declares the **plate** path a limitation:

> the unilateral plate path (`solve_plate_foundation`, `unilateral=True`) is implemented
> but **NOT qualified**: the historical SRI Q4 leaves `w` hourglass modes that a partially
> engaged one-sided foundation cannot suppress, and the MITC variant has not been validated
> against a plate contact reference.

That gap is what this case closes.  Three things were missing:

1. a **quadrature-consistent** one-sided foundation operator (the historical path lumps the
   foundation into nodal springs `k A/4`, which is not the ``∫ k NᵀN dA`` operator the
   bilateral solver uses, and which fails the patch test);
2. a **well-posed** one-sided model with an existence/uniqueness argument, so that solver
   convergence can be asserted instead of hoped for;
3. an **evidence ladder with closed-form anchors**, including the limits that the beam case
   verifies but the plate case never did.

## 2. Physical model and sign conventions

A plate of flexural rigidity `D = E t³/(12(1−ν²))` rests on a Winkler subgrade of modulus `k`
that can carry **compression only** (soil / gravel bedding cannot pull the plate down).  With
`w(x)` the transverse deflection measured *into* the foundation and `g₀ ≥ 0` an initial gap,

```
p(x) = k · (w(x) − g₀)⁺ ,            (a)⁺ = max(a, 0)
```

so `p ≥ 0` everywhere (**no tension**), and `p ≡ 0` over the lifted-off region.  Equilibrium is
the stationarity condition of

```
Π(u) = ½ uᵀ K_p u + ½ ∫_A k [(w − g₀)⁺]² dA − fᵀ u .
```

`Π` is `C¹`, piecewise quadratic and coercive once any part of the plate is engaged, hence
**convex with a unique minimiser**; the subgrade law is monotone in `w`, so no cycling active
set can satisfy the KKT conditions.  This is what makes the discrete problem well posed — the
old lumped nodal-spring formulation had no such guarantee, and indeed the SRI variant was
singular whenever a region lifted off.

Sign conventions used throughout (and asserted in the tests): `w > 0` is motion **toward** the
foundation; the Q4 Mindlin element of `tensorfem.plate` uses `γ_xz = ∂w/∂x + θ_x`, so a linear
field `w = a + b x` is shear free with `θ_x = −b`.  The plate element has no membrane DOFs, so
the model is bending-only (no in-plane/foundation friction coupling).

Two one-sided models are distinguished in the code:

| model | condition | implementation |
|---|---|---|
| tensionless Winkler (used here) | `p = k (w−g₀)⁺`, `p ≥ 0`; the bedding compresses | `solve_plate_tensionless` |
| rigid bed (Signorini), penalty | `R = κ (w−g₀)⁺`, penetration `→ 0` as `κ → ∞` | `solve_plate_unilateral_penalty` |

The rigid bed is the `k → ∞` limit of the subgrade model, so the penalty path is the *same*
solver with the penalty stiffness in place of `k`; its accuracy is therefore limited by the
penetration `q/κ`, which is measured (Q7) rather than assumed.

## 3. Discretisation

The subgrade energy is integrated with the 2×2 Gauss rule over the **engaged Gauss points** of
each element (`(w_gp − g₀) > 0`):

```
k_e = Σ_{gp engaged} k N(gp) N(gp)ᵀ detJ(gp)        (consistent, default)
k_e = diag(k A_e / 4)                                (lumped, contrast path only)
```

Before the `max`, the integrand is a polynomial of degree ≤ 2 per direction, so the 2×2 rule
integrates the fully engaged operator **exactly**.  If the zero crossing of a linear deflection
field lies on a mesh line, each element sees a linear `w⁺` and the one-sided patch test is
reproduced to round-off (Q2) — a property the lumped operator cannot have, because
`∫N_i²dA ≠ A/4` for a bilinear basis (it is `A/9` on a rectangle).  The lumped path is kept only
so the benchmark can contrast the two.

The MITC (Dvorkin–Bathe) shear treatment is mandatory for the one-sided path: the one-point SRI
shear rule leaves rank-3 hourglass modes in `w` which a partially engaged foundation cannot
suppress — the exact failure documented in the beam case's limitation §4.6.

## 4. Solvers

**Active-set Newton on the convex potential** (`solve_plate_tensionless`).  Each active set
defines a quadratic model; the engaged operator is re-assembled from `sign(w)` at the Gauss
points, the linear system solved, and the step accepted only if `Π` decreases (Armijo
backtracking).  Monotone descent on a `C¹` convex functional gives global convergence; the
loop stops on the stationarity residual together with a stable engaged set.

Two details are load-bearing and cost a debugging cycle each (see §7):

* the projected-gradient fallback must be masked to the free DOFs — the reaction residing on a
  constrained DOF is large and otherwise poisons the search direction;
* the residual must be scaled by the force magnitudes actually present (`‖K_p u‖`, `‖Cᵀp‖`),
  not by `max|f|`, otherwise the tolerance is meaningless for foundation-dominated systems.

**Independent certificates.**  (i) `solve_plate_tensionless_lbfgs` minimises the *same*
potential with L-BFGS-B (no active-set bookkeeping); agreement is a cross-check, not a
restatement.  (ii) Because `Π` is convex, the minimiser is characterised by "every perturbation
raises `Π`", which the benchmark checks with 16 random directions.

**Penalty alternative** (`solve_plate_unilateral_penalty`) for the rigid bed, with the exact
penetration law `q/κ` as its own gate.

**Element performance.**  `tensorfem.plate.q4_mindlin_stiffness` handles one element per call
and inverts the 2×2 Jacobian with `torch.linalg.inv`; on the loong64 nodes that costs ~50 ms per
element (≈8 min for a 96×96 mesh).  `q4_mindlin_batch` is a vectorised numpy twin with an
analytic 2×2 inverse; it reproduces the library kernel to 4.2e-16 (MITC) / 3.6e-16 (SRI) on
strongly distorted quads and assembles 4000 elements in 0.5 s.

## 5. Evidence ladder (all gates machine-checked)

Geometry: `E = 1e7`, `ν = 0.3`, `t = 0.02` (and `t = 0.005` for the closed-form studies),
`k = 1e6`, `D = 7.326`, characteristic length `l = (D/k)^{1/4} = 0.05203`.

### Q1 — element and operator parity

| quantity | value |
|---|---|
| `q4_mindlin_batch` vs `plate.q4_mindlin_stiffness`, MITC | 4.16e-16 (rel.) |
| same, SRI | 3.57e-16 |
| `q4_foundation_consistent` vs `winkler_foundation.q4_winkler_foundation_stiffness` | 0 (identical) |
| vectorised foundation vs library | 1.56e-16 |
| `1ᵀ K_f 1` vs `k A` (consistent) | 385929.2910920024 vs 385929.2910920025 |

Gate ≤ 1e-14 → **PASS**.

### Q2 — one-sided patch test

`w = 0.008·(0.5 − x)` (zero crossing on the `x = 0.5` mesh line), `θ_x = 0.008`; induced nodal
loads from the *engaged* pressure only, then re-solved.

| n | ndof | engaged Gauss points | relative error | active-set iterations |
|---|---|---|---|---|
| 4 | 75 | 32 / 64 (50 %) | 1.43e-13 | 2 |
| 8 | 243 | 128 / 256 (50 %) | 5.68e-13 | 3 |

Gate ≤ 1e-12 → **PASS** (the field is exact although half of the foundation is disengaged).

### Q3 — exact uniform-load solution (manufactured solution for the whole one-sided path)

Free-edged plate under uniform pressure on a tensionless Winkler bed has the elementary exact
solution `w ≡ q/k`, `p ≡ q` (zero bending energy, no edge moments).

| n | ndof | `w` error | `p` error | all engaged |
|---|---|---|---|---|
| 12 | 507 | 1.36e-13 | 9.20e-14 | yes |
| 24 | 1875 | 4.05e-13 | 2.96e-13 | yes |

Gate ≤ 1e-12 → **PASS**.

### Q4 — closed-form bilateral anchor (Navier double series)

Simply supported plate, side `6l`, uniform load `q = 1e3`, Winkler bed `k = 1e6`; reference
`plate_foundation_analytical.navier_ss_plate_uniform` (Timoshenko & Woinowsky-Krieger, *Theory
of Plates and Shells*, §30 with the Winkler term) gives `w_c = 1.135406e-3`.

| n | h/l | `w_c` | rel. error | all engaged |
|---|---|---|---|---|
| 12 | 0.50 | 1.148456e-3 | 1.15 % | yes |
| 24 | 0.25 | 1.136572e-3 | 0.103 % | yes |
| 48 | 0.125 | 1.133662e-3 | 0.154 % | yes |

2nd-order Richardson on the last pair: `w = 1.13267e-3`, rel. error **0.24 %**.  The residual
floor is the first-order transverse-shear deflection of the Mindlin model, quantified by the
thickness refinement in Q6.  Gate ≤ 1 % → **PASS**.

### Q5 — lift-off under a central point load

Free-edged plate, graded mesh (`β = 3`, half-width `16 l`, centre spacing `0.40 l … 0.118 l`),
`P = 1` at the centre.  The one-sided observable is the **contact radius**
`r_c = √(A_contact/π)`.

| n | `h_c/l` | `r_c/l` | `w(0)` | equilibrium `|∫p dA − P|` | `min p` | stationarity |
|---|---|---|---|---|---|---|
| 24 | 0.403 | 2.8568 | 5.518e-5 | 2.5e-14 | 0.0 | 1.8e-13 |
| 36 | 0.267 | 2.8328 | 5.647e-5 | 9.0e-14 | 0.0 | 2.5e-13 |
| 54 | 0.178 | 2.8279 | 5.761e-5 | 5.5e-13 | 0.0 | 1.4e-12 |
| 81 | 0.118 | 2.8344 | 5.868e-5 | 3.9e-13 | 0.0 | 6.2e-13 |

* monotonicity: **not** strict — the finest level sits 0.2 % above the third one;
* spread over the three finest levels **0.23 %**, over all four 1.02 %;
* observed order of the differences 1.59; Richardson `r_c = 2.8415 l`, gap to the finest
  **0.25 %** (gate 0.5 %);
* energy ledger `U_plate + U_foundation = ½W_ext` closes to ≤ 3e-17;
* convexity/optimality spot check: 16/16 random perturbations raise `Π`, smallest increase
  5.9e-7 (i.e. the solution is a genuine interior minimiser, not a KKT point of one active set);
* independent L-BFGS certificate: identical contact radius, energy gap 7.2e-12,
  `max|Δu|/max|u| = 1.3e-3` — the free outer plate is so compliant that displacement
  differences of that size cost no energy.

**Honest caveat.** The centre deflection `w(0)` drifts **+6.3 %** over the 3.4× refinement and is
*not* h-converged: a point load on a plate produces a logarithmically singular slope, and the
lifted outer plate is nearly energy-free.  The contact radius — the quantity that actually
characterises the one-sided model — is converged; `w(0)` is reported as unconverged rather than
dressed up with a threshold.  A closed-form lift-off radius exists for the *beam*
(`beam_foundation.lift_off_length_analytic`) but not for the plate, so this gate is a
convergence + conservation + independent-solver qualification, not an anchor.

### Q6 — degeneracy limits

`t = 0.005`, simply supported plate, `k` swept over six decades (`q = 1e3`):

| `k` | `kL⁴/D` | rel. error vs Navier | all engaged |
|---|---|---|---|
| 1e-6 | 1.1e-7 | 1.03 % | yes |
| 1e2 | 1.1e-2 | 1.03 % | yes |
| 1e4 | 1.1 | 0.999 % | yes |
| 1e6 | 1.1e2 | 0.073 % | yes |
| 1e8 | 1.1e4 | 0.070 % | yes |

The small-`k` residual is a **model** difference (Mindlin first-order shear vs Kirchhoff), proven
by thickness refinement at `k → 0`: 2.07 % (`t = 0.02`), 1.45 % (`t = 0.01`), 1.01 %
(`t = 0.005`) — monotone, consistent with an `O(t²)` offset, not with an element error.

`k → ∞` on a free-edged plate (exact solution `w = q/k`, `p = q`):

| `k` | `w_max k/q` | `p̄/q` | `w` error |
|---|---|---|---|
| 1e4 | 1.000000000006 | 1.000000000000 | 6.1e-12 |
| 1e6 | 1.000000000000 | 1.000000000000 | 5.4e-13 |
| 1e8 | 1.000000000000 | 1.000000000000 | 4.8e-14 |

Gates: `k→0` ≤ 2 % → **PASS**; `k→∞` ≤ 1e-10 → **PASS**.

### Q7 — penalty (rigid bed) alternative

Free-edged plate, uniform load `q = 2e3`, tensionless law with `κ` in place of `k`.  The exact
penetration is `q/κ` and the complementarity product is `q²/κ`.

| `κ` | penetration | exact `q/κ` | rel. error | `R·penetration` |
|---|---|---|---|---|
| 1e5 | 2.0000e-2 | 2.0e-2 | 1.6e-13 | 40.000 |
| 1e6 | 2.0000e-3 | 2.0e-3 | 1.2e-13 | 4.000 |
| 1e7 | 2.0000e-4 | 2.0e-4 | 9.1e-14 | 0.400 |
| 1e8 | 2.0000e-5 | 2.0e-5 | 1.7e-14 | 0.040 |

The observed decay ratio of the complementarity product is `1000.0000` against the expected
`κ`-ratio `1000` — the penalty regularisation is verified against its own analytical law, so the
rigid-bed penetration is `O(1/κ)` (measured), not asserted.  Gate → **PASS**.

### Figures

* `artifacts/foundation_plate_tensionless/lift_off_map.png` — contact map (engaged / lifted-off
  Gauss points) and the deflection field.
* `artifacts/foundation_plate_tensionless/radial_profile.png` — `w(r)/w(0)` against the bilateral
  Hetényi `kei` solution: the two agree inside the contact patch and split exactly at `r_c`,
  where the bilateral solution keeps pulling the plate down.
* `artifacts/foundation_plate_tensionless/limits.png` — closed-form error across the Winkler
  range and the thickness refinement of the Mindlin offset.
* `artifacts/foundation_plate_tensionless/penalty.png` — penetration against the exact `q/κ`.

## 6. Error judgement against references

| quantity | reference | measured error | < 3 % ? |
|---|---|---|---|
| element stiffness (MITC/SRI) | `plate.q4_mindlin_stiffness` | 4.2e-16 / 3.6e-16 | yes (machine) |
| foundation operator | `winkler_foundation` consistent matrix | 0 / 1.6e-16 | yes (machine) |
| one-sided patch test | exact linear field | 5.7e-13 | yes |
| uniform-load field `w`, `p` | exact `q/k`, `q` | 4.1e-13 / 3.0e-13 | yes |
| SS plate centre deflection | Navier double series | 0.154 % (Richardson 0.24 %) | yes |
| SS plate across `k` sweep | Navier double series | ≤ 1.03 % | yes |
| `k → ∞` rigid support | exact | 1e-11 | yes |
| penalty penetration | `q/κ` | 1.6e-13 | yes |
| lift-off contact radius | **no closed form** (plate) | Richardson self-consistency 0.25 % | declared |
| lift-off centre deflection | **no closed form** | **6.3 % h-drift, not converged** | **no — declared** |
| lumped nodal-spring operator (contrast) | patch test fails by construction | n/a | documented |

Every gated quantity is below 3 %; the one quantity that is *not* converged (`w(0)` in the
lift-off problem) is reported as such and is excluded from the gates.

## 7. Pitfalls found on the way (all now covered by tests or gates)

1. **Projected gradient must be masked to the free DOFs.**  Without the mask the reaction on a
   constrained DOF dominates the fallback direction; in the first implementation this destroyed
   an exact solution (the patch-test case) and made the solver crawl at `α ≈ 3e-5`.
2. **Residual scaling.**  Dividing by `max|f|` (nodal loads) instead of the force magnitudes
   present (`‖K u‖`, `‖Cᵀp‖`) makes a converged foundation-dominated solve look like a residual
   of 6.3.  The scale is now computed per iterate.
3. **Gauss-point engagement vs nodal engagement.**  Engagement decided by the sign of `w` at the
   Gauss points with the consistent operator is exact for the one-sided patch test.  A *node*
   based engagement with a consistent (non-diagonal) operator has no unique meaning; nodal
   engagement forces the lumped operator, which breaks the patch test.
4. **Batched 2×2 inverses are pathologically slow on loong64**: `np.linalg.inv` on `(1000,2,2)`
   takes 7.7 s, and `torch.linalg.inv` per element ~50 ms.  Use the analytic 2×2 inverse.
5. **The Mindlin–Kirchhoff offset must be demonstrated, not hidden.**  At `t/L = 0.064` the
   `k→0` case sits 2.1 % above the Kirchhoff series; a t-refinement proves it is the shear
   deformation.  Without that proof the closed-form gate would look like an element defect.
6. **Point-load deflection in the lift-off problem is not an `h`-convergent observable**; the
   contact radius is.  Choose the observable before writing the gate.
7. **A "spring that switches off" is not a contact problem.**  The tensionless Winkler law
   (`p = k w⁺`) has no complementarity condition `p·g = 0` — the bedding compresses.  Mixing the
   two (as the historical lumped path did) makes the discrete problem non-convex/singular.  Only
   the rigid-bed penalty path carries `R·penetration → 0`.

## 8. Limitations

* Linear-elastic Winkler subgrade only: no Pasternak/Kerr shear transfer, no friction, no
  subgrade plasticity or cyclic degradation (see `kerr_foundation.py` for the bilateral
  two-parameter models).
* Small deflection (Mindlin kinematics, no von Kármán coupling); `w/h` and `w/l` must stay small.
* No in-plane/foundation coupling (the Q4 element is bending-only), so no foundation-friction
  or gravity-driven sliding.
* The lift-off radius of a plate has no closed-form reference; it is qualified by convergence,
  conservation and an independent solver.  `w(0)` in that configuration is not converged.
* The penalty path is a regularisation: the penetration is `O(1/κ)` and must be chosen against
  the required tolerance, not "large".
* Dynamic lift-off, impact and buckling of plates with one-sided foundations are out of scope.

## 9. Reproduce

```bash
python benchmarks/verify_foundation_plate_tensionless.py     # gates + evidence.json + figures
python -m pytest tests/test_foundation_plate_tensionless.py -q   # 28 tests, ~20 s
```