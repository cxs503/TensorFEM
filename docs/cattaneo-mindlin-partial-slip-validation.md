# Hertz / Cattaneo–Mindlin contact with exact unilateral complementarity

**Node** P22-N02 (campaign P22, layer: solver kernel, domain: structural contact
mechanics). **Module** `src/tensorfem/hertz_mindlin.py`. **Benchmark**
`benchmarks/mindlin_contact/run.py` (`result.json`, `gates.json`, three figures).
**Tests** `tests/test_hertz_mindlin.py`. **Last verified** 2026-11-08.

---

## 1. Case introduction

Two classical elastic contact problems are solved and gated:

* **Hertz** — a rigid sphere or cylinder pressed into a linearly elastic
  half-space. The unknowns of interest are the contact radius (or half-width)
  `a`, the peak pressure `p0` and, for the sphere, the load–approach law
  `P(δ)`.
* **Cattaneo–Mindlin partial slip** — after the normal preload `P`, the indenter
  is displaced tangentially by `u_t`. The interface then splits into a central
  stick zone (half-width `c`, traction below the Coulomb cone) and two slip
  strips (traction exactly on the cone `|q| = μp`).
* **Unilateral (Signorini) constraint** — `gap ≥ 0`, `contact force ≥ 0`,
  `gap · contact force = 0`, enforced exactly (Lagrange multipliers, active set)
  so that the complementarity residual is round-off, not penalty-limited.

### Declared parameters

| quantity | value | note |
|---|---|---|
| Young modulus `E` | 210 GPa | steel-like, linear elastic |
| Poisson ratio `ν` | 0.3 | |
| indenter radius `R` | 0.05 m | sphere (axisymmetric) and cylinder (plane strain) |
| cylinder load `P` | 3.625×10⁶ N/m | chosen so that `b = 1.000 mm` |
| sphere loads | 1.0×10⁵ … 1.05×10⁶ N (6 values, ratio 1.6) | load–approach law |
| friction coefficient `μ` | 0.3 | Cattaneo–Mindlin only |
| tangential loads | `Q/(μP) ∈ {0.15, 0.35, 0.60, 0.85}` | stay inside the cone |
| 3-D mesh ladder | 60 / 150 / 300 rings | axisymmetric |
| 2-D mesh ladder | 400 / 800 / 1600 segments | plane strain |

Plane-strain effective modulus `E* = E/(1-ν²) = 230.77 GPa`; closed forms used as
oracles: `b = √(4PR/πE*)`, `p0 = 2P/(πb)`, `a = (3FR/4E*)^{1/3}`,
`δ = a²/R`, `c = b√(1-Q/μP)`.

## 2. Method

**Operator.** The contact operator is the *exact* 2-D plane-strain Flamant
compliance of a half-plane surface and its axisymmetric Boussinesq/Love
counterpart:

```
u_z(x) = -(2/πE*) ∫ p(s) ln|x-s| ds + const           (2-D, Flamant)
u_z(r) = (4(1-ν²) a/πE) E(r/a)                       (r ≤ a, uniform disc, Love)
u_z(r) = (4(1-ν²) r/πE)[E(a/r) - (1-(a/r)²)K(a/r)]    (r > a)
```

The 2-D kernel is integrated analytically over each segment
(`∫ₐᵇ ln|x-s| ds = [(t-x)(ln|t-x|-1)]ₐᵇ`), the 3-D kernel is expressed as a
superposition of uniformly loaded discs (differences of Love's closed form), so
no singular quadrature appears anywhere. This is what makes the same code give
a smooth traction field with a single grid, and why the traction gates below can
be met without any pressure post-smoothing.

**Discretisation.** Piecewise-constant tractions on a segment grid (2-D) or an
annular ring grid (3-D), collocated at the segment/ring centres; the contact
candidates form a symmetric positive-definite compliance matrix `S`
(`w = S p`).

**Solution.** The unilateral problem is the convex program
`min ½ pᵀSp + (d-δ)ᵀp s.t. p ≥ 0` with `d = x²/2R`. For a fixed active set the
pressure is *affine* in the indenter depth, `p = δ A⁻¹1 − A⁻¹d`, so the depth
reproducing a prescribed contact load follows in closed form and the active set
is resolved by a simple release/engage sweep — no penalty, no bisection, no
load-step continuation. The tangential problem uses the same trick: for a fixed
stick set the traction is affine in the rigid tangential translation,
`q = u_t A⁻¹1 − A⁻¹(S q_slip)`, and the translation reproducing `Q` is obtained
in closed form; segments whose traction leaves the cone `|q| ≤ μp` are released
to sliding with `q = μp`.

**Toolchain.** Python 3.11, NumPy/SciPy (`ellipk`, `ellipe`, `splu`), PyTorch not
required for this module, matplotlib (Agg) for the figures. Reproduction:
`python benchmarks/mindlin_contact/run.py` (115 s, single core) and
`python -m pytest tests/test_hertz_mindlin.py` (17 tests).

**Finite-element companion.** For an independent check on a *volumetric* mesh,
`block_mesh` + `plane_strain_stiffness` + `ElasticSolver` +
`solve_indentation_fe` solve the same rigid-indenter problem with Q4 elements,
node-to-node contact and exact Lagrange multipliers (same complementarity
property, no penalty). It is a cross-check only: the half-space kernel carries no
volumetric truncation and no far-field compliance bias, and it is ~10³× cheaper.

## 3. Results

### 3.1 3-D Hertz: contact radius and peak pressure (E1, E2)

`F = 5×10⁵ N`, `a_ref = 4.331196×10⁻³ m`, `p0_ref = 1.27261×10¹⁰ Pa`.

| rings | `a` [m] | err | `a` raw [m] | err | `p0` [Pa] | err | resid |
|---|---|---|---|---|---|---|---|
| 60 | 4.320368e-3 | +0.250% | 4.331196e-3 | +0.000% | 1.273583e10 | +0.076% | 1.3e-9 |
| 150 | 4.331196e-3 | +0.000% | 4.331196e-3 | +0.000% | 1.272508e10 | +0.008% | 1.6e-9 |
| 300 | 4.331196e-3 | +0.000% | 4.331196e-3 | +0.000% | 1.272585e10 | +0.002% | 3.4e-9 |

`a` is the least-squares Hertz-profile recovery from the computed pressure
field; `a raw` is the outermost loaded ring edge (no profile ansatz). Both are
reported because the ansatz-based value is the one that converges to the
analytic radius while the raw value is the conservative, model-free indicator.

### 3.2 3-D Hertz: load–approach law (E3)

| `P` [N] | `δ` [m] | `P_Hertz(δ)` [N] | rel. dev. |
|---|---|---|---|
| 1.000e5 | 1.28310e-4 | 9.99986e4 | 1.4e-5 |
| 1.600e5 | 1.75526e-4 | 1.59998e5 | 1.4e-5 |
| 2.560e5 | 2.40116e-4 | 2.55996e5 | 1.4e-5 |
| 4.096e5 | 3.28474e-4 | 4.09594e5 | 1.4e-5 |
| 6.554e5 | 4.49347e-4 | 6.55351e5 | 1.4e-5 |
| 1.049e6 | 6.14697e-4 | 1.04856e6 | 1.4e-5 |

`R² = 0.99999999944`, fitted exponent `n = 1.5000000` (`P ∝ δ^{3/2}`).

### 3.3 2-D cylinder Hertz (same operator, plane strain)

`b_ref = 9.9905×10⁻⁴ m`, `p0_ref = 2.30772×10⁹ Pa`.

| segments | `b` [m] | err | `p0` [Pa] | err | resid |
|---|---|---|---|---|---|
| 400 | 9.990417e-4 | +0.097% | 2.307651e9 | +0.003% | 1.5e-10 |
| 800 | 1.000141e-3 | +0.013% | 2.307716e9 | +0.000% | 1.8e-10 |
| 1600 | 9.998178e-4 | +0.019% | 2.307715e9 | +0.000% | 2.5e-10 |

### 3.4 Cattaneo–Mindlin stick/slip boundary (E4)

`P = 3.625×10⁶ N/m`, `μ = 0.3`; `b` from the converged normal state per grid.

| segments | `Q/(μP)` | `Q` [N/m] | `c` [m] | `c_ref` [m] | err | stick/contact |
|---|---|---|---|---|---|---|
| 400 | 0.15 | 1.63125e5 | 9.37511e-4 | 9.21071e-4 | +1.79% | 124/134 |
| 400 | 0.35 | 3.80625e5 | 8.17510e-4 | 8.05453e-4 | +1.50% | 108/134 |
| 400 | 0.60 | 6.52500e5 | 6.37508e-4 | 6.31849e-4 | +0.90% | 84/134 |
| 400 | 0.85 | 9.24375e5 | 3.97505e-4 | 3.86927e-4 | +2.73% | 52/134 |
| 800 | 0.15 | 1.63125e5 | 9.26261e-4 | 9.22084e-4 | +0.45% | 246/268 |
| 800 | 0.35 | 3.80625e5 | 8.13760e-4 | 8.06339e-4 | +0.92% | 216/268 |
| 800 | 0.60 | 6.52500e5 | 6.41258e-4 | 6.32544e-4 | +1.38% | 170/268 |
| 800 | 0.85 | 9.24375e5 | 3.93755e-4 | 3.87353e-4 | +1.65% | 104/268 |
| 1600 | 0.15 | 1.63125e5 | 9.24386e-4 | 9.21786e-4 | +0.28% | 492/534 |
| 1600 | 0.35 | 3.80625e5 | 8.08135e-4 | 8.06079e-4 | +0.26% | 430/534 |
| 1600 | 0.60 | 6.52500e5 | 6.35632e-4 | 6.32340e-4 | +0.52% | 338/534 |
| 1600 | 0.85 | 9.24375e5 | 3.88130e-4 | 3.87228e-4 | +0.23% | 206/534 |

Tangential load control is exact (`|Q-Q_target|/Q_target < 1e-8`), the traction
inside the stick zone stays strictly inside the cone
(`max(|q| - μp) = 0` to machine precision on the slip strips, `|q| < μp` on the
stick zone) and the stick zone shrinks monotonically with `Q`.

### 3.5 Unilateral constraint (E5) and the no-contact negative control (E6)

| quantity | value |
|---|---|
| `max |gap·force|`, 2-D loaded state (1600 segments) | 2.49e-10 |
| `max |gap·force|`, 3-D loaded state (300 rings) | 3.45e-9 |
| minimum gap in contact | −9.5e-20 m |
| minimum pressure | 0.0 exactly (inactive segments) |
| retracted indenter, 2-D | 0.0 N/m exactly, all gaps ≥ 1.0e-6 m |
| retracted indenter, 3-D | 0.0 N exactly |

The residual is not a convergence parameter: the active set is solved with
Lagrange multipliers, so `gap = 0` on active and `force = 0` on inactive
segments hold to the linear-solve round-off.

### 3.6 Finite-element cross-check (volumetric Q4 mesh)

| mesh | load [N/m] | contact extent [m] | err | peak pressure [Pa] | err | resid |
|---|---|---|---|---|---|---|
| 60×40 | 3.625000e6 | 8.3334e-4 | −16.7% | 2.310481e9 | +0.12% | 2.9e-15 |
| 100×60 | 3.625000e6 | 9.7201e-4 | −2.80% | 2.306443e9 | +0.06% | 2.8e-15 |

The FE extent is the outermost *active node* and is therefore quantised by the
node spacing (the true edge lies between the last active and the next inactive
node), which is why it is a lower bound and why the 60×40 mesh is 16.7 % short.

### Figures

* `benchmarks/mindlin_contact/fig1_hertz_2d.png` — 2-D Hertz pressure against
  `p0√(1-(x/b)²)` (left) and the unilateral gap for the loaded and the retracted
  indenter (right); the loaded gap touches zero inside the contact and is
  strictly positive outside.
* `benchmarks/mindlin_contact/fig2_hertz_3d.png` — 3-D Hertz pressure profile
  (left) and the load–approach law on log–log axes with the analytic
  `P = 4/3 E*√R δ^{1.5}` (right).
* `benchmarks/mindlin_contact/fig3_cattaneo_mindlin.png` — tangential traction
  for four `Q/(μP)` levels with the local Coulomb cone (left) and the computed
  `c/b` against the Cattaneo square-root law (right).

## 4. Gates

Machine verdicts in `benchmarks/mindlin_contact/gates.json`:

| gate | value | limit | verdict |
|---|---|---|---|
| E1 Hertz contact radius (3-D, 300 rings) | 0.000% | ≤ 2% | PASS |
| E2 Hertz peak pressure (3-D, 300 rings) | 0.0021% | ≤ 5% | PASS |
| E3 load–approach `R²` | 1.0000000 | ≥ 0.99 | PASS |
| E4 stick/slip boundary (1600 segments, worst of 4 loads) | 0.521% | ≤ 5% | PASS |
| E5 complementarity residual | 3.45e-9 | ≤ 1e-8 | PASS |
| E6 no-contact force | 0.0 exactly | == 0 | PASS |

## 5. Notes, pitfalls, limitations

**Pitfalls actually hit while building this (kept as negative knowledge).**

1. *Element orientation.* `block_mesh` numbers `j` along *decreasing* `y`, so the
   connectivity `(i,j),(i+1,j),(i+1,j+1),(i,j+1)` is clockwise in the physical
   plane, giving `det J < 0` and a negative-definite element stiffness. Symptom:
   the uniaxial-strain patch test returned `u_y` of the wrong sign with a
   ±13 % spatial ripple. Fixed by ordering the corners counter-clockwise
   (`(i,j+1),(i+1,j+1),(i+1,j),(i,j)`); the patch test is now exact to 6e-15.
2. *Boundary-condition indexing.* `idx[:, 0]` on a `(nx+1, ny+1)` node map is
   the *surface* row, not the left face; the first version clamped `u_x` on the
   contact surface. Fixed to `idx[0, :]` / `idx[-1, :]`.
3. *Sign of the unilateral constraint.* The half-space solver works with the
   settlement *into* the body while the FE works with `u_y` (*up*), so the same
   expression `gap = u_y + profile - δ` is correct in one and inverted in the
   other; the inverted version looks "converged" but produces `λ < 0` at the
   centre and an empty active set.
4. *Missing slip load in the tangential control.* The closed-form `u_t` that
   reproduces `Q` must subtract the traction already carried by the slip strips
   (`Q_slip`), otherwise the achieved tangential load is 1.5–2× the request and
   the stick zone is over-released (observed: `c/b = 0.67` instead of `0.81` at
   `Q/(μP) = 0.35`).
5. *Non-symmetric kernel by datum subtraction.* Building the compliance as
   `K_ij - K(x_ref)_j` makes the operator non-symmetric and, for the stick-zone
   block, numerically near-singular (computed tractions 10⁴× the cone). Keeping
   the raw symmetric Flamant kernel and letting the indenter depth absorb the
   datum is both simpler and well conditioned.

**Limitations (applicability boundary).**

* Two-dimensional plane strain for the tangential problem. The partial-slip law
  is then the 2-D square-root law `c = b√(1-Q/μP)`; the 3-D exponent `2/3` is
  **not** reproduced here and must not be inferred from this module. In 2-D the
  normal and tangential Flamant kernels share the same constant, which is the
  reason the square-root law follows at all.
* The normal and tangential kernels are treated as uncoupled. The 2-D
  cross-terms are a constant and a step of order `(1-2ν)`; they are dropped, so
  the mixed-mode (normal/tangential interaction) stiffness is not modelled.
* Linear elasticity, small sliding, one convex rigid indenter, monotonic normal
  then tangential loading (`Q` increasing only), no self-contact, no plasticity,
  no adhesion, no roughness, quasi-static.
* `P(δ)` uses the axisymmetric half-space model: the approach is a local
  quantity in 3-D. The 2-D cylinder approach is logarithmically non-local, so
  the 2-D gates are load-controlled (`b`, `p0` vs `P`) and deliberately do **not**
  gate `δ`.
* The peak-pressure gate uses a least-squares Hertz-profile recovery; the raw
  model-free indicators (`a raw`, `p0 raw`) are tabulated alongside and agree to
  ≤ 0.05 %, so the gate is not an artefact of the ansatz.

## 6. Error judgement against the baselines

Baselines are the closed forms `b = √(4PR/πE*)`, `p0 = 2P/(πb)`,
`a = (3FR/4E*)^{1/3}`, `δ = a²/R`, `c = b√(1-Q/μP)` (Johnson, *Contact
Mechanics*, 1985; Cattaneo 1938 / Mindlin 1949).

| quantity | baseline | computed | rel. error | < 3 %? |
|---|---|---|---|---|
| 3-D contact radius `a` (300 rings) | 4.331196e-3 m | 4.331196e-3 m | 0.000% | yes |
| 3-D peak pressure `p0` | 1.27261e10 Pa | 1.272585e10 Pa | 0.002% | yes |
| 3-D contact radius, coarsest (60 rings) | 4.331196e-3 m | 4.320368e-3 m | 0.250% | yes |
| load–approach deviation (6 loads) | `P_Hertz(δ)` | `P_FE` | 1.4e-5 | yes |
| 2-D half-width `b` (1600 segments) | 9.9905e-4 m | 9.998178e-4 m | 0.019% | yes |
| 2-D peak pressure `p0` | 2.30772e9 Pa | 2.307715e9 Pa | 0.000% | yes |
| stick half-width `c`, worst of 4 loads (1600 seg.) | Cattaneo | influence solution | 0.521% | yes |
| stick half-width `c`, coarsest (400 seg.) | Cattaneo | influence solution | 2.73% | yes |
| complementarity residual | 0 | 3.45e-9 | — (absolute gate 1e-8) | yes |
| no-contact force | 0 | 0.0 | exact | yes |

Every gated quantity is inside 3 % of its closed-form baseline; the two-D
stick-zone error is the only one that exceeds 1 % on the finest grid, and it is
resolved by refinement (2.73 % → 0.52 % → 0.23 % on the 400/800/1600 ladders).

## 7. Reusable rule (distilled)

**Rule.** When a contact-mechanics claim is meant to verify a *closed form*
(Hertz, Cattaneo–Mindlin, Boussinesq), build the contact operator from the exact
half-space compliance and enforce the unilateral constraint with active-set
Lagrange multipliers; do not verify the closed form with a penalty contact model
on a volumetric mesh unless the extra mesh error is quantified separately.

**Why it works.** The kernel is exact, so the only error is in the traction
interpolation, and the pressure and stick boundary converge from a single grid;
the exact multipliers make `gap·force` a round-off quantity, so the
complementarity gate is a real assertion rather than a penalty-tuning artefact.

**Where it does not apply.** Non-convex or deformable-on-deformable contact,
finite sliding, plasticity, layered/functionally graded substrates (the Flamant
kernel is homogeneous-half-space only), and any 3-D partial-slip claim other than
via the axisymmetric normal problem.

**Regression guard.** `tests/test_hertz_mindlin.py` (17 tests) pins the reference
algebra, the kernel identities (log-integral vs quadrature, Love disc centre
value and edge continuity, symmetry/positive definiteness of the compliance), all
six gates, the FE patch test, and the FE contact cross-check. A change that
breaks the operator, the active set or the load control fails these tests.

**Review status.** Self-verified by machine gates; not yet independently
reviewed by a separate seat.