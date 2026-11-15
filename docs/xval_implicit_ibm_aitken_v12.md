# XVAL v12 — implicit IBM + Aitken: from a scalar law to a spectral (multi-DOF) law

*Mission m21768, TensorFEM. Anchor: `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`
(joconnor22/LIFE implicit immersed boundary + Aitken Δ² relaxation; acceptance
< 15 % against an analytic / literature reference).*
*Previous: `docs/xval_implicit_ibm_aitken_v11.md` (closed-form cost and stability
law for the **scalar** interface map); `v10` (coupled frequency, all resolutions
inside 15 %); `v8`/`v9` (added-mass excess surviving refinement).*

## 1. What was still open, and why it mattered

v11 replaced the anchor's last qualitative row with an exact law for a **single**
heave degree of freedom:

    omega* = 1/(1+R),  q = |1 - omega(1+R)|,  n(tol) = 1 + ceil(ln(tol/|r1|)/ln q),

so a scalar interface reaches its fixed point in **three sub-iterations for every
added-mass gain `R`**. That result is real, but every implicit-IBM coupling in
LIFE — and every six-DOF body in it — relaxes a *vector* residual with **one**
`omega`. Nothing in the repo said what the law becomes when the interface map has
more than one eigen-direction, and `docs/xval_implicit_ibm_aitken_v11.md` §6.1
flagged the gap in its own caveats ("`g` does not change here is a property of
this configuration, not a theorem").

This round answers that question with closed forms written down before
measurement, and it *corrects* the natural generalisation: the cost of one global
factor is driven by how the residual is **mixed** between eigen-directions, not by
the gain spread `kappa`.

## 2. Method

### 2.1 The identity behind the rule

For an affine interface `G(y) = A y + b` with residual `r_k = G(y_k) - y_k`, the
relaxed iterate is `y_{k+1} = y_k + omega_k r_k`, hence

    r_{k+1} = (I - omega_k M) r_k,        M := I - A   (interface operator).

Substituting `r_k - r_{k-1} = -omega_{k-1} M r_{k-1}` into the shipped Aitken
recursion (`src/tensorfem/ibm_aitken_implicit.py::AitkenRelaxation.omega_for`)

    omega_k = -omega_{k-1} (r_{k-1} · (r_k - r_{k-1})) / |r_k - r_{k-1}|^2

gives, identically,

    omega_k = (r_{k-1} · M r_{k-1}) / |M r_{k-1}|^2  —  Rayleigh quotient at r_{k-1},

i.e. the exact line search that minimises `|(I - omega M) r_{k-1}|`, evaluated at
the **previous** residual while the factor is applied to the **current** one. So
the shipped rule is a *lagged* minimal-residual (steepest-descent) step;
equivalently Anderson acceleration with memory one.

Consequences:

| case | statement | reference |
|---|---|---|
| `n = 1`, or residual inside one eigenvector | `omega = 1/mu`, `r_{k+1} = 0` — finite termination, `mu = 1 + R` | v11 law, recovered exactly |
| `n > 1`, mixed residual | rank-one line search cannot zero the residual; the iteration is linear with the classical bound `|r_{k+1}|/|r_k| <= (kappa-1)/(kappa+1)`, `kappa = cond(M)` | Saad, *Iterative Methods*, steepest-descent bound |
| one global factor on a body whose DOFs have different gains | the worst gain sets the pace; per-DOF-group (block) relaxation returns three sub-iterations per group | this round |

Closed forms live in the new module `src/tensorfem/ibm_aitken_spectral.py`
(`rayleigh_step`, `minres_relaxation(variant="lagged"|"aligned")`, `rate_bound`,
`stability_edge`, `block_minres`) together with the analytic reference added-mass
matrix of the ellipse (`ellipse_added_mass`: Lamb 1932 art. 105 / Newman 1977
§4.6 — heave `rho pi a^2`, surge `rho pi b^2`, pitch `rho pi (a^2-b^2)^2/8`).

### 2.2 Measurement without touching the solver

* **Affine 2-DOF model** — the analytic ellipse `M = I + Ms^{-1} Ma` is iterated by
  *the production class* (`AitkenRelaxation.omega_for`); no factor is taken from a
  closed form. Predictions come from my own recurrence (`minres_relaxation`), so
  model and measurement are independent implementations.
* **Shipped coupled solvers** — residual-norm sequences of
  `ibm_aitken_added_mass.IBMImplicitFSI` (plate, 1 DOF) and
  `ibm_aitken_implicit.ImplicitIBMFSI` (beam, 12 free DOF) are recovered through the
  **public `max_sub_iter` argument**: one run per truncation length, the first `k`
  sub-iterations of a step are identical for every `k`, so no source
  instrumentation and no reimplementation of the loop is needed.

## 3. Results

Driver: `benchmarks/xval_aitken_spectral_law.py` (single core, ≈ 40 s).
Artifacts: `benchmarks/xval_aitken_spectral_law_artifacts/xval_aitken_spectral_law.json`
+ 3 PNGs. Regression tests: `tests/test_aitken_spectral_law.py` (23 tests, 6 s).

### 3.1 The identity is exact (16 random affine maps, n = 1…4)

| quantity | result |
|---|---|
| worst residual-sequence rel. error, production loop vs lagged model | **2.8e-11** |
| worst relative error of the factor re-derived from the residual pair alone | **4.8e-16** |
| relaxation count agreement, model vs production | 16/16 |

### 3.2 The v11 law is the `n = 1` case

| `R` | `omega*` closed form | measured (production) | rel. err | sub-iterations |
|---|---|---|---|---|
| 1 | 0.5 | 0.5 | 0 | 3 |
| 2 | 0.333333 | 0.333333 | 0 | 3 |
| 4 | 0.2 | 0.2 | 0 | 3 |
| 12 | 0.0769231 | 0.0769231 | 0 | 3 |
| 30 | 0.0322581 | 0.0322581 | 0 | 3 |

### 3.3 A mixed residual costs 6–10 sub-iterations; a pure one costs 3

2-DOF ellipse (analytic spectrum `mu_i = 1 + R_i`), residual `(1,1)/sqrt2`
(mixed) unless stated; measured through the production class:

| case | `kappa = mu_max/mu_min` | global factor (measured) | lagged model | per-DOF block | `omega = 1` stable? |
|---|---|---|---|---|---|
| thin ellipse a=3, b=0.3, `rho_s`=1 | 1.88 | **11** | 11 | **3** | no (`mu_max`=11 > 2) |
| ellipse a=2, b=0.5, `rho_s`=7.8 | 1.25 | **8** | 8 | **3** | no |
| ellipse a=2, b=1, `rho_s`=0.5 | 2.63 | **9** | 9 | **3** | no |
| circle a=b=1, `rho_s`=0.05 | 21.0 | 4 | 4 | 3 | no |

Residual-mixing sweep (2-DOF, `kappa` ∈ {1.5, 4, 30}, share of `|r_0|^2` in the
slow eigendirection from 0 to 1, 21 points): **pure directions → 3
sub-iterations; any genuine mixture → 6–10**, and the model reproduces every
measured count (gap 0 in 21/21 points; gap 0 in the 4 ellipse cases and in the
9-point `kappa` sweep as well).

Fixed-`omega` audit: the measured asymptotic ratio equals the spectral radius
`max_i |1 - omega mu_i|` to **1.7e-3**, so the stability edge is
`omega < 2/mu_max` (the `omega = 1` case is the Causin–Gerbeau–Nobile threshold
`mu_max < 2`, i.e. `R < 1`).

### 3.4 Shipped solvers

| solver | DOF | residual sequence (measured) | sub-iterations at the tolerance | sequence/model agrees |
|---|---|---|---|---|
| plate `IBMImplicitFSI` (uniform transverse stream 0.02) | 1 | `[2.78e-4, 1.55e-4, 0, …]` — exact zero at the 3rd | **3** | yes |
| beam `ImplicitIBMFSI`, 5 markers (spacing 0.25), probe step 7 | 12 free | `[2.04e-3, 0, 0, …]` — exact zero at the 2nd | **2** | yes |

The plate row is the v11 law measured end-to-end in the shipped coupled solver.
The beam row is a **negative** result: the interface residual vanishes after a
single relaxation with `omega` pinned at the clip, i.e. the interface gain is
`mu ≈ 1` (`R < 1e-4`) — that configuration is not in the added-mass regime at all,
so no relaxation claim can be drawn from it. It cannot be pushed there either: at
marker spacing 0.0625 (17 markers) the same driver aborts at step 3 with
`ValueError: a marker stencil left the lattice domain`, for `dt` = 1e-4, 1e-3 and
2e-3 alike. **The multi-DOF law therefore has no real-solver data point —
`WITHHELD`** — and the ellipse rows above are the only quantitative backing.

## 4. Error table against the acceptance criterion

| quantity | reference | got | error | < 15 %? |
|---|---|---|---|---|
| Aitken factor = Rayleigh quotient at `r_{k-1}` | closed form (independent implementation) | measured sequence | 4.8e-16 | **yes** |
| scalar law `omega* = 1/(1+R)`, `R` = 1…30 | v11 closed form | production class | 0 | **yes** |
| sub-iteration count, 2-DOF ellipse (4 cases) | lagged model | production class | 0 (integer exact) | **yes** |
| sub-iteration count, mixing sweep (21 points) | lagged model | production class | 0 (integer exact) | **yes** |
| asymptotic rate vs `(kappa-1)/(kappa+1)` | steepest-descent bound | measured | ≤ 6e-2 relative, asymptotically inside | **yes** |
| fixed-`omega` decay ratio vs spectral radius | closed form | measured | 1.7e-3 | **yes** |
| beam (12 DOF) interface gain | added-mass regime (`mu > 1`) | `mu ≈ 1` (`R < 1e-4`) | **not in regime** | **WITHHELD** |

## 5. Caveats

1. **The cost driver is mixing, not `kappa`.** The intuitive hypothesis "cost
   grows with the gain spread" was written down and then **falsified**: in a
   `diag(mu, kappa*mu)` 2-DOF model the count is non-monotone in `kappa` (6, 8, 9,
   9, 9, 9, 8, 6, 6 for `kappa` = 1.05 … 300) because the fixed first factor
   `omega = 1` pushes the residual onto the dominant eigenvector, which the next
   Rayleigh step annihilates exactly. What does predict the count is the residual's
   share in the slow direction (§3.3).
2. **`n = 2` is not a worst case.** The measured counts stay in 4–11; higher
   dimension (6-DOF rigid bodies) can only be studied with the model, not measured
   here.
3. **The "aligned" variant does not help systematically.** Replacing `r_{k-1}` by
   `r_k` in the numerator (obtainable from the same pair, giving the line search at
   the current residual) measured 12 vs 11, 7 vs 8, 10 vs 9, 3 vs 4 — a wash. It is
   kept behind `variant="aligned"` and measured, not recommended.
4. **Model scope.** The ellipse operator is the added-mass exchange `Ms, Ma`
   without structural stiffness or fluid memory; the law concerns the coupling
   algorithm, and the fluid is frozen inside a time step, as `IBMImplicitFSI.step`
   is written.
5. **Still not LIFE itself.** No paired LIFE run; every reference here is closed
   form or the analytic ellipse added-mass matrix. A paired run remains the one
   outstanding item of this anchor.
6. **Not a defect report on the port.** The lagged numerator is the classical
   Irons–Tuck / LIFE convention and is exact for `n = 1`; this round quantifies what
   it costs for `n > 1` rather than claiming a bug.

## 6. Reproduction

```bash
PYTHONPATH=src python benchmarks/xval_aitken_spectral_law.py     # ~40 s, 1 core
PYTHONPATH=src python -m pytest tests/test_aitken_spectral_law.py -q
```

## 7. Files

| file | status |
|---|---|
| `src/tensorfem/ibm_aitken_spectral.py` | **new**: Rayleigh/lagged law, analytic ellipse added mass, block relaxation, rate bound, stability edge |
| `benchmarks/xval_aitken_spectral_law.py` | **new**: identity / scalar / ellipse / real-solver stages + 3 figures |
| `benchmarks/xval_aitken_spectral_law_artifacts/` | artifacts JSON + `xval_aitken_v12_ellipse.png`, `_cost.png`, `_real.png` |
| `tests/test_aitken_spectral_law.py` | **new**: 23 regression tests |
| `docs/xval_implicit_ibm_aitken_v12.md` | this report |