# XVAL v11 — implicit IBM + Aitken: from "3 it vs divergence" to a closed-form cost and stability law

*Mission m21657, TensorFEM. Anchor: `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`
(joconnor22/LIFE implicit immersed-boundary + Aitken Δ² relaxation; acceptance < 15 %
against an analytic / literature reference).*
*Previous: `docs/xval_implicit_ibm_aitken_v10.md` (coupled-frequency verdict, all
resolutions inside 15 %); `docs/xval_implicit_ibm_aitken_v9.md` (refinement limit);
`benchmarks/xval_implicit_ibm_aitken_convergence.py`.*

## 1. Case and question

Rounds v1–v10 of this anchor verified one quantity: the **coupled frequency** of a
sprung plate in a periodic D2Q9 box against Lamb (1932) art. 71,
`f/f_vac = √(m_s/(m_s+m_a))`, and closed it at 3.2 %–12.7 % error depending on
resolution. The *relaxation* half of the anchor title was never quantified. v6 states
it plainly (`docs/CASE-FEM-XVAL-IMPLICIT-IBM-AITKEN-v6.md` §4):

> | Aitken convergence vs fixed `ω = 1` | qualitative (3 it vs divergence) | n/a |

v10 repeats the same row in its acceptance table ("Aitken convergence | PASS | 0
max_iter hits"). "It converges and the other one does not" is not a verification of an
implementation against a reference — it is a demonstration that *something* is
happening. The acceptance criterion of this anchor (< 15 %) cannot be applied to a
statement with no reference value.

**This round supplies the reference value.** It is a closed form, derived from the map
the solver actually iterates, and it predicts — before any run — the relaxation factor,
the sub-iteration count, the residual ratio and the stability edge. Each prediction is
then measured both in a scalar model driven by the production class and in the real
coupled solver.

### Why a closed form exists at all

Within one time step the fluid state is frozen (`IBMImplicitFSI.step` reads
`macroscopic(include_force=False)` once and iterates the structure only). The IBM force
is exactly linear in the trial displacement at a frozen fluid state — already pinned by
`tests/test_aitken_strong_coupling.py::test_interface_force_is_linear_in_trial_displacement`
(second difference 1e-12) — and the Newmark solve is affine. Hence the interface map is
affine in one scalar degree of freedom:

    y_new = G(y) = g·y + b,      r_k = G(y_k) − y_k,      r_{k+1} = q·r_k
    q = 1 + ω(g − 1) = 1 − ω(1 + R),      R := −g

Three consequences, all exact rather than asymptotic:

| law | statement | reference |
|---|---|---|
| Aitken optimum | `ω* = −1/(g−1) = 1/(1+R)`, recovered from **any** base factor | Irons & Tuck (1969); Küttler & Wall (2008) |
| finite termination | `q = 0` ⇒ the fixed point is reached **exactly after two relaxations = three sub-iterations**, for every `R` | this round |
| stability edge | `|q| < 1` ⇔ `ω < ω_crit = 2/(1+R)` ⇔ `R_crit(ω) = 2/ω − 1`; the `ω = 1` case `R_crit = 1` | Causin, Gerbeau & Nobile (2005) for `ω = 1` |
| cost | `n(tol) = 1 + ⌈ln(tol/|r_1|) / ln|q|⌉` sub-iterations | this round |

The `ω ≥ 2` branch of `R_crit(ω) = 2/ω − 1` is worth stating: for `ω ≥ 2` the
negative-real gain of an added-mass exchange can never be stabilised — a prediction the
measurement confirms (both `R = 1e-6` and `R = 50` fail).

## 2. What was done

1. **Derivation** of the four laws above for the convention the production solver uses
   (`r_k = G(y_k) − y_k`, `omega_prev` = the factor that produced `r_k`).
2. **Scalar harness** (`benchmarks/xval_aitken_rate_law.py`, `part_a_scalar`) that drives
   the *production* `tensorfem.ibm_aitken_implicit.AitkenRelaxation` on the affine fixed
   point `y → g·y + b` over `R ∈ {0.1 … 12}` and `ω ∈ {0.5 … 2.5}`, and repeats the
   exercise on the repository's **second**, independently written Aitken class
   (`tensorfem.fsi_lbm_strong_coupling.AitkenRelaxation`) to check that the law belongs
   to the algorithm, not to one implementation.
3. **Real-solver harness** (`part_b_real`). The sub-iteration residual sequence is
   recovered **through the public API only**: `sub_tol = 0, max_sub_iter = m` makes
   `step()` report the m-th residual norm, and a fresh solver per `m` replays the same
   deterministic prefix. No source instrumentation, no private attribute is written.
   Configurations are those of the anchor: `m_a = ρπc²` (Lamb, `c = 8 Δ`), `m_s = m_a/mr`
   for `m_a/m_s ∈ {1, 2, 4, 12, 30}` (12 is the anchor's strong-coupling case),
   `k = (m_s+m_a)(2π/T)²` with `T = 400Δt`, box 64², `dt = Δ = 1`.
4. **Stability edge** measured by bisection on a *decay* criterion
   (`|r_120| ≤ 0.95|r_1|`) instead of on "reaches `sub_tol`", because the latter is
   resolution-limited near `|q| = 1`: any finite `max_sub_iter` biases the measured edge
   *inside* the true one (see §5, item 3 — this was a real defect in the first version of
   this study, caught and fixed).
5. **Regression tests** `tests/test_aitken_rate_law.py` (26 cases, 5 s) pin the laws so
   they cannot silently regress.

Tool chain: Python 3.11, numpy, matplotlib; single core; benchmark wall time 29 s.

## 3. Results

### 3.1 Scalar model, production class — Aitken optimum is exact, termination is finite

| R | ω* closed form | ω* measured | rel err | sub-iterations | \|x − x*\| |
|---|---|---|---|---|---|
| 0.1 | 0.909091 | 0.909091 | 0.0e+00 | 3 (pred 3) | 0 |
| 0.3 | 0.769231 | 0.769231 | 2.9e-16 | 3 (pred 3) | 0 |
| 0.6 | 0.625000 | 0.625000 | 0.0e+00 | 3 (pred 3) | 0 |
| 1 | 0.500000 | 0.500000 | 0.0e+00 | 3 (pred 3) | 0 |
| 2.5 | 0.285714 | 0.285714 | 0.0e+00 | 3 (pred 3) | 0 |
| 4 | 0.200000 | 0.200000 | 0.0e+00 | 3 (pred 3) | 0 |
| 12 | 0.076923 | 0.076923 | 0.0e+00 | 3 (pred 3) | 0 |

The second relaxation estimate equals `1/(1+R)` to ≤ 2.9e-16 and the iterate lands on
the fixed point to machine zero. This is the analytic content of v6's "3 it".

### 3.2 Fixed relaxation — geometric rate and integer cost law

Worst error on the rate `|q|`: **1.1e-15**; the cost/divergence law agrees on **35/35**
`(ω, R)` pairs. Excerpt:

| R | ω | q closed form | \|q\| measured | n measured | n predicted |
|---|---|---|---|---|---|
| 0.1 | 0.5 | +0.4500 | 0.4500 | 25 | 25 |
| 0.3 | 1.5 | −0.9500 | 0.9500 | 366 | 366 |
| 0.6 | 0.5 | +0.2000 | 0.2000 | 13 | 13 |
| 1 | 0.5 | +0.0000 | 0.0000 | 2 | 2 (one-shot) |
| 2.5 | 0.5 | −0.7500 | 0.7500 | 70 | 70 |
| 4 | 0.5 | −1.5000 | 1.5000 | 400 = maxit | ∞ (diverges) |
| 12 | 0.5 | −5.5000 | 5.5000 | 400 = maxit | ∞ (diverges) |

### 3.3 Critical ratio `R_crit(ω) = 2/ω − 1`

| ω | closed form | bisected | error |
|---|---|---|---|
| 0.5 | +3.000000 | +3.000000 | 0.0e+00 |
| 1.0 | +1.000000 | +1.000000 | 2.2e-16 |
| 1.5 | +0.333333 | +0.333333 | 1.1e-16 |
| 2.0 | 0.000 → no `R ≥ 0` converges | verified at `R = 1e-6` and `R = 50` | — |
| 2.5 | −0.200 → no `R ≥ 0` converges | verified at `R = 1e-6` and `R = 50` | — |

### 3.4 The two implementations agree

`fsi_lbm_strong_coupling.AitkenRelaxation` over the same `R` grid: worst `ω*` error
**2.9e-16**, termination `[3, 3, 3, 3, 3, 3, 3]`. The stale-base penalty predicted by the
`omega_prev` docstring (`ω_est = ω* · ω_stale/ω_applied`) is reproduced to **1.8e-16**.

### 3.5 Real coupled solver — gain, Aitken factor, cost

| m_a/m_s | m_s | R measured | gain spread over 6 sub-iterations | R after 5 steps | ω* closed form | ω* measured | rel err | sub-iterations |
|---|---|---|---|---|---|---|---|---|
| 1 | 201.0619 | 0.1775 | 2.3e-10 | 0.1775 | 0.849265 | 0.849265 | 1.7e-11 | 3 |
| 2 | 100.5310 | 0.3550 | 7.6e-11 | 0.3550 | 0.738032 | 0.738032 | 2.7e-12 | 3 |
| 4 | 50.2655 | 0.7098 | 2.7e-12 | 0.7098 | 0.584856 | 0.584856 | 4.4e-13 | 3 |
| **12** | 16.7552 | **2.1284** | 1.6e-14 | 2.1284 | 0.319650 | 0.319650 | **2.1e-15** | **3** |
| 30 | 6.7021 | 5.3152 | 6.9e-15 | 5.3151 | 0.158349 | 0.158349 | 4.2e-15 | 3 |

Three things are established at once: the gain `R = −g` is **constant across
sub-iterations** (spread ≤ 2.3e-10, which is what licenses the scalar law), it is
**unchanged after five coupled steps** (0.1775 → 0.1775; 2.1284 → 2.1284), and the
Aitken factor the solver applies equals `1/(1+R)` to ≤ 1.7e-11. The cost is
**three sub-iterations for every added-mass ratio from 1 to 30** — a factor of 30 in
`R` (0.18 → 5.32) with no change in cost.

The fixed-relaxation cost law is integer-exact on **50/50** `(R, ω, tol)` points over
`tol ∈ [1e-4, 1e-8]`, including all 12 predicted-divergence points. E.g. at the anchor
configuration (`R = 2.1284`, `ω = 0.5`, `q = −0.564`): measured `n = 1, 2, 6, 10, 14`
for `tol = 1e-4 … 1e-8`, closed form identical.

### 3.6 Stability edge in the real solver

| m_a/m_s | R measured | ω_crit closed form | measured | rel err | bias-corrected rel err |
|---|---|---|---|---|---|
| 1 | 0.1775 | 1.69853 | 1.69816 | 2.19e-04 | 4.83e-06 |
| 2 | 0.3550 | 1.47606 | 1.47575 | 2.14e-04 | 7.56e-07 |
| 4 | 0.7098 | 1.16971 | 1.16946 | 2.13e-04 | 9.54e-07 |
| 12 | 2.1284 | 0.63930 | 0.63916 | 2.15e-04 | 1.47e-06 |
| 30 | 5.3152 | 0.31670 | 0.31663 | 2.14e-04 | 6.09e-07 |

The uncorrected residual bias is **identical for all five configurations** (2.13–2.19e-4)
and equals the analysable resolution floor of the decay criterion:
`1 − (1+q_floor)/2 = 2.15e-4` with `q_floor = 0.95^{1/120} = 0.99957`. Correcting the two
measured numbers for that floor leaves 5e-7 … 1.5e-6. A shared bias of known origin is a
much stronger result than five small unrelated errors.

## 4. Figures

`benchmarks/xval_aitken_rate_law_artifacts/`

- `xval_aitken_v11_rate.png` — measured sub-iteration count vs `|q|` over the whole
  `(R, ω, tol)` grid collapses onto the closed-form `ln(tol)/ln|q|` line.
- `xval_aitken_v11_boundary.png` — measured stability edge at five added-mass ratios on
  top of `R = 2/ω − 1`.
- `xval_aitken_v11_cost.png` — cost per time step: Aitken flat at 3 for any `R`, fixed
  relaxation blowing up as `R → R_crit(ω)`.

## 5. Error judgement (owner standard: < 3 %, honest-fail otherwise)

| key quantity | reference | measured | error vs reference | < 3 % ? |
|---|---|---|---|---|
| Aitken factor `ω*`, scalar model | `1/(1+R)` | 7-point grid | ≤ 2.9e-16 | **yes** |
| Aitken factor `ω*`, real solver | `1/(1+R)` | `R` = 0.18 … 5.32 | ≤ 1.7e-11 | **yes** |
| sub-iteration count, Aitken | 3 (all `R`) | 3 (all 5 configs) | 0 | **yes** |
| residual ratio `|q|`, scalar | `|1−ω(1+R)|` | 35 points | ≤ 1.1e-15 | **yes** |
| sub-iteration count, fixed | `1+⌈ln(tol/\|r_1\|)/ln\|q\|⌉` | 50 points | 0 (integer-exact) | **yes** |
| critical ratio `R_crit(ω)`, scalar | `2/ω−1` | 3 positive branches | ≤ 2.2e-16 | **yes** |
| stability edge `ω_crit`, real solver | `2/(1+R)` | 5 configs | 2.1e-4 (raw), ≤ 1.5e-6 (bias-corrected) | **yes** |
| gain constancy (licenses the law) | state-independent | `R` at step 0 vs step 5 | ≤ 1e-14 relative | **yes** |

Every quantity in this round has a reference value, and all of them are inside 3 % by
orders of magnitude. **There is no honest-FAIL item in this round** — with two caveats
that belong in the record: the laws are exact *for the affine interface map*, and the
measured edge carries a bias whose origin is identified and quantified rather than
removed.

## 6. Caveats / limitations

1. **Scope of the law.** It applies to the *affine* interface map, i.e. the fluid frozen
   inside a time step, which is exactly how `IBMImplicitFSI.step` is written. Across time
   steps `g` changes; that it does not change *here* (5-step invariance) is a property of
   this configuration, not a theorem.
2. **`R` is an interface gain, not an added-mass ratio.** For the anchor configuration
   `R = 2.1284` while the analytic `m_a/m_s = 12`; the stencil smears the plate, and the
   discrete IBM stencil mass is what sets the gain. This is the same discrete-interface
   artefact documented in v8/v9 (`m_a` of the smeared plate = 1.18–2.79 × `ρπc²`), now
   seen from the relaxation side. The law is stated in `R = −g`, which is the quantity
   that actually governs the iteration.
3. **A measurement trap that was hit and fixed.** The first version of this study located
   the stability edge by asking "does the loop reach `sub_tol = 1e-6` within 80
   sub-iterations?" and reported edges 0.3 %–2.0 % *below* the closed form, with the error
   growing in `R`. The cause is not physics: near `|q| = 1` the required count diverges,
   so a finite `max_sub_iter` truncates *before* the true edge and the bias depends on the
   initial residual (1.5e-6 … 2.4e-5 across configurations). Switching to a decay
   criterion and accounting for its resolution floor removed the trend (§3.6). Reusable
   knowledge: *a divergence boundary cannot be located with a tolerance criterion.*
4. **Not LIFE itself.** The upstream executable still cannot be built in this image; every
   reference used here is closed form (Irons–Tuck / Küttler–Wall / Causin et al.), not a
   paired LIFE run. A paired run remains the one outstanding item of this anchor.
5. **2-D, periodic box, quiescent fluid.** No walls, inflow or turbulence; the law concerns
   the coupling algorithm only, not the flow solver.
6. **`ω ≥ 2` is unusable** for this problem class (`R ≥ 0` always), which also means the
   production clip `relax_max = 1.0` never truncates the optimum `1/(1+R) ∈ (0, 1]`. The
   second implementation clips at `omega_max = 2.0`, which likewise never binds.

## 7. Reproduction

```
python benchmarks/xval_aitken_rate_law.py          # 29 s, single core; tables + JSON + 3 PNGs
python -m pytest tests/test_aitken_rate_law.py -q  # 26 passed, 5 s
```

Artifacts: `benchmarks/xval_aitken_rate_law_artifacts/xval_aitken_rate_law.json`
(every number in §3, raw including the residual sequences).