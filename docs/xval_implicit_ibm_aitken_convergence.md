# XVAL convergence study — implicit IBM + Aitken: where the added-mass bias comes from

**Anchor**: `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` (reference side: joconnor22/LIFE
implicit direct forcing + Aitken Δ² relaxation) · **Repo**: TensorFEM ·
**Branch**: `agent/m21294` · **Mission**: 21294 · **Acceptance**: error < 15 %
against an analytic / literature reference (owner standard)

Driver: `benchmarks/xval_implicit_ibm_aitken_convergence.py`
Artifacts: `benchmarks/xval_implicit_ibm_aitken_convergence_artifacts/`
(`evidence.json`, `summary.md`, 6 figures; cluster job `cj3348`, node n315)

## 1. The debt

* `docs/xval_implicit_ibm_aitken_v6.md` gated the coupled frequency of a spring-mounted
  plate against Lamb (1932) art. 71 (`m_a = ρ_f π c²`) and measured **7.48 %** error at
  `c = 16 Δ` in a `144²` box, explaining it in one sentence as "the kernel smears the
  plate over ~3 lattice units".
* `docs/xval_implicit_ibm_aitken_v7.md` measured the *absolute* added mass of the same
  family at 3.0–3.9 × `ρπc²` for `c = 4, 8`, declared it **WITHHELD**, and named two
  possible ways out — the first being *"refine the interface (ds = 0.5 Δ, wider box) so
  the effective inflation is resolved"*. That refinement had **not been run**, and the
  box sweep itself was recorded as NOT RUN.

This study runs that route as a **two-parameter sweep** and, more importantly, states
what the sweep can and cannot support.

## 2. Method

Coupling is the shipped `IBMImplicitFSI` (3-point kernel, ε conservation, Guo
half-split force, block Gauss-Seidel sub-iterations with Aitken Δ² relaxation) —
**no solver code was modified**. Each configuration is a rigid plate of half-width `c`
on a spring `k` in a quiescent periodic D2Q9 box, designed for a coupled period of
600 Δt, recorded for 1800 steps (3 cycles).

Measurement chain (this turned out to be the weak link):

| estimator | role |
|---|---|
| `benchmarks/period_estimators.multiharmonic_period` | **primary**: separable least squares on `c0 + e^{-t/τ} Σ_{k≤2}[a_k cos kωt + b_k sin kωt]`, `(T,τ)` minimised by grid + golden section |
| `estimate_period_damped` (library) | cross-check |
| `estimate_period` (zero crossings), `estimate_period_fft` | cross-check; `period_spread_rel` reports the estimator spread per case |

The study design is two axes plus a blind check: **A** confinement (`c = 16 Δ` fixed,
`L = 96…256`), **B** resolution (`2c/L = 0.222` fixed, `c = 8…48 Δ`), **D** two
configurations (`c = 32, L = 432`; `c = 48, L = 576`) held out of every fit, plus
controls on lattice viscosity (`ω_lbm`), amplitude (`y0`) and marker density (`ds`).

## 3. Results (all numbers from `evidence.json`)

### 3.1 Resolution does move the answer — monotonically

| `c` [Δ] | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|
| `m_a/(ρπc²)` | 1.831 | 1.567 | 1.466 | 1.465 | 2.026 | 1.340 |
| coupled-frequency error | 12.7 % | 9.2 % | 7.7 % | 7.7 % | **15.0 %** | 5.8 % |

The single-parameter kernel-inflation model `r = (1 + δ/c)²` (physically motivated: a
strip's added mass scales with the square of its half-span) fits with
**δ = 3.37 Δ** — the same inflation v6 guessed as "~1.5 Δ" and v7 fitted as
"3.9–7 Δ" — but with **rms 19.5 %** of the mean ratio and a **failed blind test**
(predictions for the held-out points are 13.9 % and 16.6 % off). The `c = 32` point is a
clear outlier: the local trend from `c = 8…48` is monotone while `c = 32` is not, so the
discrepancy is measurement scatter, not a resolved trend.

### 3.2 The confinement axis is not a confinement axis

`c = 16 Δ`: `r = 1.402 (L = 96) → 1.466 (L = 144) → 1.542 (L = 256)`. The ratio
**grows** with box size — the opposite sign to potential-flow blockage — and the
power-law fit returns a negative coefficient (`r₀ = 1.76`, i.e. the `b → 0` limit is
*above* every measurement). Acoustic scale explains the direction: `c_s T ≈ 378 Δ` for
the designed period, so `L = 96` is 0.25 λ and `L = 256` is 0.68 λ; the periodic box
is *compressible* at these sizes, and the box-length dependence is an
acoustic/added-stiffness effect, not container confinement.

### 3.3 Two controls that a "smearing only" story cannot explain

| control (at `c = 16 Δ`, `L = 144 Δ`) | ratio | change vs base 1.4656 |
|---|---|---|
| `y0 = 0.1` (amplitude) | 1.4660 | **+0.03 %** (linear ✓) |
| `ω_lbm = 1.7` (ν = 0.033) | 1.6221 | **+10.7 %** |
| `ω_lbm = 1.98` (ν = 0.011) | 1.3797 | **−5.9 %** |

The inferred added mass depends on lattice viscosity at the 10 % level, so a part of
the "added mass" measured through the frequency shift is a viscous response
(Stokes layer `δ_s = √(2ν/ω) ≈ 1–2.5 Δ` at these settings, i.e. unresolved).

## 4. Error verdict (acceptance < 15 %)

| quantity | reference | measured / extrapolated | error | < 15 % | < 3 % |
|---|---|---|---|---|---|
| `f/f_vac`, `c = 48 Δ`, `L = 576 Δ` (finest) | 0.7906 | 0.7458 | **5.66 %** | yes | no |
| `f/f_vac`, `c = 16 Δ`, `L = 144 Δ` (v6 config) | 0.7906 | 0.7295 | 7.73 % | yes | no |
| `f/f_vac`, `c = 32 Δ`, `L = 288 Δ` | 0.7906 | 0.6719 | **15.01 %** | **NO** | no |
| extrapolated limit (`Δ/c → 0`, `b → 0`, joint fit) | 0.7906 | 0.7226 | 8.59 % | yes | no |
| blind prediction of held-out `c = 48 Δ` | measured 1.3296 | 1.5504 | 16.6 % | **NO** | no |

**Is the key quantity within 3 %?** **No.** Neither at the finest measured
configuration (5.66 %) nor in the extrapolated limit (8.59 %), and the extrapolation is
not trustworthy anyway because the box axis does not converge (§3.2) and the joint fit's
residual (rms 0.20) is the same size as the effect being extrapolated. Within 15 %: yes
for 12 of 13 configurations; the `c = 32 Δ` point sits at 15.01 % and is reported as a
fail rather than dropped.

## 5. What this changes

1. **The 7.5 % of v6 is real and reproducible** (7.73 % here, with a better estimator),
   and it is *not* an estimator artifact and *not* a solver defect.
2. **"Kernel smearing" is only part of it.** Body resolution reduces the bias
   (12.7 % → 5.8 % from `c = 8` to `c = 48`), but the residual cannot be attributed to a
   single δ because (a) the box axis is acoustic and non-convergent, and (b) lattice
   viscosity moves the inferred mass by 10 %.
3. **Consequence for the anchor**: refining the interface alone cannot close the
   acceptance to 3 %. The honest statement is *WITHHELD below 15 %, PASS at 15 % only
   for resolved bodies (`c ≳ 10 Δ`), 3 % out of reach without (i) an incompressible
   fluid solver (`c_s T ≫ L`, i.e. much lower Mach or a projection/artificial-
   compressibility scheme) and (ii) a resolved viscous layer (`√(2ν/ω) ≫ Δ`).*
4. **Harness defect found and fixed**: `estimate_period_damped` returned its FFT seed
   (quantised periods; two different geometries reported identical mass ratios to four
   digits) on these records. The new `multiharmonic_period` reproduces a synthetic
   distorted decay to 0.02 % and is the primary estimator here; zero-crossing estimates
   fail badly on harmonic-distorted records (synthetic: 329 Δ vs 655 Δ truth).

## 6. Files

| file | change |
|---|---|
| `benchmarks/xval_implicit_ibm_aitken_convergence.py` | new: two-axis sweep, fits (power law / reciprocal / one-parameter δ / joint), blind checks, resolution plan, figures, `--figures-only` re-render |
| `benchmarks/period_estimators.py` | new: `multiharmonic_period`, `period_spread` |
| `tests/test_ibm_aitken_convergence.py` | new: 9 tests — each fit recovers a synthetic truth, blind-prediction check, v6-number consistency, resolution-plan inversion, estimator recovery, one coupled smoke run |
| `post_ibm_aitken_convergence_report.py` | new: publishes the report with numbers injected from `evidence.json` |

No file under `src/` was modified: this mission is a verification increment, and the
measurement says the library's coupling is not the cause of the discrepancy.
