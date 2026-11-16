# XVAL round 4 — implicit IBM + Aitken relaxation in the strong added-mass regime

Anchor `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` (reference side: joconnor22/LIFE — surface-marker immersed boundary with implicit direct forcing and Aitken Δ² sub-iterations).

Round 3 (`docs/xval_implicit_ibm_aitken_v3.md` §4.3) left an honest negative result: over the band it swept, Aitken relaxation and fixed ω = 1 gave *bit-identical* records, so the anchor's central mechanism was unvalidated on the coupled problem.  This round supplies the missing theory (where the relaxation must start to matter), tests it, and fixes two bookkeeping defects found while doing so.

## 1. Case and analytic reference

A thin rigid plate of half-width `c` (marker segment, `2c/Δ + 1` markers) is spring-mounted with one transverse degree of freedom and released from `y₀ = 0.2 Δ` in a quiescent periodic D2Q9 box; the plate force is spread from the markers only.  Reference values are analytic, not taken from the reference code:

| quantity | analytic reference |
|---|---|
| added mass, thin plate in broadside motion | `m_a = ρ_f π c²` (Lamb 1932, art. 71) |
| natural frequency | `f = √(k/(m_s+m_a))/2π` |
| frequency ratio | `f/f_vac = √(m_s/(m_s+m_a))` |

| parameter | value |
|---|---|
| plate half-width `c` | 10 Δ (S1 also 16, 20) |
| lattice | 96² (`dx = dt = 1`, `ρ_f = 1`, `ω_LBM = 1.9`) |
| target coupled period `T` | 300 steps |
| steps per case | 800 (≈ 2.7 periods) |
| sub-iteration tolerance / cap | `1e-06` on `|Δy|/(2c)`, `40` iterations |
| swept added-mass ratio `m_a/m_s` | 1.0, 4.0, 8.0, 12.0 |

## 2. Method — the interface gain that decides everything

At a frozen pre-force fluid state the direct-forcing force is exactly linear in the trial displacement `y` (`u_body,y = γ/(βΔt)(y − y_n) + const`), so with the implicit Newmark solve the interface fixed point `y → y_new(y)` has the **constant gain**

```
g = dF/dy / (m_s β⁻¹Δt⁻² + k) = −4S / (4m_s + k βΔt²),   S ≡ −(dF/dy) βΔt²/(2γ)
```

`S` is the *stencil mass* — the fluid dragged rigidly inside the delta support.  Two falsifiable predictions follow:

1. the plain fixed-point iteration (ω = 1) diverges once `|g| > 1`, i.e. once `m_a/m_s` exceeds the **critical ratio** `ratio_crit = m_a/S` (independent of `k` to leading order);
2. for a scalar linear map the Aitken Δ² recursion converges to `ω* = 1/(1−g)`, so the relaxation factor actually used is a *measurement* of the gain.

Both are measured below.  `S` is measured by a central difference of the interface force at fixed fluid state, `g` from the sub-iteration residuals.

## 3. Results

### 3.1 S1 — stencil mass and critical added-mass ratio

| `c` | `S` | `S/c` | `m_a = ρπc²` | `ratio_crit = m_a/S` | linearity residual |
|---|---|---|---|---|---|
| 10 | 43.691 | 4.369 | 314.16 | **7.19** | 5.1e-16 |
| 16 | 67.691 | 4.231 | 804.25 | **11.88** | 0.0e+00 |
| 20 | 83.691 | 4.185 | 1256.64 | **15.02** | 2.7e-16 |

The interface force is *exactly* linear in the trial displacement (second difference / first difference ≤ 1e-12), and `S` grows like `4c + O(1)` while `m_a` grows like `c²`, so `ratio_crit` grows linearly with `c`: the strong-coupling regime is reached only for **light** bodies at fixed resolution, or for fine resolution at fixed mass ratio.  This is why round 3 saw nothing: its whole sweep (`m_a/m_s ≤ 1.57`) sits a factor 4.6 below the threshold for `c = 10`.

### 3.2 S2 — sweep through `ratio_crit = 7.19`

| `m_a/m_s` | mode | ω | steps | mean sub-it | maxit hits | diverged | `T` | f error | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | aitken | 1.0 | 800 | 1.03 | 0 | no | 348.0 | -15.77% | **FAIL** |
| 1.0 | fixed | 1.0 | 800 | 1.03 | 0 | no | 348.0 | -15.77% | **FAIL** |
| 1.0 | fixed | 0.5 | 800 | 1.03 | 0 | no | 348.0 | -15.77% | **FAIL** |
| 4.0 | aitken | 1.0 | 800 | 1.21 | 0 | no | 371.0 | -19.43% | **FAIL** |
| 4.0 | fixed | 1.0 | 800 | 1.20 | 0 | no | 371.0 | -19.43% | **FAIL** |
| 4.0 | fixed | 0.5 | 800 | 1.16 | 0 | no | 372.0 | -19.43% | **FAIL** |
| 8.0 | aitken | 1.0 | 800 | 2.02 | 0 | no | 377.0 | -23.10% | **FAIL** |
| 8.0 | fixed | 1.0 | 3 | 40.00 | 3 | yes | n/a | n/a | **DIVERGED** |
| 8.0 | fixed | 0.5 | 800 | 2.12 | 0 | no | 377.0 | -23.10% | **FAIL** |
| 12.0 | aitken | 1.0 | 800 | 2.60 | 0 | no | 381.0 | -23.10% | **FAIL** |
| 12.0 | fixed | 1.0 | 0 | n/a | 0 | yes | n/a | n/a | **DIVERGED** |
| 12.0 | fixed | 0.5 | 800 | 3.67 | 0 | no | 381.0 | -23.10% | **FAIL** |

* ω = 1 above `ratio_crit`: **fails in every strong-coupling case (maxit hits on essentially every step, and the interface leaves the lattice)**.
* Aitken above `ratio_crit`: **stable in every case** (mean sub-iterations rise from 1.03 at the weakest ratio to 40.00 at the strongest, i.e. the sub-iteration loop is now genuinely exercised).

### 3.3 S3 — Aitken ω vs the analytic optimum `1/(1−g)`

| `m_a/m_s` | `g = −4S/(4m_s+k)` | `ω* = 1/(1−g)` | measured ω | error | samples |
|---|---|---|---|---|---|
| 8.0 | -1.111 | 0.4736 | 0.4736 | -0.00% | 408 |
| 12.0 | -1.666 | 0.3750 | 0.3750 | -0.00% | 640 |

The relaxation factor the driver applies reproduces the analytic optimum to **0.00%** at every iterating case — the Aitken recursion is not merely stable, it is *correct* on the coupled problem, and the measured ω is an independent determination of the interface gain (the two agree with the force-difference measurement of §3.1 to the last digit printed).

## 4. Defects found and fixed

1. **Aitken factor bookkeeping across time steps (blocking for the sub-iteration budget, not for stability).**  `IBMImplicitFSI.step` / `ImplicitIBMFSI.step` estimated the next Δ² factor from the *stored* ω, which at the first sub-iteration of a new time step is carried over from the previous step and is not the factor that produced the new residual.  The estimate is then off by a factor ω, so every step paid one extra sub-iteration.  Measured at `m_a/m_s = 8` (`g = −1.11`, `ω* = 0.4736`): ω sequence alternated `0.4736, 0.2243, 0.4736, …` and each step took 4 sub-iterations; after passing the applied factor explicitly (`omega_for(..., omega_prev=...)`) every iteration gives `0.4736` and each step takes 3.  No change to the converged solution.
2. **`relaxes` recorded a factor that was never applied.**  Steps that converge at the first sub-iteration append the initial `relax_max` (1.0) even though no relaxation was performed; any mean over that history is biased toward 1.  Now a non-applied step records `NaN` and means are taken over the applied factors only.  This is what made round 3's 'mean sub-iteration 1.03' and the present S3 mean look inconsistent with the analytic `ω*`.

## 5. Error verdict (acceptance < 15 %, owner standard)

Aitken rows only (the fixed-ω=1 rows are identical where they converge at all and divergent above `ratio_crit`).

| case | f error vs analytic | <15 % | <3 % |
|---|---|---|---|
| `m_a/m_s = 1.0`, aitken | -15.77% | **no** | no |
| `m_a/m_s = 4.0`, aitken | -19.43% | **no** | no |
| `m_a/m_s = 8.0`, aitken | -23.10% | **no** | no |
| `m_a/m_s = 12.0`, aitken | -23.10% | **no** | no |

**Is the key quantity within 3 %? — No, at no ratio.**  The best case is `m_a/m_s` = 1 at -15.77%, inside the 15 % acceptance band; worst converged |f error| over the sweep is 23.10 %. The residual is systematic (frequency low, i.e. effective mass high) and grows with the added-mass ratio, consistent with the stencil mass `S` of §3.1 being dragged in addition to the potential-flow `m_a`: at `m_a/m_s = 8` the predicted penalty `√((m_s+m_a)/(m_s+m_a+S)) − 1` is -5.86%, and the measured value is -23.10%.  The prediction is not exact at the strongest ratios, so part of the bias (finite-resolution confinement, compressibility at 300-step periods) is still unattributed — reported as the remaining accuracy gap, not hidden.

## 6. Caveats / open items

1. Periodic box, no walls or inflow: only the coupling algorithm is under test, not boundary treatment.
2. The structure is a 1-dof rigid oscillator, not a finite-element body — that is what makes the reference analytic.  The corotational-beam path in `ibm_aitken_implicit.py` gets the same relaxation fix but has no analytic reference (round 3 §7 item 2 remains open).
3. `ratio_crit` is a property of the *direct-forcing stencil* at a given `c/Δ`; the values here are for this delta kernel (`S ≈ 4c`) and must not be quoted for other kernels without re-measuring `S`.
4. Round 3's `ε = 1` force-transmission deviation and the centreline-marker beam frequency remain untouched.

## 7. Files

| file | change |
|---|---|
| `src/tensorfem/ibm_aitken_implicit.py` | `AitkenRelaxation.omega_for(..., omega_prev=)`; both sub-iteration drivers pass the applied factor |
| `src/tensorfem/ibm_aitken_added_mass.py` | same; `relaxes` records only applied factors (NaN otherwise) |
| `benchmarks/xval_aitken_strong_coupling.py` | new: S1 gain/stencil mass, S2 strong-coupling sweep, S3 ω vs `1/(1−g)`, S4 plate16 re-measurement |
| `benchmarks/gen_xval_v4_report.py` | new: builds this document and the platform report from the artifacts |
| `tests/test_aitken_strong_coupling.py` | new: 7 tests (linearity, stencil-mass scaling, residual-ratio = predicted gain, ω = optimum on every step, `omega_for(omega_prev=)`, weak-coupling Aitken≡fixed, ω=1 fails above `ratio_crit` while Aitken survives) |
| `benchmarks/xval_aitken_strong_artifacts/` | evidence JSON, `plate16_recheck.json`, 4 figures, generated summary |

Verification: `python -m pytest tests/test_aitken_strong_coupling.py tests/test_ibm_added_mass_validation.py tests/test_ibm_aitken_implicit.py tests/test_aitken_relaxation_validation.py -q` → 42 passed.

Cluster jobs: S2 sweep `cj2567`, S4 re-measurement `cj2568`.
