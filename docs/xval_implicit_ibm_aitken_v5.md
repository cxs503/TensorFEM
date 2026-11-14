# XVAL v5 — coupled added-mass frequency of the implicit IBM + Aitken solver

**Anchor**: `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` (joconnor22/LIFE implicit-IBM-Aitken recipe)
**Repo**: TensorFEM · **Branch**: `agent/m20525` · **Mission**: 20525
**Acceptance**: error < 15 % against the analytic / literature reference (owner precision standard)

---

## 1. Case, physics and the open item inherited from v4

A thin rigid plate of half-width `c` (marker segment along `x`, one transverse
DOF) is spring-mounted with stiffness `k` and released from `y0` inside a
quiescent periodic D2Q9 box; the plate is coupled to the fluid by **implicit
direct forcing with Aitken Δ² sub-iterations** (`tensorfem.ibm_aitken_added_mass`).
For a flat plate in 2-D potential flow the added mass per unit length for
broadside motion is exactly

```
m_a = rho_f * pi * c^2                       (Lamb 1932, art. 71)
(m_s + m_a) y'' + k y = 0
f    = sqrt(k / (m_s + m_a)) / 2pi
f/f_vac = sqrt(m_s / (m_s + m_a))
```

`docs/xval_implicit_ibm_aitken_v4.md` closed with this check **FAIL**:
`f_fsi/f_vac = 1.5811` vs analytic `0.7906` (error 100 %), attributed to
"`y0 = 0.2` interpreted as lattice units rather than physical units". That
attribution is wrong, and the check was never executable — see §5.

### Parameter table (shard `c10_H288`, the record this report archives)

| Symbol | Value | Note |
|---|---|---|
| `c` | 10 Δ | plate half-width, markers spaced 1 Δ |
| `H` (box) | 288 Δ × 288 Δ | periodic both directions |
| `m_a` (Lamb) | 314.159 | `rho_f π c²`, `rho_f = 1` |
| `m_s` | 523.599 | `m_a / 0.6` |
| `k` | 0.0918704 | set so the *analytic* period is `T = 600` |
| `T_analytic` | 600 | `2π sqrt((m_s+m_a)/k)` |
| `dt` | 1 | lattice time step |
| `n_steps` | 2400 | **4 analytic periods** (v4 used 300 = 0.5 period) |
| `omega_lbm` | 1.9 | D2Q9 BGK, `nu = 0.00877 Δ²/dt` |
| `sub_tol` / `max_sub_iter` | 1e-6 / 30 | interface fixed point |
| blockage `B = (2c+1)/H` | 0.0729 | |

## 2. Method

1. Run the coupled solver for 4 analytic periods (shard `c10_H288`, cluster job
   `cj2758` on the loong64 nodes; 306.9 s wall).
2. Estimate the period with **`estimate_period_damped`** (new in this mission):
   separable least squares on `y = c0 + A exp(-t/tau) cos(2pi t/T + phi)`, seeds
   from the FFT peak and `span/k`, coordinate descent on `(T, tau)`. It is
   *window-safe*: a record shorter than `min_cycles` periods is returned with
   `valid = False` instead of a silently meaningless number.
3. Recover `m_a_eff = k/(2πf)² - m_s` and compare `f` and `m_a_eff` with Lamb.
4. Reproduce the v4 failure mechanism synthetically and on the real record
   (`v5_window_bias.png`), and patch the v4 harness so an unusable record can
   no longer be reported as a physics failure (`INVALID_WINDOW` verdict).
5. Field figure: vorticity of the D2Q9 box at `t = 150` with the plate markers.

Tool chain: `numpy` + `matplotlib`; benchmark `benchmarks/xval_plate_added_mass_v5.py`,
figures `benchmarks/xval_plate_added_mass_v5_figures.py`, tests
`tests/test_plate_added_mass_period.py`.

## 3. Results

### 3.1 Archived shard `c10_H288` (evidence: `config_01_c10_H288.json`)

| Quantity | Analytic | Measured | Error |
|---|---|---|---|
| period `T` | 600.0 | 682.67 | +13.8 % |
| frequency `f` | 1.6667e-3 | 1.4649e-3 | **−12.1 %** |
| `f/f_vac` | 0.79057 | 0.69484 | −12.1 % |
| `m_a_eff` | 314.16 | 560.91 | **+78.5 %** |
| `m_a_eff` vs `ρπ(c+1.5Δ)²` | 415.48 | 560.91 | +35.0 % |
| effective half-width `c_eff` | 10.0 | 13.36 | +3.36 Δ |
| damping per cycle | — | 24.8 % | — |
| fit quality `rms/amplitude` | — | 0.0150 | — |
| record length | — | 3.51 periods | ≥ 2 ✓ |
| mean / max sub-iterations | — | 1.0 / 1 | converged, 0 max-it hits |

### 3.2 Window-sensitivity of the legacy estimators (same record, truncated)

Figure `v5_window_bias.png`: `T_est/T_ref` vs record length. For records shorter
than one period the legacy path returns the window length itself; at the v4
record length (300 = 0.5 `T`) the synthetic reproduction gives
`T_est = 315.1` (`f = 1/300` in the v4 evidence, i.e. the v4 number is the
window, not the physics). The window-safe fit is exact to < 1 % on the synthetic
decaying-offset signal with ≥ 2 periods and is flagged invalid below that.

### 3.3 Mass-ratio / relaxation cross-checks (session probes, not yet archived as shards)

| Case | window | `f` error vs Lamb | `m_a_eff` error |
|---|---|---|---|
| c=10, H=144, `m_a/m_s`=0.6 | 3.6 periods (zero-crossing, 3 extrema spacings) | −11.1 % | +67 % |
| c=10, H=160, `m_a/m_s`=0.6 | 12 cycles | −9.4 % | +57 % |
| c=20, H=160, `m_a/m_s`=2.51 | 10 cycles | **+11.9 %** | **−28 %** |

The bias is **not** monotone in the added-mass ratio and changes sign between
`m_a/m_s = 0.6` (+78 %) and `2.51` (−28 %). A pure kernel-smearing error
(`c → c + const`) cannot do that: it predicts a systematic positive shift for
every `c`. The frequency agreement (9–12 %) is therefore partly accidental:
`m_s` dominates the effective mass at ratio 0.6, so a 78 % error in `m_a` leaks
into `f` as only ~12 %.

## 4. Figures

| File | Content |
|---|---|
| `v5_field_vorticity.png` | `u_x` and vorticity of the D2Q9 box at t=150 with plate markers overlaid (field cloud) |
| `v5_history.png` | plate displacement history, 4 analytic periods |
| `v5_window_bias.png` | period estimate vs record length: legacy (collapses to window length) vs window-safe fit, v4 record length marked |
| `v5_added_mass.png` | `m_a_eff` vs `c` against Lamb and the smeared model (single archived shard) |

## 5. Caveats / limitations (adversarial notes)

1. **The v4 verdict was a harness defect, not a solver defect.** `n_steps*dt = 300`
   vs `T = 600`: the zero-crossing branch has < 2 up-crossings and the
   Hann-windowed FFT saturates at `1/(n dt) = 0.003333` — exactly the recorded
   `f_measured`. The "exactly 2×" coincidence is `T = 2 n_steps dt`. The v4
   harness is patched here: such a record now yields `verdict = INVALID_WINDOW`
   with `period_fit_valid = False`, and the misleading "unit-handling" explanation
   in v4 §3.4/§5.3 is superseded by §5 of this report.
2. **Frequency PASS is conditional.** `f` meets the < 15 % acceptance on every
   valid-window configuration measured so far, but the *attribute* it is supposed
   to validate (`m_a`) is 28–78 % off. Passing the frequency at `m_a/m_s = 0.6`
   is a weak statement about added mass and must not be quoted as an added-mass
   validation. The honest status is: **frequency-level PASS, mass-attribute-level FAIL**.
3. **Sign-changing bias is unexplained.** Candidate mechanisms: (i) the
   direct-forcing estimator at a zero-thickness plate acts on ~`3.4 Δ` of extra
   apparent half-width (smearing, always positive), (ii) the periodic images
   (positive, `B`-dependent), (iii) a *fluid added-stiffness / damping* term that
   enters the single-DOF frequency and biases `m_a_eff` in either direction
   depending on the ratio. Only (iii) can explain the sign change, and it is not
   separable from (i)+(ii) with a free-decay test alone: the clean discriminator
   is the **forced-oscillation** fit of `m_a` already implemented
   (`forced_oscillation`, `benchmarks/xval_aitken_added_mass.py`), which measures
   the force components `{y'', y', y, 1}` separately. That is the recommended next step.
4. **Damping is large** (24.8 %/cycle at `nu = 0.00877`), so a single-mode
   damped fit is adequate here but the identification is not noise-free; a
   lower-viscosity (`omega_lbm → 1.95`) repeat is advisable.
5. **Sweep incompleteness (honest).** The 7-shard box/`c` sweep (`cj2758`) had
   only shard 1 archived when this report was written (queue-bound: 30+ jobs in
   flight, 307 s per 288² shard); the box-`c` model-form fit and the
   `m_a/m_s = 1.0` Aitken-vs-fixed comparison are therefore **not reported as
   results**. `xval_plate_added_mass_v5.py --aggregate` produces them once the
   shards land; the code path is tested only through the single-shard smoke run.

## 6. Error verdict (acceptance < 15 %)

| Check | Reference | Measured | Error | < 15 % ? |
|---|---|---|---|---|
| `T` fit, c=10 H=288, 4 periods | 600.0 | 682.67 | 13.8 % | yes |
| `f`, c=10 H=288 | 1.6667e-3 | 1.4649e-3 | **12.1 %** | **yes** |
| `f`, c=10 H=144 (probe) | 1.6667e-3 | 1.4824e-3 | 11.1 % | yes |
| `f`, c=10 H=160 (probe) | 9.6613e-3 | 8.7510e-3 | 9.4 % | yes |
| `f`, c=20 H=160 (probe) | 6.5772e-3 | 7.3590e-3 | 11.9 % | yes |
| `m_a_eff`, c=10 H=288 | 314.16 | 560.91 | **78.5 %** | **no** |
| `m_a_eff`, c=10 H=160 (probe) | 314.16 | 492.31 | 56.7 % | no |
| `m_a_eff`, c=20 H=160 (probe) | 1256.6 | 903.2 | 28.1 % | no |
| estimator on synthetic 4-period signal | 600.0 | 595.8 | 0.7 % | yes |
| legacy estimator, 0.5-period record | 600.0 | 300.0 (= window) | 50 % | **invalid** (guard) |

**Answer to "is the key quantity within 3 %?":** no — the coupled frequency is
9–12 % off and the recovered added mass 28–78 % off. The 3 % band is met only by
the estimator itself (0.7 % on a synthetic reference) and by the operator-level
checks already archived in v4 (kernel partition of unity, `epsilon` conservation
to 2e-16).

## 7. Files changed

| File | Change |
|---|---|
| `src/tensorfem/ibm_aitken_added_mass.py` | `estimate_period_damped`, `period_window_ok` (window-safe period/damping estimator + guard) |
| `benchmarks/xval_plate_added_mass_v5.py` | new sweep benchmark (4-period windows, blocking-aware design, aggregate verdicts) |
| `benchmarks/xval_plate_added_mass_v5_figures.py` | figures for this report |
| `benchmarks/xval_implicit_ibm_aitken_v4.py` | window guard: `period_fit_valid`, `window_cycles`, `window_ok_2cyc`, `INVALID_WINDOW` verdict |
| `tests/test_plate_added_mass_period.py` | 4 fast invariants + 1 slow coupled-frequency acceptance test |
| `docs/xval_implicit_ibm_aitken_v5.md` | this report |

*Generated by TensorFEM XVAL pipeline, mission m20525 (cluster shard job `cj2758`).*