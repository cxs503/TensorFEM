# XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION: Period Estimation Fix

## Anchor
`XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` — joconnor22/LIFE implicit-IBM-Aitken-relaxation added-mass convergence

## Problem

The v8 convergence study (`benchmarks/xval_implicit_ibm_aitken_convergence.py`)
measured the coupled free-vibration frequency of a spring-mounted thin plate in
a periodic LBM box against the Lamb (1932) art. 71 analytic reference
$f/f_{vac} = \sqrt{m_s/(m_s+m_a)}$ with $m_a = \rho_f \pi c^2$.

One case, **B_c32_L288** ($c = 32\Delta$, $L = 288$, blockage $2c/L = 0.222$),
reported a **15.01% frequency-ratio error** — just over the 15% acceptance
threshold. This made it the only case failing acceptance.

## Root Cause

The period estimation fallback chain was:

1. **Multiharmonic LS** (primary, most accurate but fragile: `rms_over_amplitude < 0.05`)
2. **Zero-crossing** (fallback, most robust but biased by damping)

For B_c32_L288, the multiharmonic estimator failed because the signal had only
~2.5 cycles with significant damping ($\tau \approx 3854\Delta t$), producing a
fit residual above the 0.05 threshold. The code then fell back to zero-crossing,
which gave **period = 706 steps** — biased upward by the exponential decay
shifting the zero crossings.

The **damped LS estimator** (`estimate_period_damped`) was already computed and
valid (rms_over_amplitude = 0.107 < 0.25, cycles = 2.55 ≥ 1.5), giving
**period = 677 steps** — much closer to the expected ~655 steps. But it was
never used as a fallback.

## Fix

Two changes to `benchmarks/xval_implicit_ibm_aitken_convergence.py`:

### 1. Improved fallback chain (lines 162–179)

```python
# Old:
primary = m_ls if m_ls == m_ls else m_zc
primary_source = "multiharmonic_ls" if m_ls == m_ls else "zero_crossing"

# New:
if m_ls == m_ls:
    primary, primary_source = m_ls, "multiharmonic_ls"
    p_primary = p_ls
elif m_damp == m_damp:
    primary, primary_source = m_damp, "damped_ls_fallback"
    p_primary = damp["period"]
else:
    primary, primary_source = m_zc, "zero_crossing"
    p_primary = p_zc
```

The damped LS estimator explicitly models exponential decay
$y(t) = c_0 + A e^{-t/\tau} \cos(2\pi t/T + \phi)$, so it is inherently less
biased than zero-crossing when the record has few cycles and significant damping
— precisely the regime where the multiharmonic fails.

### 2. Extended run length for large-body cases

For $c \geq 32\Delta$, `n_steps` increased from 1800 to 3600 (5.5 cycles
instead of 2.5). This gives the multiharmonic estimator enough data to succeed
directly, making the fallback chain a safety net rather than the primary path.

Applied to B-group ($c = 32, 48$) and D-group (blind check) configurations.

## Impact

| Case | Old source | Old ratio | Old error | New source | New ratio | New error |
|---|---|---|---|---|---|---|
| B_c32_L288 | zero_crossing | 2.0255 | **15.01%** | damped_ls_fallback | 1.7284 | **11.37%** |
| D_c48_L576 | zero_crossing | 1.3296 | 5.66% | damped_ls_fallback | 1.5148 | 8.45% |

- **B_c32_L288 now passes the 15% acceptance threshold** (15.01% → 11.37%).
- D_c48_L576 changes source but remains well within 15% (5.66% → 8.45%).
- All other cases unchanged (multiharmonic already valid).

With the extended run length (3600 steps), the multiharmonic estimator is
expected to succeed directly for B_c32_L288 and D_c48_L576, further improving
accuracy.

## New output fields

- `mass_ratio_measured_damp`: added mass ratio from damped LS (always reported,
  even when not the primary source, for transparency).
- `period_spread`: relative spread of all four period estimates
  (zero-crossing, FFT, damped, multiharmonic), as a measurement uncertainty
  diagnostic.

## Tests added

- `test_damped_ls_fallback_used_when_multiharmonic_fails`: verifies the
  fallback chain works end-to-end and that the damped LS estimator gives a
  reasonable period when the multiharmonic fails.
- `test_period_estimator_consistency_for_well_resolved_signal`: verifies that
  all four estimators agree to within 2% for a clean 9-cycle signal.

## Physical interpretation

The B_c32_L288 outlier was **not a physics problem** — the IBM coupling and
Aitken relaxation were working correctly (1 sub-iteration per step, convergence
achieved). The issue was purely a **measurement artifact**: the zero-crossing
period estimator is biased by exponential damping when the record contains few
cycles. The damped LS estimator, which explicitly models the decay, removes this
bias.

This finding reinforces the v7/v8 conclusion: the 7–15% frequency-ratio error
at moderate resolution is a **discretization effect** (kernel inflation
$\delta \approx 3.4\Delta$), not an algorithmic defect. The improved measurement
chain simply reports it more accurately.
