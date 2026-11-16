# XVAL round 5 — attribution of the round-4 coupled-frequency deficit

Reopens PR #1120 (round 4: `xval_aitken_strong_coupling.py`). Round 4's honest verdict was `max |f - f_ref| = 23.1 %` against `f_ref = sqrt(k/(m_s+m_a))/2pi` with Lamb's `m_a = rho pi c^2`. This round asks *what carries that deficit* and predeclares the thresholds below before the runs.

## Predeclared thresholds

* `T1_spread_pct` = 2.0
* `T1_mode_pct` = 0.5
* `T2_window_pct` = 1.0
* `T3_holdout_pct` = 3.0
* `T4_model_pct` = 2.0
* `min_cycles` = 2.5
* `estimator_agreement_pct` = 1.5

## S — fresh coupled runs (constant blockage 2c/H = 0.208)

| c | box | m_a/m_s | n_steps | cycles | T (damped fit) | T (zero crossing) | m_eff | m_eff/m_a | vs Lamb | sub-it |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 96 | 6.0 | 1500 | 3.93 | 381.023 | 378.667 | 538.871 | 1.715 | -21.26% | 1.60 |
| 12 | 115 | 6.0 | 1500 | 4.03 | 372.364 | 370.000 | 737.715 | 1.631 | -19.43% | 1.84 |
| 16 | 154 | 6.0 | 1500 | 3.89 | 384.951 | 386.000 | 1410.874 | 1.754 | -22.07% | 1.59 |
| 20 | 192 | 6.0 | 1500 | 3.59 | 417.380 | 404.000 | 2449.307 | 1.949 | -25.74% | 1.01 |

## T1 — m_eff over the round-4 ratio sweep (recomputed from the stored records)

m_eff mean **535.66**, std 3.71, spread **1.99 %** (threshold 2.0 %), m_eff/m_a = **1.705**, aitken vs fixed-omega=0.5 max difference **0.000 %** (threshold 0.5 %) → **PASS**.

## T2 — window independence

| n_steps | cycles | T (fit) | T (zero crossing) |
|---|---|---|---|
| 1500 | 3.93 | 381.023 | 378.667 |
| 1100 | 2.88 | 381.023 | 378.500 |

frequency difference long vs short record: **+0.000 %** (threshold 1.0 %); fit vs zero crossing: **+0.622 %** → **PASS**.

## T3 — held-out frequency prediction from the measured m_eff

m_eff from the `m_a/m_s = 1` case alone: **534.20**.

| m_a/m_s | mode | omega | f predicted | f measured | residual |
|---|---|---|---|---|---|
| 3.0 | aitken | 1.0 | 2.698974e-03 | 2.685547e-03 | -0.50% |
| 3.0 | fixed | 1.0 | 2.698974e-03 | 2.685547e-03 | -0.50% |
| 3.0 | fixed | 0.5 | 2.698974e-03 | 2.685547e-03 | -0.50% |
| 6.0 | aitken | 1.0 | 2.634934e-03 | 2.624512e-03 | -0.40% |
| 6.0 | fixed | 1.0 | 2.634934e-03 | 2.624512e-03 | -0.40% |
| 6.0 | fixed | 0.5 | 2.634934e-03 | 2.624512e-03 | -0.40% |
| 8.0 | aitken | 1.0 | 2.616817e-03 | 2.624512e-03 | +0.29% |
| 8.0 | fixed | 0.5 | 2.616817e-03 | 2.624512e-03 | +0.29% |
| 10.0 | aitken | 1.0 | 2.605481e-03 | 2.597734e-03 | -0.30% |
| 10.0 | fixed | 0.5 | 2.605481e-03 | 2.597734e-03 | -0.30% |
| 12.0 | aitken | 1.0 | 2.597719e-03 | 2.597734e-03 | +0.00% |
| 12.0 | fixed | 0.5 | 2.597719e-03 | 2.597734e-03 | +0.00% |
| 16.0 | aitken | 1.0 | 2.587776e-03 | 2.597734e-03 | +0.38% |
| 16.0 | fixed | 0.5 | 2.587776e-03 | 2.597734e-03 | +0.38% |

worst |residual| **0.50 %** (threshold 3.0 %) → **PASS**.

## T4 — geometry of the bias

| c | 2c/H | m_eff/m_a |
|---|---|---|
| 10 | 0.208 | 1.7153 |
| 12 | 0.209 | 1.6307 |
| 16 | 0.208 | 1.7543 |

| model | parameter (fit on c=10,16) | predicted at c=20 | measured at c=20 | held-out residual | max fit residual (all three) |
|---|---|---|---|---|---|
| scale | alpha=1.2877 | 424.51 | 449.10 | -5.48% | +5.70% |
| smear | delta=3.227 | 369.69 | 449.10 | -17.68% | +17.18% |

winner: **None** — not separable over this c-range.

excluded from the geometry fit (not measurements):

* c=20: fit invalid / estimators disagree: not a measurement (rms/amp 0.335, estimator gap -3.206 %)

## Verdict

* `T1_m_eff_ratio_and_mode_independent` = True
* `T2_window_and_estimator_independent` = True
* `T3_heldout_frequency_reproduced` = True
* `T4_winner` = None
* `T4_interpretation` = not separable over this c-range
* `m_eff_over_lamb_round4_sweep` = 1.7050599968069549
* `m_eff_over_lamb_geometry_sweep` = [1.71527972277307, 1.630708784817876, 1.754277659843793]
* `fresh_worst_error_vs_lamb_pct` = 22.06796700419563
* `fresh_cases_within_15pct_vs_lamb` = False
* `explanation_upheld` = C: the deficit is the diffuse-interface added mass

## What the round-4 number was

Round 4 compared `f` to `f_ref = sqrt(k/(m_s+m_a))/2pi` and reported `max |error| = 23.10 %`, i.e. FAIL against the owner's 15 % acceptance. Recomputed here from the *same stored records* with the window-safe estimator the deficit is 19.93 %: the record-length path of round 4 (`--quick`, 400 steps, ~1.3 cycles) shows `T = nan` in its own artifact and its FFT fallback is a window artefact, not a frequency.

## Interpretation limits

* The attribution is to the **regularised interface**, not to a specific line of the kernel: T4 separates a constant scale error (C1) from a fixed smear length (C2) only over the swept half-width range; a `delta` that itself grows with `c` would mimic either.
* All runs are 2-D, at `rho_f = 1`, `dt = dx = 1`, one plate per periodic box, constant blockage `2c/H = 0.208`; nothing here bounds 3-D or free-surface behaviour.
* `m_eff` is an *inertia* statement: it says the coupled solver integrates `(m_s + m_eff) y'' + k y = 0` to the fitted accuracy. It does not by itself prove the interface force is correct -- the `m_eff` bias is exactly what a correct operator on a smoothed body geometry would also produce.

## Reproduction

```
python benchmarks/xval_aitken_strong_attribution.py          # full
python benchmarks/xval_aitken_strong_attribution.py --quick  # 2 c-values
python -m pytest tests/test_aitken_strong_attribution.py -q
```

Runtime: 2 s.
