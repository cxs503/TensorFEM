# Acoustic radiation efficiency: the missing ½ — convention audit and fix

Status: **defect found and fixed** · module `tensorfem.acoustic_radiation`
(and its consumer `tensorfem.ice_noise_chain`) · tests
`tests/test_acoustic_radiation.py`, `tests/test_acoustic_radiation_low_freq_exact.py`
· verdict table below is machine-checked, no hand-copied numbers.

## 1. What was wrong

`sigma = P_rad / (rho c S <v^2>)` is only a well-defined quantity when *both*
`P_rad` and `<v^2>` are **time** averages. The module took `P_rad` from the
Rayleigh radiation resistance with the correct harmonic factor
(`P = (rho*omega/4pi) * int int v1 v2 sin(kR)/R dS1 dS2`, which is
`(1/2) Re[int p v* dS]` for **peak** amplitudes), but divided it by

    mode_mean_square_velocity(...) = V0^2 / 4        # pre-audit

which is only the *spatial* average of the squared mode shape
(`<sin^2> = 1/4`) and omits the harmonic time average `1/2`. The 1 m square
panel mode (1,1) therefore had to satisfy

    Rayleigh route:   sigma(f = 2 kHz) = 0.503694
    Leppington route: sigma(f = 2 kHz) = 1.000000        # k/k_mn = 8.2

i.e. the two independent routes in the same module disagreed by 49.7% and no
test covered the comparison. The same ½ was missing in the closed-form
low-frequency laws of this module and in
`ice_noise_chain.monopole_sigma_lowfreq`.

## 2. Which side was wrong — three independent anchors

1. **Baffled piston, exact.** For a uniformly vibrating baffled disk of peak
   velocity `V0` the exact efficiency is `sigma = 1 - J1(2ka)/ka -> (ka)^2/2`
   at low `ka` and `-> 1` at high `ka`, with `<v^2> = V0^2/2` (time average
   only). Evaluating the module's own Rayleigh power formula for a uniform
   velocity distribution reproduces `P = rho c k^2 Q^2/(4 pi)` and, divided by
   the *time-averaged* `<v^2>`, gives exactly `(kR)^2/2` — the exact piston
   result. The power formula is right; the `1/4` denominator is not.
2. **Far above coincidence.** For a baffled panel mode `sigma -> 1`
   (Leppington, Broadbent & Heron 1982). Post-fix Rayleigh route: 1.007388 at
   2 kHz and 1.001175 at 5 kHz, agreeing with Leppington to 0.73% and 0.12%.
3. **Low-`ka` closed form, derived from scratch.** `Q = 4 V0 a b/(pi^2 m n)`,
   `P = rho c k^2 Q^2/(4 pi)`, `<v^2> = V0^2/8` give

       sigma = 2 k^2 Q^2 / (pi a b V0^2) = 32 k^2 a b / (pi^5 m^2 n^2)

   Post-fix module value at `ka = 0.1` (1 m square, mode (1,1)):
   `1.045684e-03`; closed form `32 k^2 a b/pi^5 = 1.045684e-03` (0.00%).

All three agree: the denominator (not Leppington) was the defect. The audit
note that first spotted the factor of two
(`docs/plate_radiation_chain.md` §5, "audit finding for the existing module")
attributed it to the Leppington branch; that attribution is corrected here — the
chain's own rms convention was the correct one.

## 3. Fix (minimal, one root cause)

| file | change |
|---|---|
| `src/tensorfem/acoustic_radiation.py` | `mode_mean_square_velocity` returns `V0^2/8` (time x space average for peak `V0`), derivation in the docstring |
| same | `monopole_radiation_efficiency` prefactor `16 -> 32` (`32 k^2 a b/(pi^5 m^2 n^2)`), `maidanik_radiation_efficiency` docstring follows |
| same | `dipole_radiation_efficiency` divides by `<v^2>` in code, so it doubles automatically — no edit needed (verified by the dipole-law tests) |
| same | `validate_high_frequency_limit` gained a **quadrature self-convergence guard** (see §5) |
| same | `energy_conservation_check`: `conserved` is now the honest statement `sigma > 0`; the unit cap was a convention artefact, replaced by a documented coincidence-peak cap (`2.0`) reported separately |
| `src/tensorfem/ice_noise_chain.py` | `monopole_sigma_lowfreq`: `k^2 Q^2/(pi a b V0^2) -> 2 k^2 Q^2/(pi a b V0^2)`, docstring derivation updated to `<v^2> = V0^2/8` |

Leppington and Maidanik *published* forms were left untouched: they are the
literature values and were right.

## 4. Numbers (before -> after, same worktree, machine-computed)

Panel: 1 m x 1 m, mode (1,1), air (`rho = 1.225`, `c = 343`), `n_points = 40`.

| quantity | pre-audit | post-audit | reference | pre error | post error |
|---|---|---|---|---|---|
| `<v^2>` (`V0 = 1`) | 0.25 | 0.125 | `V0^2/8` | +100% | 0% |
| sigma Rayleigh, 2 kHz | 0.503694 | 1.007388 | Leppington 1.000000 | -49.7% | **-0.73%** |
| sigma Rayleigh, 5 kHz | 0.500587 | 1.001175 | Leppington 1.000000 | -49.9% | **-0.12%** |
| sigma Rayleigh, `ka = 0.1` | 5.228e-4 | 1.045684e-3 | `32k^2ab/pi^5` = 1.045684e-3 | -50.0% | **0.00%** |
| monopole prefactor, (1,1) square | 1.0332 | 2.0641 | `32 a b k_mn^2/(pi^5 m^2 n^2)` | -50.0% | 0% |
| superseded m6164 fit `1.03` | (+0.3% vs pre-audit law) | 50.1% low | analytic 2.0641 | — | 50.1% |

The superseded `1.03` calibration constant is now explained: it was exactly
*half* of the correct (1,1)-square prefactor `2.0641`, i.e. the same missing ½
in fitted form. Its residual against the corrected law is 50.1% ((1,1)),
60.1% ((2,1) aspect 2), 38.2% ((3,1) aspect 2), 10.2% ((1,3)), 9202% ((1,2),
wrong exponent) — the negative-control test was re-baselined accordingly.

Coincidence region: the corrected Rayleigh route exceeds unity there
(measured maxima over a 100 Hz-2 kHz sweep: 1.283 (1,1) @300 Hz, 1.471 (2,2)
@500 Hz, 1.598 (1,3) @700 Hz). This is physical for an individual mode and is
quadrature converged (`n_points` 40/56/72 agree to 6 digits at 300/500/700 Hz),
so the old `sigma <= 1.05` assertions were convention artefacts, not physics;
they are now `<= 2.0` with the measured peak recorded in the test docstring.

## 5. Second finding: the high-frequency validation scored unresolved quadrature

`validate_high_frequency_limit` claims to check `sigma -> 1`. Its Rayleigh
route integrates `sin(kR)/R` on an `n x n` Gauss grid; for the 1 m panel at
`f = 10 kHz` (`ka = 183`) `n_points = 40` returns `sigma = 9.74755`, while
`n_points = 56` returns `1.00029` — the raw value is numerical garbage, and at
20 kHz even 72 points are unresolved (`16.28` vs `8.58`). Because the existing
test only asserted the Leppington branch, the broken anchor was invisible.

Fix: every point is now evaluated at `n_points` and `n_points_refine`
(default `n_points + 16`); the relative drift is reported as
`quadrature_rel_drift` with `quadrature_converged`, and `all_pass_rayleigh`
scores **only** converged rows (`converged_points`, `scored_points`,
`all_rows_rayleigh_converged` are returned). Test:
`test_high_frequency_validation_flags_quadrature_starvation` — 5 kHz passes
(sigma = 1.00117 against 1, 3% band), 10 kHz is flagged with drift 89.7%, and
`all_pass_rayleigh` stays True because it no longer looks at the starved row.

## 6. Verification performed

* `python -m pytest tests/test_acoustic_radiation.py
  tests/test_acoustic_radiation_low_freq_exact.py -q` -> **72 passed**
  (was 69 + 9 failures on the half-convention assertions; every re-baselined
  assertion carries the measured number in its docstring).
* New invariants added: cross-route agreement far above coincidence (2 kHz,
  5 kHz, <2%), absolute low-`ka` literature value (<0.5%), quadrature-starvation
  flagging, `<v^2>` definition.
* `python -m pytest tests/test_ice_noise_chain.py -q` -> **25 passed, 1 xfailed**
  (the consumer's `monopole_sigma_lowfreq` vs shared Rayleigh route stays within
  the 0.3% gate it already had).
* Maidanik / dipole / even-even parity laws: unchanged in form, still pass their
  gates against the Rayleigh route (relative comparisons are convention-free).

## 7. Limits of this audit

* No experimental data; the anchors are analytic (piston, Leppington, monopole
  law) and one in-repo independent implementation (`plate_radiation_chain`,
  rms convention, `sigma(10 f_c) = 1.00003`).
* The subsonic Leppington branch (`sigma = (k/k_mn)^2`, unit prefactor) is the
  published *band-average* form adapted to a modal wavenumber, so it sits a
  fixed factor `2.0641` below the exact per-mode monopole law for the (1,1)
  square mode. It is left as published and the relation is now pinned by
  `test_subsonic_leppington_is_half_the_exact_monopole_law` instead of being
  asserted as a 5% agreement that only held in the half convention.
* Absolute `sigma` values in `docs/acoustic-radiation-low-frequency-asymptotic.md`
  and its JSON evidence predate this audit and were computed in the half
  convention; multiply by 2 (banner added to that document).
* Users who calibrated against the pre-audit numbers see a factor 2: this is a
  deliberate, documented correction, not a silent rescale.