# Winkler-foundation reference solutions: verification audit (2026-11-02)

Scope: the analytical references used to qualify beam/plate-on-elastic-foundation
capability (SI-C / SI-2 line):

* `src/tensorfem/hetenyi_reference.py` - infinite beam (point / distributed load),
  finite beam series, plate-strip reduction;
* `src/tensorfem/plate_mindlin.py::hetenyi_reference` - infinite plate on a Winkler
  foundation under a point load.

Method: every reference is compared with an **independent** oracle that shares no
code with the closed form under test - a Green's-function quadrature
(superposition of the point-load solution), a Hankel quadrature
`w(r) = P/(2 pi) * integral of xi*J0(xi*r)/(D*xi^4+k) d xi` (Bessel function from the
A&S 9.4 polynomials, itself validated against tabulated values), a dense
finite-difference solve of `EI w'''' + k_b w = q`, the exact centre deflection and the
global equilibrium identity `integral of k*w dA = P`. The tests live in
`tests/test_foundation_reference_audit.py` (22 tests, green); every number below is
produced programmatically by `benchmarks/foundation_reference_audit.py` into
`artifacts/foundation_reference_audit.json` (figure:
`artifacts/foundation_reference_audit.png`).

## Findings

### F1 (real defect, fixed) - distributed load on an infinite beam was not the Green's integral

Pre-audit expression:

    w(x) = (q0/k_b) * [1 - 0.5*(A(lam|x-a|)*sgn(x-a) - A(lam|x-b|)*sgn(x-b))]

It is neither the integral of the point-load Green's function nor a solution of
`EI w'''' + k_b w = q`: it leaves an `O(q0)` residual in the governing ODE and tends to
`q0/k_b` in both far fields instead of decaying, i.e. it violates equilibrium for a
finite net load.

Corrected form (exact; `C(xi) = exp(-xi)*cos xi`, `G(d) = sgn(d)*(1-C(lam*|d|))`):

    w(x) = q0/(2*k_b) * [G(x-a) - G(x-b)]

| anchor | pre-audit | corrected |
|---|---|---|
| max rel. error vs Green's quadrature (181 stations, x in [-8, 10] m) | 3284x | 4.7e-8 |
| abs(w) at 40 m (q0 = 1e4 N/m over 1 m, k_b = 2e6 N/m2, lam = 0.35 1/m) | 5.00e-3 (= q0/k_b) | 1.4e-9 |
| ODE residual `(EI w'''' + k_b w - q)/q0`, 5-point FD, h = 1e-3 m | ~1.0 | <= 8.4e-3 |

### F2 (fabricated reference, replaced) - the plate reference was an exponential fit

`plate_mindlin.hetenyi_reference` returned `w0*exp(-beta*r)` and its own docstring
called it a simplified exponential-decay approximation. It is not the Kelvin
solution: it misses the shallow uplift ring (the exact solution changes sign at
`r/l = 3.96` for these parameters) and deviates by up to 2211 % from the exact
solution over `r/l` in [0.02, 4]. The function now returns the exact solution

    w(r) = -P/(2*pi*k*l^2) * kei(r/l),   l = (D/k)^(1/4),   w(0) = P/(8*sqrt(D*k))

with `kei(x) = Im K0(x*exp(i*pi/4))` implemented in `hetenyi_reference.kelvin_kei`
(ascending K0 series for x <= 14, six-term asymptotic above; branch continuity
7.0e-7).

| anchor | value |
|---|---|
| `kei` vs tabulated values (0, 1, 2, 5, 10) | max rel. 6.3e-11 |
| centre deflection vs `P/(8*sqrt(D*k))` | 9.013878e-07 m, rel. < 1e-12 |
| closed form vs Hankel quadrature (r/l = 0.02 ... 4) | max rel. 5.1e-5 |
| force equilibrium `integral k*w dA / P` (r <= 25 l, trapezoid) | 0.9999999967 |
| sign change of the exact solution | r/l = 3.96 |

### F3 (mislabel, now fail-closed) - `finite_beam_series(boundary='free')` returned the simply supported result

The Fourier-sine expansion satisfies `w = w'' = 0` at both ends, i.e. it is exact for
simply supported ends only; `boundary` was accepted and ignored. Free-end Hetenyi
end-conditioning is **not implemented** and is now rejected with a `ValueError`
instead of silently returning a differently-bounded solution. For beams long
compared with `1/lam` use the infinite-beam functions (end effects decay like
`exp(-lam*x)`).

Validation of the surviving series: a dense FD solve of the same ODE reproduces the
exact single-mode response to 6.6e-9, and the truncated series reproduces it to
< 5e-3 with a point-load sampling of a sine load (`n_terms = 600`).

## Error verdict

| quantity | baseline | relative error | < 3 % ? |
|---|---|---|---|
| beam w(x), uniform load | Green's quadrature | 4.7e-8 | yes |
| beam ODE residual | zero | 8.4e-3 of q0 (FD truncation floor) | yes |
| plate w(r) | Hankel quadrature | 5.1e-5 | yes |
| plate w(0) | `P/(8*sqrt(D*k))` | < 1e-12 | yes |
| plate equilibrium | P | 3.3e-9 | yes |
| simply supported beam (FD vs exact mode) | analytic mode | 6.6e-9 | yes |

## Boundary / what is still open

* Only the **axisymmetric infinite plate** has an exact closed form here. Finite
  plates are still judged FE-vs-FE: `tests/test_plate_mindlin.py::test_hetenyi_reproduction`
  only asserts positivity and magnitude, so it is a smoke test, not a quantitative
  cross-check. A quantitative FE-vs-exact plate comparison over mesh pairs is open
  and compute-bound (Mindlin Q4 resolving `l = 0.37 m` on a multi-metre plate).
* Free-end finite beam on a Winkler foundation (Hetenyi end-conditioning forces) is
  not implemented; the API now fails closed instead of mislabelling.  **Closed** (see
  [hetanyi-free-end-reference.md](hetanyi-free-end-reference.md)):
  `hetenyi_reference.finite_beam_bvp` implements the general end-conditioning
  solution (free / simply supported / clamped / guided) with multiple shooting,
  pinned by the Fourier series, the infinite-beam Green's function, the global
  equilibrium invariant and an independent FE model.
* `kelvin_kei` has a ~1e-7 relative accuracy floor set by the series/asymptotic
  branch split; it is a reference helper, not a production hot path.
* These references are qualification oracles only. No panel/contact/1.0-readiness
  claim changes: the open blockers listed in 0.45.0 stay open.
