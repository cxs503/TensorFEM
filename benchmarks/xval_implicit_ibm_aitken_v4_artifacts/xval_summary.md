# XVAL — implicit IBM + Aitken relaxation (TensorFEM vs LIFE recipe) — v4

Generated: 2026-11-16T02:47:23Z  |  tolerance: 15%

## Reference side
joconnor22/LIFE (GPLv3): implicit IBM direct forcing (`src/IBMNode.cpp:51-58`), epsilon conservation (`src/Objects.cpp:235-321`),
block Gauss-Seidel sub-iterations with Aitken dynamic relaxation (`src/Objects.cpp:26-60,150-213`), corotational beam (`src/FEMElement.cpp:99-116`).
Documented in TensorPlatform `docs/oss_inventory/FEAT-OSS-INVENTORY-LBM_LIFE.md`.

## Cross-validation table

| item | ref | got | kind | verdict |
|---|---|---|---|---|
| kernel partition of unity | 2.2e-16 | 2.22045e-16 | absolute | PASS |
| force transmission error, eps = 1 (sanity window only; not an agreement test) | LIFE 11-37 % (surface markers); our pass band 2-70 % is a sanity window, not agreement | 0.611166 | band | OUT_OF_REF_BAND |
| force transmission error, eps corrected | 9e-16 | 2.22045e-16 | absolute | PASS |
| vacuum free-vibration f vs analytic | 1.02167 | 1.02539 | relative | PASS |
| #sub-iterations, gain 1.5: Aitken converged while fixed omega=1 did not (qualitative) | 200 = iteration cap; omega=1 is outside 0 < omega < 2/g, divergence is guaranteed | 3 | qualitative | PASS |
| #sub-iterations, gain 1.5: Aitken vs optimal fixed relaxation omega*=1/(1-g) | 2 | 3 | relative | FAIL |
| #sub-iterations, gain 2.5: Aitken converged while fixed omega=1 did not (qualitative) | 200 = iteration cap; omega=1 is outside 0 < omega < 2/g, divergence is guaranteed | 3 | qualitative | PASS |
| #sub-iterations, gain 2.5: Aitken vs optimal fixed relaxation omega*=1/(1-g) | 2 | 3 | relative | FAIL |
| #sub-iterations, gain 4.0: Aitken converged while fixed omega=1 did not (qualitative) | 200 = iteration cap; omega=1 is outside 0 < omega < 2/g, divergence is guaranteed | 3 | qualitative | PASS |
| #sub-iterations, gain 4.0: Aitken vs optimal fixed relaxation omega*=1/(1-g) | 2 | 3 | relative | FAIL |
| plate f_fsi/f_vac vs analytic added-mass | 0.790569 | 0.734998 | relative | PASS |

`INVALID_WINDOW` = the measurement record is too short for the period
estimator (see `period_window_ok`); it is not a physics result.
`OUT_OF_REF_BAND` = inside our sanity window but outside LIFE's published
band: a configuration difference, *not* a cross-validation pass.
`qualitative` rows are convergence statements, not measured agreements.

Run mode: **mixed (structure/submerged: quick; plate: full 2400 steps, seat m22878)** (structure 1500 steps, plate 2400 steps).

## Verdict accounting

- checks: 11, pass: 7, all_pass: False
- open failures: ['force transmission error, eps = 1 (sanity window only; not an agreement test)', '#sub-iterations, gain 1.5: Aitken vs optimal fixed relaxation omega*=1/(1-g)', '#sub-iterations, gain 2.5: Aitken vs optimal fixed relaxation omega*=1/(1-g)', '#sub-iterations, gain 4.0: Aitken vs optimal fixed relaxation omega*=1/(1-g)']
- out of reference band (not counted as pass): ['force transmission error, eps = 1 (sanity window only; not an agreement test)']
- invalid windows: []

## Coupling cost

| mode | mean sub-iter | max sub-iter | maxit hits | converged all |
|---|---|---|---|---|
| Aitken | 2.00 | 2 | 0 | True |
| fixed ω=1 | 2.00 | 2 | 0 | True |

## Vacuum frequency (beam, clamped-free)

- Analytic f1 (Euler-Bernoulli, with area): 1.021669 Hz (T = 0.9788 s)
- FEM eigen: 1.021670 Hz (error: 0.0001 %)
- Section requested == simulated: True (A = 0.02)
- Record: 1500 steps x dt = 0.005 s = 7.69 periods (gate 2.0)
- Window valid: True
- Free-vibration f (damped-sinusoid fit): 1.025391 Hz (error: 0.3643 %)
- Free-vibration f (legacy mean-removed zero crossings, record only): 1.023018 Hz

## Plate added-mass frequency ratio

- c = 16, lattice = [144, 144], steps = 2400
- Analytic f_vac: 0.00210819 Hz, f_fsi: 0.00166667 Hz
- Analytic f_fsi/f_vac: 0.790569
- Measured f_fsi/f_vac (damped fit, valid = True): 0.734998
- Measured f_fsi/f_vac (legacy zero crossings): 0.730881
- Error (damped fit): 7.03 %
- Record length: 3.72 periods (gate 2.0), window valid: True
- Verdict: **PASS**

## Notes

1. ε=1 force-transmission error (61 %) exceeds LIFE's 11–37 % band by 65 %
   because our test uses centreline markers (spacing ≈ 1 Δ) rather than LIFE's
   surface markers with exact element shape-function integration. The row is
   therefore reported as OUT_OF_REF_BAND (sanity window 2–70 %) and is excluded
   from the pass count: it cannot certify IBM force conservation against LIFE.
2. The beam submerged frequency remains open: the beam's effective thickness in
   the IBM is set by the 3-point delta support (≈ 3 Δ), not a physical thickness,
   so the analytic added-mass estimate (m_a = ρ_f π b²/4) does not apply directly.
   The plate geometry (v4) closes this gap with an analytic reference.
3. Aitken relaxation shows no advantage over ω=1 at the tested added-mass ratios
   (≤ 0.23 for the beam, 0.6 for the plate) because the interface fixed point
   converges in 1 sub-iteration. The scalar-probe evidence (ω* = 1/(1−g)) stands.
4. v4.1 (mission 22704) fixes two harness defects that produced the shipped
   43.048 Hz / 1.5811 rows: the free-vibration run now receives the section its
   analytic reference describes, and both frequency rows are gated by the record
   length instead of being reported as physics failures.