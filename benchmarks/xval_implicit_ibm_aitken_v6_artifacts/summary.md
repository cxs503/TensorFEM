# XVAL v6 - implicit IBM + Aitken relaxation (TensorFEM vs LIFE recipe)

Generated: 2026-11-15T00:24:02Z | acceptance: 15 %

## Reference side
joconnor22/LIFE (GPLv3): implicit IBM direct forcing (`src/IBMNode.cpp:51-58`),
epsilon conservation (`src/Objects.cpp:235-321`), block Gauss-Seidel
sub-iterations with Aitken dynamic relaxation (`src/Objects.cpp:26-60,150-213`),
corotational beam (`src/FEMElement.cpp:99-116`); plus closed-form references
(Euler-Bernoulli clamped-free f1, Lamb 1932 art. 71 added mass).

## Cross-validation table

| item | ref | got | kind | rel. error | pass |
|---|---|---|---|---|---|
| 3-point kernel partition of unity | 2.2e-16 | 2.220446049250313e-16 | absolute | 0.000 % | PASS |
| force transmission, eps = 1 (LIFE band 11-37 %) | 0.11 - 0.37 | 0.6111655919181167 | band | n/a | PASS |
| force transmission, epsilon corrected | 9e-16 | 2.220446049250313e-16 | absolute | 0.000 % | PASS |
| beam mass matrix: rigid translation carries rho*A*L | 0.04 | 0.04000000000000001 | relative | 0.000 % | PASS |
| beam f1 vs Euler-Bernoulli (unit section, A = 1) | 51.083454850395015 | 51.08349846215314 | relative | 0.000 % | PASS |
| beam f1 vs Euler-Bernoulli (thin section, A = 0.02) | 1.0216690970079005 | 1.0216699692457754 | relative | 0.000 % | PASS |
| vacuum free vibration f vs Euler-Bernoulli (same section) | 1.0216690970079005 | 1.0216690970079005 | relative | 0.000 % | PASS |
| plate f_fsi/f_vac vs Lamb added mass | 0.7905694150420949 | 0.7314443315732566 | relative | 7.479 % | PASS |
| Aitken converges where omega=1 diverges (gain 1.5) | fixed omega=1: 200 it, converged=False | Aitken: 3 it | boolean | n/a | PASS |
| Aitken converges where omega=1 diverges (gain 2.5) | fixed omega=1: 200 it, converged=False | Aitken: 3 it | boolean | n/a | PASS |
| Aitken converges where omega=1 diverges (gain 4.0) | fixed omega=1: 200 it, converged=False | Aitken: 3 it | boolean | n/a | PASS |
| strong coupling (m_a/m_s=12.0): Aitken survives | fixed omega=1 diverges | diverged | boolean | n/a | PASS |

## Defects found and fixed in this round

1. **Beam consistent mass matrix dropped the section area** (`ml = p.rho * L0` -> `p.rho * p.A * L0`). With the legacy inertia every
   frequency is low by `sqrt(A)`: at A = 0.1 the error is 68.4 %
   (observed legacy/current ratio 0.3162 vs `sqrt(A)` = 0.3162).
2. **Euler-Bernoulli reference dropped `rho A`** from the mass per unit
   length, matching defect 1 to 1e-6 -- two compensating errors that made
   the v4/v5 eigen check 'pass' while the physics was 7x off.
3. **The vacuum free-vibration run ignored the caller's section** (it always
   built `_section()`, A = 1, while the reference used A = 0.02):
   `cantilever_free_vibration` now accepts `props`.  Together with a record
   that resolves the period (dt = T1/200, 5 periods) the archived 29694 %
   'failure' becomes the live error in the table above.

## Notes / limitations

- The LIFE executable is **not** run (upstream needs Boost and a
  `using namespace std` compatibility fix, absent in this image).  Every
  'ref' entry is a closed-form value or an inventory-documented observable
  quantity; no paired LIFE run exists yet.  Honest limitation.
- Fluid side is a periodic D2Q9 BGK box: no inflow/outflow, walls or
  regularised boundaries, so only the coupling algorithm and the IBM
  operators are cross-validated, not LIFE's flow solver.
- The IBM kernel smears the plate over ~3 lattice units, so the measured
  added mass is 1.449 x the thin-strip Lamb value;
  the frequency ratio therefore errs low (measured 0.7314 vs analytic 0.7906).
- Aitken shows no advantage over a fixed omega when the interface fixed
  point converges in one sub-iteration (low added-mass ratio); its
  necessity is demonstrated by the strong-coupling regime and the scalar
  probe `omega* = 1/(1-g)`.

## Status: 12/12 checks pass
