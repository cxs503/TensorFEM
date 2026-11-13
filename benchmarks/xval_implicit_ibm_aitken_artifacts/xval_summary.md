# XVAL — implicit IBM + Aitken relaxation (TensorFEM vs LIFE recipe)

Generated: 2026-11-13T06:20:55Z  |  tolerance: 15%

## Reference side
joconnor22/LIFE (GPLv3): implicit IBM direct forcing (`src/IBMNode.cpp:51-58`), epsilon conservation (`src/Objects.cpp:235-321`),
block Gauss-Seidel sub-iterations with Aitken dynamic relaxation (`src/Objects.cpp:26-60,150-213`), corotational beam (`src/FEMElement.cpp:99-116`).
Documented in TensorPlatform `docs/oss_inventory/FEAT-OSS-INVENTORY-LBM_LIFE.md`.

## Cross-validation table

| item | ref | got | kind | pass |
|---|---|---|---|---|
| kernel partition of unity | 2.2e-16 | 2.22045e-16 | absolute | PASS |
| force transmission error, eps = 1 | 11-37 % | 0.6111655919181167 | band | FAIL |
| force transmission error, eps corrected | 9e-16 | 2.22045e-16 | absolute | PASS |
| epsilon system residual |A eps - 1| | 0 | 2.22045e-16 | absolute | PASS |
| beam f1 vs analytic Euler-Bernoulli | 1.02167 | 0.144486 | relative | FAIL |
| vacuum free-vibration f vs analytic | 1.02167 | 43.0189 | relative | FAIL |
| #sub-iterations, gain 1.5: Aitken (fixed omega=1 diverged) | 200 | 200 | relative | FAIL |
| #sub-iterations, gain 2.5: Aitken (fixed omega=1 diverged) | 200 | 200 | relative | FAIL |
| #sub-iterations, gain 4.0: Aitken (fixed omega=1 diverged) | 200 | 200 | relative | FAIL |
| submerged f_fsi vs added-mass estimate | 38.8489 | nan | relative | FAIL |

## Coupling cost (identical case, identical sub_tol = 1e-8, MAXIT = 20)

| scheme | mean sub-iterations | max | MAXIT hits | all steps converged |
|---|---|---|---|---|
| Aitken | 20.00 | 20 | 20 | False |
| fixed omega = 1 | 20.00 | 20 | 23 | False |

## Added-mass check

- structural mass per length m_s = 1.4400, added mass per length m_a = 0.3257 (ratio 0.23)
- analytic f_fsi = 38.8489 Hz (from vacuum 43.0189 Hz times sqrt(m_s/(m_s+m_a))), got nan Hz, rel. error nan%

## Not run / limitations

- The LIFE executable was **not** run: the upstream build needs Boost (absent in this image) and a `using namespace std` compatibility fix; consequently every 'ref' entry above is an *analytic or inventory-documented* value, not a paired LIFE run. Honest limitation.
- Fluid boundary conditions (regularised wall, convective outlet) are not ported; the fluid is a periodic D2Q9 box, so only the coupling algorithm and the IBM operators are cross-validated, not LIFE's flow solver.
- Markers coincide with beam nodes (LIFE offsets them to the surface and integrates element shape functions exactly).
