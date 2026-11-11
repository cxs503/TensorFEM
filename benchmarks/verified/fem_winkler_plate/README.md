# GAP-FEM-WINKLER: Winkler Plate on Elastic Foundation vs Hetényi

## Mission
Verify TensorFEM's Mindlin-Reissner Q4 plate element with Winkler foundation
against the Hetényi (1946) analytical solution for an infinite plate under
point load.

## Physics
- **Governing equation**: D∇⁴w + kw = Pδ(r)/(2πr)
- **Hetényi solution**: w(r) = −P/(2πkl²)·kei(r/l), l = (D/k)^(1/4)
- **Centre deflection**: w(0) = P/(8√(Dk))

## Parameters (sea-ice on water)
| Parameter | Value | Unit |
|---|---|---|
| E | 3.0 GPa | Ice Young's modulus |
| ν | 0.3 | Ice Poisson's ratio |
| h | 0.15 | m, ice thickness |
| k | 10055.2 | N/m³, ρ_water·g |
| P | 10.0 | kN, point load |
| l | 3.099 | m, characteristic length |

## Acceptance Criteria
| Gate | Criterion | Threshold | Result |
|---|---|---|---|
| C1 | Centre deflection error | < 5% | 0.43% ✅ |
| C2 | Equilibrium residual | < 1% | 0.00% ✅ |
| C3 | Profile R² vs kei | > 0.99 | 1.00 ✅ |

## Convergence
| n | β | w_FEM (mm) | w_Hetényi (mm) | Error (%) | R² |
|---|---|---|---|---|---|
| 20 | 3.0 | 12.8650 | 12.9458 | 0.62 | 0.99998 |
| 30 | 3.5 | 12.9514 | 12.9458 | 0.04 | 1.00000 |
| 40 | 4.0 | 12.9790 | 12.9458 | 0.26 | 1.00000 |
| 50 | 4.5 | 12.9924 | 12.9458 | 0.36 | 1.00000 |
| 60 | 5.0 | 13.0015 | 12.9458 | 0.43 | 1.00000 |

## Artifacts
- `result.json` — Full numerical results
- `gates_verified.json` — Machine-readable gate verdicts
- `winkler_deflection_profile.png` — FEM vs Hetényi deflection profile
- `winkler_convergence.png` — Convergence plot

## References
1. Hetényi, M. (1946). *Beams on Elastic Foundation*. Univ. Michigan Press.
2. Timoshenko & Woinowsky-Krieger (1959). *Theory of Plates and Shells*, 2nd ed.
