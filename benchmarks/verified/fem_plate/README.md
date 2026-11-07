# fem_plate - Kirchhoff SSSS square plate, uniform pressure

Verdict: **PASS**  
Element: Bogner-Fox-Schmit C1 bicubic Hermite rectangle (16 dof; w, w,x, w,y, w,xy); BC: hard simple support (w = 0 and tangential slope = 0 on edges).

| n x n | n_dof | h [mm] | w_max [mm] | alpha = w D/(q a^4) | err vs 0.00406 | err vs Navier |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 500.00 | 1.440844 | 0.002771 | 31.752% | 31.792% |
| 4 | 100 | 250.00 | 1.919507 | 0.003691 | 9.080% | 9.132% |
| 6 | 196 | 166.67 | 2.025016 | 0.003894 | 4.082% | 4.138% |
| 8 | 324 | 125.00 | 2.062941 | 0.003967 | 2.286% | 2.342% |
| 10 | 484 | 100.00 | 2.080662 | 0.004001 | 1.446% | 1.504% |
| 12 | 676 | 83.33 | 2.090332 | 0.004020 | 0.988% | 1.046% |
| 16 | 1156 | 62.50 | 2.099978 | 0.004038 | 0.532% | 0.589% |
| 20 | 1764 | 50.00 | 2.104452 | 0.004047 | 0.320% | 0.377% |
| 24 | 2500 | 41.67 | 2.106886 | 0.004052 | 0.204% | 0.262% |

Reference: Timoshenko table value alpha = 0.00406, Navier series alpha = 0.004062353.
Observed convergence order: **1.944** (deflection, from below).

Gates: G2.centre_deflection = 5.316e-03 (<= 2.0e-02) -> PASS; G2.centre_deflection_vs_series = 5.892e-03 (<= 2.0e-02) -> PASS; G2.order = 1.944e+00 (>= 1.5e+00) -> PASS
