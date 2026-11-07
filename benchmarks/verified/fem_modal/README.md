# fem_modal - Euler-Bernoulli cantilever, modes 1-5

Verdict: **PASS**  
Element: 2-node cubic Hermite beam, consistent mass (tensorfem.modal).

| mode | beta_n L | f_exact [Hz] | err @ 4 el. | err @ 8 el. | err @ 16 el. | err @ 32 el. | err @ 64 el. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.875104 | 8.154 | 3.27e-05 | 2.08e-06 | 1.31e-07 | 8.09e-09 | 7.90e-10 |
| 2 | 4.694091 | 51.099 | 1.17e-03 | 8.00e-05 | 5.11e-06 | 3.21e-07 | 2.03e-08 |
| 3 | 7.854757 | 143.079 | 7.74e-03 | 6.08e-04 | 3.98e-05 | 2.51e-06 | 1.57e-07 |
| 4 | 10.995541 | 280.377 | 1.45e-02 | 2.24e-03 | 1.51e-04 | 9.62e-06 | 6.04e-07 |
| 5 | 14.137168 | 463.484 | 1.41e-01 | 5.79e-03 | 4.08e-04 | 2.62e-05 | 1.65e-06 |

Observed orders: mode_1 = 3.868, mode_2 = 3.957, mode_3 = 3.909, mode_4 = 3.697, mode_5 = 4.056.

Gates: G3.frequencies = 4.082e-04 (<= 1.0e-02) -> PASS; G3.roots_vs_table = 3.705e-06 (<= 1.0e-05) -> PASS; G3.characteristic_residual = 3.060e-15 (<= 1.0e-12) -> PASS; G3.order = 3.697e+00 (>= 1.5e+00) -> PASS
