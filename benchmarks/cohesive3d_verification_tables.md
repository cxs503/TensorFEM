| check | oracle (closed form) | measured | tolerance | status |
|---|---|---|---|---|
| `law_reduces_to_2d_bk` | 3-D law == merged 2-D BK law on delta_s2 = 0 | max scaled traction difference over 5 mode-mix ratios x 5 amplitudes | 1e-12 | PASS |
| `law_energy_mode_I` | int t . d(delta) along the envelope == G_c(direction) | mode_I: measure 500.000000 vs 500.000000 J/m^2 | 1e-05 | PASS |
| `law_energy_mode_II` | int t . d(delta) along the envelope == G_c(direction) | mode_II: measure 1199.999919 vs 1200.000000 J/m^2 | 1e-05 | PASS |
| `law_energy_mode_III` | int t . d(delta) along the envelope == G_c(direction) | mode_III: measure 1199.999919 vs 1200.000000 J/m^2 | 1e-05 | PASS |
| `law_energy_mixed_45` | int t . d(delta) along the envelope == G_c(direction) | mixed_45: measure 594.316809 vs 594.316824 J/m^2 | 1e-05 | PASS |
| `law_initiation_quadratic` | (t_n/t_n0)^2 + (t_s/t_s0)^2 == 1 at damage onset | max |quadratic criterion - 1| over 11 mode-mix angles | 1e-12 | PASS |
| `law_tangent_vs_ad` | analytic dt/d(delta) == autograd Jacobian of the traction | max relative tangent difference over 35 (angle, amplitude) points | 1e-10 | PASS |
| `law_batch_matches_pointwise` | batched update (16 softening points) == 16 independent point calls | max relative difference 0.0e+00 at damage 0.991 | 1e-12 | PASS |
| `element_patch_small_strain` | uniform opening: reaction on the top face == t * A (unit area) | measure 4615384.615385 N vs 4615384.615385 N | 1e-12 | PASS |
| `element_rigid_rotation_small_strain` | rigid rotation of the body in a fixed frame must give zero traction | max |f| / (t_n0 A) over theta = [0.01, 0.1, 0.3, 0.7, 1.4, 3.0] (reference-frame separation) | n/a | **KNOWN_LIMIT** |
| `element_rigid_rotation_corotational` | rigid rotation of the body in a fixed frame must give zero traction | max |f| / (t_n0 A) over theta = [0.01, 0.1, 0.3, 0.7, 1.4, 3.0] (current-frame separation) | 1e-10 | PASS |
| `element_rotation_objectivity_corotational` | f(R X, R u) == R f(X, u) for a rotated element geometry | relative Frobenius difference of the reaction under R (0.5, -0.2, 0.9) | 1e-09 | PASS |
| `corotational_elastic_tangent_symmetry` | tangent is the Hessian of a potential in the elastic regime => symmetric | max |K - K^T| / max |K| | 1e-12 | PASS |
| `corotational_tangent_vs_finite_difference` | d f / d u == central differences of the internal force | relative Frobenius difference at damage 0.762 (central differences, eps = 1e-9 m) | 1e-06 | PASS |
| `corotational_dissipation_rotation_invariance` | dissipated work under monotonic opening is independent of rigid rotation | max |W(theta) - W(0)| / W(0) for theta = 0.0, 0.4, 1.2 rad | 1e-12 | PASS |
| `corotational_dissipation_vs_fracture_energy` | dissipated work == G_Ic * A (trapezoid integration, 40 steps) | measure 498.8077 J vs 500.0 J | 5e-03 | PASS |
| `small_strain_dissipation_at_rotation` | same opening at theta = 0.4 rad, reference-frame separation (known limit) | measure -1559043.3 J vs 500.0 J | n/a | **KNOWN_LIMIT** |
| `law_dissipation_vs_path_integral` | closed-form W_d(a_f) == int t . d(delta) along a monotonic ray (7 mixities) | max relative difference, 0deg 500.0000; 15deg 502.2829; 30deg 522.8049 ... | 1e-05 | PASS |
| `law_dissipation_equals_fracture_energy` | W_d(a_f) == G_c(direction), BK mixed mode, modes I / II / III / 45deg / 3-D | max relative difference from G_c, 5 directions, worst 5.74e-16 | 1e-12 | PASS |
| `law_dissipation_monotone_bounded` | W_d == 0 on the elastic branch, non-decreasing, saturating at G_c | elastic max 0.0e+00 J/m^2, worst decrease 1.1e-13, overshoot 2.2e-16 | 1e-12 | PASS |
| `element_dissipation_equals_gc_times_area` | element diagnostics ``dissipated_energy`` == G_Ic * A at complete opening | small_strain 500.000000 J, corotational 500.000000 J vs 500.0 J | 1e-12 | PASS |
| `element_dissipation_rotation_invariance` | element ``dissipated_energy`` independent of a rigid rotation (corotational) | max relative spread over theta = 0.0, 0.3, 1.2 rad | 1e-14 | PASS |
| `nonproportional_path_dissipation` | open to 1.5 a_0(I) then shear to failure: 0 <= W - G_IIc and W_d ~= W_path | W_path 1200.249 J/m^2 vs G_IIc 1200.0, closed form 1199.754 (residual 4.1e-04) | 1e-03 | PASS |
| `corotational_mesh_objectivity` | n x n patches (n = 1, 2, 4) of the same unit interface give the same reaction | max relative reaction difference from t * A | 1e-11 | PASS |
| `corotational_assembly_consistency` | dense patch assembly == sum of element contributions (force and tangent) | max relative difference, 2 x 2 patch | 1e-12 | PASS |
| `corotational_compression_undamaged` | closing uses the penalty stiffness k_n and never damages the interface | reaction -9999.999995 N vs -10000.000000 N, damage 0.0e+00 | 1e-08 | PASS |
| `specimen_compliance_a20mm` | bracket [plain 5.742787e-06, 1.10x root rotation 8.535434e-06] m/N | C_fem = 7.931010e-06 m/N (+2.2% vs root rotation) | 0e+00 | PASS |
| `specimen_compliance_a30mm` | bracket [plain 1.919619e-05, 1.10x root rotation 2.592416e-05] m/N | C_fem = 2.202367e-05 m/N (-6.6% vs root rotation) | 0e+00 | PASS |
| `specimen_compliance_a40mm` | bracket [plain 4.534801e-05, 1.10x root rotation 5.827473e-05] m/N | C_fem = 4.713796e-05 m/N (-11.0% vs root rotation) | 0e+00 | PASS |
