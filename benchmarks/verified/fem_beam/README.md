# fem_beam - Euler-Bernoulli cantilever under uniform load

Verdict: **PASS**  
Element: 2-node cubic Hermite (4 dof), tensorfem.modal.beam_bending_stiffness; consistent load q*[L/2, L^2/12, L/2, -L^2/12].

| n_elem | h [mm] | w_tip FE [mm] | rel. err tip | L2 field rel. err | shear recovered / qL |
|---:|---:|---:|---:|---:|---:|
| 2 | 500.000 | 3.750000 | 4.626e-16 | 1.638e-03 | -1.000000 |
| 4 | 250.000 | 3.750000 | 1.573e-14 | 1.024e-04 | -1.000000 |
| 8 | 125.000 | 3.750000 | 1.965e-13 | 6.398e-06 | -1.000000 |
| 16 | 62.500 | 3.750000 | 1.426e-12 | 3.999e-07 | -1.000000 |
| 32 | 31.250 | 3.750000 | 1.527e-11 | 2.498e-08 | -1.000000 |
| 64 | 15.625 | 3.750000 | 8.078e-12 | 1.561e-09 | -1.000000 |

Analytic tip deflection `qL^4/(8EI)` = 3.750000 mm, tip rotation `qL^3/(6EI)` = 0.005000 rad.
Field L2 convergence order (2..32 elements): **4.000**.

Gates: G1.tip_deflection = 1.965e-13 (<= 1.0e-02) -> PASS; G1.field_order = 4.000e+00 (>= 1.5e+00) -> PASS; G1.equilibrium = 2.558e-14 (<= 1.0e-08) -> PASS

Note: nodal displacements of this element are superconvergent (exact for the quartic exact solution), so the tip error sits at round-off while the *field* error measures the real O(h^4) interpolation error.
