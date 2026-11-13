# CASE-FEM-JINTEGRAL-EDGE-CRACK - ledger acceptance run

Generated 2026-11-13T02:46:16+00:00 by `PYTHONPATH=src python benchmarks/verified/fem_sent_j_integral/ledger_acceptance.py`.

## Configuration

| item | value |
| --- | --- |
| geometry | SENT plate, W = 0.100 m, a/W = 0.30, H/W = 2.0 |
| loading | uniform tension sigma = 100.0 MPa on the loaded edge |
| material | E = 210.0 GPa, nu = 0.30, E / (1 - nu^2) |
| model | quadratic (T6) ray-fan mesh, quarter-point tip ring, upper half + symmetry |
| reference | K_I = sigma sqrt(pi a) F(a/W), Tada/Paris/Irwin 3rd ed. |

## Reference vs. computed (ledger acceptance point)

| quantity | computed (FEM) | reference (Tada) | rel. error | band | verdict |
| --- | --- | --- | --- | --- | --- |
| mode_i_k [Pa sqrt(m)] | 4.9952e+07 | 5.09592e+07 | -1.976 % | 3 % | PASS |
| shape_factor [1] | 1.62711 | 1.65992 | -1.976 % | 3 % | PASS |

Path independence (J is path independent for a stationary crack): contour fractions ['0.100', '0.150', '0.200'] give F = ['1.62435', '1.62711', '1.62455'], relative spread 0.170 % (band 1 %).

Discrete equilibrium: vertical reaction / applied load = 1.000000000000 (band 1e-9).

## Refinement ladder (a/W = 0.3, plane strain)

| level | sectors | elements | F_FEM | F_Tada | rel. error | reaction ratio |
| --- | --- | --- | --- | --- | --- | --- |
| coarse | 24 | 390 | 1.62711 | 1.65992 | -1.976 % | 1.000000000000 |
| medium | 48 | 1050 | 1.63759 | 1.65992 | -1.345 % | 1.000000000000 |
| fine | 96 | 2842 | 1.64768 | 1.65992 | -0.738 % | 1.000000000000 |

Roache GCI: not applicable (observed order -0.055 < 0.05); medium->fine relative change 0.612 %.

## Same-mesh A/B control (tip ring)

| tip ring | elements | F_FEM | F_Tada | rel. error | inside 3 % band |
| --- | --- | --- | --- | --- | --- |
| quarter_point | 390 | 1.62711 | 1.65992 | -1.976 % | yes |
| midpoint | 390 | 1.61759 | 1.65992 | -2.550 % | yes |

Quarter-point ring improves the same-mesh accuracy by 0.57 pp: **YES**. Limitation (honest): both rings are inside the 3 % band on this mesh, so the band alone does not demonstrate the necessity of the quarter-point ring.

## Verdict

Ledger band (5 %) met: **YES**. Platform band (3 %) met: **YES**.

Scope/limits: linear-elastic mode-I stationary crack, 2-D plane strain, homogeneous isotropic, no plasticity, no crack closure, no body force, H/W = 2 truncation (the height sensitivity is measured separately in `run.py --stage height`).
