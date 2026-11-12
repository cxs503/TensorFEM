# CASE-FEM-CLAMPED-PLATE-UNIFORM -- clamped square plate under uniform pressure

Reference: Timoshenko & Woinowsky-Krieger 1959, Theory of Plates and Shells, 2nd ed., Sec. 30, Table 30 (uniformly loaded rectangular plate, w_max = alpha q a^4/D), square plate, all edges built in, nu = 0.3 -> alpha = 0.00126

Independent (computed) reference: Rayleigh-Ritz N=12 alpha = 0.00126825 (the table entry is rounded to 3 digits, band +-0.40%)

## Anchors

| anchor | got | reference | rel. error | gate | verdict |
|---|---|---|---|---|---|
| reference_machinery_ss_vs_navier | 0.0040620279 | 0.0040623527 | 0.008% | 0.1% | PASS |
| reference_truncation_stability | 0.0012682503 | 0.0012655097 | 0.216% | 0.5% | PASS |
| dimensionless_group_scale_invariance | 0.0012533011 | 0.0012533011 | 0.000% | 0.0% | PASS |
| finest_mesh_vs_ritz | 0.0012669429 | 0.0012682503 | 0.103% | 1.0% | PASS |
| converged_fe_limit_vs_table | 0.0012678611 | 0.00126 | 0.624% | 3.0% | PASS |
| converged_fe_limit_vs_ritz | 0.0012678611 | 0.0012682503 | 0.031% | 1.0% | PASS |
| primary_fe_vs_timoshenko_table | 0.0012642047 | 0.00126 | 0.334% | 3.0% | PASS |
| secondary_fe_centre_moment_vs_table | 0.022674527 | 0.0231 | 1.842% | 3.0% | PASS |
| slenderness_independence_thin_range | 0.0012622875 | 0.0012682503 | 0.470% | 1.0% | PASS |
| mindlin_shear_raises_thick_plate_deflection | 0.0015021593 | 0.0012682503 | 18.443% | 100.0% | PASS |
| poisson_sweep_fe_vs_ritz | 0.0012639386 | 0.0012682565 | 0.340% | 1.0% | PASS |
| bfs_patch_constant_curvature_exact | 6.8199414e-16 | 0 | 0.000% | 0.0% | PASS |
| bfs_family_machinery_ss_vs_navier | 0.004062357 | 0.0040623527 | 0.000% | 0.5% | PASS |
| independent_family_bfs_vs_table | 0.001265382 | 0.00126 | 0.427% | 3.0% | PASS |
| independent_family_bfs_vs_ritz | 0.001265382 | 0.0012682503 | 0.226% | 1.0% | PASS |
| cross_family_bfs_vs_mindlin | 0.0012678611 | 0.001265382 | 0.196% | 1.0% | PASS |
| bfs_global_equilibrium_reaction | 1000000 | 1000000 | 0.000% | 0.0% | PASS |
| independent_family_bfs_centre_moment_vs_table | 0.02292189 | 0.0231 | 0.771% | 3.0% | PASS |

## Mesh convergence (a/t = 100, clamped)

| nx | nodes | alpha | err vs table | err vs Ritz | s |
|---|---|---|---|---|---|
| 4 | 25 | 0.00121383 | -3.664% | -4.291% | 0.4 |
| 6 | 49 | 0.00124213 | -1.418% | -2.059% | 1.0 |
| 8 | 81 | 0.00125330 | -0.532% | -1.179% | 1.8 |
| 10 | 121 | 0.00125854 | -0.116% | -0.766% | 2.8 |
| 12 | 169 | 0.00126138 | +0.109% | -0.542% | 4.0 |
| 16 | 289 | 0.00126420 | +0.334% | -0.319% | 7.9 |
| 20 | 441 | 0.00126552 | +0.438% | -0.216% | 13.5 |
| 24 | 625 | 0.00126623 | +0.495% | -0.159% | 19.2 |
| 32 | 1089 | 0.00126694 | +0.551% | -0.103% | 31.7 |

Richardson (last three meshes): order = 1.99, limit = 0.00126786 (table +0.624%, Ritz -0.031%)

## Sensitivity A -- slenderness (nx = 16)

| a/t | alpha FE | alpha Ritz (Kirchhoff) | shear effect | err vs table |
|---|---|---|---|---|
| 200 | 0.00126229 | 0.00126825 | -0.47% | +0.182% |
| 100 | 0.00126420 | 0.00126825 | -0.32% | +0.334% |
| 50 | 0.00127182 | 0.00126825 | +0.28% | +0.938% |
| 20 | 0.00132397 | 0.00126825 | +4.39% | +5.077% |
| 10 | 0.00150216 | 0.00126825 | +18.44% | +19.219% |

## Sensitivity B -- Poisson ratio (nx = 16, a/t = 100)

| nu | alpha FE | alpha Ritz | err |
|---|---|---|---|
| 0.25 | 0.00126394 | 0.00126826 | -0.340% |
| 0.30 | 0.00126420 | 0.00126825 | -0.319% |
| 0.35 | 0.00126450 | 0.00126824 | -0.295% |

## Independent family B -- BFS C1 bicubic Kirchhoff (round 2: cross-family verification)

Element: Bogner--Fox--Schmit C1 bicubic Hermite (``tensorfem.plate_bfs``), 4 dof/node, Kirchhoff bending only, sparse SuperLU -- no element code, kinematics or solver shared with the Q4 Mindlin path above.

| check | got | reference | rel. error | gate |
|---|---|---|---|---|
| patch test w=x^2+y^2+xy (2:1 elements) | 4.774e-15 | 0 (exact) | 6.82e-16 | 1e-10 |
| BFS SSSS 20x20 vs exact Navier | 0.00406236 | 0.00406235 | 0.000% | 0.5% |
| equilibrium: |reaction| vs q a^2 (40x40) | 1.000000e+06 | 1.000000e+06 | 1.4e-12 | 1e-10 |

### BFS ladder (clamped, a/t = 100)

| nx | dofs | alpha | err vs table | err vs Ritz | self-convergence vs finer |
|---|---|---|---|---|---|
| 10 | 484 | 0.00126925 | +0.734% | +0.079% | - |
| 20 | 1764 | 0.00126581 | +0.461% | -0.192% | - |
| 40 | 6724 | 0.00126538 | +0.427% | -0.226% | - |

Self-convergence order over the 10/20/40 ladder: **2.99**.  Cross-family gap to the Q4 Mindlin Richardson limit (0.00126786): **0.196%**.

### Cross-family ladder at matched meshes

| nx | alpha Q4 Mindlin | alpha BFS | family gap |
|---|---|---|---|
| 10 | 0.00125854 | 0.00126925 | 0.845% |
| 20 | 0.00126552 | 0.00126581 | 0.023% |
| 32 | 0.00126694 | 0.00126544 | 0.118% |

## Verdict

**PASS** -- primary anchor 0.00126420 vs 0.00126 = 0.33% < 3% gate.

Reproduce: `PYTHONPATH=src python benchmarks/case_clamped_plate_uniform.py`
