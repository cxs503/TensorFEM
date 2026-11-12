# CASE-FEM-CLAMPED-PLATE-UNIFORM — round 2: research, method selection and cross-family verification

Mission 16320 · repo TensorFEM · branch agent/m16320 · case script `benchmarks/case_clamped_plate_uniform.py` · artifacts `benchmarks/artifacts/CASE-FEM-CLAMPED-PLATE-UNIFORM/`

## 0. What round 1 left, and why round 2 is different

Round 1 (mission 16181) delivered a passing case: `alpha = 0.00126420` vs the Timoshenko Table 30 entry 0.00126 (0.33 %, gate 3 %). The platform's maturity review graded it **L1**: every number came from **one** discretisation family (Q4 Mindlin), extrapolated with Richardson and cross-checked against a Rayleigh–Ritz reference that shares the same Kirchhoff physics but is not an element code. A one-family limit is self-consistency; it cannot detect a family-wide modelling error (wrong load vector, wrong shear split, wrong curvature recovery). Round 2 therefore adds an **unrelated element family** and grades both families against the same external anchor, plus algebraic identities that no reference can fake (patch test, discrete equilibrium).

## 1. Research: horizontal survey and method genealogy

**Horizontal (open-source) survey.** The mandated query `SELECT * FROM oss_inventory WHERE feature_slug ILIKE '%UNIFORM%'` **cannot run from the agent role**: `oss_inventory`/`oss_solutions`/`negative_knowledge_ledger` all answer `permission denied for table ...` for the role behind `TP_PG_DB` (this is the same friction already logged in `benchmarks/artifacts/CASE-FEM-CLAMPED-PLATE-UNIFORM/ecosystem_friction.md` item 2). A cluster-side scan of `~/oss-cache` was submitted (`cj1298`) but had not been dispatched by the end of the mission (queue age 15+ min with ~460 jobs ahead), so it is **not** claimed as evidence. What *was* usable: the workstation `~/oss-cache` (LBM libraries only — no plate/FEM project) and the repository's own prior art, which turned out to be the decisive find: `src/tensorfem/plate_bfs.py` (Bogner–Fox–Schmit C1 bicubic element, `assemble_bfs_plate`/`solve_bfs_static`, sparse SuperLU) and `benchmarks/verified/fem_plate_static/run.py` (rung L2 of another anchor, SSSS+CCCC with the BFS family) already existed, with `tests/test_plate_bfs.py` guarding them. The anchor case simply never used them.

**Method genealogy (literature / textbook standard practice).**

| method | reference | role in this case |
|---|---|---|
| Kirchhoff thin-plate theory, `D = E t^3/(12(1-nu^2))`, `alpha = w_max D/(q a^4)`; clamped-square coefficient 0.00126 and centre moment 0.0231 | Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., McGraw-Hill 1959, Sec. 30 / Table 30 (and Table 35) | external anchor (the only transcribed numbers in the case) |
| Navier double-sine series (simply supported, uniform load) | same, Sec. 30/39 | closed-form oracle for the family-machinery check |
| Rayleigh–Ritz with clamped–clamped beam modes | Timoshenko (1959) Ch. 5; Young (1950) | converged independent reference for the clamped case (0.00126825) |
| C1-conforming bicubic Hermite rectangle, 4 dof/node `(w,w_x,w_y,w_xy)` | Bogner, Fox & Schmit, *Proc. 2nd Conf. Matrix Methods Struct. Mech.* 1965; Zienkiewicz, *The Finite Element Method* — the classical conforming Kirchhoff element | **added as family B** (already in `tensorfem.plate_bfs`, now actually used by the case) |
| Mindlin–Reissner C0 Q4 with selective reduced integration (SRI) / MITC4 assumed natural shear strain | Bathe & Dvorkin, *Int. J. Numer. Meth. Eng.* 21 (1985); Zienkiewicz | family A (round 1 path, unchanged) |

**Selection and migration.** Keep family A as the production path (it is what ship panels use: shear-deformable, extensible to thick plating/`a/t = 10`), and add family B as the **verification** path for thin-plate anchors. Migration plan realised here: (i) export the constant-curvature/constant-twist patch test as a library capability (`tensorfem.plate_bfs.bfs_patch_test`); (ii) let the anchor case assemble BFS meshes directly (no new element code was written — the point is independence, not novelty); (iii) add the cross-family gate + regression tests; (iv) treat the round-1 anchors as a subset (they are all still in the artifact, 11 of 18 anchors).

## 2. Method and implementation

- Case script: `benchmarks/case_clamped_plate_uniform.py` — family A (`solve_uniform_square_plate`, Q4 Mindlin, `winkler_k = 0`) and family B (`assemble_bfs_plate` + `bfs_boundary_dofs("cccc")` + `solve_bfs_static`, sparse SuperLU) solve the *same* boundary-value problem.
- New helpers in the case: `bfs_solve()` (deflection, centre moment, reaction sum), `self_convergence_order()` (order from successive differences, no exact reference).
- New library capability: `src/tensorfem/plate_bfs.py::bfs_patch_test()` — constant-curvature (2) / constant-twist (1) field `w = x^2 + y^2 + xy` on a 3x2 mesh of 2:1 rectangles, complete boundary trace prescribed, interior dofs solved; returns max error / field scale. Docstring records why the twist dof must be part of the trace (a free boundary twist dof carries a zero-corner-force natural condition that a constant-twist field violates: measured interior residual 9e-10 vs 2.9e3 at free corner twist dofs).
- New tests: `tests/test_case_clamped_plate_uniform_crossfamily.py` (8 tests: patch test to round-off on 2:1 and square patches, BFS vs Navier, BFS vs table band, cross-family agreement, refinement of the family gap, discrete equilibrium, BC validation).

## 3. Parameters (complete dimensionless set)

| quantity | value |
|---|---|
| side a | 1 m |
| thickness t | 0.01 m |
| a/t | 100 |
| E | 2.1e+11 Pa |
| nu | 0.3 |
| q | 1e+06 Pa |
| D | 19230.8 N m |
| alpha (group) | w_max D/(q a^4) |
| table band (3 printed digits) | +-0.40% |

## 4. Comparison table (ref vs got, every gate)

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

Reference sources: Timoshenko & Woinowsky-Krieger 1959 Table 30 (alpha = 0.00126, M = 0.0231); exact Navier series (0.0040623527, 401 odd terms, series-vs-801 stability < 1e-9); converged clamped Ritz N = 12 (alpha = 0.00126825; N = 10/12 spread 0.216%); Richardson limit of the Mindlin series (0.00126786, observed order 1.99).

**Does the key quantity meet the < 3 % benchmark?** Yes: family A 0.33 %, family B 0.43 %, and the two families agree with each other to 0.20 %% — an order of magnitude inside the 3 % gate and of the same size as the table's own +-0.40 % rounding band.

## 5. Convergence and sensitivity

Family A (Q4 Mindlin), clamped, a/t = 100:

| nx | alpha | err vs table | err vs Ritz |
|---|---|---|---|
| 4 | 0.00121383 | -3.664% | -4.291% |
| 6 | 0.00124213 | -1.418% | -2.059% |
| 8 | 0.00125330 | -0.532% | -1.179% |
| 10 | 0.00125854 | -0.116% | -0.766% |
| 12 | 0.00126138 | +0.109% | -0.542% |
| 16 | 0.00126420 | +0.334% | -0.319% |
| 20 | 0.00126552 | +0.438% | -0.216% |
| 24 | 0.00126623 | +0.495% | -0.159% |
| 32 | 0.00126694 | +0.551% | -0.103% |

Family B (BFS C1 Kirchhoff), clamped, a/t = 100:

| nx | dofs | alpha | err vs table | err vs Ritz |
|---|---|---|---|---|
| 10 | 484 | 0.00126925 | +0.734% | +0.079% |
| 20 | 1764 | 0.00126581 | +0.461% | -0.192% |
| 40 | 6724 | 0.00126538 | +0.427% | -0.226% |

Self-convergence order of family B over 10/20/40: **2.99**. Cross-family gap at matched meshes: 10 -> 0.845%, 20 -> 0.023%, 32 -> 0.118%. The gap is not monotone (it is 0.023 % at nx = 20 and 0.118 % at nx = 32) because the two families converge to *different* limits by design: the Mindlin family keeps transverse shear at a/t = 100, while BFS is pure Kirchhoff; the residual gap 0.196 % is dominated by discretisation, not by the physical shear term (analytic shear increment at a/t = 100 is ~3e-5 relative, i.e. two orders smaller).

Sensitivity axes (family A, unchanged from round 1): slenderness a/t in {200,100,50,20,10} — thin range stays within 0.5 %% of the Kirchhoff limit, a/t = 10 raises alpha by +18.4 %% (shear branch live); Poisson ratio nu in {0.25,0.30,0.35} — worst deviation from the Ritz reference 0.34 %%, i.e. the table's nu = 0.3 restriction is not a hidden error source.

## 6. Failure modes, limitations and artifacts

- **Reference rounding is the floor**: the table has 3 significant digits, so +-0.40 %% is irreducible; the case reports it and does not pretend to certify below it.
- **Clamped-edge moment peak not certified**: pointwise moments at a clamped edge converge slowly in both methods; the moment anchor is the centre value only.
- **BFS order 3.0, not 4**: for a smooth bicubic solution one expects O(h^4); the observed 3.0 is consistent with the clamped corner twist dof being natural, which introduces a mild boundary-layer effect. Reported, not hidden; it does not affect the value (0.43 %% error at nx = 40).
- **Free-twist patch test is a trap**: with boundary twist dofs left free, the exact constant-twist field is *not* reproduced (residual 2.9e3 vs 9e-10). That is a boundary-condition property, not an element defect; documented in the docstring and in the case docstring so the next worker does not re-derive it.
- **Research gap (honest)**: `oss_inventory` unreadable for the agent role and the cluster `~/oss-cache` scan (`cj1298`) still queued at close of mission; the horizontal survey is limited to the workstation cache (LBM-only) plus in-repo prior art.

Artifacts / repro:

- `PYTHONPATH=src python benchmarks/case_clamped_plate_uniform.py` (151.7 s single core) -> `benchmarks/artifacts/CASE-FEM-CLAMPED-PLATE-UNIFORM/case_result.json` (18 anchors, `all_gates_passed: true`), `case_result.md`, figure `docs/assets/case-clamped-plate-uniform/clamped_plate_uniform.png` (left panel now overlays both families' mesh convergence).
- `PYTHONPATH=src python -m pytest tests/test_case_clamped_plate_uniform_crossfamily.py -q` -> 8 passed (31.5 s).
- `PYTHONPATH=src python -m pytest tests/test_plate_bfs.py -q` (existing family-B regression suite, unchanged).

## 7. What this changes for the maturity level

Round 1 evidence: one family vs table + one family's Richardson limit -> L1. Round 2 evidence adds (a) a second element family that reproduces the closed-form Navier case to 1e-6 relative before grading the clamped case, (b) cross-family agreement at 0.20 %%, (c) an algebraic patch test at round-off, (d) the discrete equilibrium identity, (e) a regression test file, and (f) a reusable library patch test. The anchor number itself did not change (0.33 %% vs 0.43 %% for the two families) — what changed is that it is now supported by two independent codes instead of one.

