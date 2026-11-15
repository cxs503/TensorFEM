# lbm_table_selfcheck

Anchor `XVAL-LBM-TABLE-DRIVEN-LATTICE-SELFCHECK`.  Table-driven LB velocity sets
(D1Q3, D2Q9, D3Q15, D3Q19, D3Q27) audited in exact rational arithmetic, attacked
with corrupted tables, then run against two closed-form references with the
viscosity taken from the table alone.

* run: `python benchmarks/verified/lbm_table_selfcheck/run.py [--fast]`
* report stage: `python benchmarks/verified/lbm_table_selfcheck/make_report.py`
* report: [docs/reports/lbm-table-driven-lattice-selfcheck.md](../../../docs/reports/lbm-table-driven-lattice-selfcheck.md)
* gates: A1_tables_pass_exact=PASS, A2_mutation_reject_rate=PASS, A3_equilibrium_moments_exact=PASS, A4_channel_curvature=PASS, A5_wall_offset_and_profile=PASS, A6_taylorgreen=PASS, A7_3d_channel_same_kernels=PASS, L1_literature_table_identity=PASS
* all gates passed: **True**
