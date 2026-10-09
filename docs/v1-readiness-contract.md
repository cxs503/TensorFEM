# TensorFEM 1.0 readiness contract

`v1_readiness.evaluate_v1_readiness` is the fail-closed release contract for
the 1.0 milestone. A healthy package build is necessary but not sufficient.
The report requires all of the following before `ready_for_1_0` can be true:

- a difficult nonlinear shell step improved without changing the qualified
  path by more than the declared benchmark tolerance;
- general surface-to-surface double-deformable 3-D contact, not a planar
  node-to-triangle prototype promoted by name;
- a sparse Shell4 path qualified at no fewer than 10,000 active DOFs with a
  measured storage benefit;
- full regression, API audit and offline wheel installation;
- explicit peak and post-peak evidence on 4x4, 8x8 and 12x12 panel meshes,
  with adjacent peak changes no greater than 3%.

Missing evidence produces `blocked`; completed but nonconverged mesh evidence
produces `failed`. The evaluator never extrapolates a peak or treats an API
smoke test as physical qualification. The complete report is SHA-256 covered
so a status cannot be edited independently of its evidence summary.
