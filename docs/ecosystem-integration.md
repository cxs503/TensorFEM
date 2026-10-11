# Tensor ecosystem integration

TensorFEM is the specialist finite-element layer for structural and solid-mechanics analysis. Engineering claims must remain tied to executable verification and declared applicability limits.

## Responsibilities

- Own finite-element formulations, sparse linear/nonlinear solution, contact/fracture and structural dynamics.
- Maintain FEM-native mesh/input/output adapters and solver-specific benchmark evidence.
- Export documented fields, reactions, energy measures and artifacts for coupled workflows.
- Distinguish stable, prototype, experimental and physically unvalidated capabilities.

## Boundaries

- **TensorPhysics** coordinates execution, solver lifecycle, multiphysics exchange and provenance; it does not replace FEM kernels.
- **TensorPlatform** owns shared dataset/AI lifecycle, catalog, access control and lineage.
- **TensorDEM** models bonded particles and fragments; DEM/FEM interaction must define contact ownership and avoid double-counting forces or energy.
- **TensorLBM** and **TensorFVM** provide alternative fluid models; loads must be mapped to structural surfaces with documented sign, units, time and conservation checks.

## Coupling and verification rules

- Declare units, coordinate systems, mesh/surface conventions and physical time for every exchanged quantity.
- Separate interface-contract tests from physical-validation benchmarks.
- Record solver revision, configuration, mesh resolution, convergence history, energy/work balance and artifact checksums.
- Passing software tests alone does not certify engineering accuracy.

## Integration path

1. Preserve standalone FEM verification and benchmark workflows.
2. Publish stable adapter methods for configuration, solve/step, reactions, fields and restart where supported.
3. Let TensorPhysics coordinate coupled runs and store a reproducible run manifest.
4. Let TensorPlatform catalog validated outputs and derived training datasets.
5. Promote coupled capabilities only after independent reference comparisons and sensitivity gates pass.

## Repository map

- [TensorPlatform](https://github.com/cxs503/TensorPlatform) — shared data and AI services.
- [TensorPhysics](https://github.com/cxs503/TensorPhysics) — scientific workflow orchestration and multiphysics coordination.
- [TensorLBM](https://github.com/cxs503/TensorLBM) — lattice Boltzmann solver.
- [TensorFVM](https://github.com/cxs503/TensorFVM) — finite-volume solver.
- [TensorFEM](https://github.com/cxs503/TensorFEM) — finite-element solver.
- [TensorDEM](https://github.com/cxs503/TensorDEM) — discrete-element solver.
