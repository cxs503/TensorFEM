# Industrial P0 phase-3 qualification boundary

`industrial_p0_phase3_qualification` is the executable, fail-closed view of
the current panel-ultimate and Hertz work. It deliberately separates three
claims:

| Capability | Status | Evidence or blocker |
|---|---|---|
| Axisymmetric rigid-indenter/deformable-half-space Hertz | qualified | Q4 elasticity and contact solve on 16x16, 24x24 and 48x48 meshes; all designated fine-mesh errors are below 3% |
| Finite-rotation layered Shell4 arc integration | qualified primitive | accepted-point equilibrium plus rejected-corrector J2 history rollback |
| General 3-D deformable-to-deformable Hertz contact | blocked | no 3-D sphere/half-space solve, pressure integration and mesh/domain convergence evidence |
| Marine panel peak and post-peak strength | blocked | real Shell4 meshes, initial fields and a transactional arc driver exist, but the 4/8/12 mesh and two-step-control result matrix has not been executed |

The Hertz qualification covers load, fitted contact radius, fitted peak
pressure, pressure-distribution weighted L2 error and normalized penalty
overlap. Each is strictly below `0.03` on the 48x48 mesh and decreases over
the declared sequence. The independently recovered support reaction balances
the integrated contact force below `1e-10`. Prescribed indenter travel is not
misrepresented as a solved accuracy metric.

The panel report records the actual 4x4, 8x8 and 12x12 Shell4 input meshes,
two requested continuation controls, analytical bounds and acceptance
thresholds. It records no invented peak or post-peak values. The generic arc
driver's accepted equilibrium and rejected-step rollback are independently
qualified here; panel promotion still requires actual peak/post-peak runs,
mesh and arc-step convergence, equilibrium, and energy audits.

Both evidence sections and the canonical complete report have SHA-256 hashes.
Validation recomputes all three hashes and rejects attempts to promote either
blocked claim.

```bash
PYTHONWARNINGS='error,ignore:Failed to initialize NumPy' PYTHONPATH=src \
  python -m pytest -q tests/test_industrial_p0_phase3_qualification.py
```
