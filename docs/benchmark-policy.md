# Benchmark admission policy

TensorFEM does not advertise a numerical capability merely because a case runs.
Every supported element, solver, and analysis type must include a reproducible
benchmark with:

1. geometry, mesh and element formulation;
2. material parameters, units, loads and boundary conditions;
3. an analytical value or a traceable published benchmark reference;
4. the measured quantity and an unambiguous sampling location;
5. the relative-error definition;
6. an automated fail-closed test; and
7. a mesh-convergence check where no closed-form discrete solution exists.

The release gate is

```text
abs(computed - reference) / abs(reference) < tolerance <= 0.03
```

Patch tests use a tighter absolute/relative numerical tolerance. A benchmark
that misses the gate remains experimental and must not appear in the supported
capability matrix. Results obtained by tuning a reference value to match the
implementation are invalid.
