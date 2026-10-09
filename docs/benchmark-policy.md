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

## Research report qualification

A scalar check is not a complete FE research report. Published field reports must supply complete finite displacement/stress samples, declared sampling coordinates and reference scope, independently recomputed field errors, force balance, at least three distinct mesh levels, and the corresponding result figures and exports. The final mesh must satisfy every declared quantity below 3%; coarse meshes that fail stay visibly blocked. Zero-reference stress components are included in a tensor norm rather than divided by zero. Beam-theory interior checks do not certify boundary peaks. Uniform patch tests do not certify realistic stiffened-panel buckling or strength.

Use `scripts/rebuild_benchmark_reports.py` to recompute the three published cases and `scripts/validate_published_benchmark_reports.py` to validate their fields and export hashes. Remaining scalar or incomplete cases remain reference-only in the report builder.
