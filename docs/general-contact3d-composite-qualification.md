# General 3-D contact composite qualification

The composite gate promotes the exact scope `general_surface_to_surface` only
after four independent, hash-verified artifacts pass: curved frictionless,
curved frictional mesh, curved finite-strain friction, and a true two-deformable
three-dimensional Hertz qualification. The maximum error across their terminal
mesh/oracle metrics must be at most 3%, and every quasi-static path must retain
exact failed-step rollback.

The qualified scope is deliberately bounded to quasi-static, low-order,
faceted, nonmatching, double-deformable surface contact, including friction and
the demonstrated small/finite-strain subsets. Production-scale segmentation
and search remain a stated limitation. Self-contact candidate filtering and
dynamic frictional-impact balance are executed as boundary evidence, but a
self-contact solve and impact integration are separate capabilities and are not
silently included in the quasi-static scope.

The current repository contains the Hertz qualification contract and negative
tests, but no persisted true 3-D Hertz FE run. Therefore the production runner
correctly fails closed until a hash-verified artifact with schema
`tensorfem.hertz-3d-qualification/1.0` is supplied; it must not substitute the
Winkler paraboloid or the analytical Hertz calculator.

Run:

```text
PYTHONPATH=src python scripts/run_general_contact_qualification.py \
  --curved-frictionless .qualification/curved-surface-contact3d/report.json \
  --curved-frictional .qualification/frictional-surface-mesh/report.json \
  --curved-finite-strain-friction .qualification/curved-finite-strain-friction/report.json \
  --hertz-3d PATH/TO/TRUE_HERTZ_3D_REPORT.json \
  --output .qualification/general-contact3d/report.json
```

Missing artifacts, unsupported schemas, hash tampering, inconsistent pass
status, errors above 3%, or missing rollback all fail closed.
