# Initial-field transfer

`tensorfem.initial_fields` transfers measured or simulated fabrication fields
from ResultDB into a target structural model without assuming row order. Node
and element IDs are preferred. Nodes missing a matching ID can be matched by
global Cartesian coordinates with an explicit tolerance; callers select a
fail-closed, zero-fill, or nearest policy.

The contract supports three-component nodal `initial_imperfection` and
six-component `(xx, yy, zz, xy, yz, xz)` `residual_stress` at nodes, elements,
or ResultDB v2 integration points. Units and component layouts are checked but
not silently converted. `assert_self_equilibrated` evaluates weighted
component resultants; callers should supply tributary areas or volumes for a
physical balance check.

`solve_imported_elastic_strip` is an executable reduced structural solve: the
piecewise profile gradient enters geometric membrane strain and every nodal
axial residual-stress value enters equilibrium. Changing the imported spatial
distribution therefore changes the response; the field is not collapsed to a
scalar amplitude. `imperfect_plastic_strip_field_path` provides the analogous
incremental bilinear-plastic workflow.

`initial_fields_result_db` writes the mapped state to a ResultDB v2 initial
frame, retaining integration-point data for restart and audit. The
`imperfect_strip_inputs` adapter passes every nodal imperfection and axial
residual-stress value to reduced strip workflows. Projection from element or
integration-point fields to nodes remains solver-specific and is deliberately
not guessed.

Run `python examples/marine/initial_field_transfer.py` and
`PYTHONWARNINGS=error pytest -q tests/test_initial_fields.py`.
