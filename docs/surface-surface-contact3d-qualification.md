# Multi-point surface-to-surface contact qualification

Matching planar QUAD4 patches are coupled by four contact quadrature points per
facet. Both surfaces are independently Winkler-compliant and participate in a
single Newton equilibrium. The independent uniform-pressure oracle treats the
two foundations and penalty layer as three compliances in series.

The 1x1, 2x2 and 3x3 sequences exercise 4, 16 and 36 contact points. Gates
cover pressure error below 3%, force and moment balance, finite rigid-motion
objectivity, master/slave exchange and exact rollback after a failed increment.

Run `PYTHONPATH=src pytest -q tests/test_surface_surface_contact3d.py` and
`PYTHONPATH=src python examples/contact/surface_surface_contact3d.py`.

This is a planar, matching QUAD4, frictionless penalty-Mortar qualification.
It does not qualify curved Hertz contact, production nonmatching mortar,
friction, self-contact, finite-strain solids or impact.
