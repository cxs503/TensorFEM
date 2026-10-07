# Nonmatching surface-to-surface contact qualification

Two independently compliant planar patches use unequal QUAD4 meshes. Complete
Newton paths are run for slave/master resolutions 1/2, 2/3 and 3/4 against the
uniform-pressure three-compliance oracle. The qualification checks monotone
mesh trend, force and moment balance and a finite-rotation covariance test.

Master/slave interchange is stronger than an assembly-only comparison: the
two surface discretisations are exchanged, the geometry is reflected to retain
the contact orientation, and a second complete Newton equilibrium is solved.
Failed increments must leave both caller-owned displacement fields unchanged.

Run `PYTHONPATH=src pytest -q tests/test_surface_surface_contact3d.py` and
`PYTHONPATH=src python examples/contact/nonmatching_surface_contact3d.py`.

This qualifies planar nonmatching QUAD4 frictionless penalty-Mortar only.
`general_surface_to_surface` remains **blocked** pending a curved-surface public
benchmark, production segmentation integration and frictional two-pass contact.
