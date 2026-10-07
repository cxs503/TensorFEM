# Double-deformable 3-D contact qualification

This benchmark closes one global equilibrium path with both sides compliant.
Three master TRI3 nodes and one slave quadrature node have independent springs.
Equal and opposite loads close an initial gap while contact participates in the
same Newton solve.

The independent reaction oracle is `C=1/ks+1/(3km)` and
`R=kn*(C*F-g)/(1+kn*C)`. Gates cover error below 3%, contact force and moment
balance, finite-rotation objectivity, TRI3 mortar role interchange, and exact
rollback after a deliberately failed increment.

Run `PYTHONPATH=src pytest -q tests/test_double_deformable_contact3d.py` and
`PYTHONPATH=src python examples/contact/double_deformable_contact3d.py`.

This qualifies frictionless planar node-to-TRI3 penalty contact between two
linearly compliant bodies with current-configuration search. It does not
qualify general double-sided mortar, curved/deforming patches, self-contact,
frictional large sliding, finite-strain solids, impact, or production search.
