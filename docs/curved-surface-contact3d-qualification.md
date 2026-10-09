# Curved nonmatching surface-contact qualification

A shallow faceted parabolic cap contacts an independently compliant flat
surface. Slave/master grids are deliberately unequal. Both surfaces and the
penalty layer act as local springs in series, giving effective stiffness
`keff=1/(1/kn+1/ks+1/km)`. Axisymmetric integration of the parabolic overlap
gives the independent Winkler sphere oracle `F=pi*keff*R*delta^2`.

Three complete Newton paths check mesh trend and a below-3% fine result.
Additional gates cover force/moment balance, finite rotation covariance, a
second complete solve after exchanging surface roles, and exact rollback.

The symmetric two-pass 4/5 path builds two full automatic-differentiation
Hessians per Newton iteration and is intentionally opt-in. The measured
one-pass bracketing runs give 50.6345 and 48.9069 against the 49.7678 oracle
(1.742% and 1.730%); their raw one-pass role bias is 3.412%, which is why the
qualified implementation assembles both directional potentials and tangents
in one symmetric Newton system instead of claiming either biased pass.

Default regression runs a 2/3 two-pass construction, active-contact, balance
and rollback smoke gate:

`PYTHONPATH=src pytest -q tests/test_surface_surface_contact3d.py`

Run the full three-grid oracle/objectivity/interchange qualification explicitly:

`TENSORFEM_RUN_SLOW_CURVED_CONTACT=1 PYTHONPATH=src pytest -q tests/test_surface_surface_contact3d.py -k curved_nonmatching_complete`

or `PYTHONPATH=src python examples/contact/curved_surface_contact3d.py`.

For an atomic machine-readable report suitable for a detached job, use:

`TENSORFEM_RUN_SLOW_CURVED_CONTACT=1 PYTHONPATH=src python scripts/run_curved_contact_qualification.py`

The runner writes `.qualification/curved-surface-contact3d/report.json` only
after every fail-closed gate has passed.

This qualifies a curved, nonmatching, frictionless, double-deformable subset.
It is a Hertz-type sphere geometry with Winkler compliance, not classical
Hertz elastic halfspaces. Frictional two-pass contact, self-contact,
finite-strain solids and impact remain outside the claim.
