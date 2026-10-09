# Multi-facet Shell4 transactional state path

`tensorfem.shell4_state_path` connects the existing corotational Shell4
residual and exact autograd Hessian to Crisfield continuation on an assembled
multi-element mesh. A nodal initial-imperfection field changes the reference
geometry directly; it is not replaced by an equivalent load.

The accepted displacement is projected back to every facet and the exact
generalized deformation, elastic resultant, and energy used by the element
law are committed. Trial states are never stored. If an arc-length corrector
is rejected, the returned global and section state therefore remains at the
last accepted equilibrium. Invalid or non-finite imperfection fields and
degenerate imperfect facets fail before solution.

Evidence in `tests/test_shell4_state_path.py` covers:

- a doubly curved, two-facet assembled path;
- free-DOF equilibrium after continuation;
- equality of recovered section energy and assembled element energy;
- a deliberately rejected step with uncontaminated zero committed state;
- fail-closed initial-field validation.

## Qualification boundary

This is a real multi-element geometrically nonlinear discretization, but its
section law remains elastic and quadratic in the corotational deformation.
The one-point state is a generalized element-section state, not a layered
through-thickness Gauss-point state. Residual stress, plastic return mapping,
hardening, consistent elastoplastic tangent, follower loading, shell
postbuckling and branch switching are **not qualified** by this module. The
transaction boundary is intentionally explicit so those history variables can
be added without allowing rejected Newton trials to mutate committed state.

