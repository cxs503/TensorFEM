# Layered Shell4 integration-point plasticity

`tensorfem.layered_shell4_plasticity` provides a real, multi-element material
integration path for a flat-facet Shell4.  It is a deliberately bounded
small-strain capability, not the earlier reduced plated-strip model and not a
section-level closed-form wrapper.

## Discretization and state

Each element uses a 2x2 in-plane quadrature rule.  At every in-plane point the
membrane strain and curvature are mapped to the midpoint of every thickness
layer,

`epsilon(z) = epsilon_0 + z kappa`.

Every `(element, in-plane point, layer)` owns an independent three-dimensional
J2 plastic state.  A scalar local condensation enforces `sigma_zz = 0`; the
returned 3x3 in-plane tangent differentiates the same condensed return map.
Isotropic hardening is supported.  Mindlin transverse shear and drilling
stabilization remain elastic.

An optional `[element, 4, layer, 3]` residual-stress field is converted to an
initial compatible elastic strain before return mapping.  Consequently it is
present at zero applied displacement and participates in subsequent yielding.
The caller is responsible for supplying a self-equilibrated field; a
`[+a,-2a,+a]` three-layer pattern is used by the verification test to establish
zero initial nodal force and moment.

Global Newton iterations always integrate from the immutable committed state.
Material histories are returned only with a converged equilibrium point and
are committed between accepted load factors.  A failed increment raises
without mutating the input state.

## Verification evidence

`tests/test_layered_shell4_plasticity.py` covers:

- a uniaxial plane-stress J2 oracle including hardening and tangent comparison;
- pure membrane strain and pure curvature through the actual Shell4 B matrix;
- a self-equilibrated initial residual-stress field;
- a two-element membrane, global force balance, yield, unloading and permanent
  deformation;
- rejected-step material-history rollback; and
- systematic midpoint thickness-integration convergence in elastic bending.

Run the focused evidence with:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_layered_shell4_plasticity.py
PYTHONPATH=src .venv/bin/python examples/marine/layered_shell4_plasticity.py
```

## Qualification boundary

This qualifies small-strain, flat-facet Shell4 membrane/bending plasticity with
plane-stress J2 material points.  It does **not** qualify finite-rotation
plasticity, geometric stiffness, follower loading, arc-length continuation,
branch switching, transverse-shear yielding, anisotropy, damage, contact, or
general shell post-buckling.  Connecting these material states to the
corotational/arc-length path requires an objective finite-rotation stress and
history update and is intentionally left outside this evidence.

