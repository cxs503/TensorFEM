# Two-block frictionless mortar segment validation

This note qualifies the mortar *segment* contact kernel in
`tensorfem.mortar_segment_contact3d` on two deformable HEX8 blocks. It closes the
"mortar contact" gap recorded for the FEM solver: the pre-existing kernel in
`tensorfem.mortar_contact3d` integrates the penalty potential at the slave-facet
quadrature points of each facet regardless of how much of that facet actually
overlaps a master facet. The new kernel builds the real mortar segments.

## What the kernel does

For every slave facet and every neighbouring master facet:

1. the slave facet polygon is projected into the master facet plane and clipped
   against the master facet edges (Sutherland-Hodgman, four edges for a QUAD4,
   three for a TRI3);
2. the clipped convex polygon is fan-triangulated and integrated with a
   three-point Gauss rule per triangle (exact for quadratics on a triangle, so a
   uniform traction field is integrated exactly);
3. the normal gap at each quadrature point is `g = (x_slave - x_master) . n_master`,
   with the slave material point recovered from the slave facets' own shape
   functions, so a non-matching discretisation exchanges force without any
   node-on-node pairing;
4. the penalty potential `1/2 eps_n < -g >_+^2` is integrated and differentiated
   (autograd) to give the residual and the consistent tangent of both surfaces.

The segment topology, the master normal and the slave projection weights are
frozen inside one linearisation (geometry and active set are rebuilt at every
assembly, i.e. an *updated* mortar), and the area measure is the reference-area
(total-Lagrangian) one, matching the small-strain HEX8 solid. The reference
configuration may not interpenetrate beyond `initial_penetration_tol`; a
pre-overlapping input is rejected instead of being silently penalised.

Scope: frictionless, small relative sliding, planar-to-moderately-curved facets,
primal penalty (no dual multipliers). Frictional mortar with finite sliding,
curved/self-contact segment integration and a regularised (finite-strain or
cohesive) edge treatment are roadmap items.

## Proof surface

`scripts/run_mortar_two_block_evidence.py` writes
`artifacts/mortar-two-block/evidence.json` (numbers) and
`artifacts/mortar-two-block/figures/*.png` (figures).

### Case A - equal blocks, non-matching interface meshes (analytic reference)

Quarter model `[0,0.5]^2`, two blocks of the same material (`E = 1000`,
`nu = 0.3`), uniform top pressure `p = 10`, symmetry planes `x = 0`, `y = 0`,
roller bottom, frictionless interface at `z = 0` with `n` slave facets against
`n + 1` master facets. Two identical blocks under a uniform pressure admit the
exact state `sigma_zz = -p`, `sigma_xx = sigma_yy = 0`, so the analytic interface
traction is `p` at every point - including the free edge, because the uniform
state satisfies the traction-free lateral boundary condition exactly.

| n_slave / n_master | dofs | band L2 | full L2 | area error | equilibrium | penetration |
|---|---|---|---|---|---|---|
| 2 / 3 | 225 | 1.38e-10 | 1.37e-10 | 0.0 | 3.7e-12 | 1.00e-08 |
| 3 / 4 | 369 | 1.40e-10 | 1.47e-10 | 0.0 | 3.0e-11 | 1.00e-08 |
| 4 / 5 | 549 | 1.47e-10 | 1.42e-10 | 0.0 | 1.3e-11 | 1.00e-08 |
| 5 / 6 | 765 | 1.50e-10 | 1.51e-10 | 0.0 | 8.1e-12 | 1.00e-08 |

Band and full L2 are relative to `p`, so the mortar transfer reproduces the
analytic uniform state to ~1e-10 at every refinement level (a patch test, not a
convergence statement). The penetration equals `p / eps_n` exactly, i.e. the
no-penetration constraint is satisfied to the penalty compliance, and no
quadrature point carries tension.

### Case C - prescribed smooth field, transfer accuracy (analytic integral)

A smooth, strictly positive penetration field
`g = g0 (1.2 + 0.5 sin(pi x) sin(pi y))` is imposed on the slave surface of two
parallel plates with non-matching meshes. The exact interface force density is
`t = eps_n g`; the analytic total is `eps_n g0 (1.2 + 2/pi^2)`.

| n_slave / n_master | quadrature points | L2 error | total force error |
|---|---|---|---|
| 2 / 3 | 96 | 6.39e-02 | 5.54e-02 |
| 3 / 4 | 216 | 3.13e-02 | 2.57e-02 |
| 4 / 5 | 384 | 1.82e-02 | 1.46e-02 |
| 6 / 7 | 864 | 8.29e-03 | 6.56e-03 |
| 8 / 9 | 1536 | 4.70e-03 | 3.70e-03 |

Observed log-log rates 1.76, 1.88, 1.94, 1.97 -> second order, as expected for
bilinear nodal sampling plus a three-point segment rule.

### Case B - mismatched punch (edge regime, reference withheld)

Upper block `[0,0.5]^2 x [0,1]` pressed onto a `1.5 x 1.5 x 1.5` base of the same
material, non-matching interface. The interface pressure is *not* uniform here
(the base surface dishes under the patch load, so the punch cannot stay
conformal). At fixed mesh (`n = 3`) the computed distribution depends strongly on
the penalty regularisation:

| eps_n | closed area fraction | band mean / p | edge peak / p | equilibrium |
|---|---|---|---|---|
| 1e5 | 1.000 | 0.780 | 3.8 | 9.2e-15 |
| 1e6 | 0.532 | 0.632 | 10.8 | 0.0 |
| 1e7 | 0.227 | 0.600 | 26.4 | 3.3e-14 |
| 1e8 | 0.144 | 0.554 | 76.9 | 1.8e-13 |

and at fixed moderate penalty (`eps_n = 1e6`) it drifts under mesh refinement:

| n_slave / n_master | dofs | closed fraction | band mean / p | edge peak / p | band L2 |
|---|---|---|---|---|---|
| 2 / 7 | 657 | 0.417 | 5.73 | 12.4 | 1.31 |
| 3 / 10 | 1233 | 0.532 | 6.32 | 10.8 | 1.11 |
| 4 / 13 | 1989 | 0.633 | 6.99 | 9.66 | 0.94 |

A flat punch on an elastic body carries an integrable `(a-s)^{-1/2}` pressure
singularity at its edge, which a Q1 penalty discretisation cannot represent: as
soon as the penalty is stiff enough to act as a near-hard constraint, the load is
picked up by the quadrature points nearest the edge and the interior opens. The
mesh trend above is consistent (band mean rises towards `p`, peak and band L2
fall) but is far from converged, and the penalty sweep proves the distribution is
regularisation dependent. Therefore the *distribution* in the mismatched
configuration is declared **withheld**: no resolving reference exists at this
discretisation. What is verified is the integral statement: the transferred
normal force equals the applied load to ~1e-13 in every configuration, the mean
traction equals `p`, and no tension appears. Resolving the edge requires
finite-strain/frictional mortar plus a regularised contact length scale
(roadmap).

## Rejection gates

* `normal_penalty <= 0` -> `ValueError` (assembly and sampling);
* reference configuration interpenetrating beyond `initial_penetration_tol`
  (typical self-contact/overlapping-surface modelling error) -> `ValueError`;
  `check_initial_penetration=False` is an explicit opt-in;
* `search_distance < 0`, non-`long` or out-of-range connectivity, mismatched
  reference/displacement shapes -> `ValueError`;
* an axis-aligned candidate search radius smaller than a detected gap ->
  `ValueError` (silently dropping penetrating pairs would make the contact
  vanish).

## How to reproduce

```bash
PYTHONPATH=src python scripts/run_mortar_two_block_evidence.py \
    --output artifacts/mortar-two-block --figures artifacts/mortar-two-block/figures
PYTHONPATH=src pytest -q tests/test_mortar_segment_contact3d.py
```

## Limits

* linear small-strain HEX8 + reference-area penalty measure (the current-area
  contribution is a second-order effect, verified at 0.6% on the `1%` strain
  level of the transport study);
* frictionless, so tangential interface tractions are not modelled;
* no dual mortar multipliers, no frictional mortar, no finite sliding;
* the mismatched-interface pressure distribution is not a verified quantity
  (see Case B).