# Hot-spot path and elementary fracture qualification

`marine_hotspot_fracture` provides TensorFEM-only structural post-processing.
It samples a scalar component from a selected collinear row of shell nodes or
solid element/integration-point centroids, separates membrane and bending stress
along a through-thickness stress-classification line, performs weld-toe
extrapolation, and evaluates the elementary LEFM relations
`K_I = Y sigma sqrt(pi a)` and `J = K_I^2/E'` against a supplied toughness.

The path extractor accepts `ResultDB` and an explicitly selected step/frame of
`ResultDBv2`. Integration-point arrays are averaged within each element before
centroid-path interpolation. Inputs are finite, typed, unit-explicit SI values;
missing coordinates, incomplete path coverage, off-path clouds, duplicate
projections, and invalid fracture properties fail closed.

## Qualification evidence

The tests use independently written affine stress oracles. Nodal and element-
centroid sampling reproduce that field, while 0.4t/1.0t linear and
0.4t/0.9t/1.4t quadratic extrapolation recover the analytic toe stress. A
linear through-thickness field independently recovers its membrane and signed
surface-bending terms. A
finite-width single-edge crack uses the Tada polynomial, with K and plane-strain
J recomputed directly in the test. Every qualification relative-error gate is
strictly below 3%.

```python
path = sample_result_path(db, "S", start, end, samples=41,
                          location="node", component=0)
toe_stress = hot_spot_extrapolate(path, thickness_m=0.012, method="linear")
gate = mode_i_fracture_gate(toe_stress, crack_m=0.003,
    geometry_factor=1.12, young_pa=210e9,
    toughness_pa_sqrt_m=60e6, poisson=0.3, plane_strain=True)
```

## Scope boundary

This is not general crack-growth finite-element analysis. It does not calculate
K or J from singular elements/domain integrals, choose a crack path, remesh,
model mixed mode or plastic fracture, or establish class-rule acceptance. A
qualified crack-tip FE implementation and experimental correlation are required
before such claims. TensorLBM/CFD is neither imported nor executed.
