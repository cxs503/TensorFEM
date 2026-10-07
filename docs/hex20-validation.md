# HEX20 verification contract

The 20-node serendipity brick uses the conventional eight-corner/twelve-edge
ordering and full `3 x 3 x 3` Gauss integration. Every integration point must
have a positive, non-degenerate Jacobian; inverted or collapsed elements fail
closed instead of returning a result.

Verification covers partition of unity, Kronecker interpolation at all twenty
nodes, conforming edge-node sharing, an affine three-dimensional patch, and a
uniaxial constant-stress solution. The latter two agree with their independent
continuum solutions to machine precision, comfortably below the project 3%
qualification threshold.

The implementation intentionally remains separate from the stable shared API
until performance and distorted-mesh/cantilever convergence evidence has been
reviewed.

## Cantilever A/B benchmark

A `10 x 1 x 1` solid cantilever with a consistent uniform end shear is compared
with the independent Euler bending plus Timoshenko shear-deflection formula.
At `8 x 1 x 1`, HEX20 has `2.412%` relative error and therefore qualifies under
the strict `<3%` gate. The matching fully integrated HEX8 mesh has `43.683%`
error; HEX20 reduces that error by more than an order of magnitude. Coarser
HEX20 meshes (`4 x 1 x 1`: `4.694%`) are retained as convergence evidence but
are not advertised as qualified results.
