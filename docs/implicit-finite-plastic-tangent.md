# Local implicit finite-plastic tangent

The scalar J2 consistency equation is eliminated analytically and PyTorch
differentiates the resulting local return. An eight-point integral matrix
logarithm replaces spectral eigenvector derivatives, remaining smooth at
repeated principal stretches. `dP/dF` is assembled into a TET4 tangent by an
explicit chain rule and checked against central differences plus the existing
spectral return-map oracle.

This is a local, single-return implicit tangent. It is qualified for elastic-
plastic transitions, non-coaxial increments and near-repeated spectra. Large
increments that require adaptive material substeps must still use the AD
substep-tree tangent; this module does not silently extrapolate its local
validity. Matrix-log quadrature error and the plastic active-set switch are the
remaining piecewise-smooth boundaries.
