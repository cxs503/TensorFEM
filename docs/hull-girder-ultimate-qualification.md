# Hull-girder ideal-plastic section qualification

This TensorFEM-only benchmark checks moment–curvature integration for a symmetric
rectangular section. It uses 400 equal-area midpoint fibers with
`sigma = clamp(E kappa y, -sigma_y, sigma_y)`. The independent oracle is the
closed-form ideal-elastic-plastic solution:

`M = E I kappa` up to `kappa_y = 2 sigma_y/(E h)`, and
`M = b sigma_y h^2/4 - b sigma_y c^2/3`, `c = sigma_y/(E kappa)`, thereafter.

The qualification records the full M–kappa curve, initial yield, full-plastic
moment, yielded-fiber fraction and accumulated sectional work. Errors must be
strictly below 3%. Moment, yielded fraction and work must be monotone, and every
fiber stress must remain bounded by yield stress. Inputs require finite positive
SI geometry/material values, an even fiber count, and a strictly increasing
non-negative curvature history beginning at zero.

Run `python examples/hull_girder_ultimate_benchmark.py` from an installed or
source-configured environment.

## Capability boundary

This proves a section-fiber ideal-elastic-plastic bending implementation against
an analytic rectangular-section solution. It is **not** a complete hull-girder
progressive-collapse simulation or a classification-society rule assessment.
It excludes plate/stiffener local buckling, initial imperfections, residual
stress, corrosion, unloading/path dependence, fracture and load redistribution
between realistic structural components. It neither calls nor depends on
TensorLBM.
