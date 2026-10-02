# Finite-sliding node-to-polyline contact

## Scope

`finite_sliding_contact.py` implements a two-dimensional node-to-rigid-polyline
local contact kernel. The closest segment and projection are recomputed in the
current configuration on every increment. The normal law is a robust penalty;
the tangential law is a Coulomb elastic-predictor/plastic-corrector update.

This capability does **not** claim surface-to-surface mortar contact,
self-contact, three-dimensional contact, finite-element Hertz contact, or a
global nonlinear contact solver.

## Verification contracts

1. **Moving projection:** a point crosses a polyline vertex and the global
   closest-segment search returns the independently evaluated new segment and
   local coordinate to machine precision.
2. **30 degree inclined plane:** prescribed penetration gives
   `lambda = k_n delta`; tangential traction follows `tau=-k_t s` during stick
   and `|tau|=mu lambda` during slip. All reported force errors are below 3%.
3. **Opening/closure:** opening removes both normal and tangential traction and
   resets recoverable slip while retaining accumulated dissipation.
4. **Energy:** for one monotonic friction increment, tangential work equals
   recoverable spring energy plus Coulomb dissipation to round-off.

The analytical references are the penalty normal law and classical Coulomb
friction inequality `|tau| <= mu lambda`; the return mapping is independently
checked against these closed-form scalar expressions rather than against
another TensorFEM solver.
