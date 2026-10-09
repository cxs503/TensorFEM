# Nonlinear verification scope

The nonlinear modules are intentionally material-point and truss verification
kernels. They are not advertised as a general nonlinear structural solver.

## Total-Lagrangian truss

The bar uses Green--Lagrange strain
`E = 0.5 * (x.x/L^2 - 1)` and St. Venant--Kirchhoff axial energy. Internal force
and the consistent material-plus-geometric tangent are differentiated from that
energy. A one-bar large-displacement benchmark uses the closed form
`P = EA/2*((1+u/L)^3-(1+u/L))`; the displacement error is below `1e-9` relative.
The tangent is also checked by centered finite differences.

The symmetric two-bar shallow arch is traced by prescribed apex displacement,
which remains well posed through its limit point. Its exact reaction is
`P = EA*(h^2-y^2)*y/(a^2+h^2)^(3/2)`, where `y=h-v`. The analytical first limit
point `y=h/sqrt(3)` is reproduced with less than `0.03%` displacement error and
less than `1e-5%` load error. Ordinary stable branches can be solved with the
incremental load-control Newton solver. Load control is deliberately not claimed
to cross a limit point; displacement control is the documented path strategy.

## Plasticity return mapping

The 1-D associative model uses linear isotropic hardening and backward-Euler
return mapping. Stress, yield consistency, and the exact algorithmic tangent
`E*H/(E+H)` are checked to machine precision.

The 3-D J2 material point accepts full symmetric tensors and uses a radial return
with linear isotropic hardening. A proportional deviatoric loading benchmark
checks the updated von Mises stress against `sigma_y + H*alpha` to machine
precision. Structural plasticity, finite-strain plasticity, kinematic hardening,
and a global nonlinear continuum solver remain out of scope.
