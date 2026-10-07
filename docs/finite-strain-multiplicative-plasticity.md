# Multiplicative finite-strain logarithmic J2 plasticity

The integration-point state is `(Fp, alpha)` and kinematics use `F=Fe Fp`.
Elastic logarithmic strain follows from `Ce=Fe^T Fe`. An associative traceless
Mandel direction is integrated by `Fp[n+1]=exp(delta_gamma N) Fp[n]`; therefore
plastic volume is preserved to roundoff. Isotropic hardening uses backward
radial return, with an exact closed form for coaxial proportional paths.

First Piola stress is recovered from the rotated Kirchhoff stress. The global
TET4 solver always evaluates Newton trials from immutable committed integration
point states, so rejected increments roll back and converged results restart.

Qualification covers superposed 1.1-radian rigid rotation, `det(Fp)=1`, yield
consistency, nonnegative dissipation, analytical isochoric finite uniaxial
extension, residual stress on unloading, global Newton convergence and restart.

## Theory boundary

This is a finite-strain multiplicative model, not small-strain J2 relabelled.
The exponential update is exact for the qualified coaxial path. General
non-coaxial loading uses the stated intermediate logarithmic-strain algorithm
but is not yet broadly benchmarked. The global tangent is a centred numerical
algorithmic tangent and is independently difference-checked; a closed-form
consistent tangent, substepping error estimator and non-proportional benchmark
suite remain explicit gaps.
