# Numerical crack-tip J-integral validation

`crack_tip_j_integral` evaluates the two-dimensional Rice contour integral
directly from ordered, discrete stress and displacement-gradient samples.  It
does not infer J from K.  Polygon geometry supplies the outward normal and a
periodic trapezoidal rule integrates the energy and traction-work fluxes.

The verification fixture samples the leading Williams mode-I displacement and
stress fields on several circular and smoothly perturbed contours. Displacement
gradients are recovered by centered Cartesian differences. Tests require:

- agreement with the independent `K_I^2/E'` identity within 0.3%;
- spread below 0.5% over three radii/path shapes; and
- systematic angular-quadrature convergence from 32 to 256 points.

The accepted scope is homogeneous isotropic, small-strain 2-D linear elasticity
under mode-I loading, plane stress or plane strain, with no body force or
material interface enclosed by the contour. The supplied field must use local
coordinates with crack extension in positive x. A production analysis must
recover stress and displacement gradients from its finite-element solution and
select contours outside the singular element ring. Plastic J, mixed-mode
interaction integrals, dynamics, crack-front 3-D integration, automatic crack
growth, remeshing and toughness acceptance remain outside the qualified scope.
