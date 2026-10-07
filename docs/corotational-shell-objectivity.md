# Corotational cylindrical-shell objectivity gate

The original cylindrical Shell4 is a small-rotation stiffness and retains a
nonzero rigid-translation energy on finite angular patches. It remains
experimental. `corotational_shell` adds a separate finite-rotation kinematic
wrapper: a proper Kabsch rotation is extracted from current nodal positions,
translation and that rotation are removed, and nodal orientation matrices are
mapped into the corotated frame before the existing material stiffness is
evaluated.

The strict gate covers rigid rotations of 5, 30, 90, 150 and 179 degrees with
an arbitrary translation. Normalized energy must be below `1e-27`, and the
generalized deformation norm below `2e-14`. A non-rigid state is also subjected
to a superposed 74.5-degree rotation and translation; energy and generalized
deformation must remain invariant to `2e-11`. Invalid, reflected or scaled
orientation matrices fail closed.

This qualifies the **element kinematic objectivity measure only**. It does not
promote the curved shell to stable status: the tangent is not yet consistently
linearized through the polar/SVD frame, large-strain constitutive updates are
not included, and Scordelis--Lo is still the only classical curved-shell
response below 3%. Pinched-cylinder and hemispherical-shell qualification,
path-following and consistent nonlinear internal forces remain required.
