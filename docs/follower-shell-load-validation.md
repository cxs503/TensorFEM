# Configuration-dependent shell pressure

`follower_shell_load` integrates a pressure per current area directly on a
four-node bilinear surface,

\[
  f_a(u)=\int_{-1}^{1}\int_{-1}^{1}N_a p
  (x_{,\xi}\times x_{,\eta})\,d\xi d\eta .
\]

The signed pressure follows the element connectivity.  Translational nodal
forces are consistent; nodal couples are zero.  The returned load tangent is
the exact automatic derivative `df_ext/du`, not a dead-load approximation.
Consequently the global equilibrium tangent used by Newton and Crisfield
continuation is `K_int - lambda*K_load`.  The arc-length implementation also
uses the exact load-column `dR/dlambda = -f_ext(u)`.

## Qualification gates

The automated checks cover an analytic planar resultant and moment, covariance
under a finite rigid rotation and translation, and a central finite-difference
directional derivative.  The derivative tolerance is 3e-6 (well inside the
requested 3 percent).  Existing arc-length regression tests guard rollback,
restart, and traversal of an algorithmic limit point.  A separate analytic
one-degree-of-freedom gate, `R=u-u^3-lambda(1+0.2u)`, checks that a genuinely
configuration-dependent load crosses its maximum-load point while maintaining
equilibrium.  This is explicitly an algorithm test, not a shell benchmark. The adapter
`general_shell_pressure_arc_problem` connects this operator to multi-element
general-shell assembly without treating pressure as a fixed reference vector.

## Deliberate claim boundary

This qualifies the pressure residual, its consistent tangent, and their
continuation interface.  It does **not** qualify a public shell snap-through
benchmark.  The reviewed Karatas--Yuksel spherical-shell paper (Teknik Dergi
31, 2020, DOI 10.18400/tekderg.565095) supplies geometry in the text but its
critical path values are graphical rather than a reproducible numerical table.
No values have been digitized or invented.  A benchmark claim remains blocked
until an authoritative tabulation (including material, loading convention,
boundary conditions, and reported limit points) is available.

Run the focused gate with warnings promoted to errors:

```bash
PYTHONWARNINGS=error pytest -q tests/test_follower_shell_load.py tests/test_arc_length_shell.py
```
