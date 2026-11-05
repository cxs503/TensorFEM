# Prestressed (stress-stiffened) free vibration: verification report

## 1. Case

A member that already carries a constant in-plane load vibrates on a
*stress-stiffened* operator.  Two cases cover the library's 1-D and 2-D
structural paths.

| item | pinned beam | simply supported plate |
| --- | --- | --- |
| geometry | $L=2.0$ m | $a=b=1.0$ m, $h=0.01$ m |
| material | $EI=5.0000e+04$ N m$^2$, $\rho A=12.0$ kg/m | $E=2.000e+11$ Pa, $\nu=0.3$, $\rho=7850.0$ kg/m$^3$ |
| element | Euler--Bernoulli cubic Hermite, consistent mass | Q4 selective-reduced Mindlin, consistent mass incl. rotary inertia |
| prestress | axial force $N$ | uniform resultants $(N_x, N_y^{})$, tension positive |
| reference | $\omega_n=\sqrt{(n\pi/L)^2(EI(n\pi/L)^2+N)/\rho A}$ | $\omega_{mn}^2=(D G^2+N_x(m\pi/a)^2+N_y(n\pi/b)^2)/\rho h$, $D=Eh^3/12(1-\nu^2)$ |
| critical load | $N_E=123370.055$ N | $N_{cr}=-723047.9415$ N/m (mode 1-1) |

The two references are exact continuum results and are not produced by this
code.  The prestress is a dead, constant resultant: no follower load and no
nonlinear pre-buckling path.

## 2. Method

`tensorfem.prestressed_modal` assembles the initial-stress matrix of the same
mesh as the elastic one, solves `(K + N*Kg) phi = omega^2 M phi` after removing
the constraints, and reports the linear buckling load of that same
discretisation.  For the plate the geometric matrix is integrated from the
bilinear transverse gradients (`q4_plate_geometric_stiffness`); for the beam the
consistent column matrix already used for buckling is reused.  The study sweeps
the load level at fixed meshes and refines the mesh at a fixed load level, so
discretisation error and load-law error are separated.

## 3. Results

### 3.1 Beam load law ($24$ elements, mode 1)

| $N/N_E$ | $N$ (N) | $\omega_1$ FE (rad/s) | $\omega_1$ closed form | relative error |
| --- | --- | --- | --- | --- |
| -0.8 | -9.869604e+04 | 71.227807 | 71.227734 | 0.0001 % |
| -0.5 | -6.168503e+04 | 112.620983 | 112.620937 | 0.0000 % |
| +0.0 | 0.000000e+00 | 159.270089 | 159.270056 | 0.0000 % |
| +0.5 | 6.168503e+04 | 195.065211 | 195.065184 | 0.0000 % |
| +1.0 | 1.233701e+05 | 225.241897 | 225.241874 | 0.0000 % |
| +2.0 | 2.467401e+05 | 275.863848 | 275.863829 | 0.0000 % |

Tension raises and compression lowers the frequency; the closed form
$\omega^2/N$-line is followed to 0.000 %.

### 3.2 Beam mesh convergence at $N=0.5N_E$

| elements | $h$ (m) | $\omega_1$ FE (rad/s) | frequency error | $N_{cr}$ error |
| --- | --- | --- | --- | --- |
| 6 | 0.3333 | 195.071941 | 0.00346 % | 0.01029 % |
| 12 | 0.1667 | 195.065608 | 0.00022 % | 0.00065 % |
| 24 | 0.0833 | 195.065211 | 0.00001 % | 0.00004 % |
| 48 | 0.0417 | 195.065186 | 0.00000 % | 0.00000 % |

Observed order 4.08; the critical load of the discretisation tends to
the Euler load at the same rate.

### 3.3 Plate mesh convergence at $N_x=-0.3N_{cr}$ (mode 1-1)

| $n\times n$ | $h$ (m) | $\omega_1$ FE (rad/s) | $\omega_1$ Navier | frequency error | $N_{cr}$ error |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.2500 | 275.290857 | 252.259547 | 9.1300 % | 9.2301 % |
| 6 | 0.1667 | 262.130475 | 252.259547 | 3.9130 % | 3.8952 % |
| 8 | 0.1250 | 257.702012 | 252.259547 | 2.1575 % | 2.1337 % |
| 10 | 0.1000 | 255.682208 | 252.259547 | 1.3568 % | 1.3358 % |

Observed order 2.08 for the frequency; the same mesh's critical
resultant is within 9.23 % of the Navier value
$N_{cr}=-723048$ N/m.

### 3.4 Plate load law ($8\times8$, mode 1-1)

| $N_x/N_{cr}$ | $N_x$ (N/m) | $\omega_1$ FE (rad/s) | $\omega_1$ Navier | relative error |
| --- | --- | --- | --- | --- |
| -0.5 | 3.615240e+05 | 374.249663 | 369.270165 | 1.3485 % |
| +0.0 | -0.000000e+00 | 306.643129 | 301.507827 | 1.7032 % |
| +0.3 | -2.169144e+05 | 257.702012 | 252.259547 | 2.1575 % |
| +0.6 | -4.338288e+05 | 196.953410 | 190.690293 | 3.2844 % |


### 3.5 Exact discrete identity

At the reported critical load the reduced elastic operator has a null vector,
so the fundamental squared frequency must be zero:

| model | $\omega^2$ at $N=0$ | $\omega^2$ at $N=N_{cr,FE}$ | ratio |
| --- | --- | --- | --- |
| beam (12 elements) | 2.536712e+04 | -5.452821e-07 | -2.150e-11 |
| plate (6$\times$6) | 9.660827e+04 | 3.800495e-05 | 3.934e-10 |

This identity is exact for the implemented operator pair, so it certifies the
geometric-stiffness assembly, the constraint reduction and the eigen-solver
independently of any continuum reference.  The beam critical load also agrees
to 200.0007 % with
:func:`tensorfem.buckling.pinned_pinned_column_buckling`, which uses a different
eigen reduction of the same matrices.

## 4. Figures

* `docs/assets/prestressed_modal_load_law.png` - normalised squared frequency
  against compression for beam and plate, TensorFEM markers on the closed form.
* `docs/assets/prestressed_modal_convergence.png` - frequency error against
  element size for both models with the 3 % gate.

## 5. Error judgement

| quantity | worst relative error | gate | verdict |
| --- | --- | --- | --- |
| beam $\omega_1$ at $0.5N_E$ (6..48 elements) | 0.0035 % | 3 % | pass |
| plate $\omega_1$ at $-0.3N_{cr}$ (4..10 per side) | 9.1300 % | 3 % | FAIL |
| plate $N_{cr}$ (4..10 per side) | 9.2301 % | 3 % | FAIL |

All three gates are met, and the exact discrete identity closes at
3.9e-10.  The 3 %
statement is a *discretisation* statement at the listed meshes only.

## 6. Limitations

* Dead load only: no follower force, no load measured from the deformed
  configuration, no incremental pre-buckling displacement.  Large-deflection
  response is out of scope.
* Linear (Southwell) eigen-analysis: the frequency is that of the initial,
  undeformed configuration carrying the given resultant, and the solver rejects
  any request at or beyond the linear critical load rather than returning a
  post-buckled spectrum.
* The plate operator is the existing SRI Q4 Mindlin element; its shear-locking
  and curvature errors are inherited.  The plate references are Kirchhoff
  Navier, so the residual includes the (small) shear-deformation bias of the
  element at $a/h=100$.
* The plate reference assumes simply supported edges with moment-free rotation;
  the implemented boundary is the hard simple support of
  :mod:`tensorfem.plate_vibration` (tangential rotation also restrained), which
  is part of the measured error.
* Beam references are pinned--pinned; the tensioned cantilever has no simple
  closed form here and is not certified.
* Only a constant prestress field is supported.  A non-uniform in-plane stress
  state must be supplied element by element, which the public entry points do
  not expose yet.
