# External anchor — CASE-FEM-JINTEGRAL-EDGE-CRACK

Reference solution carried by the case (verification anchor, not a fitted parameter).

## 1. Stress-intensity handbook factor (Tada, Paris & Irwin, 3rd ed., 2000)
Single-edge-notched tension (SENT) specimen, tension `sigma` remote, crack length `a`, width `W`,
height `H`:

```
K_I = sigma * sqrt(pi a) * F(a/W)
F(a) = 1.12 - 0.231 a + 10.55 a^2 - 21.72 a^3 + 30.39 a^4      (a = a/W)
```

quoted range `a/W <= 0.6`, `H/W >= 1`, stated accuracy +-0.5 %.
Handbook: Tada, H., Paris, P. C., Irwin, G. R., *The Stress Analysis of Cracks Handbook*,
3rd ed., ASME Press, 2000 (DOI 10.1115/1.3601206), SENT chapter.

Independent cross-check used by the implementation (`tada_sent_shape_factor_closed_form`), the
secant form of the same chapter (also Anderson, *Fracture Mechanics*, Table 2.1):

```
F = sqrt( (2/(pi a)) tan(pi a/2) ) * (0.752 + 2.02 a + 0.37 (1 - sin(pi a/2))^3) / cos(pi a/2)
```

At `a/W = 0.3` both forms give `F = 1.659919` (polynomial) vs `1.6617` (secant), i.e. the
reference itself carries ~0.1 % model uncertainty, small compared with the graded 1.98 % FEM
error, so the FEM error cannot be explained away by handbook uncertainty.

## 2. Energy-release / J-integral definition
Rice, J. R., "A path independent integral and the approximate analysis of strain concentration by
notches and cracks", *J. Appl. Mech.* 35 (1968) 379-386: `J = int_C (W n_x - t . du/dx) ds`, and
for plane strain `K_I = sqrt(J E')`, `E' = E/(1 - nu^2)` (plane stress `E' = E`).

## 3. Why this anchor is admissible at L4 (not L5)
The case grades a *numerical implementation* against a closed-form handbook reference at one
geometric point plus a refinement ladder. It adds no new physical result; the far-field/height
convergence stage remains unquantified (CG non-convergence at H/W = 3.0). Hence L4 (benchmark
verification: complete case + error analysis + reproducible) is the honest ceiling.