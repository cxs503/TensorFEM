# Structural mechanics verification face: Mindlin plate bending and structural vibration

*Campaign node:* `P187-W1-N02`  
*Module:* `tensorfem.structural_mindlin_face`  
*Generated:* 2026-11-10T16:19:45Z from `artifacts/structural_mindlin_face/report.json` (every number below is injected from that file)  
*Commit:* `1d0b43490cce98a41145f65c0f0a18199e54b32c`  
*Command:* `python benchmarks/structural_mindlin_vibration_face.py` (OMP threads: 4)

## 1. Case and physical background

A flat steel panel of a ship's deck/bulkhead scale is used as the canonical structural element. Two questions decide whether a finite-element model of such a panel may be trusted, and neither was answered by the existing single-thickness, single-boundary-condition checks:

1. **Thick-panel bending.** At `h/L = 0.2` (ice belt, collision strake, insert plate) Kirchhoff theory under-predicts the deflection because it ignores transverse shear. A Mindlin (first-order shear deformation, FSDT) model must reproduce the shear amplification and must *not* lock as the plate becomes thin.
2. **Spectral response for real edge conditions.** Panel frequencies are the entry quantity of any vibro-acoustic or fatigue screening. A real stiffened panel is neither fully simply supported nor fully clamped, and free edges are the rule, not the exception.

| quantity | value |
|---|---|
| plate | 1.0 m x 1.0 m |
| Young's modulus | 200.0 GPa |
| Poisson ratio | 0.3 |
| density | 7850 kg/m^3 |
| uniform pressure | 10.0 kPa |
| thickness sweep | 0.005, 0.01, 0.02, 0.05, 0.2 (h/L) |
| element | 4-node Mindlin Q4, 3 DOF/node (w, theta_x, theta_y) |
| shear treatments | SRI (1-point), MITC4 (Dvorkin-Bathe), full 2x2 control |
| boundary conditions | SS (hard simple), CC (clamped), CF (cantilever) |

## 2. Method and tool chain

**Static half.** Reference solutions are closed form and independent of the code: the Kirchhoff Navier double-sine solution and the FSDT Navier solution of the same uniformly loaded simply supported plate, which adds the shear particular solution `q_mn / (k_s G h R)` while leaving the bending moments unchanged. The FE side assembles the Q4 Mindlin element with three shear treatments; the 2x2 shear variant is deliberately kept as a locking control, i.e. a variant that *must* fail so that a pass is meaningful.

**Vibration half.** The plate eigenproblem `K phi = omega^2 M phi` uses the consistent mass matrix including rotatory inertia. References:

* SS: the exact FSDT (Mindlin) frequency, obtained from the symmetric 3x3 generalised eigenproblem of the amplitudes `(psi_x, psi_y, w)` for each wave pair, so the thick-plate spectrum (`h/L = 0.2`) is also covered;
* CC: a polynomial Rayleigh-Ritz solution built from the exact energy functional, a discretisation that shares no algebra with the FE assembly. It reproduces Leissa's clamped-square-plate coefficients (35.9853, 73.394, 108.2, 131.6, 132.2) to better than 0.01 %;
* CF: no closed form exists, so a three-mesh Richardson extrapolation (refinement ratio 2) with the GCI is used. The extrapolation is labelled as such; no literature value is transcribed.

**Numerical environment notes.** The platform BLAS/LAPACK stack (flexiblas 3.3.0 on loongarch64) is unstable for these matrices: the LAPACK *generalised* symmetric driver either segfaults or returns a corrupt spectrum, and even the standard solver intermittently returns a large negative eigenvalue. The module therefore reduces with an explicit Cholesky factor, validates every candidate spectrum against its eigenpair residual and falls back through several LAPACK drivers.

## 3. Results

### 3.1 Static bending across the thickness range (C1, C2 and the locking control)

| h/L | FSDT/Kirchhoff | SRI error vs Kirchhoff | MITC error vs Kirchhoff | full-integration error vs Kirchhoff |
|---|---|---|---|---|
| 0.005 | 1.0001 | -0.113 % | -0.114 % | -95.754 % |
| 0.01 | 1.0005 | -0.074 % | -0.075 % | -84.920 % |
| 0.02 | 1.0021 | +0.082 % | +0.081 % | -58.345 % |
| 0.05 | 1.0130 | +1.177 % | +1.173 % | -17.168 % |
| 0.2 | 1.2073 | +20.729 % | +20.663 % | +19.281 % |

Error against the *FSDT* reference (the correct thick-plate target) for the same sweep:

| h/L | SRI | MITC | full |
|---|---|---|---|
| 0.005 | -0.126 % | -0.126 % | -95.754 % |
| 0.01 | -0.126 % | -0.126 % | -84.927 % |
| 0.02 | -0.125 % | -0.126 % | -58.431 % |
| 0.05 | -0.117 % | -0.121 % | -18.228 % |
| 0.2 | +0.003 % | -0.052 % | -1.197 % |

* **C1** (PASS): largest deviation from Kirchhoff for `h/L <= 0.05` is +1.177 % (threshold 2 %).
* **C2** (PASS): at `h/L = 0.2` the Mindlin deflection exceeds Kirchhoff by +20.729 %, matching the FSDT closed form to +0.003 %.
* **Locking control:** the full-integration variant loses 95.754 % of the deflection at `h/L = 0.005` where SRI stays within 0.126 %. The error is therefore a property of the shear treatment, not of the mesh.

### 3.2 Mesh convergence (C3)

| elements per side | h [m] | w_FE [m] | FSDT reference [m] | relative error |
|---|---|---|---|---|
| 4 | 0.25000 | 2.168366e-03 | 2.219194e-03 | -2.290 % |
| 8 | 0.12500 | 2.207796e-03 | 2.219194e-03 | -0.514 % |
| 16 | 0.06250 | 2.216395e-03 | 2.219194e-03 | -0.126 % |
| 32 | 0.03125 | 2.218497e-03 | 2.219194e-03 | -0.031 % |

Observed order of accuracy (least-squares slope of `log|error|` vs `log h`): **p = 2.059**, pairwise orders 2.16, 2.03, 2.01. C3 (PASS, threshold p >= 2): the second-order rate proves the error is discretisation error of a consistent element, not a systematic model bias.

### 3.3 Bending stress

| h/L | sample point [m] | M_x FE [N m/m] | M_x Kirchhoff [N m/m] | relative error | bending stress [MPa] |
|---|---|---|---|---|---|
| 0.01 | (0.4868, 0.4868) | 477.0 | 478.3 | -0.266 % | 28.62 |
| 0.05 | (0.4868, 0.4868) | 477.0 | 478.3 | -0.266 % | 1.14 |
| 0.2 | (0.4868, 0.4868) | 477.0 | 478.3 | -0.266 % | 0.07 |

The FE moment is sampled at the Gauss point nearest the centre and compared with the series at the *same* coordinates, so no discretisation offset enters the comparison. Largest deviation: +0.266 %.

### 3.4 Natural frequencies (C4)

First six frequencies of the square plate at `h/L = 0.01`:

| mode | SS FE [Hz] | SS exact Mindlin [Hz] | error | CC FE [Hz] | CC Rayleigh-Ritz [Hz] | error | CF FE [Hz] (finest mesh) | CF extrapolated [Hz] | error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 48.06 | 47.97 | +0.191 % | 87.77 | 87.48 | +0.332 % | 8.44 | 8.44 | +0.031 % |
| 2 | 120.69 | 119.86 | +0.695 % | 180.15 | 178.42 | +0.971 % | 20.67 | 20.66 | +0.072 % |
| 3 | 120.69 | 119.86 | +0.695 % | 180.15 | 178.42 | +0.971 % | 51.92 | 51.72 | +0.387 % |
| 4 | 193.14 | 191.67 | +0.767 % | 265.52 | 263.08 | +0.929 % | 66.25 | 66.07 | +0.273 % |
| 5 | 243.66 | 239.50 | +1.736 % | 326.84 | 319.88 | +2.178 % | 75.42 | 75.15 | +0.352 % |
| 6 | 243.66 | 239.50 | +1.736 % | 328.41 | 321.39 | +2.182 % | 132.03 | 131.47 | +0.427 % |

* **C4** (PASS): largest error against the analytic/variational references for SS and CC is 2.182 % (threshold 3 %).
* CF has no closed form: the FE error against the Richardson extrapolation is at most 0.427 % with a maximum GCI of 0.532 %, i.e. the extrapolated reference is itself uncertain at that level.

Thick-plate spectrum at `h/L = 0.2` (SS), against the exact Mindlin and Kirchhoff closed forms:

| mode | FE [Hz] | exact Mindlin [Hz] | Kirchhoff [Hz] | error vs Mindlin | error vs Kirchhoff |
|---|---|---|---|---|---|
| 1 | 851.51 | 848.36 | 959.73 | +0.371 % | -11.276 % |
| 2 | 1876.77 | 1854.98 | 2399.32 | +1.175 % | -21.779 % |
| 3 | 1876.77 | 1854.98 | 2399.32 | +1.175 % | -21.779 % |
| 4 | 2713.35 | 2681.42 | 3838.92 | +1.191 % | -29.320 % |
| 5 | 3247.89 | 3167.39 | 4798.65 | +2.542 % | -32.316 % |
| 6 | 3247.89 | 3167.39 | 4798.65 | +2.542 % | -32.316 % |

The thick plate is where Kirchhoff theory is simply wrong: it over-predicts the fundamental by the percentage shown in the last column, and the FSDT model reproduces the correct value.

### 3.5 Negative control (C5)

| density [kg/m^3] | f1 [Hz] | f1/f1(rho0) | sqrt(rho) f1 |
|---|---|---|---|
| 7850 | 48.80 | 1.000 | 4324.0 |
| 78.5 | 488.04 | 10.0000 | 4324.0 |
| 0.785 | 4880.40 | 100.0000 | 4324.0 |

C5 (PASS): the fundamental frequency grows exactly as `1/sqrt(rho)` and `sqrt(rho) f1` is invariant to all printed digits, so the mass matrix really is `rho`-proportional and a mass-free (MOCK) implementation would fail here.

### 3.6 Cylindrical shell: an honest gap

The cylindrical shell half was exercised on the standard benchmark geometry (`R = 0.5 m`, `L = 1.0 m`, `t = 0.005 m`, `E = 210 GPa`, `rho = 7800 kg/m^3`):

| FE frequency [Hz] (SS, 10x24 mesh) | DMV closed form, lowest modes [Hz] |
|---|---|
| 0.0, 0.0, 21.7, 22.2, 30.9 | 17 (m=1, n=1), 54 (m=2, n=1), 116 (m=3, n=1), 202 (m=1, n=5), 220 (m=1, n=6) |

**Verdict: NOT_VERIFIED.** the lowest FE frequencies (2.18e-05, 1.71e-03 Hz) are numerically zero: the diaphragm condition constrains only (v, w, rx) on the end rings and leaves the axial translation (and drilling rotation) free, so the first 'modes' are rigid-body modes, not the circumferential bending family whose lowest DMV member is 17 Hz. The first bending-like FE frequency is 21.7 Hz, i.e. 1.253 of the DMV value, and the repository test for this path only bounds the frequencies loosely (10 < f < 500 Hz). The shell half of C4 is therefore reported as an honest gap, not a pass.

The correct fix (a statically determinate support that removes the free axial translation and drilling rotation, plus a per-wave-number comparison with DMV) is recorded as follow-up work rather than papered over.

## 4. Figures

**thickness_sweep.png** — Centre-deflection error against the Kirchhoff closed form for three shear treatments; full integration locks while SRI and MITC stay within C1 for h/L <= 0.05.

![thickness_sweep.png](../../artifacts/structural_mindlin_face/thickness_sweep.png)

**convergence.png** — Centre-deflection error versus mesh size with the fitted observed order of accuracy.

![convergence.png](../../artifacts/structural_mindlin_face/convergence.png)

**frequencies.png** — First six natural frequencies for SS, CC and CF plates with the relative error of every mode.

![frequencies.png](../../artifacts/structural_mindlin_face/frequencies.png)

**deflection_field.png** — FE deflection field of the simply supported plate and the centreline comparison with the Kirchhoff Navier solution.

![deflection_field.png](../../artifacts/structural_mindlin_face/deflection_field.png)

**density_control.png** — Negative control: the fundamental frequency follows the exact 1/sqrt(rho) scaling, so a mass matrix that ignored density would fail this test.

![density_control.png](../../artifacts/structural_mindlin_face/density_control.png)

## 5. Pitfalls, limits and scope

* **Platform BLAS instability.** The LAPACK generalised symmetric driver segfaults on these matrices and the standard solver intermittently returns a large negative eigenvalue. All solves now validate the eigenpair residual and fall back through several drivers; the module raises rather than returning a corrupt spectrum. Runs on this host should not pin `OMP_NUM_THREADS` to 1 (see `tensorfem.plate_vibration`), and 4-16 threads was the stable range here.
* **Hard simple support.** The FE SS condition (`w = 0`, tangential rotation restrained) is exactly the FSDT-Navier SS condition, which is why the thin plate comparison closes to a fraction of a percent; a *soft* simple support would not be comparable to the Navier solution.
* **Rayleigh-Ritz is a Kirchhoff model.** The CC reference is valid for thin plates only; at `h/L = 0.2` the FSDT reference must be used instead. A cantilever (CF) Ritz reference was attempted and discarded: the polynomial and Fourier-cosine product spaces reproduced only the cylindrical-bending family (lambda = 3.516, 22.03, ...), which is a documented negative result, so CF is verified by grid convergence instead.
* **Extrapolated references are not truth.** The CF plate and the shell rows carry an explicit GCI; a `3 %` agreement there means agreement with the extrapolation, whose own uncertainty is printed alongside.
* **Scope.** Linear elasticity, small deflection, uniform pressure, no stiffeners, no added mass, no fluid coupling. Wet-mode and stiffened-panel behaviour remain outside this face.

## 6. Criteria roll-up and error judgement

| criterion | threshold | value | verdict |
|---|---|---|---|
| C1 deflection vs Kirchhoff (h/L <= 0.05) | < 2 % | 1.177 % | PASS |
| C2 h/L = 0.2 Mindlin > Kirchhoff | > 0 | +20.729 % | PASS |
| C3 convergence order | >= 2 | 2.059 | PASS |
| C4 plate frequencies (SS, CC) | < 3 % | 2.182 % | PASS |
| C5 density -> 0 control | monotone + 1/sqrt(rho) | 10.000, 100.000 | PASS |
| C6 end-to-end JSON keys | all present | deflection, stress, natural_frequencies, convergence_rate, boundary_conditions | PASS |

**Is every key quantity below 3 %?**

* centre deflection vs the Kirchhoff closed form: yes, +1.177 % for `h/L <= 0.05`;
* centre deflection vs the FSDT closed form across the whole sweep: yes, worst case 0.126 %;
* bending moment vs the Kirchhoff series at the same point: yes, 0.266 %;
* first six frequencies, SS and CC: yes, +2.182 %;
* thick-plate frequencies at `h/L = 0.2` vs exact Mindlin: yes, 2.542 %;
* CF plate frequencies: no closed form exists; agreement with the Richardson extrapolation is 0.427 % with GCI 0.532 % — this is verification by grid convergence, not against an independent solution;
* cylindrical shell frequencies: **no** — the existing shell path does not reach the DMV spectrum and is reported as an honest gap, not as a number.

## 7. Reproduction

```bash
OMP_NUM_THREADS=4 python benchmarks/structural_mindlin_vibration_face.py
python benchmarks/structural_mindlin_vibration_face_report.py
python -m pytest tests/test_structural_mindlin_face.py -q
```

Wall time of the full face: 33 s (python 3.11.6, numpy 2.4.6, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38).
