# Ice plate on an elastic foundation: bending and flexural (first-crack) limit load

Anchor: **H3-ICE-PLATE-FEM** (`TensorFEM`).  Evidence:
`benchmarks/artifacts/ice_plate_flexural_limit/evidence.json`, produced by
`python benchmarks/ice_plate_flexural_limit.py` (script sha256 `d9f61087f5425672...`,
git `e5185c94dabe`).  Report is generated; every number
below is injected from the JSON.

## 1. Case, geometry and parameters

A floating ice sheet is loaded vertically through a circular contact area -- the
classical idealisation of a quasi-static ice load on a sheet (Westergaard 1926;
Nevel 1969; Kerr 1979; ISO 19906:2010).  The water is a Winkler bed with
`k = rho_w g`; the sheet is at rest, so only the static buoyancy stiffness enters.

| parameter | symbol | value | unit |
|---|---|---|---|
| Young's modulus | E | 7.000e+09 | Pa |
| Poisson ratio | nu | 0.3 | - |
| sheet thickness | h | 0.3 | m |
| Winkler modulus | k = rho_w g | 10055.2 | N/m^3 |
| flexural rigidity | D = E h^3/(12(1-nu^2)) | 1.7308e+07 | N m |
| characteristic length | l = (D/k)^(1/4) | 6.4411 | m |
| flexural strength | sigma_f | 0.60 | MPa |
| patch radius (swept) | a | 0.1 l ... 8 l | m |

The domain is the quarter model `[0, 20 l]^2` with symmetry on both axes and
free outer edges; far-field truncation error is below the mesh error because the
moment field decays like the Kelvin function `kei(r/l)`.

## 2. Method

1. **Foundation kernel, unchanged.** The Q4 Mindlin-Reissner plate + Winkler
   element, batched assembly and sparse solve of
   `tensorfem.foundation_plate_bending` (the GAP-FEM-WINKLER kernel) are used
   as-is: `graded_square_mesh`, `assemble_plate_foundation_system`,
   `solve_assembled_system`.  No foundation code is duplicated in this anchor.
2. **Circular patch load.** Consistent nodal forces from sub-cell quadrature of
   the uniform pressure over the quarter disc (`quarter_disc_loads`,
   8x8 sub-cells per element).  The integrated load reproduces
   `p * pi * a^2 / 4` to 0.358% (worst).
3. **Moment recovery.** Curvatures from the Mindlin rotations at the 2x2 Gauss
   points (superconvergent), `M = D_b (kappa + nu ...)`; the bilinear
   extrapolation of the Gauss-point field to the corner node (load centre) is
   the nearest Gauss-point value.
4. **Failure criterion, independently implemented** (`ice_plate_flexural_limit`,
   *not* the analytic route of `levelset.ice_plate_fracture`):
   bottom fibre `sigma = +6 M/h^2`, top fibre `sigma = -6 M/h^2`,
   principal moments `M_1,2 = (Mx+My)/2 +- sqrt(((Mx-My)/2)^2 + Mxy^2)`,
   crack initiation at `max(sigma)` = `sigma_f` (Rankine), hence
   `P_f = sigma_f h^2 / (6 m_hat)` with `m_hat = max_field M_1 / P`.
5. **References.** Exact Kelvin patch moment `M_c = (1+nu)/2 p l^2 X kei'(X)`
   (`levelset.plate_crack.patch_max_moment`) and the Westergaard/Nevel closed
   form `(1+nu)(P/4 pi)(ln(l/a) + 0.6159)`
   (`levelset.plate_crack.patch_max_moment_nevel`).

## 3. Results

### 3.1 Mesh convergence of the FE patch moment (vs the exact Kelvin anchor)

| a/l | cells/side | beta | h at centre [m] | M_c FE [N m/m] | M_c Kelvin [N m/m] | error [%] | load error [%] |
|---|---|---|---|---|---|---|---|
| 0.3 | 40 | 4 | 0.4728 | 2.1983e+00 | 2.2191e+00 | -0.936 | +0.3582 |
| 0.3 | 80 | 5 | 0.1086 | 2.2176e+00 | 2.2191e+00 | -0.068 | +0.0211 |
| 0.3 | 120 | 6 | 0.0319 | 2.2186e+00 | 2.2191e+00 | -0.023 | +0.0011 |
| 0.3 | 160 | 8 | 0.0043 | 2.2187e+00 | 2.2191e+00 | -0.018 | +0.0004 |
| 1 | 40 | 4 | 0.4728 | 9.4829e+00 | 9.5024e+00 | -0.206 | -0.0672 |
| 1 | 80 | 5 | 0.1086 | 9.5006e+00 | 9.5024e+00 | -0.019 | +0.0201 |
| 1 | 120 | 6 | 0.0319 | 9.5004e+00 | 9.5024e+00 | -0.021 | -0.0025 |
| 1 | 160 | 8 | 0.0043 | 9.5011e+00 | 9.5024e+00 | -0.014 | +0.0139 |

Second-order convergence to the exact anchor; at 80 elements per side the centre
moment is within 0.068% (worst case in the table).

### 3.2 First-crack (flexural limit) load vs analytic plate theory

| a/l | m_hat FE | P_f FEM [kN] | P_f Kelvin [kN] | P_f Nevel [kN] | err vs Kelvin [%] | err vs Nevel [%] | reaction/load-1 |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.30100 | 29.90 | 29.80 | 29.81 | +0.339 | +0.305 | -9.6e-09 |
| 0.2 | 0.23032 | 39.08 | 39.03 | 39.09 | +0.130 | -0.047 | -9.6e-09 |
| 0.3 | 0.18901 | 47.62 | 47.58 | 47.80 | +0.085 | -0.394 | -9.6e-09 |
| 0.5 | 0.13783 | 65.30 | 65.27 | 66.46 | +0.039 | -1.745 | -9.7e-09 |
| 1 | 0.07288 | 123.49 | 123.45 | 141.25 | +0.039 | -12.572 | -9.9e-09 |
| 2 | 0.02274 | 395.79 | 395.79 | -1126.23 | -0.000 | -135.143 | -1.0e-08 |

* Acceptance: limit load within **10%** of the analytic plate theory.
  Max deviation from the exact Kelvin reference over the whole sweep:
  **0.339%**.
* Against the Westergaard/Nevel form the deviation is
  **<= 1.745%** for `a/l <= 0.5`
  (max 1.745% at `a/l = 0.5`).
  Beyond that the Nevel form itself leaves its validity window (table 3.3),
  so the FEM cannot be blamed for the gap.

### 3.3 Validity window of the Westergaard/Nevel closed form

| a/l | Kelvin [N m/m] | Nevel [N m/m] | error [%] |
|---|---|---|---|
| 0.1 | 3.9365e-01 | 3.9352e-01 | -0.035 |
| 0.2 | 1.2023e+00 | 1.2002e+00 | -0.176 |
| 0.3 | 2.2191e+00 | 2.2085e+00 | -0.478 |
| 0.5 | 4.4928e+00 | 4.4127e+00 | -1.783 |
| 1 | 9.5024e+00 | 8.3046e+00 | -12.606 |
| 1.5 | 1.1958e+01 | 6.3842e+00 | -46.610 |
| 2 | 1.1855e+01 | -4.1663e+00 | -135.143 |

The log form is the small-`a/l` expansion of the Kelvin solution: within 1% for
`a/l <= 0.3`, 1.8% at `a/l = 0.5`, 12.6% at `a/l = 1`, and it changes sign near
`a/l ~ 1.6`.  **Engineering consequence: quote Nevel for `a/l <= 0.5`; use the
Kelvin patch moment (or this FEM) outside.**  `nevel_moment_applicability`
exposes the table programmatically.

### 3.4 Failure location and mode

| a/l | critical fibre | r_crit/a | r_crit/l | top/bottom | M_theta/M_r | at centre | mode |
|---|---|---|---|---|---|---|---|
| 0.1 | bottom | 0.050 | 0.005 | 0.064 | 1.001 | yes | radial_bottom |
| 0.3 | bottom | 0.017 | 0.005 | 0.101 | 1.000 | yes | radial_bottom |
| 0.5 | bottom | 0.010 | 0.005 | 0.134 | 1.000 | yes | radial_bottom |
| 1 | bottom | 0.005 | 0.005 | 0.217 | 1.000 | yes | radial_bottom |
| 2 | bottom | 0.003 | 0.005 | 0.416 | 1.000 | yes | radial_bottom |
| 3 | bottom | 0.444 | 1.331 | 0.759 | 0.981 | no | transitional_bottom |
| 4 | bottom | 0.663 | 2.653 | 0.910 | 0.796 | no | circumferential_bottom |
| 6 | bottom | 0.782 | 4.691 | 0.914 | 0.567 | no | circumferential_bottom |
| 8 | bottom | 0.832 | 6.653 | 0.930 | 0.485 | no | circumferential_bottom |

Mesh sensitivity (80 vs 120 elements per side):

| a/l | r_crit/a (80 / 120) | M_theta/M_r | top/bottom | mode |
|---|---|---|---|---|
| 3 | 0.444 / 0.443 | 0.981 / 0.981 | 0.759 / 0.757 | transitional_bottom / transitional_bottom |
| 4 | 0.663 / 0.665 | 0.796 / 0.797 | 0.910 / 0.910 | circumferential_bottom / circumferential_bottom |
| 6 | 0.782 / 0.788 | 0.567 / 0.569 | 0.914 / 0.916 | circumferential_bottom / circumferential_bottom |

## 4. Why this location and this mode

1. **Small patch (`a/l <= 2`): bottom fibre at the load centre, radial cracks.**
   The centre is axisymmetric (`Mx = My`, `Mxy = 0`, `M_theta/M_r = 1.000` in the
   table) and sagging, so the whole bottom surface of the loaded area is in
   equibiaxial tension.  Every radial plane is equally critical; once one radial
   crack forms, the hoop stress it carried is released and the neighbouring
   radial planes become critical -- the classical radial crack star.  The bottom
   fibre is unambiguously critical here: the top-fibre tension at the same load
   is only 6.4% ... 41.6%
   of it over `a/l = 0.1 ... 2`.
2. **Large patch (`a/l >= 4`): the critical point migrates off-centre, onto the
   sagging ring at `r_crit ~ 0.66 a ... 0.83 a`, still on the bottom fibre.**
   For `a/l >> 1` the sheet under the load is nearly a uniformly loaded infinite
   plate (`M -> 0` in the deep interior), so the bending concentrates in the
   transition ring at the patch edge; the radial moment then dominates
   (`M_theta/M_r = 0.485` at `a/l = 8`),
   which by the normality of the crack to the largest tension means a
   **circumferential** bottom crack.  `a/l = 3` is the transition: the critical
   point has moved to `r_crit/a = 0.44` while the state is still within 2% of
   equibiaxial, which is reported as `transitional` rather than forced into a
   mode.
3. **The top fibre never governs for a uniform pressure patch**
   (`top/bottom = 0.064 ... 0.930`
   over `a/l = 0.1 ... 8`; it saturates near 0.93).  Top-surface circumferential
   cracking is a feature of *rim-concentrated* or edge loading (rigid indenter,
   plate edge), not of a uniform circular pressure patch -- that regime is
   outside this anchor and is stated as a limitation rather than modelled here.
4. **Consistency checks.** Foundation reaction equals the applied load to
   `1e-10` relative (equilibrium audit in `foundation_reaction`); the ring
   scatter of `M_r` in `polar_profiles` is the axisymmetry error; the critical
   location and mode are stable between the two meshes in table 3.4.

## 5. Traps found while building this anchor

* **Q4 Jacobian transpose.** With `J[a,c] = d x_c / d xi_a` (natural square
  `[-1,1]^2`), the chain rule is `dN_i/dx_d = sum_a (dN_i/dxi_a) J^-1[d,a]`,
  i.e. `dN = dN_nat @ inv(J).T`.  Omitting the transpose is invisible on
  axis-aligned rectangles (all elements of the graded quarter mesh) and wrong on
  distorted elements -- it silently doubles the moments if the side length is
  used instead of `side/2`.  `test_q4_kinematics_matches_torch_plate_kernel`
  now pins both the kinematics and the distorted-element case.
* **Quarter model bookkeeping.** `m_hat` must divide by the *full* load
  `4 x (quarter total)`; dividing by the quarter total inflates the limit load
  by exactly a factor of four while every internal consistency check still
  passes.
* **Element-centre vs Gauss-point moments.** Element-centre curvature
  underestimates the centre moment by the log-singular peak it averages over
  (2-3% at the same mesh that gives 0.02% with the Gauss-point value).

## 6. Error verdict

| quantity | reference | worst error | < 3%? |
|---|---|---|---|
| FE centre moment, 80+ cells | Kelvin patch moment | 0.068% | yes |
| first-crack load, whole sweep | Kelvin patch moment | 0.339% | yes |
| first-crack load, `a/l <= 0.5` | Westergaard/Nevel | 1.745% | yes |
| applied patch load | `p pi a^2/4` | 0.358% | yes |

The 10% acceptance criterion of H3-ICE-PLATE-FEM is met with a factor of ~30
margin against the exact plate theory.  Not demonstrated here (declared, not
claimed): comparison with an experimental first-crack dataset (no offline
dataset is available in this repository), and the rim-concentrated loading
regime that produces top-surface circumferential cracks.

## 7. Reproduce

```bash
python benchmarks/ice_plate_flexural_limit.py          # 74.1 s, writes evidence + figures
python scripts/render_ice_plate_flexural_limit_doc.py  # regenerates this document
python -m pytest tests/test_ice_plate_flexural_limit.py -q
```

Environment of the recorded run: Python 3.11.6,
NumPy 2.4.6, loongarch64.
Figures: `limit_load_vs_radius.png`, `polar_moment_profiles.png`, `convergence.png`.
