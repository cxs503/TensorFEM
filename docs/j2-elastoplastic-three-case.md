# Three-case qualification of the J2 (von Mises) elastoplastic surface

Campaign node **p337w1-n07-fem-elastoplastic** · TensorFEM · module
`tensorfem.j2_elastoplastic_three_case` · benchmark
`benchmarks/j2_elastoplastic_three_case/run.py` · evidence
`artifacts/j2_elastoplastic_three_case/report.json`.

Status: **REAL** (solver + oracles implemented and gated by tests), with two
explicit *withheld* claims listed in §5: the ASTM E8 experimental chord and any
closed-form claim for the hardening cylinder.

---

## 1. Case introduction

A constitutive model is only usable in a strength assessment once the *same
material card* answers three different questions correctly. This node asks them
with one card, in one code path, against oracles that the code under test never
sees:

| case | physical problem | geometry | loading | oracle |
|---|---|---|---|---|
| A | uniaxial tension of a coupon | `L = 100 mm`, `A = 10 × 10 mm` | displacement control, `ε = 0 … 5 %` | bilinear material card + independent scalar bisection (`j2_saturation_hardening.uniaxial_reference`) |
| B | pure shear / thin-wall tube torsion | `L = 100 mm`, `R = 50 mm`, `t = 2 mm` (`t/R = 0.04`) | `γ = 0 … 5 %` | closed form `τ = (σ_y0 + H λ)/√3`, `τ_y = σ_y/√3` |
| C | thick-walled pressure vessel | `R_i = 50 mm`, `R_o = 100 mm`, `L = 200 mm` | internal pressure `p`, plane strain | Hill 1950 (§5.3) closed form + quadrature for `u_r(r)` |

Material card (identical in all three cases, SI):

| `E` | `ν` | `σ_y0` | `H` | `E_t = EH/(E+H)` | `G` | `ε_y` |
|---|---|---|---|---|---|---|
| 210 GPa | 0.30 | 250 MPa | 1000 MPa | 995.24 MPa | 80.77 GPa | 0.11905 % |

Derived cylinder quantities (computed, not tabulated by hand): shear yield
`k = σ_y/√3 = 144.34 MPa`, first yield `p_y = k(1 − R_i²/R_o²) = 108.25 MPa`,
plastic collapse `p_lim = 2k ln(R_o/R_i) = 200.09 MPa`.

**Note on the pressure band.** The campaign text specifies `p = 0 … 100 MPa`
for case C. For this card that band is **entirely elastic** (`p_y = 108.25 MPa`),
so it cannot contain a plastic-zone expansion. Both are therefore reported: the
literal band as an elastic check (`cylinder_elastic_band`, §3.4) and five
*contained-plastic* levels `p/k = 1.00, 1.10, 1.20, 1.30, 1.35`
(`144 … 195 MPa`, i.e. `p < p_lim`) where `r_p(p)` exists. This is a physics
requirement, not a relaxation of the criterion.

## 2. How it was done

**Constitutive kernel under test** (`tensorfem.plasticity.update_j2`,
`tensorfem.j2_elastoplasticity.radial_return`): von Mises yield
`f(σ) = √(3J₂) − σ_y(ε_p)`, linear isotropic hardening
`σ_y(ε_p) = σ_y0 + H ε_p`, associative flow `dε_p = dλ ∂f/∂σ`, backward-Euler
radial return (Wilkins 1964) with the closed-form consistent tangent.

**Case A** — two independent routes plus a finite-element route:

1. material point: `uniaxial_stress_path` drives `σ_22 = σ_33 = 0` by a lateral
   Newton iteration that uses the *consistent tangent*, so the tangent is
   exercised rather than bypassed;
2. independent oracle: `uniaxial_reference` solves
   `E(ε − α) = σ_y(α)` by bisection on a scalar equation — a different code
   path with a different return map (implicit Voce/linear Newton);
3. FE: an eighth-model TET4 bar (`8 × 1 × 1` hexes split into 48 tets, symmetry
   planes at `x = y = z = 0`) under displacement control, Newton to
   `1e-10` relative residual. The exact solution of this bar is *homogeneous*,
   and a linear tetrahedron represents a uniform strain field exactly, so the
   case isolates the constitutive update + incremental driver with no spatial
   error to hide behind. Axial stress is measured twice: element stress and
   load-cell reaction over the quarter area.

**Case B** — material point under the strain state `ε_xy = γ/2` (isotropic J2
maps this to `τ_xy` only, which is the thin-wall tube solution), plus a TET4
patch test in which every boundary node is driven by the homogeneous field
`u_x = γy`; the patch test checks that the *assembled* code path reproduces the
material-point answer.

**Case C** — plane-strain axisymmetric Q4 annulus
(`tensorfem.thick_cylinder_plasticity`), `u_z = 0` enforced exactly so
`ε_z = 0`; the full 3-D J2 return mapping runs at every Gauss point, i.e.
`σ_z` and the yield condition are *outputs*, not assumptions. Two oracles:
Hill's closed form for `r_p(p)` and stresses, and `u_r(r)` by quadrature of the
compatibility identity `d(r²ε_θ)/dr = r ε_vol` (no plastic closed form for the
displacement is used).

**Mesh convergence (C7)** — Roache GCI via
`tensorfem.convergence_verification.evaluate_convergence_study` on `u_r(a)` with
`n_r = 16, 32, 64` and `h = (R_o − R_i)/n_r`, safety factor 1.25.

**Tool chain** — torch 2.x float64, pytest, matplotlib; run with
`python benchmarks/j2_elastoplastic_three_case/run.py`. The JSON artifact is the
single source of every number below (tables in §3 are generated *from* it, see
`benchmarks/j2_elastoplastic_three_case/render_markdown.py`).

## 3. Results

<!-- TABLES:START -->

### 3.1 Acceptance criteria (machine read from the artifact)

| criterion | value | band | verdict |
|---|---|---|---|
| C1_uniaxial_curve | 3.80e-15 | 5% | **PASS** |
| C2_yield_stress | 3.58e-16 | 2% | **PASS** |
| C3_shear_yield | 2.06e-16 | 2% | **PASS** |
| C4_plastic_radius | 0.0137 | 5% | **PASS** |
| C5_radial_displacement | 0.0208 | 10% | **PASS** |
| C6_pi_plane_circle | 1.19e-16 | 1% | **PASS** |
| C7_gci | 8.64e-04 | 5% | **PASS** |

### 3.2 Case A - uniaxial tension (`ε = 0 … 5 %`)

| ε [%] | card σ [MPa] | material point σ [MPa] | ε_p [%] | rel. error |
|---|---|---|---|---|
| 0.100 | 210.000 | 210.000 | 0.000 | 0.00 |
| 0.250 | 251.303 | 251.303 | 0.130 | 1.19e-16 |
| 0.500 | 253.791 | 253.791 | 0.379 | 3.52e-16 |
| 1.000 | 258.768 | 258.768 | 0.877 | 4.61e-16 |
| 2.000 | 268.720 | 268.720 | 1.872 | 4.44e-16 |
| 3.000 | 278.673 | 278.673 | 2.867 | 1.50e-15 |
| 4.000 | 288.626 | 288.626 | 3.863 | 8.26e-16 |
| 5.000 | 298.578 | 298.578 | 4.858 | 1.80e-15 |

FE bar (48 TET4, 36 nodes, 8 axial × 1 transverse): worst relative error of the element stress 3.80e-15, of the load-cell reaction 1.47e-14; yield stress read back from the computed curve 250.0000 MPa vs the card 250.0 MPa (3.58e-14 %).


### 3.3 Case B - pure shear (`γ = 0 … 5 %`)

| γ [%] | closed form τ [MPa] | return map τ [MPa] | ε_p [%] | rel. error |
|---|---|---|---|---|
| 0.040 | 32.308 | 32.308 | 0.000 | 0.00 |
| 0.090 | 72.692 | 72.692 | 0.000 | 0.00 |
| 0.150 | 121.154 | 121.154 | 0.000 | 0.00 |
| 0.210 | 144.441 | 144.441 | 0.018 | 0.00 |
| 0.300 | 144.740 | 144.740 | 0.070 | 0.00 |
| 0.600 | 145.736 | 145.736 | 0.242 | 0.00 |
| 1.200 | 147.728 | 147.728 | 0.587 | 0.00 |
| 2.500 | 152.043 | 152.043 | 1.335 | 5.88e-16 |
| 5.000 | 160.343 | 160.343 | 2.772 | 1.30e-15 |

`τ_y = 144.3376 MPa` vs `σ_y/√3 = 144.3376 MPa` (2.06e-16 relative); plastic shear slope `331.963 MPa` vs closed form `331.963 MPa`; torque at yield `T_y = τ_y W_t = 4534.4984 N·m` (`W_t = 2πR²t = 3.14e-05 m³`). TET4 patch test (homogeneous shear, `γ = 0.30 %`): `τ = 144.7402 MPa` vs material point `144.7402 MPa` (2.06e-16), stress spread 0.00 Pa.


### 3.4 Case C - the literal `p = 0 … 100 MPa` band is elastic

| p [MPa] | p / p_y | max ε_p | σ_r err (÷p) | σ_θ err (÷p) | u_r(a) rel. err |
|---|---|---|---|---|---|
| 20 | 0.185 | 0.0 | 0.0196 | 0.0084 | 1.75e-04 |
| 40 | 0.370 | 0.0 | 0.0196 | 0.0084 | 1.75e-04 |
| 60 | 0.554 | 0.0 | 0.0196 | 0.0084 | 1.75e-04 |
| 80 | 0.739 | 0.0 | 0.0196 | 0.0084 | 1.75e-04 |
| 100 | 0.924 | 0.0 | 0.0196 | 0.0084 | 1.75e-04 |

All five levels: `max ε_p = 0.0` — the whole band is elastic, as predicted by `p_y = 108.25 MPa`. Worst Lamé stress error 1.96 % of `p` (mesh `n_r = 48`, grading 0.1 — the `σ_r` gradient at the bore is why a uniform mesh resolves it only to a few percent), bore displacement error 0.0175 %.


### 3.5 Case C - plastic-zone growth, five contained-plastic levels

| p [MPa] | p/k | r_p FE [mm] | r_p Hill [mm] | err [%] | u_r(a) FE [µm] | u_r(a) Hill [µm] | err [%] | yielded |
|---|---|---|---|---|---|---|---|---|
| 144.34 | 1.00 | 59.902 | 59.783 | 0.199 | 71.548 | 71.323 | 0.316 | 0.203 |
| 158.77 | 1.10 | 65.057 | 64.875 | 0.281 | 85.662 | 85.295 | 0.429 | 0.297 |
| 173.21 | 1.20 | 71.523 | 71.200 | 0.454 | 105.495 | 104.836 | 0.629 | 0.430 |
| 187.64 | 1.30 | 80.639 | 79.999 | 0.800 | 137.424 | 135.843 | 1.164 | 0.609 |
| 194.86 | 1.35 | 88.031 | 86.845 | 1.366 | 166.275 | 162.894 | 2.075 | 0.766 |


Roache GCI on `u_r(a)` (`h = (R_o − R_i)/n_r`, `n_r = 16, 32, 64):

| p [MPa] | GCI (rel.) | observed order | Richardson u_r(a) [µm] | offset vs Hill [%] |
|---|---|---|---|---|
| 144.34 | 4.94e-04 | 1.148 | 71.577 | 0.356 |
| 158.77 | 6.19e-04 | 1.224 | 85.704 | 0.479 |
| 173.21 | 2.85e-04 | 1.857 | 105.519 | 0.652 |
| 187.64 | 6.49e-04 | 1.709 | 137.496 | 1.216 |
| 194.86 | 8.64e-04 | 1.799 | 166.390 | 2.146 |

Worst case C numbers: `r_p` 1.366 % (band 5 %), `u_r(a)` 2.075 % (band 10 %), `u_r(r)` profile worst 2.075 % (band 10 %). The last column is *not* a mesh error: the Richardson value of the FE sequence still sits 2.15 % from Hill at the highest pressure, which is the classical plane-strain `σ_z` inconsistency discussed in §5.5, not a discretisation defect.


### 3.6 Case C with isotropic hardening `H = 1000 MPa` (no closed form)

| p/k | r_p (H = 1000) [mm] | u_r(a) (H = 1000) [µm] | r_p (H = 0) [mm] | u_r(a) (H = 0) [µm] |
|---|---|---|---|---|
| 1.00 | 59.883 | 71.490 | 59.902 | 71.548 |
| 1.10 | 65.006 | 85.494 | 65.057 | 85.662 |
| 1.20 | 71.381 | 105.023 | 71.523 | 105.495 |
| 1.30 | 80.242 | 135.846 | 80.639 | 137.424 |
| 1.35 | 87.065 | 162.263 | 88.031 | 166.275 |

Hardening moves the answers as physics requires: at `p/k = 1.35` the interface shrinks from `88.03 mm` (perfect plasticity) to `87.06 mm` and the bore displacement from `166.27 µm` to `162.26 µm`. No oracle columns are scored here (Hill is a perfect-plasticity solution); this branch is verified by mesh refinement only.


### 3.7 C6 - yield surface in the π plane

| θ [°] | q at onset [MPa] | deviation |
|---|---|---|
| 0 | 250.000000 | 0.00 |
| 60 | 250.000000 | 1.19e-16 |
| 120 | 250.000000 | 0.00 |
| 180 | 250.000000 | 0.00 |
| 240 | 250.000000 | 1.19e-16 |
| 300 | 250.000000 | 1.19e-16 |

Worst deviation 1.19e-16 at θ = 60° (band 1 %): the implemented return map puts the yield surface on a circle in the π plane to machine precision.


### 3.8 Run metadata

| item | value |
|---|---|
| node | p337w1-n07-fem-elastoplastic |
| quick mode | False |
| wall time [s] | 264.5 |
| torch | 2.7.1 |
| cylinder mesh sizes | 16, 32, 64 |
| pressure levels [MPa] | 144.3, 158.8, 173.2, 187.6, 194.9 |
<!-- TABLES:END -->

## 4. Figures

* `fig1_uniaxial_stress_strain.png` — case A: card / material point / FE bar.
* `fig2_pure_shear.png` — case B: `τ(γ)` and `τ_y = σ_y/√3`.
* `fig3_cylinder_plastic_zone.png` — case C: `r_p(p)` and `u_r(r)` vs Hill.
* `fig4_pi_plane.png` — C6: yield surface in the π plane (6 directions).
* `fig5_gci.png` — C7: `h`-refinement of the bore displacement, five pressures.

All figures are regenerated by the runner from the same JSON; none of them
embeds a hand-entered value.

## 5. Points of attention, limitations

1. **Infinitesimal strain.** `ε = 5 %` is at the edge of the small-strain
   contract: the axial strain is 42 × `ε_y`, and a finite-strain (multiplicative
   J2, cf. `tensorfem.finite_strain_plasticity`) treatment would differ in the
   third digit. The 5 % level is reported, not extrapolated further.
2. **No experimental chord (withheld).** No ASTM E8 dataset is in the
   repository and the worktree has no network. C1 is therefore scored against
   the *analytic material card* the campaign specifies; the experimental
   comparison is registered as `withheld`, not as passed. Substituting a
   remembered/published curve by hand was deliberately not done — that would
   make the criterion unfalsifiable.
3. **Case A has no spatial error by construction.** Reporting "FE = card" is a
   statement about the constitutive integration, not about tetrahedral
   accuracy. Spatial accuracy is case C's job (and case C's `r_p` error is
   dominated by the interface reconstruction, not by the element).
4. **TET4 volumetric locking is *not* exercised** by case A: the plastic strain
   field there is uniform and therefore representable, so nothing locks. A
   *non-homogeneous* plastic TET4 problem would need BBAR/mixed formulation;
   that is outside this node's scope (see
   `docs/incompressible-locking-audit.md` for the elastic side).
5. **The classical plane-strain solution is not exactly consistent for
   `ν ≠ 1/2`.** Hill's solution sets `σ_z = (σ_r + σ_θ)/2` in the plastic zone
   (J2 flow plus `ε_z^p = 0`) but `σ_z = ν(σ_r + σ_θ)` in the elastic zone, so
   `σ_z` jumps at `r = c` and the von Mises measure of the *elastic* zone is
   `q = √(3s² + (1−2ν)² m²)` rather than the `(√3/2)(σ_θ − σ_r)` used by the
   closed form. The FE solution, which enforces continuity, therefore converges
   to a value a fraction of a percent away from the formula; that offset is
   reported separately (`richardson_relative_offset_vs_analytic`) and it is a
   property of the *oracle*, not of the mesh. It stays inside the C4/C5 bands.
6. **Hardening cylinder has no closed form.** With `H = 1000 MPa` the same
   pressure levels are re-run and reported (r_p, u_bore), but the oracle
   columns are set to `None` rather than scored against perfect-plasticity
   physics. That branch is verified only by mesh convergence.
7. **Plane strain / closed ends.** Case C models a long tube in plane strain
   (`ε_z = 0`); closed-end pressure (`σ_z = pR²/(R_o²−R_i²)`) is *not* included,
   so axial stress is an output of the constitutive update, not of a pressure
   end cap.
8. **`r_p` resolution.** `r_p` is extracted from the elastic-zone Lamé fit
   `q = C/r²` over the four elastic Gauss points nearest the interface (second
   order in `h`), with the model-free bracket `[r_yield_last, r_elastic_first]`
   reported alongside as a sanity bound.

## 6. Error judgement (explicit answer per criterion)

| criterion | quantity | oracle | band | value | < 3 % ? |
|---|---|---|---|---|---|
| C1 | `σ(ε)`, `ε = 0 … 5 %` | bilinear material card | 5 % | see §3 table | yes (machine precision) |
| C2 | `σ_y` read back from the FE curve | card `σ_y0 = 250 MPa` | 2 % | see §3 table | yes |
| C3 | `τ_y` | `σ_y/√3 = 144.338 MPa` | 2 % | see §3 table | yes |
| C4 | `r_p(p)`, 5 levels | Hill 1950 | 5 % | see §3 table | **no** (worst ≈ 1.4 %) |
| C5 | `u_r(r)`, 11 points | Hill quadrature | 10 % | see §3 table | **no** (worst ≈ 2.1 %) |
| C6 | π-plane circularity, 6 directions | `q = σ_y` | 1 % | see §3 table | yes (machine precision) |
| C7 | GCI on `u_r(a)` | Roache, 3 meshes | — | see §3 table | GCI itself ≤ 0.1 % |

The engine-wide 3 % standard is answered explicitly: C1, C2, C3, C6 are at
machine precision; **C4 (1.4 %) and C5 (2.1 %) are above the repo-wide 3 %
preference but inside the campaign's own 5 % / 10 % bands**, and the residual is
attributed to the `σ_z` inconsistency of the classical plane-strain closed form
(§5.5) rather than to the FE discretisation — the mesh-convergence study places
the discretisation error at ≤ 0.1 % of `u_r(a)`.

## 7. Reproduction

```bash
python benchmarks/j2_elastoplastic_three_case/run.py            # full (~5 min)
python benchmarks/j2_elastoplastic_three_case/run.py --quick    # smoke (~1 min)
pytest tests/test_j2_elastoplastic_three_case.py -q             # fast lane gates
pytest tests/test_j2_elastoplastic_three_case.py -q -k full_qualification  # slow lane
```