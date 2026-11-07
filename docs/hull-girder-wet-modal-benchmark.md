# Hull-girder wet modal benchmark (mission 5425, node P55-N03)

Status: **passes all four gates on the declared gate case, with two quantified
applicability limits** (see §7). Everything below is produced by
`benchmarks/hull_girder_modal/make_result.py`; the numbers in `result.json` are
recomputed on every run, none are transcribed.

## 1. Why this benchmark exists

TensorFEM already had modal analysis for plates/shells (dry modes) and beams
(`modal`, `timoshenko_beam_modal`). What it did not have is the *ship* step:
a hull girder floating in water, where the water that oscillates with the hull
adds mass and drops the bending frequencies by tens of percent. Any finite
element model of a hull girder is therefore only as trustworthy as its
reproduction of three analytic anchors:

1. the shear-flexible (Timoshenko) bending frequencies,
2. the two-dimensional section added mass of the immersed section,
3. the wet/dry frequency reduction that follows from 1 and 2.

This benchmark fixes all three anchors for a uniform box girder and quantifies
the fourth requirement any such claim needs: grid convergence.

## 2. Physical model and parameters

Uniform thin-walled box girder (equivalent hull girder), pinned–pinned, bending
about the transverse axis; vertical shear carried by the two side walls.

| quantity | symbol | gate case `girder_g` | ship-proportioned case `hull_s` |
|---|---|---|---|
| length | L | 200.0 m | 180.0 m |
| breadth | B | 16.0 m | 24.0 m |
| depth | D | 4.0 m | 14.0 m |
| wall thickness | t | 20 mm | 22 mm |
| draft | T | 2.4 m | 8.4 m |
| steel density / modulus | ρ_s, E | 7850 kg/m³, 206 GPa | same |
| Poisson ratio | ν | 0.30 | 0.30 |
| water density | ρ_w | 1025 kg/m³ | 1025 kg/m³ |
| material area | A | 0.7984 m² | 1.6701 m² |
| second moment of area | I | 2.7415 m⁴ | 61.548 m⁴ |
| shear area (2 webs) | A_s | 0.160 m² | 0.616 m² |
| effective shear coefficient | κ = A_s/A | 0.2004 | 0.3688 |
| EI | | 5.647×10¹¹ N·m² | 1.268×10¹³ N·m² |
| κGA | | 1.268×10¹⁰ N | 4.881×10¹⁰ N |
| EI/(κGA) | | 44.55 m² | 259.8 m² |
| L/D | | 50.0 | 12.9 |

**Geometry is verified before any frequency is used.** `geometry_audit()` checks
the closed-form `A` and `I` on a path that does not share their algebra:
Green's theorem on the outer/inner polygons (agreement 0 and 1.8×10⁻¹⁵
respectively) plus the thin-wall asymptote `A → 2t(B+D)`,
`I → tD²(3B+D)/6` (residual 2.0×10⁻³ and 1.2×10⁻², i.e. the expected O(t/D)).
A test asserts both.

The finite element model is the two-node Timoshenko element of
`tensorfem.timoshenko_beam_modal` (cubic/interdependent `w`, linear `θ`,
four-point Gauss). The box girder is carried onto that element by the
*equivalent rectangle* `h_eq = sqrt(12 I/A)`, `w_eq = A/h_eq`, `κ = A_s/A`,
which reproduces `EI`, `ρA`, `ρI` and `κGA` **exactly** (machine precision,
asserted in the tests) — it is a faithful carrier for the beam mechanics, not a
shape approximation.

## 3. Anchor 1 — Timoshenko corrected frequencies (gate 1)

Four independent paths for modes *n* = 1..4:

* `f_EB`: Euler–Bernoulli closed form, `(nπ/L)²·sqrt(EI/(ρA))`;
* `f_T1`: the first-order correction of the task specification,
  `f_EB·sqrt(1 − (nπ/L)²·EI/(κGA))`;
* `f_exact`: exact Hu–Timoshenko dispersion relation (quadratic in ω²);
* `f_FEM(N)`: finite element, N ∈ {20, 40, 80}.

Gate case `girder_g` (frequencies in Hz):

| n | f_EB | f_T1 | f_exact | f_FEM(80) | err f_T1 | err FEM |
|---|---|---|---|---|---|---|
| 1 | 0.37277 | 0.37072 | 0.37058 | 0.37061 | 0.035 % | 0.006 % |
| 2 | 1.49108 | 1.45793 | 1.45708 | 1.45745 | 0.059 % | 0.026 % |
| 3 | 3.35493 | 3.18466 | 3.19030 | 3.19215 | 0.177 % | 0.058 % |
| 4 | 5.96432 | 5.41450 | 5.47342 | 5.47906 | 1.077 % | 0.103 % |

Shear flexibility lowers the Euler–Bernoulli frequency by 0.6 % (mode 1) to
8.2 % (mode 4) — the correction is physically significant, and the finite
element model reproduces the exact dispersion to ≤0.11 %.

## 4. Anchor 2 — Lewis section added mass (gate 2)

Conventions: two-dimensional strip theory, zero forward speed, infinite water
depth, free surface treated as a plane of symmetry (rigid lid), so the immersed
section and its mirror image form a closed Lewis contour
`z = M(ζ + a₁/ζ + a₃/ζ³)`. The added mass follows from the closed-form
potential of the translating section,

    m₃₃ = ρ ∮ φ n_y dl = ρ ∫₀^{2π} φ(θ) x'(θ) dθ,   φ = V(y − 2M sinθ).

The implementation is verified against the three exact limits of the family
(asserted in the tests, relative error ≤1.5×10⁻¹⁶):

| limit | half-breadth b, half-draft h | m₃₃/ρ computed | exact |
|---|---|---|---|
| circle | 1.0, 1.0 | 3.141592654 | π b² = 3.141592654 |
| circle | 2.5, 2.5 | 19.63495408 | π b² = 19.63495408 |
| ellipse | 2.0, 1.0 | 12.566371 | π b² = 12.566371 |
| flat plate | 5.0, 0.05 | 78.539816 | π b² = 78.539816 |

Section fit and added mass of the gate girder section (b = 8.00 m, h = 2.40 m,
boxiest admissible member a₃ = −0.0545):

* enclosed area 69.56 m² = **90.6 %** of the doubled box area 76.8 m²;
* m₃₃ = **222.7 t/m** (ρ_w = 1025 kg/m³);
* κ₃₃ = m₃₃/(ρ_w B T) = **5.658** against the displaced mass,
  κ = m₃₃/m_struct = 35.53 against the bare steel.

The task specification writes κ₃₃ ≈ (πB/4T)·f(B/T). Within a strip model the
correction factor depends on one ratio only, `f = c/π` with
`c = m₃₃/(ρ b²)`:

| T/B | c = m₃₃/(ρb²) | κ₃₃ (Lewis) | πB/4T | f = c/π |
|---|---|---|---|---|
| 0.05 | 3.2222 | 16.111 | 15.708 | 1.026 |
| 0.10 | 3.3066 | 8.267 | 7.854 | 1.053 |
| 0.15 | 3.3950 | 5.658 | 5.236 | 1.081 |
| 0.25 | 3.5839 | 3.584 | 3.142 | 1.141 |
| 0.50 | 4.1094 | 2.055 | 1.571 | 1.308 |

So the task's `πB/(4T)` is the flat-plate limit (f → 1 as T/B → 0) and
under-predicts the box value by 8 % at T/B = 0.3 and by 31 % at T/B = 0.5. The
dashed line in fig. 4 is that comparison.

Gate 2 also checks the finite element assembly: the rigid-translation quadratic
form of the consistent added-mass matrix equals m₃₃·L to 1.7×10⁻¹⁶ (partition of
unity of the translational shape functions), so the matrix transports the
closed-form section value without loss.

## 5. Anchor 3 — wet/dry frequency ratio (gate 3)

A floating body displaces its own weight, so the vibrating mass per unit length
is the displaced water mass (structure + payload/ballast). For the gate case that
is 39.36 t/m, of which 6.27 t/m is steel and 33.09 t/m payload.

    f_wet/f_dry = 1/sqrt(1 + κ₃₃),   κ₃₃ = 5.6582
    → closed form 0.387545
    → FEM, N = 80, modes 1..4: 0.387567 / 0.387627 / 0.387711 / 0.387802
    → maximum deviation 0.066 %

The residual 6.6×10⁻⁴ is physical, not numerical: the structural mass matrix
carries rotary inertia `ρI` while the added mass does not, so the two mass
operators are not exactly proportional. Modes converge to the closed form from
above, as that argument requires. Diagnostic for the bare (unballasted) steel
girder: κ = 35.5 → ratio 0.166; such a girder cannot float at its draft, which
is why the payload mass is part of the definition.

## 6. Anchor 4 — grid convergence (gate 4)

N ∈ {20, 40, 80}, uniform refinement r = 2, Roache GCI per mode:

| n | observed order p | GCI(fine) | error vs exact (N = 80) |
|---|---|---|---|
| 1 | 2.031 | 0.008 % | 0.006 % |
| 2 | 2.030 | 0.032 % | 0.026 % |
| 3 | 2.030 | 0.073 % | 0.058 % |
| 4 | 2.030 | 0.126 % | 0.103 % |

Observed order ≈ 2.03 ≥ 2 on all four modes, i.e. the element is in its
asymptotic range and second order as a shear-flexible two-field element
should be (fig. 2 shows the errors against the O(N⁻²) reference).

## 7. Applicability limits (honest, quantified)

**(a) The first-order Timoshenko formula of the specification has a narrow
range.** It drops the ω⁴ term of the Hu–Timoshenko quadratic and uses
`sqrt(1 − x)` instead of `1/sqrt(1 + x)`, so its error grows like `x²` with
`x = (nπ/L)²·EI/(κGA)`. The formula is *undefined* once `x > 1`. For the
ship-proportioned case `hull_s` (L/D = 12.9, EI/κGA = 259.8 m²):

| n | f_exact [Hz] | f_T1 [Hz] | error f_T1 | validity index x | FEM(80) error |
|---|---|---|---|---|---|
| 1 | 1.44442 | 1.44682 | 0.17 % | 0.0791 | 0.006 % |
| 2 | 5.18872 | 4.98578 | 3.91 % | 0.3165 | 0.026 % |
| 3 | 10.19169 | 7.27944 | 28.6 % | 0.7122 | 0.058 % |
| 4 | 15.73978 | undefined | — | 1.2661 | 0.103 % |

So gate 1's 2 % band is met by the closed form on a **slender** girder
(`girder_g`, max 1.08 %) and by the **finite element model on both** girders
(max 0.11 %), but the closed form itself fails at n ≥ 2 for ship proportions.
The exact Hu–Timoshenko root in `timoshenko_beam_modal` is the reference to use
there; this benchmark does not "fix" the specification's formula, it measures
where it stops being usable.

**(b) The scalar wet/dry reduction is exact only for uniform added mass.** With
hull-like taper (beam −65 %, draft −35 % at the ends, m₃₃ from 222.7 t/m
amidships to 29.3 t/m at the ends, κ_mean = 4.182) the finite element wet/dry
ratio is 0.4016 / 0.4249 / 0.4365 / 0.4410 for modes 1..4 against the scalar
prediction 0.4393: deviations 8.6 %, 3.3 %, 0.6 %, 0.4 %. Mode 1 is most
affected because its displacement peaks where the added mass peaks. This is an
error of the *reduction*, not of the assembly (mesh refinement changes it by
<0.01 %), and it is the reason gate 3 is declared on the uniform girder only.

**(c) Lewis forms cannot represent a sharp-cornered box.** The family's
admissible ("boxiest") member stays inside its b×h bounding box and reaches
~91 % of the box area, so the reported κ₃₃ is a *lower bound* for a sharp
section; smooth flanks also mean the corner singularity (which raises the true
added mass of a square-ish section) is not captured. Therefore the section
added mass is reported as the *Lewis-equivalent* value with its area shortfall
explicit, not as an exact box value.

**(d) Model scope.** Two-dimensional strip theory, no forward speed, no
free-surface waves (rigid lid), no viscosity/flow separation, no 3D end
reduction of the added mass, no torsion/warping, no hull openings or
non-structural mass distribution, and the payload is taken as uniformly
distributed. Whipping *excitation* (slamming impulse, transient response) is out
of scope: what this benchmark anchors is the wet modal *frequency*, the
denominator of any whipping study.

## 8. Files

| path | content |
|---|---|
| `src/tensorfem/hull_girder_modal.py` | box girder section, geometry audit, Timoshenko correction, Lewis added mass, wet/dry modal assembly, GCI |
| `benchmarks/hull_girder_modal/timoshenko_comparison.py` | anchor 1 + grid convergence, `compute()` |
| `benchmarks/hull_girder_modal/wet_dry_ratio.py` | anchors 2 and 3, taper study, `compute()` |
| `benchmarks/hull_girder_modal/make_result.py` | `result.json` + figures 1–5 |
| `benchmarks/hull_girder_modal/result.json` | machine-readable gates and evidence |
| `benchmarks/hull_girder_modal/figures/` | five figures (README-level captions in `result.json`) |
| `tests/test_hull_girder_modal.py` | 25 tests, physical properties and the committed gates |

Reproduce:

```bash
PYTHONPATH=src python benchmarks/hull_girder_modal/make_result.py   # ~2 min
PYTHONPATH=src python -m pytest tests/test_hull_girder_modal.py -q  # 25 passed
```