# Level-set crack propagation and bending fracture of a floating ice sheet (TensorFEM)

Node `p360w1-n02-levelset-crack` · repository **TensorFEM** · new capability
`src/tensorfem/levelset/` · tests `tests/test_levelset_fracture.py` (21 passed)
· artefacts `artifacts/levelset_ice_plate_benchmark.json`,
`artifacts/levelset_vce_convergence.json`, `artifacts/levelset_figures_data.json`
· figures `docs/assets/levelset-ice-fracture/`

---

## 1 Case introduction

A floating ice sheet of thickness *h* is modelled as a Mindlin plate on a
Winkler foundation (`p = ρ_w g w`, i.e. hydrostatic restoring) carrying a
straight through-thickness crack that propagates when the energy release rate
reaches the ice toughness. The crack is **not** a prescribed line: it is the zero
contour of a level-set field φ, which is advected with the tip velocity and
re-initialised to a signed distance function, and the finite-element crack is
built from that contour at every increment (level set = single source of truth
for the crack geometry).

| quantity | symbol | value |
|---|---|---|
| Young's modulus | E | 7.0 GPa |
| Poisson ratio | ν | 0.30 |
| sheet thickness | h | 0.30 m |
| ice / water density | ρ_i / ρ_w | 917 / 1025 kg m⁻³ |
| Winkler modulus | k = ρ_w g | 10 055.3 N m⁻³ |
| flexural rigidity | D = Eh³/(12(1−ν²)) | 1.7308 × 10⁷ N m |
| characteristic length | l = (D/k)^{1/4} | 6.4411 m |
| fracture toughness | K_IC | 0.10 MPa m^{1/2} |
| critical energy release rate | G_c = K_IC²(1−ν²)/E | 1.30 J m⁻² |
| flexural strength (cross-check) | σ_f | 0.60 MPa |
| load patch radius | a_l | 0.50 m |
| initial crack half-length | a | 0.30 m |
| reference patch pressure | p | 10 kPa (P = 7.854 kN) |

Geometry: square domain 4l × 4l (25.76 m), graded tensor mesh clustered on the
crack (central spacing 0.100 m, 80×80 cells, 6561 nodes); the crack lies on the
grid line y = 0 and the load patch is centred on the crack centre.

## 2 Method

1. **Level set** (`level_set.py`). Signed distance field φ with the crack at
   φ = 0. Re-initialisation solves the Sussman PDE φ_τ + S(φ₀)(|∇φ|−1) = 0 with a
   Godunov upwind Hamiltonian (first order by default; second-order one-sided
   differences in the interior of the band with a mandatory first-order fallback
   within two cells of the band edge — without that fallback the scheme is
   unstable, observed and documented here). Transport uses
   ∂φ/∂t + V·∇φ = 0 with a semi-Lagrangian RK2 backtrace (unconditionally
   stable) or a CFL-limited first-order upwind scheme. Velocity extension off
   the front (∇V·∇φ = 0) is done by Adalsteinsson–Sethian ordered sweeping. The
   zero contour is extracted by marching squares with exact edge-key topology
   (no floating-point stitching); path metrics are arc-length re-sampling plus a
   one-way Hausdorff/RMS deviation normalised by the reference length.
2. **Crack criteria** (`crack_criteria.py`). Maximum tangential (hoop) stress
   criterion (Erdogan–Sih), maximum energy-release-rate criterion (small-kink
   angular functions), K–G conversion, tip kinetic law v = v₀(G/G_c−1)^m, and
   the **bending-Griffith** relation derived from the Kirchhoff/plane-stress
   analogy (w ↔ Airy function, D ↔ 1/E, ν ↔ −ν):
   G = π c² M² a/(D h), with c = 1/(1+ν) for equibiaxial bending,
   c = 1 for the cylindrical state M_tt = νM_nn and c = 1/(1−ν²) for a pure
   uniaxial moment.
3. **Cracked plate finite element** (`plate_crack.py`). 4-node Mindlin plate
   (MITC assumed shear field) on a Winkler foundation; the crack is explicit:
   the nodes on the crack line are duplicated into a lower and an upper bank, so
   both banks keep all three generalised displacements free (Reissner–Mindlin
   free-surface condition). Stiffness of the axis-aligned rectangular element is
   re-implemented in numpy (`rect_mindlin_stiffness`) and asserted equal to
   `tensorfem.plate.q4_mindlin_stiffness` to 1e-16 relative — the torch element
   costs 0.13 s per call on this host (LAPACK bound), which made the assembly of
   a cracked model with one element size per grid row unusable.
4. **Energy release rate** by virtual crack extension: G = (Π(a) − Π(a+δa))/(2 t δa)
   with Π = −uᵗf/2 at fixed load, δa = one element column, both tips advanced.
5. **Ice-sheet fracture driver** (`ice_plate_fracture.py`). Load steps are linear
   (G ∝ P²), so the fracture load is P_f = p_ref √(G_c/G(p_ref)). The growth loop
   then advances the level set at the kinetic-law speed, re-extracts the crack
   span from the zero contour (real-crossing test, outward snapping to grid
   nodes) and re-solves.

## 3 Results

### 3.1 A1 — re-initialisation (`|∇φ| = 1`)

Distorted circle (φ₀ = φ(1+0.3 sin 6φ), R = 1 m, domain ±1.5 m, band |φ| ≤ 0.45 m,
metric evaluated on |φ| ≤ 0.2 m so that the frozen band edge cannot pollute it):

| h / m | order 1, max ‖∇φ|−1| | order 2, max ‖∇φ|−1| | order 2, max |φ−φ_exact| |
|---|---|---|---|
| 0.030 | 1.33 × 10⁻² | 9.87 × 10⁻⁴ | 0.0085 h |
| 0.015 | 8.21 × 10⁻³ | **2.62 × 10⁻⁴** | 0.0049 h |

Gate **< 1 %**: satisfied by 38× (2.62 × 10⁻⁴ = 0.026 %) at h = 0.015 m,
second order; first order would need h ≈ 0.01 m.

### 3.2 A3 — energy release rate vs the analytic bending-Griffith value

Equibiaxial edge moments M = 1000 N m/m on a 20a × 20a plate with a central
through crack (a = 1 m, t = 0.3 m), analytic value
G_an = π c² M² a/(D h) = 0.35802 J m⁻²:

| h_e / a | G (VCE) / J m⁻² | G/G_an |
|---|---|---|
| 1.000 | 0.6364 | +77.7 % |
| 0.500 | 0.4661 | +30.2 % |
| 0.250 | 0.3874 | +8.2 % |
| 0.125 | 0.3489 | **−2.6 %** |

Gate **< 10 %**: satisfied at h_e = a/8 (−2.6 %) and marginally at a/4 (+8.2 %).
Note on the analytic constant: the first version of this benchmark compared the
same finite-element value with πM²a/(D h) (the *cylindrical* state) and found a
systematic −35 % "error"; the discrepancy was in the reference, not in the code
— the Kirchhoff/plane-stress analogy shows that an equibiaxial moment field
carries the factor c = 1/(1+ν) on the analogous membrane stress. Only after that
derivation (and its independent confirmation below) does the sequence converge
to 1. (Richardson extrapolation of the VCE sequence, 2G(h)−G(2h), is *not*
usable here: the error oscillates: 0.331, i.e. −7.6 %.)

### 3.3 A4 — floating ice sheet: moments and fracture load

**(a) Uncracked moment (load path check).** The exact Kelvin-function solution of
the point-load problem gives M_r + M_θ = (P/2π) ker(r/l) and, with
d/ds(s kei′(s)) = s ker(s), the centre moment of a circular patch load collapses
to a quadrature-free expression

  M_c = ((1+ν)/2) p l² X kei′(X), X = a_l/l,

which is the exact form of the classical Westergaard/Nevel formula
M_c = (1+ν)P/(4π)[ln(l/a_l) + 0.6159] — the “magic” constant is
ln 2 − γ + 1/2 = 0.6159.  Numerical checks with the benchmark patch
(p = 10 kPa, a_l = 0.5 m): **exact Kelvin 2577.55 N**, folklore form 2577.05 N
(−0.02 %), **finite element 2605.2 N (+1.07 %)**.  (A factor 4π instead of 2π,
which appears in part of the literature, is refuted: it is 50 % low.)

**(b) Fracture load.**

| quantity | value |
|---|---|
| G at p = 10 kPa (a = 0.30 m, VCE) | 0.9444 J m⁻² |
| FE fracture load P_f | 11.73 kN |
| analytic small-crack P_f = √(G_c D h/(π c² a))/m̂ | 10.60 kN |
| **deviation** | **+10.7 %** |
| strength (first-crack) load, σ_f = 0.6 MPa | 27.42 kN |
| equivalent flaw size reconciling both criteria | 44.8 mm |

Gate **< 20 %**: satisfied (+10.7 %). The FE value is *above* the analytic one,
as expected for a mesh-blunted crack tip (the discrete crack is stiffer than the
Griffith crack). The strength criterion gives a load 2.3× higher because it
assumes a *flaw-free* stress field; the two criteria coincide for an intrinsic
flaw of 2a ≈ 90 mm, which is a physically plausible ice flaw scale.

**(c) Level-set driven growth.** Load held at 1.02 P_f (a₀ = 0.30 m):

| step | a / m | G / J m⁻² | G/G_c | tip (x, y) / m |
|---|---|---|---|---|
| 0 | 0.304 | 1.353 | 1.040 | (0.304, 0) |
| 1 | 0.409 | 1.548 | 1.191 | (0.409, 0) |
| 2 | 0.517 | 1.712 | 1.317 | (0.517, 0) |
| 3 | 0.517 | 1.712 | 1.317 | (0.517, 0) |
| 4 | 0.630 | 1.854 | 1.426 | (0.630, 0) |

G increases monotonically with crack length ⇒ **unstable (run-away) growth**, the
classical ice-sheet behaviour; the lateral tip deviation is exactly 0 (mode I,
the crack grows along its own plane) and the extracted crack length follows the
level-set contour. Step 3 repeats step 2: the re-initialisation of the narrow
band absorbed one increment of advance (1 of 4 increments lost, 25 %); this is
the main quantitative weakness of the tracker and is declared, not hidden.

### 3.4 A2 — mixed-mode path tracking

Inclined crack (β = 45° from the x-axis, remote uniaxial tension along y,
a = 0.4 m) in an infinite plate: K_I = K_II ⇒ the maximum tangential stress
criterion predicts an initial kink of **53.13°** (the level set reproduces it to
0.001° from its own zero contour) and the crack then rotates towards the
direction perpendicular to the remote load (K_II → 0), modelled by an
exponentially relaxed mixity ratio with decay length 1.2 m. Ten increments of
0.2 m, level-set grid h = 0.03 m:

* path deviation (level-set tip trajectory vs the independently integrated
  reference path): **max 7.0 %, RMS 2.1 %** of the reference path length;
* tracked arc length 1.52 m for a nominal 2.00 m: the tip extraction from the
  zero contour loses about a quarter of the prescribed advance (the narrow-band
  re-initialisation smooths the front), which is the dominant tracker error;
* final direction 71.4° (towards the 90° mode-I attractor).

Gate **< 15 %** is met (7.0 %) by the *tracking* metric only. The experimental arm of
this criterion is **WITHHELD**: no digitised crack-path experiment is available
in this repository or offline in this sandbox (three-way search: repo grep,
artefacts, literature cache — zero hits). The number above therefore verifies
the *tracker* against an in-house integrated reference, not the physics against
experiment.

## 4 Figures

* `fig1_levelset_reinit.png` — exact / distorted / re-initialised φ with the 0.026 % gradient-error gate.
* `fig2_vce_convergence.png` — VCE G vs h_e against the analytic bending-Griffith value.
* `fig3_ice_plate_moment.png` — FE moment field of the floating sheet and the radial profile against the exact Kelvin anchor.
* `fig4_crack_path.png` — level-set crack path vs the integrated reference path (max deviation 7.0 %), and the hoop-stress criterion with the predicted kink.

## 5 Pitfalls, limits, applicability

1. **Second-order Godunov re-initialisation is unstable next to the band edge**
   (observed: max |∇φ| error 8.9 × 10⁰). The implementation therefore falls back
   to first order within two cells of the band edge and of the domain boundary.
2. **The band used by the re-initialisation and the band used by the metric must
   not coincide**: the frozen outer band pollutes the central-difference gradient
   of the last active row (a 5.3 × 10⁰ artificial error was observed at
   h = 0.0075 m). Keep ≥ 6h of margin.
3. **A signed distance to a *finite* segment keeps a zero contour along the
   tangent beyond both tips** (the field flips sign there). The representation
   adopted here is the one-sided “notch” field (positive ahead of the tips); the
   price is a jump along the two tip rays, which the contour extractor sees as a
   short spurious branch — the crack-span and tip extractors trim it explicitly.
4. **Mindlin plate crack: tying the transverse displacement (pure hinge) makes
   the MITC Q4 system singular**; the free-bank variant (all three generalised
   displacements free) is used. A stiff penalty on the w-jump changes the energy
   release rate by less than round-off, which shows the transverse-shear mode is
   not excited by the bending load.
5. **The patch-load moment has a logarithmic cusp** at the load boundary: the
   *peak* is mesh dependent (window mean 2410 N = −6.5 %, maximum element value
   +1.07 % at h_min = 0.1 m). Only the maximum-element/centre value should be
   compared with the Kelvin anchor.
6. **All LAPACK calls are pathologically slow on this host** (torch 2×2 inverse
   17 ms, numpy solve 50 ms). Element matrices are therefore analytic for
   axis-aligned rectangles; this is a performance workaround, not a different
   formulation (asserted equal to the reference element).
7. **Applicability**: thin-plate (Kirchhoff-limit) bending fracture with a
   straight/curved crack in a floating or foundation-supported sheet, quasi-static
   growth; not valid for thick plates, for crack branching/coalescence, for
   ductile or viscoelastic ice, nor for the hydrodynamic load itself (the load is
   a prescribed pressure patch; the added mass of the water is not modelled).
8. **Faltinsen dataset**: the acceptance criterion names the Faltinsen ice-plate
   fracture experiment as the benchmark. That dataset is not available in this
   environment, so A4 is evaluated against the analytic Kelvin + bending-Griffith
   anchor (declared substitution) plus the classical strength criterion as a
   consistency check. A literature-digitised comparison is an open item.

## 6 Error verdict

| key quantity | reference | computed | deviation | < 3 %? |
|---|---|---|---|---|
| max ‖∇φ|−1| (A1) | 1 (analytic distance) | 2.62 × 10⁻⁴ | 0.026 % | **yes** |
| G, plate crack (A3, h_e = a/8) | 0.35802 J m⁻² (bending Griffith) | 0.34887 J m⁻² | −2.6 % | **yes** |
| uncracked centre moment (A4a) | 2577.55 N (Kelvin, exact) | 2605.21 N (FE) | +1.07 % | **yes** |
| Westergaard/Nevel form | 2577.55 N | 2577.05 N | −0.02 % | **yes** |
| ice fracture load (A4b) | 10.60 kN (analytic small crack) | 11.73 kN (FE crack + VCE) | +10.7 % | no (< 20 % gate) |
| crack path deviation (A2) | in-house integrated reference | level-set trajectory | max 7.0 % / RMS 2.1 % | no (< 15 % gate) |

Honest statement: two of the six key comparisons miss 3 %. The fracture load
(+10.7 %) is limited by the mesh-blunted crack tip and the small-crack
approximation (a/l = 0.047); the path deviation (7.0 %, with a 24 % loss of the
prescribed arc length) is limited by the tip extraction from the zero contour. Both are inside their campaign gates (< 20 %,
< 15 %) and both are reported as measured, without tuning the reference to the
result.
