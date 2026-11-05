# Stiffened-panel (orthotropic) free vibration: verification record

Target: `tensorfem.stiffened_panel_vibration` + the orthotropic rigidity path of
`tensorfem.plate.q4_mindlin_stiffness`.  Evidence file:
`benchmarks/stiffened_panel_vibration_evidence.json`
(regenerate with `OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/stiffened_panel_vibration.py`).
Tests: `tests/test_stiffened_panel_vibration.py`.

## 1. Why this case exists

Before this change the library could solve

* orthotropic **statics** of a stiffened ship panel
  (`marine_structures.stiffened_panel_sine_benchmark`, Navier oracle), and
* **isotropic** plate free vibration (`plate_vibration`, verified against the
  Navier sine solution and Leissa's clamped coefficient),

but it had no orthotropic vibration path at all -- i.e. no way to compute the
natural frequencies of a real stiffened deck or side shell, which is the entry
quantity of every panel vibro-acoustic, slamming-fatigue screening and
springing check.  That is a capability gap, and it is now closed with a
reference that is *not* the solver's own output.

## 2. Method

1. **Element.** `q4_mindlin_stiffness` accepts an optional
   `PlateRigidity(dx, dy, dxy, dt, sx, sy)`.  With `rigidity=None` the
   historical isotropic matrix is returned bit-for-bit (verified: the maximum
   relative deviation is `0.0`, not merely small, for both the `sri` and the
   `mitc` shear treatments).
2. **Smearing.** `smeared_panel_rigidity` implements Timoshenko's equivalent
   orthotropic plate for flat-bar stiffeners on one face:
   `dx = Dp + E (Is + A e^2)/s` with `e=(h+depth)/2`, `dxy = nu sqrt(dx dy)`,
   `dt = G h^3/12`, `sx = sy = kappa G h`.  The attached-plating area is *not*
   added twice (the plate's own `Dp` already carries it).
3. **Reference.** The Navier modes of the *first-order shear deformation*
   orthotropic plate are exact eigenfunctions, so the frequency of mode `(m,n)`
   is the smallest root of the 3x3 generalised problem
   (see the module docstring).  Using FSDT instead of Kirchhoff removes the
   shear bias from the comparison instead of bounding it -- for this panel the
   Kirchhoff value is 0.036% high at `a/h=100` but 3.4% high at `a/h=10`.
4. **Mode matching.** Each computed mode is labelled by its normalised overlap
   with `sin(m pi x/a) sin(n pi y/b)`; every overlap in the tables below is
   `1.0000`, so the comparison is made mode-by-mode and not on a sorted
   spectrum that could be scrambled by mode crossing.

## 3. Results

### 3.1 Panel A -- synthetic orthotropic panel, `dx/dy = 6`

Steel `E = 200 GPa`, `nu = 0.3`, `rho = 7850 kg/m^3`, `h = 10 mm`,
`a x b = 1.2 x 0.8 m`, SS1 on all edges,
`dx = 1.099e5`, `dy = 1.832e4`, `dxy = 1.345e4`, `dt = 6.410e3 N m`,
`sx = sy = 6.41e8 N/m`, `H = dxy + 2 dt = 2.63e4 N m`.

| mode (m,n) | FSDT Navier [Hz] | FE 8x6 | FE 12x9 | FE 16x12 | error at 16x12 |
|---|---|---:|---:|---:|---:|
| (1,1) | 69.689 | 71.386 (2.43%) | 70.435 (1.07%) | 70.107 | **0.60%** |
| (1,2) | 176.728 | 198.956 (12.58%) | 186.005 (5.25%) | 181.838 | **2.89%** |
| (2,1) | 187.208 | 200.680 (7.20%) | 193.002 (3.10%) | 190.430 | **1.72%** |
| (2,2) | 278.141 | 306.850 (10.32%) | 290.316 (4.38%) | 284.882 | **2.42%** |

Observed convergence order of the fundamental over 8x6 -> 16x12: **2.02**
(slope-2 reference in `docs/assets/stiffened-panel-convergence.png`); the error
sequence is monotone decreasing from above, which is the expected behavior of a
consistent displacement element against an exact reference.

Mode-shape check: the FE (1,2) mode against the analytic sine product differs by
**1.98e-7** in the max norm after sign alignment
(`docs/assets/stiffened-panel-mode-shape.png`).  On this mesh the sine mode is
represented to machine-level accuracy, so the remaining frequency error is a
bending-resolution error, not a shape error.

### 3.2 Panel B -- deck grillage, 3.2 x 2.4 m, 12 mm plating, 200x12 flat bars at 0.8 m

Smeared rigidity: `dx = dy = 8.773e6 N m` (277 times the unstiffened plate
rigidity `D = 3.16e1 N m`), `dxy = 2.632e6`, `dt = 1.108e4`, `sx = sy = 7.69e8`.
Mesh 20x16.

| mode (m,n) | FSDT Navier [Hz] | FE [Hz] | error |
|---|---:|---:|---:|
| (1,1) | 106.01 | 106.43 | **0.40%** |
| (2,1) | 221.71 | 223.88 | **0.98%** |
| (1,2) | 336.88 | 342.71 | **1.73%** |
| (2,2) | 411.45 | 417.67 | **1.51%** |

Sensitivities (closed-form f1 only, no FE cost):

| variation | f1 [Hz] | change |
|---|---:|---:|
| unstiffened plate (same `h`) | 7.81 | -92.6% vs stiffened |
| smeared panel, flat bars | 106.01 | reference |
| plating 12 -> 14 mm | 99.14 | -6.48% |
| stiffener torsion included in `dt` | 106.19 | +0.17% |

## 4. Error judgement against the 3% criterion

| quantity | reference | worst error | < 3%? |
|---|---|---|---|
| Panel A first 4 modes, 16x12 | closed-form FSDT Navier | 2.89% ((1,2)) | **yes** |
| Panel A fundamental, 16x12 | closed-form FSDT Navier | 0.60% | **yes** |
| Panel B ship grillage first 4 modes, 20x16 | closed-form FSDT Navier | 1.73% ((1,2)) | **yes** |
| isotropic reduction of the FSDT reference vs Kirchhoff Navier | `kirchhoff_angular_frequency` | 0.036% at `a/h=100` | **yes** |
| orthotropic element vs isotropic element (identity) | built-in isotropic matrix | 0.0 | **yes** (exact) |
| FE (1,2) mode shape vs analytic sine | `sin sin` product | 1.98e-7 | **yes** |

**Absolute** frequency accuracy of a real stiffened panel against measurement or
a shell/3-D reference is **not** established here: no experimental or
full-3-D reference for a stiffened panel at these dimensions was available in
this repository, so the 3% statement above is strictly "FE discretisation vs the
closed form of the *same* smeared rigidity".  It says nothing about the smearing
error itself.

## 5. Caveats and scope limits (do not over-read)

* **Smearing validity.** `smearing_wavelength_warning` reports the modes whose
  half wavelength drops below `2 x` the stiffener spacing.  For Panel B those are
  `(1,2)`, `(2,2)`, `(3,1)`, `(3,2)`, `(1,3)`, `(2,3)`, `(3,3)` at 0.8 m spacing:
  the nodal lines start to fall *between* stiffeners and the equivalent
  orthotropic model is no longer justified even though the FE-vs-Navier error
  stays inside 3%.  The numbers in 3.2 are therefore a discretisation check, not
  a licence to use the smeared model for those modes.
* **Effective breadth** of the attached plating is not recomputed; `dx`/`dy` are
  declared inputs.  With the eccentric-area formula used here, adding effective
  plate breadth can only *increase* `dx`/`dy`, i.e. the tabulated frequencies are
  on the low side of a bread-based idealisation.
* **Stiffener torsion** is excluded by default (`twist_stiffeners=False`); the
  option exists but its per-unit-width convention is not qualified by any
  independent reference.  Its effect on f1 in Panel B is +0.17%.
* **Not covered:** non-uniform thickness, bulb/angle profiles (only flat bars),
  in-plane prestress, added mass / fluid coupling, damping, stiffener pull-out,
  clamping (only SS1 is verified here), and the free/clamped boundaries of the
  isotropic module are untouched by this work.
* **Element rank.** The default SRI Q4 shear rule still has the two documented
  spurious mechanisms (`docs/element-rank-deficiency-audit.md`); the solver
  fails closed if a near-zero eigenvalue survives on a supported boundary, and
  that guard is active in every run above.
* **Cost.** Dense algebra: the 20x16 Panel B solve is ~1.7e3 DOF; beyond about
  24x24 elements the isotropic module's sparse-solver limitation applies
  unchanged.