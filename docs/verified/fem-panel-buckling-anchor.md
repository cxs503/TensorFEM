# Stiffened-panel buckling anchor (`fem_panel_buckling`)

**Status.** Operator verification PASS; the campaign's ultimate-strength criteria are
**WITHHELD** (b, d, e) or **FAIL** (c) on the evidence this repository can currently
produce, and the report says so instead of inferring them. Every number below is
injected from `benchmarks/verified/fem_panel_buckling/result.json` by
`render_doc.py`; none is transcribed by hand.

Module: `tensorfem.stiffened_panel_buckling`. Driver:
`benchmarks/verified/fem_panel_buckling/run.py`. Tests:
`tests/test_stiffened_panel_buckling.py` (11 cases, ≈30 s).

## 1. Case

Longitudinally stiffened panel unit cell (campaign node
`p127w2-n08-fem-structural-ultimate`): plate field `800 x 400 mm`,
plate thickness `6.0 mm`, tee stiffener web
`150 x 10 mm` plus flange
`50 x 10 mm`, AH36 steel
`E = 206 GPa`, `nu = 0.3`, `sigma_y = 355 MPa` (bilinear hardening `E_t = 1000 MPa` is carried by the nonlinear path, see §5).

| quantity | value |
|---|---|
| plate slenderness `beta = (b/t) sqrt(sigma_y/E)` | 2.7675 |
| classical plate buckling `sigma_cr/ sigma_y` | 0.4720 |
| closed-form plate limiting stress | 210.198 MPa |
| closed-form stiffener limiting stress (governing mode: column) | 351.995 MPa |
| closed-form panel ultimate (area weighted) | 274.651 MPa |
| elastic section modulus (plate extreme fibre) | 344.3771 x 10^-6 m^3 |
| plastic section modulus | 193.5000 x 10^-6 m^3 |

## 2. Method

1. **Mesh.** The unit cell is `span x spacing/2` wide and is taken from
   `stiffened_panel_unit_fe.build_panel_unit`, i.e. exactly the cell the
   repository's nonlinear panel path integrates (plate field, web strip, flange
   strip, symmetry planes at the stiffener centreline and the mid-bay).
2. **Elastic bending operator.** Assembled from the verified MITC4 flat shell
   (`shell4_stiffness(..., plate_shear="mitc")`). The layered Shell4 used by the
   nonlinear path integrates transverse shear at one point and therefore carries
   hourglass modes; its pure bending operator is singular and cannot support an
   eigenproblem.
3. **Prebuckling state.** Under uniform axial end compression a prismatic panel has
   no linear redistribution, so every member carries the same stress and the
   initial-stress resultants are `Nx = sigma t`, `Ny = Nxy = 0`.
4. **Geometric matrix.** `shell4_buckling.shell4_geometric_stiffness` — the verified
   transverse initial-stress operator of the same facet.
5. **Bifurcation.** `(K + lambda K_G) phi = 0` restricted to the bending subspace
   `(w, rx, ry)` (the membrane and drilling blocks are uncoupled from it in a linear
   facet and `K_G` touches only `w`, so the restriction is exact). The pencil is
   mapped to a standard symmetric problem with the Cholesky factor of `K` and solved
   with LAPACK; the residual of every reported mode is recorded in the evidence.

## 3. Results

### 3.1 Simply supported plate field against the closed form

| mesh | elements | bending dofs | `sigma_cr` [MPa] | vs closed form | mode residual |
|---|---|---|---|---|---|
| 10x5 | 50 | 353 | 176.878 | +5.557 % | 5.5e-18 |
| 20x10 | 200 | 1303 | 169.410 | +1.100 % | 7.6e-18 |
| 40x20 | 800 | 5003 | 167.411 | -0.093 % | 1.1e-10 |
| 12x6 | 72 | 495 | 173.797 | +3.718 % | 6.4e-18 |
| 24x12 | 288 | 1851 | 168.641 | +0.641 % | 7.7e-18 |
| 32x16 | 512 | 3235 | 167.832 | +0.159 % | 7.8e-18 |

Closed form: `k = 4.000` (m = 2,
half wavelength 400 mm), `sigma_cr =
 167.566 MPa`.

Grid convergence on the `10x5 / 20x10 / 40x20` ladder: observed order
`p = 1.901`, `GCI_fine = 0.5459 %`,
Richardson limit `166.680 MPa`. The observed order near 2
is the expected second-order behaviour of the MITC4 bending operator and is an
independent check that the implementation is not accidentally first order.

### 3.2 Stiffened unit cell

| mesh | elements | `sigma_cr` [MPa] | `k_eff` | `sigma_cr/sigma_y` | first mode |
|---|---|---|---|---|---|
| 8x2 | 32 | 360.245 | 8.599 | 1.0148 | 360.245 MPa |
| 16x4 | 96 | 304.976 | 7.280 | 0.8591 | 304.976 MPa |
| 32x8 | 320 | 293.814 | 7.014 | 0.8276 | 293.814 MPa |

Observed order `p = 2.308`, `GCI_fine =
 1.2018 %`.

Clamped-edge finite-element companion (same mesher and element, plate with both long
edges clamped, `40x20`): `sigma_cr =
 292.855 MPa`, `k_eff = 6.991`. The cell
sits at `k_eff = 7.014`, i.e. within
 0.33 % of the clamped
limit: the 150x10 web plus 50x10 flange restrain the plate edge almost as a clamp,
so the campaign's `k = 4` simply-supported value is **not** the buckling load of this
panel and cannot be used as its oracle. The cell first bifurcates at
 `0.828 sigma_y`, i.e. above the area-weighted closed-form
ultimate: this geometry is collapse-governed, not elastic-buckling-governed.

## 4. Gates

| gate | quantity | value | threshold | verdict |
|---|---|---|---|---|
| `plate_finest_vs_classical` | \|sigma_cr(FE) / sigma_cr(classical) - 1\| on the finest simply supported plate mesh | 0.000929195 | <= 0.05 | PASS |
| `plate_medium_vs_classical` | \|sigma_cr(FE) / sigma_cr(classical) - 1\| on the middle plate mesh | 0.0110033 | <= 0.05 | PASS |
| `plate_gci_fine` | GCI_fine of the simply supported plate buckling ladder | 0.0054586 | <= 0.03 | PASS |
| `panel_gci_fine` | GCI_fine of the stiffened-cell buckling ladder | 0.012018 | <= 0.03 | PASS |
| `panel_at_least_simply_supported` | k_eff of the stiffened cell vs the idealised simply supported coefficient | 3.01368 | >= 0 | PASS |
| `panel_at_most_clamped` | k_eff of the stiffened cell vs the clamped-edge finite-element companion | 0.0270929 | >= 0 | PASS |

Overall operator verdict: **PASS**.

### 4.1 Campaign criteria

| criterion | status | evidence |
|---|---|---|
| (a)_buckling_vs_classical | **PASS** | simply supported plate 40x20; measured 0.0929 % vs 5.0 % |
| (b)_ultimate_vs_Paik2007_experiment | **WITHHELD** | primary experimental table unavailable; repository policy forbids quoting it from memory (see benchmarks/paik_thayamballi_2003_specimens.json) |
| (c)_postbuckling_path_vs_experiment | **FAIL** | the panel path in the repository is geometrically linear: no limit point, no descending branch, no energy absorption to compare (measured: applied stress 0.325 sigma_y with maximum plastic strain 0.0e+00 and a deflection-versus-load proportionality residual of 1.8e-14) |
| (d)_smith_Mkappa_vs_fem_full_model | **WITHHELD** | no geometrically nonlinear full-section FE exists in the repository, so there is no full-model M-kappa to compare the Smith curve against |
| (e)_csr_section_modulus | **WITHHELD** | the rule design bending moment of the applicable CSR edition is not available; only the actual section moduli are reported |
| (f)_mesh_convergence_gci | **PASS** | simply supported plate ladder 10x5 / 20x10 / 40x20, r = 2; measured 0.5459 % vs 3.0 % |

## 5. The limitation this anchor had to establish

The campaign asks for a post-buckling limit point and a Smith `M-kappa` curve from
finite elements. `tensorfem.layered_shell4_plasticity` is a small-strain,
small-rotation layered shell whose element tangent contains the material tangent
only: there is **no geometric stiffness**, so the panel path cannot lose stability.
The driver records the measurement of that fact:

| quantity | value |
|---|---|
| arc-length path (mesh 6x2, 30 steps, ds = 1.2e-04 m, 3 layers) | executed |
| final average stress | 115.466 MPa (0.3253 `sigma_y`) |
| applied stress / classical plate buckling stress | 0.689 |
| maximum equivalent plastic strain | 0.0e+00 |
| deflection-versus-load proportionality residual | 1.82e-14 |
| Newton iterations per step | [2] |
| limit point detected | False |

The trace is *exactly* proportional (residual at the round-off level) while the
applied stress passes 0.69 of the classical plate buckling stress with zero plastic
strain: no bifurcation is present at all. A longer run on the twelve-by-four cell
(120 steps) behaves identically and reaches `0.635 sigma_y` with two Newton
iterations per step and zero plastic strain, so this is a property of the element,
not of the run length. An element with a geometric term must instead amplify the
seeded imperfection like `1/(1 - sigma/sigma_cr)` and depart from proportionality
long before its critical load.

Consequences, stated plainly:

* Campaign (c) (post-buckling path and energy absorption) is an **honest FAIL**: there
  is no descending branch to integrate, so no energy absorption exists to compare.
* Campaign (b) (ultimate strength versus the Paik & Thayamballi table) and (d) (Smith
  `M-kappa` versus a full-model finite-element `M_u`) are **WITHHELD**: the primary
  experimental table is not available in this environment and repository policy
  forbids quoting it from memory, and no geometrically nonlinear full-section model
  exists to compare against.
* Campaign (e) is a **chain** quantity here: the actual section moduli are reported
  (§1), the required modulus needs the applicable CSR edition's design bending
  moment, which is not assumed.

What the anchor *does* certify is the buckling operator, against an oracle the solve
path never consumes, with a measured convergence order and mode residuals.

## 6. Error statement (campaign demand: is each key quantity within 3 %?)

1. Simply supported plate `sigma_cr` at `40x20`: -0.093 % — **within 3 %** (the 3 % line is crossed already at `12x6`).
2. GCI of that ladder: 0.5459 % — **within 3 %**.
3. Stiffened cell GCI: 1.2018 % — **within 3 %**.
4. Cell `sigma_cr` versus the closed-form simply-supported plate value: +75.342 % — **not within 3 %**, and that
   is the physics: the closed form omits the stiffener's rotational restraint. The
   comparison that is meaningful (a clamped-edge companion run with the same element)
   closes to 0.33 %.
5. Ultimate strength versus experiment: **no error can be stated**, the reference table
   is WITHHELD; the anchor reports no ultimate strength at all rather than a number
   that would look like agreement.

## 7. Limitations

* Linearised (eigenvalue) buckling: prebuckling resultants are not updated, so the anchor carries no post-buckling reserve and no ultimate strength.
* Rotational geometric stiffness terms are omitted; the simply supported plate agreement with the closed form quantifies that omission.
* The repository's panel path (stiffened_panel_unit_fe) is geometrically linear and cannot supply a limit point, so campaign criteria (b), (c) and (d) are WITHHELD.
* The buckling operator uses the MITC4 flat shell rather than the layered Shell4 of the nonlinear path, because the layered element's one-point shear field carries hourglass modes that make the pure bending operator singular.
* No experimental value is quoted anywhere in this anchor.
* The panel path's own bilinear hardening (`E_t = 1000 MPa`) is a parameter of the
  layered element; this anchor does not exercise plasticity at all.
* The buckling operator is a *linearised* analysis: the prebuckling resultants are
  frozen, so neither load redistribution before collapse nor the interaction of
  local buckling with yielding is represented.

## 8. Reproduce

```bash
PYTHONPATH=src python benchmarks/verified/fem_panel_buckling/run.py panel path
PYTHONPATH=src python benchmarks/verified/fem_panel_buckling/run.py plate0 plate1 plate2
PYTHONPATH=src python benchmarks/verified/fem_panel_buckling/run.py clamped0 clamped1
PYTHONPATH=src python benchmarks/verified/fem_panel_buckling/run.py   # assemble + figures
PYTHONPATH=src python benchmarks/verified/fem_panel_buckling/render_doc.py
PYTHONPATH=src pytest -q tests/test_stiffened_panel_buckling.py
```

Each stage caches its own JSON under `_cache/`, so a slow mesh can be solved in its
own call; delete `_cache/` (or pass `--force`) to regenerate from scratch.

Figures: `benchmarks/verified/fem_panel_buckling/figures/buckling_convergence.png`,
`.../buckling_modes.png`,
`.../linear_path_evidence.png`.

