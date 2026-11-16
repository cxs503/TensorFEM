# Geometrical wetting (Akai 2018 / Wenzel) — honest validation audit

**Anchor** `XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING`
**Module** `src/tensorfem/lbm_wetting.py`
**Runner** `benchmarks/verified/lbm_akai2018_geometrical_wetting/run.py`

## 1. Case and motivation

The `lanl/MF-LBM` `akai2018-geometrical-wetting` pattern validates **geometrical
wetting**: the contact angle is *not* imposed as a free parameter but emerges
from the **geometry** of the solid surface.  The natural analytical benchmark is
the **Wenzel (1936)** relation for a rough surface,

```
cos(theta_eff) = r * cos(theta),   r = 1 + 2*h/w   (2-D pillar)
```

where `theta` is the intrinsic angle on the flat wall and `theta_eff` is the
effective (apparent) angle seen at the edge of a droplet resting on a surface
carrying a rectangular pillar of width `w` and height `h`.

This task is a **validation-debt** audit: the `lbm_wetting` module was already
merged with a `geometric_wetting_xval` / `run_droplet_validation` that *claims*
to reproduce prescribed contact angles.  The audit re-derives the execution
path independently and asks whether the claim is backed by real evidence.

## 2. Method

* **Vectorized bounce-back** (`_lbm_step_vec`): the reference `_lbm_step` used a
  per-cell Python loop for the no-slip wall; the vectorized form uses a
  boolean-mask gather.  G1 proves the two are bit-identical (max |df| == 0).
* **Periodic streaming** (`_shift`): the reference streamed with zero-fill
  (mass lost at open boundaries); the fix uses `np.roll` (periodic wrap) so the
  only mass exchange is through the wall and the wetting reconstruction.
* **Local-density BC** (`apply_boundary_condition`): the reference injected a
  fixed `wall_density` (liquid) into every boundary node; the fix reconstructs
  with the *local* density so the BC only rotates the interface normal.
* **Wenzel helper** (`_wenzel_effective_angle`): exact `acos(r*cos(theta))`,
  verified to machine precision (G2).
* **Geometrical case** (`run_geometrical_wetting_validation`): a droplet on a
  surface with a central pillar; the effective angle is measured and compared
  against the Wenzel prediction.

## 3. Results

| gate | criterion | verdict | evidence |
|---|---|---|---|
| G1 | vectorized step == reference | **PASS** | max \|df\| = 0.0 |
| G2 | Wenzel helper exact | **PASS** | max \|error\| = 0.0 |
| G3 | flat wall: measured == intrinsic (<3%) | **FAIL** | field -> NaN |
| G4 | rough wall: measured tracks Wenzel (<15%) | **FAIL** | field -> NaN |
| G5 | sweep monotone | **FAIL** | every case returns NaN-default |

## 4. Honest finding (blocking)

The full droplet LBM simulation is **numerically unstable**: the density field
blows up to `NaN`/overflow within a few hundred steps.  Critically,
`measure_contact_angle` **returns a hard-coded default `90.0`** whenever it
cannot find a valid interface (fewer than 3 interface points — which is exactly
what happens when the field is `NaN`).  So the existing
`run_droplet_validation` / `geometric_wetting_xval` **reports `passed=True` and
`error_pct≈0` for every case even though the field is garbage** — the
"contact-angle validation" is an **overstated false pass**, not real evidence.

The three fixes delivered here (vectorized step, periodic streaming,
local-density BC) are *necessary* reductions toward stability but are **not yet
sufficient**: the instability is driven by the geometric-wetting reconstruction
itself (the rotated-normal equilibrium injection at boundary nodes), which needs
a proper interface-tracking / surface-tension treatment to be stable.  That is
out of scope for this single audit and is left as an honest `not_run`.

## 5. Error judgement

* G1/G2: exact (0 %), well inside the <3 % / <15 % standards.
* G3/G4/G5: **cannot be judged** — the reference field is `NaN`, so no
  meaningful error can be computed.  This is declared honestly; the existing
  "0 % error / passed" numbers in the module are **not** trustworthy and must
  not be cited as validation evidence.

## 6. Disposition

**Request changes** on the `lbm_wetting` geometric-wetting feature: the
contact-angle validation is a false pass (NaN field + hard-coded 90.0 default).
The vectorized step, periodic streaming, and Wenzel helper are sound and
reusable; the full droplet solver needs a stable interface-reconstruction
scheme before the Akai-2018 / Wenzel benchmark can be meaningfully run.
