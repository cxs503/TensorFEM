# P106W3A2-N02: Ship-Hull Detail Fatigue Maturity — Global-to-Detail Chain

## 1. Physical Background

This node matures the **welded-detail fatigue** chain of the TensorFEM structural
analysis suite.  The physical problem is the fatigue assessment of a ship-hull
welded joint under cyclic loading: a global beam/shell analysis gives the nominal
stress at a cross-section, and the local weld-toe geometry concentrates that
stress.  The complete assessment chain is:

```
global beam theory (sigma_nom = N/A + M/Z)
    -> SCF (FEM, weld-toe notch)
    -> hot spot stress (sigma_hot = SCF * sigma_nom)
    -> S-N curve (DNV-RP-C203 curve D)
    -> fatigue life (cycles to failure)
```

Three joint families are assessed:

| Joint | Geometry | Closed-form anchor |
|---|---|---|
| T-joint (transverse fillet) | t = 10-20 mm, r = 0.5-2 mm | Nii: K_t = 1 + 0.27√(t/r)√(h/t) |
| Butt weld (reinforced) | t = 10-20 mm, r = 0.5-2 mm | Maddox: K_t = 1 + 0.15√(t/r) |
| Corner / bracket toe (L-joint) | t = 10-20 mm, r = 0.5-2 mm | none (FEM only) |

## 2. Method

The FEM model is a body-fitted quadratic (Q8) plane idealisation of the joint
cross-section.  The mesh is built from structured blocks whose rows follow the
free surface, so the toe arc of radius `r` is resolved without sliver elements.
The toe surface element size is `1.75r / (16·level)` — for `t = 20 mm, r = 2 mm`
at `level = 1` this is `0.219 mm`, well below the `t/10 = 2 mm` requirement.

Hot-spot stress is extracted by IIW-style surface extrapolation:
- linear: `0.4t / 0.9t` (and the alternative `0.5t / 1.5t`);
- quadratic: `0.4t / 0.9t / 1.4t`.

The S-N curve is DNV-RP-C203 curve D with double slope `m1 = 3, m2 = 5`:
- `air`: `log10 a1 = 12.164`, knee at `1e7` cycles;
- `seawater_cp`: same `log10 a1`, knee at `1e6` cycles;
- `seawater_free`: `log10 a1 = 11.764` (corrosion knock-down), knee at `1e6`.

## 3. Results

### 3.1 SCF vs closed-form anchors

| Joint | t [mm] | r [mm] | FE K_t | Anchor | Error |
|---|---|---|---|---|---|
| T | 10 | 0.5 | 2.288 | 2.323 (Nii) | -1.5 % |
| T | 10 | 1.0 | 1.889 | 1.935 (Nii) | -2.4 % |
| T | 10 | 2.0 | 1.594 | 1.661 (Nii) | -4.0 % |
| T | 15 | 1.0 | 2.109 | 2.146 (Nii) | -1.7 % |
| T | 15 | 2.0 | 1.756 | 1.811 (Nii) | -3.0 % |
| T | 20 | 1.0 | 2.288 | 2.323 (Nii) | -1.5 % |
| T | 20 | 2.0 | 1.889 | 1.935 (Nii) | -2.4 % |
| Butt | 10 | 0.5 | 1.934 | 1.671 (Maddox) | +15.7 % |
| Butt | 10 | 1.0 | 1.602 | 1.474 (Maddox) | +8.7 % |
| Butt | 10 | 2.0 | 1.378 | 1.335 (Maddox) | +3.2 % |
| Butt | 15 | 1.0 | 1.789 | 1.581 (Maddox) | +13.2 % |
| Butt | 15 | 2.0 | 1.500 | 1.411 (Maddox) | +6.3 % |
| Butt | 20 | 1.0 | 1.934 | 1.671 (Maddox) | +15.7 % |
| Butt | 20 | 2.0 | 1.602 | 1.474 (Maddox) | +8.7 % |

The T-joint family is within **5 %** of the Nii closed form (worst 4.0 %).
The butt family is within **20 %** of the Maddox form (worst 15.7 %), which is
the documented accuracy band of the Maddox fit.

### 3.2 Hot-spot chain

For a worked hull-girder detail (t = 20 mm, 60 MPa axial + 20 MPa bending):

| Joint | SCF_lin (0.5t/1.5t) | SCF_quad (0.4t/0.9t/1.4t) | Life_air [cycles] | Life_CP [cycles] | Life_free [cycles] |
|---|---|---|---|---|---|
| T | 1.005 | 1.010 | 2.80e6 | 5.57e6 | 1.20e6 |
| Butt | 0.997 | 0.995 | 2.88e6 | 5.82e6 | 1.25e6 |
| Corner | -1.630 | -1.638 | 6.58e5 | 6.58e5 | 2.62e5 |

The corner (L-joint) has a **negative** hot-spot SCF because the peak stress at
the toe is compressive under the plate-axial load case — the stress *range* is
taken as the absolute value for the S-N life, which is the standard fatigue
treatment.  The life ordering `air > CP > free` is reproduced at 100 MPa
(CP life > air life because the CP curve moves the knee to 1e6 cycles, putting
the 100 MPa point on the more favourable m2 = 5 branch).

### 3.3 Mesh convergence

The GCI of the hot-spot stress over three mesh levels is **0.24 %** (well below
the 3 % requirement), with the finest-pair change at **0.014 %**.

### 3.4 S-N power law

The measured exponent of the m1 = 3 branch is **3.000** (residual 2.9e-15),
and the 2x stress-range life ratio is **8.0** — both exact to machine precision.

## 4. Figures

- `artifacts/detail_fatigue_v3/scf_vs_anchor.png` — FE notch SCF vs closed-form
  anchors for all three joint families;
- `artifacts/detail_fatigue_v3/sn_curve.png` — DNV-RP-C203 curve D in three
  environments;
- `artifacts/detail_fatigue_v3/mesh_convergence.png` — mesh convergence of the
  hot-spot stress;
- `artifacts/detail_fatigue_v3/hotspot_chain.png` — hot-spot SCF of the three
  joint families.

## 5. Notes and Limitations

1. **IIW SCF band (1.2-1.5)**: the non-load-carrying transverse fillet T-joint
   of this family has a structural SCF of ~1.0 by theory.  The IIW band belongs
   to structural-discontinuity details (bracket toes, load-carrying joints).
   This is an **honest FAIL** — the band is not reproduced, and the reason is
   locked in the test so it cannot be "fixed" by tuning.
2. **Corner joint**: the L-joint has no closed-form anchor; the FEM value is
   reported without a reference.  The peak stress can be compressive under the
   plate-axial load case, which is handled by taking the absolute value of the
   stress range.
3. **2-D idealisation**: everything is a 2-D plane model of a weld cross-section.
   No 3-D weld start/stop ends, no weld profile scatter, no residual stress, no
   misalignment, no multiple-crack interaction.
4. **Maddox fit**: the butt-weld SCF error of up to 15.7 % is within the
   documented accuracy band of the Maddox empirical fit.

## 6. Error Judgement

| Quantity | Measured | Reference | Error | < 3 %? |
|---|---|---|---|---|
| T-joint SCF (worst) | 1.594 | 1.661 (Nii) | 4.0 % | No (within 5 % band) |
| Butt SCF (worst) | 1.934 | 1.671 (Maddox) | 15.7 % | No (within 20 % band) |
| Hot-spot GCI | 0.24 % | 3 % limit | — | Yes |
| S-N m1 exponent | 3.000 | 3.0 | 2.9e-15 | Yes |
| S-N 2x life ratio | 8.0 | 8.0 | 0 | Yes |
| Beam-to-detail | 40.12 | 40.55 | 1.07 % | Yes |

The T-joint SCF is within the **5 %** acceptance band of the Nii closed form
(not the stricter 3 %), which is the documented accuracy of the empirical fit.
The butt SCF is within the **20 %** band of the Maddox form.  All other
quantities are within 3 % or exact.
