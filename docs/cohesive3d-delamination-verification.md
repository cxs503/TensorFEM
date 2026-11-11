# 3-D cohesive element: specimen-level verification (DCB / ENF)

This note documents `benchmarks/cohesive3d_delamination.py`, the specimen-level
verification面 of the 3-D cohesive element (`src/tensorfem/cohesive3d.py`), and
the repair of the three defects that made its first version unusable (cross-audit
`docs/m14958_cohesive3d_delamination_audit.md`, which graded the feature **L1**
and recommended `reopened`).

Element-level verification lives in `tests/test_cohesive3d.py` (44 tests:
traction-separation law, consistent tangent, patch tests, rigid-body objectivity,
mesh objectivity) and in `benchmarks/cohesive3d_verification.py`.  What was
missing is a check that the element reproduces the **global** fracture response
of a delaminating specimen.

## 1. What is verified, and against what

| route | specimen | anchor (closed form) | threshold |
|---|---|---|---|
| `dcb_linear_compliance` | DCB, a = 45 mm | `C = 4a³/(E b h³)` | 2 % |
| `dcb_linear_serr` | DCB, a = 45 mm | `G_I = 6P²a²/(E b²h³)`, from the FE compliance derivative `P²/(2b) dC/da` | 5 % |
| `enf_linear_compliance` | ENF, a = 45 mm < L = 75 mm | `C = (2L³+3a³)/(96 E I)` | 15 % |
| `enf_linear_serr` | ENF, a = 45 mm | `G_II = 9P²a²/(16 E b²h³)`, same energy route | 15 % |
| `diagnostics.dcb_cohesive_zone_propagation` | DCB, growing damage front | LEFM onset load `√(G_Ic E b²h³/(6a²))` | diagnostic only |

Specimen: LX = 150 mm, b = 12 mm, h = 3 mm per arm, E = 70 GPa, ν = 0.33,
k_n = 1e16 Pa/m, k_s = 4e15 Pa/m, t_n0 = 80 MPa, G_Ic = 300 J/m², G_IIc = 800 J/m²,
BK η = 2.2.  Verification mesh 300×4 (0.5 mm along x), process zone
`l_ch = E G_Ic/t_n0² = 3.3 mm`.

The SERR route deliberately does **not** evaluate crack-tip fields: it takes the
*energy* definition `G = P²/(2b) dC/da` from two FE solves whose crack front is
moved by ±one element.  That removes the nodal-force/opening bookkeeping that
the first version got wrong by 13 orders of magnitude.

## 2. Results (from `benchmarks/cohesive3d_delamination.json`)

| criterion | ref | FE | rel. error | pass |
|---|---|---|---|---|
| DCB compliance | 1.60714e-5 m/N | 1.60779e-5 m/N | 0.04 % | ✓ |
| DCB SERR | 4.46429e-2 J/m² | 4.28592e-2 J/m² | 4.0 % | ✓ |
| ENF compliance | 6.15699e-6 m/N | 6.64268e-6 m/N | 7.4 % | ✓ |
| ENF SERR | 4.18527e-3 J/m² | 4.69179e-3 J/m² | 11.8 % | ✓ |

**Is every key quantity within 3 %?** No, and that is reported as such: the DCB
compliance is at 0.04 % but the three others sit at 4–12 %.  The DCB SERR
residual (4 %) is the finite-difference + root-compliance discretisation error;
the ENF residuals (7 %/12 %) are dominated by the beam-theory **model form**
itself (the closed form neglects transverse shear, root rotation and the
eccentric load application), which is why the ENF thresholds — predeclared at
15 %, the platform precision line — are wider than the DCB ones.  A
discretisation study (300×4 vs 200×4 vs 150×6 vs 200×6) shows the ENF compliance
converging to ≈ +7 % above the closed form, i.e. the residual is not mesh error.

Cohesive-zone propagation (diagnostic): peak load 64.4 N vs LEFM 82.0 N
(−21 %), damage front advanced 1.9 % of the ligament, Newton converged through
the growing front but stops at the limit point.  The −21 % gap is the finite
process-zone size (`l_ch/a ≈ 7 %`) plus the coarse ramp, and is *not* held as a
criterion: at that resolution a 15 % band would measure the process-zone length,
not the element.  Recorded as a diagnostic, with the mesh in the artifact.

## 3. The three defects this repair fixes

1. **DCB rig inverted.** The first version clamped the mouth (`x = 0`) and
   loaded the bonded root (`x = LX`), with the bonded ligament at
   `x ∈ [LX−a, LX]`.  The two arms were therefore held together over the whole
   loaded region: FE compliance 1.12e-9 m/N against 1.607e-5 m/N (99.99 % error)
   and a crack-tip opening of 1e-10 m.  Fix: bonded ligament from `x = a`,
   clamped far end, open at the mouth.
2. **Prescribed DOFs left inside the unknown set.** `simulate_displacement`
   wrote the mouth displacement into `u` but never removed those DOFs from
   `free`, so the Newton solve immediately relaxed them back — the specimen never
   saw a load (peak load 0 N, `converged_all = False` in every mesh).  Fix:
   `free = setdiff(unknown, fixed ∪ prescribed)`.
3. **ENF was not an ENF.** The crack length was 105 mm > L = 75 mm, so the load
   crossed the crack and the top arm acted as a cantilever (compliance 2× the
   closed form); and the pre-crack faces were free, so the arms passed through
   each other (5.4 µm penetration measured), letting the "beam" split into two
   independent cantilevers (+24 % compliance).  Fix: `a = 45 mm < L` and a
   unilateral penalty contact on the pre-crack node pairs, resolved by an active
   set in both the linear solve and the Newton loop.

Also removed: the mixed-mode MMB check of the first version.  Its `got` and
`ref` were built from the same linear VCCT numbers (self-verifying, audit
finding M1) and the rig needs the same contact treatment; mixed-mode SPECIMEN
verification is left to a follow-up.  The mixed-mode *law* (BK) remains covered
by the element suite against the 2-D parent implementation.

## 4. Reproduction

```bash
python benchmarks/cohesive3d_delamination.py          # ~8 min, writes json + png
python benchmarks/cohesive3d_delamination.py --quick  # ~55 s coarse-mesh smoke
python -m pytest tests/test_cohesive3d_delamination.py -q   # ~55 s, 5 tests
```

`tests/test_cohesive3d_delamination.py` is the CI-sized regression gate: the
same code paths on a coarse mesh with the pre-declared `QUICK_TOL` bands, plus
three structural assertions that fail on exactly the defects above (DCB
compliance > 1e-6 m/N, ENF contact penetration < 1e-8 m, cohesive-zone Newton
converges with damage growth and a non-zero load), plus a self-consistency check
of the stored artifact whose references are recomputed from the closed forms.

## 5. Limitations

* Anchors are beam-theory closed forms (ASTM D5528 / D7905 idealisations), not
  experimental data: the routes reach "closed-form anchored", not
  "experiment anchored".
* Mode-II and mixed-mode *specimen* routes carry a 7–12 % model-form residual;
  no mixed-mode specimen check ships yet.
* The cohesive-zone route is diagnostic; a steady-state energy criterion
  (dissipated energy per unit crack area over a known growth increment, ASTM
  D5528 area method) is the natural next step.
* Runtime of the production route is ~8 min (dominated by the per-element
  Python loop over `cohesive_hex8_response`, ~11 ms per element).