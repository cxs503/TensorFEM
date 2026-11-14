# fem_sphere_pressure_membrane — CASE-SPHERE-PRESSURE, uniform-pressure sphere membrane stress

Verdict: **PASS** (11/11 gating gates in `gates_verified.json`). The
predeclared prediction about the *reference's* validity band is **not** a gate: it is kept
under `predictions` with `outcome: "falsified"` (`P1.reference_band_reaches_t_over_R_0.05`) and is graded separately.
Driver: `run_verification.py` · Report builder: `build_report.py` (`--write` / `--check`) ·
Evidence: `result.json`, `gates_verified.json` · Figures: rendered on demand by
`build_report.py --figures` into `artifacts/CASE-SPHERE-PRESSURE/figures/` and **not committed** (the certificate
declares no figure artifact, so a committed figure path here would be an `E_ARTIFACT_MISSING` defect)

Repro (13 s, single process, float64, no cluster needed):

```
python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py
python3 benchmarks/verified/fem_sphere_pressure_membrane/build_report.py --write --check
python3 benchmarks/verified/fem_sphere_pressure_membrane/build_report.py --figures   # optional PNGs
```

Model: `tensorfem.axisym_shell.AxisymShell` (2-node Reissner–Mindlin ring element,
`meridian_radius = R` so the meridian is a true circular arc, SRI on the transverse
shear) driven through `tensorfem.axisym_shell_benchmarks.sphere_pressure`. No new
element and no new solver: the case is a *case-pool* requalification of an existing
capability.

Case: `R = 10.0 m`, `t = 0.04 m`, `R/t = 250`,
`E = 6.825e+07 Pa`, `ν = 0.3`, `p = 1.0 Pa`.

Reference (machine-derived, see the report): `σ = pR/(2t) = 125 Pa`,
`w = pR²(1-ν)/(2Et) = 1.282051e-05 m`.

| facets (pole → equator) | `σ_θ` err vs `pR/2t` | `σ_φ` err | anisotropy | `w` err |
|---:|---:|---:|---:|---:|
| 2 | +1.237e-02 % | +2.429e-02 % | +2.611e-02 % | +5.105e-02 % |
| 4 | +1.128e-03 % | +1.955e-03 % | +2.311e-03 % | +6.960e-03 % |
| 8 | +7.309e-05 % | +1.204e-04 % | +1.525e-04 % | +6.262e-04 % |
| 16 | +4.603e-06 % | +7.001e-06 % | +9.359e-06 % | +4.812e-05 % |
| 32 | +2.828e-07 % | +3.643e-07 % | +5.716e-07 % | +3.313e-06 % |
| 64 | +2.534e-08 % | +2.535e-08 % | +3.479e-08 % | +2.129e-07 % |

Gates: 11/11 gating gates pass (every gate carries a numeric `threshold` and an
explicit `operator`, so `tools/verified_anchor_audit.py` recomputes each verdict). The
predeclared prediction(s) live outside the gating set (`P1.reference_band_reaches_t_over_R_0.05` with
`outcome: "falsified"`); being a prediction it can neither launder nor hide the gating
verdict, and it is never counted as a gate here.

Controls: the same element on the same mesh with straight chords fails at
+2.079e+02 % (8 facets); the merged
flat-facet Shell4 path fails at +1.332e+04 %
(24 facets), reproducing the recorded 13317 %.

Limitations: thin-shell (Reissner–Mindlin) theory only — the case validates the element
against *shell* theory, and the shell formula's own distance from 3-D elasticity is
measured separately (+4.013e-01 % at the case `t/R`).
See `docs/sphere-pressure-membrane-case.md`.
