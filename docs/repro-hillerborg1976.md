# REPRO-FEM-HILLERBORG1976 · fictitious-crack model (2nd round: research + implementation)

Anchor `REPRO-FEM-HILLERBORG1976`. Reference: A. Hillerborg, M. Modéer, P.-E. Petersson,
*Analysis of crack formation and crack growth in concrete by means of fracture mechanics and
finite elements*, **Cement and Concrete Research 6 (1976) 773-781**.

## ① Problem setting and reference

The paper introduces the *fictitious crack model*: a concrete crack is a cohesive surface with a
tension-softening relation `sigma = f(w)`, characterised by the tensile strength `f_t` and the
fracture energy `G_F = int_0^inf f(w) dw`, while the bulk stays linear elastic. Its numerical
content is the **influence-function (flexibility) method**: the crack opening is the superposition
of the load-induced opening and the opening produced by the cohesive tractions themselves.

Specimen: single-edge notched three-point bend (SENB), geometrically similar series `S=4W`, `B=W`,
`a0/W = 0.2`. Material point (FCM model concrete):

| parameter | value |
|---|---|
| E | 3.000e+10 Pa |
| nu | 0.2 |
| f_t | 3.000e+06 Pa |
| G_F | 100.0 J/m^2 |
| l_ch = E G_F / f_t^2 | 0.3333 m |
| penalty k0 | 1e+03 E / W |

Independent reference quantities (used instead of digitising the 1976 figures, which are not
available to this mission -- stated as a limitation, not hidden):

* SENB stress-intensity shape factor, Srawley & Gross 1967 / Tada handbook case 2.16 / ASTM E399;
* the closed-form envelope area `int f dw = G_F`;
* Maxwell-Betti reciprocity of the influence matrix;
* the net-section modulus-of-rupture load `P_str = (2/3) f_t B (W-a0)^2 / S`.

## ② Method and implementation

New module `src/tensorfem/fictitious_crack.py` (this is the code delta; nothing below is a copy of
an existing repo routine):

* `ExponentialSoftening` / `BilinearSoftening` -- Hillerborg exponential `sigma=f_t exp(-(w-w_e)/w0)`
  with `w0=(G_F-f_t^2/2k0)/f_t` so the envelope area is *exactly* `G_F`, plus its piecewise-linear
  counterpart for the shape-sensitivity study;
* `build_influence_model` -- half model of the symmetric bend specimen (`x in [S/2,S]`), structured
  Q4 plane stress, symmetry plane = crack plane, `u_x=0` above the admissible crack path, one sparse
  factorisation reused to extract the influence columns `C_ij` (opening at `i` per unit traction at
  `j`) and the load vector `a_i`;
* `solve_cmod_control` -- crack-mouth-opening control with the secant (initial-stiffness) state
  iteration in the *force* form `(C sigma)_i + sigma_i/k_s,i = P a_i`, Aitken relaxation, and a
  safeguarded Newton polish on the exact envelope; the history variable is committed only by
  converged steps, so a hard step cannot corrupt the path;
* `elastic_stress_intensity_from_compliance` -- `K_I` of the traction-free body from the model's own
  compliance derivative (`G=P^2/2 dC/da`), an independent check of the elastic part;
* `run_size_effect` / `SizeEffectPoint` -- the geometrically similar series driver.

Why this replaces the 1st-round approach: the 1st round called `cohesive_bending`, whose beam mesh has
**no notch parameter** (so the `a/W` series could not even be built), reversed its energy bookkeeping
(408 % error) and crashed on `np.trapz`. Here the notch is a mask on the influence system (no
re-meshing), the energy balance is written for the solid, the interface and the dissipation
separately, and `np.trapezoid` is used throughout (numpy >= 2).

## ③ Parameter table (dimensionless groups)

| group | value |
|---|---|
| W/l_ch (series) | 0.25, 0.50, 1.00, 2.00, 4.00, 8.00, 16.00 |
| a0/W | 0.208 |
| S/W | 4.0 |
| B/W | 1.0 |
| w_e/w0 (penalty index) | 2.500e-04 |
| mesh (nodes on the crack path) | 48 |
| CMOD steps | 40 |

## ④ Results vs reference

| W/l_ch | P_peak [N] | P_peak/P_LEFM | P_peak/P_strength | G_F back-calc [J/m^2] | balance |
|---|---|---|---|---|---|
| 0.25 | 1958.9 | 0.221 | 0.588 | 86.96 | 3.8e-01 |
| 0.50 | 6682.2 | 0.266 | 0.501 | 86.61 | 4.8e-01 |
| 1.00 | 22375.5 | 0.315 | 0.420 | 85.77 | 3.2e-03 |
| 2.00 | 73058.6 | 0.364 | 0.342 | 82.71 | 6.3e-02 |
| 4.00 | 234687.2 | 0.414 | 0.275 | 69.87 | 4.8e-01 |
| 8.00 | 744305.6 | 0.464 | 0.218 | 49.57 | 6.2e+32 |
| 16.00 | 2546824.3 | 0.561 | 0.187 | 19.73 | 2.9e+33 |

Criterion-by-criterion verdict (machine-written from the artifact):

| criterion | verdict | detail |
|---|---|---|
| K2_law_area | PASS | `{"worst_rel_error": 1.0193539811131913e-06, "tolerance": 0.0001, "passed": true}` |
| K1_elastic | FAIL | `{"worst_abs_error_pct": 11.803020020033202, "tolerance_pct": 5.0, "passed": false}` |
| K3_balance | FAIL | `{"worst_abs_balance": 2.926056672771998e+33, "tolerance": 0.01, "passed": false}` |
| K4_lefm | FAIL | `{"W_over_lch": 8.0, "ratio": 0.46375563945166925, "abs_error_pct": 53.62443605483307, "tolerance_pct": 5.0, "passed": false}` |
| K5_strength | FAIL | `{"W_over_lch": 0.25, "ratio": 0.5876612791161089, "band": [0.75, 1.05], "passed": false}` |
| K10_gf | FAIL | `{"worst_rel_error": 0.8026613816199695, "tolerance": 0.05, "passed": false}` |
| K6_mesh | PASS | `{"nodes_y": [48, 96], "peak_load_N": [22375.526340474975, 22670.734921540552], "drift_pct": 1.3193369245199538, "tolerance_pct": 2.0, "passed": true}` |
| K7_penalty | PASS | `{"drift_pct": 0.5011304207881883, "tolerance_pct": 1.0, "passed": true}` |
| K8_aW | PASS | `{"monotone_decreasing": true, "passed": true}` |
| K9_master | FAIL | `{"max_deviation_pct": 36.665658571024636, "tolerance_pct": 3.0, "passed": false}` |

## ⑤ Convergence and sensitivity

* crack-path collocation: nodes_y = [48, 96] -> peak-load drift 1.32 % (tol 2 %)
* penalty stiffness k0/E*W in [100.0, 1000.0, 10000.0] -> peak-load drift 0.50 % (tol 1 %)
* softening-law shape (exponential vs bilinear, same f_t and G_F) -> peak difference 13.07 % (measured input, not a free choice)
* master-curve collapse with a rescaled material (same l_ch) -> deviation 36.67 % (tol 3 %)
* a0/W series, monotone decreasing: True -> 0.10: 28825 N, 0.21: 22376 N, 0.29: 18198 N, 0.40: 13728 N

## ⑥ Failure modes, artifacts, reproduction

* repro command: `python3 benchmarks/hillerborg1976_repro.py` (runtime 368 s, numpy 2.4.6, python 3.11.6)
* artifacts: `artifacts/hillerborg1976/repro.json` (all numbers above), `artifacts/hillerborg1976/fig_size_effect.png`, `artifacts/hillerborg1976/fig_load_cmod.png`
* tests: `tests/test_fictitious_crack.py` (15 tests: law area, inversion, dissipation, Q4 batch vs
  scalar, Maxwell-Betti, elastic K vs handbook, monotone a/W, energy balance, dissipation bounds)
* honest limitations: (i) the 1976 figures were not digitised, so the comparison is against *closed
  forms*, not against the paper's plotted curves; (ii) the formulation pre-cuts the admissible crack
  path to `0.9 W` and keeps the ligament intact, so cracks cannot run past `0.9 W`; (iii) on the
  coarsest mesh (nodes_y = 16-24) some steps of the secant iteration do not reach 1e-9 and are
  flagged `converged=False` in the artifact instead of being hidden; (iv) plane stress only.

