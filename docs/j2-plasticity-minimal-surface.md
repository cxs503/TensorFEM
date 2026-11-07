# J2 (von Mises) elastoplasticity — minimum verifiable surface

**Mission**: IN14-D4 / m4951 — fills the m605 gap “no elastoplastic material”
in TensorFEM. **Component**: `tensorfem.j2_elastoplasticity`
(`src/tensorfem/j2_elastoplasticity.py`). **Evidence grade**: **REAL** for the
material-point surface below (closed-form radial return + closed-form consistent
tangent, machine-checked against analytic anchors); **ROADMAP** for cyclic /
kinematic hardening / multi-dimensional complex paths (declared, not claimed).

## 1. Case introduction

Ship structural assessment needs a local metal-plasticity model before any hull
girder, frame or bracket limit-state claim can be made. The classical choice for
ductile structural steel is the **von Mises ($J_2$) yield surface with linear
isotropic hardening**, integrated by a **backward-Euler radial return mapping**
and linearised with the **algorithmic (consistent) tangent** — the tangent is
what makes the global Newton iteration quadratically convergent, so it is part
of the deliverable, not an optimisation.

The verification target is a *material point* (integration point), because a
constitutive law must first be correct in isolation: geometry / element errors
must not be able to hide a wrong return mapping. The material parameters used
throughout are those of a generic structural steel:

| Quantity | Value | Note |
|---|---|---|
| Young's modulus $E$ | 2.1000e+11 Pa | 210 GPa (structural steel) |
| Poisson ratio $\nu$ | 0.30 | — |
| Initial yield stress $\sigma_y$ | 3.0000e+08 Pa | 300 MPa |
| Isotropic hardening $H$ | 1.0000e+10 Pa | 10 GPa ($\approx E/21$) |
| Shear modulus $G=E/2(1+\nu)$ | 8.076923e+10 Pa | derived |
| Bulk modulus $K=E/3(1-2\nu)$ | 1.750000e+11 Pa | derived |
| Yield strain $\varepsilon_y=\sigma_y/E$ | 1.428571e-03 | derived |
| Plastic tangent $E H/(E+H)$ | 9.545455e+09 Pa | derived |

Governing equations (infinitesimal strain, associative flow):

* additive split $\varepsilon = \varepsilon^e + \varepsilon^p$, isotropic
  elasticity $\sigma = K\,\mathrm{tr}\varepsilon^e\,\mathbf 1 + 2G\,\mathrm{dev}\varepsilon^e$;
* yield function $f(\sigma,\alpha) = q(\sigma) - (\sigma_y + H\alpha)$ with
  $q = \sqrt{\tfrac32 \mathbf s:\mathbf s}$ (von Mises);
* flow rule $\dot\varepsilon^p = \dot\gamma\,\tfrac32\,\mathbf s/q$,
  hardening $\dot\alpha = \dot\gamma$, Kuhn–Tucker
  $\dot\gamma \ge 0,\ f \le 0,\ \dot\gamma f = 0$;
* radial return: $\Delta\gamma = \langle f^{tr}\rangle/(3G+H)$ with
  $f^{tr} = q^{tr} - \sigma_y - H\alpha_n$, then
  $\mathbf s = \bigl(1 - 3G\Delta\gamma/q^{tr}\bigr)\mathbf s^{tr}$;
* consistent tangent $\mathbb C = K\,\mathbf 1\otimes\mathbf 1
  + 2G\,a\,\mathbb I_{\mathrm{dev}} + c\,\mathbf s^{tr}\otimes\mathbf s^{tr}$,
  $a = 1-3G\Delta\gamma/q^{tr}$, $c = 3G\,(\mathrm da/\mathrm dq^{tr})/q^{tr}$.

The return mapping and tangent kernels are the ones already present in
`tensorfem.plasticity` (single source of truth, unchanged); this mission adds the
**qualification surface**: fail-closed material validation, a posteriori
residual audits, and analytic anchors.

## 2. How it was done

1. **Read before writing**: audited `plasticity.py` (`update_j2`,
   `j2_consistent_tangent`), `solid_plasticity.py`, `global_plasticity.py` and
   the 11 existing plasticity test files. The kernels existed; the *verifiable
   surface* (input rejection, residual audit, analytic anchors, artifact) did
   not, which is why m605 reported “no elastoplastic capability”.
2. **New module** `j2_elastoplasticity.py` (material-point API): validated
   `J2Material`, `radial_return` returning stress/state/`6x6` tangent/`Δγ`,
   `return_map_residuals` (six independent a posteriori identities),
   `uniaxial_stress_path` (lateral strains solved by Newton **with the
   consistent tangent**, so $\sigma_{22}=\sigma_{33}=0$ exactly),
   `yield_probe` (bisection to the yield onset, compared with the analytic
   surface), `pure_shear_yield_point`, and `verify_minimum_surface` (the whole
   programmatic gate set).
3. **Conventions**: symmetric $3\times3$ tensors internally; Voigt order
   $[xx,yy,zz,xy,yz,zx]$ with **engineering** shear $\gamma=2\varepsilon$, so the
   tangent maps $\mathrm d\gamma$ to $\mathrm d\sigma$ (no hidden factor 2).
4. **Fail-closed validation**: $E\le 0$, $\nu\notin(-1,0.5)$, $\sigma_y\le 0$,
   $H<0$, non-finite input, non-symmetric or non-`3x3` strain all raise
   `ValueError`/`TypeError` before any arithmetic.
5. **Tool chain**: pure `torch` float64 (no randomness, no external solver),
   pytest 16 cases, matplotlib figures, JSON artifact with SHA-256 ledger.
   Command: `PYTHONPATH=src python3 -m pytest tests/test_j2_minimum_surface.py -q`.

## 3. Results

### 3.1 Uniaxial piecewise anchor (gate: relative error $< 10^{-10}$)

The lateral-strain Newton converged in a handful of iterations at every step;
the axial response is piecewise linear with the analytic slopes.

| Quantity | Numeric | Analytic | Relative error | < 3 % |
|---|---|---|---|---|
| Elastic branch slope $d\sigma/d\varepsilon$ | 2.100e+11 Pa | $E$ = 2.100e+11 Pa | 1.453e-16 | **YES** |
| Plastic branch slope $d\sigma/d\varepsilon$ | 9.545e+09 Pa | $EH/(E+H)$ = 9.545e+09 Pa | 5.795e-15 | **YES** |
| Max axial stress vs closed form (6 strain levels) | see JSON | piecewise closed form | 1.748e-16 | **YES** |

The plastic branch is **softer** than the elastic one by construction
($EH/(E+H) \ll E$), and the transition happens at
$\varepsilon_{11} = \varepsilon_y = 1.428571e-03$, where the
equivalent plastic strain first becomes positive.

### 3.2 Pure shear and general yield onset (gate: $< 1\,\%$)

| Quantity | Numeric | Analytic | Relative error | < 3 % |
|---|---|---|---|---|
| Shear yield strain $\gamma_y$ | 2.144e-03 | $\sigma_y/(\sqrt3 G)$ = 2.144e-03 | 2.022e-16 | **YES** |
| Shear yield stress $\tau_y$ | 1.732e+08 Pa | $\sigma_y/\sqrt3$ = 1.732e+08 Pa | 1.721e-16 | **YES** |
| $q(\sigma)$ at onset | — | $\sigma_y$ | 1.987e-16 | **YES** |

Four deviatoric directions were probed by bisection; each onset state satisfies
$|q/\sigma_y - 1| < 1\,\%$ (largest
1.987e-16):

| Direction | Onset scale (numeric) | Analytic scale | Scale rel. error | $q/\sigma_y-1$ |
|---|---|---|---|---|
| uniaxial | 1.857e-03 | 1.857e-03 | 2.335e-16 | 0 |
| pure shear | 2.144e-03 | 2.144e-03 | 2.022e-16 | 1.987e-16 |
| equibiaxial shear | 1.072e-03 | 1.072e-03 | 2.022e-16 | 1.987e-16 |
| combined | 1.362e-03 | 1.362e-03 | 0 | 0 |

### 3.3 A posteriori return-mapping residuals (gate: $< 10^{-12}$)

A four-step **non-proportional** path (tension → shear → shear reversal →
combined reloading) was integrated with transactional state updates. Each
returned state is re-audited *independently of the algorithm*: yield condition,
radial-return stress identity, consistency parameter, flow rule, hardening law
and complementarity.

| Residual (normalised) | Max over path |
|---|---|
| yield | 2.838e-16 |
| stress identity | 7.096e-17 |
| consistency | 0 |
| flow rule | 3.679e-16 |
| hardening | 3.401e-16 |
| complementarity | 2.838e-16 |

Maximum residual over the path: **3.679e-16** for
4 increments — at double-precision round-off, i.e. the return
mapping is an *exact* solve of the discrete equations, not just an accurate one.

### 3.4 Consistent tangent (gate: $< 10^{-6}$)

The plastic-branch `6x6` algorithmic tangent agrees with a central difference of
the return mapping to **4.192e-10** relative
error, and is softer than the elastic tangent as expected. This also audits the
pre-existing kernel: the closed-form tangent in `plasticity.py` is correct in the
engineering-shear Voigt convention.

### 3.5 Figures

![fig_uniaxial_stress_strain.png](../artifacts/j2_minimal/fig_uniaxial_stress_strain.png)

*Uniaxial-stress path: radial return (markers) against the piecewise closed form (line). Elastic slope E and plastic slope EH/(E+H) are reproduced to double precision (relative error 1.45e-16 and 5.79e-15).*

![fig_yield_surface.png](../artifacts/j2_minimal/fig_yield_surface.png)

*Deviatoric-plane yield locus. Open circles are the first plastically returning state found by sweeping 24 deviatoric directions; all lie on the analytic locus q=sigma_y (max |q/sigma_y-1| = 2.90e-05, gate < 1%).*

![fig_return_map_residuals.png](../artifacts/j2_minimal/fig_return_map_residuals.png)

*A posteriori residuals of the four-step non-proportional path (yield, radial-return stress identity, consistency parameter, flow rule, hardening law, complementarity), all below the 1e-12 gate (max 3.68e-16).*

![fig_consistent_tangent.png](../artifacts/j2_minimal/fig_consistent_tangent.png)

*Plastic-branch 6x6 algorithmic tangent (left) and a central difference of the return mapping (right); the relative difference is 4.19e-10 (gate < 1e-6).*

### 3.6 Gate summary

| Gate | Value | Criterion | Verdict |
|---|---|---|---|
| `uniaxial_elastic_slope_rel_error` | 1.453e-16 | < 1e-10 | PASS |
| `uniaxial_plastic_slope_rel_error` | 5.795e-15 | < 1e-10 | PASS |
| `uniaxial_stress_rel_error` | 1.748e-16 | < 1e-10 | PASS |
| `pure_shear_yield_surface_rel_error` | 1.987e-16 | < 1% | PASS |
| `pure_shear_gamma_rel_error` | 2.022e-16 | < 1% | PASS |
| `yield_surface_max_rel_error` | 1.987e-16 | < 1% | PASS |
| `return_map_max_residual` | 3.679e-16 | < 1e-12 | PASS |
| `consistent_tangent_rel_error` | 4.192e-10 | < 1e-6 | PASS |
| `pytest_tests` | 16.000000 | >= 6 (incl. 6 negative cases) | PASS |

Raw numbers: `artifacts/j2_minimal/verification.json`.

## 4. Error judgement (explicit)

Every key quantity is compared with a **closed-form** reference, not with a
previous run of the same code:

| Key quantity | Reference | Relative error | < 3 %? |
|---|---|---|---|
| Uniaxial elastic slope | $E$ | 1.453e-16 | **YES** |
| Uniaxial plastic slope | $EH/(E+H)$ | 5.795e-15 | **YES** |
| Uniaxial stress (6 levels) | piecewise closed form | 1.748e-16 | **YES** |
| Shear yield strain | $\sigma_y/(\sqrt3 G)$ | 2.022e-16 | **YES** |
| Shear yield stress | $\sigma_y/\sqrt3$ | 1.721e-16 | **YES** |
| Yield onset on $q=\sigma_y$ (4 directions) | analytic yield surface | 1.987e-16 | **YES** |
| Return-mapping consistency | $f=0$, flow/hardening identities | 3.679e-16 | **YES** |
| Consistent tangent | central difference of return mapping | 4.192e-10 | **YES** |

Residuals are absolute-zero targets (round-off limited), so “< 3 %” is
satisfied with 10–14 orders of magnitude of margin; no quantity required a
tolerance relaxation to pass.

## 5. Notes, pitfalls and limitations

* **Convention trap**: mixing true-shear and engineering-shear Voigt
  conventions silently halves/doubles the shear block of the tangent. The module
  fixes engineering shear and verifies the tangent against finite differences;
  the pure-shear check ($\tau=G\gamma$, not $\tfrac12 G\gamma$) caught exactly
  this mistake during development.
* **Lateral-strain solve**: driving a 3-D material point “uniaxially” by
  $\varepsilon_{11}$ alone is *not* a uniaxial stress state; the lateral
  strains must be solved. Using the consistent tangent inside that Newton solve
  both enforces $\sigma_{22}=\sigma_{33}=0$ to $10^{-8}\sigma_y$ and
  exercises the tangent on every step.
* **Perfect plasticity** ($H=0$) is supported; the uniaxial tangent then
  vanishes to round-off (relative to $E$), not to an absolute zero.
* **Scope**: small strain, isotropic linear elasticity, $J_2$ surface, **linear
  isotropic** hardening, monotonic and proportional-increment paths. **Not**
  claimed: cyclic plasticity / Bauschinger effect, kinematic or combined
  hardening, rate/temperature dependence, damage/softening ($H<0$ is rejected by
  design), finite strain, localisation or mesh objectivity.
* **Roadmap (declared, not delivered here)**: (i) kinematic/combined hardening
  with a back-stress tensor and its consistent tangent; (ii) cyclic
  load–unload–reverse verification against a stabilised hysteresis anchor;
  (iii) multi-dimensional non-proportional paths with corner (non-smooth)
  return mapping; (iv) element-level qualification of a hull-girder local model
  using this material point as the constitutive kernel.

## 6. Audit chain and reproducibility

| Artifact | SHA-256 |
|---|---|
| `src/tensorfem/j2_elastoplasticity.py` | `b1e661562ee858eafae26498cbb783f737ec834a94ac27bd3dd7221b259f23c3` |
| `src/tensorfem/plasticity.py` | `c0ed4be13a94c692dabf9a7e8ce3fb914a4f4e31a60fcb0da21ee606fb32da60` |
| `tests/test_j2_minimum_surface.py` | `1c6425a65613471f56dfbc2b64d0dd7636906112340c44a0f5f989a1dd5873e6` |
| `scripts/j2_minimal_verification.py` | `968d756b07b22c741d00503979212696da82057dd11973341a96bb7d6b779c01` |

* module `tensorfem.j2_elastoplasticity` (this mission) + unchanged kernel
  `tensorfem.plasticity`; JSON artifact records both digests;
* generated on branch `agent/m4951`; the artifact commit adds these files
  on top, so the audit identity is the SHA-256 ledger above (file content,
  not commit id);
* tests: `PYTHONPATH=src python3 -m pytest tests/test_j2_minimum_surface.py -q` → 16 passed (16 collected test ids), of which
  7 are fail-closed **rejection** cases: negative hardening ×3, non-positive
  yield stress ×3, and invalid constants / NaN / asymmetric-or-wrong-shape
  strain / wrong material type;
* figures are regenerated deterministically (no RNG) by
  `scripts/j2_minimal_verification.py`; this document is rendered from the JSON
  by `scripts/render_j2_doc.py`, so no number is transcribed by hand.

### Reproduce

```bash
cd <repo>
PYTHONPATH=src python3 scripts/j2_minimal_verification.py   # JSON + 4 PNG
PYTHONPATH=src python3 scripts/render_j2_doc.py             # this document
PYTHONPATH=src python3 -m pytest tests/test_j2_minimum_surface.py -q
```
