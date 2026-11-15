# XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION

Anchor: `XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION`
External reference: Young-Laplace law ΔP = σ/R (classical capillarity; Yuan & Schaefer 2006, Phys. Fluids 18, 042101)
Acceptance: Laplace-law fit R² ≥ 0.95 and per-point error < 15%
**Verdict: honest-FAIL (partial)** — R² = 0.9881 (PASS), but per-point errors up to 53.9% (FAIL)

Every number in this document is injected from
`docs/assets/lbm_shan_chen_laplace/results.json` by
`examples/lbm_shan_chen_laplace_pressure_xval.py` (no hand-typed value).

## 1. Problem setting and reference

The Young-Laplace law is the fundamental relation for curved fluid interfaces:

2429017\Delta P = rac{\sigma}{R}2429017

where ΔP is the pressure jump across the interface, σ is the surface tension,
and R is the radius of curvature.

For the Shan-Chen pseudo-potential LBM, the thermodynamic equation of state
(Yuan & Schaefer 2006, Phys. Fluids 18, 042101) is:

2429017P = c_s^2 ho_{	ext{tot}} + rac{c_s^2}{2} \sum_{s,s'} G_{ss'} \psi_s \psi_{s'}2429017

For D2Q9, ^2 = 1/3$. With {aa} = G_{bb} = 0$ and {ab} = G_{ba} = G$:

2429017P = c_s^2 (ho_a + ho_b) + c_s^2 G \psi_a \psi_b2429017

A circular droplet of the dense phase (component a, $ho_{	ext{in}}$) in the
dilute phase (component b, $ho_{	ext{out}}$) should exhibit a pressure jump
consistent with the Young-Laplace law.

## 2. Method

The solver `src/tensorfem/lbm_shan_chen_mcmp.py` implements the two-component
Shan-Chen MCMP model with:
- D2Q9 lattice, SRT collision, Guo (2002) forcing
- Barycentric velocity (validated against lbmpy)
- Exponential pseudo-potential $\psi = ho_0(1 - e^{-ho/ho_0})$

The `pressure_field()` method computes the thermodynamic pressure from the EOS.
For each droplet radius R, a circular drop is initialized with a tanh profile
and equilibrated. The radial pressure profile is averaged, and ΔP is measured
as the difference between the plateau values inside and outside the interface.

## 3. Parameters

| parameter | value |
|---|---|
| domain | 64 × 64, periodic |
| relaxation | $	au_a = 	au_b = 1.0$ ($\omega = 1$) |
| interaction | {ab} = g_{ba} = 5.0$ |
| pseudo potential | exponential, $ho_0 = 1.0$ |
| droplet radii | 8, 12, 16, 20, 24 |
| $ho_{	ext{in}}$ | 1.4 |
| $ho_{	ext{out}}$ | 0.2 |
| equilibration | 3000 steps per radius |
| interface width | 3.0 |

## 4. Results

| R | R_interface | P_inside | P_outside | ΔP measured | ΔP predicted (σ/R) | error | spurious current | pass |
|---|---|---|---|---|---|---|---|---|
| 8 | 9.50 | 0.736541 | 0.730147 | 0.006394 | 0.004579 | 28.40% | 2.24e-03 | ✗ |
| 12 | 12.50 | 0.721519 | 0.716667 | 0.004852 | 0.003052 | 37.09% | 2.46e-03 | ✗ |
| 16 | 16.50 | 0.706423 | 0.702554 | 0.003869 | 0.002289 | 40.82% | 2.45e-03 | ✗ |
| 20 | 19.50 | 0.694352 | 0.690568 | 0.003784 | 0.001831 | 51.60% | 2.53e-03 | ✗ |
| 24 | 23.50 | 0.690409 | 0.687096 | 0.003313 | 0.001526 | 53.94% | 2.49e-03 | ✗ |


Surface tension estimate: **σ = 0.0366** (from linear fit of ΔP vs 1/R)
Fit quality: **R² = 0.9881**

## 5. Figures

![fig1_laplace_law.png](docs/assets/lbm_shan_chen_laplace/fig1_laplace_law.png)
![fig2_radial_profiles.png](docs/assets/lbm_shan_chen_laplace/fig2_radial_profiles.png)
![fig3_spurious_currents.png](docs/assets/lbm_shan_chen_laplace/fig3_spurious_currents.png)
![fig4_laplace_errors.png](docs/assets/lbm_shan_chen_laplace/fig4_laplace_errors.png)

## 6. Error assessment

- Laplace law fit R² = 0.9881 ≥ 0.95 → **PASS**
- Maximum per-point Laplace error = 53.94% ≥ 15% → **FAIL**
- Maximum spurious current = 2.53e-03 (should be ≪ 1 for equilibrium)
- Maximum mass drift = 1.75e-13 (should be ≪ 1)

**Overall verdict: honest-FAIL (partial)** — the Laplace law scaling (ΔP ∝ 1/R) is
verified with R² = 0.9881, but the absolute pressure jump values deviate
from the fitted prediction by up to 53.9%.

## 7. Failure mode analysis

The per-point errors are high because the droplets have not fully equilibrated
within 3000 steps. Evidence:
- The coexisting densities (ρ_in_eq ≈ 1.6-1.7, ρ_out_eq ≈ 0.12-0.19) differ
  significantly from the initial values (1.4, 0.2), indicating ongoing phase separation.
- The spurious currents have plateaued at ~2.5×10⁻³, suggesting the system is
  close to but not at equilibrium.
- The pressure profiles show a gradual gradient in the "outside" region rather
  than a flat plateau, indicating incomplete relaxation.

**Remediation**: Running 10,000+ steps per radius (or using a better initial
condition closer to the Maxwell coexistence densities) would reduce the errors.
The cluster job cj3507 was submitted for this purpose but remains queued.

## 8. Reproduction

```bash
cd /srv/tensor/agent/work/m21532
PYTHONPATH=src python examples/lbm_shan_chen_laplace_pressure_xval.py
```

Artifacts: `docs/assets/lbm_shan_chen_laplace/results.json`, `fig*.png`
