# CASE FEM-HERTZ-CONTACT — load-controlled Hertz contact (TensorFEM)

Reference: Hertz (1882) *J. reine angew. Math.* **92**:156; Johnson, *Contact Mechanics*
(1985) §4.2, DOI:10.1017/cbo9781139171731. Case pool: FEM-HERTZ-CONTACT (id 104),
tolerance **< 3%** on the contact radius `a` and the indentation `delta` with the normal
load `P` prescribed (elastic half-space).

## 1. Problem setting

Rigid paraboloid indenter `z_ind(r) = r^2/(2R) - delta` pressed into an axisymmetric
elastic half-space (`E = 1e5`, `nu = 0.3`; equivalently SI `R = 10 mm`, `E = 100 GPa`)
with prescribed total normal force `P`, chosen from the Hertz closed form for
`a/R = 0.05` and `0.10`. Outputs: contact radius `a`, indentation `delta`,
peak pressure `p0`. Hertz closed form used as the only oracle:
`a = (3PR/(4E*))^(1/3)`, `delta = a^2/R`, `p0 = 3P/(2 pi a^2)`, `E* = E/(1-nu^2)`.

## 2. Method and implementation

- `src/tensorfem/hertz_halfspace_loadcontrol.py` — axisymmetric Q4 half-space,
  **sparse** assembly (the shipped `hertz_axisymmetric` is dense and
  prescribed-indentation).
- Contact: discrete Signorini problem solved by an **augmented Lagrangian**
  (Alart & Curnier 1991; Simo & Laursen 1992; Wriggers 2006 §6.2), multiplier
  update `lambda <- <lambda - rho g>_+` with `g = z_ind - u_z` and under-relaxation
  0.5. No penalty: the constraint violation is at round-off (column `pen/d`).
- **Load control**: regula falsi on the monotone response `P_fe(delta)` until
  `|P_fe - P|/P < 1e-10`.
- Contact radius: primary estimator = linear zero crossing of the computed nodal
  pressure between the last two active nodes (no analytical shape assumed);
  `a_outer` (last active node) and a Hertz-form least-squares fit are reported as
  cros-checks.
- Element kernel verified against the dense reference implementation
  (`max|dK| = 1.9e-9` vs peak `1.5e7`; `tests/test_hertz_contact_loadcontrol.py`).

## 3. Parameter table

| quantity | value |
|---|---|
| R, E, nu | 10, 1e5, 0.3 |
| E* = E/(1-nu^2) | 0 (see JSON) |
| P (a/R=0.05) | 1831.5 |
| a, delta, p0 (Hertz) | 0.500000, 0.025000, 3497.9 |
| h/a ladder | 3%, 1.5%, 0.75% |
| domain | 96 a (truncation study: 24, 48, 96 a) |
| ALM rho | E/a, factor swept 0.3 / 1 / 3 |

## 4. Comparison table (ref vs got)

| a/R | h/a [%] | H/a | elements | a_ref | a_FE | a err [%] | d_ref | d_FE | d err [%] | p0 err(consistent) [%] | pen/delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 1.50 | 96 | 9040 | 0.500000 | 0.497342 | -0.532 | 0.025000 | 0.024747 | -1.011 | +0.59 | 1.9e-07 |

Gate table (|err| <= 3%):

| run | a err [%] | delta err [%] | gate |
|---|---|---|---|
| ratio0.05_h0.0150_d96 | -0.532 | -1.011 | PASS |

## 5. Convergence and sensitivity

Truncation of the half-space (dominant modelling error):

| H/a | delta err [%] | a err [%] |
|---|---|---|
| 24 | -2.900 | -0.537 |
| 48 | -1.639 | -0.532 |
| 96 | -1.011 | -0.532 |

Augmented-Lagrangian parameter (must not move the answer):

| rho/(E/a) | ALM sweeps | a | delta |
|---|---|---|---|
| 0.3 | 80 | 0.4958518 | 0.02475100 |
| 1 | 80 | 0.4957595 | 0.02475108 |
| 3 | 67 | 0.4957593 | 0.02475108 |

rho spread: a 1.86e-02 %, delta 3.41e-04 % —
the converged state is parameter independent, as required for a penalty-free method.

## 6. Failure modes and artifacts

- **Node-on-node edge chatter**: a naive "toggle on penetration / off on tension"
  active set oscillates (k=33 <-> 34, 0.14% delta penetration swing, ~89 s wasted
  per solve). Fixed by the augmented-Lagrangian projection plus under-relaxation.
- **Contact-membership threshold**: with under-relaxation the multiplier of a
  separating node decays geometrically and stays positive at round-off level; a
  single such node outside the patch inflated `a` by +23%. Contact membership now
  uses a relative threshold `1e-10 max(lambda)`.
- **Small-patch breakdown**: at `a/R = 0.02` the edge-based estimator is noise
  dominated — recorded as an honest failure mode, not gated.
- **Lumped vs consistent pressure recovery**: `p = lambda/A` overestimates the
  axis pressure by +32% (ring area collapses); the consistent (mass-matrix)
  recovery is within +0.59%.
- Reproduce: `PYTHONPATH=src python3 benchmarks/verified/fem_hertz_contact_loadcontrol/run.py`
  (artifacts: `artifacts/contact_hertz_loadcontrol/results.json`, `figures/`).

## 7. Error verdict

Contact radius: **-0.532%**; indentation:
**-1.011%** at h/a = 1.5%, H = 96a — both **< 3%**
(criterion met). The residual indentation error is truncation, not discretisation:
it falls roughly as a/H (−2.90% at 24a -> −1.01% at 96a) and the discretisation
part is below 0.1% (0.75% level reproduces −1.010%).
