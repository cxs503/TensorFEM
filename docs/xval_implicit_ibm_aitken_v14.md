# XVAL v14 — the interface gain of the implicit-IBM beam driver is `1 + O(dt²)`: why Aitken relaxation is inert

Anchor: `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`
(joconnor22/LIFE implicit immersed boundary + Aitken Δ² relaxation; acceptance < 15 % against
an analytic / literature reference). · repo **TensorFEM** · branch `agent/m22049`
Previous: `docs/CASE-FEM-XVAL-IBM-SUBITER-GAIN.md` (v13, gain `mu ≈ 1` diagnosed as "frozen
fluid field"); `docs/xval_implicit_ibm_aitken_v12.md` (v12, multi-DOF law `WITHHELD`);
`v11` (scalar closed form `omega* = 1/(1+R)`, `q = |1-omega(1+R)|`).

## 1. What was open

Two rounds left the same hole:

* v12 §3.4: beam `ImplicitIBMFSI`, 12 free DOF — gain `mu ≈ 1`, residual `[2.0e-3, 0, 0, …]`;
  "**the multi-DOF law has no real-solver data point — `WITHHELD`**".
* v13 §5: the gain is a property of *the drivers*, not of the Δ² formula; "what would make the
  gain real … a certified added-mass-operator implementation is a separate work item".

Both rounds varied the **mass ratio** and the **relaxation factor**. Neither varied the one
parameter the interface force law is explicitly written in: the **time step**. And neither asked
the cheapest possible question — *is the beam driver evaluating the interface force at the same
state as the plate driver, whose gain is known to be real* (v6: `f_fsi/f_vac` within 7.5 % of
Lamb)? This round does both.

Three natural hypotheses, all written down before measurement:

| # | hypothesis | verdict |
|---|---|---|
| H1 | *Parity*: the beam uses a frozen predictor velocity (`beam.node_velocities()`), the plate uses the Newmark-consistent velocity of the **iterate** (`body.body_velocity(y_k)`); restoring parity restores the gain | **falsified** (§3.2) |
| H2 | v13's closed-form frozen-fluid bound `gamma dt M_f/(M_s + k beta dt^2)` predicts the gain | **rejected** — 42× to 1.5e5× too high (§3.3) |
| H3 | the gain collapses with `dt`, so it cannot be exercised at any `dt` that resolves the structure | **confirmed**, exponent 1.3–1.5, certified by the full Jacobian (§3.1, §3.4) |

## 2. Method

### 2.1 Case

The v6/certificate submerged cantilever, unchanged (D2Q9 box 96×72, `dx = 1`, BGK `omega = 1.8`,
16→10 corotational elements, beam scaled to `0.45 * 96`, released from a `0.02 * scale` static
deflection, `rho_f = 1`). Sections `rho_s` = 0.2 (light) and 1.0. Time step
`dt ∈ {1, 1/2, 1/5, 1/10, 1/20, 1/50, 1e-2, 5e-3, 2e-3}` — from the driver's own plate default
(`dt = 1`) down to the value the beam dynamics actually need (~2e-3, the v6 record).

### 2.2 Gain

`mu` is recovered from the residual traces of the shipped loop run at `relax_mode="fixed"`,
`omega ∈ {1, 1/2}` through `tensorfem.ibm_subiter_gain.gain_from_trace` (the fit that removes the
sign ambiguity of a single factor). `fit_error` is reported alongside every number: when the gain
falls below the fit resolution it is quoted as an **upper bound**, never as `mu = 1`.

### 2.3 The two code changes (both opt-in, default = archived behaviour)

1. `CorotationalBeam2D.velocity_for(u, dt)` — the Newmark correction relation at an arbitrary
   trial displacement. Tested to be *identical* (0.0 and 8.9e-16) to both the recursive form
   inside `newmark_solve` and the closed form `SpringMountedBody.body_velocity` uses.
2. `ImplicitIBMFSI(velocity_mode=…, coupling_scale=…)` — `"predictor"` (default, bit-identical
   to v1–v13), `"iterate"` (force evaluated at the iterate's velocity, i.e. plate parity);
   `coupling_scale` multiplies the marker load (default `1.0`, bit-identical).

Neither changes the shipped Aitken formula, the force law, Newmark or the ε system.

### 2.4 The Jacobian certificate

The shipped map `u -> G(u)` is evaluated through the public API only (the six lines of `step`
that build it), so the Jacobian is the Jacobian of **the shipped code**, not of a model:

```
J = dG/du   over the free DOFs (central differences, h = 1e-6 * |u|),   M = I - J
```

`M` is then fed to the v12 lagged-minres model (`minres_relaxation(M, r0, first_omega=1)`) and
the predicted sub-iteration count compared with the production Aitken loop at the same relative
tolerance.

## 3. Results

Driver `benchmarks/xval_aitken_interface_gain_scaling.py` (single core, 35 s);
artifacts `benchmarks/xval_aitken_interface_gain_scaling_artifacts/` (JSON + 2 PNG);
regression `tests/test_aitken_velocity_mode_gain.py` (8 tests, 9 s).

### 3.1 The gain collapses with the time step (H3)

`mu - 1`, fit from `omega ∈ {1, 1/2}` (`fit_error` = largest measured/model ratio gap):

| `dt` | `rho_s` = 0.2 | `fit_error` | `rho_s` = 1.0 | `fit_error` |
|---|---|---|---|---|
| 1 | **2.00e-2** | 5.6e-3 | **5.0e-3** | 1.7e-3 |
| 1/2 | 7.0e-3 | 1.1e-3 | < 4.9e-4 | 4.9e-4 |
| 1/5 | < 4.3e-4 | 4.3e-4 | < 1.4e-4 | 1.4e-4 |
| 1e-2 | < 8.6e-7 | 8.6e-7 | < 1.0e-7 | 1.0e-7 |
| 2e-3 | **< 1.1e-8** | 1.1e-8 | **< 5.3e-9** | 5.3e-9 |

log–log slope of `mu - 1` vs `dt` (`rho_s` = 0.2, the only curve with >1 resolvable point):
**1.51** (`predictor`), **1.28** (`iterate`). So the gain dies at least linearly and in fact
faster than linearly; at the step the beam dynamics need it is below `1e-8`.

In other words the *shipped* loop is a **constant map**: `r_1/r_0 = 1 - omega` to 1e-2 for every
factor (test `test_relaxation_factor_only_rescales_a_constant_map`), so the Aitken recursion —
which is exact to 4.8e-16 (v11/v12) — has nothing to solve. This is v13's `mu ≈ 1` *explained*, and
it also explains why it survived thirteen rounds: every earlier study held `dt` fixed at a value
where the gain is `1e-8`.

### 3.2 H1 is falsified: velocity parity is not the cause

`velocity_mode="iterate"` changes the traces by `< 4e-9` relative and the gain by `< 1e-3`
(|difference| below the fit granularity) at every `dt ≤ 1e-2`; at `dt = 1` the gain moves from
2.00e-2 to 1.9e-2 (`rho_s` = 0.2) and stays 5.0e-3 (`rho_s` = 1.0). The two drivers therefore agree
*when the split is meaningful* — the beam's `mu ≈ 1` was never a parity defect. Recorded as a
falsified hypothesis, with the test that pins it
(`test_velocity_modes_agree_on_the_gain`).

### 3.3 H2 is rejected: v13's frozen-fluid bound is not tight

| `rho_s` | `dt` | bound `gamma dt M_f/(M_s+k beta dt^2)` | measured `mu-1` | ratio |
|---|---|---|---|---|
| 0.2 | 1 | 8.35e-1 | 2.00e-2 | **42×** |
| 0.2 | 1e-1 | 8.35e-2 | < 1.9e-4 | **≥ 4.4e2×** |
| 0.2 | 2e-3 | 1.67e-3 | < 1.1e-8 | **≥ 1.5e5×** |

(`M_f = 44.0`, `M_s = 26.34` at `rho_s` = 0.2, i.e. `M_f/M_s = 1.67`.) The bound is 42× high where
it can be checked at all and diverges as `dt → 0`, so it over-estimates ∂F/∂u by a factor growing
with `dt^-0.3…-0.5`. It remains an upper bound (it never under-predicts), but it cannot be used as
a quantitative reference.

A scalar prefactor cannot close the gap either: `coupling_scale` acts **exactly linearly** on the
interface Jacobian (measured `‖J‖₂(50)/‖J‖₂(1) = 50` to <10 %, test
`test_coupling_scale_acts_linearly_on_the_gain`), so a multiplier moves the whole curve up without
changing its `dt` exponent.

### 3.4 Jacobian certificate (v12's `WITHHELD` row, at matrix level)

| `dt` | `‖J‖₂` | `rho(I-J)` | `kappa = cond(I-J)` | count: v12 model / shipped | ratio `r_1/r_0` model / measured |
|---|---|---|---|---|---|
| 1 | 5.41e2 | **1.2958** | 1.30 | 3 / **2** | 0.295 / 0.292 |
| 2e-3 | 1.81e-5 | **1.000000** | 1.00 | 1 / **2** | 1e-5 / 7e-8 |

At the resolved step the interface operator is the **identity to 1e-6**: the line search returns
`omega* = 1` for every step, the exact line-search step and no relaxation are the same algorithm,
and the residual cannot decay geometrically (it dies by the map being constant, in one sweep).
So v12's missing "real-solver multi-DOF data point" exists — and is **degenerate** (`kappa = 1`),
which is exactly the honest content of `WITHHELD`: not a missing measurement, but a measurement of
a quantity that is `1` by construction.

At `dt = 1`, where the gain is genuine (`rho(I-J) = 1.296`, `kappa = 1.30`), the v12 lagged-minres
model driven by the measured operator predicts **3** sub-iterations and the shipped loop takes
**2** — a one-sweep conservative gap, with the *first* residual ratio reproduced to 1 % (0.295 vs
0.292). That is the first real-solver, multi-DOF, non-degenerate data point for the v12 model, and
it is inside the acceptance band on the quantities it can check.

## 4. Error table against the acceptance criterion (owner standard)

| quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| `velocity_for` vs Newmark correction relation | recursive form / closed form | identical | 0 / 8.9e-16 | **yes** |
| gain, `dt ≤ 1e-2`, `predictor` vs `iterate` | equality (H1) | Δ < 1e-3 | ≤ fit granularity | **yes** |
| gain at resolved `dt` vs 0 (constant map) | 0 | < 1.1e-8 | 1.1e-8 absolute | **yes** |
| Jacobian at resolved `dt` vs `J = 0` | 0 | 1.81e-5 | `rho(I-J)-1 = 6e-7` | **yes** |
| sub-iterations at `dt = 1` | v12 model on measured `J` | 2 | 1 sweep (3 vs 2) | **no** |
| gain vs v13 closed-form bound | bound | 42×–1.5e5× below | **bound not tight** | **no** |
| gain vs Lamb added mass `1 + m_a/m_s` | `m_a = rho pi c^2`: `mu-1` = 1.2e-3 … 0.1 | ≤ 2.0e-2 (only at `dt = 1`) | not comparable (the driver never reaches the regime) | **no — `WITHHELD`** |

The anchor's necessity claim therefore **stays `WITHHELD`**, now with a scaling law instead of a
single configuration: `mu - 1 ∝ dt^1.3…1.5`, so the frozen-fluid scheme is gain-free at every step
that resolves the structure. It is *not* a defect of the Δ² rule.

## 5. Caveats / what would close the anchor

1. **What is missing is a term, not a factor.** ∂F/∂u in the shipped law comes from the *velocity*
   term `2 (dt/dx) rho_i u_body`, i.e. `∂F/∂u ∝ 1/dt` times the structural compliance `∝ dt^2`;
   the added-mass operator needs `∂F/∂u ∝ m_a/(beta dt^2)` (an *acceleration* term). The measured
   exponent (1.3–1.5) is not the pure `1` this argument gives, so an additional `dt` dependence
   (marker-position dependence of the interpolated momentum) is present and is **not resolved
   here**. A certified fix therefore needs a trial-collision / added-mass-operator loop (Breugem
   2012; Uhlmann 2005) — a real work item, flagged by v13 and still open.
2. **`dt = 1` is not a physical simulation.** It is used only because the *sub-iteration fixed
   point* is a property of the map and not of the time integration; the trajectory at `dt = 1` is
   meaningless and no physical number is quoted from it.
3. **Jacobian at the predictor, not at the fixed point.** `J` is measured at the first iterate of a
   step (that is the state the sweep starts from). The scalar trace fit is taken at the fixed
   point; the 15× quantitative disagreement between `rho(I-J)-1` and the trace-fit gain at `dt = 1`
   is that difference, and is reported rather than averaged away.
4. **One driver family.** Only `ImplicitIBMFSI` (beam) is measured. The plate `IBMImplicitFSI`
   reaches a genuine gain (v6: 7.5 % vs Lamb) because its default is `dt = 1`; it is not re-audited
   here.
5. `residual_traces` remains diagnostic-only (default empty); no archived record changes.

## 6. Reproduction

```bash
PYTHONPATH=src python benchmarks/xval_aitken_interface_gain_scaling.py     # 35 s, 1 core
PYTHONPATH=src python -m pytest tests/test_aitken_velocity_mode_gain.py -q  # 8 passed, 9 s
PYTHONPATH=src python -m pytest tests/test_ibm_aitken_implicit.py \
    tests/test_ibm_subiter_gain.py tests/test_aitken_spectral_law.py -q     # 55 passed
```

## 7. Files

| file | status |
|---|---|
| `src/tensorfem/ibm_aitken_implicit.py` | + `CorotationalBeam2D.velocity_for`; `ImplicitIBMFSI(velocity_mode=, coupling_scale=)` (both opt-in, defaults bit-identical); `velocity_mode` in the step record |
| `benchmarks/xval_aitken_interface_gain_scaling.py` | **new**: gain-vs-`dt` stage, bound stage, Jacobian stage, 2 figures |
| `benchmarks/xval_aitken_interface_gain_scaling_artifacts/` | `xval_aitken_interface_gain_scaling.json`, `xval_aitken_v14_gain.png`, `xval_aitken_v14_jacobian.png` |
| `tests/test_aitken_velocity_mode_gain.py` | **new**: 8 regression tests |
| `docs/xval_implicit_ibm_aitken_v14.md` | this report |

## 8. Figures

* `xval_aitken_v14_gain.png` — left: `mu-1` vs `dt` (log–log, both sections, both velocity modes)
  against `∝dt` and `∝dt²` guide lines: the measured curves fall as steeply as `dt^1.3`;
  right: the shipped residual trace at `dt = 2e-3`, dead after one sweep.
* `xval_aitken_v14_jacobian.png` — left: `rho(I-J)` of the shipped map at `dt = 1` (1.296) and
  `dt = 2e-3` (1.000000); right: sub-iteration counts, v12 model vs shipped loop.

Field contour plots are not applicable: the measured object is the coupling iteration, not a flow
field.