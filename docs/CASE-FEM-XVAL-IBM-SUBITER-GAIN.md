# CASE-FEM-XVAL-IBM-SUBITER-GAIN — does the implicit-IBM sub-iteration actually iterate?

Anchor: `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` (joconnor22/LIFE recipe) · repo **TensorFEM**
Acceptance (ledger): < 15 % against the reference recipe · Round: **v13** (v1–v12 archived)

## 1. Case / what was open

The anchor ports LIFE's implicit immersed-boundary scheme: inside every time step the
fluid–structure interface is driven to a fixed point by sub-iterations that are under-relaxed
with Aitken's Δ² rule. Twelve earlier rounds validated the IBM operator chain, the beam mass
matrix defect, the residual-measure (certificate) defect and the spectral law of the Aitken
map. The last round (v12) closed with a **negative** row it could not explain:

> beam `ImplicitIBMFSI`, 12 free DOF — interface gain `mu ≈ 1` (`R < 1e-4`), residual sequence
> `[2.04e-3, 0, 0, …]`, i.e. converged after one relaxation with `omega` pinned at the clip.
> **The multi-DOF law has no real-solver data point — `WITHHELD`.**

`WITHHELD` there means "the production driver cannot demonstrate the effect the anchor claims".
This round asks the two questions that settle it:

1. **Is the gain `mu ≈ 1` a property of one configuration, or of the driver?**
2. **If it is the driver — where does the iterate's influence go?**

Configuration used by every measurement below (the v6 submerged cantilever, unchanged):
D2Q9 box 96×72 (`dx = 1`), BGK `omega = 1.8`, `dt = 2e-3`, 16 corotational-beam elements
(33 nodes, 30 free DOF of which 10 are rotations), beam scaled to `0.45 * 96` and released from
a `0.05 * scale` static deflection; `rho_f` and `rho_s` sweep the added-mass ratio.

| parameter | value |
|---|---|
| lattice / spacing | 96×72, `dx = 1` (control: 64×48 and 128×96) |
| beam elements | 16 (control: 12 and 20) |
| `dt` | 2e-3 |
| fluid density `rho_f` | 0.05, 0.2, 1.0, 2.0 |
| structure density `rho_s` | 0.2, 1.0, 5.0 |
| relaxation factors probed | `omega` = 1, 1/2, 1/4 (no Aitken inside the probe) |
| sub-iterations per probe | 3, `sub_tol = 0` (so the full trace is recorded) |

## 2. Method

**Gain from the residual trace.** For a map `u -> G(u)` iterated as
`u_{k+1} = u_k + omega (G(u_k) - u_k)`, the interface residual obeys
`r_k = (1 - omega mu) r_{k-1}` near the fixed point, with `mu` the interface gain
(`mu = 1 + m_a/m_s` for an added-mass-limited interface, Causin–Gerbeau–Nobile 2005). The driver
is run with `relax_mode="fixed"` at three factors and the two leading residuals are read off
`ImplicitIBMFSI.residual_traces` (a new, diagnostic-only record — the shipped Aitken path is
byte-identical, see the regression tests). `mu` is the value that fits all three measured ratios
`|1 - omega mu|` (`tensorfem.ibm_subiter_gain.gain_from_trace`); a single factor cannot
distinguish `mu = 1 + rho` from `1 - rho`.

**What the gain should be.** Two independent references per configuration:

* the added-mass prediction `mu = 1 + m_a/m_s` with Lamb's flat-plate value
  `m_a = rho_f pi c^2` (per unit length, `c = thickness/2` of the scaled section) —
  `tensorfem.ibm_subiter_gain.added_mass_ratio`;
* the *frozen-fluid discrete bound*: the interface force enters the momentum equation only
  locally, so a single sweep can feel at most the marker-local fluid mass
  `M_f = sum_i rho_i eps_i ds_i`, giving `|dG/du| = gamma dt M_f / (M_s + k beta dt^2)`
  (`discrete_gain_bound`).

**Answer-independence.** The same coupled case is run with Aitken, with fixed `omega = 1` and
with fixed `omega = 0.05`; equal answers ⇒ the relaxation factor has no leverage.

## 3. Results

### 3.1 The gain is exactly 1 in every configuration

`mu` fitted from the three traces, 14 configurations (12 in the `rho_f × rho_s` sweep, 2
resolution controls):

| `rho_f` | `rho_s` | `mu` measured | `mu - 1` | residual ratios (`omega` = 1, 1/2, 1/4) | `mu - 1` predicted by Lamb | frozen-fluid bound | inert? |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.2 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.096 | 4.5e-6 | yes |
| 0.05 | 1.0 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.019 | 9.0e-7 | yes |
| 0.05 | 5.0 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.004 | 1.8e-7 | yes |
| 0.2 | 0.2 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.386 | 1.8e-5 | yes |
| 0.2 | 1.0 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.077 | 3.6e-6 | yes |
| 0.2 | 5.0 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.015 | 7.2e-7 | yes |
| 1.0 | 0.2 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 1.930 | 8.9e-5 | yes |
| 1.0 | 1.0 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.386 | 1.8e-5 | yes |
| 1.0 | 5.0 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.077 | 3.6e-6 | yes |
| 2.0 | 0.2 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 3.860 | 1.8e-4 | yes |
| 2.0 | 1.0 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.772 | 3.6e-5 | yes |
| 2.0 | 5.0 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.154 | 7.2e-6 | yes |
| 1.0 (64×48, 12 el.) | 1.0 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.387 | 3.1e-5 | yes |
| 1.0 (128×96, 20 el.) | 1.0 | 1.000000 | 0 | 0.000000, 0.500000, 0.750000 | 0.385 | 1.2e-5 | yes |

Fit error ≤ 7e-8 everywhere (`summary.json`). The signature is exact, not approximate:

> when the relaxation factor is swept, the second residual is **`(1 - omega) r_0` to round-off**
> (measured `1.7e-9`, `0.500000`, `0.750000` for `omega = 1, 1/2, 1/4`).

That is the fingerprint of a **constant return map**: `G(u)` does not depend on `u` at all, so
one sweep lands on the fixed point and *any* relaxation factor is either a no-op (`omega = 1`) or
pure deceleration. The interface is not being iterated — it is solved once and re-solved.

### 3.2 Where the iterate's influence goes

The interface load is evaluated from the **frozen pre-collision fluid field**
(`fluid.macroscopic(include_force=False)` once per time step) and the body velocity through
`beam.node_velocities()`. Within a sweep the only quantities that could respond to the trial
iterate are (a) the marker stencil geometry (the trial displacement re-stencils the delta
functions — a `O(dx)` effect, measured `≤ 1.7e-9` in the residual ratio) and (b) the structural
inertia (already inside the Newmark solve). The added-mass feedback that makes an interface hard
to relax — the fluid's own velocity response to the interface force — is **not part of the
sweep**: it is an explicit, one-step-lagged effect in this driver.

The scale mismatch is visible in the same table: the marker-local fluid mass the sweep can feel
is `M_f = 3e-2 … 3.6e-2` (all configurations: `M_f ≈ 0.018·rho_f·S_markers`), two to three
orders of magnitude below the potential-flow added mass `m_a = rho_f pi c^2` (the frozen-fluid
bound column vs the Lamb column). The sub-iteration can therefore never see an added-mass gain,
whatever `rho_f/rho_s` is.

### 3.3 Consequence for the answer (relaxation is not neutral)

Same coupled case (48×36 control lattice, 8 elements, 24 steps), three relaxation schemes:

| scheme | sub-iterations per step | MAXIT hits | tip record vs Aitken | verdict |
|---|---|---|---|---|
| Aitken Δ² | 2 (every step) | 0 | — | reference |
| fixed `omega = 1` | 2 (every step) | 0 | **0.000e+00** (bit identical) | identical |
| fixed `omega = 0.05` | 30 (clipped) | 24/24 | 5.47 % | **wrong answer** |

The Aitken factor produced by the shipped loop is the clip `omega = 1.0` on **every** step
(recorded `relax` field: 24/24 steps), which is consistent with `mu = 1`: the Δ² estimate is
never asked for a smaller factor because the residual is already at round-off by the second
sweep. Conversely an unnecessarily small factor is *not* harmless: with the same 30-sweep
ceiling the under-relaxed run never converges (`MAXIT` on every step), and the committed —
under-relaxed — state biases the tip record by 5.5 %. Relaxation therefore *costs* accuracy
unless the gain justifies it, and the shipped `MAXIT` policy turns a mis-set factor into a
silent 5 % error.

## 4. Figures

`benchmarks/xval_ibm_subiter_gain_artifacts/xval_ibm_subiter_gain.png`

* **left**: measured `mu - 1` vs the Lamb added-mass prediction `m_a/m_s` (log–log), with the
  frozen-fluid bound `|dG/du|`; the measured points sit on the `1e-6` inert line while the
  prediction spans `0.004 … 3.86` — the gap is the finding;
* **right**: residual traces of the production driver at `omega = 1, 1/2, 1/4` — the decay is
  `(1 - omega)` per sweep exactly, i.e. no iterate dependence.

(Field contour plots are not applicable: the measured object is the coupling iteration, not a
flow field. The fluid states entering the coupling are the archived v6/v8 fields.)

## 5. Caveats / limitations

* **This is a finding about the shipped drivers, not about the Aitken formula.** The Δ² recursion
  itself is correct and was verified to 4.8e-16 against an independent implementation (v11/v12);
  what this round shows is that the drivers in this repository never put it in a situation where
  it matters. The anchor's *necessity* claim is therefore unsupported **by this code path**
  (`WITHHELD`, now with a diagnosed cause), not false.
* **Single driver family.** Two drivers were probed (beam `ImplicitIBMFSI` thoroughly; the plate
  `IBMImplicitFSI` shares the same "frozen fluid field" structure and its v12 row showed the same
  `mu ≈ 1`). Other FSI paths in the repository (`fsi_lbm_strong_coupling`, `fsi_wet_modal`) hold
  their own relaxers and were not re-audited here.
* **What would make the gain real**: the interface force must be evaluated against the fluid
  state *including its response to that force* — i.e. a trial collision inside the sweep
  (Breugem/Uhlmann strong coupling) or an explicit added-mass operator. A prototype of that loop
  was built and measured in this round: the force recursion `F <- 2(dt/dx) rho u_b - m(F)` with a
  lagged trial field oscillates (`r_{k+1}/r_k ≈ -1`) unless the structural response is in the
  loop, and with it the residual ratio becomes mass-independent (0.89 … 1.02 for
  `m_a/m_s = 0.1 … 0.8`), i.e. it tracks the *fluid* recursion rather than `m_a/m_s`. Those
  numbers are recorded here as a negative result for the cheap variants; a certified
  added-mass-operator implementation is a separate work item, not part of this round.
* **Long records are queued.** The full-length relaxation/physics stages of
  `benchmarks/xval_ibm_subiter_gain.py` (160-step records on the 96×72 lattice, wet/dry
  frequency shift) were submitted to the compute gate as job `cj3827`; the numbers in §3.3 come
  from the 24-step short record of the same script (period estimates from a 24-step record are
  degenerate and are *not* quoted). The gain table §3.1 does not depend on record length.
* The probe records `residual_traces` on the production class; the field is diagnostic-only
  (default empty, nothing in the Aitken path reads it) and the archived v6 record reproduces
  bit-identically (regression test `test_default_relaxation_is_untouched_by_the_trace`).

## 6. Error judgement (owner standard)

| key quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| residual ratio `r_1/r_0` at `omega` = 1/2, 1/4 | constant-map law `1 - omega` | 0.500000, 0.750000 | ≤ 4e-7 relative | **yes** |
| interface gain `mu` (14 configs) | `1.0` (constant map) | `1.000000` | ≤ 7e-8 (fit) | **yes** |
| Aitken tip record vs fixed `omega = 1` | equality expected if `mu = 1` | 0.000e+00 | **yes** |
| interface gain `mu` vs Lamb added-mass prediction | `1 + m_a/m_s` = 1.004 … 4.86 | `1.000000` | **0.4 % … 79 %** | **no — this is the defect** |
| frozen-fluid bound vs closed form `gamma dt M_f/(M_s + k beta dt^2)` | analytic | analytic | 0 (identity) | **yes** |
| under-relaxed (`omega = 0.05`) tip record vs Aitken | converged answer | 5.47 % deviation, 24/24 MAXIT | **no** (documented: MAXIT truncation biases the committed state) |

The one quantity that fails the 3 % bar is the one the round is about: the shipped driver's
interface gain is 1 where the physics requires `1 + m_a/m_s`; at `rho_f = 2`, `rho_s = 0.2` the
shortfall is 79 %. That is reported as the defect it is — not as a coupling-accuracy failure of
the coupled answers, which are unaffected (the sweep-independent tip histories show the fixed
point is reached in one sweep either way).

## 7. Reproduction

```bash
PYTHONPATH=src python benchmarks/xval_ibm_subiter_gain_figure.py    # 14-config gain sweep + short-record relax, ~50 s
PYTHONPATH=src python benchmarks/xval_ibm_subiter_gain.py --stage all   # full-length stages (compute gate: cj3827)
PYTHONPATH=src python -m pytest tests/test_ibm_subiter_gain.py -q    # 14 passed, ~6 s
PYTHONPATH=src python -m pytest tests/test_ibm_aitken_implicit.py tests/test_aitken_interface_certificate.py -q  # 44 passed
```

## 8. Files

| file | status |
|---|---|
| `src/tensorfem/ibm_subiter_gain.py` | **new**: gain-from-trace fit, marker-local mass, frozen-fluid bound, Lamb reference |
| `src/tensorfem/ibm_aitken_implicit.py` | + `residual_traces` diagnostic record (default empty; Aitken path unchanged) |
| `benchmarks/xval_ibm_subiter_gain.py` | **new**: gain / bound / relax / physics / fig stages |
| `benchmarks/xval_ibm_subiter_gain_figure.py` | **new**: cheap artifact generator (summary.json + figure) |
| `benchmarks/xval_ibm_subiter_gain_artifacts/` | `summary.json`, `xval_ibm_subiter_gain.png` |
| `tests/test_ibm_subiter_gain.py` | **new**: 14 tests (synthetic map, constant-map certificate on the production driver, bounds) |
| `docs/CASE-FEM-XVAL-IBM-SUBITER-GAIN.md` | this report |