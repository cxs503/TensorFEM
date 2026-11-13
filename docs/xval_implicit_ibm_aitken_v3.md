# XVAL — implicit IBM + Aitken relaxation: added-mass cross-validation (round 3)

Anchor `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION` (m18276, 2026-11-13).
Reference side: **joconnor22/LIFE** (Lattice Boltzmann – immersed boundary – finite
element; implicit direct forcing + Aitken Δ² sub-iterations).
Reference values below are **analytic**, not copied from the reference code:

| quantity | analytic reference |
|---|---|
| added mass, thin plate in broadside motion | `m_a = rho_f π c²` (Lamb 1932, art. 71) |
| natural frequency | `f = √(k/(m_s+m_a)) / 2π` |
| frequency shift | `f/f_vacuum = √(m_s/(m_s+m_a))` |

Prior rounds: `docs/xval_implicit_ibm_aitken.md` (round 1, port),
`docs/xval_implicit_ibm_aitken_v2.md` (round 2, `u_km1` order + analytic-frequency
fixes; open items §6: "submerged frequency needs longer run", "ε=1 transmission
deviation").  This round closes the first open item, and while closing it found a
**blocking defect in the coupling order**.

---

## 1. Case

A **thin rigid plate** of half-width `c` (marker segment, `2c/Δ + 1` markers,
spacing `Δ`), spring-mounted with one transverse dof, immersed in a quiescent
periodic D2Q9 box; markers are the plate itself (LIFE's surface-marker placement),
the plate force is spread only from the markers (no interior forcing).

| parameter | value |
|---|---|
| plate half-width `c` | 16 Δ (also 10, 20 in the resolution study) |
| lattice | 144² (also 96², 160²), `dx = dt = 1`, `ρ_f = 1`, `ω_LBM = 1.9` (ν = 0.0075) |
| body mass `m_s` | `m_a/0.6` = 1340 (added-mass ratio 0.6) |
| spring `k` | `(m_s+m_a)(2π/T)²`, target period `T = 600` steps |
| initial condition | released from `y₀ = 0.2 Δ`, fluid at rest |
| sub-iteration | `sub_tol = 1e-6` on `|Δy|/(2c)`, `max_sub_iter = 30` |

A plate is used deliberately instead of a closed body: a closed marker ring traps
the interior fluid, which is dragged only through the interface and adds a
non-analytic `≈1.5 m_a` of inertia (measured on a ring, `c = 10`), and a circular
boundary smears to an effective radius `c + O(Δ)` (a 10 % added-mass error at
`c = 10 Δ`).  For a plate there is no interior and the thickness-induced error is
`O((Δ/2c)²) ≈ 0.1 %`.

## 2. Method / toolchain

`src/tensorfem/ibm_aitken_added_mass.py` (new) + the existing IBM/LBM stack
`src/tensorfem/ibm_aitken_implicit.py`:

1. read the **pre-force** fluid velocity of the time level (`macroscopic(include_force=False)`);
2. drive the interface fixed point: marker positions → interpolate `ρ_i, m_i` →
   direct-forcing force `F = 2(ρ_i u_body − m_i)/dt` → implicit Newmark structure →
   Aitken Δ² relaxation of `y` until `|Δy|/(2c) ≤ sub_tol`;
3. apply the converged Lagrange force **inside the same LBM collision**
   (`LBMFluid2D.step(force=...)`, Guo half split) — not one step later;
4. frequency from the displacement record: zero-crossing estimate + Hann-windowed
   FFT peak (the FFT value is used when the record holds < 2 crossings).

Three experiments: **E0** structure alone (control), **E1** prescribed motion →
least-squares added mass `F_y = −m_a ÿ − c ẏ + O(y)`, **E2** coupled free
vibration, sweep over `m_a/m_s ∈ {0.15…1.57}` and over the relaxation mode.

## 3. Defect found and fixed (blocking)

`ImplicitIBMFSI` / `IBMImplicitFSI` advanced the fluid **first** and spread the
Lagrange force at the *end* of the step, so the force derived at level `k` was
applied in the collision of level `k+1` — a one-step lag in the two-way feedback.
With the lag, the coupled plate is **unstable**: the interface force alternates in
sign and grows by a factor ≈2 per step

```
F: 0.07, −0.20, 0.86, −1.24, 3.06, −5.65, 12.1, −24.0, 49.4, −99.7, 202, −400, 764, −1468, …
```

(`c = 10`, `m_a/m_s ≈ 0.15`, `Δ|u|` reaches 3371 and `ρ` reaches −4.98e4) and the
markers leave the lattice at step 21 in **every** configuration tried, including
`m_a/m_s = 0.15`.  Moving the force into the same collision removes the
instability: the same configuration now oscillates smoothly for ≥ 1200 steps with
`maxit_hits = 0`, `y: 0.2 → −0.153` over the first 55 steps.

## 4. Results

### 4.1 E0 — control: the structure alone (implicit Newmark)

| quantity | analytic | computed | error |
|---|---|---|---|
| `f` (`m = 200`, `k = 1.43`, `T ≈ 74.3`) | 0.01345782 | 0.01346154 | **+0.028 %** ✅ |

### 4.2 E2 — coupled frequency vs analytic added mass

| `c` | lattice | target `T` | steps | mode | measured `T` | `f/f_vac` | ref `f/f_vac` | error | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 96² | 300 | 600 | Aitken | 338.0 | 0.888 | 0.791 | **−11.2 %** | ✅ <15 % |
| 16 | 144² | 600 | 900 | Aitken | 682.7 | 0.695 | 0.791 | **−12.1 %** | ✅ <15 % |
| 16 | 144² | 600 | 900 | fixed ω=1 | 682.7 | 0.695 | 0.791 | **−12.1 %** | ✅ <15 % |
| 20 | 160² | 600 | 1200 | Aitken | 640.0 | 0.741 | 0.791 | **−6.2 %** | ✅ <15 % |
| 20 | 160² | 120 | 180 | Aitken | 97 | 0.741 | 0.791 | **+31.9 %** | ❌ |

Artifacts: `benchmarks/xval_added_mass_artifacts/evidence_plate16.json`,
`evidence.json` (quick sweep), `xval_added_mass_{history,frequency,force}.png`,
`xval_added_mass_summary.md`.

Reading of the numbers: the deviation is **frequency dependent**, which identifies
its two physical sources.

* `T = 120` steps → the acoustic wavelength is only ≈ 3.5 plate widths
  (`c_s T/L ≈ 1.7`): the flow is **not incompressible**, the added mass is
  under-predicted and the frequency comes out 32 % high.  The staggered implicit
  coupling cannot fix that — it is the compressibility of the LBM fluid.
* `T = 600…3000` steps → the added mass is **over-predicted** by 15–35 %
  (`m_a` fit 1718 vs 1257 analytic at `c = 20`), i.e. the measured frequency is
  6–12 % low.  The excess is dominated by the *stencil mass* of the direct-forcing
  operator: `F = 2ρ(u_b − u*)/dt` enforces no-slip in one step, so the fluid inside
  the `3×3` delta support (`≈ 9·(2c+1) ρ ≈ 370` for `c = 20`) is accelerated as if
  rigidly attached — `1257 + 370 = 1627` vs 1718 measured, the remainder being
  periodic-image confinement (`c/h = 0.125`).

### 4.3 Aitken vs fixed ω = 1 (honest negative result)

At the tested ratios the interface fixed point converges in **1.0 sub-iteration**
on average (residual `≤ 1e-6`), so Aitken and `ω = 1` give bit-identical records.
The added-mass ratios reachable in the stable band of this staggered scheme are
simply too mild for the relaxation to matter.  The scalar-probe evidence for
Aitken (`docs/xval_implicit_ibm_aitken_v2.md` §3.2, `ω* = 1/(1−g)` recovered to
machine precision) stands; what this round *falsifies* is the expectation that a
plate at `m_a/m_s = 0.6` needs it.

## 5. Performance work (same files)

`IBMInterface.move` dominated the cost of long runs.  Vectorised, semantics
preserved (all 27 pre-existing tests pass, ε identical to the loop version to
`3.6e-15`, uniform-force transmission error still `4.4e-16`):

| operation | before | after |
|---|---|---|
| `epsilon_conservation` assembly + solve | 90 ms (double Python loop) | 8.5 ms (separable `A = (Wx·Dx)(Wy·Dy)ds`, banded candidates) |
| `find_support` per marker | 0.2 ms (numpy on 1-element arrays) | 0.03 ms (scalar kernel `_dirac_scalar`) |
| `interpolate` / `spread` | per-marker Python loops | `bincount` / `np.add.at` over one flat stencil table |
| one coupled step (`c = 20`, 160²) | ≈ 1.1 s (unusable) | 0.053 s |

## 6. Error verdict (acceptance: <15 %, owner standard)

* E0 control: **+0.03 %** ✅
* Coupled frequency, design band (incompressible, `T ≥ 600`): **−6.2 %, −12.1 %** ✅
  `< 15 %` met on every converged case; **`< 3 %` is NOT met** — the residual 6–12 %
  is the direct-forcing stencil mass plus image confinement, quantified above, and
  it is systematic (always low frequency), not scatter.
* High-frequency case (`T = 120`): **+31.9 %** ❌ — outside the acceptance band,
  attributed to acoustic compressibility (`c_s T / L ≈ 1.7`), documented as a
  validity limit of the configuration rather than a solver defect.
* E1 added-mass fit at `c = 20`: outside the 15 % band (`+36.7 %`, `m_a = 1718` vs
  1257); same stencil-mass decomposition as §4.2.  Reported as **FAIL**, not
  hidden.
* Stability: one-step lag → divergence at step 21 (all ratios); after the fix
  → stable ≥ 1200 steps, 0 `maxit` hits.

## 7. Caveats / open items

1. The periodic box has no walls and no inflow; only the coupling algorithm is
   under test, not boundary treatment.
2. Both open items of round 2 are only partly closed: the "submerged frequency"
   item is closed for a plate (analytic reference) and still **open for the
   centreline-marker beam**, where the effective thickness is set by the delta
   support and no analytic reference exists — the beam path's own default
   (`cantilever_free_vibration`, scaled section) remains unvalidated.
3. The full sweep (`m_a/m_s ∈ {0.15…1.57} × {aitken, fixed}`, 1500 steps) was
   submitted as cluster job **cj1884** (TensorFEM, 2 cores) and was still queued
   when this report was written; the numbers above come from the local
   `--quick`/medium runs stored in the artifacts directory.
4. `ε = 1` force-transmission deviation of round 2 (61 % vs LIFE's 11–37 %) is
   untouched this round.

## 8. Files changed

| file | change |
|---|---|
| `src/tensorfem/ibm_aitken_added_mass.py` | new: plate/ring marker geometries, 1-dof sprung body, implicit IBM + Aitken driver, E0/E1/E2 experiments, period estimation |
| `src/tensorfem/ibm_aitken_implicit.py` | `LBMFluid2D.step(force=…)` / `macroscopic(include_force=…)`; interface force applied inside the same collision (lag removed) in `ImplicitIBMFSI.step`; vectorised `find_support` / `epsilon_conservation` / `interpolate` / `spread` |
| `benchmarks/xval_aitken_added_mass.py` | new: `--quick` / full sweep, artifacts + three figures + summary |
| `tests/test_ibm_added_mass_validation.py` | new: 8 tests (analytic refs, estimators, Newmark purity, lag-fix stability regression) |
| `benchmarks/xval_added_mass_artifacts/` | evidence JSON, `y_*` records, figures, summary md |
| `docs/xval_implicit_ibm_aitken_v3.md` | this report |

Verification run: `python -m pytest tests/test_ibm_added_mass_validation.py
tests/test_ibm_aitken_implicit.py tests/test_aitken_relaxation_validation.py -q`
→ **35 passed** (20 s).