# `fem_kirsch_plate` — CASE-KIRSCH-PLATE: stress concentration of a circular hole, `K_t = 3`

Verification anchor of the benchmark case pool (`CASE-KIRSCH-PLATE`). One
self-contained closed-form oracle, two independent discretisations, a refinement
ladder, graded gates and machine-readable evidence — the same layout as the
IN36-N01 anchors in the parent directory.

```
result.json        every number (ladder, cross-check, sensitivity studies) — generated
gates_verified.json schema tensorfem.verified-gates.v1, 14 gates, per-gate value/threshold/verdict
kirsch_field.png            sigma_xx/T on the quarter plate (d/W = 0.02, n = 48)
kirsch_convergence.png      |K_t - 3|/3 vs h for both element families
kirsch_hoop_profile.png     hole-edge hoop stress vs the exact 1 - 2 cos 2θ
kirsch_width_trend.png      gross and net K_t vs d/W against Howland/Heywood
run.py                      the driver
```

## 1. Problem and reference (where the number comes from)

A plate of width `W` and thickness `t` with a central circular hole of diameter
`d = 2a` carries a uniform far-field uniaxial tension `T` (gross stress
`σ0 = T`). Kirsch solved the **infinite** plate exactly. In polar coordinates
centred on the hole, with `θ` measured from the load axis:

```
σ_rr = T/2 (1 - a²/r²) + T/2 (1 - 4a²/r² + 3a⁴/r⁴) cos 2θ
σ_θθ = T/2 (1 + a²/r²) - T/2 (1 + 3a⁴/r⁴) cos 2θ
σ_rθ = -T/2 (1 + 2a²/r² - 3a⁴/r⁴) sin 2θ
```

> S. Timoshenko & J.N. Goodier, *Theory of Elasticity*, 3rd ed. (McGraw-Hill,
> 1970), **Art. 39** "Stresses in a plate with a circular hole", eqs. (a)–(c);
> original solution: G. Kirsch, *VDI* **42** (1898) 797.

On `r = a` this reduces to `σ_θθ(a, θ) = T (1 - 2 cos 2θ)`: `-T` at the poles on
the load axis, and

```
K_t = σ_θθ(a, π/2) / T = 3                        (1)
```

at the two poles perpendicular to the load. `K_t` depends on neither `E` nor
`ν` (for traction-only boundary conditions the stress field of a plane problem
is independent of the elastic constants), and on no length scale.

**Zero hand-copy.** The reference `3` is *evaluated* from (1) inside
`run.py::oracle_selfcheck` and inside `tests/test_kirsch_plate_anchor.py`; the
same routine is then verified by its own boundary conditions
(`|σ_rr|, |σ_rθ| ≤ 1e-12 T` on `r = a`, uniaxial far field `σ_xx → T`,
`σ_yy → 0` at `r = 200a`). No handbook digit enters the evidence chain — the
citation above locates the formula, the code supplies the number.

Finite width. For `d/W > 0` the peak stress is *not* `3T`; the gross-basis
factor `σ_peak/T` grows with `d/W` while the net-basis factor
`K_tn = σ_peak/(T/(1-d/W))` falls below 3. The anchor therefore never claims
`K_t = 3` at finite width: it (i) drives `d/W` to 0.02, (ii) Richardson-
extrapolates in `h`, and (iii) grades the whole `d/W = 0.1 … 0.5` band against
the Howland/Heywood fit `K_tn = 2 + (1-d/W)³` already carried by
`tensorfem.finite_width_hole` (comment there: Howland 1930 as tabulated in
Peterson, *Stress Concentration Factors*, 3rd ed., Chart 4.1, ±1 % read-off).

## 2. Method and implementation

Two independent discretisations of the same physics, both **existing production
paths** — this anchor adds no solver code, it certifies what the library already
does:

1. **Traction-driven finite-width quarter plate** — `tensorfem.finite_width_hole`
   (Q4 plane stress, starburst rays from the hole, geometric grading 6.0).
   Quarter model `0 ≤ x ≤ L`, `0 ≤ y ≤ W/2`, hole quadrant at the origin:
   `u_x = 0` on `x = 0`, `u_y = 0` on `y = 0` (symmetry), hole edge and the
   lateral edge `y = W/2` traction free, uniform tension `T` consistent-nodal
   loaded on `x = L`. Nothing of the exact solution is injected, so this run
   carries the discretisation error *and* the finite-width/finite-length model
   error. `K_t` is the recovered nodal `σ_xx` at the apex node `(0, a)`, with
   the whole edge profile returned so the peak location is *checked*
   (all levels: `θ_peak = 90.000000°`) rather than assumed.
2. **Dirichlet-driven Q8 polar annulus** — `tensorfem.structural_canonical_face
   ::solve_kirsch_q8_concentration` (8-node serendipity plane stress, exact
   Kirsch displacement prescribed on `r = 2a`, hole edge left free). Different
   element family, mesh topology and boundary formulation; isolates the
   discretisation error.

Post-processing: `K_t^{(gross)} = σ_θθ(a, π/2)/T`,
`K_t^{(net)} = K_t^{(gross)} (1 - d/W)`; recovery error judged against (1);
observed order and GCI from three successive dyadic meshes (Roache, safety
factor 1.25); traction equilibrium checked as
`|ΣR_x - T·W/2·t| / (T·W/2·t)`.

### Parameter table (all dimensionless groups)

| quantity | symbol | anchor value | study range |
|---|---|---|---|
| hole/width ratio | `d/W` | 0.02 | 0.02, 0.05, 0.10, 0.2, 0.3, 0.4, 0.5 |
| plate half-length | `L/W` | 1.5 | 1.0, 1.5, 2.5, 4.0 |
| Poisson ratio | `ν` | 0.3 | 0.0, 0.30, 0.45 |
| stiffness | `E/T` | `2e11` (E in Pa, T = 1 MPa ⇒ `E/T = 2e5`) | 2 GPa, 200 GPa |
| thickness | `t/W` | 0.01 | fixed (plane stress, `t` cancels) |
| mesh density | `1/n`, `n = n_boundary = n_radial` | 1/48 | 1/12, 1/24, 1/48 |
| Q8 annulus density | `1/n_θ` | 1/96 | 1/24, 1/48, 1/96 |
| annulus extent | `R_out/a` | 2.0 | 2.0 |

`K_t` is a pure number: absolute stress, `E`, `t` and the unit system drop out,
which is why the table has no dimensional entry other than the `E/T` ratio used
to prove the invariance.

## 3. Results

Ladder (traction-driven quarter plate, `L/W = 1.5`, `grading = 6`):

| `d/W` | `n` | elements | DOFs | `K_t^{(gross)}` | `K_tn` | err vs 3 | edge-profile RMS/3 | load balance |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 12 | 144 | 338 | 2.879925 | 2.822327 | 4.002 % | 2.141 % | 7.5e-15 |
| 0.02 | 24 | 576 | 1250 | 2.977466 | 2.917917 | 0.751 % | 0.622 % | 7.3e-15 |
| **0.02** | **48** | **2304** | **4802** | **3.001408** | 2.941379 | **0.047 %** ✔ | 0.318 % | 5.5e-14 |
| 0.05 | 12 | 144 | 338 | 2.861285 | 2.718221 | 4.624 % | 1.995 % | 5.5e-16 |
| 0.05 | 24 | 576 | 1250 | 2.970483 | 2.821959 | 0.984 % | 0.531 % | 6.2e-15 |
| 0.05 | 48 | 2304 | 4802 | 3.001349 | 2.851281 | 0.045 % ✔ | 0.283 % | 1.7e-14 |
| 0.10 | 12 | 144 | 338 | 2.874571 | 2.587114 | 4.181 % | 1.900 % | 1.2e-14 |
| 0.10 | 24 | 576 | 1250 | 2.991240 | 2.692116 | 0.292 % ✔ | 0.718 % | 1.4e-14 |
| 0.10 | 48 | 2304 | 4802 | 3.025609 | 2.723048 | 0.854 % | 0.741 % | 3.3e-14 |

Q8 annulus cross-check (`R_out/a = 2`, exact Kirsch displacement on the rim):

| `n_θ × n_r` | nodes | `K_t` | err vs 3 | `‖u-fe − u-exact‖₂/‖u-exact‖₂` |
|---|---|---|---|---|
| 24 × 6 | 480 | 3.019739 | 0.658 % | 9.49e-5 |
| 48 × 12 | 1824 | 3.005729 | 0.191 % | 5.89e-6 |
| 96 × 24 | 7104 | 3.001476 | **0.049 %** ✔ | 3.62e-7 |

Reference vs computed (the campaign table):

| quantity | reference (source) | computed | error | verdict |
|---|---|---|---|---|
| `K_t`, infinite plate, uniaxial | 3 (Timoshenko & Goodier 3rd ed. Art. 39 eq. (1)) | 3.001408 (`d/W`=0.02, `n`=48) | 0.047 % | ✔ < 5 % |
| `K_t`, Richardson limit, `d/W`=0.02 | 3 | 3.009196 (p = 2.03, GCI = 0.32 %) | 0.307 % | ✔ < 1 % |
| `K_t`, Richardson limit, `d/W`=0.10 | 3 | 3.039961 (p = 1.76, GCI = 0.59 %) | 1.33 % | ✔ < 5 % |
| `K_t`, independent Q8 element | 3 | 3.001476 (`n_θ`=96) | 0.049 % | ✔ < 5 % |
| `σ_θθ(a,θ)` shape | `1 − 2cos 2θ` | RMS dev. 0.318 % of `K_t` | — | ✔ < 5 % |
| gross `K_t`, `d/W` = 0.1 … 0.5 | Howland/Heywood fit | 3.0205, 3.1328, 3.3486, 3.7107, 4.2919 | ≤ 0.98 % | ✔ < 3 % |
| net `K_tn`, `d/W` = 0.1 … 0.5 | Peterson Chart 4.1 (2.73, 2.51, 2.35, 2.24, 2.16) | 2.7185, 2.5063, 2.3440, 2.2264, 2.1459 | ≤ 0.65 % | ✔ < 3 % |
| traction equilibrium | statics | 1.0e-12 (max, all runs) | — | ✔ < 1e-10 |

## 4. Convergence and sensitivity

**Mesh convergence (4 studies, 12 solves).** The error falls monotonically and
dyadically: 4.00 % → 0.751 % → 0.047 % at `d/W` = 0.02 (factor 5.3, 16×);
observed order `p = 2.03` (0.02), `1.82` (0.05), `1.76` (0.10) from three dyadic
meshes, GCI_fine ≤ 0.59 %. `p ≈ 2` is *not* a general claim for Q4 stress: it is
the starburst-mesh + apex-node behaviour of this geometry. The Q8 annulus gives
`p = 1.72` on its own dyadic ladder.

**Sensitivity 1 — Poisson ratio and stiffness (`ν ∈ {0, 0.3, 0.45}`,
`E ∈ {2, 200} GPa`).** `K_t` = 2.95646, 2.97747, 2.99698, 2.97747 at
`d/W` = 0.02, `n` = 24. Changing `E` by 100× reproduces `K_t` to 1.2e-14 relative
(round-off in the sparse solve), confirming the exact `ν`/`E` independence;
the 1.4 % spread across `ν` is *numerical* (linear elasticity makes the stress
field elastic-constant independent, yet the recovered nodal peak at fixed mesh
moves by ~0.7 % per `ν` step) and has the same size as the mesh error at
`n` = 24 (0.75 %) — i.e. it measures recovery error, not physics. That is why
the gate is 1 %, not 0.1 %.

**Sensitivity 2 — plate length (`L/W ∈ {1.0, 1.5, 2.5, 4.0}`).** With the
resolution matched (`n = n_boundary = n_radial ∝ L`, so the physical element
size at the hole is fixed) `K_t` = 2.97263, 2.97747, 2.97918, 2.97994: a 0.22 %
band over `L/W ∈ {1, 1.5, 2.5}` that shrinks monotonically to the long-plate
value — load introduction is a ≤ 0.25 % effect at the designated `L/W` = 1.5.
⚠ **Confounder found the hard way**: with `n` held *fixed* while `L` grows, the
same rays spread over a longer plate and the angular spacing near the apex
coarsens; `K_t` then drifts 2.995 → 2.978 → 2.931 → 2.843 (5.3 %) and looks
exactly like a Saint-Venant effect. It is discretisation. Both tables are in
`result.json` (`saint_venant`, `saint_venant_fixed_n_confounder`).

**Sensitivity 3 — finite width (`d/W` = 0.1 … 0.5, `n` = 40).** Gross `K_t`
rises 3.0205 → 4.2919 and net `K_tn` falls 2.7185 → 2.1459, both monotone, both
within 0.98 % of Howland/Heywood, and the trial with `d/W` = 0.02/0.05 brackets
the Kirsch limit from the finite-width side. At fixed `d/W` the FE converges to
the *finite-width* value, not to 3 — which is why the ladder, the Richardson
limit and the trend gate are all reported.

## 5. Failure modes, scope and traps

* **Coarse meshes under-predict, never over-predict** `K_t` (2.880 → 3.001);
  a coarse-mesh pass therefore cannot be mistaken for a conservative result,
  but a coarse-mesh *fail* (4.0 %) says nothing about the model.
* **`θ` sampling**: at `n` = 12 the peak is resolved at the apex node to 1e-12
  degrees here, but the edge-profile RMS error (2.1 %) shows the *profile* is
  rougher than the apex value: do not read `K_t` off an arbitrary edge node.
* **Finite width/length must be removed, not ignored** — see the two sensitivity
  studies; the reported `3` is a limit, and the anchor proves it is approached
  from the right side.
* **The Q8 cross-check is Dirichlet-driven** (exact displacement on `r = 2a`),
  so it can only see discretisation error; it cannot certify the far-field
  model. That is the whole point of pairing it with the traction-driven plate.
* **Scope**: plane stress (thin plate), linear isotropic elasticity, single
  circular hole, no plasticity/contact/thermal effects. A thick plate with a
  hole (`σ_zz ≠ 0`) is a different problem (`ν`-dependent) and is *not* covered.
* **Runtime** 87 s single-node (`loongarch64`, 8 threads) for the whole anchor;
  no cluster job needed. The 5 s/level cost at `n` = 48 is the local sparse
  solve, not an ecosystem bottleneck.

## 6. Reproduce

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/run.py
PYTHONPATH=src pytest tests/test_kirsch_plate_anchor.py -q     # 6 tests, ~15 s
```

`run.py` rewrites `result.json`, `gates_verified.json` and the four PNGs; the
test re-derives the oracle, checks the artifacts are self-consistent (every
gated value equals the number in `result.json`, no gate passes on a stale file)
and re-solves the coarse plate end to end.

> **Figure provenance (m15393 audit).** The four figures are regenerated by `run.py`
> into this directory, but the mission publisher filters `*.png` outside `docs/`
> (`tp_pusher.py` `_EXCLUDE2`), so the committed copy of this anchor was merged into
> main without them. Version-controlled mirrors live in `docs/assets/kirsch_plate/`
> (byte-identical to the `d3db3dbf` blobs; guarded by
> `tests/test_kirsch_plate_anchor.py::test_figures_tracked_mirror`).
