# ASTM E399 specimens and Griffith anchors for the AT2 phase-field fracture model

Status: implemented, verified against closed-form fracture-mechanics calibrations.
Module: `tensorfem.phase_field_specimens`
Benchmark driver: `benchmarks/phase_field_fracture/run_astm_specimens.py`
Figures: `benchmarks/phase_field_fracture/make_astm_figures.py`
Tests: `tests/test_phase_field_specimens.py`

## 1. What was added and why

`tensorfem.phase_field_fracture` (AT2, 2-D plane strain, Amor split, history
variable `H`, monolithic Newton + a history-interleaved driver) carries the
damage physics but no *specimen*: it cannot answer the question a fracture
mechanics test asks, namely "what load does this standard coupon carry before
the crack runs?". This work adds that layer, and with it the pieces the
acceptance criteria need:

| Item | Where |
|---|---|
| C(T) compact tension geometry with pin-bored holes (element removal) | `ct_setup` |
| SE(B) three-point bend geometry with rigid rollers | `seb_setup` |
| SENT (Miehe 2010) geometry with displacement-controlled grips | `sent_setup` |
| consistent cosine pin pressure on a carved hole (exact resultant) | `pin_load_forces` |
| rigid-pin / rigid-roller displacement control | `rigid_two_sided`, `rigid_control_dofs` |
| ASTM E399 shape factors, C(T) and SE(B) | `astm_ct_factor`, `astm_seb_factor` |
| Irwin / Griffith anchors, plane strain | `irwin_k_from_gc`, `griffith_stress`, `sent_critical_stress` |
| analytical AT2 notch profile + zero-load relaxation | `initial_damage_field`, `relax_damage` |
| peak-load identification, energy ledger, crack path, `l -> 0` extrapolation | `peak_load`, `ledger_residuals`, `damage_path`, `fit_limit` |
| multi-dof displacement path with an exact external-work integral | `run_control_path` |
| true alternating (u-block / d-block) solver | `PhaseFieldSolver.alternating_step` |

## 2. Conventions (the traps)

* **C(T) crack length is measured from the load line.** ASTM E399 defines `W`
  as the distance from the pin-centre plane to the back face and `a` from that
  same plane, so the notch cut from the front face is `a + 0.25 W` long. Getting
  this backwards shifts `a/W` by 0.25 and the ASTM prediction by ~2x; the first
  calibration attempt in this work did exactly that (`K_fem/K_ASTM = 0.49`).
* **The pins do not slide.** Two horizontal restraints, one per pin bore, remove
  the rigid-body modes. Placing them at the back face instead changes the
  apparent `K` by ~4 % (measured), so the pin placement is the physically
  consistent one.
* **Removed elements leave orphan nodes.** Their damage dof has no equation;
  without pinning, the tangent matrix is exactly singular (`_pin_orphans`).
* **Rigid pins heat the contact stress.** The pin bores are damage-locked inside
  `1.8 r` (`lock_circle_damage`) because the linear-elastic contact singularity
  is an artefact of the rigid-pin idealisation, not a physical crack nucleus.
* **Initial condition matters for the energy ledger.** A sharp `0/1` notch and
  an analytical `exp(-r/l)` field both cost spurious fracture energy at the first
  step (measured: ledger residual 2.3e4 %, resp. 2.7e3 %, at the first step). The
  zero-load relaxation of the `d` sub-problem (`relax_damage`, 14 alternating
  iterations to 3e-9) removes it: the ledger residual drops to < 0.05 %.
* **`d_max` must exclude prescribed damage.** The notch is held at `d = 1`, so a
  raw `max(d)` stop criterion trips on the first step; `run_control_path` reports
  `d_max_free` over the free damage dofs.

## 3. Verification (numbers in `artifacts/phase_field_fracture/`)

* `ct_lefm` – energy-release rate of a slit C(T) crack (central difference in
  `a`, pin loaded) vs the ASTM E399 `K = P/(B sqrt(W)) f(a/W)` calibration.
* `ct_peak` / `seb_peak` – phase-field peak load under rigid-pin / rigid-roller
  displacement control vs the ASTM E399 peak-load prediction built from
  `K_Ic = sqrt(Gc E')`.
* `lconv` – SENT peak stress against the finite-width Griffith anchor for three
  regularisation lengths, with a `l -> 0` least-squares limit.
* `staggered` – true alternating scheme against monolithic Newton, and the
  coupled-residual history of the alternating loop.

The per-case numbers, the error percentages and the explicit pass/fail of every
acceptance criterion are recorded in the report of the mission that produced this
document; the raw records (load-displacement paths, residual histories, damage
fields) live in the JSON artefacts next to it.

## 4. Limitations

* 2-D plane strain, structured Q4 mesh, element-removal holes: the bore boundary
  is stair-cased, which biases the C(T) calibration low by ~5 % at `h = l/2`
  (measured, `ct_lefm`). Refining `h` at fixed `l` does not remove the bias
  (5.7 % at `h = l/4`), so it is a geometry-idealisation, not a discretisation,
  error.
* The `l -> 0` extrapolation is a one-parameter (`sigma = sigma_0 + C l`) fit of
  three points; it is a trend statement, not a converged limit.
* The alternating loop updates `H` after every u-block solve. On strongly
  damaging steps this lets the damage front creep, so the propagation runs use
  the monolithic Newton solver; the alternating scheme is demonstrated on the
  sub-critical to initiating part of the path where it reaches the 1e-6 coupled
  residual.
* Peak loads are measured on the equilibrated part of the path only; the step on
  which the post-peak branch becomes unresolvable is reported
  (`peak.n_nonconverged`) rather than hidden.