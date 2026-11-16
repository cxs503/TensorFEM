# CASE-FEM-PLATE-PATCH-PLASTIC-LADDER-RAMP-GUARD (anchor GAP-PP-3-ICE-STRUCT-PLASTIC-SET)

Verification-integrity round on the plate-panel elastoplastic capacity case. This
delivery does **not** add a new constitutive capability: it repairs the *evidence
chain* of the three-mesh convergence acceptance item, banks two already-paid-for
compute results that were never written to the repository, and states the honest
status of the acceptance item. Every number below is emitted by
`scripts/patch_plate_plastic_mesh_ladder.py` / `tensorfem.plate_patch_plastic_limit.py`
or copied from `artifacts/provenance/cj3424/PROVENANCE.json`; nothing is transcribed
by hand.

## 1. Case (unchanged, for reference)

| quantity | value |
|---|---|
| plate | simply supported, a = b = 1.000 m, t = 20 mm (t/a = 0.020), in-plane displacements free |
| patch load | 500 x 500 mm centred = 25 % of the area |
| material | DH36 @ -40 C, sigma_y = 383800000000000.0 MPa, E = 210 GPa, nu = 0.3, H = 1 MPa |
| element | layered Mindlin Shell4 (Q4, 2x2 Gauss bending), 4 layers, exact through-thickness integration |
| analytic first-yield pressure | 869233646805.9257 MPa |
| admissible bracket | strip lower 818773333333.3334 MPa < p_c < 1842240000000.0000 MPa |

## 2. What was found (defects in the evidence chain, not in the solver)

**Defect 1 - the delivered ladder mixed two different prescribed ramps.**
The repository held one ladder record (`nx = 8`, 5 load points, 0.4 p_y step, top
1.8 p_y) next to a dense record (`nx = 12`, 21 points, 0.1 p_y step) that the
analysis step had never been given (it was still inside compute job `cj3424`). The
two limits are

| record | ramp | points | limit (tight) | criterion spread | elapsed |
|---|---|---|---|---|---|
| nx = 8  | coarse 0.4 p_y | 5  | 1199742500866.4797 MPa | n/a (single definition fired) | 49 s |
| nx = 12 | dense 0.1 p_y  | 21 | 1825362001052.7627 MPa | 0.0676 | 1052 s |

A +52.1 % difference between the two is
**not** a mesh effect: a coarse ramp under-resolves the plastic tangent, so the
two-line / stiffness criteria fire in the wrong interval. `analyse()` used to glob
every record and difference them regardless, i.e. the convergence statement was
open to being produced by the ramp rather than by the mesh.

**Defect 2 - `analyse()` crashed on any artifact that predates the
`johansen_twist` bound** (`KeyError: 'johansen_twist'` at the criterion gate), so
the analysis step could not even be run over legacy records; two tests in
`tests/test_plate_patch_plastic_mesh_ladder.py` were red on the delivered tree for
this reason.

**Defect 3 - the criterion-gate annotation leaked onto the membrane records**,
which are explicitly *not* judged against bending bounds.

## 3. What was changed

1. `run_one` now records the prescribed ramp (`ramp.factors_over_py`, step, top,
   point count) in every ladder artifact - the comparability key of a ladder.
2. `ramp_signature()` / `comparable_ladders()` / `_ramp_label()`: `analyse()`
   groups records per geometry by (ramp, layers, hardening, yield stress, patch).
   When more than one family exists, only the largest family is admitted to the
   convergence / Richardson machinery; every other record is moved to
   `summary["records_excluded"]` **with the reason printed and stored** - never
   silently dropped, never silently differenced.
3. The `johansen_twist` gate falls back to the yield-line bound when the bound is
   absent from the artifact's reference set.
4. The criterion-gate flag is written only to the in-plane-free (bending-only)
   records.
5. `artifacts/provenance/cj3424/` banks the two compute results recovered from
   job `cj3424` together with their provenance note (they were paid for on
   11-15 and never committed).

Reproduce:

```bash
PYTHONPATH=src python -m pytest tests/test_plate_patch_plastic_mesh_ladder.py -q
PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --analyse
```

## 4. Reference solutions of the case (closed form, independent of any FE run)

| bound | value [MPa] | admissible? |
|---|---|---|
| strip lower bound | 818773333333.3334 | lower |
| Ilyushin sine upper bound (terms = 1) | 1515181667655.2383 | **no** - it lies below the FE collapse of the same continuum problem, so it is not an upper bound for a smooth patch mechanism |
| yield-line upper bound (twist retained) | 1842240000000.0000 | yes |
| Johansen twist upper bound | 2379982060089.8833 | yes (looser) |
| tightest admissible upper | 1842240000000.0000 | - |
| admissible bracket width | 2.25 (> 1) | - |

**Criterion check.** `criterion_discrimination` on the admissible bracket:
verdict `not_discriminative`, bracket width 2.25 against
the 1.111 needed for a 10 % test, i.e. the best
possible relative error any solver could achieve inside this bracket is
55.6 %. The ledger acceptance
"limit load within 10 % of the Ilyushin/upper-bound analysis" is therefore
**not discriminative on this case as written**: a single number inside the bracket
cannot both be within 10 % of the upper bound and be a convergence statement. This
is a property of the bound set, not of the solver, and it is why the verdicts below
are reported per bound.

## 5. Error judgement (honest, acceptance item by item)

| acceptance item | measured | threshold | verdict |
|---|---|---|---|
| elastic -> plastic transition vs analytic first yield | +0.79 % (nx = 8) / +0.20 % (nx = 12), elastic field | < 5 % | **met** |
| plastic limit load vs **admissible** upper bound (yield line, twist retained) | nx = 12: -0.92 % | < 10 % | **met on the single dense mesh** |
| plastic limit load vs Ilyushin sine bound | nx = 12: +20.47 % | < 10 % | **not met** (and inadmissible yardstick, see §4) |
| 3-level mesh convergence (8 / 12 / 16 on one ramp) | two comparable meshes not yet available: the coarse nx = 8 record is excluded by the new guard, and the corrected dense 8 / 12 / 16 runs are queued (cj4681 / cj4682 / cj4683) | monotone + Richardson | **not_run** |
| permanent deflection (unloading) on the same run | nx = 12 unloading did not converge -> withheld; nx = 8 coarse-ramp value 5.21 mm is not certified | reported | **withheld** |

Overstated claims retired by this round:

* the nx = 8 "limit load" 1.1997 MPa is a coarse-ramp artefact and is **not** a
  mesh-converged value;
* any statement that the case already has a three-mesh convergence evidence:
  it does not - it had one coarse record and one dense record that were never
  comparable.

## 6. Caveats / next step

* The dense-ramp campaign for nx = 8 / 12 / 16 on an identical prescribed ramp is
  queued (`run_compute` jobs cj4681, cj4682, cj4683). When they land, run
  `--analyse` and `scripts/patch_plate_plastic_ladder_figures.py`; the guard now
  makes the Richardson estimate impossible to obtain from mixed ramps, so the
  regenerated document either has a genuine three-mesh order or says `not_run`.
* The nx = 12 record shows a 6.8 % spread between five limit-load definitions and
  a truncated ramp; a converged number must be quoted with that spread, not as a
  single figure.
* The whole comparison is a bending-only (in-plane free) statement: the layered
  flat shell has no membrane coupling, so grillage-panel (`membrane`) records are
  excluded from every acceptance table (unchanged policy, now enforced by tests).
