# CASE-BENDING-PATCH — round-2 repair of the evidence gate set

**Anchor**: `benchmarks/verified/fem_bending_patch` · **mission**: CASE-BENDING-PATCH ·
**scope**: the three remediation items left by the independent audit
(`docs/board_audit/m15654/AUDIT.md`, verdict m16138) plus one defect the regeneration
exposed. The case-pool tolerance (`2%`), the exact tolerance (`1e-12`), the order bound
(`1.5`) and the locking-control bound (`0.5`) are **unchanged** — no threshold was moved
to make a gate pass.

## 1. What was wrong

### 1.1 F-CRIT: the Poisson cross-check graded a closed form against itself

The gate `poisson_crosscheck` was computed as

```python
meas = free['w_exact_kirchhoff'] / sym['w_exact_kirchhoff']   # both closed forms
rel_error = abs(meas - 1/(1-nu^2)) / (1/(1-nu^2))            # == 0 for any element
```

The two closed forms differ only by the rigidity, so their ratio is `1/(1-nu^2)` by
algebra: the gate reported exactly `0` on every run
for every mesh, independently of the element. It could not detect a Poisson-coupling
defect, yet the report called it "the physical cross-check".

### 1.2 F4: two different criteria blocks

`result.json` carried `['case_pool_tolerance', 'exact_tolerance']` and `gates_verified.json`
`['case_pool_tolerance', 'exact_tolerance_affine', 'locking_control_min', 'observed_order_min']`; a consumer reading only `result.json` never saw
`observed_order_min` or `locking_control_min`.

### 1.3 F3: the free-edge gate had a thin margin and an unexplained residual

It passed at 1.758 % against a 2% bound (margin 1.14x) without saying whether that residual was
discretization or a model difference.

### 1.4 New: the committed artefacts were stale, and the order gate was ill-posed

Re-running the driver on the current `main` (after the MITC covariant shear-sampling
repair, PR #765 / `docs/mitc-skew-invariance.md`) reproduces every affine gate, but the
MITC distorted-patch error collapsed from ~1e-3 to round-off:

| ladder n | MITC w error (committed artefact) | MITC w error (regenerated) |
|---|---|---|
| 4 | 9.118e-03 | 3.634e-16 |
| 8 | 2.109e-03 | 6.643e-16 |
| 16 | 6.133e-04 | 8.328e-16 |
| 32 | 1.792e-04 | 1.135e-15 |

The old `order_w_min` gate took `min(order_sri, order_mitc)`; a least-squares slope
through round-off points is noise, and it came out negative, so the anchor would have gone
**red on `main`** for a change that made the element better. The gate was repaired by
making it well posed, not by lowering it (§2.3).

## 2. What this round changes

### 2.1 The cross-check now divides the two computed solutions

`src/tensorfem/bending_patch.py` gains `strip_rigidity_crosscheck(symmetry, free,
poisson)`, which divides the **computed** tip deflections `w_tip` of the two FE solves and
returns, separately and explicitly flagged, the closed-form ratio. `run.py` evaluates it on
every matched mesh/treatment pair and gates the **worst** pair against the unchanged 2%
bound. The result records `operands = "computed w_tip"` and `is_identity = false`, so a
consumer can tell a measurement from an identity.

| n_x | treatment | w_tip symmetry | w_tip free | ratio | theory | error |
|---|---|---|---|---|---|---|
| 4 | mitc | 25999.999998 | 28069.244482 | 1.079586 | 1.098901 | 1.758 % |
| 4 | sri | 25999.999999 | 28224.555776 | 1.085560 | 1.098901 | 1.214 % |
| 8 | mitc | 26000.000000 | 28165.685698 | 1.083296 | 1.098901 | 1.420 % |
| 8 | sri | 26000.000000 | 28368.410148 | 1.091093 | 1.098901 | 0.711 % |
| 16 | mitc | 25999.999999 | 28203.097183 | 1.084735 | 1.098901 | 1.289 % |
| 16 | sri | 26000.000000 | 28424.344939 | 1.093244 | 1.098901 | 0.515 % |
| 32 | mitc | 26000.000007 | 28214.113768 | 1.085158 | 1.098901 | 1.251 % |
| 32 | sri | 26000.000057 | 28441.379201 | 1.093899 | 1.098901 | 0.455 % |

Worst pair `n_x=4/mitc`: 1.758 % —
gate value before `0`, after `0.017576` (bound 2%, unchanged). The closed-form ratio is
still shown (1.098901, residual 0.000 %) and labelled an identity.

### 2.2 One criteria block, and the width study that explains the free-edge margin

`result.json.criteria == gates_verified.json.criteria == {"case_pool_tolerance": 0.02, "exact_tolerance_affine": 1e-12, "locking_control_min": 0.5, "observed_order_min": 1.5}`.
The free-edge residual is decomposed by refining the width (`n_y`), where an anticlastic
edge layer would live. If it were through-width discretization it would fall with `n_y`;
it does not:

| n_x | n_y | treatment | w tip | beam closed form | error |
|---|---|---|---|---|---|
| 8 | 2 | sri | 28368.410148 | 28571.428571 | 0.711 % |
| 8 | 2 | mitc | 28165.685698 | 28571.428571 | 1.420 % |
| 8 | 4 | sri | 28187.537314 | 28571.428571 | 1.344 % |
| 8 | 4 | mitc | 28156.905976 | 28571.428571 | 1.451 % |
| 8 | 8 | sri | 28172.751276 | 28571.428571 | 1.395 % |
| 8 | 8 | mitc | 28160.629408 | 28571.428571 | 1.438 % |
| 8 | 16 | sri | 28171.512620 | 28571.428571 | 1.400 % |
| 8 | 16 | mitc | 28165.149168 | 28571.428571 | 1.422 % |
| 8 | 32 | sri | 28171.237400 | 28571.428571 | 1.401 % |
| 8 | 32 | mitc | 28167.707136 | 28571.428571 | 1.413 % |
| 16 | 2 | sri | 28424.344939 | 28571.428571 | 0.515 % |
| 16 | 2 | mitc | 28203.097183 | 28571.428571 | 1.289 % |
| 16 | 4 | sri | 28230.755981 | 28571.428571 | 1.192 % |
| 16 | 4 | mitc | 28198.860910 | 28571.428571 | 1.304 % |
| 16 | 8 | sri | 28215.727301 | 28571.428571 | 1.245 % |
| 16 | 8 | mitc | 28203.961926 | 28571.428571 | 1.286 % |
| 16 | 16 | sri | 28214.912697 | 28571.428571 | 1.248 % |
| 16 | 16 | mitc | 28208.939392 | 28571.428571 | 1.269 % |
| 16 | 32 | sri | 28214.777097 | 28571.428571 | 1.248 % |
| 16 | 32 | mitc | 28211.893541 | 28571.428571 | 1.258 % |

* `sri` at n_x = 16: 0.515 % (n_y = 2) → 1.248 % (n_y = 16) → 1.248 % (n_y = 32); the last
  refinement moves the value by 0.038 %.
* `mitc` at n_x = 16: 1.289 % (n_y = 2) → 1.269 % (n_y = 16) → 1.258 % (n_y = 32); the last
  refinement moves the value by 0.822 %.

With element size across the width 0.00625 against
plate thickness 0.01, the through-width mesh is finer
than the edge layer it must resolve, so the surviving residual is the difference between
the 1-D beam closed form and the Mindlin plate strip with free edges. The gate is
therefore a *model-comparison* gate; it is reported with its margin and not relaxed.

### 2.3 A well-posed convergence-order gate

`observed_order()` now takes a floor (`ORDER_FLOOR = 1e-10` relative error,
declared in code before the regenerated numbers were inspected) and excludes round-off
rows; with fewer than two rows left it returns `nan`, and the gate value becomes the `min`
over the treatments that still have a defined order. Serialisation writes `null` instead
of `NaN`, so the artefact stays valid JSON.

| treatment | rows used | max w error | observed order | status |
|---|---|---|---|---|
| sri | 4 | 1.542e-02 | 1.72 | discretization-limited |
| mitc | 0 | 1.135e-15 | n/a | round-off limited, order not defined |

## 3. Gate set after the repair

| gate | value before | value after | threshold | passed |
|---|---|---|---|---|
| oracle_selfcheck | 1.41903e-13 | 1.41903e-13 | 1e-10 | PASS |
| affine_exact_w | 8.03113e-16 | 8.03113e-16 | 1e-12 | PASS |
| affine_exact_rot | 4.19691e-16 | 4.81868e-16 | 1e-12 | PASS |
| affine_exact_moment | 9.39362e-16 | 7.12476e-16 | 1e-12 | PASS |
| affine_exact_energy_sri | 1.54638e-15 | 1.54638e-15 | 1e-12 | PASS |
| skew_energy_mitc | 0.00396991 | 9.02056e-16 | 0.02 | PASS |
| distorted_w | 0.0154152 | 0.0154152 | 0.02 | PASS |
| distorted_moment | 0.00751051 | 0.00751051 | 0.02 | PASS |
| distorted_energy | 0.00165008 | 0.00155139 | 0.02 | PASS |
| strip_symmetry_w | 2.17532e-09 | 2.17532e-09 | 0.02 | PASS |
| strip_symmetry_theta | 1.36863e-09 | 1.36863e-09 | 0.02 | PASS |
| strip_free_w | 0.0175764 | 0.0175764 | 0.02 | PASS |
| poisson_crosscheck | 0 | 0.0175764 | 0.02 | PASS |
| order_w_min | 1.71737 | 1.71737 | 1.5 | PASS |
| locking_control_strip | 0.978528 | 0.978528 | 0.5 | PASS |
| locking_control_patch | 0.0210966 | 0.0210966 | 0.02 | PASS |
| equilibrium_residual | 4.26326e-14 | 4.26326e-14 | 1e-08 | PASS |

Verdict: **True**, failed gates `[]`, undefined terms `[]`.

## 4. Honest limits of this repair

* The cross-check value is now a real measurement, and it is *worse* than the old 0.0:
  1.758 % worst case, i.e. a 1.14x margin. It is the same physical residual as
  `strip_free_w` — the two gates are algebraically linked through the free-edge strip — so
  this repair adds no independent accuracy claim; it stops a vacuous gate from being
  counted as one.
* The residual does not vanish under through-width refinement, so `strip_free_w` and
  `poisson_crosscheck` grade model-to-model agreement (Mindlin free-edge strip vs 1-D beam),
  not convergence to an exact solution. Both remain inside 2%.
* The MITC distorted-patch numbers changed because the element changed (PR #765), not
  because this round tuned anything; the artefact is regenerated from `src/`, never edited.
* `run.py --quick` (two ladder levels) cannot define an order, so `order_w_min` fails by
  construction there — that is pre-existing behaviour of the reduced mode, not a regression;
  the full four-level run is the authoritative one.
* Not run here: GitHub CI on x86; a loong64 cross-architecture rerun of the driver;
  `n_y = 64`; page-level verification of the textbook criterion (offline).
* The platform-side items of the audit (`tensor_benchmark_cases` anchor row, F1;
  `oss_solutions` selection records, F6) are **not** closed by this round: they are not
  repository artefacts and were not touched.

## 5. Reproduction

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python benchmarks/verified/fem_bending_patch/run.py
OMP_NUM_THREADS=2 PYTHONPATH=src python -m pytest tests/test_bending_patch.py -q
OMP_NUM_THREADS=2 PYTHONPATH=src python tools/verified_anchor_audit.py
```
