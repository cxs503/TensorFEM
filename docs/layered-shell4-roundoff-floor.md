# Layered Shell4 round-off floor: retiring a machine-dependent assertion

Status: **closed verification debt**.  The finite-rotation layered Shell4
stress-free regression used to assert `max|stress| < 1e-10` and
`max|force| < 1e-10` and, because that constant was not platform independent,
the whole test carried `@pytest.mark.xfail(reason="machine-dependent ...",
strict=False)` (issue #8).  An `xfail` that depends on the CPU is not evidence:
on this host the test *xpasseheld* (`14 passed, 1 xpassed`); on a different
BLAS build the same code produced `1.5e-10` and the assertion failed.  The
mark is now retired and the assertion is written against an objective
criterion.

## 1. Why an absolute constant cannot work

The assembly is a sum of `O(ops)` floating-point terms of magnitude `scale`,
so a "zero" quantity can only be resolved down to

```
floor ≈ ops · eps · scale ,      eps = 2.22e-16 for float64
```

The tested model has `E = 2e5`, so `eps·E = 4.4e-11`: the retired `1e-10`
threshold was only ~2× above the noise band, i.e. *inside* it on any build
whose operation-count amplification exceeded 2.3.  Measured amplification on
this host already reaches 3.63 (§3), so the historical failure was expected
behaviour of an under-specified assertion, not a formulation bug.

## 2. Replacement criterion

`src/tensorfem/roundoff_scale.py` provides

```
roundoff_bound(scale, dtype=float64, ops=1024) == 1024 · eps(dtype) · scale
roundoff_ratio(value, scale, ...)               == |value| / bound
```

`scale` must be the magnitude of the largest term entering the quantity —
`E` for a stress, `E·t` for a force — never the value under test.  The
operation budget `ops = 1024` keeps a ≥280× margin over the largest measured
amplification (3.63) while remaining ≥9 orders below any physically meaningful
inconsistency (a 1 % strain signal), so the check still fails loudly on a real
formulation error.  The bound is exported through the module, not through
`tensorfem.__all__`, so the frozen public-API snapshot is untouched.

`tests/test_finite_rotation_layered_shell4.py` now asserts

```
max|force|  <= roundoff_bound(E·t)      # 3.64e-09 N
max|stress| <= roundoff_bound(E)        # 4.55e-08 Pa
```

and the `xfail` mark is gone; `tests/test_roundoff_scale.py` pins the criterion
itself (linearity in `scale`/`ops`, dtype tracking, input validation, ratio
semantics) and audits the case.

## 3. Measurement (`scripts/audit_roundoff_floor.py`)

Same warped patch (`warp = +0.012 / −0.007 / +0.004 m`), zero displacement,
coordinate origin translated by `0 … 1e8 m`.  A consistent strain measure is
translation invariant, so any drift of the stress-free floor across origins is
round-off.

| origin offset [m] | max abs stress [Pa] | floor/(eps·E) | floor/bound | signal/bound |
| --- | --- | --- | --- | --- |
| 0 | 4.149e-11 | 0.934 | 0.0009 | 6.054e+09 |
| 1e2 | 1.612e-10 | 3.629 | 0.0035 | 6.054e+09 |
| 1e4 | 1.609e-10 | 3.622 | 0.0035 | 6.054e+09 |
| 1e6 | 8.053e-11 | 1.813 | 0.0018 | 6.054e+09 |
| 1e8 | 1.465e-10 | 3.298 | 0.0032 | 6.054e+09 |

Programmatic source: `artifacts/layered-shell4-roundoff-audit.json`, figure
`docs/assets/roundoff_floor/layered_shell4_roundoff_floor.png`.

Observations:

* the floor stays between 0.9 and 3.6 times `eps·E` over eight decades of
  origin offset — flat, i.e. the formulation is translation robust and the
  residual is pure round-off;
* the objective bound is used at ≤0.4 % of its capacity on this host, so the
  same assertion holds on builds up to 280× noisier;
* the two floors above `1e-10` (origins `1e2`, `1e4`, `1e8`) are exactly the
  historical failure mode, documented by
  `test_retired_absolute_threshold_was_inside_the_noise_band`;
* a genuine 1 % consistency error would read ~6e9 times the bound, so the
  criterion keeps full discriminating power.

## 4. Verification

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
    tests/test_roundoff_scale.py tests/test_finite_rotation_layered_shell4.py -q
```

The former `xfail` case is now a plain pass.  Nothing else in
`tests/test_finite_rotation_layered_shell4.py` changed: the rigid-rotation
clauses keep their empirical `4e-7` thresholds, which sit four orders above the
floor measured here and were never machine dependent.

## 5. Limitations

* The criterion bounds round-off for a *single* assembled quantity of known
  term scale.  It is not a convergence or accuracy statement, and it must not
  be reused to loosen a physical tolerance.
* `ops = 1024` is calibrated on measured amplifications ≤ 3.63 for this
  element (24 DOF, 3 layers, single patch).  Elements whose assembly sums many
  more terms (thick layered shells, long sparse accumulation chains) may need a
  larger budget; the helper takes `ops` as an argument for that reason.
* Only float64 and float32 are exercised; a mixed-precision path would need its
  own eps and a re-measured amplification.
* Other absolute "zero" thresholds elsewhere in the suite were **not** migrated
  by this change; ~300 absolute thresholds remain, most of them far above
  their noise band.  A repo-wide migration is out of scope here and would need
  a per-model term-scale audit.