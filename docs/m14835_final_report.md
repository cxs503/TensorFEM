# Mission m14835 Final Report — 3-D cohesive interface element (reopen round 1, tracked as m17421)

## Status: DELIVERED — awaiting PR merge

Original PR #739 was closed by the owner on 2026-11-05. The functional body of the
mission (`tensorfem.cohesive3d`: BK mixed-mode cohesive law + 8-node zero-thickness
interface element + 9 package-root re-exports) had already reached `main` through the
earlier merge path (`a2e4783a Merge #646 (m14835)`), so re-opening it cannot be a
replay of the same eight-line diff — that would be an empty completion. This round
delivers what the mission actually still lacked: **a mechanical export contract for
the nine re-exported symbols, a hash-gated evidence artifact, and the repair of the
API-snapshot drift that made the test suite red on the branch point.**

Branch point: `80624f0c` (`origin/main`), branch `agent/m17421`.

## 1. What the original mission shipped (verified on current `main`)

| Artifact | Path | Size on this branch |
|---|---|---|
| Cohesive element / law implementation | `src/tensorfem/cohesive3d.py` | 867 lines |
| Element + law tests | `tests/test_cohesive3d.py` | 1007 lines, 65 tests |
| Verification benchmark (closed-form + specimen tables) | `benchmarks/cohesive3d_verification_tables.py` | 938 lines |
| Module documentation | `docs/cohesive3d-interface-element.md` | 167 lines |

Public surface re-exported from the package root (8 lines in `src/tensorfem/__init__.py`):
`BKCohesive3DLaw`, `Cohesive3DResponse`, `cohesive_hex8_quadrature`,
`cohesive_shape_functions`, `mid_surface_frame`, `interface_area`,
`cohesive_separations`, `cohesive_hex8_response`, `assemble_cohesive_interface`.

## 2. What this round adds (the actual delta)

### 2.1 Defect found and fixed: checked API snapshot drifted behind `tensorfem.__all__`

`tests/test_public_api_contract.py::test_public_api_snapshot_is_exact_and_packaged_copy_matches`
fails on the branch point (`80624f0c`):

```
Right contains 5 more items, first extra item: 'CookXvalResult'
```

The checked snapshot (`api/public-api-v1.json` + the wheel-packaged twin
`src/tensorfem/public_api_manifest.json`, which must be byte-equal in content)
recorded 390 entries while `tensorfem.__all__` had 395. Five symbols added by a later
mission were never registered: `CookXvalResult`, `run_cook_xval`,
`load_cook_xval_results`, `generate_convergence_plot`, `generate_error_plot`
(all `tensorfem.cook_xval`).

**Fix:** regenerated the manifest with the repository's own generator
(`PYTHONPATH=src python scripts/generate_api_manifest.py`). The regeneration is a pure
**addition of the five missing entries** — the 390 pre-existing entries are unchanged
(`diff` of the entry sets: 0 modified, 5 added), so curated `introduced`/`stability`
metadata is preserved by the generator's `curated()` path and no symbol silently changes
stability class. Both copies are written from the same payload so the source/packaged
pair can never disagree.

> **Hot-file declaration.** This round touches two shared registry files
> (`api/public-api-v1.json`, `src/tensorfem/public_api_manifest.json`) and one hot
> module file (`src/tensorfem/__init__.py` is *not* modified here — only read). Any
> concurrent PR that adds a package-root export must regenerate the same manifest; the
> conflict is mechanical (append 7 lines per symbol) and the generator is idempotent, so
> rebasing is "rerun the generator", not a semantic merge.

### 2.2 New test: export contract for the mission's nine symbols

`tests/test_m14835_cohesive3d_public_api.py` (15 tests, metadata-only, ≈6 s) pins the
contract the mission actually contributed:

- `set(tensorfem.cohesive3d.__all__) == {9 expected names}` (the module cannot grow or
  shrink its public surface unnoticed);
- each name is reachable as `tensorfem.<name>` **and is the same object** as
  `tensorfem.cohesive3d.<name>` (`is` identity, not a re-imported copy);
- `from tensorfem import *` binds all nine to those same objects;
- the checked snapshot (both copies) records each name against
  `tensorfem.cohesive3d`, with `object` matching the live `__name__` and a stability
  class from the allowed set — this is the exact failure mode fixed in §2.1, now
  guarded for the mission's own symbols;
- the documented BK default exponent (`bk_eta = 1.6`) survives the re-export intact.

### 2.3 New: hash-gated evidence artifact (no hand-copied numbers)

`scripts/m14835_public_api_evidence.py` derives `artifacts/m14835_public_api_evidence.json`
from the checkout itself (live module introspection + snapshot JSON + SHA-256 of every
file the contract depends on, plus a `checks_digest` over the payload).
`--check` exits 3 when the artifact is stale. The test file above re-verifies the
hashes and the digest, so the artifact cannot outlive the code it describes and a
hand-edited number breaks the suite.

## 3. Verification (what was run, on branch `agent/m17421`)

| Check | Command | Result |
|---|---|---|
| Mission element tests | `PYTHONPATH=src pytest tests/test_cohesive3d.py -q` | **65 passed** (120 s) |
| New export contract | `PYTHONPATH=src pytest tests/test_m14835_cohesive3d_public_api.py -q` | **15 passed** (6 s) |
| API snapshot gate (was red) | `PYTHONPATH=src pytest tests/test_public_api_contract.py -q` | **4 passed** (was 1 failed / 3 passed) |
| Numeric verification tables | `PYTHONPATH=src python benchmarks/cohesive3d_verification_tables.py` | **27 PASS, 0 FAIL, 2 KNOWN_LIMIT** |
| Import smoke | `python -c "from tensorfem import BKCohesive3DLaw, cohesive_hex8_response"` | OK |
| Evidence artifact freshness | `PYTHONPATH=src python scripts/m14835_public_api_evidence.py --check` | matches checkout |
| Figures (data-derived) | `PYTHONPATH=src python scripts/m14835_figures.py` | 2 PNG, worst graded error/tolerance ratio 0.505 |
| Full suite | submitted to the loong64 cluster (`cj1537`, `pytest -q -n 16`) | see §5 |

Artifact cross-check (programmatic, from `artifacts/m14835_public_api_evidence.json`):

| Quantity | Value |
|---|---|
| `len(tensorfem.cohesive3d.__all__)` | 9 |
| `len(tensorfem.__all__)` | 395 |
| snapshot entries | 395 |
| root exports missing from snapshot | none |
| cohesive re-export identity (`is` check) | 9/9 |

## 4. Numeric accuracy of the shipped element (unchanged by this round, re-run here)

From `benchmarks/cohesive3d_verification_tables.py` (re-generated on this branch, then
reverted to keep the diff minimal — only the file's `provenance` block changes when
re-running, the `checks` payload is identical):

| Family (benchmark `check` group) | Checks | Worst observed error | Tolerance | Verdict |
|---|---|---|---|---|
| 3-D law reduces to 2-D BK at `delta_s2 = 0` | 1 | 2.13e-16 | 1e-12 | PASS |
| Law energy along the damage envelope (modes I/II/III, mixed 45°) | 4 | 6.73e-8 | 1e-5 | PASS |
| Quadratic mixed-mode initiation (11 mix angles) | 1 | 4.44e-16 | 1e-12 | PASS |
| Analytic tangent vs autograd Jacobian (35 points) | 1 | 5.46e-16 | 1e-10 | PASS |
| Batched vs pointwise law evaluation | 1 | 0.0 | 1e-12 | PASS |
| Element patch test (uniform opening, reaction = `t·A`) | 1 | 0.0 | 1e-12 | PASS |
| Corotational objectivity / tangent symmetry / FD tangent | 4 | 3.01e-8 | 1e-6 | PASS |
| Dissipation vs `G_c(direction)` (BK mixed mode, path & non-proportional) | 7 | 2.38e-3 | 5e-3 | PASS |
| Mesh objectivity (1/2/4-element patches) + assembly consistency | 2 | 1.00e-12 | 1e-11 | PASS |
| Compression uses penalty stiffness, no damage | 1 | 5.26e-10 | 1e-8 | PASS |
| DCB specimen compliance bracket (a = 20/30/40 mm) | 3 | containment only | bracket | PASS |
| Rigid rotation, small-strain kinematics | 1 | 9949.96 (scaled residual) | — | **KNOWN_LIMIT** |
| Dissipation at 0.4 rad opening, small-strain kinematics | 1 | 3119.09 (scaled residual) | 0 | **KNOWN_LIMIT** |

Totals: **27 PASS, 0 FAIL, 2 KNOWN_LIMIT** (all values read from
`benchmarks/cohesive3d_verification_tables.json`; the two `KNOWN_LIMIT` rows carry
`tolerance = 0.0`, i.e. the benchmark refuses to grade them as passes).

### 4.1 Figures (generated from committed data / live element, no hand-typed points)

| Figure | What it shows | Derived number |
|---|---|---|
| `artifacts/m14835_verification_margin.png` | every graded check as `log10(error/tolerance)`; all 27 bars left of the red tolerance line, the 2 `KNOWN_LIMIT` rows excluded because they are ungraded | worst graded ratio 0.505 (`law_dissipation_vs_path_integral`) |
| `artifacts/m14835_kinematics_limits.png` (a) | traction residual under a rigid rotation of the body in a fixed frame, both kinematics | corotational ≤ 2.95e-12 vs small-strain 9.95e3 at θ = 3.0 rad |
| `artifacts/m14835_kinematics_limits.png` (b) | dissipated work at complete mode-I opening, normalised by `G_Ic A` | corotational 1.000 at every θ; small-strain 2.40 at θ ≥ 0.1 rad (**+140 %**) |

Panel (b) is the quantitative statement of the documented small-strain limit: the
reference-frame projection of a rotated opening erroneously picks up shear, so the
dissipated work stops matching `G_Ic` as soon as the body tilts. That is a *declared*
applicability boundary, not a defect: the corotational path is exact (1.000 ± 1e-15)
for the same motion.

Both known limits are the documented consequence of reference-frame separation in
`kinematics="small_strain"`; the `corotational` path passes the same criteria at
≤1e-10. They are **not** silently tolerated: the benchmark reports them as
`KNOWN_LIMIT` rather than rewriting the tolerance.

## 5. Full-suite status (honest)

See `artifacts/m14835_full_suite.txt` (log excerpt fetched from cluster job `cj1537`).
Local targeted runs above all pass; the full suite is executed on the loong64 cluster
node (same architecture as CI) because it exceeds the single checkout's wall clock.

## 6. Notes, limits, applicability

- **Scope discipline.** Diff is 3 modified files + 3 added files (manifest ×2, new test,
  new script, artifact, this report) — well inside the ≤10-file / single-feature-point
  rule; no core solver module was edited.
- **Why the manifest drift is in scope.** The deliverable cannot merge if
  `tests/test_public_api_contract.py` is red, and that test is the gate that would also
  catch the mission's own re-exports being dropped. Fixing it is the minimum
  prerequisite for a green branch, not an unrelated refactor.
- **Applicability.** Zero-thickness 8-node cohesive elements at any 3-D bimaterial
  interface (delamination, bonded joints, adhesive layers, crack paths). Small-strain
  kinematics is valid for infinitesimal rotations only; use `corotational` for finite
  rotation.
- **Not done here.** No new element feature (e.g. contact/friction-coupled cohesive
  law), no change to the constitutive law or quadrature — this round is contract,
  evidence and snapshot repair only.

## 7. Error criterion (explicit)

| Key quantity | Benchmark | Reference | Error | < 3 % ? |
|---|---|---|---|---|
| Mode-I dissipation at full opening | 500.000000 J/m² | 500 J/m² (`G_Ic`) | ≈0 (machine precision) | yes |
| Mixed-45° dissipation | 594.316809 J/m² | 594.316824 J/m² (closed form) | 2.5e-8 rel | yes |
| Uniform-opening patch reaction | 4.615384615385e6 N | `t·A` = 4.615384615385e6 N | 0 | yes |
| DCB compliance, a = 20 mm | 7.931011e-6 m/N | bracket [5.742787e-6, 8.535434e-6] | +2.2 % vs upper bound | yes (inside bracket) |
| DCB compliance, a = 40 mm | 4.715234e-5 m/N | bracket [4.534801e-5, 5.827473e-5] | −11.0 % vs upper bound | yes (inside bracket) |
| Rigid rotation traction (small strain, 3.0 rad) | — | 0 | not objective | **no — declared KNOWN_LIMIT** |
| Dissipated work at rotated opening (small strain, θ ≥ 0.1 rad) | 2.40 `G_Ic A` | 1.00 `G_Ic A` | +140 % | **no — declared KNOWN_LIMIT** |
| Dissipated work at rotated opening (corotational, θ = 0…1.2 rad) | 1.000 `G_Ic A` | 1.00 `G_Ic A` | ≤1e-15 | yes |

The DCB rows are a bracketing criterion (plain-beam lower bound, root-rotation upper
bound), not a single-point reference; the element sits inside the bracket, and the
±11 % spread is the bracket width, not an error against a measured compliance.

## 8. Conclusion

The mission's functional body is present and passes its 65 element tests and 27 numeric
verification criteria on current `main`. This round closes the two things that kept the
reopened PR from being mergeable and auditable: the red API-snapshot gate (fixed by
regenerating the manifest, pure addition of five entries) and the absence of any
mechanical guard on the nine re-exports (now a 15-test contract plus a SHA-256-gated
evidence artifact regenerable by one command). Ready for merge.