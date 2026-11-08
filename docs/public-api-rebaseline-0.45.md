# Public API manifest re-baseline (0.45.0) — 12-symbol drift closure

Status: **CLOSED** — `scripts/audit_public_api.py` PASS (365 symbols);
`tests/test_public_api_contract.py` 4/4 green.

## 1. Debt origin

Report #7718 (TET10 integration audit, m6462) flagged on HEAD:

> `scripts/audit_public_api.py` 在 HEAD 上即报 `__all__` 漂移 365 vs 期望 353，
> 属既有欠账，re-baseline 需单独走 API 复核，本单不动。

Root cause: two merged missions exported new top-level symbols without updating
the reviewed snapshot `api/public-api-v1.json` (and its packaged copy
`src/tensorfem/public_api_manifest.json`):

| commit | mission | module | symbols |
|---|---|---|---|
| `e520521` (Merge #273, m4951) | J2 minimum verifiable surface | `tensorfem.j2_elastoplasticity` | `J2Material`, `J2Return`, `radial_return`, `return_map_residuals`, `uniaxial_stress_path`, `pure_shear_yield_point`, `verify_minimum_surface` |
| `46c58b9` (m5257) | structural verification ladder | `tensorfem.structural_verification` | `solve_cantilever_udl`, `kirchhoff_ss_plate_uniform_load`, `cantilever_modal_ladder`, `euler_bernoulli_cantilever_tip`, `PLATE_ALPHA_TIMOSHENKO` |

All 12 are legitimate, documented, tested exports (dedicated test files
`tests/test_j2_minimum_surface.py`, `tests/test_structural_verification.py`,
`tests/test_structural_verification_face.py`); none is an accidental leak.
`verify_minimum_surface` is the J2 *minimum verifiable surface* driver, not a
minimal-surface solver (name kept for API stability).

## 2. Method (surgical splice, not full regeneration)

A full `generate_api_manifest.py` regeneration was **not** safe: the repository
history is squashed (the "add" commit of every module now resolves to the
2026-10-07 squash `d4cf378`, version 0.45.0), so the generator's
`introduced()` re-derives `0.45.0` for **all** 353 pre-existing entries and
would destroy the curated historical versions (0.1.0 … 0.44.0).

Instead:

1. Diffed `tensorfem.__all__` (365) against the manifest (353): 12 additions,
   0 removals, order of the 353 unchanged.
2. Spliced the 12 entries at their exact `__all__` positions (5 after
   `solve_linear_buckling`, 7 appended after `read_result_db_compatible`),
   preserving every pre-existing entry byte-for-byte (verified: 0 of 353
   modified).
3. New entries follow the generator's own policy: `introduced: "0.45.0"`
   (both source commits are 0.45.0-era, verified via
   `git show <c>:pyproject.toml`), `stability: "experimental"` — consistent
   with the 27 pre-existing 0.45.0 entries.
4. Wrote both copies identically (policy: "Regenerate both manifest copies").

Cross-check: after the generator fix below, a fresh regeneration reproduces
the spliced manifest **exactly** (365 entries, `==` True).

## 3. Generator defect fixed (data-loss guard)

`scripts/generate_api_manifest.py::build()` now loads the existing manifest
and preserves the reviewed `introduced`/`stability` of names already present
(`curated()`), so the documented workflow

```bash
PYTHONPATH=src python scripts/generate_api_manifest.py --write api/public-api-v1.json
cp api/public-api-v1.json src/tensorfem/public_api_manifest.json
PYTHONPATH=src python scripts/audit_public_api.py
```

can no longer clobber curated metadata on squashed history. Fresh-manifest
behavior is unchanged. Regression test:
`tests/test_public_api_contract.py::test_generator_preserves_reviewed_metadata_on_regeneration`.

## 4. Verification

- `python3 scripts/audit_public_api.py` → `API audit: PASS; 365 symbols;
  {'stable': 47, 'experimental': 229, 'internal_candidate': 45, 'beta': 44}`
  (before: FAIL, 353 vs 365).
- `pytest tests/test_public_api_contract.py` → 4 passed (before: 1 failed).
- `pytest tests/test_j2_minimum_surface.py tests/test_structural_verification.py`
  → 16 passed, 3 failed — the 3 failures
  (`test_committed_verified_artifacts_are_consistent[fem_beam|fem_plate|fem_modal]`)
  are **pre-existing on HEAD** (reproduced with the change stashed); they
  concern committed verified artifacts filtered by the gate in m5257 and are
  out of scope here (separate debt).

## 5. Notes / limits

- `introduced` for the 12 new symbols is the version at the exporting commit,
  which equals the first release carrying them (0.45.0); no earlier release
  contained them.
- Stability promotion (experimental → beta/stable) still requires the
  documented review (tests + docs + reviewed manifest update); this change
  only records the current reviewed state.
- The 3 pre-existing artifact-consistency failures above remain open debt.
