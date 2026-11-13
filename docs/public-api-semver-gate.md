# Release-to-release compatibility gate (`scripts/public_api_compat.py`)

`api/public-api-v1.json` already carries the promise in its own `policy` block:

| stability | policy sentence stored in the manifest |
|---|---|
| `stable` | semver compatibility |
| `beta` | deprecate before removal |
| `experimental` | may change with release notes |
| `internal_candidate` | kept for compatibility; move to submodules after deprecation |

Until this change nothing enforced it. `scripts/audit_public_api.py` and
`tests/test_public_api_contract.py` are **point-in-time** checks: they compare the
snapshot against the tree it ships in. Since the manifest and the code are edited
together, they are structurally unable to notice that a name *used to exist* and no
longer does — a stable symbol can be deleted, renamed, moved to another module or
demoted to `experimental` and the snapshot stays "exactly consistent".

## What the gate adds

`scripts/public_api_compat.py` takes two reviewed snapshots (a path or
`git:<rev>:api/public-api-v1.json`) and turns every difference into the **minimum
version bump the policy demands**:

| change | `stable` | `beta` | `experimental` | `internal_candidate` |
|---|---|---|---|---|
| added | minor | minor | minor | minor |
| removed | **major** | **major** | minor | minor |
| object renamed (import name changes) | **major** | **major** | minor | minor |
| module moved (`entry.module`) | **major** | **major** | minor | minor |
| stability demoted | **major** | minor (that is the deprecation path) | minor | minor |
| stability promoted | patch | patch | patch | patch |
| `__all__` reordered | patch | patch | patch | patch |

`--mode pr` (default) charges only the **breaking** rows, because the version is
bumped at release, not on every pull request: a purely additive PR passes while
being reported as `accumulated_minor_delta`. `--mode release` charges the whole
delta, so a release cannot ship new API surface under an unchanged number.
`--zero-major-minor-is-breaking` opts into the SemVer 0.x convention where the
minor position carries breaking changes.

```bash
# pull-request / push gate
python scripts/public_api_compat.py --base "git:$BASE_SHA:api/public-api-v1.json"
# release gate: everything since the previous reviewed snapshot must be billed
python scripts/public_api_compat.py --mode release --base git:<rev>:api/public-api-v1.json
```

Exit codes: `0` compatible, `1` violation (each one printed as
`VIOLATION: <kind> <name> needs <bump>`), `2` malformed input (duplicate names,
unknown stability class, bad version string) — a manifest this gate cannot parse is
a failure, not a pass.

`.github/workflows/ci.yml` runs the `pr` mode against the base commit of every
pull request and push. When the event carries no usable baseline (new branch, force
push, shallow tip) the job says so instead of failing on an artefact of the payload.

## Measured behaviour on this repository

Reproduce with `PYTHONPATH=src python scripts/m18005_compat_baseline_compare.py`;
the numbers below come from `artifacts/m18005_public_api_compat_baseline.json`.

**Detection contrast** — two breaks injected into a scratch copy of the tree, with
module *and* manifest edited together exactly as a real release edits them:

| injected fault | snapshot audit (`audit_public_api.py`) | this gate |
|---|---|---|
| `TrussModel` (stable) deleted from `__init__` + both manifests | PASS (not detected) | FAIL: `removed TrussModel needs major` |
| `solve_linear_static` demoted `stable -> experimental` | PASS (not detected) | FAIL: `stability_changed needs major` |
| **detection rate** | **0 / 2** | **2 / 2** |

**Real history** — every consecutive pair of commits that touched the manifest:

| base | head | entries | changes | `pr` mode | `release` mode |
|---|---|---|---|---|---|
| fe26e5fe | a86365e7 | 390 → 395 | added 5 | none / PASS | minor / FAIL |
| bae35ec1 | fe26e5fe | 370 → 390 | added 20 | none / PASS | minor / FAIL |
| aa59084f | bae35ec1 | 365 → 370 | added 5 | none / PASS | minor / FAIL |
| 1dabcbe5 | aa59084f | 353 → 365 | added 12 | none / PASS | minor / FAIL |
| 06f76270 | 1dabcbe5 | 346 → 353 | added 7 | none / PASS | minor / FAIL |
| d4cf378e | 06f76270 | 341 → 346 | added 5 | none / PASS | minor / FAIL |

Two facts worth keeping: the shipped top-level API has **only ever grown** (no
removal, rename, module move or demotion in the recorded history), and all of that
growth happened under the single declared version `0.45.0` — which is exactly what
the `release` mode flags: should `0.45.0` be released from this state, the manifest
must be re-baselined at `0.46.0` first.

## Limits

* Surface only. It cannot see a behaviour change inside a stable function; that is
  what benchmark/verification anchors are for.
* Deprecation *scheduling* is not machine-checked: `tensorfem.deprecation` records
  `since`/`remove` at runtime, but the manifest has no per-entry lifecycle field, so
  the gate cannot verify that a `beta` removal was announced a minor release earlier.
  Closing that gap needs a manifest schema change, deliberately kept out of this
  change so it does not collide with the open manifest-shape work.
* `git:<rev>:<path>` needs an unshallowed clone; CI checks out with
  `fetch-depth: 0`.