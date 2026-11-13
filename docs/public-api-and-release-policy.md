# Public API and release audit

TensorFEM 0.19 keeps every existing top-level export while establishing an auditable path to a stable v1 API. The reviewed snapshot is `api/public-api-v1.json`; an identical copy is packaged for runtime inspection (`src/tensorfem/public_api_manifest.json`, declared as package data in `pyproject.toml` and `MANIFEST.in`).

## Stability classes

- `stable`: covered by semantic-version compatibility.
- `beta`: supported, but may change after a documented deprecation period.
- `experimental`: evaluable functionality whose interface may change in release notes.
- `internal_candidate`: benchmark, qualification, demo, or post-processing helpers that remain importable for compatibility but should be imported from their defining submodule.

The initial inventory contained 310 names (47 stable, 44 beta, 175 experimental, 44 internal candidates). The snapshot is regenerated as the surface grows; at 0.45.0 it records 395 names (52 stable, 44 beta, 254 experimental, 45 internal candidates) — the authoritative counts are whatever `scripts/audit_public_api.py --json` prints from the checkout, never this paragraph. Classification is deliberately conservative; promotion requires tests, documentation, and a reviewed manifest update. Nothing was removed during the original audit.

## Machine-readable contract

`api/public-api-manifest.schema.json` is the JSON Schema (draft 2020-12) for the manifest: closed key sets at both levels, `stability` enum, and `^\d+\.\d+(\.\d+)?$` patterns for `package_version`/`introduced`. `scripts/audit_public_api.py` enforces the same rules directly, so the fast gate has no JSON-Schema dependency; when `jsonschema` happens to be installed it is used as a second opinion and reports `jsonschema=passed|failed|unavailable`. `tests/test_public_api_manifest_gate.py` keeps the schema document and the in-script rules from drifting apart.

## Compatibility workflow

The audit fails closed on three independent surfaces:

1. **Runtime drift** — any top-level addition, removal, reorder, module move, or classification change against `tensorfem.__all__`.
2. **Contract shape** — unknown or missing manifest keys, invalid `stability`/`introduced`/`module`/`name` values, a wrong `schema` id, or a stability class with no `policy` sentence.
3. **Packaged-copy parity** — a stale or missing `src/tensorfem/public_api_manifest.json`, i.e. a wheel that would advertise a different API than the one reviewed. This check lives in the audit (not only in pytest) because the release evidence path runs the audit.

Regenerate both manifest copies only as part of an intentional API review:

```bash
PYTHONPATH=src python scripts/generate_api_manifest.py --write api/public-api-v1.json
cp api/public-api-v1.json src/tensorfem/public_api_manifest.json
PYTHONPATH=src python scripts/audit_public_api.py
PYTHONPATH=src python scripts/m14835_public_api_evidence.py   # refresh the hash-gated evidence
```

Outside the repository tree (for example auditing a downloaded manifest) use `--manifest PATH --no-package-check`; the packaged-copy check is meaningless without the tree.

Before a supported symbol is removed or renamed, decorate it with `tensorfem.deprecation.deprecated`, document the replacement and target removal release, and retain it for at least one published minor release. Internal candidates should first move to submodule-first documentation, then receive the same warning cycle before leaving the top level.

## Release gate

The fast audit checks required metadata and API drift. The full gate builds both artifacts offline, verifies package data, installs the wheel into a temporary virtual environment using already installed dependencies, imports the package with warnings treated as errors, and executes the installed CLI:

```bash
PYTHONPATH=src python scripts/release_smoke.py
PYTHONPATH=src python scripts/release_smoke.py --build
```

The temporary environment uses `--system-site-packages` solely to reuse the current local PyTorch installation; installation itself uses `--no-index --no-deps`. This gate does not claim dependency isolation from PyTorch, only artifact and entry-point isolation from the source tree.

Current capability labels printed by the CLI predate this manifest and are not the compatibility contract. The JSON manifest is authoritative until those labels are reconciled in a separately reviewed change.