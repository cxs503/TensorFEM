# Public API and release audit

TensorFEM 0.19 keeps every existing top-level export while establishing an auditable path to a stable v1 API. The reviewed snapshot is `api/public-api-v1.json`; an identical copy is packaged for runtime inspection.

## Stability classes

- `stable`: covered by semantic-version compatibility.
- `beta`: supported, but may change after a documented deprecation period.
- `experimental`: evaluable functionality whose interface may change in release notes.
- `internal_candidate`: benchmark, qualification, demo, or post-processing helpers that remain importable for compatibility but should be imported from their defining submodule.

The initial inventory contains 310 names: 47 stable, 44 beta, 175 experimental, and 44 internal candidates. Classification is deliberately conservative; promotion requires tests, documentation, and a reviewed manifest update. Nothing was removed during this audit.

## Compatibility workflow

Any top-level addition, removal, reorder, module move, or classification change fails `scripts/audit_public_api.py`. Regenerate both manifest copies only as part of an intentional API review:

```bash
PYTHONPATH=src python scripts/generate_api_manifest.py --write api/public-api-v1.json
cp api/public-api-v1.json src/tensorfem/public_api_manifest.json
PYTHONPATH=src python scripts/audit_public_api.py
```

Before a supported symbol is removed or renamed, decorate it with `tensorfem.deprecation.deprecated`, document the replacement and target removal release, and retain it for at least one published minor release. Internal candidates should first move to submodule-first documentation, then receive the same warning cycle before leaving the top level.

## Release gate

The fast audit checks required metadata and API drift. The full gate builds both artifacts offline, verifies package data, installs the wheel into a temporary virtual environment using already installed dependencies, imports the package with warnings treated as errors, and executes the installed CLI:

```bash
PYTHONPATH=src python scripts/release_smoke.py
PYTHONPATH=src python scripts/release_smoke.py --build
```

The temporary environment uses `--system-site-packages` solely to reuse the current local PyTorch installation; installation itself uses `--no-index --no-deps`. This gate does not claim dependency isolation from PyTorch, only artifact and entry-point isolation from the source tree.

Current capability labels printed by the CLI predate this manifest and are not the compatibility contract. The JSON manifest is authoritative until those labels are reconciled in a separately reviewed change.
