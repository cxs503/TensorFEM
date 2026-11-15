# TensorFEM Knowledge Asset Catalog

## Overview

This module implements the **knowledge asset operations** layer for TensorFEM,
providing structured management of verification evidence with version chain
tracking, lifecycle management, and integrity-checked dataset export.

## What is a Knowledge Asset?

A knowledge asset is a machine-readable representation of verified engineering
knowledge in TensorFEM. Each asset carries:

- **Unique identifier** (e.g., `anchor.fem_cantilever`)
- **Type**: `anchor`, `gate`, `benchmark`, or `capability`
- **Version chain**: history of lifecycle transitions (draft → reviewed → published → deprecated)
- **SHA-256 integrity**: cryptographic digest of canonical JSON
- **Provenance**: source module, git reference, directory
- **Evidence**: quantitative verification results (gates passed/failed, errors)
- **Tags**: domain classification (beam, plate, shell, marine, etc.)

## Quick Start

```python
from tensorfem.knowledge_assets import build_catalog, export_dataset, validate_asset

# Build the complete catalog from anchor certificates
catalog = build_catalog()

# Inspect summary
print(f"Total assets: {catalog.summary['total']}")
print(f"By type: {catalog.summary['by_type']}")
print(f"By lifecycle: {catalog.summary['by_lifecycle']}")

# Filter by lifecycle
published = catalog.by_lifecycle("published")
print(f"Published assets: {len(published)}")

# Filter by type
anchors = catalog.by_type("anchor")
print(f"Anchor assets: {len(anchors)}")

# Filter by tag
beam_assets = catalog.by_tag("beam")
print(f"Beam-related assets: {len(beam_assets)}")

# Validate all assets
for asset in catalog.assets:
    errors = validate_asset(asset)
    if errors:
        print(f"{asset.asset_id}: {errors}")

# Export published dataset for downstream use
dataset = export_dataset(catalog, lifecycle="published")
print(f"Dataset integrity: {dataset['integrity_sha256']}")
```

## Lifecycle State Machine

```
draft ──→ reviewed ──→ published ──→ deprecated
  │          │
  └──────────┴──→ deprecated
```

- **draft**: Initial state; evidence incomplete or missing gates/runner
- **reviewed**: Evidence present but not yet certified (uncertified grade)
- **published**: Certified anchor with all gates passing
- **deprecated**: Terminal state; asset superseded or withdrawn

## Asset Types

| Type | Description | Example |
|------|-------------|---------|
| `anchor` | Verification anchor with gates | `anchor.fem_cantilever` |
| `capability` | Solver capability summary | `capability.linear_static` |
| `gate` | Individual verification gate | (embedded in anchor evidence) |
| `benchmark` | Benchmark result | (embedded in anchor evidence) |

## Integrity Verification

All assets carry SHA-256 integrity digests computed from canonical JSON
(sorted keys, no whitespace, no NaN). To verify:

```python
from tensorfem.knowledge_assets import build_catalog, validate_asset

catalog = build_catalog()
for asset in catalog.assets:
    errors = validate_asset(asset)
    assert errors == [], f"{asset.asset_id} failed: {errors}"
```

## Dataset Export

The `export_dataset()` function produces a versioned, integrity-checked
export suitable for:

- Training data for ML models
- Cross-solver comparison
- Meta-analysis of verification results
- Downstream consumption by other tools

```python
dataset = export_dataset(
    catalog,
    lifecycle="published",
    asset_types=("anchor", "capability"),
)
```

The export includes:
- Schema version
- Export timestamp
- Filtered assets
- Catalog integrity reference
- Dataset integrity digest

## Files

| File | Purpose |
|------|---------|
| `src/tensorfem/knowledge_assets.py` | Core module |
| `tests/test_knowledge_assets.py` | Test suite (44 tests) |
| `docs/knowledge-asset-catalog-report.md` | Generated catalog report |
| `artifacts/knowledge_asset_catalog.json` | Full catalog JSON |
| `artifacts/knowledge_asset_dataset_published.json` | Published dataset export |

## Schema

### Knowledge Asset (v1)

```json
{
  "schema": "tensorfem.knowledge-asset/1.0",
  "asset_id": "anchor.fem_cantilever",
  "asset_type": "anchor",
  "title": "Anchor: fem_cantilever",
  "version": "v1.0",
  "lifecycle": "published",
  "version_chain": [
    {"version": "v1.0", "lifecycle": "draft", "sha256": ""},
    {"version": "v1.0", "lifecycle": "reviewed", "sha256": ""},
    {"version": "v1.0", "lifecycle": "published", "sha256": "abc123..."}
  ],
  "provenance": {
    "source": "benchmarks/verified/fem_cantilever",
    "directory": "benchmarks/verified/fem_cantilever"
  },
  "evidence": {
    "gates_total": 10,
    "gates_passed": 10,
    "gates_failed": 0,
    "grade": "certified",
    "certified": true
  },
  "tags": ["beam", "certified"],
  "integrity_sha256": "def456..."
}
```

## Relation to FEAT-TK-ASSET-OPS

This module implements the TensorFEM-side contribution to the platform-level
**FEAT-TK-ASSET-OPS** (知识资产运营三件套):

1. **管理 (Management)**: Version chain tracking, lifecycle state machine,
   integrity verification
2. **前端呈现 (Frontend Presentation)**: Structured catalog with summary
   statistics, tag-based filtering, Markdown report generation
3. **训练调用 (Training Invocation)**: Versioned dataset export with
   SHA-256 integrity for downstream ML/training use
