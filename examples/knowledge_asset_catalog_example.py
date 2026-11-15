#!/usr/bin/env python3
"""Example: Using the TensorFEM Knowledge Asset Catalog.

This script demonstrates how to:
1. Build the knowledge asset catalog
2. Query assets by type, lifecycle, and tags
3. Validate integrity
4. Export a versioned dataset for downstream use
"""
from __future__ import annotations

import json
from pathlib import Path

from tensorfem.knowledge_assets import (
    build_catalog,
    export_dataset,
    validate_asset,
    catalog_report,
)


def main():
    # 1. Build the catalog
    print("=" * 60)
    print("Building TensorFEM Knowledge Asset Catalog")
    print("=" * 60)
    catalog = build_catalog()
    print(f"Total assets: {catalog.summary['total']}")
    print(f"  Anchors: {catalog.summary.get('anchors_total', 0)}")
    print(f"  Certified: {catalog.summary.get('certified_anchors', 0)}")
    print(f"  Gates: {catalog.summary.get('gates_total', 0)}")
    print()

    # 2. Query by type
    print("Assets by type:")
    for t, count in sorted(catalog.summary.get("by_type", {}).items()):
        print(f"  {t}: {count}")
    print()

    # 3. Query by lifecycle
    print("Assets by lifecycle:")
    for lc in ("draft", "reviewed", "published", "deprecated"):
        count = len(catalog.by_lifecycle(lc))
        if count > 0:
            print(f"  {lc}: {count}")
    print()

    # 4. Query by tag
    print("Top tags:")
    tag_counts: dict[str, int] = {}
    for asset in catalog.assets:
        for tag in asset.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {tag}: {count}")
    print()

    # 5. Validate all assets
    print("Validating assets...")
    invalid = []
    for asset in catalog.assets:
        errors = validate_asset(asset)
        if errors:
            invalid.append((asset.asset_id, errors))
    if invalid:
        print(f"  {len(invalid)} assets with validation issues:")
        for aid, errs in invalid[:5]:
            print(f"    {aid}: {errs}")
    else:
        print("  All assets valid ✓")
    print()

    # 6. Export published dataset
    print("Exporting published dataset...")
    dataset = export_dataset(catalog, lifecycle="published")
    print(f"  Dataset assets: {dataset['asset_count']}")
    print(f"  Dataset integrity: {dataset['integrity_sha256'][:32]}...")
    print()

    # 7. Show sample asset detail
    published_anchors = [a for a in catalog.by_lifecycle("published") if a.asset_type == "anchor"]
    if published_anchors:
        sample = published_anchors[0]
        print(f"Sample asset: {sample.asset_id}")
        print(f"  Title: {sample.title}")
        print(f"  Gates: {sample.evidence.get('gates_total', 'N/A')}")
        print(f"  Tags: {', '.join(sample.tags)}")
        print(f"  Version chain: {len(sample.version_chain)} entries")
        print()

    # 8. Generate report
    print("Generating catalog report...")
    report = catalog_report(catalog)
    report_path = Path("docs/knowledge-asset-catalog-report.md")
    report_path.write_text(report)
    print(f"  Report written to: {report_path}")
    print()

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
