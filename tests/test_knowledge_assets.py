"""Tests for tensorfem.knowledge_assets module."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from tensorfem.knowledge_assets import (
    SCHEMA,
    DATASET_SCHEMA,
    LIFECYCLE_STATES,
    LIFECYCLE_TRANSITIONS,
    KnowledgeAsset,
    VersionEntry,
    AssetCatalog,
    build_catalog,
    export_dataset,
    catalog_report,
    validate_asset,
    canonical_json,
    compute_sha256,
    _infer_lifecycle,
    _infer_tags,
)


# --------------------------------------------------------------------------- #
# Helper tests
# --------------------------------------------------------------------------- #

class TestCanonicalJson:
    def test_deterministic(self):
        obj = {"b": 2, "a": 1, "c": [3, 4]}
        h1 = canonical_json(obj)
        h2 = canonical_json(obj)
        assert h1 == h2

    def test_sorted_keys(self):
        obj = {"z": 1, "a": 2}
        result = canonical_json(obj)
        assert result.index(b'"a"') < result.index(b'"z"')

    def test_no_nan(self):
        with pytest.raises(ValueError):
            canonical_json({"x": float("nan")})


class TestComputeSha256:
    def test_consistent(self):
        obj = {"key": "value"}
        h1 = compute_sha256(obj)
        h2 = compute_sha256(obj)
        assert h1 == h2
        assert len(h1) == 64  # hex digest length

    def test_different_objects_different_hash(self):
        h1 = compute_sha256({"a": 1})
        h2 = compute_sha256({"a": 2})
        assert h1 != h2


# --------------------------------------------------------------------------- #
# Lifecycle inference
# --------------------------------------------------------------------------- #

class TestInferLifecycle:
    def test_certified_is_published(self):
        assert _infer_lifecycle({"certified": True}) == "published"

    def test_no_gates_is_draft(self):
        assert _infer_lifecycle({"defects": ["E_NO_GATES"]}) == "draft"

    def test_no_runner_is_draft(self):
        assert _infer_lifecycle({"defects": ["E_NO_RUNNER"]}) == "draft"

    def test_uncertified_is_reviewed(self):
        assert _infer_lifecycle({"grade": "uncertified"}) == "reviewed"

    def test_default_draft(self):
        assert _infer_lifecycle({}) == "draft"


# --------------------------------------------------------------------------- #
# Tag inference
# --------------------------------------------------------------------------- #

class TestInferTags:
    def test_certified_tag(self):
        tags = _infer_tags({"grade": "certified"})
        assert "certified" in tags

    def test_beam_tag(self):
        tags = _infer_tags({"anchor": "fem_cantilever"})
        assert "beam" in tags

    def test_plate_tag(self):
        tags = _infer_tags({"anchor": "fem_plate_static"})
        assert "plate" in tags

    def test_shell_tag(self):
        tags = _infer_tags({"anchor": "fem_shell_scordelis_lo"})
        assert "shell" in tags

    def test_marine_tag(self):
        tags = _infer_tags({"anchor": "marine_panel_ultimate"})
        assert "marine" in tags

    def test_no_gates_tag(self):
        tags = _infer_tags({"defects": ["E_NO_GATES"]})
        assert "no_gates" in tags

    def test_contradicted_tag(self):
        tags = _infer_tags({"gates_contradicted": 3})
        assert "contradicted" in tags


# --------------------------------------------------------------------------- #
# KnowledgeAsset
# --------------------------------------------------------------------------- #

class TestKnowledgeAsset:
    def test_create_valid(self):
        asset = KnowledgeAsset(
            asset_id="test.anchor",
            asset_type="anchor",
            title="Test Anchor",
            version="v1.0",
            lifecycle="draft",
        )
        assert asset.asset_id == "test.anchor"
        assert asset.lifecycle == "draft"

    def test_compute_integrity(self):
        asset = KnowledgeAsset(
            asset_id="test.anchor",
            asset_type="anchor",
            title="Test Anchor",
            version="v1.0",
            lifecycle="draft",
        )
        sha = asset.compute_integrity()
        assert len(sha) == 64
        assert asset.integrity_sha256 == ""  # not set yet

    def test_integrity_stable_across_version_chain(self):
        """Integrity hash should NOT change when version_chain is modified."""
        asset = KnowledgeAsset(
            asset_id="test.anchor",
            asset_type="anchor",
            title="Test Anchor",
            version="v1.0",
            lifecycle="draft",
        )
        sha1 = asset.compute_integrity()
        # Add version chain entries
        asset.version_chain = [
            VersionEntry("v1.0", "draft", "hash1"),
            VersionEntry("v1.0", "reviewed", "hash2"),
        ]
        sha2 = asset.compute_integrity()
        assert sha1 == sha2, "integrity should be stable regardless of version_chain"

    def test_integrity_matches(self):
        asset = KnowledgeAsset(
            asset_id="test.anchor",
            asset_type="anchor",
            title="Test Anchor",
            version="v1.0",
            lifecycle="draft",
        )
        sha = asset.compute_integrity()
        asset.integrity_sha256 = sha
        errors = validate_asset(asset)
        assert errors == []

    def test_integrity_mismatch(self):
        asset = KnowledgeAsset(
            asset_id="test.anchor",
            asset_type="anchor",
            title="Test Anchor",
            version="v1.0",
            lifecycle="draft",
            integrity_sha256="wrong_hash",
        )
        errors = validate_asset(asset)
        assert any("integrity mismatch" in e for e in errors)

    def test_validate_missing_id(self):
        asset = KnowledgeAsset(
            asset_id="",
            asset_type="anchor",
            title="Test",
            version="v1.0",
            lifecycle="draft",
        )
        errors = asset.validate()
        assert any("asset_id" in e for e in errors)

    def test_validate_invalid_type(self):
        asset = KnowledgeAsset(
            asset_id="test",
            asset_type="invalid",
            title="Test",
            version="v1.0",
            lifecycle="draft",
        )
        errors = asset.validate()
        assert any("asset_type" in e for e in errors)

    def test_validate_invalid_lifecycle(self):
        asset = KnowledgeAsset(
            asset_id="test",
            asset_type="anchor",
            title="Test",
            version="v1.0",
            lifecycle="invalid",
        )
        errors = asset.validate()
        assert any("lifecycle" in e for e in errors)

    def test_canonical_dict_includes_version_chain(self):
        asset = KnowledgeAsset(
            asset_id="test.anchor",
            asset_type="anchor",
            title="Test Anchor",
            version="v1.0",
            lifecycle="draft",
            version_chain=[VersionEntry("v1.0", "draft", "abc")],
        )
        d = asset.canonical_dict()
        assert "version_chain" in d
        assert "integrity_sha256" not in d

    def test_content_dict_excludes_version_chain(self):
        asset = KnowledgeAsset(
            asset_id="test.anchor",
            asset_type="anchor",
            title="Test Anchor",
            version="v1.0",
            lifecycle="draft",
            version_chain=[VersionEntry("v1.0", "draft", "abc")],
        )
        d = asset._content_dict()
        assert "version_chain" not in d
        assert "integrity_sha256" not in d

    def test_version_chain_invalid_transition(self):
        asset = KnowledgeAsset(
            asset_id="test",
            asset_type="anchor",
            title="Test",
            version="v1.0",
            lifecycle="published",
            version_chain=[
                VersionEntry("v1.0", "published", ""),
                VersionEntry("v1.0", "draft", ""),
            ],
        )
        errors = asset.validate()
        assert any("invalid transition" in e for e in errors)


# --------------------------------------------------------------------------- #
# AssetCatalog
# --------------------------------------------------------------------------- #

class TestAssetCatalog:
    def test_empty_catalog(self):
        catalog = AssetCatalog()
        assert catalog.assets == []
        assert catalog.summary == {}

    def test_by_type(self):
        assets = [
            KnowledgeAsset("a1", "anchor", "A1", "v1.0", "draft"),
            KnowledgeAsset("c1", "capability", "C1", "v1.0", "published"),
        ]
        catalog = AssetCatalog(assets=assets)
        assert len(catalog.by_type("anchor")) == 1
        assert len(catalog.by_type("capability")) == 1

    def test_by_lifecycle(self):
        assets = [
            KnowledgeAsset("a1", "anchor", "A1", "v1.0", "draft"),
            KnowledgeAsset("a2", "anchor", "A2", "v1.0", "published"),
        ]
        catalog = AssetCatalog(assets=assets)
        assert len(catalog.by_lifecycle("draft")) == 1
        assert len(catalog.by_lifecycle("published")) == 1

    def test_by_tag(self):
        assets = [
            KnowledgeAsset("a1", "anchor", "A1", "v1.0", "draft", tags=["beam"]),
            KnowledgeAsset("a2", "anchor", "A2", "v1.0", "draft", tags=["plate"]),
        ]
        catalog = AssetCatalog(assets=assets)
        assert len(catalog.by_tag("beam")) == 1
        assert len(catalog.by_tag("plate")) == 1

    def test_compute_integrity(self):
        catalog = AssetCatalog(
            assets=[KnowledgeAsset("a1", "anchor", "A1", "v1.0", "draft")]
        )
        sha = catalog.compute_integrity()
        assert len(sha) == 64


# --------------------------------------------------------------------------- #
# Build catalog
# --------------------------------------------------------------------------- #

class TestBuildCatalog:
    def test_empty_certificates(self):
        catalog = build_catalog(cert_path=Path("/nonexistent/path.json"))
        assert catalog.summary["total"] == 0

    def test_with_real_certificates(self):
        repo_root = Path(__file__).resolve().parents[1]
        cert_path = repo_root / "benchmarks" / "verified" / "anchor_certificates.json"
        if cert_path.is_file():
            catalog = build_catalog(cert_path=cert_path)
            assert catalog.summary["total"] > 0
            assert "anchor" in catalog.summary.get("by_type", {})
            # Check integrity
            assert len(catalog.integrity_sha256) == 64

    def test_exclude_draft(self):
        repo_root = Path(__file__).resolve().parents[1]
        cert_path = repo_root / "benchmarks" / "verified" / "anchor_certificates.json"
        if cert_path.is_file():
            catalog_all = build_catalog(cert_path=cert_path, include_draft=True)
            catalog_no_draft = build_catalog(cert_path=cert_path, include_draft=False)
            assert len(catalog_no_draft.assets) <= len(catalog_all.assets)

    def test_all_assets_valid(self):
        """All assets built from real certificates should pass validation."""
        repo_root = Path(__file__).resolve().parents[1]
        cert_path = repo_root / "benchmarks" / "verified" / "anchor_certificates.json"
        if cert_path.is_file():
            catalog = build_catalog(cert_path=cert_path)
            invalid = []
            for asset in catalog.assets:
                errors = validate_asset(asset)
                if errors:
                    invalid.append((asset.asset_id, errors))
            assert invalid == [], f"Invalid assets: {invalid}"


# --------------------------------------------------------------------------- #
# Export dataset
# --------------------------------------------------------------------------- #

class TestExportDataset:
    def test_export_published(self):
        assets = [
            KnowledgeAsset("a1", "anchor", "A1", "v1.0", "published"),
            KnowledgeAsset("a2", "anchor", "A2", "v1.0", "draft"),
        ]
        catalog = AssetCatalog(assets=assets, version="v1.0")
        catalog.integrity_sha256 = catalog.compute_integrity()

        dataset = export_dataset(catalog, lifecycle="published")
        assert dataset["schema"] == DATASET_SCHEMA
        assert dataset["asset_count"] == 1
        assert "integrity_sha256" in dataset

    def test_export_filters_by_type(self):
        assets = [
            KnowledgeAsset("a1", "anchor", "A1", "v1.0", "published"),
            KnowledgeAsset("c1", "capability", "C1", "v1.0", "published"),
        ]
        catalog = AssetCatalog(assets=assets, version="v1.0")
        catalog.integrity_sha256 = catalog.compute_integrity()

        dataset = export_dataset(catalog, lifecycle="published", asset_types=("anchor",))
        assert dataset["asset_count"] == 1
        assert dataset["assets"][0]["asset_type"] == "anchor"


# --------------------------------------------------------------------------- #
# Catalog report
# --------------------------------------------------------------------------- #

class TestCatalogReport:
    def test_report_contains_summary(self):
        catalog = AssetCatalog(
            assets=[KnowledgeAsset("a1", "anchor", "A1", "v1.0", "published")],
            summary={"total": 1, "by_type": {"anchor": 1}, "by_lifecycle": {"published": 1}},
        )
        report = catalog_report(catalog)
        assert "TensorFEM Knowledge Asset Catalog Report" in report
        assert "**Total assets**: 1" in report

    def test_report_contains_integrity(self):
        catalog = AssetCatalog(
            assets=[KnowledgeAsset("a1", "anchor", "A1", "v1.0", "published")],
            summary={"total": 1},
        )
        catalog.integrity_sha256 = catalog.compute_integrity()
        report = catalog_report(catalog)
        assert "integrity_sha256" in report or "Integrity" in report

    def test_report_contains_verification_code(self):
        catalog = AssetCatalog(assets=[], summary={})
        report = catalog_report(catalog)
        assert "validate_asset" in report


# --------------------------------------------------------------------------- #
# Lifecycle transitions
# --------------------------------------------------------------------------- #

class TestLifecycleTransitions:
    def test_all_states_covered(self):
        for state in LIFECYCLE_STATES:
            assert state in LIFECYCLE_TRANSITIONS

    def test_draft_can_go_to_reviewed(self):
        assert "reviewed" in LIFECYCLE_TRANSITIONS["draft"]

    def test_reviewed_can_go_to_published(self):
        assert "published" in LIFECYCLE_TRANSITIONS["reviewed"]

    def test_published_can_go_to_deprecated(self):
        assert "deprecated" in LIFECYCLE_TRANSITIONS["published"]

    def test_deprecated_is_terminal(self):
        assert LIFECYCLE_TRANSITIONS["deprecated"] == ()
