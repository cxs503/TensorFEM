#!/usr/bin/env python3
"""Tests for tools/knowledge_governance.py — cross-anchor knowledge governance."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from knowledge_governance import (
    _combined_similarity,
    _jaccard,
    _normalise,
    _token_overlap,
    build_entity_clusters,
    extract_knowledge_objects,
    find_contradictions,
    find_duplicate_candidates,
    load_certificates,
    KnowledgeObject,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

CERT_PATH = REPO_ROOT / "benchmarks" / "verified" / "anchor_certificates.json"


@pytest.fixture(scope="module")
def cert():
    return load_certificates(CERT_PATH)


@pytest.fixture(scope="module")
def objects(cert):
    return extract_knowledge_objects(cert)


@pytest.fixture(scope="module")
def objects_cert_only(cert):
    return extract_knowledge_objects(cert, cert_only=True)


# --------------------------------------------------------------------------- #
# Similarity helpers
# --------------------------------------------------------------------------- #

class TestNormalise:
    def test_lowercase_and_strip(self):
        assert _normalise("Hello World!") == ["hello", "world"]

    def test_removes_function_words(self):
        """Only English function words are removed, not physics terms."""
        result = _normalise("the max relative error of the deflection")
        # Function words removed
        assert "the" not in result
        assert "of" not in result
        # Physics terms preserved
        assert "deflection" in result
        assert "error" in result
        assert "max" in result
        assert "relative" in result

    def test_empty_string(self):
        assert _normalise("") == []

    def test_numbers_preserved(self):
        result = _normalise("error < 0.01 at N=40")
        assert "40" in result

    def test_physics_terms_preserved(self):
        """Key physics terms must survive normalisation."""
        result = _normalise("cantilever tip deflection relative error")
        assert "cantilever" in result
        assert "tip" in result
        assert "deflection" in result


class TestJaccard:
    def test_identical(self):
        assert _jaccard(["a", "b"], ["a", "b"]) == 1.0

    def test_disjoint(self):
        assert _jaccard(["a", "b"], ["c", "d"]) == 0.0

    def test_partial(self):
        assert _jaccard(["a", "b"], ["b", "c"]) == 1 / 3

    def test_empty_both(self):
        assert _jaccard([], []) == 1.0

    def test_empty_one(self):
        assert _jaccard(["a"], []) == 0.0


class TestTokenOverlap:
    def test_full_overlap(self):
        assert _token_overlap(["a", "b"], ["a", "b", "c"]) == 1.0

    def test_no_overlap(self):
        assert _token_overlap(["a", "b"], ["c", "d"]) == 0.0

    def test_empty_a(self):
        assert _token_overlap([], ["a", "b"]) == 0.0


class TestCombinedSimilarity:
    def test_identical_strings(self):
        s = "cantilever tip deflection relative error"
        assert _combined_similarity(s, s) == 1.0

    def test_similar_strings(self):
        a = "cantilever tip deflection relative error N=40"
        b = "cantilever tip deflection relative error N=20"
        # High similarity but not perfect (N=40 vs N=20 differ)
        assert _combined_similarity(a, b) > 0.7

    def test_dissimilar_strings(self):
        a = "cantilever tip deflection"
        b = "hertz contact pressure distribution"
        assert _combined_similarity(a, b) < 0.3


# --------------------------------------------------------------------------- #
# Knowledge object extraction
# --------------------------------------------------------------------------- #

class TestExtractKnowledgeObjects:
    def test_non_empty(self, objects):
        assert len(objects) > 0

    def test_all_have_anchor(self, objects):
        assert all(obj.anchor for obj in objects)

    def test_all_have_gate_id(self, objects):
        assert all(obj.gate_id for obj in objects)

    def test_certified_filter(self, objects, objects_cert_only):
        # cert_only should be a subset
        anchors_cert_only = {obj.anchor for obj in objects_cert_only}
        for obj in objects_cert_only:
            assert obj.certified
        # All cert_only objects should be in the full set
        full_anchors = {obj.anchor for obj in objects}
        assert anchors_cert_only.issubset(full_anchors)

    def test_tokenisation(self, objects):
        for obj in objects:
            assert isinstance(obj.quantity_tokens, list)


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #

class TestDuplicateCandidates:
    def test_finds_candidates(self, objects):
        candidates = find_duplicate_candidates(objects, threshold=0.3)
        assert len(candidates) > 0

    def test_sorted_by_similarity(self, objects):
        candidates = find_duplicate_candidates(objects, threshold=0.3)
        sims = [c.similarity for c in candidates]
        assert sims == sorted(sims, reverse=True)

    def test_higher_threshold_fewer_candidates(self, objects):
        c_low = find_duplicate_candidates(objects, threshold=0.3)
        c_high = find_duplicate_candidates(objects, threshold=0.7)
        assert len(c_high) <= len(c_low)

    def test_winkler_plate_cluster(self, objects):
        """fem_winkler_plate and fem_winkler_plate_k_sweep should be detected."""
        candidates = find_duplicate_candidates(objects, threshold=0.5)
        winkler_pairs = [
            c for c in candidates
            if {"fem_winkler_plate", "fem_winkler_plate_k_sweep"} <= {c.anchor_a, c.anchor_b}
        ]
        assert len(winkler_pairs) >= 1

    def test_no_self_pairs(self, objects):
        candidates = find_duplicate_candidates(objects, threshold=0.3)
        for c in candidates:
            assert c.anchor_a != c.anchor_b


# --------------------------------------------------------------------------- #
# Contradiction detection
# --------------------------------------------------------------------------- #

class TestContradictions:
    def test_finds_contradictions(self, objects):
        contradictions = find_contradictions(objects)
        assert len(contradictions) > 0

    def test_no_self_contradictions(self, objects):
        contradictions = find_contradictions(objects)
        for c in contradictions:
            assert c.anchor_a != c.anchor_b

    def test_severity_levels(self, objects):
        contradictions = find_contradictions(objects)
        severities = {c.severity for c in contradictions}
        assert severities.issubset({"hard", "soft"})

    def test_values_present(self, objects):
        contradictions = find_contradictions(objects)
        for c in contradictions:
            assert c.value_a is not None
            assert c.value_b is not None


# --------------------------------------------------------------------------- #
# Entity clustering
# --------------------------------------------------------------------------- #

class TestEntityClusters:
    def test_builds_clusters(self, objects):
        candidates = find_duplicate_candidates(objects, threshold=0.5)
        clusters = build_entity_clusters(objects, candidates, threshold=0.5)
        # Should find at least the winkler cluster
        assert len(clusters) >= 1

    def test_clusters_have_multiple_anchors(self, objects):
        candidates = find_duplicate_candidates(objects, threshold=0.5)
        clusters = build_entity_clusters(objects, candidates, threshold=0.5)
        for cl in clusters:
            assert cl["size"] >= 2
            assert len(cl["anchors"]) >= 2

    def test_cluster_id_is_hash(self, objects):
        candidates = find_duplicate_candidates(objects, threshold=0.5)
        clusters = build_entity_clusters(objects, candidates, threshold=0.5)
        for cl in clusters:
            assert len(cl["cluster_id"]) == 8  # 8-char hex


# --------------------------------------------------------------------------- #
# Integration: full pipeline
# --------------------------------------------------------------------------- #

class TestFullPipeline:
    def test_artifact_writable(self, tmp_path, cert):
        """Verify the full analysis produces valid JSON."""
        from knowledge_governance import (
            extract_knowledge_objects,
            find_duplicate_candidates,
            find_contradictions,
            build_entity_clusters,
        )

        objects = extract_knowledge_objects(cert)
        candidates = find_duplicate_candidates(objects)
        contradictions = find_contradictions(objects)
        clusters = build_entity_clusters(objects, candidates)

        artifact = {
            "schema": "tensorfem.knowledge-governance/1",
            "summary": {
                "total_anchors": cert.get("anchors", 0),
                "knowledge_objects": len(objects),
                "duplicate_candidates": len(candidates),
                "contradictions": len(contradictions),
                "entity_clusters": len(clusters),
            },
            "duplicate_candidates": [
                {"anchor_a": c.anchor_a, "anchor_b": c.anchor_b, "similarity": c.similarity}
                for c in candidates
            ],
            "contradictions": [
                {"anchor_a": c.anchor_a, "anchor_b": c.anchor_b, "severity": c.severity}
                for c in contradictions
            ],
            "entity_clusters": clusters,
        }

        out = tmp_path / "test_kg.json"
        with open(out, "w") as f:
            json.dump(artifact, f)

        # Round-trip
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["schema"] == "tensorfem.knowledge-governance/1"
        assert loaded["summary"]["knowledge_objects"] == len(objects)
