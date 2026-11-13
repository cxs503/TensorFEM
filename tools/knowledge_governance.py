#!/usr/bin/env python3
"""Cross-anchor knowledge governance for TensorFEM verification registry.

Purpose
-------
The existing ``verified_anchor_audit.py`` checks *within* a single anchor for
internal certificate consistency.  This tool operates **across** anchors to
provide the three pillars of knowledge governance at scale:

1. **Entity Resolution (Deduplication)**
   Identify anchors that test the same or overlapping physical cases by
   analysing gate quantities, anchor names, and directory metadata.  Outputs
   a similarity matrix and candidate duplicate pairs.

2. **Cross-Anchor Contradiction Detection**
   Find cases where different anchors claim contradictory values for the same
   physical quantity (e.g. two anchors reporting different reference
   deflections for the same cantilever case).

3. **Knowledge Object Registry**
   A structured, machine-readable registry of all knowledge objects (gates)
   with their relationships, provenance, and governance status.

Why this matters
----------------
As the verification registry grows (36 anchors, 325 gates as of 2026-11),
manual oversight becomes impossible.  Without cross-anchor governance:
- Duplicate anchors waste compute and confuse users
- Contradictory claims erode trust in the registry
- No single source of truth for "what has been verified"

Design principles
-----------------
- **Zero hand-copying**: all numbers come from ``anchor_certificates.json``
- **Deterministic**: sorted outputs, no timestamps, reproducible
- **Conservative**: similarity thresholds are tunable; false positives are
  reported as *candidates* requiring human review, not automatic merges
- **Non-destructive**: reads only; writes analysis artifacts, never modifies
  the source registry

Usage
-----
    python tools/knowledge_governance.py                     # full analysis
    python tools/knowledge_governance.py --cert-only         # certified anchors only
    python tools/knowledge_governance.py --similarity-threshold 0.6  # adjust dedup threshold
    python tools/knowledge_governance.py --write             # write artifacts
    python tools/knowledge_governance.py --report            # print markdown report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
CERT_PATH = REPO_ROOT / "benchmarks" / "verified" / "anchor_certificates.json"
ARTIFACT_JSON = REPO_ROOT / "artifacts" / "knowledge_governance.json"
REPORT_MD = REPO_ROOT / "docs" / "knowledge-governance-report.md"

# --------------------------------------------------------------------------- #
# Similarity helpers
# --------------------------------------------------------------------------- #

#: Common English function words to normalise before similarity comparison.
#: Physics/engineering terms (deflection, stress, strain, etc.) are deliberately
#: **not** removed — they carry the semantic meaning needed for entity resolution.
_NORMALISE_RE = re.compile(r"[^a-z0-9_]+")
_STOP_WORDS = frozenset({
    "the", "of", "and", "for", "in", "on", "at", "to", "a", "an", "is",
    "must", "be", "over", "with", "from", "by", "as", "or", "not", "no",
    "that", "this", "these", "those", "it", "its", "are", "was", "were",
    "has", "have", "had", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "need", "than", "then", "when", "where", "which",
    "while", "who", "whom", "whose", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "so", "just", "also", "back", "out", "up", "down", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "once", "here", "there",
    "any", "but", "if", "because", "until", "although", "though",
    "declared", "recomputed", "status", "defects", "warnings",
})


def _normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = _NORMALISE_RE.sub(" ", text.lower()).split()
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity of two token lists."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _token_overlap(a: list[str], b: list[str]) -> float:
    """Fraction of tokens in a that also appear in b."""
    if not a:
        return 0.0
    sb = set(b)
    return sum(1 for t in a if t in sb) / len(a)


def _combined_similarity(a: str, b: str) -> float:
    """Combined similarity: weighted Jaccard + token overlap."""
    ta, tb = _normalise(a), _normalise(b)
    j = _jaccard(ta, tb)
    o = (_token_overlap(ta, tb) + _token_overlap(tb, ta)) / 2
    return 0.6 * j + 0.4 * o


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class KnowledgeObject:
    """A single gate from an anchor, normalised for cross-anchor comparison."""
    anchor: str
    gate_id: str
    quantity_raw: str
    quantity_tokens: list[str]
    value: float | None
    threshold: float | None
    operator: str | None
    certified: bool

    @property
    def quantity_normalised(self) -> str:
        return " ".join(self.quantity_tokens)


@dataclass
class DuplicateCandidate:
    """A pair of anchors that may be testing the same physical case."""
    anchor_a: str
    anchor_b: str
    similarity: float
    shared_gates: int
    evidence: list[str]


@dataclass
class Contradiction:
    """A cross-anchor contradiction: same quantity, incompatible values."""
    anchor_a: str
    anchor_b: str
    gate_a: str
    gate_b: str
    quantity_summary: str
    value_a: float
    value_b: float
    threshold_a: float | None
    threshold_b: float | None
    severity: str  # "hard" | "soft"


@dataclass
class GovernanceReport:
    schema: str = "tensorfem.knowledge-governance/1"
    total_anchors: int = 0
    certified_anchors: int = 0
    total_gates: int = 0
    knowledge_objects: list[dict] = field(default_factory=list)
    duplicate_candidates: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    entity_clusters: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #

def load_certificates(path: Path = CERT_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_knowledge_objects(cert: dict, cert_only: bool = False) -> list[KnowledgeObject]:
    """Extract all gates as normalised knowledge objects."""
    objects = []
    for record in cert.get("records", []):
        if cert_only and not record.get("certified"):
            continue
        anchor = record["anchor"]
        certified = record.get("certified", False)
        for gate in record.get("gate_detail", []):
            qty = gate.get("quantity", "")
            tokens = _normalise(qty)
            objects.append(KnowledgeObject(
                anchor=anchor,
                gate_id=gate.get("id", ""),
                quantity_raw=qty,
                quantity_tokens=tokens,
                value=gate.get("value"),
                threshold=gate.get("threshold"),
                operator=gate.get("operator"),
                certified=certified,
            ))
    return objects


def find_duplicate_candidates(
    objects: list[KnowledgeObject],
    threshold: float = 0.5,
) -> list[DuplicateCandidate]:
    """Find anchor pairs with high gate-level similarity.

    Two anchors are candidates for deduplication if they share gates with
    high quantity similarity.  This catches cases like ``fem_beam`` and
    ``fem_beam_static`` which may overlap in what they verify.
    """
    # Group objects by anchor
    by_anchor: dict[str, list[KnowledgeObject]] = {}
    for obj in objects:
        by_anchor.setdefault(obj.anchor, []).append(obj)

    anchors = sorted(by_anchor.keys())
    candidates = []

    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            gates_a = by_anchor[a]
            gates_b = by_anchor[b]

            # Compute pairwise gate similarity
            similarities = []
            shared = 0
            evidence = []
            for ga in gates_a:
                for gb in gates_b:
                    sim = _combined_similarity(ga.quantity_raw, gb.quantity_raw)
                    if sim >= threshold:
                        shared += 1
                        similarities.append(sim)
                        evidence.append(
                            f"{ga.gate_id} ↔ {gb.gate_id} (sim={sim:.3f}): "
                            f"{ga.quantity_raw[:60]}…"
                        )

            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                # Also factor in anchor name similarity
                name_sim = _combined_similarity(a, b)
                combined = 0.7 * avg_sim + 0.3 * name_sim

                candidates.append(DuplicateCandidate(
                    anchor_a=a,
                    anchor_b=b,
                    similarity=round(combined, 4),
                    shared_gates=shared,
                    evidence=evidence[:5],  # cap evidence
                ))

    # Sort by similarity descending
    candidates.sort(key=lambda c: c.similarity, reverse=True)
    return candidates


def find_contradictions(objects: list[KnowledgeObject]) -> list[Contradiction]:
    """Find cross-anchor contradictions: same quantity, incompatible values.

    A contradiction occurs when two gates from different anchors describe
    the same physical quantity but report values that cannot both be true
    (e.g. different reference values for the same beam deflection).
    """
    contradictions = []

    # Group by normalised quantity tokens (first 3 tokens as key)
    by_quantity: dict[str, list[KnowledgeObject]] = {}
    for obj in objects:
        if obj.value is None:
            continue
        # Use first 3 significant tokens as grouping key
        key = "_".join(obj.quantity_tokens[:3]) if obj.quantity_tokens else ""
        if key:
            by_quantity.setdefault(key, []).append(obj)

    for key, group in by_quantity.items():
        if len(group) < 2:
            continue

        # Check all pairs within the group
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a.anchor == b.anchor:
                    continue

                # Check if values are compatible
                # Two values are contradictory if they differ by more than
                # both thresholds allow
                val_diff = abs(a.value - b.value)
                avg_val = (abs(a.value) + abs(b.value)) / 2
                if avg_val == 0:
                    continue

                rel_diff = val_diff / avg_val

                # Hard contradiction: values differ by >10x the tighter threshold
                min_threshold = min(
                    t for t in [a.threshold, b.threshold] if t is not None
                ) if any(t is not None for t in [a.threshold, b.threshold]) else None

                if min_threshold is not None and rel_diff > 10 * min_threshold:
                    severity = "hard"
                elif rel_diff > 0.5:
                    severity = "soft"
                else:
                    continue

                contradictions.append(Contradiction(
                    anchor_a=a.anchor,
                    anchor_b=b.anchor,
                    gate_a=a.gate_id,
                    gate_b=b.gate_id,
                    quantity_summary=a.quantity_raw[:80],
                    value_a=a.value,
                    value_b=b.value,
                    threshold_a=a.threshold,
                    threshold_b=b.threshold,
                    severity=severity,
                ))

    contradictions.sort(key=lambda c: (0 if c.severity == "hard" else 1, c.quantity_summary))
    return contradictions


def build_entity_clusters(
    objects: list[KnowledgeObject],
    candidates: list[DuplicateCandidate],
    threshold: float = 0.5,
) -> list[dict]:
    """Build clusters of anchors that may represent the same entity.

    Uses union-find on the duplicate candidate graph.
    """
    # Union-Find
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], x)
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Initialize
    all_anchors = {obj.anchor for obj in objects}
    for a in all_anchors:
        parent[a] = a

    # Union based on candidates above threshold
    for c in candidates:
        if c.similarity >= threshold:
            union(c.anchor_a, c.anchor_b)

    # Build clusters
    clusters: dict[str, list[str]] = {}
    for a in all_anchors:
        root = find(a)
        clusters.setdefault(root, []).append(a)

    # Filter to clusters with >1 member
    result = []
    for root, members in sorted(clusters.items()):
        if len(members) > 1:
            result.append({
                "cluster_id": hashlib.md5(",".join(sorted(members)).encode()).hexdigest()[:8],
                "anchors": sorted(members),
                "size": len(members),
            })

    return result


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #

def generate_report(
    cert: dict,
    objects: list[KnowledgeObject],
    candidates: list[DuplicateCandidate],
    contradictions: list[Contradiction],
    clusters: list[dict],
) -> str:
    """Generate a markdown report."""
    lines = [
        "# TensorFEM Knowledge Governance Report",
        "",
        "## Overview",
        "",
        f"- **Total anchors**: {cert.get('anchors', 0)}",
        f"- **Certified anchors**: {cert.get('certified', 0)}",
        f"- **Contradicted anchors**: {cert.get('contradicted', 0)}",
        f"- **Uncertified anchors**: {cert.get('uncertified', 0)}",
        f"- **Total gates**: {cert.get('gates_total', 0)}",
        f"- **Knowledge objects analysed**: {len(objects)}",
        f"- **Duplicate candidates found**: {len(candidates)}",
        f"- **Cross-anchor contradictions**: {len(contradictions)}",
        f"- **Entity clusters**: {len(clusters)}",
        "",
        "## Entity Clusters (Potential Duplicates)",
        "",
    ]

    if clusters:
        for cl in clusters:
            lines.append(f"### Cluster `{cl['cluster_id']}` ({cl['size']} anchors)")
            lines.append("")
            lines.append(f"Anchors: {', '.join(cl['anchors'])}")
            lines.append("")
            # Find evidence from candidates
            for c in candidates:
                if c.anchor_a in cl["anchors"] and c.anchor_b in cl["anchors"]:
                    lines.append(f"- **{c.anchor_a} ↔ {c.anchor_b}** (similarity={c.similarity:.3f})")
                    for ev in c.evidence[:3]:
                        lines.append(f"  - {ev}")
            lines.append("")
    else:
        lines.append("No entity clusters found at current similarity threshold.")
        lines.append("")

    lines.append("## Top Duplicate Candidates")
    lines.append("")

    if candidates:
        lines.append("| Anchor A | Anchor B | Similarity | Shared Gates |")
        lines.append("|----------|----------|------------|--------------|")
        for c in candidates[:15]:
            lines.append(f"| {c.anchor_a} | {c.anchor_b} | {c.similarity:.4f} | {c.shared_gates} |")
        lines.append("")
    else:
        lines.append("No duplicate candidates found.")
        lines.append("")

    lines.append("## Cross-Anchor Contradictions")
    lines.append("")

    if contradictions:
        hard = [c for c in contradictions if c.severity == "hard"]
        soft = [c for c in contradictions if c.severity == "soft"]
        lines.append(f"- **Hard contradictions**: {len(hard)}")
        lines.append(f"- **Soft contradictions**: {len(soft)}")
        lines.append("")

        if hard:
            lines.append("### Hard Contradictions")
            lines.append("")
            for c in hard[:10]:
                lines.append(f"- **{c.anchor_a}** ({c.gate_a}) vs **{c.anchor_b}** ({c.gate_b})")
                lines.append(f"  - Quantity: {c.quantity_summary}")
                lines.append(f"  - Values: {c.value_a:.6e} vs {c.value_b:.6e}")
                lines.append("")
    else:
        lines.append("No cross-anchor contradictions detected.")
        lines.append("")

    lines.append("## Knowledge Object Registry Summary")
    lines.append("")

    # Group by anchor
    by_anchor: dict[str, int] = {}
    for obj in objects:
        by_anchor[obj.anchor] = by_anchor.get(obj.anchor, 0) + 1

    lines.append("| Anchor | Gates | Certified |")
    lines.append("|--------|-------|-----------|")
    for anchor in sorted(by_anchor.keys()):
        certified = any(obj.certified for obj in objects if obj.anchor == anchor)
        lines.append(f"| {anchor} | {by_anchor[anchor]} | {'✓' if certified else '✗'} |")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Entity resolution**: Jaccard + token overlap similarity on gate quantity descriptions, combined with anchor name similarity (weights 0.6/0.4 for gate, 0.7/0.3 for candidate scoring)")
    lines.append("- **Contradiction detection**: Groups gates by normalised quantity tokens; flags pairs where relative value difference exceeds 10× the tighter threshold (hard) or 50% (soft)")
    lines.append("- **Clustering**: Union-Find on duplicate candidate graph at similarity threshold 0.5")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Similarity is text-based; semantic equivalence (e.g. 'tip deflection' vs 'end displacement') may be missed")
    lines.append("- Contradiction detection requires numeric values; gates without values are excluded")
    lines.append("- Thresholds are tunable; results should be reviewed by domain experts before action")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-anchor knowledge governance for TensorFEM")
    parser.add_argument("--cert-only", action="store_true", help="Analyse certified anchors only")
    parser.add_argument("--similarity-threshold", type=float, default=0.5, help="Dedup similarity threshold")
    parser.add_argument("--write", action="store_true", help="Write artifacts")
    parser.add_argument("--report", action="store_true", help="Print markdown report")
    parser.add_argument("--cert-path", type=str, default=str(CERT_PATH), help="Path to anchor_certificates.json")
    args = parser.parse_args()

    cert = load_certificates(Path(args.cert_path))
    objects = extract_knowledge_objects(cert, cert_only=args.cert_only)
    candidates = find_duplicate_candidates(objects, threshold=args.similarity_threshold)
    contradictions = find_contradictions(objects)
    clusters = build_entity_clusters(objects, candidates, threshold=args.similarity_threshold)

    report = GovernanceReport(
        total_anchors=cert.get("anchors", 0),
        certified_anchors=cert.get("certified", 0),
        total_gates=cert.get("gates_total", 0),
        knowledge_objects=[{
            "anchor": obj.anchor,
            "gate_id": obj.gate_id,
            "quantity_tokens": obj.quantity_tokens[:5],
            "value": obj.value,
            "threshold": obj.threshold,
            "operator": obj.operator,
            "certified": obj.certified,
        } for obj in objects],
        duplicate_candidates=[{
            "anchor_a": c.anchor_a,
            "anchor_b": c.anchor_b,
            "similarity": c.similarity,
            "shared_gates": c.shared_gates,
            "evidence": c.evidence,
        } for c in candidates],
        contradictions=[{
            "anchor_a": c.anchor_a,
            "anchor_b": c.anchor_b,
            "gate_a": c.gate_a,
            "gate_b": c.gate_b,
            "quantity_summary": c.quantity_summary,
            "value_a": c.value_a,
            "value_b": c.value_b,
            "threshold_a": c.threshold_a,
            "threshold_b": c.threshold_b,
            "severity": c.severity,
        } for c in contradictions],
        entity_clusters=clusters,
        summary={
            "total_anchors": cert.get("anchors", 0),
            "certified_anchors": cert.get("certified", 0),
            "total_gates": cert.get("gates_total", 0),
            "knowledge_objects": len(objects),
            "duplicate_candidates": len(candidates),
            "contradictions": len(contradictions),
            "entity_clusters": len(clusters),
            "hard_contradictions": len([c for c in contradictions if c.severity == "hard"]),
            "soft_contradictions": len([c for c in contradictions if c.severity == "soft"]),
        },
    )

    if args.write:
        ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_JSON, "w") as f:
            json.dump({
                "schema": report.schema,
                "summary": report.summary,
                "duplicate_candidates": report.duplicate_candidates,
                "contradictions": report.contradictions,
                "entity_clusters": report.entity_clusters,
            }, f, indent=2, sort_keys=True)
        print(f"Wrote {ARTIFACT_JSON}")

        md = generate_report(cert, objects, candidates, contradictions, clusters)
        REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_MD, "w") as f:
            f.write(md)
        print(f"Wrote {REPORT_MD}")

    if args.report:
        md = generate_report(cert, objects, candidates, contradictions, clusters)
        print(md)

    # Always print summary
    print(f"\n=== Knowledge Governance Summary ===")
    print(f"Anchors analysed: {report.summary['total_anchors']}")
    print(f"Knowledge objects: {report.summary['knowledge_objects']}")
    print(f"Duplicate candidates: {report.summary['duplicate_candidates']}")
    print(f"Contradictions: {report.summary['contradictions']} (hard={report.summary['hard_contradictions']}, soft={report.summary['soft_contradictions']})")
    print(f"Entity clusters: {report.summary['entity_clusters']}")


if __name__ == "__main__":
    main()
