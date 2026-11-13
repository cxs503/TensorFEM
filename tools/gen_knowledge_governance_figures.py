#!/usr/bin/env python3
"""Generate knowledge governance figures for the TensorFEM report.

Produces:
- artifacts/kg_similarity_heatmap.png: similarity matrix heatmap
- artifacts/kg_contradiction_bars.png: contradiction severity bar chart
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ARTIFACT_DIR = REPO_ROOT / "artifacts"
CERT_PATH = REPO_ROOT / "benchmarks" / "verified" / "anchor_certificates.json"


def _load_cert():
    with open(CERT_PATH) as f:
        return json.load(f)


def _load_kg():
    kg_path = ARTIFACT_DIR / "knowledge_governance.json"
    if kg_path.exists():
        with open(kg_path) as f:
            return json.load(f)
    return None


def gen_similarity_heatmap() -> str | None:
    """Generate a similarity heatmap of anchor pairs."""
    if not HAS_MPL:
        return None

    kg = _load_kg()
    if not kg:
        return None

    # Get all anchors from the certificate
    cert = _load_cert()
    anchors = sorted(set(
        r["anchor"] for r in cert.get("records", [])
        if r.get("certified")
    ))

    n = len(anchors)
    # Build similarity matrix
    sim_matrix = np.zeros((n, n))
    anchor_idx = {a: i for i, a in enumerate(anchors)}

    for c in kg.get("duplicate_candidates", []):
        a, b = c["anchor_a"], c["anchor_b"]
        if a in anchor_idx and b in anchor_idx:
            i, j = anchor_idx[a], anchor_idx[b]
            sim_matrix[i, j] = c["similarity"]
            sim_matrix[j, i] = c["similarity"]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    # Labels
    short_labels = [a.replace("fem_", "") for a in anchors]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=8)

    ax.set_title("Cross-Anchor Similarity Matrix (Certified Anchors)", fontsize=12)
    plt.colorbar(im, ax=ax, label="Similarity")

    out = ARTIFACT_DIR / "kg_similarity_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def gen_contradiction_bars() -> str | None:
    """Generate a bar chart of contradictions by severity."""
    if not HAS_MPL:
        return None

    kg = _load_kg()
    if not kg:
        return None

    contradictions = kg.get("contradictions", [])
    hard = sum(1 for c in contradictions if c.get("severity") == "hard")
    soft = sum(1 for c in contradictions if c.get("severity") == "soft")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Hard", "Soft"], [hard, soft], color=["#d62728", "#ff7f0e"])
    ax.set_ylabel("Count")
    ax.set_title("Cross-Anchor Contradictions by Severity", fontsize=12)
    for bar, val in zip(bars, [hard, soft]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontweight="bold")

    out = ARTIFACT_DIR / "kg_contradiction_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def gen_anchor_gate_distribution() -> str | None:
    """Generate a bar chart of gates per anchor."""
    if not HAS_MPL:
        return None

    cert = _load_cert()
    records = cert.get("records", [])

    anchors = []
    gates = []
    colors = []
    for r in sorted(records, key=lambda x: x.get("gates", 0), reverse=True):
        if r.get("gates", 0) > 0:
            anchors.append(r["anchor"].replace("fem_", ""))
            gates.append(r["gates"])
            colors.append("#2ca02c" if r.get("certified") else "#d62728")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(anchors)), gates, color=colors)
    ax.set_xticks(range(len(anchors)))
    ax.set_xticklabels(anchors, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Gates")
    ax.set_title("Gates per Anchor (Green=Certified, Red=Not Certified)", fontsize=12)

    out = ARTIFACT_DIR / "kg_anchor_gate_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    if not HAS_MPL:
        print("matplotlib not available, skipping figures")
        sys.exit(0)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    paths = []
    for gen in [gen_similarity_heatmap, gen_contradiction_bars, gen_anchor_gate_distribution]:
        p = gen()
        if p:
            paths.append(p)
            print(f"Generated {p}")

    if not paths:
        print("No figures generated")
