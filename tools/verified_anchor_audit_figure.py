"""Figure for the verified-anchor certificate audit (docs/ figures).

Reads only ``benchmarks/verified/anchor_certificates.json`` -- the artifact the
auditor writes -- so no number in the figure is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "benchmarks/verified/anchor_certificates.json"
OUT = REPO / "docs/assets/anchor-certificates/certificate_audit.png"


def main() -> Path:
    audit = json.loads(AUDIT.read_text())
    records = audit["records"]
    names = [r["anchor"].replace("fem_", "") for r in records]
    agrees = [r["gates"] - r["gates_unrecomputable"] - r["gates_contradicted"] for r in records]
    unrecomputable = [r["gates_unrecomputable"] for r in records]
    contradicted = [r["gates_contradicted"] for r in records]
    order = np.argsort([-r["gates"] for r in records])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), gridspec_kw={"width_ratios": [2.1, 1.0]})

    ax = axes[0]
    x = np.arange(len(records))
    a = np.array(agrees)[order]
    u = np.array(unrecomputable)[order]
    c = np.array(contradicted)[order]
    ax.bar(x, a, color="#2b7bba", label="verdict recomputed, agrees")
    ax.bar(x, u, bottom=a, color="#e8a33d", label="verdict not machine-checkable")
    ax.bar(x, c, bottom=a + u, color="#c0392b", label="contradiction under declared operator")
    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("gates in gates_verified.json")
    ax.set_title(
        f"Anchor certificates: {audit['anchors']} anchors, {audit['gates_total']} gates, "
        f"{audit['certified']} certified"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    labels = ["certified", "contradicted", "uncertified"]
    values = [audit["certified"], audit["contradicted"], audit["uncertified"]]
    bars = ax.barh(labels, values, color=["#2e8b57", "#c0392b", "#7f8c8d"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)
    ax.set_xlabel("anchors")
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_title("Registry grades before the fail-closed gate")
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "TensorFEM verified-anchor certificate audit "
        f"(git tracking checked: {audit['git_tracking_checked']})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(OUT)
    return OUT


if __name__ == "__main__":
    main()
