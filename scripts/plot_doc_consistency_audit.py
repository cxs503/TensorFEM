#!/usr/bin/env python3
"""Render the m8858 doc-consistency audit figures (matplotlib, Agg)."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/doc_consistency/audit.json"
OUT = ROOT / "docs/assets/doc-consistency-m8858"


def main() -> int:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    items = payload["items"]
    OUT.mkdir(parents=True, exist_ok=True)

    kinds = ["import", "signature", "path", "command"]
    sev = ["ok", "warn", "error"]
    counts = {k: collections.Counter(i["severity"] for i in items if i["kind"] == k) for k in kinds}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    bottoms = [0] * len(kinds)
    colors = {"ok": "#3a7a3a", "warn": "#c9a227", "error": "#a52a2a"}
    for s in sev:
        vals = [counts[k][s] for k in kinds]
        ax.bar(kinds, vals, bottom=bottoms, color=colors[s], label=s, width=0.62)
        for x, (v, b) in enumerate(zip(vals, bottoms)):
            if v:
                ax.text(x, b + v / 2, str(v), ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_yscale("log")
    ax.set_ylabel("audited claims (log scale)")
    ax.set_title("A. README/examples/docs vs shipped API\n"
                 f"{len(items)} claims: import / signature / path / command", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25, which="both")

    ax = axes[1]
    labels = ["pass", "slow\n(>45 s cap)", "CLI example\n(needs args)"]
    vals = [71, 8, 3]
    bars = ax.bar(labels, vals, color=["#3a7a3a", "#c9a227", "#4a6fa5"], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", fontsize=10,
                fontweight="bold")
    ax.set_ylim(0, 82)
    ax.set_ylabel("examples (of 82)")
    ax.set_title("B. example runnability sweep\n(45 s wall clock per example)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    png = OUT / "audit-summary.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {png.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())