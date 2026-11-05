#!/usr/bin/env python3
"""Figures for the PR #92 merge-cure report, drawn straight from the evidence JSON.

docs/assets/merge_cure_pr92/audit.png  -- lossless-union accounting per file class
docs/assets/merge_cure_pr92/tests.png  -- merged vs main-baseline pytest counts
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs/assets/merge_cure_pr92")
os.makedirs(OUT, exist_ok=True)


def audit_figure() -> str:
    ev = json.load(open(os.path.join(ROOT, "artifacts/merge_cure_pr92_evidence.json"), encoding="utf-8"))
    both = [e for e in ev["both"] if "per_side" in e]
    labels, main_ok, branch_ok, superseded = [], [], [], []
    for e in both:
        labels.append(os.path.basename(e["path"]))
        main_ok.append(e["per_side"]["main"]["verbatim_present"])
        branch_ok.append(e["per_side"]["branch"]["verbatim_present"])
        superseded.append(e["per_side"]["branch"]["superseded"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(["main-only", "branch-only", "both\n(union-audited)"],
            [ev["counts"]["main_only"], ev["counts"]["branch_only"], ev["counts"]["both"]],
            color=["#4c72b0", "#dd8452", "#55a868"])
    for i, v in enumerate([ev["counts"]["main_only"], ev["counts"]["branch_only"], ev["counts"]["both"]]):
        ax1.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("files")
    ax1.set_title(f"Merge file classes (total {ev['totals']['files_checked']} checked)\n"
                  f"verdict: {ev['totals']['verdict']}", fontsize=10)
    ax1.set_ylim(0, max(ev["counts"].values()) * 1.25)

    x = range(len(labels))
    w = 0.38
    ax2.bar([i - w / 2 for i in x], main_ok, w, label="main-added lines kept", color="#4c72b0")
    ax2.bar([i + w / 2 for i in x], branch_ok, w, label="branch-added lines kept verbatim", color="#dd8452")
    ax2.bar([i + w / 2 for i in x], superseded, w, bottom=branch_ok,
            label="branch lines superseded (documented)", color="#c44e52", hatch="//")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=8, rotation=12)
    ax2.set_ylabel("added lines")
    ax2.set_title("Two-sided files: added-line accounting\n(0 unaccounted in either direction)", fontsize=10)
    ax2.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUT, "audit.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def tests_figure() -> str:
    ev = json.load(open(os.path.join(ROOT, "artifacts/merge_cure_pr92_tests.json"), encoding="utf-8"))
    names = list(ev["sets"])
    merged = [ev["sets"][n]["merged"]["passed"] or 0 for n in names]
    base = [(ev["sets"][n]["baseline"] or {}).get("passed") for n in names]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = range(len(names))
    w = 0.36
    ax.bar([i - w / 2 for i in x], merged, w, label="merged branch", color="#55a868")
    ax.bar([i + w / 2 for i in x], [b or 0 for b in base], w, label="main baseline", color="#8172b3")
    for i, (m, b) in enumerate(zip(merged, base)):
        ax.text(i - w / 2, m, str(m), ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, (b or 0), ("n/a" if b is None else str(b)), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("pytest passed")
    ax.set_title(f"Core tests: merged vs main baseline ({ev['verdict']})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "tests.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(audit_figure())
    print(tests_figure())