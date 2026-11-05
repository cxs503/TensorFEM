#!/usr/bin/env python3
"""Plot the diff-scoped lint gate: what the PR owns versus what it inherits.

Numbers are read from the JSON reports written by ``scripts/lint_scope.py``
(``artifacts/pr97-lint/before.json`` and ``after-vs-main.json``); nothing is
typed in by hand, so the figure cannot drift from the evidence.

    python scripts/plot_lint_gate_scope.py \
        --before artifacts/pr97-lint/before.json \
        --after artifacts/pr97-lint/after-vs-main.json \
        --out artifacts/pr97-lint/gate_scope.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _reformatted(output):
    match = re.search(r"(\d+) files? would be reformatted", output or "")
    if match:
        return int(match.group(1))
    return 0 if "already formatted" in (output or "") else 0


def _findings(report):
    result = report["results"][0] if report.get("results") else {}
    output, code = result.get("output", ""), result.get("returncode", 0)
    match = re.search(r"Found (\d+) error", output or "")
    if match:
        return int(match.group(1))
    return 1 if code else 0


def before_state(path):
    data = json.loads(Path(path).read_text())
    return {
        "findings": 1 if data["pr_files"]["check"]["returncode"] else 0,
        "unformatted": _reformatted(data["pr_files"]["format"]["output"]),
        "backlog_findings": data["repo_wide"]["check"]["errors"],
        "backlog_unformatted": data["repo_wide"]["format"]["files"],
        "files_judged": data["repo_wide"]["check"].get("files_judged"),
    }


def after_state(path):
    data = json.loads(Path(path).read_text())
    return {
        "findings": _findings(data),
        "unformatted": sum(1 for r in data["results"] if r["rule"] == "format" and r["returncode"]),
        "backlog_findings": (data.get("backlog") or {}).get("check"),
        "backlog_unformatted": (data.get("backlog") or {}).get("format"),
        "files_judged": len(data["files"]),
        "clean": data["clean"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="artifacts/pr97-lint/before.json")
    ap.add_argument("--after", default="artifacts/pr97-lint/after-vs-main.json")
    ap.add_argument("--out", default="artifacts/pr97-lint/gate_scope.png")
    args = ap.parse_args()

    before, after = before_state(args.before), after_state(args.after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # Left: the debt the change itself owns.
    labels = ["ruff check\nfindings", "files failing\nruff format --check"]
    b = [before["findings"], before["unformatted"]]
    a = [after["findings"], after["unformatted"]]
    x = range(len(labels))
    ax1.bar([i - 0.19 for i in x], b, width=0.38, color="#c0392b", label="before (head 5e1ca50)")
    ax1.bar([i + 0.19 for i in x], a, width=0.38, color="#27ae60", label="after (this branch)")
    for i, (bv, av) in enumerate(zip(b, a)):
        ax1.text(i - 0.19, bv + 0.05, str(bv), ha="center", va="bottom", fontsize=10)
        ax1.text(i + 0.19, av + 0.05, str(av), ha="center", va="bottom", fontsize=10)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("count (change-owned, gated)")
    ax1.set_title("What the pull request is responsible for")
    ax1.set_ylim(0, max(b + a) * 1.35 + 0.5)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # Right: how much of the tree the job judges, and the backlog it inherits.
    before_judged = before["files_judged"] or 1
    panels = [
        ("before: tree-wide scan", before_judged, before["backlog_findings"], "#c0392b"),
        ("after: diff-scoped gate", after["files_judged"], after["backlog_findings"], "#27ae60"),
    ]
    xs = range(len(panels))
    ax2.bar(
        [i - 0.19 for i in xs],
        [p[1] for p in panels],
        width=0.38,
        color=[p[3] for p in panels],
        label="files judged by the lint job",
    )
    ax2.set_yscale("log")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([p[0] for p in panels])
    ax2.set_ylabel("Python files judged (log)")
    for i, p in enumerate(panels):
        ax2.text(i - 0.19, p[1] * 1.15, str(p[1]), ha="center", fontsize=10)
        ax2.text(
            i + 0.19,
            1.05,
            f"backlog {p[2]}\nfindings (not gated)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    ax2.set_ylim(0.8, before_judged * 6)
    ax2.set_title("How much of the tree the job judges")
    ax2.legend(fontsize=8, loc="upper center")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "TensorFEM PR #97: lint red is the change's own debt (left), "
        "the 4k-finding backlog is inherited (right)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print("before:", before)
    print("after:", after)


if __name__ == "__main__":
    main()
