"""Render the PR #56 evidence figures from frozen artifacts (no re-solve).

Figure 1: Cook membrane recovered-field convergence, log-log, against the 3 % gate
          and an observed-order-2 guide, using benchmarks/cook_stress_recovery.json.
Figure 2: two-sided merge accounting — lines added per side on the three conflicted
          files and how many survive in the merged tree (content-loss audit).
"""
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/merge_m3148"
OUT.mkdir(parents=True, exist_ok=True)


def fig_convergence():
    data = json.loads((ROOT / "benchmarks/cook_stress_recovery.json").read_text())
    levels = data["levels"]
    n = [row["divisions"] for row in levels]
    err = [100.0 * row["max_relative_error"] for row in levels]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(n, err, "o-", lw=1.8, ms=6, label="recovered nodal field (2x2 Gauss + nodal avg)")
    ax.axhline(3.0, color="crimson", ls="--", lw=1.3, label="3 % acceptance gate")
    # order-2 guide anchored on the finest point
    guide = [err[-1] * (n[-1] / k) ** 2 for k in n]
    ax.loglog(n, guide, "k:", lw=1.1, label="observed order 2 guide")
    for x, y in zip(n, err):
        ax.annotate(f"{y:.3f}%", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xlabel("mesh divisions per edge n")
    ax.set_ylabel("max pointwise von Mises error vs frozen Q8 reference  [%]")
    ax.set_title("Cook membrane: recovered stress field convergence\n(PR #56 union, artifact %s)" % data.get("stress_recovery_qualified"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "cook_recovery_convergence.png", dpi=130)
    plt.close(fig)
    return [(k, v) for k, v in zip(n, err)]


def line_sets(path, base, ref):
    diff = subprocess.run(["git", "diff", "-U0", base, ref, "--", path],
                          cwd=ROOT, capture_output=True, text=True).stdout
    out = set()
    for line in diff.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            text = line[1:].strip()
            if text and not text.startswith(("<<<", "===", ">>>")):
                out.add(text)
    return out


def fig_merge_accounting():
    base = subprocess.run(["git", "merge-base", "HEAD^1", "HEAD^2"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    files = ["CHANGELOG.md", "docs/benchmark-qualification-status.md",
             "src/tensorfem/additional_benchmark_fields.py"]
    rows = []
    for path in files:
        cur = {l.strip() for l in (ROOT / path).read_text().split("\n")}
        ours = line_sets(path, base, "HEAD^1")
        theirs = line_sets(path, base, "HEAD^2")
        rows.append((path.split("/")[-1],
                     len(ours), len(ours & cur),
                     len(theirs), len(theirs & cur)))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    idx = range(len(rows))
    w = 0.2
    ax.bar([i - 1.5 * w for i in idx], [r[1] for r in rows], w, label="ours: lines added")
    ax.bar([i - 0.5 * w for i in idx], [r[2] for r in rows], w, label="ours: preserved")
    ax.bar([i + 0.5 * w for i in idx], [r[3] for r in rows], w, label="main: lines added")
    ax.bar([i + 1.5 * w for i in idx], [r[4] for r in rows], w, label="main: preserved")
    for i, r in enumerate(rows):
        ax.text(i - 1.0 * w, max(r[1], r[3]) + 3, f"{r[2]}/{r[1]}", ha="center", fontsize=8)
        ax.text(i + 1.0 * w, max(r[1], r[3]) + 3, f"{r[4]}/{r[3]}", ha="center", fontsize=8)
    ax.set_xticks(list(idx))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("added lines (verbatim match in merged tree)")
    ax.set_title("PR #56 three-way merge: two-sided content retention audit")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "merge_retention_audit.png", dpi=130)
    plt.close(fig)
    return rows


if __name__ == "__main__":
    print("convergence points (n, err%):", fig_convergence())
    print("retention rows:", fig_merge_accounting())
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))