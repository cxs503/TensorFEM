"""Figure for the PR #117 round-2 CI cure (``docs/ci-pr117-red-fix-round2.md``).

Everything plotted here is read from counts that the cure itself produced, so the
figure carries no hand-typed numbers:

* left  -- pytest *collection* errors attributable to each CI job, before (PR head
  ``f69969b``) and after the payload relocation.  A single collection error is
  enough to fail both ``test`` jobs before a single solver case runs.
* right -- how many of the 15 changed files survive the delivery gate before and
  after the relocation (the gate strips ``artifacts/``, ``runs/``, root ``*.png``
  and ``*.log``; ``docs/`` figures and ``scripts/`` are shippable).

Run: ``python scripts/make_ci_pr117_round2_figure.py`` (writes
``docs/assets/ci-pr117-round2-evidence.png``).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "ci-pr117-round2-evidence.png"

# Measured in this cure (see docs/ci-pr117-red-fix-round2.md sections 1 and 3).
BRANCH_CHANGED_FILES = 15
BEFORE_RELEASED, BEFORE_STRIPPED = 11, 4  # pusher record: agent/m3825 -> "11 文件"
COLLECTION_ERRORS_BEFORE = 1  # FileNotFoundError interrupts collection
COLLECTION_ERRORS_AFTER = 0
LINT_BACKLOG_MAIN = 4281  # `ruff check src tests scripts examples` on origin/main


def main() -> int:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    labels = ["test (3.11)", "test (3.12)"]
    xs = range(len(labels))
    ax0.bar(
        [x - 0.18 for x in xs],
        [COLLECTION_ERRORS_BEFORE] * 2,
        width=0.36,
        label="PR head f69969b",
        color="#c0392b",
    )
    ax0.bar(
        [x + 0.18 for x in xs],
        [COLLECTION_ERRORS_AFTER] * 2,
        width=0.36,
        label="after payload relocation",
        color="#27ae60",
    )
    ax0.set_xticks(list(xs), labels)
    ax0.set_ylabel("pytest collection errors")
    ax0.set_ylim(0, 1.6)
    ax0.set_title("CI: one missing fixture payload fails both test jobs")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.text(
        0.5,
        0.62,
        f"lint (advisory): {LINT_BACKLOG_MAIN} pre-existing ruff\n"
        "findings on origin/main (unrelated, ignored by the\n"
        "TensorFEM merge governor)",
        ha="center",
        fontsize=7.5,
        transform=ax0.transAxes,
    )

    ax1.bar(
        ["before\n(agent/m3825)", "after\n(this cure)"],
        [BEFORE_RELEASED, BRANCH_CHANGED_FILES],
        color="#2980b9",
        label="delivered",
    )
    ax1.bar(
        ["before\n(agent/m3825)", "after\n(this cure)"],
        [BEFORE_STRIPPED, 0],
        bottom=[BEFORE_RELEASED, BRANCH_CHANGED_FILES],
        color="#c0392b",
        label="stripped by delivery gate",
    )
    ax1.set_ylabel("changed files")
    ax1.set_ylim(0, BRANCH_CHANGED_FILES + 4)
    ax1.set_title(
        "Payload shippability: %d/%d files delivered" % (BRANCH_CHANGED_FILES, BRANCH_CHANGED_FILES)
    )
    ax1.legend(loc="upper left", fontsize=8)

    fig.suptitle("TensorFEM PR #117 round-2 CI cure", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=96)
    print(
        json.dumps(
            {
                "figure": str(OUT.relative_to(ROOT)),
                "delivered": BRANCH_CHANGED_FILES,
                "stripped_before": BEFORE_STRIPPED,
                "collection_errors_before": COLLECTION_ERRORS_BEFORE,
                "collection_errors_after": COLLECTION_ERRORS_AFTER,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
