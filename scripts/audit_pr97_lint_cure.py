"""Audit the lint state of the files owned by PR #97 (TensorFEM, branch agent/m3593).

Why this script exists
----------------------
PR #97 ("abaqus_io: fail-closed keyword gate") was reported red on the advisory
``lint (advisory)`` job.  The repository carries a large *pre-existing* ruff
backlog, so "the lint job is red" says nothing about whether the pull request
itself is the cause.  This script makes the attribution machine-checkable:

* it measures the ruff state of the files the pull request actually touches
  (``check`` + ``format --check``), and
* it measures the same two numbers for the whole tree, both at the pull
  request head and at its parent commit, so the *increment* attributable to
  the pull request is explicit and cannot be hand-waved.

Usage
-----
    python scripts/audit_pr97_lint_cure.py            # head tree only
    python scripts/audit_pr97_lint_cure.py --baseline # head + parent commit

The parent commit is materialised in a throw-away ``git worktree`` so the head
tree under test is never disturbed.  Outputs:

    artifacts/pr97_lint_cure/audit.json
    artifacts/pr97_lint_cure/lint_cure.png
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "pr97_lint_cure"

#: The Python files this pull request adds or modifies (see `git show --stat`).
PR_FILES = (
    "src/tensorfem/abaqus_io.py",
    "tests/test_abaqus_io.py",
    "scripts/audit_abaqus_keyword_gate.py",
)

#: Trees the CI lint job feeds to ruff.
LINT_TARGETS = ("src", "tests", "scripts", "examples")

_COUNT_RE = re.compile(r"Found (\d+) error")
_REFORMAT_RE = re.compile(r"Would reformat")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def _ruff_version() -> str:
    return _run([sys.executable, "-m", "ruff", "--version"], REPO_ROOT).stdout.strip()


def _check(targets: list[str], cwd: Path) -> dict[str, object]:
    """Return the finding count and the raw tail of ``ruff check``."""
    proc = _run([sys.executable, "-m", "ruff", "check", *targets], cwd)
    match = _COUNT_RE.search(proc.stdout + proc.stderr)
    findings = int(match.group(1)) if match else 0
    return {
        "exit_code": proc.returncode,
        "findings": findings,
        "clean": proc.returncode == 0,
        "tail": (proc.stdout + proc.stderr).strip().splitlines()[-3:],
    }


def _format(targets: list[str], cwd: Path) -> dict[str, object]:
    """Return how many files ``ruff format --check`` would reformat."""
    proc = _run([sys.executable, "-m", "ruff", "format", "--check", *targets], cwd)
    lines = (proc.stdout + proc.stderr).splitlines()
    unformatted = sum(1 for line in lines if _REFORMAT_RE.search(line))
    return {
        "exit_code": proc.returncode,
        "unformatted_files": unformatted,
        "clean": proc.returncode == 0,
    }


def _measure(tree: Path, label: str) -> dict[str, object]:
    pr_files = [f for f in PR_FILES if (tree / f).exists()]
    return {
        "tree": label,
        "pr_files_present": len(pr_files),
        "pr_files": {
            name: {"check": _check([name], tree), "format": _format([name], tree)}
            for name in pr_files
        },
        "tree_check": _check(list(LINT_TARGETS), tree),
        "tree_format": _format(list(LINT_TARGETS), tree),
    }


def _parent_worktree() -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    holder: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="tensorfem-base-")
    path = Path(holder.name) / "tree"
    _run(["git", "worktree", "add", "--detach", str(path), "HEAD~1"], REPO_ROOT)
    if not path.exists():
        holder.cleanup()
        raise RuntimeError("could not materialise the parent commit as a worktree")
    return path, holder


def _plot(report: dict[str, object], out_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is a dev extra
        return False

    head = report["head"]
    labels = [
        "PR-owned files\n(ruff check)",
        "PR-owned files\n(format)",
        "whole tree\n(ruff check)",
    ]
    after = [
        sum(int(v["check"]["findings"]) for v in head["pr_files"].values()),
        sum(int(v["format"]["unformatted_files"]) for v in head["pr_files"].values()),
        int(head["tree_check"]["findings"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    base = report.get("baseline")
    if base:
        before = [
            sum(int(v["check"]["findings"]) for v in base["pr_files"].values()),
            sum(int(v["format"]["unformatted_files"]) for v in base["pr_files"].values()),
            int(base["tree_check"]["findings"]),
        ]
        idx = range(len(labels))
        axes[0].bar(
            [i - 0.2 for i in idx], before, width=0.4, label="parent commit", color="#b0413e"
        )
        axes[0].bar(
            [i + 0.2 for i in idx], after, width=0.4, label="PR head (cured)", color="#2f7f6f"
        )
        axes[0].set_xticks(list(idx))
        axes[0].legend()
    else:
        axes[0].bar(labels, after, color="#2f7f6f")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("count")
    axes[0].set_title("Ruff findings: attribution by scope", fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    names = list(head["pr_files"])
    check_counts = [int(head["pr_files"][n]["check"]["findings"]) for n in names]
    fmt_counts = [int(head["pr_files"][n]["format"]["unformatted_files"]) for n in names]
    short = [Path(n).name for n in names]
    axes[1].barh(short, check_counts, color="#2f7f6f", label="check findings")
    axes[1].barh(short, fmt_counts, color="#d9a03f", alpha=0.65, label="files needing format")
    axes[1].set_xlabel("count (0 == clean)")
    axes[1].set_title("PR-owned files after cure", fontsize=10)
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("PR #97 lint cure audit (ruff 0.15, line-length 100)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also measure the parent commit in a throw-away worktree",
    )
    parser.add_argument("--json", type=Path, default=ARTIFACT_DIR / "audit.json")
    parser.add_argument("--png", type=Path, default=ARTIFACT_DIR / "lint_cure.png")
    args = parser.parse_args(argv)

    report: dict[str, object] = {"ruff": _ruff_version(), "pr_files": list(PR_FILES)}
    report["head"] = _measure(REPO_ROOT, "head")

    if args.baseline:
        base, holder = _parent_worktree()
        try:
            report["baseline"] = _measure(base, "parent")
        finally:
            _run(["git", "worktree", "remove", "--force", str(base)], REPO_ROOT)
            holder.cleanup()

    head_pr_check = sum(int(v["check"]["findings"]) for v in report["head"]["pr_files"].values())
    head_pr_fmt = sum(
        int(v["format"]["unformatted_files"]) for v in report["head"]["pr_files"].values()
    )
    report["verdict"] = {
        "pr_files_ruff_clean": head_pr_check == 0 and head_pr_fmt == 0,
        "pr_file_check_findings": head_pr_check,
        "pr_file_unformatted": head_pr_fmt,
    }
    if "baseline" in report:
        base = report["baseline"]
        report["verdict"]["tree_findings_increment"] = int(
            report["head"]["tree_check"]["findings"]
        ) - int(base["tree_check"]["findings"])
        report["verdict"]["tree_unformatted_increment"] = int(
            report["head"]["tree_format"]["unformatted_files"]
        ) - int(base["tree_format"]["unformatted_files"])

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    plotted = _plot(report, args.png)
    report["figure"] = str(args.png) if plotted else None
    args.json.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {args.json}")
    shutil.os.path.exists(args.png) and print(f"wrote {args.png}")
    return 0 if report["verdict"]["pr_files_ruff_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
