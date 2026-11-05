#!/usr/bin/env python3
"""Emit the evidence record for the change-scoped lint gate.

The CI ``lint`` job runs ``ruff check`` / ``ruff format --check`` over the
Python files the change touches (``scripts/lint_scope.sh``).  This script
measures exactly what the job measures, and additionally re-measures the same
files as they exist at the base revision, so the record shows the gate moving
from red to green instead of asserting it.

Run from the repository root::

    python scripts/lint_gate_evidence.py --out artifacts/lint-gate/evidence.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINTED_TREES = ("src", "tests", "scripts", "examples", "benchmarks")


def _run(args, cwd=REPO):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def _git(*args):
    result = _run(["git", *args])
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _rev(ref):
    result = _run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"])
    return result.stdout.strip() if result.returncode == 0 else ""


def _scope(base):
    """Python files the gate lints for this change (empty when nothing changed)."""
    result = _run(["bash", str(REPO / "scripts" / "lint_scope.sh"), base])
    if result.returncode != 0:
        raise SystemExit(f"lint_scope.sh failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line]


def _ruff_check(paths, cwd=REPO):
    """(error_count, statistics) for ``ruff check`` over ``paths``."""
    if not paths:
        return 0, {}
    stats = _run(["ruff", "check", "--statistics", *paths], cwd=cwd).stdout
    counts = {}
    for line in stats.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            counts[parts[1]] = int(parts[0])
    return sum(counts.values()), counts


def _ruff_format_unformatted(paths, cwd=REPO):
    """Files ``ruff format --check`` rejects among ``paths``."""
    if not paths:
        return []
    result = _run(["ruff", "format", "--check", *paths], cwd=cwd)
    if result.returncode == 0:
        return []
    return [
        line.split("Would reformat: ", 1)[1].strip()
        for line in result.stdout.splitlines()
        if "Would reformat: " in line
    ]


def _materialise(before, files):
    """Write the pre-change content of ``files`` into a scratch tree."""
    tmp = Path(tempfile.mkdtemp(prefix="lint-gate-before-"))
    shutil.copy(REPO / "pyproject.toml", tmp / "pyproject.toml")
    present = []
    for rel in files:
        blob = _run(["git", "show", f"{before}:{rel}"])
        if blob.returncode != 0:
            continue  # file did not exist before this change
        content = blob.stdout
        target = tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        present.append(rel)
    return tmp, present


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", default="", help="diff base for the lint scope (default: origin/main)"
    )
    parser.add_argument(
        "--before",
        default="",
        help="revision holding the pre-change content of the scoped files "
        "(default: the parent of HEAD)",
    )
    parser.add_argument("--out", default="artifacts/lint-gate/evidence.json")
    args = parser.parse_args(argv)

    base = args.base or _rev("origin/main")
    if not base:
        raise SystemExit("cannot resolve origin/main; pass --base")
    base = _rev(base) or base
    before = args.before or _rev("HEAD~1") or base

    scope = _scope(base)
    current_errors, current_stats = _ruff_check(scope)
    current_unformatted = _ruff_format_unformatted(scope)

    tmp, before_present = _materialise(before, scope)
    try:
        before_errors, before_stats = _ruff_check(before_present, cwd=tmp)
        before_unformatted = _ruff_format_unformatted(before_present, cwd=tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    whole_tree_errors, whole_tree_stats = _ruff_check(list(LINTED_TREES))
    whole_tree_unformatted = _ruff_format_unformatted(list(LINTED_TREES))

    record = {
        "gate": "ci.yml :: lint (advisory) -- ruff check / ruff format --check",
        "scope_rule": "Python files changed vs the base revision under " + ", ".join(LINTED_TREES),
        "base_revision": base,
        "before_revision": before,
        "head_revision": _git("rev-parse", "HEAD").strip(),
        "changed_python_files": scope,
        "changed_files_before_this_change": {
            "present": before_present,
            "ruff_check_errors": before_errors,
            "ruff_check_statistics": before_stats,
            "ruff_format_unformatted": before_unformatted,
        },
        "changed_files_at_head": {
            "ruff_check_errors": current_errors,
            "ruff_check_statistics": current_stats,
            "ruff_format_unformatted": current_unformatted,
        },
        "whole_tree_backlog": {
            "ruff_check_errors": whole_tree_errors,
            "ruff_check_statistics": whole_tree_stats,
            "ruff_format_unformatted_count": len(whole_tree_unformatted),
        },
        "verdict": {
            "changed_files_lint_clean": current_errors == 0,
            "changed_files_format_clean": not current_unformatted,
            "gate_was_red_before_this_change": before_errors > 0 or bool(before_unformatted),
            "gate_is_green_at_head": current_errors == 0 and not current_unformatted,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record["verdict"], indent=2))
    print(
        json.dumps(record["changed_files_before_this_change"]["ruff_format_unformatted"], indent=2)
    )
    print(f"evidence -> {out}")
    return 0 if record["verdict"]["gate_is_green_at_head"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
