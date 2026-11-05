#!/usr/bin/env python3
"""Diff-scoped ruff gate for the CI ``lint (advisory)`` job.

The tree carries a large pre-existing ruff backlog (thousands of ``ruff
check`` findings and hundreds of files ``ruff format`` would rewrite), so a
repo-wide scan fails on *every* commit -- on ``main`` as much as on a pull
request -- and its red carries no information about the change under review.
This gate therefore lints and format-checks only the Python files the change
touches, and *reports* the grandfathered tree-wide backlog without gating on
it, so the number stays visible and can only be paid down deliberately.

Exit codes: 0 clean, 1 violations found, 2 environment error (fail closed --
a missing ruff, a missing git history or an unresolvable base must never be
read as "no findings").
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Directories the CI job has always covered; used for ``--all`` and for the
#: advisory backlog measurement.
LINT_ROOTS = ("src", "tests", "scripts", "examples")

_FOUND_RE = re.compile(r"Found (\d+) error")
_REFORMAT_RE = re.compile(r"(\d+) files? would be reformatted")


def _run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _git(repo, *args):
    return _run(["git", *args], repo)


def resolve_base(repo, base):
    """Return the commit the diff is taken against.

    The merge base is preferred so that a branch that has merged ``main`` is
    only judged on what it itself introduces.  Falls back to the plain commit
    when no merge base exists (e.g. a shallow or unrelated history), and
    raises ``ValueError`` when the ref is unusable.
    """
    if not base:
        raise ValueError("no comparison base given (--base or $LINT_BASE)")
    status, out = _git(repo, "merge-base", base, "HEAD")
    if status == 0 and out:
        return out.strip().splitlines()[0]
    status, out = _git(repo, "rev-parse", "--verify", f"{base}^{{commit}}")
    if status != 0:
        raise ValueError(f"cannot resolve comparison base {base!r}: {out}")
    return out.strip()


def changed_python_files(repo, merge_base, include_worktree=False):
    """Python files touched between ``merge_base`` and the worktree."""
    status, out = _git(
        repo, "diff", "--name-only", "--diff-filter=ACMRT", merge_base, "HEAD", "--", "*.py"
    )
    if status != 0:
        raise ValueError(f"git diff failed: {out}")
    names = [line.strip() for line in out.splitlines() if line.strip()]
    if include_worktree:
        for extra in (
            ("diff", "--name-only", "--", "*.py"),
            ("ls-files", "--others", "--exclude-standard", "--", "*.py"),
        ):
            status, out = _git(repo, *extra)
            if status != 0:
                raise ValueError(f"git {' '.join(extra)} failed: {out}")
            names += [line.strip() for line in out.splitlines() if line.strip()]
    seen, ordered = set(), []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    # A file deleted in the worktree cannot be linted.
    return [name for name in ordered if (Path(repo) / name).is_file()]


def tracked_python_files(repo):
    """Tracked Python files under the roots the CI job has always covered."""
    status, out = _git(repo, "ls-files", "--", "*.py")
    if status != 0:
        raise ValueError(f"git ls-files failed: {out}")
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return [name for name in names if name.split("/", 1)[0] in LINT_ROOTS]


def measure_backlog(repo, ruff):
    """Count the grandfathered tree-wide backlog (never gates)."""
    backlog = {"check": None, "format": None}
    status, out = _run([ruff, "check", *LINT_ROOTS, "--statistics"], repo)
    match = _FOUND_RE.search(out)
    backlog["check"] = int(match.group(1)) if match else 0
    status, out = _run([ruff, "format", "--check", *LINT_ROOTS], repo)
    match = _REFORMAT_RE.search(out)
    backlog["format"] = int(match.group(1)) if match else 0
    return backlog


def _write_report(path, report):
    if not path:
        return
    target = Path(path)
    if target.parent and str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lint and format-check the Python files a change touches."
    )
    parser.add_argument("--repo", default=".", help="repository root (default: cwd)")
    parser.add_argument(
        "--base",
        help="commit/ref the change is diffed against (default: $LINT_BASE)",
    )
    parser.add_argument(
        "--all", action="store_true", help="gate the whole tracked tree instead of the diff"
    )
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="also include staged, unstaged and untracked Python files",
    )
    parser.add_argument("--report", help="write a JSON report to this path")
    parser.add_argument(
        "--measure-backlog",
        action="store_true",
        help="also record the grandfathered tree-wide backlog in the report",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    ruff = os.environ.get("RUFF") or shutil.which("ruff")
    report = {
        "repo": str(repo),
        "mode": "all" if args.all else "changed",
        "base": args.base or os.environ.get("LINT_BASE"),
        "merge_base": None,
        "files": [],
        "results": [],
        "ruff": None,
        "backlog": None,
        "clean": False,
    }

    if ruff is None:
        print("lint_scope: ruff is not installed (fail closed)", file=sys.stderr)
        _write_report(args.report, report)
        return 2
    _, version = _run([ruff, "--version"], repo)
    report["ruff"] = version

    try:
        if args.all:
            files = tracked_python_files(repo)
        else:
            base = report["base"]
            report["merge_base"] = resolve_base(repo, base)
            files = changed_python_files(repo, report["merge_base"], args.worktree)
    except ValueError as exc:
        print(f"lint_scope: {exc}", file=sys.stderr)
        _write_report(args.report, report)
        return 2
    report["files"] = files

    if not files:
        print("lint_scope: no Python files in scope -- nothing to lint")
        report["clean"] = True
        _write_report(args.report, report)
        return 0

    failures = 0
    for rule, command in (
        ("check", [ruff, "check", *files]),
        ("format", [ruff, "format", "--check", *files]),
    ):
        status, out = _run(command, repo)
        print(f"$ {' '.join(command)}\n{out or '(no output)'}")
        report["results"].append(
            {"rule": rule, "command": command, "returncode": status, "output": out}
        )
        failures += status != 0

    if args.measure_backlog:
        report["backlog"] = measure_backlog(repo, ruff)
        print(f"lint_scope: grandfathered backlog {json.dumps(report['backlog'])}")

    report["clean"] = failures == 0
    _write_report(args.report, report)
    print(
        f"lint_scope: {len(files)} file(s) in scope, "
        + ("clean" if report["clean"] else f"{failures} check(s) failed")
    )
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
