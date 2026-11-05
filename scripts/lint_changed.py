#!/usr/bin/env python3
"""Incremental ruff gate: lint only the Python files a change touches.

Why this exists (see ``docs/lint-ratchet.md``): the tree carries a large
pre-existing ruff backlog -- thousands of ``E701``/``E702`` multi-statement
lines plus hundreds of files that ``ruff format`` would rewrite.  A repo-wide
``ruff check src tests scripts examples`` therefore reports ~4.2k findings and
``ruff format --check`` reports ~560 unformatted files on every run, so the CI
lint job was red no matter what a pull request did, and its red carried no
information about the change under review.

This script keeps that legacy debt grandfathered while gating *new and modified*
code:

* the comparison base is the PR target branch (``origin/main``) or the previous
  pushed commit; the changed set is ``git diff`` against the merge base of that
  base and ``HEAD``, so committed and not-yet-committed work are both covered
  and the identical command can be run locally before pushing;
* untracked, non-ignored ``*.py`` files count as changed -- a brand-new file is
  gated before its first commit, when it is cheapest to fix;
* only ``*.py`` files under ``src/ tests/ scripts/ examples/`` are considered,
  matching the CI invocation this replaces;
* ``ruff check`` and ``ruff format --check`` run on that set only.

``--all`` restores the repo-wide invocation for the day the backlog is cleared.
Once ``--all`` is green, CI can drop ``continue-on-error`` and lint becomes a
blocking gate.

Exit codes: ``0`` clean (including "nothing changed"), ``1`` lint findings,
``2`` usage or environment error (unknown base, ruff missing).  The gate fails
closed on ``2`` so a broken base resolution is visible instead of silently
vouching for the change.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOTS = ("src", "tests", "scripts", "examples")
"""Package roots the CI lint job covers."""

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run ``cmd``; a missing executable reports like a shell would (exit 127)."""
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", f"{exc.strerror or exc}: {cmd[0]}")


def _git(cwd: Path, *args: str) -> str:
    """Run git and return stdout, raising RuntimeError on failure."""
    proc = _run(["git", *args], cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _resolves(cwd: Path, rev: str) -> bool:
    """True when ``rev`` names a commit in this repository."""
    return (
        _run(["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], cwd).returncode == 0
    )


def _merge_base(cwd: Path, base: str, head: str) -> str:
    """Best merge base of ``base`` and ``head``; ``base`` itself if unrelated."""
    proc = _run(["git", "merge-base", base, head], cwd)
    if proc.returncode != 0 or not proc.stdout.strip():
        return base
    return proc.stdout.strip()


def _python_files(cwd: Path, roots: tuple[str, ...], *extra: str) -> list[str]:
    """``*.py`` paths under ``roots`` selected by an ``ls-files``/``diff`` call."""
    out = _git(cwd, *extra, "--", *roots)
    return [line for line in out.splitlines() if line.endswith(".py")]


def _tracked(cwd: Path, roots: tuple[str, ...]) -> list[str]:
    """Tracked files under ``roots`` (used by ``--all``)."""
    return _python_files(cwd, roots, "ls-files", "--cached")


def _changed(cwd: Path, base: str, roots: tuple[str, ...]) -> tuple[list[str], str]:
    """Python files differing from ``base``, including uncommitted/untracked work."""
    head = _git(cwd, "rev-parse", "HEAD").strip()
    merge_base = _merge_base(cwd, base, head)
    # `git diff <merge-base>` compares against the working tree, so staged and
    # unstaged edits are included and no extra HEAD comparison is needed.
    names = set(_python_files(cwd, roots, "diff", "--name-only", "--diff-filter=ACMR", merge_base))
    names.update(_python_files(cwd, roots, "ls-files", "--others", "--exclude-standard"))
    return sorted(names), merge_base


def _existing(cwd: Path, files: list[str]) -> list[str]:
    """Drop paths that were deleted before the gate ran."""
    return [f for f in files if (cwd / f).is_file()]


def _ruff_command(explicit: str | None, cwd: Path) -> list[str]:
    """Resolve the ruff executable, failing closed when it is unavailable."""
    candidates: list[list[str]] = []
    if explicit:
        candidates.append(shlex.split(explicit))
    else:
        found = shutil.which("ruff")
        if found:
            candidates.append([found])
        candidates.append([sys.executable, "-m", "ruff"])
    for cmd in candidates:
        if _run([*cmd, "--version"], cwd).returncode == 0:
            return cmd
    raise RuntimeError("ruff is not available (install the dev extra: pip install -e '.[dev]')")


def _execute(cmd: list[str], cwd: Path) -> dict[str, object]:
    """Run one ruff command, echoing the command plus its output."""
    print(f"$ {shlex.join(cmd)}")
    proc = _run(cmd, cwd)
    output = (proc.stdout + proc.stderr).rstrip()
    if output:
        print(output)
    print(f"  -> exit {proc.returncode}")
    return {"rule": cmd[1], "command": cmd, "returncode": proc.returncode, "output": output}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        default=os.environ.get("LINT_CHANGED_BASE") or "HEAD^",
        help=(
            "comparison base (default: $LINT_CHANGED_BASE or HEAD^). In CI pass the PR "
            "target branch (origin/main) or the previous pushed commit."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="lint every tracked Python file in the roots (pre-ratchet behaviour)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="repository root to lint from (default: current directory)",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help=f"package roots to restrict to (default: {' '.join(DEFAULT_ROOTS)})",
    )
    parser.add_argument(
        "--ruff",
        default=None,
        help="ruff command to use (default: the `ruff` on PATH, else `python -m ruff`)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write a JSON summary here (files gated, per-command exit codes)",
    )
    parser.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="treat an empty change set as an error instead of a pass",
    )
    args = parser.parse_args(argv)

    cwd = Path(args.root).resolve()
    if not cwd.is_dir():
        print(f"lint_changed: --root {cwd} is not a directory", file=sys.stderr)
        return EXIT_ERROR
    roots = tuple(args.paths)

    if args.all:
        files = _existing(cwd, sorted(set(_tracked(cwd, roots))))
        base = merge_base = None
        mode = "all"
    else:
        if not _resolves(cwd, args.base):
            print(
                f"lint_changed: comparison base {args.base!r} does not resolve to a commit; "
                "pass --base explicitly (a CI checkout needs fetch-depth: 0)",
                file=sys.stderr,
            )
            return EXIT_ERROR
        changed, merge_base = _changed(cwd, args.base, roots)
        files = _existing(cwd, changed)
        base = args.base
        mode = "changed"

    print(f"lint_changed: mode={mode} base={base} merge_base={merge_base}")
    if not files:
        print("lint_changed: no Python files to lint")
        if args.fail_on_empty:
            print("lint_changed: --fail-on-empty set but the change set is empty", file=sys.stderr)
            return EXIT_ERROR
    else:
        print(f"lint_changed: gating {len(files)} file(s)")
        for name in files:
            print(f"  {name}")

    try:
        ruff = _ruff_command(args.ruff, cwd)
    except RuntimeError as exc:
        print(f"lint_changed: {exc}", file=sys.stderr)
        return EXIT_ERROR
    version = _run([*ruff, "--version"], cwd).stdout.strip()

    results: list[dict[str, object]] = []
    if files:
        results.append(_execute([*ruff, "check", *files], cwd))
        results.append(_execute([*ruff, "format", "--check", *files], cwd))
    failed = [r for r in results if r["returncode"] != 0]

    if args.report is not None:
        payload = {
            "mode": mode,
            "base": base,
            "merge_base": merge_base,
            "ruff": version,
            "files": files,
            "results": results,
            "clean": not failed,
        }
        args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"lint_changed: report written to {args.report}")

    if failed:
        names = ", ".join(str(r["rule"]) for r in failed)
        print(f"lint_changed: FAIL ({names}) -- fix the files listed above")
        return EXIT_FINDINGS
    print("lint_changed: OK")
    return EXIT_CLEAN


if __name__ == "__main__":
    raise SystemExit(main())
