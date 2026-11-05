#!/usr/bin/env python3
"""Scope the CI ruff gate to the files a push/PR actually touches.

Why this exists
---------------
The tree carries a large pre-existing ruff backlog, so the whole-tree run
(``ruff check src tests scripts examples``) is red on ``main`` itself: no
feature PR can ever turn that check green, and a red advisory job hides the
one signal that *is* attributable to the PR -- whether the PR adds new lint
debt of its own.  This helper computes the ratchet scope instead: the ``.py``
files this push/PR adds or modifies inside the linted roots (``src``,
``tests``, ``scripts``, ``examples``).  CI then runs ``ruff check`` and
``ruff format --check`` on exactly that list, so:

  * a PR whose own files are clean is green (pre-existing debt is reported
    separately and is not charged to the PR),
  * a PR that adds lint debt stays red -- the gate can only get stricter,
  * burning down the backlog implicates nobody (touching a legacy file puts it
    in scope, so the ratchet never forgives a file someone is already editing).

The zero-list case is green: a docs-only or JSON-only change has nothing to
lint.

Fail-open: if git cannot answer (shallow clone, unknown base sha, detached
weirdness) the helper prints a notice and emits an empty list.  A lint scope
helper must never be the reason a pipeline dies -- the whole-tree informational
step still reports the backlog for humans.

Usage
-----
    python scripts/ci_lint_scope.py --base <sha> --out /tmp/lint-files.txt
    python scripts/ci_lint_scope.py --base <sha>            # list on stdout

Exit status is 0 unless ``--strict`` is given and the scope could not be
resolved from git at all.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LINT_ROOTS = ("src", "tests", "scripts", "examples")
NORM = "0" * 40


def _git(repo: Path, *args: str) -> tuple[bool, str]:
    """Run git in ``repo``; return (ok, stdout-or-error-message)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - git absent
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()
    return True, proc.stdout


def _is_commit(repo: Path, rev: str) -> bool:
    ok, _ = _git(repo, "cat-file", "-e", f"{rev}^{{commit}}")
    return ok


def resolve_base(repo: Path, base: str | None) -> tuple[str | None, str]:
    """Pick the comparison base, newest-first: explicit sha, then HEAD^."""
    if base and base.strip() and base.strip() != NORM:
        cand = base.strip()
        if _is_commit(repo, cand):
            return cand, f"base={cand[:12]}"
        return None, f"base {cand[:12]} not present locally (shallow clone?)"
    ok, _ = _git(repo, "rev-parse", "--verify", "HEAD^")
    if ok:
        return "HEAD^", "no base given, falling back to HEAD^"
    return None, "no base given and HEAD has no parent"


def changed_files(repo: Path, base: str, roots: tuple[str, ...] = LINT_ROOTS) -> list[str]:
    """Added/modified/deleted-agnostic .py files touched since the merge-base."""
    ok, out = _git(
        repo, "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD", "--", "*.py"
    )
    if not ok:
        # A tag/ref that only differs at the merge base edge is rare; retry
        # with the plain two-dot form before giving up.
        ok, out = _git(
            repo, "diff", "--name-only", "--diff-filter=ACMR", base, "HEAD", "--", "*.py"
        )
        if not ok:
            return []
    picked = []
    for line in out.splitlines():
        rel = line.strip()
        if not rel:
            continue
        if not any(rel == r or rel.startswith(r + "/") for r in roots):
            continue
        if (repo / rel).is_file():
            picked.append(rel)
    return sorted(dict.fromkeys(picked))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base", default="", help="comparison sha (pull_request.base.sha / push before)"
    )
    ap.add_argument("--repo", default=".", help="repository root (default: cwd)")
    ap.add_argument(
        "--root", action="append", default=None, help=f"linted root (default: {LINT_ROOTS})"
    )
    ap.add_argument("--out", default="", help="write the newline-separated file list here")
    ap.add_argument(
        "--strict", action="store_true", help="exit 2 when the scope cannot be resolved"
    )
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    roots = tuple(args.root) if args.root else LINT_ROOTS
    base, why = resolve_base(repo, args.base)
    if base is None:
        print(f"::warning title=ruff scope::could not resolve a base ({why}); nothing scoped")
        files: list[str] = []
    else:
        files = changed_files(repo, base, roots)
        scope = "changed" if files else "no changed python file"
        print(f"::notice title=ruff scope::{why}, {len(files)} file(s) in scope ({scope})")

    payload = "".join(f + "\n" for f in files)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    if base is None and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
