#!/usr/bin/env python3
"""Ruff formatting gate: ratchet the *changed* Python files.

Why this exists
---------------
The tree carries a large pre-existing ``ruff format`` backlog (600+ files), so a
tree-wide ``ruff format --check`` can never be green and therefore carries no
signal: the CI step is red for every branch, including branches that only touch
documentation.  This gate keeps the signal where it matters -- the code a change
actually touches -- without forcing an unrelated repo-wide reformat:

* a Python file **added** by the change must be format-clean;
* a file that was **already format-clean at the merge base** must stay clean;
* a file that was **already dirty at the merge base** is only warned about;
  formatting it is a burn-down item, not a merge blocker, so touching legacy
  code never drags a huge reformat diff into an otherwise small change.

Exit status: 0 when no *new* formatting debt was introduced, 1 otherwise.

``--all`` switches to a plain tree-wide check; combined with ``--count`` it
exits 0 and prints the remaining backlog size (used by the advisory burn-down CI
step, never as a gate).

Usage::

    python scripts/lint_gate.py --base origin/main
    python scripts/lint_gate.py --all --count
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Directories the CI lint job is responsible for (mirrors the ruff invocation).
SCOPE = ("src", "tests", "scripts", "examples")

# Mirror [tool.ruff] in pyproject.toml.  Passed explicitly so that the checks we
# run against *extracted* base revisions (which live outside the tree and would
# otherwise fall back to ruff's defaults) use the same settings as in-tree files.
LINE_LENGTH = 100
TARGET_VERSION = "py311"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=root)


def _repo_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise SystemExit(f"not inside a git work tree: {proc.stderr.strip()}")
    return Path(proc.stdout.strip())


def _ruff(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    exe = shutil.which("ruff")
    cmd = [exe] if exe else [sys.executable, "-m", "ruff"]
    return subprocess.run([*cmd, *args], capture_output=True, text=True, cwd=root)


def _format_check(root: Path, target: str | Path) -> bool:
    """True when ``target`` is already ruff-format clean."""
    proc = _ruff(
        root,
        "format",
        "--check",
        "--no-cache",
        "--line-length",
        str(LINE_LENGTH),
        "--target-version",
        TARGET_VERSION,
        str(target),
    )
    if proc.returncode not in (0, 1):
        raise SystemExit(f"ruff format failed on {target}: {proc.stderr.strip()}")
    return proc.returncode == 0


def _resolve_base(root: Path, explicit: str | None) -> tuple[str, str] | tuple[None, None]:
    """Resolve the merge base (commit sha) and the label used to name it."""
    candidates = [explicit] if explicit else ["origin/main", "main", "HEAD~1"]
    for candidate in candidates:
        if not candidate:
            continue
        if _git(root, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}").returncode:
            continue
        merge_base = _git(root, "merge-base", candidate, "HEAD")
        if merge_base.returncode == 0 and merge_base.stdout.strip():
            return merge_base.stdout.strip(), candidate
    return None, None


def _changed_python_files(root: Path, base: str) -> tuple[list[str], set[str]]:
    """Python files inside SCOPE that the change added or modified."""
    tracked = _git(root, "diff", "--name-only", "--diff-filter=ACMRT", base, "--", *SCOPE)
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", *SCOPE)
    added = {line.strip() for line in untracked.stdout.splitlines() if line.strip()}
    changed = {line.strip() for line in tracked.stdout.splitlines() if line.strip()}
    files = sorted(
        rel
        for rel in changed | added
        if rel.endswith(".py") and rel.split("/")[0] in SCOPE and (root / rel).is_file()
    )
    return files, added


def _was_formatted_at_base(root: Path, base: str, rel: str, scratch: Path) -> bool | None:
    """Formatting state of ``rel`` at ``base``; None when the file did not exist."""
    proc = _git(root, "show", f"{base}:{rel}")
    if proc.returncode != 0:
        return None
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    probe = scratch / f"{digest}_{Path(rel).name}"
    probe.write_text(proc.stdout, encoding="utf-8")
    return _format_check(root, probe)


def _scope_targets(root: Path) -> list[str]:
    return [d for d in SCOPE if (root / d).exists()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="ref to ratchet against (default: origin/main)")
    parser.add_argument(
        "--all", action="store_true", help="check the whole scope instead of the changed files"
    )
    parser.add_argument(
        "--count", action="store_true", help="report the backlog size and always exit 0"
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    targets = _scope_targets(root)
    if not targets:
        print("lint_gate: no scope directories present, nothing to check")
        return 0

    if args.all:
        proc = _ruff(
            root,
            "format",
            "--check",
            "--no-cache",
            "--line-length",
            str(LINE_LENGTH),
            "--target-version",
            TARGET_VERSION,
            *targets,
        )
        if proc.returncode not in (0, 1):
            raise SystemExit(f"ruff format failed: {proc.stderr.strip()}")
        pending = [
            line.split(":", 1)[1].strip()
            for line in proc.stdout.splitlines()
            if line.startswith("Would reformat:")
        ]
        print(f"lint_gate: {len(pending)} file(s) would be reformatted across {', '.join(targets)}")
        if args.count:
            return 0
        return 0 if not pending else 1

    base, label = _resolve_base(root, args.base)
    if base is None:
        print(f"lint_gate: no merge base against {args.base or 'origin/main'} — skipping ratchet")
        return 0

    files, added = _changed_python_files(root, base)
    if not files:
        print(
            f"lint_gate: no Python files changed against {label} ({base[:9]}) — nothing to ratchet"
        )
        return 0

    problems: list[str] = []
    warnings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="lint-gate-") as scratch_name:
        scratch = Path(scratch_name)
        for rel in files:
            clean_now = _format_check(root, rel)
            if clean_now:
                continue
            base_state = None if rel in added else _was_formatted_at_base(root, base, rel, scratch)
            if base_state is None:
                problems.append(f"new file is not ruff-formatted: {rel}")
            elif base_state:
                problems.append(f"file became unformatted: {rel} (clean at {label})")
            else:
                warnings.append(f"pre-existing format debt kept as-is: {rel}")

    for line in warnings:
        print(f"lint_gate: WARN {line}")
    for line in problems:
        print(f"lint_gate: FAIL {line}")
    checked = len(files)
    print(
        f"lint_gate: {checked} changed Python file(s) vs {label} ({base[:9]}): "
        f"{len(problems)} new debt, {len(warnings)} pre-existing"
    )
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
