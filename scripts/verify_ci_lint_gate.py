#!/usr/bin/env python3
"""Local mirror of the two-tier lint gate in .github/workflows/ci.yml.

Why this exists: the CI lint job used to run ``ruff check`` /
``ruff format --check`` over the whole tree, which is red for *every* commit
because of a pre-existing backlog (~4.3k findings, mostly semicolon-packed
E701/E702 statements, plus ~620 files older than ``ruff format``).  PR #125 was
reported ``lint (advisory)=failure`` although its own diff was lint-clean.

The gate is now two-tier (see ci.yml).  This script reproduces both tiers
locally so a branch can be checked before pushing:

* verdict tier  -> changed .py files only (default:  diff vs ``--base``);
* report tier   -> whole-tree counters, informational.

Usage::

    python scripts/verify_ci_lint_gate.py --base 7accc4b
    python scripts/verify_ci_lint_gate.py --base origin/main --json out.json

Exit code is 0 when the verdict tier is green, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

TREES = ("src", "tests", "scripts", "examples")
SCOPE = re.compile(r"^(%s)/" % "|".join(TREES))


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def changed_python_files(root: Path, base: str) -> list[str]:
    """Same filter as the ``Collect changed Python files`` CI step."""
    rc, out = run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", base, "HEAD", "--", "*.py"],
        root,
    )
    if rc != 0:
        raise SystemExit(f"git diff against {base!r} failed:\n{out}")
    return [ln for ln in out.splitlines() if ln and SCOPE.match(ln)]


def ruff_available() -> str:
    if not shutil.which("ruff"):
        raise SystemExit("ruff not installed (pip install 'ruff>=0.6')")
    return run(["ruff", "--version"], Path.cwd())[1].strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="HEAD~1", help="base commit for the diff (default: HEAD~1)")
    ap.add_argument("--root", default=".", help="repository root (default: cwd)")
    ap.add_argument("--json", default="", help="write the machine-readable summary here")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    version = ruff_available()
    files = changed_python_files(root, args.base)

    rc_check, out_check = (0, "(no changed python files)")
    rc_fmt, out_fmt = (0, "(no changed python files)")
    if files:
        rc_check, out_check = run(["ruff", "check", "--no-cache", *files], root)
        rc_fmt, out_fmt = run(["ruff", "format", "--check", *files], root)

    rc_repo, out_repo = run(["ruff", "check", *TREES, "--statistics", "--no-cache"], root)
    rc_repo_fmt, out_repo_fmt = run(["ruff", "format", "--check", *TREES], root)
    backlog = re.search(r"Found (\d+) errors?", out_repo)
    unformatted = re.search(r"(\d+) files? would be reformatted", out_repo_fmt)

    verdict_ok = rc_check == 0 and rc_fmt == 0
    summary = {
        "ruff": version,
        "base": args.base,
        "changed_python_files": files,
        "tier_changed": {
            "ruff_check_rc": rc_check,
            "ruff_format_rc": rc_fmt,
            "ok": verdict_ok,
        },
        "tier_repo_informational": {
            "ruff_check_errors": int(backlog.group(1)) if backlog else 0,
            "files_needing_format": int(unformatted.group(1)) if unformatted else 0,
            "ruff_check_rc": rc_repo,
            "ruff_format_rc": rc_repo_fmt,
        },
    }

    print(f"ruff: {version}")
    print(f"diff base: {args.base}")
    print(f"changed python files ({len(files)}):")
    for f in files:
        print(f"  {f}")
    print("\n--- tier 1: changed files (job verdict) ---")
    print(out_check.strip())
    print(out_fmt.strip())
    print("\n--- tier 2: repo backlog (informational) ---")
    print(
        "ruff check findings: %s | files needing format: %s"
        % (
            summary["tier_repo_informational"]["ruff_check_errors"],
            summary["tier_repo_informational"]["files_needing_format"],
        )
    )
    print("\nVERDICT:", "GREEN" if verdict_ok else "RED")

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
