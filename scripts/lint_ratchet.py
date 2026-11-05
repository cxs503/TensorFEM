#!/usr/bin/env python3
"""Ruff lint ratchet for the TensorFEM tree.

Why this exists
---------------
The tree carries a large pre-existing ruff backlog (semicolon-packed
statements, unused imports, star imports and friends).  Gating on a clean tree
today would either block every pull request or force a single unreviewable
commit over hundreds of files, while declaring the check "advisory" only hides
the regression signal -- it does not remove the red.

So the backlog is frozen in ``scripts/lint_baseline.json`` and ruff runs as a
ratchet: pre-existing debt is tolerated, new debt is rejected.  The full-tree
``ruff check`` / ``ruff format --check`` runs stay in CI as advisory steps so
the drain progress stays visible, and the bug-class rule gate
(``E9,F63,F7,F82,F811``) runs on top as an unconditional requirement.

Commands
--------
    python scripts/lint_ratchet.py            # verify the tree against the baseline
    python scripts/lint_ratchet.py --update   # re-freeze the current tree
    python scripts/lint_ratchet.py --summary  # report the frozen debt, always exit 0

Exit status is 0 when the tree is at or below the frozen baseline, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE_PATH = Path("scripts/lint_baseline.json")
LINT_PATHS = ("src", "tests", "scripts", "examples")
SCHEMA = 1


def _ruff() -> str:
    exe = shutil.which("ruff")
    if exe is None:
        raise SystemExit("ruff not found; install the dev extras: pip install -e '.[dev]'")
    return exe


def _run(args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([_ruff(), *args], cwd=root, capture_output=True, text=True)


def _rel(path: str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - files outside the tree are not linted
        return Path(path).as_posix()


def ruff_version(root: Path | None = None) -> str:
    """Version string of the ruff binary that produced the numbers."""
    proc = subprocess.run([_ruff(), "--version"], cwd=root, capture_output=True, text=True)
    return proc.stdout.strip()


def collect_check(root: Path, paths: tuple[str, ...] = LINT_PATHS) -> dict[str, dict[str, int]]:
    """Per-file ruff rule counts, ``{relative path: {rule: count}}``.

    Files without violations are omitted, which keeps the frozen baseline small.
    """
    proc = _run(["check", "--no-cache", "--output-format", "json", *paths], root)
    # ruff exits 1 when it reports violations; anything else is a tool failure.
    if proc.returncode not in (0, 1):
        raise SystemExit(f"ruff check could not run (exit {proc.returncode}):\n{proc.stderr}")
    counts: dict[str, Counter[str]] = {}
    for item in json.loads(proc.stdout or "[]"):
        counts.setdefault(_rel(item["filename"], root), Counter())[item["code"]] += 1
    return {path: dict(sorted(counts[path].items())) for path in sorted(counts)}


def collect_format(root: Path, paths: tuple[str, ...] = LINT_PATHS) -> list[str]:
    """Relative paths of files ``ruff format`` would rewrite.

    ``ruff format --check`` has no stable machine-readable output format, so the
    human line ``Would reformat: <path>`` is parsed; it is the only guaranteed
    field of that output.
    """
    proc = _run(["format", "--check", *paths], root)
    return sorted(
        _rel(line.split("Would reformat: ", 1)[1].strip(), root)
        for line in proc.stdout.splitlines()
        if line.startswith("Would reformat: ")
    )


def snapshot(root: Path | None = None) -> dict:
    """Full lint state of the tree in the baseline file format."""
    root = (root or Path.cwd()).resolve()
    files = collect_check(root)
    totals: Counter[str] = Counter()
    for counts in files.values():
        totals.update(counts)
    return {
        "schema": SCHEMA,
        "ruff_version": ruff_version(root),
        "paths": list(LINT_PATHS),
        "check": {"totals": dict(sorted(totals.items())), "files": files},
        "format": {"unformatted": collect_format(root)},
    }


def regression_problems(current: dict, baseline: dict) -> list[str]:
    """Human readable list of new lint debt (empty means the ratchet holds)."""
    problems: list[str] = []
    if baseline.get("ruff_version") != current.get("ruff_version"):
        problems.append(
            f"ruff version changed: baseline {baseline.get('ruff_version')!r} "
            f"!= current {current.get('ruff_version')!r} -- "
            "re-freeze with --update after reviewing the new violations"
        )
        return problems
    frozen = baseline["check"]["files"]
    for path, counts in current["check"]["files"].items():
        allowed = frozen.get(path, {})
        for rule, count in counts.items():
            budget = allowed.get(rule, 0)
            if count > budget:
                problems.append(f"{path}: {rule} {budget} -> {count} (+{count - budget})")
    unformatted = set(baseline["format"]["unformatted"])
    problems.extend(
        f"{path}: ruff format would rewrite a file that is not in the baseline"
        for path in current["format"]["unformatted"]
        if path not in unformatted
    )
    return problems


def _debt(current: dict) -> tuple[int, int, int]:
    violations = sum(current["check"]["totals"].values())
    return violations, len(current["check"]["files"]), len(current["format"]["unformatted"])


def render_summary(label: str, state: dict) -> str:
    violations, files, unformatted = _debt(state)
    return (
        f"{label}: {violations} ruff violations in {files} files, "
        f"{unformatted} files ruff format would rewrite ({state.get('ruff_version')})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ruff lint ratchet over the frozen backlog.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--update", action="store_true", help="re-freeze the current tree")
    parser.add_argument("--summary", action="store_true", help="report the frozen debt only")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    baseline_path = args.baseline or root / BASELINE_PATH
    current = snapshot(root)

    if args.update:
        baseline_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(render_summary(f"frozen {baseline_path.relative_to(root)}", current))
        return 0

    if not baseline_path.exists():
        print(f"missing baseline {baseline_path}; create it with --update", file=sys.stderr)
        return 1
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    print(render_summary("frozen baseline", baseline))
    print(render_summary("current tree   ", current))

    problems = regression_problems(current, baseline)
    if args.summary:
        return 0
    if problems:
        print(
            f"\nnew lint debt ({len(problems)} item(s)); fix it or justify a --update:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("ratchet OK: no ruff violation or format debt beyond the frozen baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
