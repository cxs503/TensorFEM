#!/usr/bin/env python3
"""Ratchet gate for the repository-wide ruff backlog.

Why this exists
---------------
``main`` carries a large *pre-existing* ruff backlog (4268 findings in 344 of
546 python files, plus 619 files that ``ruff format`` would rewrite).  A
whole-tree ``ruff check`` therefore exits non-zero on every commit, PR #131
included, and the CI job could only ever be red.  Rewriting the whole tree is
a separate, much larger change; the gate must not pretend the backlog is gone
either.

The gate implemented here is a *ratchet*: for every python file touched by the
change set, the number of ruff findings may not grow and no newly unformatted
code may be introduced.  The whole-tree backlog is still reported (advisory) so
the number is visible and can be burned down deliberately.

Rules
-----
For each changed ``*.py`` file (renames are compared against their old path):

* ``findings(head) <= findings(base)`` -- no new findings.  A brand new file
  therefore has to be clean (``base == 0``).
* ``format_clean(head)`` unless the file was *already* unformatted at base --
  no newly added unformatted code.

An unusable base commit (shallow clone, brand new ref, ``workflow_dispatch``)
skips the ratchet instead of failing: the check must never be red because of
missing history.

Usage
-----
    python scripts/lint_ratchet.py --base origin/main
    python scripts/lint_ratchet.py --base "$BASE_SHA" --summary artifacts/lint-ratchet.json

Exit status is 0 when the change set adds no debt, 1 when it does.
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

FINDING_RE = re.compile(r"^.+:\d+:\d+: [A-Z]+[0-9]+")
DEFAULT_ADVISORY_PATHS = ("src", "tests", "scripts", "examples")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], root)


def repo_root() -> Path:
    res = _git(Path.cwd(), "rev-parse", "--show-toplevel")
    if res.returncode != 0:
        sys.exit(f"lint-ratchet: not inside a git checkout: {res.stderr.strip()}")
    return Path(res.stdout.strip())


def changed_python_files(root: Path, base: str, head: str) -> list[tuple[str, str | None]]:
    """Return ``(head_path, base_path or None)`` pairs for changed python files."""
    res = _git(root, "diff", "--name-status", "-M", "--diff-filter=d", base, head, "--", "*.py")
    if res.returncode != 0:
        raise RuntimeError(f"git diff {base}..{head} failed: {res.stderr.strip()}")
    pairs: list[tuple[str, str | None]] = []
    for line in res.stdout.splitlines():
        fields = line.split("\t")
        status = fields[0]
        if status.startswith("R") and len(fields) >= 3:
            pairs.append((fields[2], fields[1]))
        elif len(fields) >= 2:
            # A = added (no base version); M/T/C = the old path is the same path.
            pairs.append((fields[1], fields[1] if status[0] != "A" else None))
    return pairs


def _ruff_config(root: Path) -> str:
    return str(root / "pyproject.toml")


def count_findings(target: Path, root: Path) -> int:
    """Number of ruff findings in ``target`` (ruff exit 2 is a hard error)."""
    res = _run(
        [
            "ruff",
            "check",
            "--no-cache",
            "--config",
            _ruff_config(root),
            "--output-format=concise",
            str(target),
        ],
        root,
    )
    if res.returncode not in (0, 1):
        raise RuntimeError(f"ruff check failed on {target}: {res.stderr.strip()}")
    return sum(1 for line in res.stdout.splitlines() if FINDING_RE.match(line))


def format_clean(target: Path, root: Path) -> bool:
    res = _run(
        ["ruff", "format", "--check", "--config", _ruff_config(root), str(target)],
        root,
    )
    if res.returncode not in (0, 1):
        raise RuntimeError(f"ruff format failed on {target}: {res.stderr.strip()}")
    return res.returncode == 0


def _materialise(root: Path, base: str, path: str, scratch: Path) -> Path | None:
    """Write the ``base`` revision of ``path`` into ``scratch``; None if absent."""
    res = _git(root, "show", f"{base}:{path}")
    if res.returncode != 0:
        return None
    target = scratch / Path(path).name
    target.write_text(res.stdout)
    return target


def backlog_snapshot(root: Path, paths: list[str]) -> dict[str, object]:
    """Whole-tree backlog, reported for visibility only -- never fails the gate."""
    existing = [p for p in paths if (root / p).exists()]
    if not existing:
        return {"paths": paths, "findings": None, "unformatted": None}
    check = _run(["ruff", "check", "--no-cache", *existing, "--output-format=concise"], root)
    fmt = _run(["ruff", "format", "--check", *existing], root)
    findings = sum(1 for line in check.stdout.splitlines() if FINDING_RE.match(line))
    unformatted = sum(1 for line in fmt.stdout.splitlines() if line.startswith("Would reformat"))
    return {"paths": existing, "findings": findings, "unformatted": unformatted}


def evaluate(root: Path, base: str, head: str) -> dict[str, object]:
    pairs = changed_python_files(root, base, head)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="lint-ratchet-") as tmp:
        scratch = Path(tmp)
        for head_path, base_path in pairs:
            head_target = root / head_path
            if not head_target.is_file():
                continue  # deleted in head: nothing to ratchet
            base_target = (
                _materialise(root, base, base_path, scratch) if base_path is not None else None
            )
            head_findings = count_findings(head_target, root)
            base_findings = count_findings(base_target, root) if base_target else 0
            head_fmt = format_clean(head_target, root)
            base_fmt = format_clean(base_target, root) if base_target else True
            reasons = []
            if head_findings > base_findings:
                reasons.append(f"findings {base_findings} -> {head_findings}")
            if not head_fmt and base_fmt:
                reasons.append("new file/formatting is not `ruff format` clean")
            rows.append(
                {
                    "file": head_path,
                    "base_file": base_path,
                    "base_findings": base_findings,
                    "head_findings": head_findings,
                    "base_format_clean": base_fmt,
                    "head_format_clean": head_fmt,
                    "regressions": reasons,
                }
            )
    return {"base": base, "head": head, "files": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--base", default="", help="base commit/ref to ratchet against")
    parser.add_argument("--head", default="HEAD", help="head commit/ref (default: HEAD)")
    parser.add_argument("--summary", default="", help="write a JSON summary of the ratchet here")
    parser.add_argument(
        "--advisory-paths",
        nargs="*",
        default=list(DEFAULT_ADVISORY_PATHS),
        help="paths whose whole-tree backlog is reported (advisory only)",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    if shutil.which("ruff") is None:
        print("lint-ratchet: ruff not installed, skipping (install the dev extra).")
        return 0

    base = args.base.strip()
    have_base = bool(base) and _git(root, "cat-file", "-e", f"{base}^{{commit}}").returncode == 0
    if not have_base:
        print(
            "lint-ratchet: no usable base commit "
            f"(base={base!r}); nothing to diff against, skipping the ratchet."
        )
        return 0

    result = evaluate(root, base, args.head)
    rows = result["files"]
    assert isinstance(rows, list)

    print(f"lint-ratchet: {base}..{args.head} -> {len(rows)} changed python file(s)")
    header = f"{'file':<52} {'findings':>16} {'format':>13}  verdict"
    print(header)
    print("-" * len(header))
    regressions = 0
    for row in rows:
        assert isinstance(row, dict)
        verdict = "ok" if not row["regressions"] else "REGRESSION: " + "; ".join(row["regressions"])
        regressions += bool(row["regressions"])
        before = "ok" if row["base_format_clean"] else "dirty"
        after = "ok" if row["head_format_clean"] else "dirty"
        print(
            f"{str(row['file']):<52} {row['base_findings']:>7} -> {row['head_findings']:<7}"
            f" {before:>6} -> {after:<6}  {verdict}"
        )

    snapshot = backlog_snapshot(root, args.advisory_paths)
    print(
        "\nlint-ratchet: repo-wide backlog (advisory, not a gate): "
        f"{snapshot['findings']} ruff findings, {snapshot['unformatted']} file(s) "
        "`ruff format --check` would rewrite"
    )

    if args.summary:
        payload = dict(result)
        payload["backlog"] = snapshot
        payload["regressions"] = regressions
        out = Path(args.summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"lint-ratchet: summary written to {out}")

    if regressions:
        print(f"\nlint-ratchet: FAIL -- {regressions} file(s) add lint debt")
        return 1
    print("\nlint-ratchet: PASS -- no new lint debt in this change set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
