#!/usr/bin/env python3
"""Lint ratchet for TensorFEM: a change may not *add* lint debt.

Why this exists
---------------
``.github/workflows/ci.yml`` linted the whole tree::

    ruff check src tests scripts examples
    ruff format --check src tests scripts examples

The tree carries a large pre-existing backlog (thousands of ``E701``/``E702``
packed statements, plus unused imports, unused variables and star imports), so
that job was red on *every* ref including ``main``.  A red check therefore said
nothing about the pull request under review: PR #127 inherited it while adding
zero findings of its own.

What this gate does instead
---------------------------
* It measures only the Python files the change touches (``src``, ``tests``,
  ``scripts``, ``examples``).
* Each such file is compared against *its own* version at the merge base: the
  per-rule-code finding count may not grow, and a file that was already
  ``ruff format`` clean may not become unformatted.  A file that the change
  adds did not exist at the merge base, so it must be completely clean.
* A probe with a deliberately broken file proves the measurements are not
  empty, so a green result can never come from linting nothing (fail closed).
* The whole-tree backlog is left alone; it can only shrink, and its size is
  still printed for the record.

Exit codes: ``0`` no regression, ``1`` the change adds debt, ``2`` the gate
could not run (missing base ref / missing ruff / blind probe).  Exit 2 is a
failure on purpose -- a gate that cannot see must not report success.

Usage::

    python scripts/lint_ratchet.py --base origin/main
    python scripts/lint_ratchet.py --base origin/main --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

SCOPE_DIRS = ("src", "tests", "scripts", "examples")
PROBE_SOURCE = '"""probe"""\nimport os\n\n\ndef f():\n    x = 1; y = 2\n    return x\n'
PROBE_MIN_FINDINGS = 2
STAT_LINE = re.compile(r"^\s*(\d+)\t[A-Z]+\d+\t")


class GateError(RuntimeError):
    """The gate could not take a measurement; treat as failure."""


def sh(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def repo_root() -> Path:
    cp = sh(["git", "rev-parse", "--show-toplevel"], Path.cwd())
    if cp.returncode != 0:
        raise GateError(f"not inside a git work tree: {cp.stderr.strip()}")
    return Path(cp.stdout.strip())


def resolve_base(root: Path, base: str) -> str:
    cp = sh(["git", "rev-parse", "--verify", f"{base}^{{commit}}"], root)
    if cp.returncode != 0:
        raise GateError(
            f"cannot resolve base ref {base!r}; fetch the base branch first "
            "(actions/checkout needs fetch-depth: 0 to have a merge base)"
        )
    return cp.stdout.strip()


def changed_files(root: Path, base: str, scope: tuple[str, ...]) -> dict[str, str]:
    """``{path: "added"|"modified"}`` for in-scope Python files touched by the change."""
    cp = sh(["git", "diff", "--name-status", "--no-renames", f"{base}...HEAD", "--", *scope], root)
    if cp.returncode != 0:
        raise GateError(f"git diff failed: {cp.stderr.strip()[:400]}")
    changed: dict[str, str] = {}
    for line in cp.stdout.splitlines():
        if not line.strip():
            continue
        status, path = line.split("\t", 1)
        if status.startswith("D") or not path.endswith(".py"):
            continue
        in_scope = any(path == d or path.startswith(d + "/") for d in scope)
        if in_scope:
            changed[path] = "added" if status.startswith("A") else "modified"
    return dict(sorted(changed.items()))


def ruff_config(root: Path) -> list[str]:
    cfg = root / "pyproject.toml"
    return ["--config", str(cfg)] if cfg.exists() else []


def findings(ruff: str, root: Path, paths: list[str], config: list[str]) -> dict[str, Counter[str]]:
    """Per-file Counter of rule codes reported by ``ruff check``."""
    per_file: dict[str, Counter[str]] = {p: Counter() for p in paths}
    if not paths:
        return per_file
    cp = sh([ruff, "check", "--no-cache", "--output-format", "json", *config, *paths], root)
    if cp.returncode not in (0, 1):
        raise GateError(f"ruff check failed (exit {cp.returncode}): {cp.stderr.strip()[:600]}")
    try:
        reported = json.loads(cp.stdout or "[]")
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise GateError(f"cannot parse ruff JSON output: {exc}") from exc
    for item in reported:
        name = item.get("filename") or item.get("file") or ""
        rel = os.path.relpath(os.path.abspath(os.path.join(root, name)), root).replace(os.sep, "/")
        per_file.setdefault(rel, Counter())[item["code"]] += 1
    return per_file


def is_formatted(ruff: str, root: Path, path: str, config: list[str]) -> bool:
    cp = sh([ruff, "format", "--check", "--quiet", *config, path], root)
    if cp.returncode not in (0, 1):
        raise GateError(f"ruff format failed on {path}: {cp.stderr.strip()[:400]}")
    return cp.returncode == 0


def materialise_base(root: Path, base: str, paths: list[str], dest: Path) -> list[str]:
    """Write the base version of ``paths`` under ``dest``; return those that existed."""
    present: list[str] = []
    for path in paths:
        cp = sh(["git", "show", f"{base}:{path}"], root)
        if cp.returncode != 0:
            continue  # added by the change: no base copy, so it starts from clean
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(cp.stdout, encoding="utf-8")
        present.append(path)
    return present


def probe(ruff: str, workdir: Path, config: list[str]) -> dict:
    """Prove the measurements are live: a seeded violation must be reported."""
    seed = workdir / "_lint_ratchet_probe.py"
    seed.write_text(PROBE_SOURCE, encoding="utf-8")
    found = sum(findings(ruff, workdir, [seed.name], config)[seed.name].values())
    unformatted = not is_formatted(ruff, workdir, seed.name, config)
    seed.unlink()
    if found < PROBE_MIN_FINDINGS or not unformatted:
        raise GateError(f"probe went blind: {found} findings / unformatted={unformatted}")
    return {"findings": found, "unformatted": int(unformatted)}


def whole_tree_totals(ruff: str, root: Path, scope: tuple[str, ...]) -> dict[str, int]:
    """Backlog size for the record only -- never used to fail the gate."""
    present = [d for d in scope if (root / d).is_dir()]
    check = sh([ruff, "check", "--no-cache", "--statistics", *present], root)
    fmt = sh([ruff, "format", "--check", *present], root)
    return {
        "findings": sum(
            int(match.group(1))
            for match in (STAT_LINE.match(line) for line in check.stdout.splitlines())
            if match
        ),
        "unformatted": sum(
            1 for line in fmt.stdout.splitlines() if line.startswith("Would reformat")
        ),
    }


def ruff_version(ruff: str) -> str:
    out = sh([ruff, "--version"], Path.cwd()).stdout.strip()
    return out or "unknown"


def build_report(root: Path, base_ref: str, scope: tuple[str, ...], ruff: str) -> dict:
    base = resolve_base(root, base_ref)
    head = sh(["git", "rev-parse", "HEAD"], root).stdout.strip()
    config = ruff_config(root)
    changed = changed_files(root, base, scope)

    if not changed:
        return {
            "base_ref": base_ref,
            "base_sha": base,
            "head_sha": head,
            "ruff": ruff_version(ruff),
            "probe": {"findings": None, "unformatted": None},
            "files": [],
            "regressions": [],
            "whole_tree": whole_tree_totals(ruff, root, scope),
            "note": "no in-scope Python file changed",
        }

    paths = list(changed)
    head_findings = findings(ruff, root, paths, config)
    head_fmt = {path: is_formatted(ruff, root, path, config) for path in paths}

    with tempfile.TemporaryDirectory(prefix="lint-ratchet-base-") as tmp:
        tmp_root = Path(tmp)
        base_present = materialise_base(root, base, paths, tmp_root)
        base_findings = findings(ruff, tmp_root, base_present, config)
        base_fmt = {path: is_formatted(ruff, tmp_root, path, config) for path in base_present}
        probed = probe(ruff, tmp_root, config)

    rows: list[dict] = []
    regressions: list[dict] = []
    for path in paths:
        before, after = base_findings.get(path, Counter()), head_findings.get(path, Counter())
        grew = {
            code: after[code] - before.get(code, 0)
            for code in after
            if after[code] > before.get(code, 0)
        }
        fmt_before = base_fmt.get(path, True)  # a new file starts from clean
        fmt_after = head_fmt[path]
        reasons = []
        if grew:
            reasons.append("new findings")
        if fmt_before and not fmt_after:
            reasons.append("newly unformatted")
        row = {
            "path": path,
            "change": changed[path],
            "base_findings": sum(before.values()),
            "head_findings": sum(after.values()),
            "growth": grew,
            "base_formatted": fmt_before,
            "head_formatted": fmt_after,
            "reasons": reasons,
        }
        rows.append(row)
        if reasons:
            regressions.append(row)

    return {
        "base_ref": base_ref,
        "base_sha": base,
        "head_sha": head,
        "ruff": ruff_version(ruff),
        "probe": probed,
        "files": rows,
        "regressions": regressions,
        "whole_tree": whole_tree_totals(ruff, root, scope),
    }


def print_report(report: dict) -> None:
    print(
        f"lint ratchet: base {report['base_sha'][:8]} ({report['base_ref']}) -> "
        f"head {report['head_sha'][:8]}  ({report['ruff']})"
    )
    probe_ = report["probe"]
    if probe_["findings"] is not None:
        print(
            f"probe: {probe_['findings']} findings / "
            f"unformatted={bool(probe_['unformatted'])} (gate is live)"
        )
    if not report["files"]:
        print(report.get("note", "nothing to check"))
    else:
        print(f"{'file':<50} {'change':<9} {'base':>5} {'head':>5}  format    verdict")
        for row in report["files"]:
            if not row["head_formatted"]:
                fmt = "unformatted" if row["base_formatted"] else "kept-debt"
            else:
                fmt = "clean"
            verdict = "REGRESSION" if row["reasons"] else "ok"
            print(
                f"{row['path']:<50} {row['change']:<9} {row['base_findings']:>5} "
                f"{row['head_findings']:>5}  {fmt:<9} {verdict}"
            )
            if row["reasons"]:
                print(f"    reason: {', '.join(row['reasons'])}  new findings: {row['growth']}")
    whole = report["whole_tree"]
    print(
        f"whole-tree backlog (reported, not gated): {whole['findings']} findings / "
        f"{whole['unformatted']} unformatted files"
    )
    if report["regressions"]:
        print(f"\nFAIL: {len(report['regressions'])} file(s) got worse")
    else:
        print("\nOK: this change adds no lint debt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        default=os.environ.get("LINT_RATCHET_BASE", "origin/main"),
        help="base ref; the merge base is used (`git diff base...HEAD`)",
    )
    parser.add_argument("--scope", default=",".join(SCOPE_DIRS))
    parser.add_argument("--ruff", default="ruff")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    scope = tuple(part.strip() for part in args.scope.split(",") if part.strip())
    if shutil.which(args.ruff) is None:
        print(f"lint ratchet: FAIL - {args.ruff} is not installed", file=sys.stderr)
        return 2
    try:
        report = build_report(repo_root(), args.base, scope, args.ruff)
    except GateError as exc:
        print(f"lint ratchet: FAIL - {exc}", file=sys.stderr)
        return 2
    print_report(report)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if report["regressions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
