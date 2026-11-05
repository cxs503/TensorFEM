#!/usr/bin/env python3
"""Ruff *debt ratchet*: freeze the pre-existing lint debt, fail on any increase.

Why not a plain `ruff check` gate
---------------------------------
The tree carries a large legacy backlog (4 271 `ruff check` violations in 347
files, 624 files that `ruff format` would rewrite - see
``docs/lint-debt-baseline.md``).  A blocking ``ruff check src tests scripts
examples`` could therefore only be made green by a repository-wide rewrite,
which would bury every other CI signal and touch files a focused PR has no
business touching.  The alternative, leaving the job ``continue-on-error``,
hides new debt just as effectively: a PR can arrive with fresh E702/F401 noise
and nothing complains.

This gate splits the two questions:

* **new debt** - *blocking*.  Any ``file -> rule`` pair whose count grows, any
  rule that shows up in a file that did not have it, and any file that needs
  reformatting but is not in the frozen set fail the job.
* **legacy debt** - *reported, never blocking*.  Debt that goes down is printed
  as progress and can be re-frozen in a reviewable commit with
  ``--update-baseline``.

The ledger lives in ``docs/lint-baseline.json`` (per-file, per-rule counts plus
the frozen unformatted-file set and a SHA-256 signature).

Usage
-----
    python scripts/lint_gate.py --check              # CI gate, exit 1 on new debt
    python scripts/lint_gate.py --update-baseline    # re-freeze (review the diff!)
    python scripts/lint_gate.py --check --report artifacts/lint-report.json

Exit codes: ``0`` no new debt, ``1`` new debt, ``2`` tooling problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1
DEFAULT_SCOPES = ("src", "tests", "scripts", "examples")
BASELINE_RELPATH = "docs/lint-baseline.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
REFORMAT_PREFIX = "Would reformat:"


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
class RuffError(RuntimeError):
    """Raised when ruff itself cannot be run or its output cannot be parsed."""


def _ruff(ruff: str, args: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [ruff, *args], cwd=str(root), capture_output=True, text=True, check=False
        )
    except OSError as exc:  # pragma: no cover - ruff binary missing
        raise RuffError(f"cannot execute ruff ({ruff}): {exc}") from exc


def _relpath(filename: str, root: Path) -> str:
    """Normalise a ruff-reported filename to a posix path relative to *root*."""
    path = Path(filename)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(root)
        except ValueError:
            pass
    return path.as_posix()


def collect_check(ruff: str, scopes: tuple[str, ...], root: Path) -> dict[str, dict[str, int]]:
    """Return ``{file: {rule: count}}`` for ``ruff check`` over *scopes*."""
    proc = _ruff(ruff, ["check", *scopes, "--output-format=json"], root)
    # 0 = clean, 1 = violations found (the expected legacy state).  Anything
    # else is a real ruff failure (bad config, unparsable file) and must never
    # be read as "green".
    if proc.returncode not in (0, 1):
        raise RuffError(f"ruff check failed rc={proc.returncode}: {proc.stderr.strip()[:500]}")
    try:
        items = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuffError(f"cannot parse ruff json output: {exc}") from exc
    ledger: dict[str, dict[str, int]] = {}
    for item in items:
        rel = _relpath(item["filename"], root)
        codes = ledger.setdefault(rel, {})
        codes[item["code"]] = codes.get(item["code"], 0) + 1
    return dict(sorted(ledger.items()))


def collect_unformatted(ruff: str, scopes: tuple[str, ...], root: Path) -> list[str]:
    """Return the sorted list of files ``ruff format --check`` would rewrite."""
    proc = _ruff(ruff, ["format", "--check", *scopes], root)
    if proc.returncode not in (0, 1):
        raise RuffError(f"ruff format --check failed rc={proc.returncode}: {proc.stderr[:500]}")
    out = set()
    for stream in (proc.stdout, proc.stderr):
        for line in (stream or "").splitlines():
            line = line.strip()
            if line.startswith(REFORMAT_PREFIX):
                out.add(_relpath(line[len(REFORMAT_PREFIX) :].strip(), root))
    return sorted(out)


def ruff_version(ruff: str, root: Path) -> str:
    proc = _ruff(ruff, ["--version"], root)
    return (proc.stdout or proc.stderr or "unknown").strip()


def tool_drift(baseline: dict, running_tool: str) -> str | None:
    """Return a message when the running ruff cannot be compared with *baseline*.

    The ledger is a per-file, per-rule count frozen with one ruff build.  A
    different build may add, drop or recategorise rules, so comparing across
    versions reads as "new debt" (or as suspiciously *little* debt) when the
    real problem is the tool.  That case is refused as a tooling problem
    (exit code 2) instead of being guessed at.
    """
    frozen_tool = (baseline.get("tool") or "").strip()
    running = (running_tool or "").strip()
    if not frozen_tool or frozen_tool == running:
        return None
    return (
        f"ruff version drift: ledger frozen with '{frozen_tool}', running '{running}'."
        "  Pin the same build (pyproject 'dev' extra) or re-freeze deliberately"
        " with --update-baseline"
    )


def totals(ledger: dict[str, dict[str, int]], unformatted: list[str]) -> dict[str, int]:
    return {
        "check": sum(sum(codes.values()) for codes in ledger.values()),
        "files_with_check_debt": len(ledger),
        "unformatted": len(unformatted),
    }


def signature(ledger: dict[str, dict[str, int]], unformatted: list[str]) -> str:
    """Stable digest of the ledger, so a baseline edit is visible in review."""
    payload = json.dumps(
        {"check": ledger, "unformatted": unformatted}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rule_totals(ledger: dict[str, dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for codes in ledger.values():
        for code, count in codes.items():
            out[code] = out.get(code, 0) + count
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


# --------------------------------------------------------------------------- #
# comparison (pure functions - the part the tests pin down)
# --------------------------------------------------------------------------- #
def regressions(
    baseline_check: dict[str, dict[str, int]],
    baseline_unformatted: list[str],
    current_check: dict[str, dict[str, int]],
    current_unformatted: list[str],
) -> list[str]:
    """New debt: grown counts, new ``file -> rule`` pairs, newly unformatted files."""
    problems: list[str] = []
    for path in sorted(current_check):
        frozen = baseline_check.get(path, {})
        for code in sorted(current_check[path]):
            now = current_check[path][code]
            before = frozen.get(code, 0)
            if now > before:
                verdict = f"new ({now}x)" if before == 0 else f"{before} -> {now}"
                problems.append(f"{path}: {code} {verdict}")
    frozen_files = set(baseline_unformatted)
    for path in sorted(current_unformatted):
        if path not in frozen_files:
            problems.append(f"{path}: needs 'ruff format' (not in frozen set)")
    return problems


def improvements(
    baseline_check: dict[str, dict[str, int]],
    baseline_unformatted: list[str],
    current_check: dict[str, dict[str, int]],
    current_unformatted: list[str],
) -> list[str]:
    """Debt that went away - reported as progress, never as a failure."""
    wins: list[str] = []
    for path in sorted(baseline_check):
        frozen = baseline_check[path]
        now = current_check.get(path, {})
        for code in sorted(frozen):
            before = frozen[code]
            remaining = now.get(code, 0)
            if remaining < before:
                wins.append(f"{path}: {code} {before} -> {remaining}")
    still_unformatted = set(current_unformatted)
    for path in sorted(baseline_unformatted):
        if path not in still_unformatted:
            wins.append(f"{path}: formatted")
    return wins


# --------------------------------------------------------------------------- #
# baseline I/O
# --------------------------------------------------------------------------- #
def load_baseline(path: Path) -> dict:
    if not path.is_file():
        raise RuffError(f"baseline not found: {path} (create it with --update-baseline)")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuffError(f"baseline is not valid json: {exc}") from exc
    if data.get("schema") != SCHEMA:
        raise RuffError(f"baseline schema {data.get('schema')} != {SCHEMA}")
    data.setdefault("check", {})
    data.setdefault("unformatted", [])
    return data


def build_baseline(
    ruff: str,
    scopes: tuple[str, ...],
    root: Path,
    check: dict[str, dict[str, int]],
    unformatted: list[str],
) -> dict:
    return {
        "schema": SCHEMA,
        "tool": ruff_version(ruff, root),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scopes": list(scopes),
        "signature": signature(check, unformatted),
        "totals": totals(check, unformatted),
        "check": check,
        "unformatted": unformatted,
    }


def render_report(
    current_check: dict[str, dict[str, int]],
    current_unformatted: list[str],
    baseline: dict,
    problems: list[str],
    wins: list[str],
) -> str:
    """Markdown summary - also fed to $GITHUB_STEP_SUMMARY by the CI job."""
    now = totals(current_check, current_unformatted)
    frozen = baseline.get("totals", {})
    lines = [
        "## Lint debt ratchet",
        "",
        "| metric | frozen baseline | now | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("check", "files_with_check_debt", "unformatted"):
        before = frozen.get(key, 0)
        after = now[key]
        lines.append(f"| {key} | {before} | {after} | {after - before:+d} |")
    lines += [
        "",
        "### per-rule totals",
        "",
        "| rule | frozen | now |",
        "| --- | ---: | ---: |",
    ]
    frozen_rules = rule_totals(baseline.get("check", {}))
    for code, count in rule_totals(current_check).items():
        lines.append(f"| {code} | {frozen_rules.get(code, 0)} | {count} |")
    lines += ["", f"### new debt ({len(problems)})", ""]
    lines += [f"- FAIL {p}" for p in problems[:60]] or ["- none"]
    if len(problems) > 60:
        lines.append(f"- ... and {len(problems) - 60} more")
    lines += ["", f"### debt burned down ({len(wins)})", ""]
    lines += [f"- OK {w}" for w in wins[:30]] or ["- none"]
    if len(wins) > 30:
        lines.append(f"- ... and {len(wins) - 30} more")
    return "\n".join(lines)


def emit_step_summary(report: str) -> None:
    """Append the report to the GitHub job summary when running in Actions."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    try:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")
    except OSError:  # pragma: no cover - summary is a nicety, not a gate
        pass


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ruff debt ratchet gate")
    parser.add_argument(
        "--check", action="store_true", help="gate mode (default when nothing else is given)"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=f"re-freeze {BASELINE_RELPATH} at the current state",
    )
    parser.add_argument(
        "--baseline", default=None, help=f"ledger path (default {BASELINE_RELPATH})"
    )
    parser.add_argument("--report", default=None, help="write the json report here")
    parser.add_argument("--root", default=str(REPO_ROOT), help="repository root")
    parser.add_argument("--scopes", nargs="+", default=list(DEFAULT_SCOPES))
    parser.add_argument("--ruff", default=None, help="ruff executable (default: from PATH)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    ruff = args.ruff or shutil.which("ruff")
    if not ruff:  # pragma: no cover - environment problem
        print("lint_gate: ruff not found on PATH (pip install '.[dev]')", file=sys.stderr)
        return 2
    baseline_path = Path(args.baseline) if args.baseline else root / BASELINE_RELPATH
    scopes = tuple(args.scopes)
    try:
        current_check = collect_check(ruff, scopes, root)
        current_unformatted = collect_unformatted(ruff, scopes, root)
        if args.update_baseline:
            baseline = build_baseline(ruff, scopes, root, current_check, current_unformatted)
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(
                json.dumps(baseline, indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )
            report = render_report(current_check, current_unformatted, baseline, [], [])
            print(report)
            emit_step_summary(report)
            print(f"\nlint_gate: baseline re-frozen at {baseline_path}")
            return 0
        baseline = load_baseline(baseline_path)
    except RuffError as exc:
        print(f"lint_gate: {exc}", file=sys.stderr)
        return 2

    drift = tool_drift(baseline, ruff_version(ruff, root))
    if drift:
        print(f"lint_gate: {drift}", file=sys.stderr)
        return 2

    problems = regressions(
        baseline.get("check", {}),
        baseline.get("unformatted", []),
        current_check,
        current_unformatted,
    )
    wins = improvements(
        baseline.get("check", {}),
        baseline.get("unformatted", []),
        current_check,
        current_unformatted,
    )
    report = render_report(current_check, current_unformatted, baseline, problems, wins)
    print(report)
    emit_step_summary(report)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "pass": not problems,
                    "tool": ruff_version(ruff, root),
                    "baseline_signature": baseline.get("signature"),
                    "totals": totals(current_check, current_unformatted),
                    "baseline_totals": baseline.get("totals", {}),
                    "rule_totals": rule_totals(current_check),
                    "regressions": problems,
                    "improvements": wins,
                },
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if problems:
        print(
            f"\nlint_gate: FAIL - {len(problems)} new lint debt item(s) vs baseline",
            file=sys.stderr,
        )
        return 1
    now = totals(current_check, current_unformatted)
    frozen = baseline.get("totals", {})
    print(
        "\nlint_gate: PASS - no new debt "
        f"(check {frozen.get('check', 0)} -> {now['check']}, "
        f"unformatted {frozen.get('unformatted', 0)} -> {now['unformatted']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
