#!/usr/bin/env python3
"""Audit the slow-test registry from a pytest ``--durations`` log.

The fast lane (``pytest --fast``) is only as honest as the ``slow`` markers it
acts on, and the markers are hand-placed.  Two drift modes matter:

* a *newly added* heavyweight test is not marked, so the fast lane silently
  pays it on every run (registration gap);
* a *stale* entry stays marked after its case was made cheap, so the full lane
  runs it while the fast lane no longer does (cosmetic exclusion).

This tool re-derives the candidate set from measurements instead of trusting
memory: it parses a ``pytest --durations=0`` log, ranks cases by wall time, and
diffs the measurement against ``tests/slow_lane_manifest.txt``.

    # what would the fast lane drop, measured on this machine?
    python scripts/slow_lane_audit.py artifacts/pytest_durations.log --threshold 2.0

    # machine-readable, for the report / for CI
    python scripts/slow_lane_audit.py artifacts/pytest_durations.log --json

    # drift gate: exit 1 when a test above the threshold is not registered
    python scripts/slow_lane_audit.py artifacts/pytest_durations.log \
        --threshold 2.0 --check

The threshold is a *policy* number, not a verdict: a case is registered when it
is heavyweight *in kind* (convergence sweep, long nonlinear path, industrial
factorisation) even if this machine happens to run it in 1.5 s, because the
cost is what it does on the slowest supported target (loong64 CI).  The audit
therefore reports, it does not rewrite the registry; ``--write`` is opt-in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "tests" / "slow_lane_manifest.txt"

# ``pytest --durations=0`` prints e.g.
#     12.34s call     tests/test_foo.py::test_bar
#      0.02s setup    tests/test_foo.py::test_baz
_DURATION = re.compile(r"^\s*(?P<seconds>[0-9.]+)s\s+(?P<phase>call|setup|teardown)\s+(?P<nodeid>\S+)\s*$")


def parse_durations(text: str) -> dict[str, float]:
    """Total per-test wall time from a pytest durations log (call+setup+teardown)."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _DURATION.match(line)
        if match is None:
            continue
        nodeid = match.group("nodeid")
        totals[nodeid] = totals.get(nodeid, 0.0) + float(match.group("seconds"))
    return totals


def parse_junit(path: Path) -> dict[str, float]:
    """Per-test wall time from a JUnit XML report (``--junitxml``), xdist-merged.

    JUnit carries no node ids, so they are reconstructed from ``classname``
    (the dotted module path) and ``name`` (the test, parametrisation suffix
    included).  That is the same shape ``pytest`` collects; a class-scoped test
    is reconstructed as ``module.py::Class::test``.
    """
    totals: dict[str, float] = {}
    for case in ET.parse(path).iter("testcase"):
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        if not name:
            continue
        parts = [part for part in classname.split(".") if part]
        if len(parts) >= 2 and parts[-1][:1].isupper():
            # module path + class-scoped test: tests/test_x.py::TestClass::test_y
            module_path = "/".join(parts[:-1]) + ".py"
            nodeid = f"{module_path}::{parts[-1]}::{name}"
        elif parts:
            module_path = "/".join(parts) + ".py"
            nodeid = f"{module_path}::{name}"
        else:
            nodeid = name
        try:
            seconds = float(case.get("time") or 0.0)
        except ValueError:
            seconds = 0.0
        totals[nodeid] = totals.get(nodeid, 0.0) + seconds
    return totals


def base_node_id(node_id: str) -> str:
    """Strip a parametrisation suffix: ``...::test_x[a-1]`` -> ``...::test_x``."""
    return node_id.split("[", 1)[0]


def read_manifest(path: Path = MANIFEST) -> list[str]:
    """Registered slow node ids, comments and blank lines ignored, order kept."""
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def audit(totals: dict[str, float], manifest: list[str], threshold: float) -> dict:
    registered = set(manifest)

    def key(nodeid: str) -> str:
        # ``--durations`` reports the same node id pytest collects; parametrised
        # ids contain no whitespace, so no normalisation is needed beyond that.
        return nodeid

    measured: dict[str, float] = {}
    for nodeid, seconds in totals.items():
        base = base_node_id(nodeid)
        measured[base] = measured.get(base, 0.0) + seconds
    over = {n: s for n, s in measured.items() if s >= threshold}
    return {
        "threshold_seconds": threshold,
        "tests_measured": len(measured),
        "measured_seconds_total": round(sum(measured.values()), 3),
        "over_threshold": dict(sorted(over.items(), key=lambda kv: -kv[1])),
        "registered": sorted(registered),
        "unregistered_over_threshold": sorted(n for n in over if n not in registered),
        "registered_under_threshold": sorted(n for n in registered if n in measured and n not in over),
        "registered_not_measured": sorted(n for n in registered if n not in measured),
        "manifest_size": len(registered),
    }


def _print_report(result: dict, *, limit: int) -> None:
    over = result["over_threshold"]
    total = result["tests_measured"]
    print(f"measured {total} tests, {result['measured_seconds_total']:.1f}s total")
    print(f"threshold {result['threshold_seconds']:.2f}s: {len(over)} tests above it")
    print(f"registry: {result['manifest_size']} entries ({MANIFEST.relative_to(REPO_ROOT)})")
    if over:
        print("\n  seconds  node id")
        for nodeid, seconds in list(over.items())[:limit]:
            flag = "" if nodeid in result["registered"] else "  <- NOT REGISTERED"
            print(f"  {seconds:7.2f}  {nodeid}{flag}")
        if len(over) > limit:
            print(f"  ... {len(over) - limit} more")
    if result["registered_under_threshold"]:
        print("\nregistered but under the threshold on this machine:")
        for nodeid in result["registered_under_threshold"]:
            print(f"  {nodeid}")
    if result["registered_not_measured"]:
        print("\nregistered but not present in this run (stale or deselected):")
        for nodeid in result["registered_not_measured"]:
            print(f"  {nodeid}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "log",
        type=Path,
        nargs="?",
        help="pytest log produced with --durations=0 (omit when using --junitxml)",
    )
    parser.add_argument("--junitxml", type=Path, help="JUnit XML from --junitxml (xdist-merged)")
    parser.add_argument("--threshold", type=float, default=2.0, help="seconds; slow-candidate cut-off (default 2.0)")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--json", action="store_true", help="emit the audit as JSON")
    parser.add_argument("--limit", type=int, default=40, help="rows printed in the human report")
    parser.add_argument("--check", action="store_true", help="exit 1 if a test above the threshold is unregistered")
    parser.set_defaults(check=False)
    args = parser.parse_args(argv)

    if args.junitxml is not None:
        if not args.junitxml.is_file():
            parser.error(f"no such JUnit report: {args.junitxml}")
        totals = parse_junit(args.junitxml)
        source = str(args.junitxml)
    elif args.log is not None:
        if not args.log.is_file():
            parser.error(f"no such durations log: {args.log}")
        totals = parse_durations(args.log.read_text(errors="replace"))
        source = str(args.log)
    else:
        parser.error("give a --durations log or --junitxml")
    if not totals:
        parser.error(f"{source}: no per-test timings found")
    result = audit(totals, read_manifest(args.manifest), args.threshold)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_report(result, limit=args.limit)

    if args.check and result["unregistered_over_threshold"]:
        print(
            f"\nFAIL: {len(result['unregistered_over_threshold'])} test(s) run longer than "
            f"{args.threshold:.2f}s without a 'slow' marker registration",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())