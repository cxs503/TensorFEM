#!/usr/bin/env python3
"""Condense a pytest JUnit XML run into a diffable test-baseline record.

The repository keeps no machine-readable record of "which tests are green on
this tree", so every drift hunt (new red vs pre-existing red vs environment
red) has to be re-derived by hand.  This tool turns a ``--junitxml`` artefact
into a small JSON baseline:

    python scripts/summarize_test_baseline.py runs/junit.xml \
        --output artifacts/test-baseline.json

and can diff two such records:

    python scripts/summarize_test_baseline.py runs/junit.xml \
        --baseline artifacts/test-baseline.json --check

``--check`` exits non-zero when a test that is green in the baseline turned
red (a *new* red), which is what a CI drift gate needs.  Tests that are red in
both records are *pre-existing* red and do not fail the gate; they are listed
so the report can classify them.

The signature is a SHA-256 over the sorted ``nodeid:outcome`` lines, so two
runs of the same tree on the same environment collide on purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "tensorfem.test-baseline/1"

# JUnit outcome precedence: one element can carry several child tags, the
# worst one wins so the record never hides a failure behind a skip.
_WORST_FIRST = ("error", "failure", "skipped")


def _outcome(case: ET.Element) -> str:
    for tag in _WORST_FIRST:
        node = case.find(tag)
        if node is not None:
            if tag == "skipped":
                # pytest encodes xfail/xpass inside the skip element.
                kind = node.get("type", "")
                if kind == "pytest.xfail":
                    return "xfailed"
                return "skipped"
            return "failed" if tag == "failure" else "error"
    return "passed"


def _message(case: ET.Element) -> str:
    for tag in _WORST_FIRST:
        node = case.find(tag)
        if node is not None:
            text = (node.get("message") or node.text or "").strip()
            return " ".join(text.split())[:400]
    return ""


def collect(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    tests: list[dict] = []
    for suite in suites:
        for case in suite.iter("testcase"):
            name = case.get("classname", "")
            nodeid = f"{name}::{case.get('name')}" if name else case.get("name", "")
            entry = {"nodeid": nodeid, "outcome": _outcome(case)}
            time = case.get("time")
            if time is not None:
                entry["time_s"] = round(float(time), 3)
            message = _message(case)
            if message:
                entry["message"] = message
            tests.append(entry)
    tests.sort(key=lambda t: t["nodeid"])
    counts: dict[str, int] = {}
    for test in tests:
        counts[test["outcome"]] = counts.get(test["outcome"], 0) + 1
    digest = hashlib.sha256(
        "\n".join(f"{t['nodeid']}:{t['outcome']}" for t in tests).encode()
    ).hexdigest()
    try:
        import torch

        torch_version = torch.__version__
    except Exception:  # pragma: no cover - torch is a hard dependency, be safe
        torch_version = "unavailable"
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": path.name,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch_version,
        },
        "totals": {"collected": len(tests), **counts},
        "signature": digest,
        "tests": tests,
    }


def diff(baseline: dict, current: dict) -> dict:
    before = {t["nodeid"]: t["outcome"] for t in baseline["tests"]}
    after = {t["nodeid"]: t["outcome"] for t in current["tests"]}
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = [
        {"nodeid": n, "baseline": before[n], "current": after[n]}
        for n in sorted(set(before) & set(after))
        if before[n] != after[n]
    ]
    # Regression = green before, not green now.  A pre-existing failure that
    # turns into a skip is drift worth reporting but not a new red.
    green = {"passed", "xfailed"}
    regressions = [c for c in changed if c["baseline"] in green and c["current"] not in green]
    recoveries = [c for c in changed if c["baseline"] not in green and c["current"] in green]
    return {
        "baseline_signature": baseline.get("signature"),
        "current_signature": current.get("signature"),
        "identical": baseline.get("signature") == current.get("signature"),
        "added": added,
        "removed": removed,
        "changed": changed,
        "regressions": regressions,
        "recoveries": recoveries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("junit", type=Path, help="pytest --junitxml artefact")
    parser.add_argument("--output", type=Path, help="write the condensed record here")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="compare against an existing record instead of only summarising",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 on a new red (green in baseline, not green now)",
    )
    args = parser.parse_args(argv)

    current = collect(args.junit)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(current, indent=2) + "\n")

    totals = current["totals"]
    print(
        "collected={collected} passed={passed} failed={failed} error={error} "
        "skipped={skipped} xfailed={xfailed}".format(
            collected=totals.get("collected", 0),
            passed=totals.get("passed", 0),
            failed=totals.get("failed", 0),
            error=totals.get("error", 0),
            skipped=totals.get("skipped", 0),
            xfailed=totals.get("xfailed", 0),
        )
    )
    print(f"signature {current['signature']}")

    red = [t for t in current["tests"] if t["outcome"] in {"failed", "error"}]
    for test in red:
        print(f"RED {test['nodeid']}")
        if test.get("message"):
            print(f"    {test['message']}")

    if not args.baseline:
        return 0

    baseline = json.loads(args.baseline.read_text())
    report = diff(baseline, current)
    for test in report["regressions"]:
        print(f"REGRESSION {test['nodeid']}: {test['baseline']} -> {test['current']}")
    for test in report["recoveries"]:
        print(f"RECOVERED {test['nodeid']}: {test['baseline']} -> {test['current']}")
    for nodeid in report["added"]:
        print(f"NEW {nodeid}")
    for nodeid in report["removed"]:
        print(f"GONE {nodeid}")
    print("identical" if report["identical"] else "drifted")
    if args.check and report["regressions"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())