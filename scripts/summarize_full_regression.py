"""Match sharded pytest JUnit cases to a frozen collection, retaining all outcomes."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def collected_cases(path):
    cases = {}
    for line in path.read_text().splitlines():
        if not line.startswith("tests/") or "::" not in line:
            continue
        file, rest = line.split("::", 1)
        test, bracket, parameters = rest.partition("[")
        *classes, name = test.split("::")
        key = (
            ".".join([file.removesuffix(".py").replace("/", "."), *classes]),
            name + bracket + parameters,
        )
        if key in cases:
            raise ValueError(f"ambiguous collected test {line}")
        cases[key] = line
    return cases


def message(text):
    text = re.sub(r"/workspace/scratch/[^/]+/fem-[\w-]+/", "", text or "")
    text = re.sub(r"(?:gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]+)", "[redacted]", text)
    return " ".join(text.split())[:480]


def summarize(collection, directories, out, revision):
    expected = collected_cases(collection)
    attempts = defaultdict(list)
    extra = []
    for directory in directories:
        for file in sorted(directory.glob("*.xml")):
            for case in ET.parse(file).iter("testcase"):
                key = (case.get("classname", ""), case.get("name", ""))
                problem = next((e for e in case if e.tag in {"error", "failure", "skipped"}), None)
                status = problem.tag if problem is not None else "passed"
                if problem is not None and problem.get("type") == "pytest.xfail":
                    status = "xfailed"
                reason = message(problem.get("message")) if problem is not None else ""
                row = (status, case.get("time", ""), reason, directory.name + "/" + file.name)
                if key in expected:
                    attempts[key].append(row)
                else:
                    extra.append(
                        {"classname": key[0], "name": key[1], "outcome": status, "reason": reason}
                    )
    out.mkdir(parents=True, exist_ok=True)
    counts, repeat_conflicts = Counter(), []
    table = io.StringIO(newline="")
    writer = csv.writer(table, lineterminator="\n")
    writer.writerow(
        ["nodeid", "outcome", "seconds", "reason", "junit_batch", "all_observed_outcomes"]
    )
    for key, node in expected.items():
        rows = attempts[key]
        states = sorted(set(r[0] for r in rows))
        if len(states) > 1:
            repeat_conflicts.append({"nodeid": node, "outcomes": states})
        status, seconds, reason, batch = rows[0] if rows else ("unaccounted", "", "", "")
        writer.writerow([node, status, seconds, reason, batch, "|".join(states)])
        counts[status] += 1
    payload = gzip.compress(table.getvalue().encode(), mtime=0)
    (out / "results.csv.gz").write_bytes(payload)
    data = dict(
        revision=revision,
        collected=len(expected),
        with_junit_outcome=sum(bool(rows) for rows in attempts.values()),
        counts=dict(counts),
        repeat_conflicts=repeat_conflicts,
        unmatched_junit_cases=extra,
        collection_sha256=hashlib.sha256(collection.read_bytes()).hexdigest(),
        table_sha256=hashlib.sha256(payload).hexdigest(),
    )
    (out / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: data[k] for k in ("collected", "with_junit_outcome", "counts")}))
    return not counts["unaccounted"] and not extra


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", required=True, type=Path)
    ap.add_argument("--junit", nargs="+", required=True, type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", required=True)
    args = ap.parse_args()
    if not summarize(args.collection, args.junit, args.out, args.revision):
        raise SystemExit(2)
