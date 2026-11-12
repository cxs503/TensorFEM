#!/usr/bin/env python3
"""Consistency audit of the verified-benchmark gate corpus.

A gate file is only worth something if a machine can read it and disagree with
it.  This tool re-derives every verdict stored in
``benchmarks/verified/*/gates_verified.json`` from the raw fields and reports
where a file

* claims an overall PASS while one of its own gates failed
  (``pass_with_failed_gate``);
* stores ``passed = true`` for a numeric gate whose ``actual`` violates its own
  ``threshold`` (``gate_arithmetic``);
* declares a verification tolerance looser than the platform acceptance limit
  of 15 % (``tolerance_above_platform_limit``);
* is unreadable under the declared schema (``schema``).

The output is a machine-readable baseline.  ``tests/test_gates_consistency.py``
fails when new violations appear, so the corpus can only get cleaner.

Usage:
    python tools/gates_consistency_audit.py [--write]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "benchmarks" / "verified"
OUT = REPO / "docs" / "assets" / "gates_consistency" / "gates_consistency.json"

PASS_WORDS = ("PASS", "OK", "✅")
PLATFORM_MAX_ACCEPTABLE = 0.15  # owner standard: >15 % error is not acceptable


def _numeric(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _direction(gate: dict) -> str:
    """'min' for lower-bound gates (order >= p), 'max' for upper-bound gates."""
    text = f"{gate.get('id', '')} {gate.get('description', '')}".lower()
    if any(tok in text for tok in (">=", "at least", "minimum", "order")):
        return "min"
    return "max"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:  # synthetic files outside the repo (tests)
        return str(path)


def load_gates(path: Path) -> tuple[list[dict], object, dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload, None, {}
    gates = payload.get("gates")
    if gates is None:
        for key in ("criteria_results", "checks"):
            if isinstance(payload.get(key), list):
                gates = payload[key]
                break
    overall = payload.get("overall_verdict", payload.get("passed", payload.get("verdict")))
    return list(gates or []), overall, payload


def overall_is_pass(overall) -> bool:
    if overall is True:
        return True
    if isinstance(overall, str):
        return overall.strip().upper().startswith(PASS_WORDS)
    return False


def audit_file(path: Path) -> list[dict]:
    findings: list[dict] = []
    try:
        gates, overall, payload = load_gates(path)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the corpus scan
        return [
            {
                "file": _rel(path),
                "rule": "schema",
                "detail": f"unreadable gate file: {exc!r}",
            }
        ]
    rel = _rel(path)

    failed = [g for g in gates if isinstance(g, dict) and g.get("passed") is False]
    if failed and overall_is_pass(overall):
        findings.append(
            {
                "file": rel,
                "rule": "pass_with_failed_gate",
                "detail": "overall verdict claims PASS while gates failed: "
                + ", ".join(str(g.get("id", "?")) for g in failed),
                "overall": overall if isinstance(overall, (str, bool)) else str(overall),
            }
        )

    for g in gates:
        if not isinstance(g, dict):
            continue
        threshold = _numeric(g.get("threshold"))
        actual = _numeric(g.get("actual"))
        if threshold is None or actual is None:
            continue
        derived = bool(actual >= threshold) if _direction(g) == "min" else bool(actual <= threshold)
        if g.get("passed") is True and not derived:
            findings.append(
                {
                    "file": rel,
                    "rule": "gate_arithmetic",
                    "detail": f"{g.get('id')}: actual={actual:.6g} violates threshold={threshold:.6g} "
                    "but passed=true",
                }
            )
        text = f"{g.get('id', '')} {g.get('description', '')}".lower()
        error_gate = any(tok in text for tok in ("error", "accuracy", "displacement within"))
        if (
            g.get("passed") is True
            and error_gate
            and 0.0 < actual > PLATFORM_MAX_ACCEPTABLE
            and threshold > 0.0
        ):
            findings.append(
                {
                    "file": rel,
                    "rule": "tolerance_above_platform_limit",
                    "detail": f"{g.get('id')}: tolerance {threshold:.3g} (actual {actual:.3g}) "
                    f"exceeds the {PLATFORM_MAX_ACCEPTABLE:.0%} platform limit",
                }
            )
    # an order gate is only meaningful if the sequence it was fitted to is monotone
    order_gates = [
        g
        for g in gates
        if isinstance(g, dict)
        and "order" in f"{g.get('id', '')} {g.get('description', '')}".lower()
        and g.get("passed") is True
    ]
    monotone_gate = next(
        (
            g
            for g in gates
            if isinstance(g, dict) and "monoton" in f"{g.get('id', '')} {g.get('description', '')}".lower()
        ),
        None,
    )
    if order_gates and monotone_gate is not None and monotone_gate.get("passed") is False:
        findings.append(
            {
                "file": rel,
                "rule": "order_gate_despite_nonmonotone",
                "detail": f"{order_gates[0].get('id')} passes while "
                f"{monotone_gate.get('id')} failed: a fitted order on a non-monotone "
                "sequence is not evidence of a real convergence rate",
            }
        )
    return findings


def audit_corpus(corpus: Path = CORPUS) -> dict:
    files = sorted(corpus.glob("*/gates_verified.json"))
    findings: list[dict] = []
    for path in files:
        findings.extend(audit_file(path))
    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
    return {
        "schema": "tensorfem.gates-consistency.v1",
        "corpus": _rel(corpus),
        "files_scanned": len(files),
        "findings": findings,
        "counts_by_rule": by_rule,
        "total_findings": len(findings),
        "platform_max_acceptable_error": PLATFORM_MAX_ACCEPTABLE,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Verified-gate corpus consistency audit",
        "",
        f"- gate files scanned: {report['files_scanned']}",
        f"- findings: {report['total_findings']}",
        "",
        "| rule | count |",
        "|---|---|",
    ]
    for rule, count in sorted(report["counts_by_rule"].items()):
        lines.append(f"| {rule} | {count} |")
    lines += ["", "## Findings", ""]
    if not report["findings"]:
        lines.append("none — every stored verdict is reproducible from its own fields.")
    else:
        lines += ["| file | rule | detail |", "|---|---|---|"]
        for f in report["findings"]:
            lines.append(f"| {f['file']} | {f['rule']} | {f['detail']} |")
    return "\n".join(lines) + "\n"


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the baseline JSON + markdown")
    args = ap.parse_args()
    report = audit_corpus()
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, indent=2))
    for f in report["findings"]:
        print(f"  [{f['rule']}] {f['file']}: {f['detail']}")
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        (OUT.parent / "gates_consistency.md").write_text(render_markdown(report))
        print(f"wrote {OUT}")
    return report


if __name__ == "__main__":
    main()