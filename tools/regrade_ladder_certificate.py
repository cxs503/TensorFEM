#!/usr/bin/env python3
"""Re-grade every shipped refinement-ladder certificate from its own artifact.

Why this exists
---------------
A refinement-ladder anchor (``benchmarks/verified/<anchor>/``) publishes two
things that must agree:

* ``result.json`` -- the measured ladder (``levels`` + ``reference``) and the
  ``criteria`` it was graded under, i.e. a ``tensorfem.refinement-gate/1.0``
  block;
* ``gates_verified.json`` -- the ``tensorfem.verified-gates.v1`` envelope the
  certificate audit reads.

``tools/verified_anchor_audit.py`` checks that each *gate* is recomputable from
its own ``actual``/``threshold``/``operator``.  It cannot check the stronger
property this tool checks: that the **stored gates are exactly what the stored
levels produce under the stored criteria**.  Without that, a hand-edited
threshold, a gate copied from an older run, or a ``result.json`` refreshed
without re-grading its certificate all pass the audit -- because every gate
still agrees with itself.

The re-grade runs in ``tensorfem.refinement_gate.regrade_ladder``: no solver, no
caller memory, no per-anchor recipe.  If the artifact lost its ``criteria``
block the tool fails closed instead of guessing thresholds from the gates it is
supposed to check.

Usage
-----
    python tools/regrade_ladder_certificate.py            # human table
    python tools/regrade_ladder_certificate.py --check    # exit 1 on any drift
    python tools/regrade_ladder_certificate.py --json     # machine record
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DIR = REPO_ROOT / "benchmarks" / "verified"
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorfem.refinement_gate import (  # noqa: E402
    SCHEMA as LADDER_SCHEMA,
    LadderCertificateError,
    certificate_drift,
    regrade_ladder,
)

GATE_SCHEMA = "tensorfem.verified-gates.v1"
ENVELOPE_NAME = "gates_verified.json"

#: ``criteria`` entries in the envelope that duplicate the ladder block.  The
#: envelope is allowed to phrase them as prose (older anchors do), in which case
#: there is nothing to compare numerically.
SHARED_CRITERIA = ("tolerance", "min_order", "coarsest_sanity")

_PREFIX = re.compile(r"^G\d+_")


def bare_gate_id(gate_id: str) -> str:
    """Drop the per-anchor numbering, keep the ladder prefix.

    ``G5_asymptotic_range_consistency`` (pinched cylinder) names the same ladder
    gate as ``asymptotic_range_consistency`` in its ``result.json``; the
    Scordelis-Lo envelope additionally prefixes the ladder name
    (``cylindrical.accuracy_finest``) because that anchor ships two ladders.
    """

    return _PREFIX.sub("", str(gate_id))


def ladder_hint(pointer: str) -> str | None:
    """Ladder name a block belongs to, read from its position in ``result.json``.

    ``/ladder_grading/flat_facet`` -> ``"flat_facet"``; a single-ladder anchor
    (``/ladder_grading``) -> ``None``.
    """

    parts = [p for p in pointer.split("/") if p]
    if len(parts) >= 2 and not parts[-1].startswith("[") and parts[-1] != "ladder_grading":
        return parts[-1]
    return None


def find_blocks(payload: object, path: str = "") -> list[tuple[str, dict]]:
    """Every ``tensorfem.refinement-gate/1.0`` block inside ``payload``."""

    found: list[tuple[str, dict]] = []
    if isinstance(payload, dict):
        if payload.get("schema") == LADDER_SCHEMA:
            found.append((path or "/", payload))
        for key, value in payload.items():
            found.extend(find_blocks(value, f"{path}/{key}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(find_blocks(value, f"{path}[{index}]"))
    return found


def discover(verified_dir: Path) -> list[tuple[str, str, dict]]:
    """``(anchor, json_pointer, block)`` for every shipped ladder block."""

    out: list[tuple[str, str, dict]] = []
    for result_path in sorted(verified_dir.glob("*/result.json")):
        anchor = result_path.parent.name
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for pointer, block in find_blocks(payload):
            out.append((anchor, pointer, block))
    return out


def _envelope_gate(gates: dict, gate_id: str, ladder: str | None) -> dict | None:
    for candidate in ([f"{ladder}.{gate_id}", gate_id] if ladder else [gate_id]):
        if candidate in gates:
            return gates[candidate]
    return None


def _envelope_issues(anchor_dir: Path, block: dict, ladder: str | None = None) -> list[dict]:
    """Compare one ladder block against the anchor's certificate envelope."""

    path = anchor_dir / ENVELOPE_NAME
    if not path.is_file():
        return [{"anchor": anchor_dir.name, "kind": "missing_envelope",
                 "detail": f"{ENVELOPE_NAME} not shipped"}]
    envelope = json.loads(path.read_text())
    if envelope.get("schema") != GATE_SCHEMA:
        return [{"anchor": anchor_dir.name, "kind": "envelope_schema",
                 "detail": f"schema={envelope.get('schema')!r}"}]

    issues: list[dict] = []
    gates = {}
    for gate in envelope.get("gates") or []:
        if isinstance(gate, dict):
            gates.setdefault(bare_gate_id(gate.get("id")), gate)

    block_gates = {bare_gate_id(g.get("id")): g for g in block.get("gates") or []
                   if isinstance(g, dict)}
    for gate_id, gate in block_gates.items():
        envelope_gate = _envelope_gate(gates, gate_id, ladder)
        if envelope_gate is None:
            issues.append({"anchor": anchor_dir.name, "kind": "gate_absent_from_envelope",
                           "gate": gate_id})
            continue
        for field in ("status", "operator", "actual", "threshold"):
            if envelope_gate.get(field) != gate.get(field):
                issues.append({
                    "anchor": anchor_dir.name,
                    "kind": "gate_differs_from_envelope",
                    "gate": gate_id,
                    "field": field,
                    "envelope": envelope_gate.get(field),
                    "ladder": gate.get(field),
                })

    criteria = block.get("criteria") or {}
    envelope_criteria = envelope.get("criteria") or {}
    for name in SHARED_CRITERIA:
        declared = envelope_criteria.get(name)
        if isinstance(declared, (int, float)) and not isinstance(declared, bool):
            if declared != criteria.get(name):
                issues.append({
                    "anchor": anchor_dir.name,
                    "kind": "criteria_differs_from_envelope",
                    "field": name,
                    "envelope": declared,
                    "ladder": criteria.get(name),
                })
    return issues


def audit(verified_dir: Path = VERIFIED_DIR) -> dict:
    """Full re-grade sweep over the shipped ladder certificates."""

    blocks = discover(verified_dir)
    records = []
    for anchor, pointer, block in blocks:
        record: dict[str, object] = {
            "anchor": anchor,
            "pointer": pointer,
            "gates": len(block.get("gates") or []),
            "status": block.get("status"),
        }
        try:
            regraded = regrade_ladder(block)
            drift = certificate_drift(block)
            record["regraded_status"] = regraded.get("status")
            record["drift"] = drift
        except LadderCertificateError as exc:
            record["drift"] = []
            record["error"] = str(exc)
        record["envelope_issues"] = _envelope_issues(
            verified_dir / anchor, block, ladder_hint(pointer))
        record["ok"] = (not record.get("error") and not record["drift"]
                        and not record["envelope_issues"])
        records.append(record)

    return {
        "schema": "tensorfem.ladder-regrade/1",
        "blocks": len(records),
        "gates": sum(int(r["gates"]) for r in records),
        "regraded_blocks": sum(1 for r in records if "regraded_status" in r),
        "drift_fields": sum(len(r["drift"]) for r in records),
        "errors": [{"anchor": r["anchor"], "pointer": r["pointer"], "error": r["error"]}
                   for r in records if r.get("error")],
        "ok": all(r["ok"] for r in records),
        "records": records,
    }


def _print_table(record: dict) -> None:
    print(f"anchors={record['blocks']} gates={record['gates']} "
          f"regraded={record['regraded_blocks']} drift={record['drift_fields']} "
          f"errors={len(record['errors'])}")
    for entry in record["records"]:
        mark = "ok  " if entry["ok"] else "FAIL"
        print(f"  [{mark}] {entry['anchor']}{entry['pointer']} "
              f"gates={entry['gates']} status={entry['status']}")
        if entry.get("error"):
            print(f"         error: {entry['error']}")
        for item in entry["drift"]:
            target = item["id"] or item["scope"]
            print(f"         drift {target}.{item['field']}: "
                  f"stored={item['stored']!r} regraded={item['regraded']!r}")
        for issue in entry["envelope_issues"]:
            print(f"         envelope {issue['kind']}: "
                  + ", ".join(f"{k}={v!r}" for k, v in issue.items()
                              if k not in ("anchor", "kind")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if any certificate is not reproducible from its own artifact")
    parser.add_argument("--json", action="store_true", help="print the full record as JSON")
    args = parser.parse_args(argv)

    record = audit()
    if args.json:
        json.dump(record, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_table(record)
    if args.check and not record["ok"]:
        print("certificate re-grade FAILED: a shipped certificate is not reproducible "
              "from its own result.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())