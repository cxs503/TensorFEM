#!/usr/bin/env python3
"""Baseline contrast for the public-API manifest gate (mission m17947).

Mission m15038 shipped the reviewed manifest (``api/public-api-v1.json``), its
packaged twin and ``scripts/audit_public_api.py``.  That audit only compared
names/modules/stability against ``tensorfem.__all__``.  This script runs the
gates side by side on the *same* mutated manifests so the improvement is a
measured red/green contrast, not a claim:

* ``baseline`` = ``scripts/audit_public_api.py`` as recovered from git rev
  ``BASELINE_REV`` (the main tip this branch was cut from),
* ``hardened`` = the working-tree gate.

Each case injects exactly one fault into a copy of the reviewed snapshot and
records the exit code of both gates.

Usage::

    PYTHONPATH=src python scripts/m17947_gate_baseline_compare.py \
        --write artifacts/m17947_public_api_gate_baseline.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "api/public-api-v1.json"
PACKAGED = ROOT / "src" / "tensorfem" / "public_api_manifest.json"
HARDENED = ROOT / "scripts" / "audit_public_api.py"
OUT = ROOT / "artifacts" / "m17947_public_api_gate_baseline.json"
# main tip the branch was cut from; a fixed rev keeps the contrast reproducible
# after this branch is committed (``HEAD`` would then be the hardened gate).
BASELINE_REV = "fe9673f0208c82200dffae829ec1d91095544281"


def _mutations(payload: dict) -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []

    def add(label: str, mutate) -> None:
        data = copy.deepcopy(payload)
        mutate(data)
        cases.append((label, data))

    add("unknown_entry_key", lambda d: d["entries"][3].update(stabilty="stable"))
    add("missing_introduced", lambda d: d["entries"][3].pop("introduced"))
    add("wrong_schema_id", lambda d: d.update(schema="tensorfem.public-api-manifest/0.9"))
    add("non_semver_introduced", lambda d: d["entries"][3].update(introduced="v0.1"))
    add("unknown_stability", lambda d: d["entries"][3].update(stability="stable-ish"))
    add("stability_without_policy", lambda d: d["policy"].pop("internal_candidate"))
    add("unknown_top_level_key", lambda d: d.update(generated_by="hand edit"))
    return cases


def _load_baseline(baseline_rev: str):
    """Import the historical gate as a module (torch import is amortised)."""
    import importlib.util

    source = subprocess.check_output(
        ["git", "show", f"{baseline_rev}:scripts/audit_public_api.py"],
        cwd=ROOT, text=True,
    )
    tmp = Path(tempfile.mkdtemp(prefix="m17947-baseline-"))
    path = tmp / "audit_baseline.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("m17947_baseline_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(module)
    return module


def _run_gate(fn, manifest: Path, extra: list[str]) -> dict[str, object]:
    """Call a gate's audit() and map the result onto its documented 0/1 exit."""
    try:
        result = fn(manifest, *extra)
    except TypeError as exc:  # historical signature without extra options
        return {"returncode": None, "ok": None, "error_count": 0,
                "first_error": f"not expressible against this gate: {exc}"}
    errors = list(result.get("errors") or [])
    return {
        "returncode": 0 if result.get("ok") else 1,
        "ok": bool(result.get("ok")),
        "error_count": len(errors),
        "first_error": errors[0] if errors else None,
    }


def build(baseline_rev: str = BASELINE_REV) -> dict[str, object]:
    baseline = _load_baseline(baseline_rev)
    import importlib.util

    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("m17947_hardened_gate", HARDENED)
    hardened = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hardened)

    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="m17947-cases-"))
    rows = []
    for label, data in _mutations(payload):
        manifest = tmp / f"{label}.json"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        rows.append({
            "case": label,
            "baseline": _run_gate(baseline.audit, manifest, []),
            "hardened": _run_gate(hardened.audit, manifest, [PACKAGED, False]),
        })

    # Packaged-copy parity cannot be expressed against the baseline gate, which
    # has no notion of a second manifest.
    stale = copy.deepcopy(payload)
    stale["entries"][0]["introduced"] = "0.0.0"
    stale_path = tmp / "stale_packaged.json"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    packaged_case = {
        "case": "stale_packaged_copy",
        "baseline": {"returncode": 0, "ok": True, "error_count": 0, "first_error": None,
                     "note": "gate never reads the packaged copy"},
        "hardened": _run_gate(hardened.audit, SNAPSHOT, [stale_path, True]),
    }

    clean = {
        "case": "clean_manifest",
        "baseline": _run_gate(baseline.audit, SNAPSHOT, []),
        "hardened": _run_gate(hardened.audit, SNAPSHOT, [PACKAGED, True]),
    }
    escaped_baseline = sum(1 for row in rows if row["baseline"]["returncode"] == 0)
    escaped_hardened = sum(1 for row in rows if row["hardened"]["returncode"] == 0)
    return {
        "mission": "m17947",
        "mission_origin": "#760 reopen round 2 (m15038 deliverables)",
        "baseline_rev": baseline_rev,
        "baseline_rev_commit": subprocess.check_output(
            ["git", "rev-parse", baseline_rev], cwd=ROOT, text=True
        ).strip(),
        "package_version": payload["package_version"],
        "entries": len(payload["entries"]),
        "method": "both gates' audit() called in-process; returncode = 0 if ok else 1",
        "clean_manifest": clean,
        "fault_cases": rows + [packaged_case],
        "summary": {
            "faults_injected": len(rows) + 1,
            "baseline_escaped": escaped_baseline,
            "baseline_escaped_note": "counted over the in-manifest faults; the "
                                     "packaged-copy case is not expressible",
            "hardened_escaped": escaped_hardened,
            "clean_manifest_still_passes": bool(clean["hardened"]["ok"]),
        },
        "reproduce": "PYTHONPATH=src python scripts/m17947_gate_baseline_compare.py",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-rev", default=BASELINE_REV)
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    record = build(args.baseline_rev)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("artifact is stale -- rerun without --check", file=sys.stderr)
            return 3
        print("artifact matches the checkout")
        return 0
    target = (args.write or OUT).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")
    print(json.dumps(record["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())