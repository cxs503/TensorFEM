#!/usr/bin/env python3
"""Fail closed when TensorFEM's top-level API drifts from its reviewed snapshot."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import tensorfem

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "api/public-api-v1.json"
VALID_STABILITY = {"stable", "beta", "experimental", "internal_candidate"}


def audit(path: Path = MANIFEST) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    expected = [entry["name"] for entry in entries]
    actual = list(tensorfem.__all__)
    errors: list[str] = []
    if expected != actual:
        errors.append(f"top-level __all__ drift: expected {len(expected)}, found {len(actual)}")
    if len(expected) != len(set(expected)):
        errors.append("manifest contains duplicate names")
    for entry in entries:
        name = entry["name"]
        if entry.get("stability") not in VALID_STABILITY:
            errors.append(f"{name}: invalid stability")
        if not hasattr(tensorfem, name):
            errors.append(f"{name}: missing top-level object")
            continue
        module = getattr(getattr(tensorfem, name), "__module__", "tensorfem")
        if module != entry.get("module"):
            errors.append(f"{name}: module drift {entry.get('module')} -> {module}")
    if payload.get("package_version") != tensorfem.__version__:
        errors.append("manifest package_version differs from tensorfem.__version__")
    counts = Counter(entry.get("stability") for entry in entries)
    return {"ok": not errors, "errors": errors, "count": len(entries), "stability": dict(counts)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else
          f"API audit: {'PASS' if result['ok'] else 'FAIL'}; {result['count']} symbols; {result['stability']}")
    for error in result["errors"]:
        print(f"ERROR: {error}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
