#!/usr/bin/env python3
"""Fail closed when TensorFEM's top-level API drifts from its reviewed snapshot.

The gate has three independent failure surfaces:

1. **Runtime drift** -- ``tensorfem.__all__`` / ``__module__`` no longer match the
   reviewed snapshot.
2. **Contract shape** -- the snapshot itself is malformed (unknown or missing
   keys, bad ``stability``/``introduced`` values, wrong ``schema`` id, a stability
   class with no ``policy`` sentence).  The rules mirror
   ``api/public-api-manifest.schema.json`` but are enforced here so the fast gate
   never needs a JSON-Schema runtime (``jsonschema`` is used as a *second*
   opinion when it happens to be installed).
3. **Packaged-copy parity** -- the wheel ships
   ``src/tensorfem/public_api_manifest.json``; if that twin drifts from the
   reviewed snapshot the published package advertises a different API than the
   one that was reviewed, so the audit fails even when the sources are clean.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import tensorfem

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "api/public-api-v1.json"
PACKAGED = ROOT / "src/tensorfem/public_api_manifest.json"
SCHEMA_DOC = ROOT / "api/public-api-manifest.schema.json"

SCHEMA_ID = "tensorfem.public-api-manifest/1.0"
VALID_STABILITY = {"stable", "beta", "experimental", "internal_candidate"}
TOP_LEVEL_KEYS = {"schema", "package_version", "policy", "entries"}
ENTRY_KEYS = {"name", "module", "object", "stability", "introduced"}

SEMVER = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MODULE = re.compile(r"^tensorfem(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _load(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"{label}: missing manifest {path}")
        return None
    except json.JSONDecodeError as exc:
        errors.append(f"{label}: not valid JSON ({exc})")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label}: top level must be an object")
        return None
    return payload


def _check_contract(payload: dict[str, Any], label: str, errors: list[str]) -> list[dict[str, Any]]:
    """Validate the manifest against the reviewed contract (closed key set)."""
    unknown = set(payload) - TOP_LEVEL_KEYS
    if unknown:
        errors.append(f"{label}: unknown top-level keys {sorted(unknown)}")
    missing = TOP_LEVEL_KEYS - set(payload)
    if missing:
        errors.append(f"{label}: missing top-level keys {sorted(missing)}")

    if payload.get("schema") != SCHEMA_ID:
        errors.append(f"{label}: schema id {payload.get('schema')!r} != {SCHEMA_ID!r}")

    version = payload.get("package_version")
    if not isinstance(version, str) or not SEMVER.match(version):
        errors.append(f"{label}: invalid package_version {version!r}")
    elif version != tensorfem.__version__:
        errors.append("manifest package_version differs from tensorfem.__version__")

    policy = payload.get("policy")
    if not isinstance(policy, dict):
        errors.append(f"{label}: policy must be an object")
        policy = {}
    else:
        unknown_policy = set(policy) - VALID_STABILITY
        if unknown_policy:
            errors.append(f"{label}: unknown policy classes {sorted(unknown_policy)}")
        for stability in sorted(VALID_STABILITY - set(policy)):
            errors.append(f"{label}: stability class {stability!r} has no policy sentence")
        for stability, sentence in policy.items():
            if not isinstance(sentence, str) or not sentence.strip():
                errors.append(f"{label}: policy[{stability!r}] must be a non-empty string")

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append(f"{label}: entries must be a non-empty list")
        return []

    clean: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        where = f"{label}: entry[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} is not an object")
            continue
        extra = set(entry) - ENTRY_KEYS
        if extra:
            errors.append(f"{where} has unknown keys {sorted(extra)}")
        absent = ENTRY_KEYS - set(entry)
        if absent:
            errors.append(f"{where} is missing keys {sorted(absent)}")
        name = entry.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            errors.append(f"{where} has invalid name {name!r}")
            where = f"{label}: entry[{index}] {name}"
        if entry.get("stability") not in VALID_STABILITY:
            errors.append(f"{where}: invalid stability {entry.get('stability')!r}")
        elif entry.get("stability") not in policy:
            errors.append(f"{where}: stability {entry.get('stability')!r} has no policy sentence")
        module = entry.get("module")
        if not isinstance(module, str) or not MODULE.match(module):
            errors.append(f"{where}: invalid module {module!r}")
        obj = entry.get("object")
        if not isinstance(obj, str) or not obj.strip():
            errors.append(f"{where}: object must be a non-empty string")
        introduced = entry.get("introduced")
        if not isinstance(introduced, str) or not SEMVER.match(introduced):
            errors.append(f"{where}: invalid introduced {introduced!r}")
        if not (extra or absent) and isinstance(name, str):
            clean.append(entry)
    return clean


def _jsonschema_second_opinion(payload: dict[str, Any], errors: list[str]) -> str:
    """Cross-check with the published JSON Schema when ``jsonschema`` is present."""
    try:
        import jsonschema
    except Exception:
        return "unavailable"
    try:
        schema = json.loads(SCHEMA_DOC.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - covered by the schema test
        errors.append(f"schema document unreadable: {exc}")
        return "unavailable"
    validator_cls = getattr(jsonschema, "Draft202012Validator", None) or getattr(
        jsonschema, "Draft7Validator"
    )
    problems = sorted(validator_cls(schema).iter_errors(payload), key=lambda e: list(e.path))
    for problem in problems[:10]:
        location = "/".join(str(part) for part in problem.path) or "<root>"
        errors.append(f"schema violation at {location}: {problem.message}")
    return "passed" if not problems else "failed"


def audit(
    path: Path = MANIFEST,
    packaged: Path | None = PACKAGED,
    check_package: bool = True,
) -> dict[str, object]:
    errors: list[str] = []
    payload = _load(Path(path), "snapshot", errors)
    if payload is None:
        return {"ok": False, "errors": errors, "count": 0, "stability": {}, "packaged": None}

    entries = _check_contract(payload, "snapshot", errors)
    expected = [entry["name"] for entry in entries]
    actual = list(tensorfem.__all__)

    if expected != actual:
        errors.append(f"top-level __all__ drift: expected {len(expected)}, found {len(actual)}")
        only_snapshot = [name for name in expected if name not in actual]
        only_runtime = [name for name in actual if name not in expected]
        if only_snapshot:
            errors.append(f"names in snapshot but not in __all__: {only_snapshot[:10]}")
        if only_runtime:
            errors.append(f"names in __all__ but not in snapshot: {only_runtime[:10]}")
    if len(expected) != len(set(expected)):
        duplicates = sorted(name for name, n in Counter(expected).items() if n > 1)
        errors.append(f"manifest contains duplicate names: {duplicates[:10]}")
    for entry in entries:
        name = entry["name"]
        if not hasattr(tensorfem, name):
            errors.append(f"{name}: missing top-level object")
            continue
        module = getattr(getattr(tensorfem, name), "__module__", "tensorfem")
        if module != entry.get("module"):
            errors.append(f"{name}: module drift {entry.get('module')} -> {module}")

    schema_state = _jsonschema_second_opinion(payload, errors)

    packaged_state: str | None = None
    if check_package and packaged is not None:
        twin = _load(Path(packaged), "packaged", errors)
        if twin is None:
            packaged_state = "missing"
        elif twin == payload:
            packaged_state = "in_sync"
        else:
            packaged_state = "drift"
            errors.append(
                "packaged copy src/tensorfem/public_api_manifest.json drifted from "
                "api/public-api-v1.json; regenerate both copies in one review"
            )

    counts = Counter(entry.get("stability") for entry in entries)
    return {
        "ok": not errors,
        "errors": errors,
        "count": len(entries),
        "stability": dict(counts),
        "packaged": packaged_state,
        "jsonschema": schema_state,
        "schema": payload.get("schema"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--packaged", type=Path, default=PACKAGED)
    parser.add_argument(
        "--no-package-check",
        action="store_true",
        help="skip packaged-copy parity (only for auditing a manifest outside the tree)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(args.manifest, args.packaged, not args.no_package_check)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"API audit: {'PASS' if result['ok'] else 'FAIL'}; "
            f"{result['count']} symbols; {result['stability']}; "
            f"packaged={result['packaged']}; jsonschema={result['jsonschema']}"
        )
    for error in result["errors"]:
        print(f"ERROR: {error}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())