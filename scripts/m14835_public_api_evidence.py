#!/usr/bin/env python3
"""Emit ``artifacts/m14835_public_api_evidence.json`` for mission m14835.

The mission's only code change was the eight-line re-export of nine
``tensorfem.cohesive3d`` symbols from the package root.  Prose describing that is
not evidence: this script derives the record from the checkout itself --
the symbol table read off the live module, the checked snapshot entries, the
line counts, and the SHA-256 of every file the contract depends on -- so the
artifact cannot outlive the code it describes (``tests/
test_m14835_cohesive3d_public_api.py`` fails once a hash no longer matches).

Usage::

    PYTHONPATH=src python scripts/m14835_public_api_evidence.py [--check]

``--check`` recomputes and exits 3 when the committed artifact is stale instead
of rewriting it, mirroring ``benchmarks/cohesive3d_verification.py --check``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "m14835_public_api_evidence.json"

COHESIVE_MODULE = "tensorfem.cohesive3d"
TRACKED_SOURCES = (
    "src/tensorfem/__init__.py",
    "src/tensorfem/cohesive3d.py",
    "tests/test_cohesive3d.py",
    "tests/test_m14835_cohesive3d_public_api.py",
    "api/public-api-v1.json",
    "src/tensorfem/public_api_manifest.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover - shallow/exported checkout
        return "unknown"


def build() -> dict:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    import tensorfem

    module = importlib.import_module(COHESIVE_MODULE)
    snapshot = json.loads((ROOT / "api/public-api-v1.json").read_text(encoding="utf-8"))
    recorded = {entry["name"]: entry for entry in snapshot["entries"]}

    symbols = []
    for name in module.__all__:
        exported = getattr(tensorfem, name)
        entry = recorded.get(name, {})
        symbols.append(
            {
                "name": name,
                "module": getattr(exported, "__module__", None),
                "reexport_identity_ok": exported is getattr(module, name),
                "in_root_all": name in tensorfem.__all__,
                "snapshot_stability": entry.get("stability"),
                "snapshot_introduced": entry.get("introduced"),
                "snapshot_module": entry.get("module"),
            }
        )

    payload = {
        "mission": "m14835",
        "round": "reopen-1 (tracked as m17421)",
        "schema": "tensorfem.mission-evidence/1",
        "package_version": tensorfem.__version__,
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "cohesive_module": COHESIVE_MODULE,
        "counts": {
            "cohesive_exports": len(module.__all__),
            "root_all": len(tensorfem.__all__),
            "snapshot_entries": len(snapshot["entries"]),
            "root_exports_missing_from_snapshot": sorted(
                set(tensorfem.__all__) - set(recorded)
            ),
        },
        "symbols": symbols,
        "line_counts": {
            rel: len((ROOT / rel).read_text(encoding="utf-8").splitlines())
            for rel in TRACKED_SOURCES
            if (ROOT / rel).suffix == ".py"
        },
        "source_sha256": {rel: sha256(ROOT / rel) for rel in TRACKED_SOURCES},
        "reproduce": "PYTHONPATH=src python scripts/m14835_public_api_evidence.py",
    }
    payload["checks_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on a stale artifact")
    args = parser.parse_args()
    payload = build()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        # The recorded commit is the branch point the evidence was derived at;
        # committing the artifact itself moves HEAD, which must not make the
        # hashes stale.  Adopt the recorded commit/branch before comparing.
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current:
            recorded = json.loads(current)
            payload["commit"] = recorded.get("commit", payload["commit"])
            payload["branch"] = recorded.get("branch", payload["branch"])
            payload["checks_digest"] = hashlib.sha256(
                json.dumps(
                    {k: v for k, v in payload.items() if k != "checks_digest"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if current != text:
            print("artifact is stale -- rerun without --check", file=sys.stderr)
            return 3
        print("artifact matches the checkout")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())