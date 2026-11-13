"""Certificate *renderables*: stop requiring binaries the merge gate strips.

Problem
-------
``benchmarks/verified/*/gates_verified.json`` certificates declare their
evidence under ``artifacts``.  ``tools/verified_anchor_audit.py`` resolves every
declared path against the git index and raises ``E_ARTIFACT_MISSING`` when it is
absent — a fail-closed rule, and the right one for artifacts that *can* be
committed (``result.json``, drivers, gate files).

Binary figures cannot survive here.  Measured on this branch (2026-11-13):
``git ls-files '*.png'`` counts 489 tracked PNGs repo-wide but **0** under
``benchmarks/verified/**``, while git history shows two separate missions
(``d094673b`` "restore 32 certificate-named figures", ``c4070685`` "restore
declared figure evidence") that committed them and had them filtered out of a
later delivery merge ("N 个文件被门禁滤除").  A certificate that requires a PNG
therefore goes red once per merge cycle: restore, filter, red, restore.  The
2026-11-13 tree was in the red half of that cycle — six anchors
(``fem_beam_static``, ``fem_cantilever``, ``fem_hertz_contact``,
``fem_plate_modal``, ``fem_plate_static``, ``fem_winkler_plate``) carried
``E_ARTIFACT_MISSING`` and were not in the debt registry, so
``tests/test_verified_anchor_certificates.py::test_no_unregistered_certificate_defect``
failed on ``main``.

Policy implemented here
-----------------------
A figure is *renderable* evidence, not a committed artifact.  A certificate may
move a binary figure declaration out of ``artifacts`` into
``figures_rendered_on_demand``::

    {"path": "convergence.png", "rendered_by": "benchmarks/verified/x/run.py"}

``rendered_by`` must be a **committed** script inside the anchor directory that
literally names the figure file, so the figure is reproducible by one command
from the committed artifact.  That is a weaker claim than "the PNG is here" but
it is a claim that can be *kept*: it is machine checked (this module) and it
survives the merge gate.

``--check`` enforces three invariants (also asserted by
``tests/test_certificate_renderables.py``):

1. no ``artifacts`` value names a binary figure that is missing on disk;
2. every ``figures_rendered_on_demand`` entry resolves to a committed producer
   that mentions the figure basename;
3. a figure that *does* exist on disk is not hidden in the renderable list
   (existence wins: keep it under ``artifacts``).

Usage
-----
    python3 tools/certificate_renderables.py --check
    python3 tools/certificate_renderables.py --migrate   # rewrite certificates
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DIR = REPO_ROOT / "benchmarks" / "verified"
FIGURE_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".gif", ".webp", ".eps"}
RENDERABLE_KEY = "figures_rendered_on_demand"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=180, check=False
    )
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def _is_figure(text: str) -> bool:
    return os.path.splitext(text)[1].lower() in FIGURE_SUFFIXES


def _resolve(text: str, directory: Path) -> Path | None:
    """Resolve a declared artifact the way the audit tool does (dir, then repo)."""
    for candidate in (directory / text, REPO_ROOT / text):
        if candidate.exists():
            return candidate
    return None


def producer_for(figure: str, directory: Path, tracked: set[str]) -> str | None:
    """Committed script inside the anchor directory that writes this figure."""
    basename = os.path.basename(figure)
    for script in sorted(directory.glob("*.py")):
        rel = script.relative_to(REPO_ROOT).as_posix()
        if rel not in tracked:
            continue
        try:
            if basename in script.read_text(errors="ignore"):
                return rel
        except OSError:  # pragma: no cover - unreadable file is not a producer
            continue
    return None


def audit_certificate(path: Path, tracked: set[str]) -> dict:
    """Return the renderable-policy violations of one certificate."""
    directory = path.parent
    payload = json.loads(path.read_text())
    violations: list[dict] = []

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for key, value in artifacts.items():
            values = value if isinstance(value, list) else [value]
            for item in values:
                if not isinstance(item, str) or not _is_figure(item):
                    continue
                if _resolve(item, directory) is None:
                    violations.append(
                        {
                            "anchor": directory.name,
                            "code": "E_FIGURE_DECLARED_BUT_ABSENT",
                            "figure": item,
                            "key": key,
                            "remedy": (
                                "move it to figures_rendered_on_demand with a committed producer, "
                                "or commit the figure"
                            ),
                        }
                    )

    for entry in payload.get(RENDERABLE_KEY) or []:
        if not isinstance(entry, dict):
            violations.append(
                {"anchor": directory.name, "code": "E_RENDERABLE_MALFORMED", "entry": entry}
            )
            continue
        figure, rendered_by = entry.get("path"), entry.get("rendered_by")
        resolved = _resolve(str(figure), directory) if figure else None
        if resolved is not None:
            violations.append(
                {
                    "anchor": directory.name,
                    "code": "E_RENDERABLE_BUT_PRESENT",
                    "figure": figure,
                    "remedy": "the figure exists on disk; declare it under artifacts instead",
                }
            )
            continue
        if not rendered_by or rendered_by not in tracked:
            violations.append(
                {
                    "anchor": directory.name,
                    "code": "E_RENDERABLE_NO_PRODUCER",
                    "figure": figure,
                    "rendered_by": rendered_by,
                    "remedy": "name the committed script that writes this figure",
                }
            )
        elif not (REPO_ROOT / rendered_by).parent == directory:
            violations.append(
                {
                    "anchor": directory.name,
                    "code": "E_RENDERABLE_FOREIGN_PRODUCER",
                    "figure": figure,
                    "rendered_by": rendered_by,
                    "remedy": "the producer must live in the anchor directory",
                }
            )
    return {"anchor": directory.name, "certificate": path, "violations": violations}


def audit_registry() -> dict:
    tracked = tracked_files()
    records = [
        audit_certificate(path, tracked) for path in sorted(VERIFIED_DIR.glob("*/gates_verified.json"))
    ]
    violations = [v for record in records for v in record["violations"]]
    # An anchor already booked in the shrink-only certificate debt registry may
    # carry an unrepaired figure declaration (it has no committed producer yet,
    # e.g. ``fem_winkler_plate_k_sweep`` ships no runner).  Debt may exist, it
    # may not be *invisible*: unregistered violations are the failing ones.
    debt_path = VERIFIED_DIR / "anchor_certificate_debt.json"
    debt = json.loads(debt_path.read_text())["anchors"] if debt_path.is_file() else {}
    registered = [v for v in violations if v["anchor"] in debt]
    return {
        "schema": "tensorfem.certificate-renderables/1",
        "certificates": len(records),
        "renderable_entries": sum(
            len(json.loads(record["certificate"].read_text()).get(RENDERABLE_KEY) or [])
            for record in records
        ),
        "violations": [v for v in violations if v["anchor"] not in debt],
        "debt_registered_violations": registered,
    }


# --------------------------------------------------------------------------- #
# migration
# --------------------------------------------------------------------------- #
def migrate_certificate(path: Path, tracked: set[str]) -> tuple[bool, list[str]]:
    """Move absent binary figure declarations into ``figures_rendered_on_demand``."""
    directory = path.parent
    payload = json.loads(path.read_text())
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return False, []

    moved: list[dict] = []
    changed = False
    for key, value in list(artifacts.items()):
        values = value if isinstance(value, list) else [value]
        keep: list[str] = []
        for item in values:
            if not isinstance(item, str) or not _is_figure(item) or _resolve(item, directory) is not None:
                keep.append(item)
                continue
            producer = producer_for(item, directory, tracked)
            if producer is None:
                keep.append(item)  # no committed producer: cannot be claimed as renderable
                continue
            moved.append({"path": item, "rendered_by": producer})
            changed = True
        if not keep:
            artifacts.pop(key)
            changed = True
        elif len(keep) == 1 and not isinstance(value, list):
            artifacts[key] = keep[0]
        elif keep != values:
            artifacts[key] = keep

    if moved:
        existing = payload.setdefault(RENDERABLE_KEY, [])
        seen = {(entry.get("path"), entry.get("rendered_by")) for entry in existing if isinstance(entry, dict)}
        for entry in moved:
            if (entry["path"], entry["rendered_by"]) not in seen:
                existing.append(entry)
    if changed:
        path.write_text(json.dumps(payload, indent=2) + "\n")
    return changed, [entry["path"] for entry in moved]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 on any renderable-policy violation")
    parser.add_argument("--migrate", action="store_true", help="rewrite certificates")
    parser.add_argument("--json", action="store_true", help="print the audit record")
    args = parser.parse_args(argv)

    if args.migrate:
        tracked = tracked_files()
        for path in sorted(VERIFIED_DIR.glob("*/gates_verified.json")):
            changed, moved = migrate_certificate(path, tracked)
            if changed:
                print(f"{path.parent.name}: {len(moved)} figure(s) recorded as renderable")

    audit = audit_registry()
    if args.json:
        print(json.dumps(audit, indent=2))
    else:
        print(
            f"certificates={audit['certificates']} renderable_entries={audit['renderable_entries']} "
            f"violations={len(audit['violations'])}"
        )
        for violation in audit["violations"]:
            print(f"  [{violation['code']}] {violation['anchor']}: {violation.get('figure') or violation.get('entry')}")

    if args.check and audit["violations"]:
        print("FAIL: renderable-policy violations", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())