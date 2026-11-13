#!/usr/bin/env python3
"""Release-to-release compatibility gate for ``api/public-api-v1.json``.

``scripts/audit_public_api.py`` answers *"does the snapshot describe the code in
the tree right now?"*.  It is a **point-in-time** check and never compares two
releases, so the promise the manifest makes in its own ``policy`` block --
``stable = "semver compatibility"`` -- was unenforced.  A stable name could be
deleted, renamed, moved to another module or demoted to ``experimental`` and
every existing gate would still pass, because manifest and code are edited
together and the snapshot is always "consistent".

This module adds the missing **release-to-release** axis:

    change in the manifest  ->  minimum version bump the policy demands

Rules, one per policy sentence stored in the manifest:

* ``stable``             -- "semver compatibility": removal, object rename,
  module move or demotion needs a major bump.
* ``beta``               -- "deprecate before removal": removal/rename/move needs
  a major bump; demotion is allowed in a minor release (that *is* the documented
  way to start a deprecation).
* ``experimental``       -- "may change with release notes": a break needs a
  minor bump (released and noted, never slipped into a patch).
* ``internal_candidate`` -- "kept for compatibility": break needs a minor bump.

Two execution modes, because "additive" and "breaking" mean different things
before and after the version number is written:

* ``--mode pr`` (default) -- pull-request gate.  Additions and promotions are
  reported as *accumulated minor-release delta* and never block: the version is
  bumped at release, not on every PR.  Any **breaking** change must already be
  covered by the declared bump.
* ``--mode release`` -- release gate.  The whole delta against the reviewed
  baseline (additions included) must be covered by the declared bump, so a new
  release cannot ship unreleased API surface under an unchanged number.

Pure stdlib, no ``tensorfem`` import: cheap enough for every PR, and it cannot
be perturbed by the code it is checking.

Usage
-----
    python scripts/public_api_compat.py --base <spec> [--head <spec>] [--json]
    # <spec> is a filesystem path or git:<rev>:<path>
    python scripts/public_api_compat.py \
        --base git:d4cf378e:api/public-api-v1.json --head api/public-api-v1.json
    python scripts/public_api_compat.py --mode release --base git:... --head ...
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: Bump magnitudes, weakest first.
BUMP_ORDER = {"none": 0, "patch": 1, "minor": 2, "major": 3}

#: Stability classes, weakest guarantee first.
STABILITY_ORDER = {"internal_candidate": 0, "experimental": 1, "beta": 2, "stable": 3}

#: Change kinds that withdraw or move surface a user may already depend on.
BREAKING_KINDS = {"removed", "module_moved", "object_renamed"}

SEMVER = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)(?:\.(?P<patch>[0-9]+))?$")


class ManifestError(ValueError):
    """Raised when a manifest cannot serve as a compatibility baseline."""


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_manifest(spec: str) -> dict[str, Any]:
    """Load a manifest from a filesystem path or a ``git:<rev>:<path>`` spec."""
    if spec.startswith("git:"):
        try:
            _, rev, path = spec.split(":", 2)
        except ValueError as exc:
            raise ManifestError(f"bad git spec {spec!r}; expected git:<rev>:<path>") from exc
        try:
            raw = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{rev}:{path}"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ManifestError(f"cannot read {spec} from git: {exc}") from exc
    else:
        raw = Path(spec).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{spec}: not valid JSON ({exc})") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise ManifestError(f"{spec}: manifest must be an object with an 'entries' list")
    return payload


def _entries(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    """Index entries by name, rejecting manifests this gate cannot compare."""
    by_name: dict[str, dict[str, Any]] = {}
    for entry in payload["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ManifestError(f"{label}: every entry needs a string 'name'")
        name = entry["name"]
        if name in by_name:
            raise ManifestError(f"{label}: duplicate name {name!r}")
        if entry.get("stability") not in STABILITY_ORDER:
            raise ManifestError(
                f"{label}: {name!r} has invalid stability {entry.get('stability')!r}"
            )
        if not isinstance(entry.get("module"), str):
            raise ManifestError(f"{label}: {name!r} has no module")
        by_name[name] = entry
    return by_name


def parse_version(version: str) -> tuple[int, int, int]:
    match = SEMVER.match(version or "")
    if not match:
        raise ManifestError(f"invalid package_version {version!r}")
    return (int(match["major"]), int(match["minor"]), int(match["patch"] or 0))


def version_bump(base: str, head: str) -> str:
    """Classify the bump ``base -> head`` performs (``none`` when unchanged)."""
    b, h = parse_version(base), parse_version(head)
    if h == b:
        return "none"
    if h[0] != b[0]:
        return "major"
    if h[1] != b[1]:
        return "minor"
    return "patch"


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #
def diff_manifests(base: dict[str, Any], head: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered list of API changes between two manifests."""
    before = _entries(base, "base")
    after = _entries(head, "head")
    changes: list[dict[str, Any]] = []

    for name, entry in after.items():
        old = before.get(name)
        if old is None:
            changes.append({"kind": "added", "name": name, "stability": entry["stability"]})
            continue
        if old["stability"] != entry["stability"]:
            changes.append(
                {
                    "kind": "stability_changed",
                    "name": name,
                    "from": old["stability"],
                    "to": entry["stability"],
                }
            )
        if old["module"] != entry["module"]:
            changes.append(
                {
                    "kind": "module_moved",
                    "name": name,
                    "from": old["module"],
                    "to": entry["module"],
                    "stability": entry["stability"],
                }
            )
        if old.get("object") != entry.get("object"):
            changes.append(
                {
                    "kind": "object_renamed",
                    "name": name,
                    "from": old.get("object"),
                    "to": entry.get("object"),
                    "stability": entry["stability"],
                }
            )
    for name, entry in before.items():
        if name not in after:
            changes.append({"kind": "removed", "name": name, "stability": entry["stability"]})
    base_order = [entry["name"] for entry in base["entries"]]
    head_order = [entry["name"] for entry in head["entries"]]
    if base_order != head_order and set(before) == set(after):
        changes.append({"kind": "reordered", "name": "<__all__ order>"})
    return changes


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def change_bump(change: dict[str, Any]) -> str:
    """Version bump a single API change demands under the policy sentence."""
    kind, stability = change["kind"], change.get("stability")
    if kind == "added":
        return "minor"
    if kind == "reordered":
        return "patch"
    if kind == "stability_changed":
        if STABILITY_ORDER[change["to"]] >= STABILITY_ORDER[change["from"]]:
            return "patch"  # a promotion only strengthens the guarantee
        return "major" if change["from"] == "stable" else "minor"
    if kind in BREAKING_KINDS:
        return "major" if stability in {"stable", "beta"} else "minor"
    return "major"  # pragma: no cover - defensive


def is_breaking(change: dict[str, Any]) -> bool:
    """A change is breaking when it needs more than the accumulated minor delta."""
    if change["kind"] in BREAKING_KINDS:
        return True
    if change["kind"] == "stability_changed":
        return STABILITY_ORDER[change["to"]] < STABILITY_ORDER[change["from"]]
    return False


def required_bump(
    changes: list[dict[str, Any]],
    *,
    breaking_only: bool = False,
    zero_major_minor_is_breaking: bool = False,
) -> str:
    """Weakest bump that keeps the promise for ``changes``.

    ``breaking_only`` restricts the calculation to changes that withdraw surface
    (the pull-request gate), so purely additive PRs do not have to carry a
    version bump.  ``zero_major_minor_is_breaking`` applies the SemVer 0.x
    convention where the minor position carries breaking changes.
    """
    need = "none"
    for change in changes:
        if breaking_only and not is_breaking(change):
            continue
        bumped = change_bump(change)
        if zero_major_minor_is_breaking and bumped == "major":
            bumped = "minor"
        if BUMP_ORDER[bumped] > BUMP_ORDER[need]:
            need = bumped
    return need


def classify(
    base: dict[str, Any],
    head: dict[str, Any],
    *,
    mode: str = "pr",
    zero_major_minor_is_breaking: bool = False,
) -> dict[str, Any]:
    """Full verdict for one release transition."""
    if mode not in {"pr", "release"}:
        raise ManifestError(f"unknown mode {mode!r}")
    base_version, head_version = base.get("package_version"), head.get("package_version")
    changes = diff_manifests(base, head)
    actual = version_bump(base_version, head_version)
    need = required_bump(
        changes,
        breaking_only=(mode == "pr"),
        zero_major_minor_is_breaking=zero_major_minor_is_breaking,
    )
    compliant = BUMP_ORDER[actual] >= BUMP_ORDER[need]

    counts: dict[str, int] = {}
    for change in changes:
        counts[change["kind"]] = counts.get(change["kind"], 0) + 1

    violations = []
    for change in changes:
        if mode == "pr" and not is_breaking(change):
            continue
        bumped = change_bump(change)
        if zero_major_minor_is_breaking and bumped == "major":
            bumped = "minor"
        if BUMP_ORDER[bumped] > BUMP_ORDER[actual]:
            violations.append({**change, "needs": bumped, "actual_bump": actual})

    breaking = [change for change in changes if is_breaking(change)]
    return {
        "mode": mode,
        "base_version": base_version,
        "head_version": head_version,
        "entries": {"base": len(base["entries"]), "head": len(head["entries"])},
        "changes": changes,
        "counts": counts,
        "breaking": breaking,
        "accumulated_minor_delta": counts.get("added", 0),
        "required_bump": need,
        "actual_bump": actual,
        "compliant": compliant,
        "violations": violations,
        "zero_major_minor_is_breaking": zero_major_minor_is_breaking,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="baseline manifest path or git:<rev>:<path>")
    parser.add_argument("--head", default=str(ROOT / "api/public-api-v1.json"))
    parser.add_argument("--mode", choices=("pr", "release"), default="pr")
    parser.add_argument("--json", action="store_true", help="emit the full verdict as JSON")
    parser.add_argument(
        "--zero-major-minor-is-breaking",
        action="store_true",
        help="accept a 0.x minor bump in place of a major bump (SemVer 0.x convention)",
    )
    args = parser.parse_args(argv)
    try:
        verdict = classify(
            load_manifest(args.base),
            load_manifest(args.head),
            mode=args.mode,
            zero_major_minor_is_breaking=args.zero_major_minor_is_breaking,
        )
    except ManifestError as exc:
        print(f"API compatibility gate: ERROR; {exc}")
        return 2
    if args.json:
        print(json.dumps(verdict, indent=2, sort_keys=True))
    else:
        print(
            f"API compatibility gate [{verdict['mode']}]: {'PASS' if verdict['compliant'] else 'FAIL'}; "
            f"{verdict['base_version']} -> {verdict['head_version']}; "
            f"requires {verdict['required_bump']} bump, found {verdict['actual_bump']}; "
            f"changes {verdict['counts']}; breaking {len(verdict['breaking'])}"
        )
        for violation in verdict["violations"]:
            print(f"VIOLATION: {violation['kind']} {violation['name']} needs {violation['needs']}")
    return 0 if verdict["compliant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
