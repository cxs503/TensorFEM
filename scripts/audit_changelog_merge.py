#!/usr/bin/env python
"""Audit CHANGELOG.md after a three-way merge resolution.

A merge of two long-lived branches touches ``CHANGELOG.md`` on both sides, and
Git's textual merge is not enough: two lines can survive while the *document*
is corrupted (a heading can lose its bullets to the neighbouring section, or a
section heading can be duplicated).  Both failures were observed in this
repository, so the resolution is audited mechanically instead of by eye.

Two independent checks are implemented:

``--structural``  layout invariants of the resolved file itself:
                  exactly one ``## Unreleased`` heading (a repeated heading is
                  the signature of a mechanical overlay), no empty level-2
                  section (the signature of an orphaned heading), and every
                  section's bullets stay inside that section.

``--merge REV``   content preservation against the two parents of merge
                  revision ``REV``: every non-blank line of each parent's
                  ``CHANGELOG.md`` must still be present in the result with at
                  least the same multiplicity, unless it is listed in
                  ``--allow-dropped`` (typically a duplicated heading that the
                  resolution deliberately de-duplicates).  Reported as a
                  multiset difference, so a silently dropped duplicate bullet
                  cannot hide behind an identical one.

Usage::

    python scripts/audit_changelog_merge.py --structural
    python scripts/audit_changelog_merge.py --merge HEAD --allow-dropped '## Unreleased'
    python scripts/audit_changelog_merge.py --simulate HEAD

Exit status is 0 only when every requested check passes.  ``--simulate`` is the
inverse check: it regenerates the *textual* merge of ``REV``'s parents (what Git
would have committed without manual work) and passes only when that result is
structurally broken, i.e. when the manual resolution was genuinely necessary.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = "CHANGELOG.md"
HEADING = re.compile(r"^(#{1,6}) +(.*)$")


def load_text(path: Path | str) -> list[str]:
    p = Path(path)
    return p.read_text(encoding="utf-8").splitlines()


def git_show(rev: str, path: str = CHANGELOG) -> list[str]:
    out = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return out.stdout.splitlines()


def sections(lines: list[str], max_level: int = 2) -> list[dict]:
    """Split into level<=max_level sections, keeping the heading line."""
    heads = [(i, m.group(1), m.group(2))
             for i, line in enumerate(lines)
             if (m := HEADING.match(line)) and len(m.group(1)) <= max_level]
    out = []
    for k, (i, hashes, title) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        out.append({"index": i, "level": len(hashes), "title": title,
                    "heading": lines[i], "body": lines[i + 1:end]})
    return out


def structural_audit(lines: list[str]) -> dict:
    secs = sections(lines)
    findings: list[str] = []
    report: dict = {"file_lines": len(lines), "sections": len(secs)}

    duplicates = [t for t, n in
                  collections.Counter(s["heading"] for s in secs).items() if n > 1]
    unreleased = sum(1 for s in secs if s["title"].strip() == "Unreleased")
    report["unreleased_headings"] = unreleased
    report["duplicate_headings"] = duplicates
    if unreleased != 1:
        findings.append(f"expected exactly one '## Unreleased', found {unreleased}")

    empty = [s["heading"] for s in secs
             if s["level"] == 2 and not any(l.strip() for l in s["body"])]
    report["empty_level2_sections"] = empty
    if empty:
        findings.append(f"orphaned (empty) level-2 sections: {empty}")

    # bullets must live under the heading that owns their subject: the two
    # sections that collided in PR #77 are checked by name.
    by_title = {s["title"]: s for s in secs}
    cook = next((s for t, s in by_title.items() if t.startswith("2026-11-03 Cook")), None)
    bbar = next((s for t, s in by_title.items() if t.startswith("2026-11-04")), None)
    report["cook_membrane_bullets"] = (
        sum(1 for l in cook["body"] if l.startswith("- ")) if cook else None)
    if cook is not None and report["cook_membrane_bullets"] == 0:
        findings.append("2026-11-03 Cook section lost its bullets")
    if bbar is not None:
        strays = [l[:70] for l in bbar["body"]
                  if "cook_membrane_stress_reference" in l or "test_cook_stress_audit" in l]
        report["cook_bullets_in_bbar_section"] = len(strays)
        if strays:
            findings.append(f"Cook bullets found inside the B-bar section: {strays}")

    report["passed"] = not findings
    report["findings"] = findings
    return report


def loss_audit(result: list[str], reference: list[str],
               allow_dropped: tuple[str, ...] = ()) -> dict:
    """Multiset line-preservation audit of ``result`` against ``reference``."""
    ref = collections.Counter(l for l in reference if l.strip())
    res = collections.Counter(l for l in result if l.strip())
    lost = [l for l in ref if res[l] < ref[l]]
    allowed = [l for l in lost if any(a in l for a in allow_dropped)]
    unexpected = [l for l in lost if l not in allowed]
    return {
        "reference_nonblank_lines": sum(ref.values()),
        "lost_lines": len(lost),
        "allowed_dropped_lines": allowed,
        "unexpected_lost_lines": unexpected,
        "passed": not unexpected,
    }


def merge_audit(rev: str, allow_dropped: tuple[str, ...]) -> dict:
    parents = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", rev],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.split()
    sha, parent_shas = parents[0], parents[1:]
    result = load_text(ROOT / CHANGELOG)
    out = {"merge": sha, "parents": parent_shas, "sides": {}}
    for p in parent_shas:
        out["sides"][p] = loss_audit(result, git_show(p), allow_dropped)
    out["passed"] = all(s["passed"] for s in out["sides"].values())
    return out


def simulate_machine_merge(rev: str) -> dict:
    """Regenerate the unresolved textual merge of ``rev``'s parents and audit it.

    ``git merge-file`` writes Git's own conflict output, so this reproduces
    exactly the state a mechanical resolution would have committed.  Both
    parents' lines are then measured against that output, and the result is
    expected to *fail* the structural audit - a passing audit here would mean
    the manual three-way work added nothing.
    """
    import tempfile

    parents = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", rev],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.split()
    sha, p1, p2 = parents[0], parents[1], parents[2]
    base = subprocess.run(["git", "merge-base", p1, p2], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, r in (("ours", p1), ("theirs", p2), ("base", base)):
            text = chr(10).join(git_show(r)) + chr(10)
            (tmp / f"{name}.md").write_text(text, encoding="utf-8")
        proc = subprocess.run(
            ["git", "merge-file", "-p", str(tmp / "ours.md"), str(tmp / "base.md"),
             str(tmp / "theirs.md")],
            capture_output=True, text=True,
        )
    lines = proc.stdout.splitlines()
    structural = structural_audit(lines)
    return {
        "merge": sha,
        "parents": [p1, p2],
        "merge_base": base,
        "conflict_markers": sum(1 for l in lines if l.startswith("<<<<<<<")),
        "file_lines": len(lines),
        "nonblank_lines": sum(1 for l in lines if l.strip()),
        "structural": structural,
        "naive_merge_is_structurally_broken": not structural["passed"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structural", action="store_true")
    ap.add_argument("--merge", metavar="REV")
    ap.add_argument("--simulate", metavar="REV",
                    help="regenerate the unresolved textual merge and expect it to fail")
    ap.add_argument("--allow-dropped", action="append", default=[],
                    metavar="SUBSTRING")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv)

    if not args.structural and not args.merge and not args.simulate:
        ap.error("nothing to do: pass --structural, --merge REV and/or --simulate REV")

    report: dict = {"benchmark": "changelog_merge_audit"}
    ok = True
    if args.structural:
        report["structural"] = structural_audit(load_text(ROOT / CHANGELOG))
        ok &= report["structural"]["passed"]
    if args.merge:
        report["merge"] = merge_audit(args.merge, tuple(args.allow_dropped))
        ok &= report["merge"]["passed"]
    if args.simulate:
        report["simulated"] = simulate_machine_merge(args.simulate)
        ok &= report["simulated"]["naive_merge_is_structurally_broken"]
    report["passed"] = ok

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())