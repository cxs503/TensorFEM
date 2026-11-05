#!/usr/bin/env python3
"""Re-audit the PR #61 reconciliation merge for silent content loss.

The merge that heals PR #61 is a three-way merge of ``main`` into
``agent/m3202``.  Textual cleanliness is not evidence: a mechanical
"take one side" overlay can drop content without ever producing a conflict
marker, and this repository has that failure mode on record.  This script
re-derives, straight from the git object database, the *union obligation* of
the merge and fails closed when any obligation is unmet.

Union obligation, for a file touched on both sides of the merge with base
``B``, main result ``A`` and branch result ``C``:

* **tier 1 — auto-merged files** (git produced the text): the merged line
  multiset must contain every line that is unique to ``A`` or to ``C``.
* **tier 2 — hand-resolved files** (a human/agent rewrote the hunk): the
  merged text must keep every *atom* of both sides (backticked identifiers,
  markdown link targets, file paths, numbers) and every content word, so a
  rewrap is allowed but a dropped assertion is not.

Deletions are audited separately: a file deleted on one side and untouched on
the other must be absent from the merge result; a delete/modify pair must be
reported for review instead of being resolved silently.

Usage::

    python3 artifacts/pr61_merge_union_check.py                 # write evidence
    python3 artifacts/pr61_merge_union_check.py --check          # exit 1 on loss
    python3 artifacts/pr61_merge_union_check.py --merge origin/main --check      # negative control
    python3 artifacts/pr61_merge_union_check.py --merge agent/m3202 --check      # negative control

The two negative controls reproduce the mechanical overlays (main only, branch
only) that the audit has to reject; see docs/pr61-merge-reconciliation.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
MAIN = "origin/main"
BRANCH = "agent/m3202"
MERGE = "HEAD"
OUT = REPO / "artifacts" / "pr61_merge_union_evidence.json"
BASE_OVERRIDE: str | None = None

# Hunks that could not be unioned mechanically and were rewritten by hand.
HAND_RESOLVED = {"CHANGELOG.md", "docs/marine-hydrostatics.md"}

ATOM_RE = re.compile(
    # Prose is hard-wrapped, so a backticked span may be split across lines.
    # Letting the pattern cross newlines would mint whole paragraphs as a single
    # "atom" and report them as lost on any rewrap, so spans stay intra-line.
    r"`[^`\n]+`"                     # backticked identifier / formula
    r"|\]\([^)]+\)"                  # markdown link target
    r"|[\w./-]+\.(?:py|md|json|png)"  # file path
    r"|(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?"  # number
)
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9-]{3,}|[\u4e00-\u9fff]")


def git(*args: str) -> str:
    return subprocess.run(
        ("git",) + args, cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def blob(rev: str, path: str) -> str | None:
    """Text content of ``rev:path``; ``None`` when absent.

    Objects are read as bytes and decoded leniently so binary evidence assets
    (``.png``, ``.pdf``) cannot abort the audit.
    """
    proc = subprocess.run(("git", "show", f"{rev}:{path}"), cwd=REPO, capture_output=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def name_status(base: str, rev: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in git("diff", "--name-status", "--no-renames", base, rev).splitlines():
        if line.strip():
            status, path = line.split("\t")[:2]
            rows[path] = status
    return rows


def atoms(text: str) -> collections.Counter:
    return collections.Counter(ATOM_RE.findall(text))


def words(text: str) -> set:
    return {w.lower() for w in WORD_RE.findall(text)}


def audit_file(path: str, base_lines: list[str], a: str, c: str, m: str) -> dict:
    # Tier choice follows the unit that actually carries meaning. Markdown prose
    # is hard-wrapped, so one sentence spans several lines: a sentence-level
    # union legitimately destroys line identity while losing no assertion. Prose
    # is therefore held to the atom+word obligation. Code and data files are
    # line-atomic (a line is the statement), so they keep the strict
    # line-multiset rule, where a dropped line *is* dropped content.
    prose = path.endswith(".md")
    rec: dict = {
        "path": path,
        "tier": "2-atom+word" if (prose or path in HAND_RESOLVED) else "1-line",
        "tier_rule": "prose-atom+word" if prose else "line-atomic",
    }
    lost_atoms: collections.Counter = collections.Counter()
    lost_lines = 0
    lost_words: set = set()
    if rec["tier"] == "1-line":
        merged_counter = collections.Counter(m.splitlines())
        for side, text in (("main", a), ("branch", c)):
            unique = collections.Counter(text.splitlines()) - collections.Counter(base_lines)
            lost = unique - merged_counter
            rec[f"lines_unique_{side}"] = sum(unique.values())
            rec[f"lines_lost_from_{side}"] = sum(lost.values())
            lost_lines += sum(lost.values())
            if lost:
                rec.setdefault("examples_lost", []).extend(k[:160] for k in list(lost)[:3])
    else:
        merged_atoms, merged_words = atoms(m), words(m)
        for side, text in (("main", a), ("branch", c)):
            side_unique = "\n".join(
                (collections.Counter(text.splitlines())
                 - collections.Counter(base_lines)).elements()
            )
            side_atoms = atoms(side_unique)
            missing = side_atoms - merged_atoms
            lost_atoms += missing
            rec[f"atoms_unique_{side}"] = sum(side_atoms.values())
            rec[f"atoms_lost_from_{side}"] = sum(missing.values())
            missing_words = words(side_unique) - merged_words
            lost_words |= missing_words
            rec[f"words_lost_from_{side}"] = len(missing_words)
            if missing:
                rec.setdefault("examples_lost", []).extend(list(missing)[:3])
        rec["lines_total_merge"] = len(m.splitlines())
    failed = bool(lost_lines or lost_atoms or lost_words)
    rec["verdict"] = "FAIL-union" if failed else "union-preserved"
    rec["_failed"] = failed
    if failed and "examples_lost" not in rec:
        rec["examples_lost"] = sorted(lost_words)[:5]
    return rec


def main() -> int:
    parse = argparse.ArgumentParser()
    parse.add_argument("--check", action="store_true")
    parse.add_argument("--repo", help="repository / worktree to audit")
    parse.add_argument("--merge", help="revision holding the merge result")
    parse.add_argument(
        "--base",
        help="override the merge base (historical fork point). Needed when the "
        "target branch was merged in already: merge-base(main, branch) then "
        "collapses onto main and every audit degenerates to zero files.",
    )
    parse.add_argument("--main", help="revision of the target branch")
    parse.add_argument(
        "--branch",
        help="revision of the PR source branch as it was BEFORE the merge "
        "(default: agent/m3202; pass the pre-merge head to audit the merge "
        "commit itself, otherwise merge-base collapses onto main and the "
        "audit is vacuous)",
    )
    parse.add_argument("--out", help="JSON evidence output path")
    args = parse.parse_args()

    global REPO, MAIN, BRANCH, MERGE, OUT, BASE_OVERRIDE
    if args.repo:
        REPO = pathlib.Path(args.repo).resolve()
    if args.main:
        MAIN = args.main
    if args.branch:
        BRANCH = args.branch
    if args.base:
        BASE_OVERRIDE = args.base
    if args.merge:
        MERGE = args.merge
    OUT = pathlib.Path(args.out) if args.out else REPO / "artifacts" / "pr61_merge_union_evidence.json"

    base = (BASE_OVERRIDE or git("merge-base", MAIN, BRANCH).strip()).strip()
    base = git("rev-parse", base).strip()
    shas = {
        "merge_base": base,
        "main": git("rev-parse", MAIN).strip(),
        "branch": git("rev-parse", BRANCH).strip(),
        "merge": git("rev-parse", MERGE).strip(),
        "merge_tree": git("rev-parse", f"{MERGE}^{{tree}}").strip(),
    }
    side_main, side_branch = name_status(base, MAIN), name_status(base, BRANCH)
    both = sorted(set(side_main) & set(side_branch))

    files: list[dict] = []
    failures: list[str] = []
    single_side_content_checked = 0
    for path in both:
        text_a, text_c, text_m = blob(MAIN, path), blob(BRANCH, path), blob(MERGE, path)
        if text_m is None:
            failures.append(f"{path}: absent from merge result")
            files.append({"path": path, "verdict": "FAIL-absent"})
            continue
        if text_a is None or text_c is None:
            files.append({"path": path, "verdict": "modify-delete-pair (review)"})
            continue
        rec = audit_file(path, (blob(base, path) or "").splitlines(), text_a, text_c, text_m)
        if rec.pop("_failed"):
            failures.append(
                f"{path}: lost {rec.get('lines_lost_from_main', 0)}/{rec.get('lines_lost_from_branch', 0)} "
                f"lines, {rec.get('atoms_lost_from_main', 0)}/{rec.get('atoms_lost_from_branch', 0)} atoms, "
                f"{rec.get('words_lost_from_main', 0)}/{rec.get('words_lost_from_branch', 0)} words"
            )
        files.append(rec)

    deletions: list[dict] = []
    for label, side, other_map in (("main", side_main, side_branch), ("branch", side_branch, side_main)):
        for path, status in side.items():
            if status != "D":
                continue
            rec = {
                "deleted_by": label,
                "path": path,
                "touched_by_other_side": bool(other_map.get(path, "D") != "D"),
                "present_in_merge": blob(MERGE, path) is not None,
            }
            if rec["touched_by_other_side"]:
                rec["verdict"] = "modify-delete-needs-review"
            elif rec["present_in_merge"]:
                rec["verdict"] = "FAIL-deletion-not-propagated"
                failures.append(f"{path}: deletion by {label} not propagated")
            else:
                rec["verdict"] = "deletion-propagated"
            deletions.append(rec)

    for path in sorted(set(side_main) ^ set(side_branch)):
        want = (side_main.get(path) or side_branch.get(path)) != "D"
        got = blob(MERGE, path)
        if (got is not None) != want:
            failures.append(f"{path}: single-side change not carried into merge")
            continue
        if got is None:
            continue
        # A single-side change must land byte-for-byte; a mechanical overlay can
        # carry the path while silently dropping hunks, so compare the content.
        src = MAIN if path in side_main else BRANCH
        single_side_content_checked += 1
        if got != blob(src, path):
            failures.append(f"{path}: single-side content altered by the merge")

    changed = git("diff", "--name-only", base, MERGE).split()
    markers = [p for p in changed
               if re.search(r"(?m)^<<<<<<< ", blob(MERGE, p) or "")]

    evidence = {
        "task": "TensorFEM PR #61 reconciliation (agent/m3202 <- main)",
        "shas": shas,
        "counts": {
            "files_main_side": len(side_main),
            "files_branch_side": len(side_branch),
            "files_both_sides": len(both),
            "files_main_only": len(set(side_main) - set(side_branch)),
            "files_branch_only": len(set(side_branch) - set(side_main)),
        },
        "hand_resolved": sorted(HAND_RESOLVED),
        "files_both_sides": files,
        "deletions": deletions,
        "files_changed_total": len(changed),
        "unresolved_conflict_markers": markers,
        "single_side_content_checked": single_side_content_checked,
        "failures": failures,
        "verdict": (
            # Zero shared files means the base collapsed onto one side: the
            # union obligation was never exercised. Reporting UNION_OK here
            # would be a false green, so it is called out explicitly.
            "NOT_AUDITED_NO_SHARED_FILES" if not both
            else "UNION_OK" if not failures and not markers
            else "UNION_FAIL"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(evidence["counts"], ensure_ascii=False))
    for rec in files:
        print(f"  {rec['verdict']:<16} tier{rec.get('tier', '?')[0]} {rec['path']}")
        for key in ("lines_lost_from_main", "lines_lost_from_branch",
                    "atoms_lost_from_main", "atoms_lost_from_branch"):
            if rec.get(key):
                print(f"      {key} = {rec[key]}")
    for rec in deletions:
        print(f"  deletion {rec['verdict']:<32} {rec['path']}")
    if markers:
        print("conflict markers left in:", markers)
    try:
        shown = OUT.resolve().relative_to(REPO)
    except ValueError:
        shown = OUT
    print(evidence["verdict"], "->", shown)
    return 0 if evidence["verdict"] == "UNION_OK" or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())