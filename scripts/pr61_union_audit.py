#!/usr/bin/env python3
"""Independent union audit of the PR #61 reconciliation merge (TensorFEM).

Why this exists
---------------
PR #61 (``agent/m3202`` -> ``main``) went dirty because ``main`` advanced while
the branch was open.  The healing merge is a three-way merge; the danger is a
*mechanical overlay* that silently drops one side's content while leaving no
conflict marker behind (this repository has that failure mode on record).  A
clean ``git status`` is therefore not evidence.

This script re-derives the merge obligation straight from the git object
database and fails closed:

* **conflict set** — ``git merge-tree`` is used as an independent oracle: the
  files it cannot merge mechanically must be exactly the declared
  hand-resolved set, so "we resolved it by hand" cannot be used to hide a file
  git would have merged fine;
* **union blobs** — for every path changed on either side, the merged blob must
  be one of {main blob, branch blob, declared hand resolution}; nothing else is
  an acceptable third value, and deletions must propagate;
* **hand-resolved files** — the resolution must keep every atom (numbers,
  backticked identifiers, link targets, file paths; *multiset*, so a repeated
  measurement cannot silently lose one occurrence) and every content word
  (*set*, because a prose rewrap legitimately merges two sentences into one);
* **post-merge hygiene** — merged content is immutable: no commit after the merge
  commit may modify or delete a path that the merge took from either side.  The
  PR's own new evidence files (added after the merge) may of course evolve.
  No conflict marker may survive anywhere.

Negative controls are part of the evidence: the same checker is run against the
two mechanical overlays (main-only, branch-only) and against a single-line
deletion of the delivered text.  A checker that cannot reject those is
worthless, so ``--check`` fails when a control does not fire.

Reproducibility
---------------
The PR coordinates are frozen constants below: the branch tip moved when the
merge was committed and again when evidence was added, so re-deriving them from
live refs would silently change the question being answered.  Override with
``--base``/``--main``/``--branch``/``--merge``/``--tip``.

Usage::

    python3 scripts/pr61_union_audit.py            # write evidence JSON
    python3 scripts/pr61_union_audit.py --check    # exit 1 on any failure

Evidence: ``docs/assets/pr61_union_audit_evidence.json``.

The checker lives under ``scripts/`` and its evidence under ``docs/assets/``
because the delivery gate drops ``artifacts/`` from the committed tree; the
gate below (``tests/test_pr61_merge_union_audit.py``) needs a payload that
travels with the repository.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

# Frozen PR #61 coordinates (see docs/pr61-merge-reconciliation.md).
BASE = "e8f6759aab2d203db641b2e94502ee5865848e3f"  # merge-base(main, branch)
MAIN = "7accc4b9cfb45369958c432a8aa2f0ebbb846a5b"  # main tip at merge time
BRANCH = "2ff75a8158492b1d2232348d1a1d8e002ce1c009"  # agent/m3202 tip pre-merge
MERGE = "bafc0faf2b438de53fb2da52e8478997687c69ad"  # three-way merge commit
TIP = "HEAD"  # branch tip to verify

# Files git could not merge mechanically; the resolution was rewritten by hand.
HAND_RESOLVED = ("CHANGELOG.md", "docs/marine-hydrostatics.md")

OUT = REPO / "docs" / "assets" / "pr61_union_audit_evidence.json"

ATOM_RE = re.compile(
    r"`[^`]+`"  # backticked identifier / formula
    r"|\]\([^)]+\)"  # markdown link target
    r"|[\w./-]+\.(?:py|md|json|png|xml|toml)"  # file path
    r"|(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?"  # number
)
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9-]{3,}|[\u4e00-\u9fff]")
GIT_NOTE_RE = re.compile(r"^(Auto-merging|CONFLICT|Merge|Removing|Adding) ")


def git(*args: str) -> str:
    """Run git in the repository, raising on failure."""
    proc = subprocess.run(("git",) + args, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:400]}")
    return proc.stdout


def tree_map(rev: str) -> dict[str, str]:
    """``path -> blob sha`` for every file of ``rev`` (one git call)."""
    files: dict[str, str] = {}
    for entry in git("ls-tree", "-r", "-z", rev).split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        files[path] = meta.split()[2]
    return files


def blob_text(rev: str, path: str) -> str | None:
    """Lenient text of ``rev:path`` (binary evidence assets must not abort)."""
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


def words(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text)}


def unique_text(side: str, base: str) -> str:
    """Lines present on ``side`` beyond the multiset the base already had."""
    return "\n".join(
        (collections.Counter(side.splitlines()) - collections.Counter(base.splitlines())).elements()
    )


def union_verdict(
    path: str, base_text: str, main_text: str, branch_text: str, merged_text: str
) -> dict:
    """Atom (multiset) + content-word (set) containment of both sides in the merge."""
    side_main, side_branch = unique_text(main_text, base_text), unique_text(branch_text, base_text)
    m_atoms, m_words = atoms(merged_text), words(merged_text)

    rec: dict = {"path": path, "tier": "atoms(multiset)+content-words(set)", "failures": []}
    for label, side in (("main", side_main), ("branch", side_branch)):
        side_atoms, side_words = atoms(side), words(side)
        lost_atoms = side_atoms - m_atoms
        lost_words = side_words - m_words
        rec[f"atoms_unique_{label}"] = sum(side_atoms.values())
        rec[f"atoms_lost_from_{label}"] = sum(lost_atoms.values())
        rec[f"words_unique_{label}"] = len(side_words)
        rec[f"words_lost_from_{label}"] = len(lost_words)
        if lost_atoms or lost_words:
            rec["failures"].append(
                f"{label}: atoms lost {sorted(lost_atoms)[:6]}, words lost {sorted(lost_words)[:8]}"
            )
        rec[f"words_occurrences_{label}"] = len(WORD_RE.findall(side))

    both_unique = bool(side_main.strip()) and bool(side_branch.strip())
    if both_unique and merged_text in (main_text, branch_text):
        rec["failures"].append(
            "merge text is byte-identical to one side although both sides "
            "carry unique content (one-sided overlay)"
        )
    rec["merged_equals_main"] = merged_text == main_text
    rec["merged_equals_branch"] = merged_text == branch_text
    rec["merged_lines"] = len(merged_text.splitlines())
    rec["verdict"] = "union-preserved" if not rec["failures"] else "FAIL-union"
    return rec


def check(merge: str, tip: str) -> dict:
    """Full audit of merge commit ``merge`` and branch tip ``tip``."""
    shas = {
        "merge_base": git("rev-parse", BASE).strip(),
        "main": git("rev-parse", MAIN).strip(),
        "branch_pre_merge": git("rev-parse", BRANCH).strip(),
        "merge_commit": git("rev-parse", merge).strip(),
        "merge_tree": git("rev-parse", f"{merge}^{{tree}}").strip(),
        "tip": git("rev-parse", tip).strip(),
        "tip_tree": git("rev-parse", f"{tip}^{{tree}}").strip(),
    }
    t_main, t_branch, t_merge, t_tip = (tree_map(r) for r in (MAIN, BRANCH, merge, tip))
    changed_main, changed_branch = name_status(BASE, MAIN), name_status(BASE, BRANCH)
    failures: list[str] = []

    # --- 1. git's own view of the conflict set ------------------------------
    mt = subprocess.run(
        ("git", "merge-tree", "--write-tree", "--name-only", MAIN, BRANCH),
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    notes = [ln for ln in mt.stdout.splitlines() if ln.strip() and not GIT_NOTE_RE.match(ln)]
    oracle = {
        "auto_merge_tree": notes[0] if notes else None,
        "conflicted_files": sorted(notes[1:]),
        "merge_tree_exit_code": mt.returncode,
    }
    if oracle["conflicted_files"] != sorted(HAND_RESOLVED):
        failures.append(
            f"conflict set {oracle['conflicted_files']} != declared hand-resolved "
            f"{sorted(HAND_RESOLVED)}"
        )

    # --- 2. main must be an ancestor of the merge (PR fast-forwardable) -----
    main_is_ancestor = (
        subprocess.run(("git", "merge-base", "--is-ancestor", MAIN, merge), cwd=REPO).returncode
        == 0
    )
    if not main_is_ancestor:
        failures.append(
            f"{MAIN[:8]} is not an ancestor of {shas['merge_commit'][:8]}: the PR would stay dirty"
        )

    # --- 3. union blob rule for every path changed on either side ----------
    carried: list[dict] = []
    for path in sorted(set(changed_main) | set(changed_branch)):
        status = changed_main.get(path) or changed_branch.get(path)
        allowed = {"main": t_main.get(path), "branch": t_branch.get(path)}
        got = t_merge.get(path)
        if path in HAND_RESOLVED:
            carried.append(
                {
                    "path": path,
                    "status": status,
                    "side": "both",
                    "merged_blob": got,
                    "verdict": "hand-resolved",
                }
            )
            continue
        if path in changed_main and path in changed_branch:
            failures.append(f"{path}: changed on both sides but not declared hand-resolved")
            carried.append(
                {
                    "path": path,
                    "status": status,
                    "side": "both",
                    "merged_blob": got,
                    "verdict": "FAIL-undeclared-conflict",
                }
            )
            continue
        side = "main" if path in changed_main else "branch"
        ok = got == allowed[side]
        if not ok:
            failures.append(
                f"{path}: merge blob {str(got)[:10]} is neither the {side} blob "
                f"{str(allowed[side])[:10]} nor a declared hand resolution"
            )
        carried.append(
            {
                "path": path,
                "status": status,
                "side": side,
                "expected_blob": allowed[side],
                "merged_blob": got,
                "verdict": "carried" if ok else "FAIL-content-mismatch",
            }
        )

    # --- 4. union obligation for the hand-resolved files -------------------
    hand: list[dict] = []
    for path in HAND_RESOLVED:
        base_text, main_text, branch_text, merged_text = (
            blob_text(r, path) for r in (BASE, MAIN, BRANCH, merge)
        )
        if None in (main_text, branch_text, merged_text):
            failures.append(f"{path}: missing on one of base/main/branch/merge")
            hand.append({"path": path, "verdict": "FAIL-absent"})
            continue
        rec = union_verdict(path, base_text or "", main_text, branch_text, merged_text)
        for msg in rec.pop("failures"):
            failures.append(f"{path}: {msg}")
        hand.append(rec)

    # --- 5. nothing added that neither side carries, no surviving markers ---
    expected_paths = set(t_main) | set(t_branch)
    unexpected = sorted(set(t_merge) - expected_paths)
    if unexpected:
        failures.append(f"merge introduced paths that neither side has: {unexpected[:5]}")
    changed_total = git("diff", "--name-only", BASE, merge).split()
    markers = [
        p
        for p in changed_total
        if p in t_merge
        and p.rsplit(".", 1)[-1] in {"md", "py", "json", "toml", "txt", "cfg"}
        and re.search(r"(?m)^<<<<<<< ", blob_text(merge, p) or "")
    ]
    if markers:
        failures.append(f"conflict markers survived in {markers[:5]}")

    # --- 6. post-merge hygiene: the tip may only add files ------------------
    tip_vs_merge = name_status(merge, tip)
    merged_content = set(changed_main) | set(changed_branch)
    touched_after = sorted(p for p in tip_vs_merge if p in merged_content)
    if touched_after:
        failures.append(
            f"commit(s) after the merge modified/deleted merged content: {touched_after[:5]}"
        )
    tip_added = sorted(p for p, st in tip_vs_merge.items() if st == "A")
    tip_owned = sorted(p for p, st in tip_vs_merge.items() if st != "A")

    # --- 7. negative controls: the checker must reject mechanical overlays ---
    controls = []
    for label, pick in (("main-only overlay", "main"), ("branch-only overlay", "branch")):
        fired = []
        for path in HAND_RESOLVED:
            texts = {"main": blob_text(MAIN, path) or "", "branch": blob_text(BRANCH, path) or ""}
            rec = union_verdict(
                path, blob_text(BASE, path) or "", texts["main"], texts["branch"], texts[pick]
            )
            if rec["verdict"] == "FAIL-union":
                fired.append(path)
        controls.append(
            {
                "control": label,
                "rejected_files": fired,
                "verdict": "control-fires" if fired else "CONTROL-BLIND",
            }
        )
        if not fired:
            failures.append(f"negative control '{label}' was not rejected by the checker")

    changelog = blob_text(merge, "CHANGELOG.md") or ""
    dropped = next(
        (ln for ln in changelog.splitlines() if "marine_large_angle_gz" in ln and "1.33e-15" in ln),
        None,
    ) or next((ln for ln in changelog.splitlines() if "1.33e-15" in ln), None)
    if dropped:
        mutated = changelog.replace(dropped + "\n", "", 1)
        rec = union_verdict(
            "CHANGELOG.md",
            blob_text(BASE, "CHANGELOG.md") or "",
            blob_text(MAIN, "CHANGELOG.md") or "",
            blob_text(BRANCH, "CHANGELOG.md") or "",
            mutated,
        )
        controls.append(
            {
                "control": f"single-line deletion: {dropped.strip()[:70]}",
                "rejected_files": ["CHANGELOG.md"] if rec["verdict"] == "FAIL-union" else [],
                "verdict": "control-fires" if rec["verdict"] == "FAIL-union" else "CONTROL-BLIND",
            }
        )
        if rec["verdict"] != "FAIL-union":
            failures.append("single-line deletion control was not rejected by the checker")
    else:
        controls.append({"control": "single-line deletion", "verdict": "control-unavailable"})
        failures.append("could not build the single-line deletion control")

    return {
        "task": "TensorFEM PR #61 reconciliation (agent/m3202 <- main): union audit",
        "shas": shas,
        "git_version": git("--version").strip(),
        "counts": {
            "files_main_side": len(changed_main),
            "files_branch_side": len(changed_branch),
            "files_both_sides": len(set(changed_main) & set(changed_branch)),
            "files_main_only": len(set(changed_main) - set(changed_branch)),
            "files_branch_only": len(set(changed_branch) - set(changed_main)),
            "files_changed_total": len(changed_total),
            "files_carried_verbatim_or_hand": len(carried),
            "files_added_after_merge": len(tip_added),
        },
        "merge_tree_oracle": oracle,
        "main_is_ancestor": main_is_ancestor,
        "hand_resolved": list(HAND_RESOLVED),
        "union": hand,
        "carried_files": carried,
        "deletions": [
            {
                "path": p,
                "status": st,
                "side": "main" if p in changed_main else "branch",
                "present_in_merge": p in t_merge,
            }
            for p, st in sorted({**changed_main, **changed_branch}.items())
            if st == "D"
        ],
        "added_after_merge": tip_added,
        "evolved_after_merge": tip_owned,
        "unexpected_paths": unexpected,
        "unresolved_conflict_markers": markers,
        "negative_controls": controls,
        "failures": failures,
        "verdict": "UNION_OK" if not failures else "UNION_FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo")
    parser.add_argument("--base")
    parser.add_argument("--main")
    parser.add_argument("--branch")
    parser.add_argument("--merge", default=MERGE)
    parser.add_argument("--tip", default=TIP)
    parser.add_argument("--out")
    parser.add_argument("--check", action="store_true", help="exit 1 unless the audit passes")
    args = parser.parse_args()

    global REPO, BASE, MAIN, BRANCH, OUT
    if args.repo:
        REPO = pathlib.Path(args.repo).resolve()
    BASE = args.base or BASE
    MAIN = args.main or MAIN
    BRANCH = args.branch or BRANCH
    if args.out:
        OUT = pathlib.Path(args.out)

    evidence = check(args.merge, args.tip)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(evidence["counts"], ensure_ascii=False))
    print(f"conflict set (git oracle): {evidence['merge_tree_oracle']['conflicted_files']}")
    print(f"main is ancestor of merge: {evidence['main_is_ancestor']}")
    for rec in evidence["union"]:
        print(
            f"  {rec['verdict']:<16} {rec['path']}  atoms lost "
            f"{rec.get('atoms_lost_from_main', 0)}/{rec.get('atoms_lost_from_branch', 0)}, "
            f"words lost {rec.get('words_lost_from_main', 0)}/"
            f"{rec.get('words_lost_from_branch', 0)}, lines {rec.get('merged_lines', 0)}"
        )
    bad = [c for c in evidence["carried_files"] if c["verdict"].startswith("FAIL")]
    print(
        f"  carried files: {len(evidence['carried_files']) - len(bad)}/"
        f"{len(evidence['carried_files'])} verbatim-or-hand"
    )
    print(f"  added after merge: {evidence['counts']['files_added_after_merge']} (additions only)")
    for c in evidence["negative_controls"]:
        print(f"  control {c['verdict']:<15} {c['control']}")
    for msg in evidence["failures"]:
        print(f"  FAIL {msg}")
    print(evidence["verdict"], "->", OUT)
    return 0 if evidence["verdict"] == "UNION_OK" or not args.check else 1


if __name__ == "__main__":
    sys.exit(main())
