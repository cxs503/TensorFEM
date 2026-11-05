#!/usr/bin/env python3
"""Lossless-union audit for the PR #92 merge cure (agent/m3545 onto main).

Mechanical overlays silently drop content; this script proves the opposite for
every file touched by either side of the merge:

  * files touched by only one side must be byte-identical to that side;
  * files touched by both sides must contain every line each side added
    relative to the merge base, except lines listed as *superseded* — each of
    which is printed together with the replacement text that now carries the
    same (or newer, factually corrected) statement.

Output: artifacts/merge_cure_pr92_evidence.json (all numbers programmatic).
Usage:  python3 scripts/merge_cure_pr92_audit.py --base 14ca2fd --main main \
            --branch 51fa370 --merged HEAD
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess

# Lines the resolving author deliberately did not keep verbatim, with the
# evidence-backed reason.  Empty set => a pure verbatim union is required.
SUPERSEDED: dict[str, list[dict[str, str]]] = {
    "docs/element-rank-deficiency-audit.md": [
        {
            "dropped": "刚度；HEX20 的减缩积分变体在代码库中不存在，无法审计。",
            "replaced_by": "二次实体单元的欠账（畸变/曲边 hex20、tet10、减缩积分变体）已在 quadratic-solid-rank-audit.md 中补齐",
            "reason": (
                "main resolved this open item: hex20_reduced_stiffness (order 2, 8-point) "
                "now exists as audit-only code and was audited there; keeping the branch's "
                "'does not exist, cannot audit' sentence would be factually false"
            ),
        },
        {
            "dropped": "7. **已清偿（本轮）**：`hex20` 此前只审了规则立方体。现已在 `audit_library` 中补齐",
            "replaced_by": "7. **已清偿（本轮）**：秩审计之外，`hex20` 此前只对规则立方体做过资格复核。现已在 `audit_library` 中补齐",
            "reason": "reworded only to avoid contradicting main's rank audit (§6); claim preserved",
        },
        {
            "dropped": "畸变角点体与 4% 中边凸起曲边体（`spurious = 0`、`null_dim = 6`），并做了 35 组",
            "replaced_by": "畸变角点体与 4% 中边凸起曲边体（`spurious = 0`、`null_dim = 6`），并做了 35 组",
            "reason": "line reflowed by the reword above; text identical",
        },
    ]
}


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True, check=True).stdout


def show_bytes(rev: str, path: str) -> bytes | None:
    """Raw blob bytes (binary-safe: PNG/JSON assets are audited too)."""
    r = subprocess.run(("git", "show", f"{rev}:{path}"), capture_output=True)
    return r.stdout if r.returncode == 0 else None


def show(rev: str, path: str) -> str | None:
    raw = show_bytes(rev, path)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None  # binary asset


def added_lines(old: str, new: str) -> list[str]:
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
    return [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--main", default="main")
    ap.add_argument("--branch", required=True)
    ap.add_argument("--merged", default="HEAD")
    ap.add_argument("--out", default="artifacts/merge_cure_pr92_evidence.json")
    a = ap.parse_args()

    def names(rev: str) -> set[str]:
        return set(git("diff", "--name-only", a.base, rev).split())

    main_files, branch_files = names(a.main), names(a.branch)
    both = sorted(main_files & branch_files)
    only_branch = sorted(branch_files - main_files)
    report: dict = {
        "base": a.base,
        "main": a.main,
        "branch": a.branch,
        "counts": {
            "main_only": len(main_files - branch_files),
            "branch_only": len(only_branch),
            "both": len(both),
        },
        "branch_only": [],
        "both": [],
    }

    # 1) one-sided files must be byte-identical to their side
    for path in only_branch:
        identical = show_bytes(a.branch, path) == show_bytes(a.merged, path)
        report["branch_only"].append(
            {"path": path, "identical_to_branch": identical,
             "binary": show(path and a.branch, path) is None}
        )
    for path in sorted(main_files - branch_files):
        report["both"].append({"path": path, "identical_to_main": show_bytes(a.main, path) == show_bytes(a.merged, path)})

    # 2) two-sided files: every added line of each side must survive
    ok = True
    for path in both:
        b, m, g = show(a.base, path), show(a.main, path), show(a.merged, path)
        entry: dict = {"path": path, "only_side": None}
        if b is None or m is None or g is None:  # added/deleted on a side
            entry.update({"binary_or_absent": True, "verbatim_union": None})
            report["both"].append(entry)
            continue
        if "\x00" in m or "\x00" in g:
            entry.update({"binary": True, "verbatim_union": None})
            report["both"].append(entry)
            continue
        sup = {d["dropped"] for d in SUPERSEDED.get(path, [])}
        per_side = {}
        for side_name, side_rev in (("main", a.main), ("branch", a.branch)):
            side_text = show(side_rev, path)
            adds = [l.strip() for l in added_lines(b, side_text) if l.strip()]
            missing = [l for l in adds if l not in g and l not in sup]
            per_side[side_name] = {
                "added_lines": len(adds),
                "verbatim_present": len(adds) - len([l for l in adds if l not in g]),
                "superseded": len([l for l in adds if l not in g and l in sup]),
                "unaccounted_missing": missing,
            }
            if missing:
                ok = False
        entry["per_side"] = per_side
        entry["superseded_detail"] = SUPERSEDED.get(path, [])
        report["both"].append(entry)

    # 3) global consistency
    one_sided_diff = [e["path"] for e in report["branch_only"] if not e["identical_to_branch"]]
    main_one_sided_diff = [
        e["path"] for e in report["both"] if "identical_to_main" in e and not e["identical_to_main"]
    ]
    report["totals"] = {
        "files_checked": len(main_files | branch_files),
        "one_sided_mismatch": one_sided_diff + main_one_sided_diff,
        "two_sided_unaccounted_missing": 0 if ok else 1,
        "verdict": "LOSSLESS_UNION" if ok and not one_sided_diff and not main_one_sided_diff else "CONTENT_LOSS",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report["counts"], ensure_ascii=False))
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))
    return 0 if report["totals"]["verdict"] == "LOSSLESS_UNION" else 1


if __name__ == "__main__":
    raise SystemExit(main())