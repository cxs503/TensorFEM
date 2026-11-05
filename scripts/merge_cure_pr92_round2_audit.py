#!/usr/bin/env python3
"""Lossless-union audit for the PR #92 (agent/m3545) → main three-way merge, round 2.

The round-1 cure merged main@7a8f4af into agent/m3545, but main then moved on
(`723e12c`, the m3255 line) and the PR went dirty again.  This round merges the
*current* main tree into the branch, and this script certifies that the result
is a semantic union: every path of either side survives, no side's own edit is
replaced by the other side's stale copy, and the genuine add/add collisions
(two independently written implementations of the same physics) are both kept.

Conventions
-----------
``base``   = ``7accc4b`` (merge-base of main and the branch);
``main``   = GitHub main tree at merge time (``723e12c``);
``branch`` = ``agent/m3545`` head before this round (``fb9c212``);
``merged`` = the merge result (this tree).

Usage::

    PYTHONPATH=src python scripts/merge_cure_pr92_round2_audit.py \
        --base 7accc4b --main <main-sha> --branch <branch-sha> --merged HEAD
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: add/add collisions: the canonical path keeps main's implementation, the
#: branch's independent implementation keeps its own, explicitly renamed, path.
RENAMED_FROM_BRANCH = {
    "src/tensorfem/plate_linear_stress_buckling.py": "src/tensorfem/plate_bending_buckling.py",
    "tests/test_plate_linear_stress_buckling.py": "tests/test_plate_bending_buckling.py",
    "benchmarks/plate_linear_stress_buckling.py": "benchmarks/plate_bending_buckling.py",
    "benchmarks/plate_linear_stress_buckling_evidence.json":
        "benchmarks/plate_bending_buckling_evidence.json",
    "docs/assets/plate_linear_stress_buckling.png": "docs/assets/plate_bending_buckling.png",
}
#: both sides edited the prose; the merge keeps one union paragraph.
UNION_PATHS = ["docs/benchmark-qualification-status.md"]

#: A rename also has to rewrite references; the rewrite is required to be a pure
#: textual substitution, so the *code* stays byte-identical.
#: both sides edited the prose; the merge keeps one union paragraph.
UNION_PATHS = ["docs/benchmark-qualification-status.md"]

#: A rename also has to rewrite references; the rewrite is required to be a pure
#: textual substitution, so the *code* stays byte-identical.
RENAME_SUBSTITUTIONS = {
    name: [("plate_bending_buckling", "plate_linear_stress_buckling")]
    for name in ("src/tensorfem/plate_linear_stress_buckling.py",
                 "tests/test_plate_linear_stress_buckling.py",
                 "benchmarks/plate_linear_stress_buckling.py",
                 "benchmarks/plate_linear_stress_buckling_evidence.json",
                 "docs/assets/plate_linear_stress_buckling.png")
}
#: branch docs whose only edits are the module/asset path references
REFERENCE_ONLY_PATHS = ["docs/marine-plate-linear-stress-buckling-validation.md"]


def git(*args: str, cwd: pathlib.Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


def tree(rev: str) -> dict[str, tuple[str, str]]:
    out = {}
    for line in git("ls-tree", "-r", rev).splitlines():
        meta, path = line.split("\t", 1)
        mode, _typ, sha = meta.split()
        out[path] = (mode, sha)
    return out


def blob(rev: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                          capture_output=True, check=True).stdout


def classify(base, main, branch) -> dict[str, list[str]]:
    plan: dict[str, list[str]] = {k: [] for k in (
        "branch_only_edit", "main_only_edit", "both_same", "conflict",
        "added_by_branch", "added_by_main", "deleted_by_main", "deleted_by_branch",
        "untouched")}
    for path in sorted(set(base) | set(main) | set(branch)):
        b, m, t = base.get(path), main.get(path), branch.get(path)
        if b == m == t:
            plan["untouched"].append(path)
        elif m == b and t != b:
            plan["deleted_by_branch" if t is None else "branch_only_edit"].append(path)
        elif t == b and m != b:
            plan["deleted_by_main" if m is None else "main_only_edit"].append(path)
        elif m == t:
            plan["both_same"].append(path)
        elif t is None:
            plan["deleted_by_branch"].append(path)
        elif m is None:
            plan["deleted_by_main"].append(path)
        else:
            plan["conflict"].append(path)
    return plan


def audit(plan, base, main, branch, merged, revs=None) -> list[str]:
    fails = []
    renamed_src = set(RENAMED_FROM_BRANCH.values())
    for path in plan["branch_only_edit"] + plan["added_by_branch"]:
        if path in renamed_src or path in REFERENCE_ONLY_PATHS:
            continue
        if merged.get(path) != branch.get(path):
            fails.append(f"branch-side edit lost: {path}")
    for path in plan["main_only_edit"] + plan["added_by_main"]:
        if merged.get(path) != main.get(path):
            fails.append(f"main-side edit lost: {path}")
    for path in plan["both_same"]:
        if merged.get(path) != main.get(path):
            fails.append(f"identical-on-both-sides changed: {path}")
    for path in plan["untouched"]:
        if merged.get(path) != base.get(path):
            fails.append(f"untouched path changed: {path}")
    for path in plan["deleted_by_branch"]:
        if path in merged and path not in renamed_src:
            fails.append(f"branch deletion silently reverted: {path}")
    for path in plan["deleted_by_main"]:
        if path in merged:
            fails.append(f"main deletion silently reverted: {path}")

    # add/add collisions: both implementations must survive.  The branch copy
    # must equal its pre-merge blob modulo the declared reference rewrites; the
    # main copy at the canonical path must be untouched.
    for new, old in RENAMED_FROM_BRANCH.items():
        if new not in merged:
            fails.append(f"branch implementation missing after rename: {new}")
            continue
        if old in main and old in merged and merged[old] != main.get(old):
            fails.append(f"main implementation altered: {old}")
        if new.endswith(".png"):
            if sha(blob(revs["branch"], old)) != sha(blob(revs["merged"], new)):
                fails.append(f"branch figure altered by rename: {new}")
            continue
        want = blob(revs["branch"], old).decode("utf-8")
        for a, b in RENAME_SUBSTITUTIONS.get(new, []):
            want = want.replace(a, b)
        got = blob(revs["merged"], new).decode("utf-8")
        if want != got:
            fails.append(f"branch implementation not a pure rename: {new}")
    for path in REFERENCE_ONLY_PATHS:
        want = blob(revs["branch"], path).decode("utf-8")
        for a, b in RENAME_SUBSTITUTIONS["benchmarks/plate_linear_stress_buckling.py"]:
            want = want.replace(a, b)
        if want != blob(revs["merged"], path).decode("utf-8"):
            fails.append(f"branch reference rewrite not pure: {path}")
    # semantic collisions: main's canonical modules/tests/bench must still exist
    for path in ("src/tensorfem/plate_bending_buckling.py",
                 "tests/test_plate_bending_buckling.py",
                 "benchmarks/plate_bending_buckling.py"):
        if path not in merged:
            fails.append(f"canonical main path vanished: {path}")
    return fails


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_union_docs(merged, base, main, branch, revs=None) -> dict:
    """Both sides' prose claims must survive in the merged paragraph."""
    out = {}
    for path in UNION_PATHS:
        text = blob(revs["merged"], path).decode("utf-8")
        m_text = blob(revs["main"], path).decode("utf-8")
        b_text = blob(revs["branch"], path).decode("utf-8")
        markers = [k for k in ("<<<<<<<", "=======", ">>>>>>>") if k in text]
        # the two closure claims each side added, and both doc links, must be present
        need = [
            "plate-bending-buckling-validation.md",
            "marine-plate-linear-stress-buckling-validation.md",
            "沿板宽线性变化的预屈曲压缩",
            "0.28%",
        ]
        out[path] = {
            "conflict_markers": markers,
            "contains": {k: (k in text) for k in need},
            "main_len": len(m_text),
            "branch_len": len(b_text),
            "merged_len": len(text),
            "ok": not markers and all(k in text for k in need),
        }
    return out


def main_() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="7accc4b")
    ap.add_argument("--main", required=True)
    ap.add_argument("--branch", default="fb9c212")
    ap.add_argument("--merged", default="HEAD")
    ap.add_argument("--json-out", default="artifacts/merge_cure_pr92_round2_evidence.json")
    args = ap.parse_args()

    base, m, b, g = (tree(args.base), tree(args.main), tree(args.branch), tree(args.merged))
    plan = classify(base, m, b)
    revs = {'branch': args.branch, 'merged': args.merged}
    fails = audit(plan, base, m, b, g, revs)
    docs = check_union_docs(g, base, m, b, {"main": args.main, "branch": args.branch, "merged": args.merged})
    fails += [f"doc union failed: {p}" for p, v in docs.items() if not v["ok"]]

    report = {
        "task": "PR #92 (agent/m3545) -> main, round-2 lossless three-way merge",
        "revisions": {"base": git("rev-parse", args.base).strip(),
                      "main": git("rev-parse", args.main).strip(),
                      "branch": git("rev-parse", args.branch).strip(),
                      "merged": git("rev-parse", args.merged).strip()},
        "counts": {k: len(v) for k, v in plan.items()},
        "plan": {k: v for k, v in plan.items() if k != "untouched"},
        "renamed_from_branch": RENAMED_FROM_BRANCH,
        "union_docs": docs,
        "status": "PASS" if not fails else "FAIL",
        "failures": fails,
    }
    out = ROOT / args.json_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in ("counts", "status", "failures")},
                     ensure_ascii=False, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main_())