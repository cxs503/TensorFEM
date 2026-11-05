#!/usr/bin/env python3
"""Independent cross-check of the PR #61 merge tree (no shared code with
`scripts/audit_pr61_merge_union.py`).

Two obligations, both machine-decided:

  A) tree containment: every path that main changed since the fork point and
     the branch did not touch must be byte-identical in the merge tree.  This
     is the "nothing from main silently vanished" obligation, and it is
     deliberately weaker-but-independent: it reads blobs straight from git.
  B) content containment on the hand-resolved files.  Code/data must keep every
     token of both sides (line-atomic: a dropped token is dropped content).
     Prose may be reflowed, so prose is judged on *atoms* -- backticked
     identifiers, link targets and numbers -- which is where an assertion's
     substance lives; a different word set is reported and must be justified
     file-by-file in PROSE_REWORDED, otherwise the run fails closed.  Tolerating
     prose word drift silently is exactly how a rewritten sentence eats an
     assertion unannounced.

    python3 scripts/crosscheck_pr61_merge.py --base e8f6759 --main 7accc4b \
        --branch 6b4c4fa --merge HEAD --out docs/evidence/pr61/word-containment.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def blob_bytes(rev: str, path: str) -> bytes | None:
    """Raw blob bytes; text decoding is a separate concern (binary assets exist)."""
    r = subprocess.run(["git", "cat-file", "blob", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def blob(rev: str, path: str) -> str | None:
    raw = blob_bytes(rev, path)
    return None if raw is None else raw.decode("utf-8", "replace")


def name_status(base: str, rev: str) -> dict[str, str]:
    out = {}
    for line in git("diff", "--name-status", base, rev).splitlines():
        status, _, path = line.partition("\t")
        out[path] = status
    return out


PROSE_SUFFIXES = (".md", ".rst", ".txt")

# Prose files whose merge is a *reword*, not a superset, and why the assertions
# survive anyway.  Anything not listed here fails closed on a word deficit.
PROSE_REWORDED = {
    "docs/marine-hydrostatics.md": (
        "main's 'this module still is not a large-angle GZ curve' and the "
        "branch's 'large-angle GZ curves are provided separately' are merged "
        "into 'This module itself still provides no large-angle GZ curve; two "
        "separate services cover that question' + the module names; every "
        "backticked identifier, link target and number of both sides survives "
        "(see obligation B atoms), only the copula was rewritten."
    ),
}


def tokens(text: str) -> collections.Counter:
    return collections.Counter(re.findall(r"[A-Za-z0-9_`./%-]{2,}", text))


def atoms(text: str) -> collections.Counter:
    """Semantic atoms: identifiers, link targets, numbers (never prose words)."""
    return collections.Counter(
        re.findall(r"`[^`\n]+`|\[[^\]\n]+\]\([^)\n]+\)|https?://\S+|"
                   r"(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?", text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--merge", default="HEAD")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = git("rev-parse", args.base).strip()
    main_rev = git("rev-parse", args.main).strip()
    branch_rev = git("rev-parse", args.branch).strip()
    merge_rev = git("rev-parse", args.merge).strip()

    side_main = name_status(base, main_rev)
    side_branch = name_status(base, branch_rev)
    raw = git("diff", "--name-status", base, merge_rev)
    merged_paths = {}
    for line in raw.splitlines():
        status, _, path = line.partition("\t")
        merged_paths[path] = status

    failures: list[str] = []

    # ---- obligation A: main-only paths survive byte-for-byte ----
    main_only = sorted(set(side_main) - set(side_branch))
    identical = checked = 0
    for path in main_only:
        if side_main[path].startswith("D"):  # deleted on main: must be gone
            if blob_bytes(merge_rev, path) is not None:
                failures.append(f"A: {path} deleted on main but alive in the merge")
            checked += 1
            continue
        want, got = blob_bytes(main_rev, path), blob_bytes(merge_rev, path)
        checked += 1
        if want is None:
            continue
        if got is None:
            failures.append(f"A: {path} present on main, missing from the merge")
        elif want == got:
            identical += 1
        else:
            failures.append(f"A: {path} differs from main although the branch never touched it")

    # ---- obligation B: per-side token superset on the shared files ----
    shared = sorted(set(side_main) & set(side_branch))
    shared_records = []
    for path in shared:
        got = blob(merge_rev, path)
        if got is None:
            failures.append(f"B: shared file {path} missing from the merge")
            continue
        prose = path.endswith(PROSE_SUFFIXES)
        unit = atoms if prose else tokens
        merged_units = unit(got)
        rec = {"path": path, "unit": "atoms" if prose else "tokens",
               "missing_main": {}, "missing_branch": {}, "word_drift": {}}
        for label, rev, key in (("main", main_rev, "missing_main"),
                                ("branch", branch_rev, "missing_branch")):
            side = blob(rev, path)
            if side is None:
                continue
            missing = unit(side) - merged_units
            if missing:
                rec[key] = dict(missing.most_common(10))
                failures.append(f"B: {path} lost {sum(missing.values())} {rec['unit']} "
                                f"from {label}: {list(missing)[:5]}")
            if prose:
                drift = tokens(side) - tokens(got)
                if drift:
                    rec["word_drift"][label] = dict(drift.most_common(10))
        if prose and rec["word_drift"] and path not in PROSE_REWORDED:
            failures.append(f"B: prose {path} changed wording without a recorded "
                            f"justification (PROSE_REWORDED)")
        if path in PROSE_REWORDED:
            rec["justification"] = PROSE_REWORDED[path]
        shared_records.append(rec)

    # ---- vacuity guard: an audit that exercised nothing is not a pass ----
    vacuous = not checked or not shared
    evidence = {
        "task": "independent cross-check of the PR #61 merge tree",
        "shas": {"base": base, "main": main_rev, "branch": branch_rev, "merge": merge_rev},
        "obligation_a": {
            "main_only_paths": len(main_only),
            "checked": checked,
            "byte_identical": identical,
            "mismatches": len([f for f in failures if f.startswith("A:")]),
        },
        "obligation_b": {
            "shared_paths": len(shared),
            "prose_reworded": sorted(PROSE_REWORDED),
            "records": shared_records,
            "mismatches": len([f for f in failures if f.startswith("B:")]),
        },
        "merge_diff_paths": len(merged_paths),
        "failures": failures,
        "verdict": ("NOT_AUDITED_VACUOUS" if vacuous
                    else "CROSSCHECK_OK" if not failures else "CROSSCHECK_FAIL"),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: evidence[k] for k in ("obligation_a", "obligation_b")}, ensure_ascii=False))
    print(evidence["verdict"], "->", out)
    return 0 if evidence["verdict"] == "CROSSCHECK_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())