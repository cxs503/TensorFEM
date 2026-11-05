"""Lossless-union audit for the PR #78 round-2 merge (agent/m3364 <- main).

The round-2 merge is `main` @ 7a8f4af onto `agent/m3364` @ b04bbf3 whose common
ancestor is 7accc4b.  Exactly one file (CHANGELOG.md) conflicted textually, but a
mechanical overlay has silently dropped content in this repository before, so the
merge is audited mechanically rather than by inspection:

  A) path survival      every path that exists on either side must still exist in
                        the merged tree (no side's addition may vanish)
  B) both-side survival for every file modified on both sides, all line-level
                        additions of *both* sides must be present in the merge
                        (semantic union, not ours/theirs wholesale)
  C) union-file integrity
                        the union file must contain each side's new section
                        verbatim (byte-exact substring, not "looks similar")
  D) no live markers    no conflict marker at column 0 anywhere in the tree
  E) no silent edits    every path changed by exactly one side is byte-identical
                        (same blob hash) to that side in the merged tree
                        (marker *detectors* legitimately embed the strings, so the
                        check is anchored at column 0 and both markers must appear)

Usage:  python scripts/audit_union_round2.py [--json out.json]
Exit code 0 == all gates green.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

BASE = "7accc4b9cfb45369958c432a8aa2f0ebbb846a5b"
OURS = "HEAD"
THEIRS = "tensorfem_local/main"

MARKER_RE = re.compile(r"^(?:<{7}(?: |$)|>{7}(?: |$)|={7}$)", re.M)

# files that both sides rewrote; the audit asserts a byte-exact containment of each
# side's added block, and records the per-file resolution note.
UNION_FILES = {
    "CHANGELOG.md": "both sides open with a new dated section; keep both, newest first",
}


def git(*args: str) -> str:
    proc = subprocess.run(["git", *args], capture_output=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args, proc.stdout, proc.stderr)
    return proc.stdout.decode("utf-8", "surrogateescape")


def changed_paths(ref: str) -> list[str]:
    out = git("diff", "--name-only", BASE, ref)
    return sorted(p for p in out.splitlines() if p)


def added_lines(ref: str, path: str) -> list[str]:
    """Non-trivial (>=8 chars) lines the ref adds to path relative to BASE."""
    try:
        out = git("diff", "-U0", BASE, ref, "--", path)
    except subprocess.CalledProcessError:
        return []
    lines = []
    for line in out.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            body = line[1:].strip()
            if len(body) >= 8:
                lines.append(body)
    return lines


def side_block(ref: str, path: str) -> str:
    """The verbatim added block of `ref` for `path` (contiguous diff body)."""
    out = git("diff", "-U0", BASE, ref, "--", path)
    body = [ln[1:] for ln in out.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")]
    return "\n".join(body)


def exists(ref: str, path: str) -> bool:
    try:
        git("cat-file", "-e", f"{ref}:{path}")
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="runs/m3364_merge_round2/union_audit_round2.json")
    args = ap.parse_args()

    ours_paths = changed_paths(OURS)
    theirs_paths = changed_paths(THEIRS)
    both = sorted(set(ours_paths) & set(theirs_paths))

    on_disk = set(git("ls-files", "--cached").splitlines())

    survival = []
    for path in sorted(set(ours_paths) | set(theirs_paths)):
        in_ours, in_theirs = exists(OURS, path), exists(THEIRS, path)
        in_merge = path in on_disk
        survival.append({
            "path": path, "in_ours": in_ours, "in_theirs": in_theirs,
            "in_merge": in_merge, "ok": in_merge or not (in_ours or in_theirs),
        })

    both_side = []
    for path in both:
        if not os.path.exists(path):
            both_side.append({"path": path, "ok": False, "reason": "absent from merge"})
            continue
        cur = open(path, encoding="utf-8", errors="surrogateescape").read()
        a, b = added_lines(OURS, path), added_lines(THEIRS, path)
        lost_a = [x for x in a if x not in cur]
        lost_b = [x for x in b if x not in cur]
        verbatim = {}
        for side, ref in (("ours", OURS), ("theirs", THEIRS)):
            block = side_block(ref, path)
            verbatim[side] = {
                "chars": len(block),
                "present_verbatim": block.strip() != "" and block.strip() in cur,
            }
        both_side.append({
            "path": path,
            "resolution": UNION_FILES.get(path, "unrecorded"),
            "ours_added_lines": len(a), "theirs_added_lines": len(b),
            "ours_lost": len(lost_a), "theirs_lost": len(lost_b),
            "lost_samples": (lost_a[:3] + lost_b[:3]),
            "verbatim": verbatim,
            "ok": not lost_a and not lost_b and all(v["present_verbatim"] for v in verbatim.values()),
        })

    identity = []
    for path in sorted(set(ours_paths) ^ set(theirs_paths)):
        side = OURS if path in ours_paths else THEIRS
        if path not in on_disk:
            continue
        want = git("rev-parse", f"{side}:{path}").strip()
        got = git("hash-object", path).strip()
        identity.append({"path": path, "side": "ours" if side == OURS else "theirs",
                         "byte_identical": want == got})
    identity_bad = [i for i in identity if not i["byte_identical"]]

    live_markers = []
    for path in sorted(on_disk):
        if not path.endswith((".md", ".py", ".txt", ".json", ".toml", ".yml", ".yaml", ".cfg")):
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        found = MARKER_RE.findall(text)
        if any(m.startswith("<") for m in found) and any(m.startswith(">") for m in found):
            live_markers.append({"path": path, "markers": sorted(set(m.strip() for m in found))[:4]})

    result = {
        "base": BASE,
        "ours": {"ref": OURS, "sha": git("rev-parse", OURS).strip(), "changed": len(ours_paths)},
        "theirs": {"ref": THEIRS, "sha": git("rev-parse", THEIRS).strip(), "changed": len(theirs_paths)},
        "both_modified": both,
        "gates": {
            "A_path_survival": all(s["ok"] for s in survival),
            "B_both_side_text": all(f["ok"] for f in both_side),
            "C_verbatim_union": all(v["present_verbatim"]
                                    for f in both_side if "verbatim" in f
                                    for v in f["verbatim"].values()),
            "D_no_live_markers": not live_markers,
            "E_single_side_identity": not identity_bad,
        },
        "path_survival": survival,
        "both_side_files": both_side,
        "live_markers": live_markers,
        "single_side_identity": identity,
        "single_side_mismatch": identity_bad,
        "checked_paths": len(set(ours_paths) | set(theirs_paths)),
    }
    result["all_ok"] = all(result["gates"].values())

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in ("gates", "all_ok", "both_modified", "checked_paths")},
                     indent=2, ensure_ascii=False))
    for f in both_side:
        print("  both-side:", f["path"], "ok=", f["ok"], f.get("verbatim"))
    print("  single-side paths checked:", len(identity), "mismatch:", len(identity_bad))
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())