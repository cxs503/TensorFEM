#!/usr/bin/env python3
"""Push the merged tree to GitHub branch ``agent/m3545`` through the Git Data API.

The branch must become a single commit whose *parent is current main* (``723e12c``)
with the merged tree — that is what turns PR #92 from ``dirty`` into ``clean``
(GitHub can then fast-forward/merge it).  Only blobs whose content is not already
reachable from main's tree are uploaded.

Usage::

    python scripts/merge_cure_pr92_round2_push.py --ref agent/m3545 --parent 723e12c \\
        --tree-from HEAD [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

REPO = "cxs503/TensorFEM"
ROOT = pathlib.Path(__file__).resolve().parents[1]
TOK = re.search(r"//[^:@\s]+:([^@\s]+)@github\.com",
                open("/root/.git-credentials").read()).group(1)


def gh(method: str, path: str, body=None):
    for attempt in range(6):
        req = urllib.request.Request(
            "https://api.github.com" + path,
            data=json.dumps(body).encode() if body is not None else None, method=method)
        req.add_header("Authorization", "Bearer " + TOK)
        req.add_header("Accept", "application/vnd.github+json")
        try:
            r = urllib.request.urlopen(req, timeout=180)
            payload = r.read()
            return json.loads(payload) if payload else {}
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          check=True).stdout


def tree(rev: str) -> dict[str, tuple[str, str]]:
    out = {}
    for line in git("ls-tree", "-r", rev).decode().splitlines():
        meta, path = line.split("\t", 1)
        mode, _typ, sha = meta.split()
        out[path] = (mode, sha)
    return out


def main_() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="agent/m3545")
    ap.add_argument("--parent", default="723e12c")
    ap.add_argument("--tree-from", default="HEAD")
    ap.add_argument("--message", default="m3545: lossless union with main 723e12c (PR #92 round-2 cure)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    parent = subprocess.run(["git", "rev-parse", args.parent], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    remote_parent = gh("GET", f"/repos/{REPO}/git/commits/{parent}")
    old_head = gh("GET", f"/repos/{REPO}/git/ref/heads/{args.ref}")["object"]["sha"]
    print("parent(main)", parent, "| old branch head", old_head)

    local_tree = tree(args.tree_from)
    remote_tree = {e["path"]: (e["mode"], e["sha"])
                   for e in gh("GET", f"/repos/{REPO}/git/trees/{parent}?recursive=1")["tree"]
                   if e["type"] == "blob"}
    changed = {p: v for p, v in local_tree.items() if remote_tree.get(p) != v}
    deleted = [p for p in remote_tree if p not in local_tree]
    print("changed", len(changed), "deleted", len(deleted))
    if args.dry_run:
        print("\n".join(sorted(changed)))
        return 0

    entries = []
    for path, (mode, _sha) in sorted(changed.items()):
        content = git("show", f"{args.tree_from}:{path}")
        blob = gh("POST", f"/repos/{REPO}/git/blobs",
                  {"content": base64.b64encode(content).decode(), "encoding": "base64"})
        entry = {"path": path, "mode": mode, "type": "blob", "sha": blob["sha"]}
        entries.append(entry)
    for path in deleted:
        entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})

    new_tree = gh("POST", f"/repos/{REPO}/git/trees",
                  {"base_tree": remote_parent["tree"]["sha"], "tree": entries})
    commit = gh("POST", f"/repos/{REPO}/git/commits",
                {"message": args.message, "tree": new_tree["sha"], "parents": [parent]})
    gh("PATCH", f"/repos/{REPO}/git/refs/heads/{args.ref}",
       {"sha": commit["sha"], "force": True})
    print(json.dumps({"old_head": old_head, "new_head": commit["sha"],
                      "tree": new_tree["sha"], "parent": parent,
                      "local_tree": git("rev-parse", f"{args.tree_from}^{{tree}}").decode().strip()},
                     indent=1))
    pr = gh("GET", f"/repos/{REPO}/pulls/92")
    print("PR92:", pr["state"], "mergeable", pr.get("mergeable"),
          "state", pr.get("mergeable_state"), "head", pr["head"]["sha"][:8])
    return 0


if __name__ == "__main__":
    sys.exit(main_())