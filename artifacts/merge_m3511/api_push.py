"""Push local branch tip to GitHub through the Git Data API.

`git push` to github.com is not available in this sandbox (git://443 blocked / command
policy), so branch updates go through the REST Git Data API, the same transport
tools/pr_rebuild.py and tp_pusher use. Only the objects that GitHub does not already
have (blobs for the merge resolutions + the new evidence files) are uploaded; trees are
based on the current main tree.

Credentials are read from /root/.git-credentials; nothing is printed.
"""

import base64
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

WT = Path("/root/tp-agent/work/m4018")
REPO = "cxs503/TensorFEM"
REF = "refs/heads/agent/m3511"
MAIN = "723e12cb594d36c9d366c06ea4237d99429688cd"
OLD_TIP = "f9bd976b241ff797bc04759c8b35276676f41aa8"
AUTHOR = {"name": "TensorFEM agent m4018", "email": "tp-agent@tensor.local"}

TOKEN = re.search(
    r"//[^:@\s]+:([^@\s]+)@github\.com", Path("/root/.git-credentials").read_text()
).group(1)


def api(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(6):
        req = urllib.request.Request("https://api.github.com" + path, data=data, method=method)
        req.add_header("Authorization", "Bearer " + TOKEN)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (500, 502, 503, 504) and attempt < 5:
                time.sleep(2 * (attempt + 1))
                continue
            raise SystemExit(f"HTTP {exc.code} {method} {path}: {detail}")
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 5:
                raise SystemExit(f"network failure {method} {path}: {exc}")
            time.sleep(2 * (attempt + 1))
    raise SystemExit("unreachable")


def git(*args: str, binary: bool = False):
    out = subprocess.run(["git", *args], cwd=WT, capture_output=True, check=True).stdout
    return out if binary else out.decode().strip()


def local_commits() -> list[str]:
    head = git("rev-parse", "HEAD")
    merge = git("rev-parse", "HEAD^")
    return [merge, head]  # oldest first


def tree_entries(ref: str) -> list[dict]:
    """Entries that differ from MAIN (additions/modifications/deletions)."""
    raw = subprocess.run(
        ["git", "diff", "--raw", "--no-abbrev", "-z", "--no-renames", MAIN, ref],
        cwd=WT,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    entries = []
    i = 0
    while i < len(raw):
        chunk = raw[i]
        if not chunk.startswith(b":"):
            i += 1
            continue
        meta = chunk.decode().split()
        new_mode, new_sha = meta[1], meta[3]
        path = raw[i + 1].decode()
        i += 2
        entries.append(
            {"path": path, "mode": new_mode, "sha": None if set(new_sha) == {"0"} else new_sha}
        )
    return entries


def blob_exists(sha: str, known: set[str]) -> bool:
    return sha in known


def main() -> None:
    heads = local_commits()
    tree_head = git("rev-parse", "HEAD^{tree}")
    tree_merge = git("rev-parse", "HEAD^^{tree}")

    main_commit = api("GET", f"/repos/{REPO}/commits/{MAIN}")
    main_tree = main_commit["commit"]["tree"]["sha"]
    main_tree_listing = api("GET", f"/repos/{REPO}/git/trees/{main_tree}?recursive=1")
    if main_tree_listing.get("truncated"):
        raise SystemExit("main tree listing truncated")
    known = {e["sha"] for e in main_tree_listing.get("tree", []) if e.get("sha")}
    branch_tree = api("GET", f"/repos/{REPO}/git/commits/{OLD_TIP}")["tree"]["sha"]
    branch_listing = api("GET", f"/repos/{REPO}/git/trees/{branch_tree}?recursive=1")
    if branch_listing.get("truncated"):
        raise SystemExit("branch tree listing truncated")
    known |= {e["sha"] for e in branch_listing.get("tree", []) if e.get("sha")}
    print(f"blobs already on GitHub: {len(known)}")

    def build_tree(diff_ref: str, base: str) -> str:
        entries = tree_entries(diff_ref)
        uploaded = 0
        payload = []
        for e in entries:
            sha = e["sha"]
            if sha and not blob_exists(sha, known):
                content = git("cat-file", "blob", sha, binary=True)
                created = api(
                    "POST",
                    f"/repos/{REPO}/git/blobs",
                    {"content": base64.b64encode(content).decode(), "encoding": "base64"},
                )
                if created["sha"] != sha:
                    raise SystemExit(
                        f"blob sha mismatch for {e['path']}: local {sha} remote {created['sha']}"
                    )
                known.add(sha)
                uploaded += 1
            payload.append(
                {
                    "path": e["path"],
                    "mode": e["mode"],
                    "type": "blob",
                    "sha": sha,
                }
            )
        tree = api("POST", f"/repos/{REPO}/git/trees", {"base_tree": base, "tree": payload})
        print(f"tree {tree['sha']} for {diff_ref}: {len(payload)} entries, {uploaded} new blobs")
        return tree["sha"]

    # commit 1: the merge commit (parents = old branch tip, main)
    merge_tree = build_tree(heads[0], main_tree)
    assert merge_tree == tree_merge, f"merge tree {merge_tree} != local {tree_merge}"
    merge_msg = subprocess.run(
        ["git", "log", "-1", "--format=%B", heads[0]],
        cwd=WT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    merge_commit = api(
        "POST",
        f"/repos/{REPO}/git/commits",
        {"message": merge_msg, "tree": merge_tree, "parents": [OLD_TIP, MAIN], "author": AUTHOR},
    )
    print(
        "merge commit:",
        merge_commit["sha"],
        "tree ok" if merge_commit["tree"]["sha"] == tree_merge else "TREE MISMATCH",
    )

    # commit 2: evidence files on top
    ev_tree_api = api("GET", f"/repos/{REPO}/git/trees/{merge_tree}?recursive=1")
    ev_known = {e.get("sha") for e in ev_tree_api.get("tree", []) if e.get("sha")}
    entries = tree_entries(heads[1])
    payload, uploaded = [], 0
    for e in entries:
        sha = e["sha"]
        if sha and sha not in ev_known:
            content = git("cat-file", "blob", sha, binary=True)
            created = api(
                "POST",
                f"/repos/{REPO}/git/blobs",
                {"content": base64.b64encode(content).decode(), "encoding": "base64"},
            )
            if created["sha"] != sha:
                raise SystemExit(
                    f"blob sha mismatch for {e['path']}: local {sha} remote {created['sha']}"
                )
            uploaded += 1
        payload.append({"path": e["path"], "mode": e["mode"], "type": "blob", "sha": sha})
    ev_tree = api("POST", f"/repos/{REPO}/git/trees", {"base_tree": merge_tree, "tree": payload})
    assert ev_tree["sha"] == tree_head, f"evidence tree {ev_tree['sha']} != local {tree_head}"
    ev_msg = subprocess.run(
        ["git", "log", "-1", "--format=%B", heads[1]],
        cwd=WT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    ev_commit = api(
        "POST",
        f"/repos/{REPO}/git/commits",
        {
            "message": ev_msg,
            "tree": ev_tree["sha"],
            "parents": [merge_commit["sha"]],
            "author": AUTHOR,
        },
    )
    print(f"evidence commit: {ev_commit['sha']} ({uploaded} new blobs)")

    ref = api(
        "PATCH",
        f"/repos/{REPO}/git/{REF}",
        {
            "sha": ev_commit["sha"],
            "force": False,
            "message": "merge main into agent/m3511 (PR #89 cure)",
        },
    )
    print("ref updated:", ref.get("object", {}).get("sha"), "->", REF)
    print(json.dumps({"merge_commit": merge_commit["sha"], "evidence_commit": ev_commit["sha"]}))


main()
