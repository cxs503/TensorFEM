#!/usr/bin/env python3
"""Login + fetch tensor report by id (read-only helper)."""
import json
import os
import sys
import urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]


def login() -> str:
    body = json.dumps({"username": user, "password": pw}).encode()
    req = urllib.request.Request(
        f"{base}/auth/login",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        d = json.load(r)
    tok = d.get("access_token") or d.get("token")
    if not tok:
        raise SystemExit(f"login failed: {d}")
    return tok


def main() -> None:
    tok = login()
    rid = sys.argv[1] if len(sys.argv) > 1 else "19643"
    req = urllib.request.Request(
        f"{base}/tensor/reports/{rid}",
        headers={"Authorization": f"Bearer {tok}"},
    )
    with urllib.request.urlopen(req) as r:
        d = json.load(r)
    d = d.get("data", d)
    print("TITLE:", d.get("title"))
    print("MISSION:", d.get("mission_id"))
    md = d.get("markdown", "")
    print("LEN:", len(md))
    print(md[:8000])


if __name__ == "__main__":
    main()
