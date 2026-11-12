#!/usr/bin/env python3
"""Helper: authenticated GET against the tensor platform API (read-only)."""
import json
import os
import sys
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PWD = os.environ["TENSOR_AGENT_PASS"]


def _req(path, data=None, token=None):
    url = BASE + path
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = "Bearer " + token
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, headers=hdr,
                               method="POST" if body else "GET")
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode())


def login():
    d = _req("/auth/login", {"username": USER, "password": PWD})
    return d.get("access_token") or d.get("token")


if __name__ == "__main__":
    tok = login()
    path = sys.argv[1]
    out = _req(path, token=tok)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:20000])