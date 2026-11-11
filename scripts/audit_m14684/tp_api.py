"""Minimal TensorPlatform API helper for the m14959 audit mission.

Keeps credentials out of the shell: the login payload is read from the
worktree-local get_token.py, and the bearer token is cached in-process only.
"""
from __future__ import annotations

import json
import re
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
_TOKEN = None


def _creds():
    src = open("/root/tp-agent/work/m14959/get_token.py").read()
    user = re.search(r"'user':\s*'([^']+)'", src).group(1)
    pw = re.search(r"'pass':\s*'([^']+)'", src).group(1)
    return user, pw


def token() -> str:
    global _TOKEN
    if _TOKEN:
        return _TOKEN
    user, pw = _creds()
    req = urllib.request.Request(
        f"{BASE}/auth", data=json.dumps({"user": user, "pass": pw}).encode(),
        headers={"Content-Type": "application/json"})
    _TOKEN = json.loads(urllib.request.urlopen(req).read())["token"]
    return _TOKEN


def call(path: str, method: str = "GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}/{path.lstrip('/')}", data=data,
                                 method=method,
                                 headers={"Authorization": f"Bearer {token()}",
                                          "Content-Type": "application/json"})
    try:
        raw = urllib.request.urlopen(req).read()
    except urllib.error.HTTPError as exc:  # surface the server message
        return {"status": exc.code, "raw": exc.read().decode()[:2000]}
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw.decode()[:2000]}


if __name__ == "__main__":
    import sys
    print(json.dumps(call(sys.argv[1]), ensure_ascii=False, indent=1)[:6000])