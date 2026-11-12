#!/usr/bin/env python3
"""Post the m15714 audit report to the platform (uses only stdlib)."""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, method="POST",
                                 data=json.dumps(payload).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


code, data = post("/auth/login", {"username": USER, "password": PASS})
tok = data["access_token"]
body = json.load(open("/tmp/report.json"))
code, out = post("/tensor/reports", body, tok)
print(code, json.dumps({k: out.get(k) for k in ("id", "title", "mission_id")}, ensure_ascii=False))