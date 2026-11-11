#!/usr/bin/env python3
"""m14915: fetch full markdown of selected reports."""
import json, os, sys, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

def req(method, path, body=None, token=None):
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read().decode())

tok = req("POST", "/auth/login", {"username": user, "password": pw}).get("access_token", "")
ids = [int(x) for x in sys.argv[1:]]
for rid in ids:
    d = req("GET", f"/tensor/reports/{rid}", token=tok)
    md = d.get("markdown") or ""
    print(f"===== REPORT {rid} | {d.get('title')} =====")
    print(md[:6000])
    print("...[truncated]" if len(md) > 6000 else "")
