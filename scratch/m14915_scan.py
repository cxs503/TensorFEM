#!/usr/bin/env python3
"""m14915: login + recent reports scan (limitation/todo sections)."""
import json, os, urllib.request, urllib.error

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
print("token_len", len(tok))

d = req("GET", "/tensor/reports?limit=25", token=tok)
items = d if isinstance(d, list) else d.get("items", d.get("reports", []))
print("n_reports", len(items))
for r in items:
    print("----", r.get("id"), "|", r.get("scenario_key"), "|", r.get("title"))
    md = r.get("markdown") or ""
    low = md.lower()
    for kw in ["局限", "待办", "下一步", "limitation", "next step", "todo"]:
        idx = low.find(kw.lower())
        if idx >= 0:
            print("   [", kw, "]", md[idx:idx+350].replace("\n", " ")[:350])
            break
