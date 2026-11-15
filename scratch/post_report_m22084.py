#!/usr/bin/env python3
"""Post the m22084 report using the standard agent credentials (env)."""
import json
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
report = json.loads((REPO / "scratch" / "report_m22084.json").read_text())

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

login = json.dumps({"username": user, "password": pw}).encode()
req = urllib.request.Request(base + "/auth/login", data=login,
                             headers={"Content-Type": "application/json"})
try:
    token = json.loads(urllib.request.urlopen(req).read())["token"]
except Exception:
    # fallback field names
    login = json.dumps({"user": user, "pass": pw}).encode()
    req = urllib.request.Request(base + "/auth", data=login,
                                headers={"Content-Type": "application/json"})
    token = json.loads(urllib.request.urlopen(req).read())["token"]

body = json.dumps(report).encode()
req = urllib.request.Request(
    base + "/tensor/reports", data=body,
    headers={"Content-Type": "application/json",
             "Authorization": f"Bearer {token}"},
)
resp = json.loads(urllib.request.urlopen(req).read())
print(json.dumps(resp, ensure_ascii=False)[:600])
