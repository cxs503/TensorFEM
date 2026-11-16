#!/usr/bin/env python3
"""Publish the m23702 GAP-PP-3 3-grid evidence-consistency fix report."""
from __future__ import annotations
import json, os, urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "report_m23702.json"


def login() -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def main() -> int:
    token = login()
    payload = json.loads(PAYLOAD.read_text())
    req = urllib.request.Request(BASE + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    print("posted:", resp.get("id", resp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
