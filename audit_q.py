#!/usr/bin/env python3
"""Audit helper: query tensor platform API for benchmark case + mission m15330 evidence."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def req(method, path, body=None, token=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def main():
    st, login = req("POST", "/auth/login", {"username": USER, "password": PASS})
    assert st == 200, (st, login)
    tok = login["access_token"]
    out = {}

    # 1) benchmark case rows
    st, cases = req("GET", "/benchmark/cases?anchor_slug=CASE-CANTILEVER-TIP", token=tok)
    out["benchmark_cases_status"] = st
    out["benchmark_cases"] = cases

    # 2) mission m15330 record
    for path in ("/missions/15330", "/missions/m15330"):
        st, m = req("GET", path, token=tok)
        if st == 200:
            out["mission_status"] = st
            out["mission"] = m
            break
        out.setdefault("mission_fail", []).append((path, st))

    # 3) reports for mission 15330
    st, reps = req("GET", "/tensor/reports?mission_id=15330", token=tok)
    out["reports_status"] = st
    out["reports"] = reps

    print(json.dumps(out, ensure_ascii=False, indent=1)[:12000])


if __name__ == "__main__":
    main()
