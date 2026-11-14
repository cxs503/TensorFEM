#!/usr/bin/env python3
"""Fetch benchmark-case ground truth for CASE-FEM-CANTILEVER-TIP from the platform API."""
import json
import os
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
OUT = os.environ.get("OUT", "/tmp/m19999_bench.json")


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
        return e.code, (e.read().decode() or "")[:300]


def main():
    st, login = req("POST", "/auth/login", {"username": USER, "password": PASS})
    tok = login["access_token"] if st == 200 else None
    out = {"login_status": st}
    candidates = [
        "/benchmark/cases?anchor_slug=CASE-FEM-CANTILEVER-TIP",
        "/tensor/benchmark/cases?anchor_slug=CASE-FEM-CANTILEVER-TIP",
        "/benchmark-cases?anchor_slug=CASE-FEM-CANTILEVER-TIP",
        "/benchmarks/cases?anchor_slug=CASE-FEM-CANTILEVER-TIP",
        "/tensor/maturity/gate?repo=TensorFEM",
        "/missions/19999",
        "/mission/19999",
    ]
    for p in candidates:
        s, d = req("GET", p, token=tok)
        out[p] = {"status": s, "data": d if s == 200 else str(d)[:200]}
    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()