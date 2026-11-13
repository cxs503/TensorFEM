#!/usr/bin/env python3
"""Audit helper: login + GET endpoints on TensorPlatform API."""
import json
import os
import sys
import urllib.request
import urllib.error

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def req(method, path, body=None, tok=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"err": str(e)}
    except Exception as e:
        return -1, {"err": str(e)}


def main():
    st, d = req("POST", "/auth/login", {"username": USER, "password": PASS})
    if st not in (200, 201):
        print("LOGIN FAIL", st, d)
        sys.exit(1)
    tok = d.get("access_token") or d.get("token")
    open("/tmp/tok.txt", "w").write(tok)
    print("login ok", st)
    for p in sys.argv[1:]:
        st, d = req("GET", p, tok=tok)
        print("==", p, st)
        print(json.dumps(d, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
