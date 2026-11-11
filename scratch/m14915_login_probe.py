#!/usr/bin/env python3
import json, os, urllib.request, urllib.error

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

def try_login(path, body):
    url = base + path
    data = json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method="POST")
    r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return "OK " + resp.read().decode()[:200]
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} " + e.read().decode()[:150]
    except Exception as e:
        return "ERR " + str(e)[:150]

bodies = [
    {"user": user, "pass": pw},
    {"username": user, "password": pw},
]
for path in ["/auth/login", "/auth/token", "/login", "/auth"]:
    for b in bodies:
        print(path, list(b.keys()), "->", try_login(path, b))
