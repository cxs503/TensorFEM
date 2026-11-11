"""Pre-work probe: query platform knowledge surfaces for GAP-FEM-PARALLEL.

Pure stdlib (urllib) so it runs under the sandbox policy that blocks curl.
Usage: python3 scripts/kb_probe.py
"""
import json
import os
import urllib.parse
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def _req(path, data=None, token=None, method=None):
    url = BASE + path
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode()
    try:
        return json.loads(raw)
    except Exception:
        return raw


def login():
    for payload in ({"user": USER, "pass": PASS},
                    {"username": USER, "password": PASS}):
        try:
            d = _req("/auth", payload)
            if isinstance(d, dict):
                for k in ("token", "access_token"):
                    if k in d:
                        return d[k]
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError("login failed")


def main():
    tok = login()
    print("token ok, len", len(tok))
    probes = [
        "/knowledge?domain=TensorFEM&ktype=capability",
        "/tensor/knowledge?domain=TensorFEM",
        "/knowledge/search?q=%s" % urllib.parse.quote("domain decomposition"),
        "/negative-knowledge?keywords=TensorFEM",
        "/oss/solutions?feature_slug=GAP-FEM-PARALLEL",
        "/oss_solutions?feature_slug=GAP-FEM-PARALLEL",
    ]
    for p in probes:
        try:
            d = _req(p, token=tok)
            s = json.dumps(d)[:1500]
            print(f"\n=== {p} ===\n{s}")
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {p} === ERR {e}")


if __name__ == "__main__":
    main()