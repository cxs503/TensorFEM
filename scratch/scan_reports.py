import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")


def req(path, data=None, token=None):
    r = urllib.request.Request(base + path, method="POST" if data is not None else "GET")
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    body = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(r, body, timeout=60) as f:
        return json.load(f)


tok = req("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                          "password": os.environ["TENSOR_AGENT_PASS"]})
token = tok.get("access_token") or tok.get("token")
open("/tmp/tok.txt", "w").write(token)
d = req("/tensor/reports?limit=15", token=token)
items = d if isinstance(d, list) else (d.get("items") or d.get("data") or [])
for r in items:
    print("-", r.get("id"), (r.get("title") or "")[:80])