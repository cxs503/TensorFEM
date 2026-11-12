import json, os, urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PWD = os.environ["TENSOR_AGENT_PASS"]


def call(path, data=None, token=None, method=None):
    url = BASE + path
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"))
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


tok = call("/auth/login", {"username": USER, "password": PWD})
tok = tok.get("access_token") or tok.get("token")
open("/tmp/tok", "w").write(tok)
d = call("/tensor/reports?limit=20", token=tok)
items = d.get("items") or d.get("reports") or d
for r in items:
    print(r.get("id"), "|", r.get("scenario_key"), "|", (r.get("title") or "")[:100],
          "| mission", r.get("mission_id"))