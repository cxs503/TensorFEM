import json, os, urllib.request
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
def req(path, body=None, token=None, method=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    r.add_header("Content-Type", "application/json")
    if token: r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.load(resp)
tok = req("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]})
token = tok.get("access_token") or tok.get("token")
open("/root/tp-agent/work/m2386/.scratch/tok.txt","w").write(token or "")
print("TOKEN", bool(token))
d = req("/tensor/reports?limit=25", token=token)
items = d if isinstance(d, list) else d.get("items", d.get("reports", []))
print("reports:", len(items))
for r in items:
    print(r.get("id"), "|", r.get("mission_id"), "|", (r.get("title") or "")[:90])
