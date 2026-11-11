import json, os, urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]
data = json.dumps({"username": user, "password": pw}).encode()
req = urllib.request.Request(base + "/auth/login", data=data, headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=20)
    d = json.load(r)
except Exception as e:
    print("ERR", e)
    raise SystemExit(1)
tok = d.get("access_token") or d.get("token") or d.get("data", {}).get("access_token")
print((tok or "")[:30])
open(os.path.join(os.path.dirname(__file__), "tok.txt"), "w").write(tok or "")