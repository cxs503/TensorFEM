import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")
def post(path, payload, token=None):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + token} if token else {})},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                           "password": os.environ["TENSOR_AGENT_PASS"]})
token = tok.get("access_token") or tok.get("token")
print("login ok:", bool(token))
rep = json.load(open("/tmp/report.json"))
out = post("/tensor/reports", rep, token)
print("report id:", out.get("id") or out.get("report_id") or out)
