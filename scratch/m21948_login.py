import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

body = json.dumps({"username": user, "password": pw}).encode()
req = urllib.request.Request(base + "/auth/login", data=body,
                             headers={"Content-Type": "application/json"}, method="POST")
try:
    d = json.load(urllib.request.urlopen(req))
    tok = d.get("access_token") or d.get("token")
    print("LOGIN_OK", len(tok or ""))
    open("/tmp/tok.txt", "w").write(tok)
except Exception as e:
    print("LOGIN_FAIL", e)
    # try alternate endpoint
    req2 = urllib.request.Request(base + "/auth", data=body,
                                  headers={"Content-Type": "application/json"}, method="POST")
    d = json.load(urllib.request.urlopen(req2))
    tok = d.get("token") or d.get("access_token")
    print("LOGIN_OK_ALT", len(tok or ""))
    open("/tmp/tok.txt", "w").write(tok)
