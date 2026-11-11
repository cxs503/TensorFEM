import os, json, urllib.request

base = os.environ["TENSOR_API_BASE"]
body = json.dumps({
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
}).encode()
req = urllib.request.Request(base + "/auth/login", data=body,
                            headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req))
tok = d.get("access_token") or d.get("token")
open("/tmp/tok.txt", "w").write(tok)
print(tok[:20])