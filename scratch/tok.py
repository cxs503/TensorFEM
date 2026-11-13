import json, os, urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
user = os.environ.get("TENSOR_AGENT_USER", "")
pw = os.environ.get("TENSOR_AGENT_PASS", "")

def post(path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

for body in ({"username": user, "password": pw}, {"user": user, "pass": pw}):
    try:
        d = post("/auth/login", body)
    except Exception:
        try:
            d = post("/auth", body)
        except Exception as e:
            print("ERR", e)
            continue
    tok = d.get("access_token") or d.get("token")
    if tok:
        open("/tmp/tok.txt", "w").write(tok)
        print("OK len", len(tok), "keys", list(d)[:6])
        break
else:
    print("FAILED")