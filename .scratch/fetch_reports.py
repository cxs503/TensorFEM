import json, os, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
U = os.environ.get("TENSOR_AGENT_USER")
P = os.environ.get("TENSOR_AGENT_PASS")


def post(path, body, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get(path, token=None):
    req = urllib.request.Request(BASE + path)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


if __name__ == "__main__":
    tok = post("/auth/login", {"username": U, "password": P})["access_token"]
    open("/tmp/tok_m6848.txt", "w").write(tok)
    for q in ["/tensor/reports?mission_id=6848", "/tensor/reports?limit=5"]:
        try:
            d = get(q, tok)
            print("==", q)
            print(json.dumps(d, ensure_ascii=False)[:4000])
        except Exception as e:
            print(q, "ERR", e)