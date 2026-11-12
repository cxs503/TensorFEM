import os, json, urllib.request, urllib.error

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER")
PWD = os.environ.get("TENSOR_AGENT_PASS")


def req(path, method="GET", body=None, token=None):
    url = BASE + "/" + path.lstrip("/")
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


def login():
    st, d = req("auth/login", "POST", {"username": USER, "password": PWD})
    tok = None
    if isinstance(d, dict):
        tok = d.get("access_token") or d.get("token")
        if not tok and isinstance(d.get("data"), dict):
            tok = d["data"].get("access_token") or d["data"].get("token")
    return tok, st, d


if __name__ == "__main__":
    tok, st, d = login()
    print("login status", st, "token_len", len(tok or ""))
    if not tok:
        print(json.dumps(d)[:800])
    else:
        open("/tmp/tok_m15654.txt", "w").write(tok)