import json, os, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
user = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
pw = os.environ.get("TENSOR_AGENT_PASS", "")

def req(method, path, body=None, tok=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if tok:
        headers["Authorization"] = "Bearer " + tok
    r = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    return json.loads(urllib.request.urlopen(r).read())

tok = req("POST", "/auth/login", {"username": user, "password": pw}).get("access_token")

# try search / filter params
for q in ("/tensor/reports?limit=5&scenario_key=CASE-FEM-HEAT-L2",
          "/tensor/reports?limit=5&q=HEAT",
          "/tensor/reports?limit=5&search=heat"):
    try:
        d = req("GET", q, tok=tok)
        items = d.get("items") or d.get("reports") or d
        if isinstance(items, dict):
            items = items.get("items", [])
        print(q, "->", len(items))
        for r in items[:5]:
            print("   ", r.get("id"), "|", (r.get("title") or "")[:80])
    except Exception as e:
        print(q, "ERR", e)
