import json, os, urllib.request, urllib.parse

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
tok = open("/tmp/tok.txt").read().strip()

def get(path, params=None):
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=40).read())
    except Exception as e:
        return {"__err__": str(e)}

for kw in ("TensorFEM", "immersed boundary", "Aitken relaxation", "FSI coupling"):
    r = get("/tensor/negative-knowledge/search", {"q": kw, "top_k": 5, "min_score": 0.15})
    print("== neg", kw)
    if "__err__" in r:
        print(r); continue
    for it in (r.get("items") or r.get("results") or [])[:5]:
        print("  -", str(it.get("title"))[:80], "|", str(it.get("failure_reason"))[:160])
    print("  total", r.get("total"))

# my mission node
for nid in (17361, 17428):
    r = get(f"/tensor/campaigns/nodes/{nid}")
    print("== node", nid, json.dumps(r, ensure_ascii=False)[:900])