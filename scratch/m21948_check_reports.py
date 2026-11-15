import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
tok = open("/tmp/tok.txt").read().strip()

def get(path):
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + tok})
    return json.load(urllib.request.urlopen(req))

d = get("/tensor/reports?limit=60")
rows = d if isinstance(d, list) else (d.get("items") or d.get("reports") or d.get("data") or [])
print("n_reports:", len(rows))
for r in rows:
    print(r.get("id"), "|", (r.get("title") or "")[:95], "|", r.get("scenario_key", ""))
