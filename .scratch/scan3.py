import json, os, urllib.request
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
tok = open(".scratch/tok.txt").read().strip()
r = urllib.request.Request(BASE+"/tensor/reports?limit=200"); r.add_header("Authorization","Bearer "+tok)
d = json.load(urllib.request.urlopen(r, timeout=40))
items = d if isinstance(d,list) else d.get("items", d.get("reports",[]))
print("total", len(items))
for x in items:
    t=(x.get("title") or "")
    if ("FEM" in t) or ("fem" in t.lower()):
        print(x.get("id"), x.get("mission_id"), t[:100])
