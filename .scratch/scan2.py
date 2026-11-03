import json, os, urllib.request
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
tok = open(".scratch/tok.txt").read().strip()
def req(path):
    r = urllib.request.Request(BASE+path)
    r.add_header("Authorization","Bearer "+tok)
    with urllib.request.urlopen(r, timeout=40) as resp:
        return json.load(resp)
for p in ["/tensor/reports?limit=200", "/tensor/missions?limit=60"]:
    try:
        d = req(p)
    except Exception as e:
        print(p, "ERR", e); continue
    items = d if isinstance(d, list) else d.get("items", d.get("reports", d.get("missions", [])))
    print("###", p, len(items))
    for r in items:
        if p.startswith("/tensor/reports"):
            t = (r.get("title") or "")
            if "FEM" in t or "Shell" in t or "板" in t or "壳" in t:
                print(r.get("id"), r.get("mission_id"), t[:95])
        else:
            print(r.get("id"), r.get("status"), (r.get("repo") or r.get("branch") or "")[:18], (r.get("goal") or r.get("title") or "")[:80].replace("\n"," "))
