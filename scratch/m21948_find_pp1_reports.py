import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
tok = open("/tmp/tok.txt").read().strip()

def get(path):
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + tok})
    return json.load(urllib.request.urlopen(req))

found = []
total = None
off = 0
while True:
    d = get(f"/tensor/reports?limit=100&offset={off}")
    rows = d if isinstance(d, list) else (d.get("items") or d.get("reports") or d.get("data") or [])
    if total is None:
        total = d.get("total") if isinstance(d, dict) else None
    if not rows:
        break
    for r in rows:
        blob = json.dumps(r, ensure_ascii=False)
        if "GAP-PP-1" in blob or "gap_pp1" in blob or "ice_struct_contact" in blob or "patch_from_contact" in blob:
            found.append((r.get("id"), (r.get("title") or "")[:100], r.get("scenario_key", "")))
    if len(rows) < 100:
        break
    off += 100
    if off > 3000:
        print("stopped at 3000")
        break

print("total field:", total, "scanned:", off + len(rows))
print("GAP-PP-1 related reports:")
for f in found:
    print(f)
print("total found:", len(found))
