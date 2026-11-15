import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
tok = open("/tmp/tok.txt").read().strip()

def get(path):
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + tok})
    return json.load(urllib.request.urlopen(req))

d = get("/tensor/reports/19692")
r = d.get("data", d) if isinstance(d, dict) else d
print("id:", r.get("id"))
print("title:", r.get("title"))
print("scenario_key:", r.get("scenario_key"))
print("mission_id:", r.get("mission_id"))
print("created_at:", r.get("created_at"))
md = r.get("markdown") or ""
print("markdown_len:", len(md))
# print the limitations / todo sections
import re
for m in re.finditer(r"(?im)^#{1,3}\s*.*(局限|待办|TODO|下一步|next)", md):
    s = m.start()
    print("---- section at", s, "----")
    print(md[s:s+1200])
