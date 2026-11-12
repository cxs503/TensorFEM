import json, os, urllib.request, sys
base = os.environ["TENSOR_API_BASE"].rstrip("/")
token = open("/tmp/tok.txt").read().strip()
r = urllib.request.Request(base + "/tensor/reports/" + sys.argv[1])
r.add_header("Authorization", "Bearer " + token)
d = json.load(urllib.request.urlopen(r, timeout=60))
md = d.get("markdown") or d.get("summary") or ""
print(d.get("title"))
print("=====")
print(md[:3000])