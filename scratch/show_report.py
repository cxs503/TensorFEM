import json, os, sys, urllib.request

BASE = os.environ["TENSOR_API_BASE"]
tok = open("/tmp/tok").read().strip()


def get(path):
    req = urllib.request.Request(BASE + path)
    req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


for rid in sys.argv[1:]:
    d = get("/tensor/reports/%s" % rid)
    r = d.get("item") or d
    md = r.get("markdown") or ""
    print("=" * 30, rid, r.get("title"))
    # print limitation section
    low = md.lower()
    for key in ("局限", "待办", "limitation"):
        i = low.rfind(key)
        if i > 0:
            print(md[max(0, i - 200): i + 1800])
            break