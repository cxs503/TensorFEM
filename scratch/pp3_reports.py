import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

body = json.dumps({"user": user, "pass": pw}).encode()
req = urllib.request.Request(base + "/auth", data=body,
                             headers={"Content-Type": "application/json"})
tok = json.load(urllib.request.urlopen(req))["token"]

req = urllib.request.Request(base + "/tensor/reports?limit=20",
                             headers={"Authorization": "Bearer " + tok})
d = json.load(urllib.request.urlopen(req))
items = d if isinstance(d, list) else d.get("items", d.get("reports", []))
for r in items:
    print(r.get("id"), "|", r.get("scenario_key", ""), "|", (r.get("title") or "")[:90])
