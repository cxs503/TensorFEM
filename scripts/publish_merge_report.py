"""Publish the PR #56 merge report to the platform reports API (token from env)."""
import json
import os
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PWD = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + token} if token else {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, r.read().decode()[:800]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:800]


status, body = post("/auth/login", {"username": USER, "password": PWD})
print("login:", status)
token = None
try:
    data = json.loads(body)
    token = data.get("access_token") or data.get("token")
except Exception:
    pass
if not token:
    raise SystemExit("no token: " + body)

report = json.loads(open("/tmp/report_3148.json").read())
print("report title:", report["title"][:70])
status, body = post("/tensor/reports", report, token)
print("post report:", status, body)