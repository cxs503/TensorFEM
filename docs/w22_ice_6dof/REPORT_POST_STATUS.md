# W22 ice-6DOF report — platform post status

- Report (rendered, committed): `papers/reports/w22_ice_6dof2.md`
- Payload builder: `python3 benchmarks/w22_ice_6dof_report.py --payload /tmp/m14517_report.json`
  (reads only `docs/w22_ice_6dof/*.json` + `docs/w22_ice_6dof/figures/*.png`; payload ~98 kB,
  4 PNG figures, 54 kB total)
- Post target: `POST $TENSOR_API_BASE/tensor/reports` with `Authorization: Bearer <token>`,
  body `mission_id = 14517`, `scenario_key = ice`.

## Status at delivery time (2026-11-11 ~08:20 UTC)

The platform API on `127.0.0.1:8010` was **unreachable** (`ConnectionRefusedError(111)` on
`/health`, three retries with backoff; it answered `/health` normally at 07:30 UTC at the
start of this mission). Therefore the report could not be POSTed from this worktree, and no
receipt exists. This note is the audit trail for that failure — not a claim that the report
was published.

Retry (once the API answers `/health`):

```bash
python3 benchmarks/w22_ice_6dof_report.py --payload /tmp/m14517_report.json
python3 - <<'PY'
import json, os, urllib.request
base = os.environ["TENSOR_API_BASE"]
req = urllib.request.Request(base + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
token = json.loads(urllib.request.urlopen(req, timeout=15).read())
body = open("/tmp/m14517_report.json", "rb").read()
req = urllib.request.Request(base + "/tensor/reports", data=body,
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer " + (token.get("access_token") or token.get("token"))},
    method="POST")
print(urllib.request.urlopen(req, timeout=60).read()[:400])
PY
# then verify: GET $TENSOR_API_BASE/tensor/reports?mission_id=14517
```
