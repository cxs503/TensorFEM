import json
import sys

sys.path.insert(0, "/root/tp-agent")
import tp_worker  # noqa: E402

report = json.load(open("/tmp/report.json", encoding="utf-8"))
token = tp_worker.login()
status, data = tp_worker.api("/tensor/reports", report, token)
print("status:", status)
if isinstance(data, dict):
    print("keys:", sorted(data.keys()))
    print("id:", data.get("id"), "mission:", data.get("mission_id"))
else:
    print(str(data)[:400])