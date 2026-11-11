#!/usr/bin/env python3
"""Check recent reports and feature ledger."""
import urllib.request, json

# Get token
auth_url = 'http://127.0.0.1:8010/api/v1/auth/login'
auth_data = json.dumps({"username": "eco_admin", "password": "EcoTest!2026pass"}).encode()
req = urllib.request.Request(auth_url, data=auth_data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req)
token = json.loads(resp.read())['access_token']

# Get reports
reports_url = 'http://127.0.0.1:8010/api/v1/tensor/reports?limit=20'
req2 = urllib.request.Request(reports_url, headers={'Authorization': f'Bearer {token}'})
resp2 = urllib.request.urlopen(req2)
data = json.loads(resp2.read())

print("=== Data keys:", list(data.keys()))
items = data.get('items', [])
print(f"\n=== Recent Reports ({len(items)}) ===")
for r in items[:20]:
    title = r.get('title', r.get('scenario_key', '?'))
    tags = r.get('tags', [])
    rid = r.get('id', r.get('report_id', '?'))
    print(f"  [{rid}] {str(title)[:70]}  tags={tags}")
    # Check for limitations/todo
    md = r.get('markdown', '')
    if '局限' in md or '待办' in md or 'TODO' in md or 'limitation' in md.lower():
        print(f"    -> HAS limitations/todo section")
