#!/usr/bin/env python3
import urllib.request, json

# Get token
url = 'http://127.0.0.1:8010/api/v1/auth/login'
data = json.dumps({'username': 'eco_admin', 'password': 'EcoTest!2026pass'}).encode()
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req)
token = json.loads(resp.read())['access_token']

# Post report
with open('/tmp/report.json', 'r') as f:
    report_data = f.read().encode()

url = 'http://127.0.0.1:8010/api/v1/tensor/reports'
req = urllib.request.Request(url, data=report_data, headers={
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
})
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print(f"Report posted successfully: {result}")
