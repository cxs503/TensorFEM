import os
import urllib.request, json

url = 'http://127.0.0.1:8010/api/v1/auth'
data = json.dumps({'user': 'eco_admin', 'pass': os.environ["TENSOR_API_PASSWORD"]}).encode()
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
try:
    resp = urllib.request.urlopen(req)
    print(json.loads(resp.read())['token'])
except Exception as e:
    print(f"Error: {e}")
