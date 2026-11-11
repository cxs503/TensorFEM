import json, urllib.request
d = json.loads(urllib.request.urlopen('http://127.0.0.1:8010/openapi.json', timeout=20).read())
for p in sorted(d['paths']):
    if any(k in p for k in ('auth','oss','knowledge','kb','gap','capab','report')):
        print(p, list(d['paths'][p]))
