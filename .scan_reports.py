import urllib.request, json, os, re

base = os.environ.get('TENSOR_API_BASE', 'http://127.0.0.1:8010/api/v1').rstrip('/')
payload = json.dumps({'username': os.environ['TENSOR_AGENT_USER'], 'password': os.environ['TENSOR_AGENT_PASS']}).encode()
req = urllib.request.Request(base + '/auth/login', data=payload, headers={'Content-Type': 'application/json'})
body = json.loads(urllib.request.urlopen(req).read())
token = body.get('token') or body.get('access_token')

# fetch specific TensorFEM reports by id
for rid in [8924, 8883]:
    req = urllib.request.Request(base + f'/tensor/reports/{rid}', headers={'Authorization': 'Bearer ' + token})
    try:
        r = json.loads(urllib.request.urlopen(req).read())
    except Exception as e:
        print(rid, 'ERR', e); continue
    md = r.get('markdown') or ''
    print('#####', rid, (r.get('title') or '')[:70], 'len=', len(md))
    for kw in ['局限', '待办', 'TODO', 'Outstanding', '未验证', '欠账', '尚未', '尚缺', '缺口', 'Limitation', 'Open item', 'future work', 'Future work']:
        for m in re.finditer(re.escape(kw), md):
            i = m.start()
            snippet = md[max(0,i-60):i+300].replace('\n', ' | ')
            print('  ['+kw+']', snippet[:320])
