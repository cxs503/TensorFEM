import json, os, urllib.request
base = "http://127.0.0.1:8010/api/v1"
def login():
    d = json.dumps({'user':'eco_admin','pass':'EcoTest!2026pass'}).encode()
    r = urllib.request.Request(base+'/auth', data=d, headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r).read())['token']
tok = login()
open('/tmp/tok','w').write(tok)
def get(p):
    r = urllib.request.Request(base+p, headers={'Authorization':'Bearer '+tok})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=15).read())
    except Exception as e:
        return {'ERR': repr(e)[:200]}
inv = get('/oss/inventory')
print('INV type', type(inv), list(inv)[:5] if isinstance(inv, dict) else len(inv))
items = inv if isinstance(inv, list) else inv.get('items') or inv.get('data') or []
print('n items', len(items))
cohe = [i for i in items if 'COHES' in json.dumps(i).upper()]
print('n cohesive', len(cohe))
for i in cohe[:10]:
    print(json.dumps(i)[:400])
