import json, os, urllib.request
base = "http://127.0.0.1:8010/api/v1"
env = {}
for line in open('/root/tp-agent/agent.env'):
    line=line.strip()
    if '=' in line and not line.startswith('#'):
        k,v = line.split('=',1); env[k]=v
d = json.dumps({'user':env['TENSOR_AGENT_USER'],'pass':env['TENSOR_AGENT_PASS']}).encode()
r = urllib.request.Request(base+'/auth', data=d, headers={'Content-Type':'application/json'})
tok = json.loads(urllib.request.urlopen(r, timeout=15).read())['token']
open('/tmp/tok','w').write(tok)
def get(p):
    r = urllib.request.Request(base+p, headers={'Authorization':'Bearer '+tok})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=20).read())
    except Exception as e:
        return {'ERR': repr(e)[:200]}
inv = get('/oss/inventory')
items = inv if isinstance(inv, list) else (inv.get('items') or inv.get('data') or [])
print('inv keys', list(inv)[:6] if isinstance(inv,dict) else 'list', 'n', len(items))
s = json.dumps(inv)
import re
print('sample', s[:500])
