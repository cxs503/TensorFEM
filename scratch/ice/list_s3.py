import urllib.request, urllib.parse, xml.etree.ElementTree as ET, re, sys
base="https://iacs.s3.af-south-1.amazonaws.com/"
ns={"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
tok=None; keys=[]
prefix=sys.argv[1] if len(sys.argv)>1 else ""
for _ in range(60):
    q={"list-type":"2","max-keys":"1000","prefix":prefix}
    if tok: q["continuation-token"]=tok
    url=base+"?"+urllib.parse.urlencode(q)
    raw=urllib.request.urlopen(url,timeout=60).read()
    root=ET.fromstring(raw)
    for c in root.findall("s:Contents",ns):
        keys.append(c.find("s:Key",ns).text)
    trunc=root.find("s:IsTruncated",ns).text
    tokEl=root.find("s:NextContinuationToken",ns)
    if trunc!="true" or tokEl is None: break
    tok=tokEl.text
print("total",len(keys))
pat=re.compile(sys.argv[2] if len(sys.argv)>2 else r'(I1|I2|I3|polar)',re.I)
hits=[k for k in keys if pat.search(k.split('/')[-1])]
print("hits",len(hits))
for h in hits[:80]: print(h)
