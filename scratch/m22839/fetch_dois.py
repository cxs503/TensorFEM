import json, urllib.request, urllib.error, urllib.parse

DOIS = [
    "10.1017/S0022112094001771",
    "10.1016/0021-9991(82)90058-4",
    "10.1103/PhysRevE.65.046308",
    "10.1063/1.1471914",
    "10.1063/1.1474820",
]
out = {}
for d in DOIS:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(d)
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            m = json.load(r)["message"]
        out[d] = {
            "http": 200,
            "title": m.get("title"),
            "authors": [(a.get("family"), a.get("given")) for a in m.get("author", [])],
            "container": m.get("container-title"),
            "volume": m.get("volume"),
            "page": m.get("page"),
            "year": m.get("issued", {}).get("date-parts"),
            "doi": m.get("DOI"),
        }
    except urllib.error.HTTPError as e:
        out[d] = {"http": e.code, "error": str(e)}
    except Exception as e:  # noqa
        out[d] = {"http": None, "error": repr(e)}
print(json.dumps(out, indent=2, ensure_ascii=False))