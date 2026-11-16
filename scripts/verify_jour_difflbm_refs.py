#!/usr/bin/env python3
"""Independent DOI resolution of the JOUR-DIFFLBM fluid-core references.

Anchor: ``JOUR-DIFFLBM-REF-VERIFY`` (立项引用逐条 DOI 核证).

The audited mission (m22758) verified the three ``References`` entries of
``src/tensorfem/lbm_cavity_diff.py`` by *inspection only* ("本机无外网，未做 DOI
在线解析").  This script replaces inspection with an **external anchor**: every
DOI is resolved against Crossref and the resolved bibliographic record
(title / authors / year / journal / volume / pages) is compared with what the
source file claims.  A reference is graded

* ``RESOLVED_MATCH``  — DOI resolves and the record matches the citation;
* ``DOI_INVALID``     — DOI does not resolve at all (registrar 404);
* ``CHIMERA``         — DOI resolves, but the citation stitches together two
                        different works (title/authors from one paper,
                        journal/volume/pages from another);
* ``UNRESOLVED``      — network unavailable and no cached record.

Resolution results are cached in ``artifacts/refverify_m22758/crossref_cache.json``
so the check is reproducible offline (``--offline``).

Usage
-----
    python3 scripts/verify_jour_difflbm_refs.py            # resolve online, refresh cache
    python3 scripts/verify_jour_difflbm_refs.py --offline  # cached records only
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "tensorfem" / "lbm_cavity_diff.py"
EXAMPLES = ROOT / "examples" / "lbm_cavity_diff_validation.py"
OUT = ROOT / "artifacts" / "refverify_m22758"
CACHE = OUT / "crossref_cache.json"
MATRIX = OUT / "verification_matrix.json"

UA = "tensorplatform-board-audit/1.0 (mailto:audit@tensorplatform.local)"

# ISO-4 abbreviations used in the source file -> full Crossref container titles
_JOURNAL_ABBREV = {
    "J. Fluid Mech.": "Journal of Fluid Mechanics",
    "J. Comput. Phys.": "Journal of Computational Physics",
    "Phys. Fluids": "Physics of Fluids",
}

# The three citations exactly as they stand in the module References block.
DECLARED = {
    "Ladd": {
        "doi": "10.1017/S0022112094001771",
        "journal": "J. Fluid Mech.",
        "volume": "271",
        "pages": "285-309",
        "year": 1994,
        "authors": ["Ladd"],
        "title_fragments": ["numerical simulations of particulate suspensions"],
        "claimed_use": "moving-wall half-way bounce-back closure (lid injection)",
    },
    "Ghia": {
        "doi": "10.1016/0021-9991(82)90058-4",
        "journal": "J. Comput. Phys.",
        "volume": "48",
        "pages": "387-411",
        "year": 1982,
        "authors": ["Ghia", "Shin"],
        "title_fragments": ["high-re solutions for incompressible flow"],
        "claimed_use": "lid-driven cavity benchmark (Re = 100 centreline)",
    },
    "Guo": {
        "doi": "10.1063/1.1474820",
        "journal": "Phys. Fluids",
        "volume": "14",
        "pages": "2007-2010",
        "year": 2002,
        "authors": ["Guo", "Zheng", "Shi"],
        "title_fragments": ["discrete lattice effects on the forcing term"],
        "claimed_use": "discrete forcing term (Guo forcing, BGK collision)",
    },
}


def _get(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def resolve(doi: str, offline: bool, cache: dict) -> dict:
    """Return {'status': 'ok'|'404'|'no-network', 'record': {...}|None}."""
    if offline:
        rec = cache.get(doi)
        if rec:
            # a cached *404* is evidence too -- replay the registrar verdict
            return {"status": rec["status"], "record": rec.get("record"),
                    "source": "cache"}
        return {"status": "no-network", "record": None, "source": "cache-miss"}
    try:
        msg = _get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
        m = msg["message"]
        rec = {
            "title": (m.get("title") or [""])[0],
            "container_title": (m.get("container-title") or [""])[0],
            "volume": m.get("volume"),
            "page": m.get("page"),
            "issued": (m.get("issued") or {}).get("date-parts", [[None]])[0],
            "author_families": [a.get("family") for a in m.get("author", [])],
            "type": m.get("type"),
        }
        return {"status": "ok", "record": rec, "source": "crossref"}
    except urllib.error.HTTPError as e:
        return {"status": "404" if e.code == 404 else f"http-{e.code}",
                "record": None, "source": "crossref"}
    except Exception as e:  # network down / DNS / timeout
        return {"status": "no-network", "record": None,
                "source": f"crossref-error:{type(e).__name__}"}


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def grade(decl: dict, res: dict) -> tuple[str, list[str], dict]:
    """Grade one citation against its resolved record.

    Returns ``(grade, notes, element_flags)`` where ``element_flags`` records
    the six bibliographic elements checked (title/authors/year/journal/volume/
    pages) as booleans -- the audit figure and table are rendered from it.
    """
    notes: list[str] = []
    flags = {k: False for k in ("title", "authors", "year", "journal",
                                "volume", "pages")}
    if res["status"] == "no-network":
        return "UNRESOLVED", ["no network and no cached record"], flags
    if res["status"] != "ok" or not res["record"]:
        return ("DOI_INVALID",
                [f"DOI does not resolve (registrar status {res['status']})"], flags)
    rec = res["record"]
    title_ok = any(_norm(f) in _norm(rec["title"]) for f in decl["title_fragments"])
    authors_ok = all(any(a.lower() in (f or "").lower() for f in rec["author_families"])
                     for a in decl["authors"])
    year_ok = decl["year"] in (rec["issued"] or [])
    journal_ok = _JOURNAL_ABBREV.get(decl["journal"], decl["journal"]) == rec["container_title"]
    vol_ok = (rec["volume"] or "") == decl["volume"]
    pages_ok = _norm(rec["page"]) == _norm(decl["pages"])
    flags.update(title=title_ok, authors=authors_ok, year=year_ok,
                 journal=journal_ok, volume=vol_ok, pages=pages_ok)
    for what in ("title", "authors", "year", "journal", "volume", "pages"):
        if not flags[what]:
            notes.append(f"mismatch: {what}")
    if all(flags.values()):
        return "RESOLVED_MATCH", notes, flags
    if title_ok and authors_ok and year_ok and not (journal_ok and vol_ok and pages_ok):
        # the DOI names the work whose *title* is cited, but the journal/volume/
        # pages in the citation belong to some other record -> frankenstein cite
        return "CHIMERA", notes, flags
    return ("DOI_INVALID" if not title_ok else "CHIMERA"), notes, flags


def code_usage() -> dict:
    """Is the reference actually backing code (not a decorative citation)?"""
    src = SRC.read_text()
    # drop the module docstring: a citation listed in References must not count
    # as "code-backed" merely because the bibliography mentions the author
    body = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
    ex = EXAMPLES.read_text() if EXAMPLES.exists() else ""
    closure = re.search(r"graph-preserving lid injection", body) is not None and \
        re.search(r"2\.0 \* W9\[i\] / CS2", body) is not None
    ghia = "GHIA" in ex
    forcing = re.search(r"guo|forcing|force_?term", body, re.I) is not None
    return {
        "Ladd": {"used": bool(closure),
                 "where": "src/tensorfem/lbm_cavity_diff.py::_stream "
                          "(2.0*W9[i]/CS2 * rho * (c_i.u_lid) wall correction)"},
        "Ghia": {"used": bool(ghia),
                 "where": "examples/lbm_cavity_diff_validation.py GHIA reference table"},
        "Guo": {"used": bool(forcing),
                "where": "no forcing term anywhere in the module (grep: no 'guo'/"
                         "no force coupling)"},
    }


def disambiguate(offline: bool) -> dict:
    """For an unresolvable DOI, find (a) the DOI that really owns the cited
    *title* and (b) the paper that owns the cited *journal/volume/pages*.

    This is what turns a bare "DOI_INVALID" into an actionable repair spec.
    """
    if offline:
        c = OUT / "disambiguation.json"
        return json.loads(c.read_text()) if c.exists() else {}

    def search(q: str, key: str) -> list[dict]:
        url = "https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q)
        try:
            items = _get(url)["message"]["items"]
        except Exception:
            return []
        out = []
        for m in items:
            out.append({
                "doi": m.get("DOI"),
                "title": (m.get("title") or [""])[0],
                "container_title": (m.get("container-title") or [""])[0],
                "volume": m.get("volume"),
                "page": m.get("page"),
                "issued": (m.get("issued") or {}).get("date-parts", [[None]])[0],
                "author_families": [a.get("family") for a in m.get("author", [])],
            })
        return out

    d = {
        "cited_title_owner": search(
            "Discrete lattice effects on the forcing term in the lattice "
            "Boltzmann method Guo Zheng Shi", "title"),
        "cited_venue_owner": search(
            "An extrapolation method for boundary conditions in lattice "
            "Boltzmann method Guo Zheng Shi Physics of Fluids 2002", "venue"),
    }
    (OUT / "disambiguation.json").write_text(json.dumps(d, indent=2, ensure_ascii=False))
    return d


def _repair_spec(rows: dict, dis: dict) -> list[dict]:
    """Machine-generated repair list: one entry per reference that fails."""
    spec = []
    for name, r in rows.items():
        if r["grade"] == "RESOLVED_MATCH" and r["code_usage"]["used"]:
            continue
        item = {"ref": name, "grade": r["grade"],
                "declared_doi": r["declared"]["doi"], "actions": []}
        if r["grade"] != "RESOLVED_MATCH":
            top = next((h for h in (dis.get("cited_title_owner") or [])
                        if h["title"].lower().startswith(
                            r["declared"]["title_fragments"][0])), None)
            if top:
                item["actions"].append(
                    f"replace DOI with {top['doi']} (the work actually titled "
                    f"'{top['title']}', {top['container_title']} "
                    f"{top['volume']} ({top['issued'][0] if top['issued'] else '?'}))")
            venue = next((h for h in (dis.get("cited_venue_owner") or [])
                          if (h["volume"] or "") == r["declared"]["volume"]
                          and _norm(h["page"]) == _norm(r["declared"]["pages"])), None)
            if venue:
                item["actions"].append(
                    f"the cited venue '{r['declared']['journal']} "
                    f"{r['declared']['volume']}, {r['declared']['pages']}' belongs to a "
                    f"different paper: {venue['doi']} '{venue['title']}' — do not "
                    f"attach it to this title")
        if not r["code_usage"]["used"]:
            item["actions"].append(
                "no code path uses this reference (no forcing term in "
                "src/tensorfem/lbm_cavity_diff.py): either implement the forcing "
                "term or drop the citation / mark it as background reading")
        if item["actions"]:
            spec.append(item)
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    usage = code_usage()
    rows = {}
    for name, decl in DECLARED.items():
        res = resolve(decl["doi"], args.offline, cache)
        cache[decl["doi"]] = res
        g, notes, flags = grade(decl, res)
        rows[name] = {
            "declared": decl,
            "resolution": res,
            "grade": g,
            "notes": notes,
            "element_flags": flags,
            "code_usage": usage[name],
        }
    dis = disambiguate(args.offline)
    CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    MATRIX.write_text(json.dumps({
        "anchor": "JOUR-DIFFLBM-REF-VERIFY",
        "audited_mission": 22758,
        "audited_commit": "794447ec",
        "audit_mission": 22811,
        "source": "src/tensorfem/lbm_cavity_diff.py References block",
        "references": rows,
        "summary": {
            "n_references": len(rows),
            "n_resolved_match": sum(r["grade"] == "RESOLVED_MATCH" for r in rows.values()),
            "n_chimera": sum(r["grade"] == "CHIMERA" for r in rows.values()),
            "n_doi_invalid": sum(r["grade"] == "DOI_INVALID" for r in rows.values()),
            "n_unresolved": sum(r["grade"] == "UNRESOLVED" for r in rows.values()),
            "n_code_backed": sum(r["code_usage"]["used"] for r in rows.values()),
        },
        "disambiguation": dis,
        "repair_spec": _repair_spec(rows, dis),
    }, indent=2, ensure_ascii=False))

    print(f"{'ref':6} {'grade':16} {'code-backed':12} notes")
    for name, r in rows.items():
        print(f"{name:6} {r['grade']:16} {str(r['code_usage']['used']):12} "
              f"{'; '.join(r['notes'])[:90]}")
    m = json.loads(MATRIX.read_text())["summary"]
    print("summary:", json.dumps(m, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())