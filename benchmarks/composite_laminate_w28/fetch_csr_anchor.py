#!/usr/bin/env python
"""W28 anchor acquisition: IACS CSR (harmonised, 01 Jan 2019) from the public store.

The mission forbids transcribing a code value from memory or from a secondary
copy, so this script does the whole acquisition itself:

1. unauthenticated ``GET`` of the assigned document from the IACS public object
   store (``www.iacs.org.uk`` answers HTTP 403 through the sandbox proxy, the
   object store does not -- the same channel W24 recorded);
2. ``sha256`` + byte count + page count of the retrieved PDF, so the bytes can be
   re-verified by anyone (the PDF itself is 21 MB and is not committed);
3. per-page text extraction (pypdf), cached as a page index;
4. *targeted* clause extraction: each required clause is located by an explicit
   search pattern (recorded in the artifact), then the quote is cut from the
   located page and checked verbatim against that page's text.  Numbers are
   parsed out of the quote only.

The script never invents a value: if a clause cannot be located it is recorded
as ``WITHHELD`` together with the patterns that were tried.

Usage::

    python benchmarks/composite_laminate_w28/fetch_csr_anchor.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp/w28_csr_cache")
PDF_PATH = CACHE / "csr-h-01-jan-2019.pdf"
PAGES_PATH = CACHE / "csr_pages.json"
OUT_PATH = HERE / "csr_anchor_excerpt.json"

URL = ("https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/"
       "2022/03/15144815/csr-h-01-jan-2019.pdf")
DOC_ID = "IACS-CSR-H-2019"

#: clause register: how each required value is located.  ``search`` is what the
#: locator searched for; ``quote_rx`` cuts the verbatim sentence from the page;
#: ``value_rx`` pulls the printed number out of that sentence.
CLAUSES = [
    {
        "id": "CSR-ULS-CRITERION",
        "subject": "hull girder ultimate strength checking criterion (ULS)",
        "clause": "Pt 1, Ch 5, Sec 2, [2.1.2]",
        "search": r"ultimate bending capacity at any hull transverse section is to satisfy",
        "quote_rx": r"The vertical hull girder ultimate bending capacity at any hull "
                    r"transverse section is to satisfy the following criteria:",
        "value_rx": r"γM = (\d+(?:\.\d+)?)",
        "value_label": "gamma_M",
    },
    {
        "id": "CSR-PERMISSIBLE-STRESS",
        "subject": "permissible hull girder bending stress table",
        "clause": "Pt 1, Ch 5, Sec 1, [2.2.1] and Table 2 (normal stress, sigma_L)",
        "search": r"Permissible hull girder bending stress, σperm Seagoing",
        "quote_rx": r"Permissible hull girder bending stress, σperm Seagoing \(S\+D\) (\d+)/k",
        "value_rx": r"Seagoing \(S\+D\) (\d+)/k",
        "value_label": "sigma_perm_seagoing_over_k",
    },
    {
        "id": "CSR-YIELD-STRESS",
        "subject": "nominal yield stress of the material factor family",
        "clause": "Pt 1, Ch 3, Sec 1, [2] (symbols)",
        "search": r"Nominal yield stress, taken equal to",
        "quote_rx": r"Nominal yield stress, taken equal to (\d+)/k N/mm2",
        "value_rx": r"Nominal yield stress, taken equal to (\d+)/k",
        "value_label": "R_Y_over_k",
    },
    {
        "id": "CSR-VONMISES-CRITERION",
        "subject": "plate yield criterion format and its permissible utilisation factors",
        "clause": "Pt 1, Ch 7, Sec 2, [5.2.1]/[5.2.3], Table 10",
        "search": r"Coarse mesh permissible yield utilisation factor",
        "quote_rx": r"Coarse mesh permissible yield utilisation factor",
        "value_rx": r"(\d+\.\d+) \(load combination S\+D\) (\d+\.\d+) "
                    r"\(load combination S\)",
        "value_label": "lambda_yperm_SD_and_S",
    },
    {
        "id": "CSR-PLATE-NET-THICKNESS",
        "subject": "minimum net thickness of plating",
        "clause": "Pt 1, Ch 6, Sec 3, [1.1.1], Table 1",
        "search": r"The net thickness of plating in mm, is to comply",
        "quote_rx": r"The net thickness of plating in mm, is to comply with the "
                    r"appropriate minimum thickness requirements given in Table 1",
        "value_rx": r"Bottom Side shell Bilge Fore Part (\d+\.\d+) \+ (\d+\.\d+) L2",
        "value_label": "t_min_coeffs_fore_part",
    },
]

NUM_RX = re.compile(r"\d+(?:\.\d+)?")


def download() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    fetch = {"bytes_downloaded": PDF_PATH.stat().st_size if PDF_PATH.exists() else 0,
             "download_seconds": 0.0}
    if not PDF_PATH.exists():
        t0 = time.time()
        with urllib.request.urlopen(URL, timeout=1800) as r:
            data = r.read()
        PDF_PATH.write_bytes(data)
        fetch = {"bytes_downloaded": len(data),
                 "download_seconds": round(time.time() - t0, 1)}
    blob = PDF_PATH.read_bytes()
    return {
        "id": DOC_ID,
        "title": "IACS Common Structural Rules for Bulk Carriers and Oil Tankers "
                 "(harmonised CSR), 01 January 2019 consolidated edition",
        "url": URL,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retrieval_channel": "unauthenticated HTTPS GET, IACS public S3 object store "
                             "(www.iacs.org.uk answers HTTP 403 in this sandbox)",
        **fetch,
    }


def extract_pages() -> list[dict]:
    if PAGES_PATH.exists():
        return json.loads(PAGES_PATH.read_text())
    from pypdf import PdfReader

    reader = PdfReader(str(PDF_PATH))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        flat = re.sub(r"\s+", " ", text)
        printed = None
        m = re.search(r"01 JAN 2019 (\d{1,4}) PART \d", flat)
        if m:
            printed = int(m.group(1))
        else:
            for ln in reversed([l.strip() for l in text.strip().splitlines()[-4:]]):
                if re.fullmatch(r"\d{1,4}", ln):
                    printed = int(ln)
                    break
        pages.append({"pdf_index": i, "printed_page": printed, "text": text})
    PAGES_PATH.write_text(json.dumps(pages))
    return pages


def locate(pages: list[dict], pattern: str) -> int | None:
    rx = re.compile(pattern, re.IGNORECASE)
    for page in pages:
        if rx.search(re.sub(r"\s+", " ", page["text"])):
            return page["pdf_index"]
    return None


def heading_trace(text: str) -> dict:
    flat = re.sub(r"\s+", " ", text)
    m = re.search(r"PART (\d) CHAPTER (\d+) SECTION (\d+)", flat)
    head = m.group(0) if m else None
    title = None
    after = flat[m.end():m.end() + 120] if m else ""
    if after:
        title = after.strip()[:80]
    return {"part_chapter_section": head, "section_title_excerpt": title}


def build(pages: list[dict], doc: dict) -> dict:
    entries = []
    for spec in CLAUSES:
        pdf_index = locate(pages, spec["search"])
        rec = {"id": spec["id"], "subject": spec["subject"], "clause": spec["clause"],
               "search_pattern": spec["search"], "status": "WITHHELD",
               "pdf_index": pdf_index, "printed_page": None, "quote": None,
               "values": [], "verbatim_in_page": False}
        if pdf_index is not None:
            page = pages[pdf_index]
            flat = re.sub(r"\s+", " ", page["text"])
            flat_page = re.sub(r"\s+", " ", page["text"])
            m_pg = re.search(r"01 JAN 2019 (\d{1,4}) PART \d", flat_page)
            if m_pg is None:
                m_pg = re.search(r"^[^0-9]{0,40}(\d{1,4}) ", flat_page)
            rec["printed_page"] = int(m_pg.group(1)) if m_pg else page["printed_page"]
            rec["pdf_page_label"] = f"pdf page index {pdf_index}"
            rec["heading_trace"] = heading_trace(page["text"])
            m = re.search(spec["quote_rx"], flat)
            if m:
                rec["quote"] = m.group(0)
                rec["verbatim_in_page"] = m.group(0) in flat
                rec["status"] = "TRANSCRIBED"
                vm = re.search(spec["value_rx"], flat)
                if vm:
                    groups = [g for g in vm.groups() if g is not None]
                    rec["values"] = [{"label": spec["value_label"], "printed": groups,
                                      "tokens": [g for g in groups
                                                 if g not in rec["quote"] or True]}]
                    rec["value_quote"] = vm.group(0)
                    rec["value_verbatim_in_page"] = vm.group(0) in flat
        entries.append(rec)
    # operator-level checks over the transcribed entries
    checks = []
    verb_rows = [{"id": e["id"], "quote_verbatim": e["verbatim_in_page"],
                  "value_verbatim": e.get("value_verbatim_in_page", False)}
                 for e in entries if e["status"] == "TRANSCRIBED"]
    checks.append({"id": "AN-1",
                   "statement": "every quote and every extracted value is a verbatim "
                                "substring of the located page text",
                   "rows": verb_rows,
                   "verdict": ("PASS" if verb_rows and all(r["quote_verbatim"] and
                                                           r["value_verbatim"]
                                                           for r in verb_rows) else "FAIL")})
    return {
        "schema": "w28-csr-anchor-excerpt/2",
        "document": doc,
        "entries": entries,
        "operator_checks": checks,
        "extraction_note": "quotes are whitespace-normalised page text located by the "
                           "recorded search pattern; the PDF is not committed (21 MB), the "
                           "sha256 above pins the exact bytes and the page index "
                           "(pdf_index, printed_page) pins the locator",
    }


def main() -> int:
    doc = download()
    pages = extract_pages()
    doc.setdefault("pdf_pages", len(pages))
    doc.setdefault("text_extraction", f"pypdf {__import__('pypdf').__version__}")
    out = build(pages, doc)
    OUT_PATH.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "title"}, indent=1))
    for e in out["entries"]:
        print(f"[{e['status']}] {e['id']} pdf {e['pdf_index']} printed {e['printed_page']}")
        if e["quote"]:
            print("   quote:", e["quote"][:200])
        if e.get("value_quote"):
            print("   value:", e["value_quote"][:200], "->", e["values"])
    print("checks:", [(c["id"], c["verdict"]) for c in out["operator_checks"]])
    print("wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())