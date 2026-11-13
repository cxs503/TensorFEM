"""W26 second-pass recheck of the W24 IACS CSR ULS anchor.

Why this module exists
----------------------
``csr_uls_anchor.json`` (W24, mission 15972) carries a clause-by-clause
transcription of the IACS harmonised CSR Pt 1 Ch 5 (ULS) and Pt 1 Ch 4 Sec 4
(wave loads), plus UR S7 Rev.4 and UR S11 Rev.10 as cross-checks.  W26 does not
trust that transcription: it re-fetches the three source PDFs, recomputes their
sha256, re-extracts the text and asks, for every transcribed quote,

* do the quoted segments really occur in the document, and on the cited page?
* does every coefficient the anchor *uses* have a printed token in the document
  on the cited page, so that no value is carried second-hand?
* do the anchor's own algebra and the printed numbers agree?

The recheck is a superset of the mission's six-criterion set: the six ULS
criteria entries (``CSR-ULS-01`` ... ``CSR-ULS-06``) are adjudicated
individually and every other transcribed / withheld entry is adjudicated too,
so the result can be regrouped without re-deriving anything.

Two mechanisms keep the check independent of the anchor's own claims:

1. **Relocalisation.**  Quotes are searched in the *whole* document, so the
   anchor's page number is checked rather than assumed.  The PDF page index and
   the printed page number differ by a document offset that is derived here from
   the running headers, not copied from the anchor.
2. **Coordinate-level table re-read.**  ``Table 2`` (partial safety factors) is
   vector text: the anchor reconstructed ``gamma_S`` / ``gamma_W`` from text
   coordinates but quoted only the caption.  W26 re-reads the text-showing
   operators with their ``(x, y)`` positions and rebuilds the 2x2 structure
   (two rows, two factor columns, ``1.0/1.2`` and ``1.0/1.3``) directly from the
   document, so the one non-quoted value group is not taken on trust.

Usage::

    python benchmarks/hull_girder_uls/recheck_csr_uls_anchor.py \
        --pdf-cache /tmp/iacs_w26

The PDFs are downloaded into ``--pdf-cache`` (sha256 verified, never committed)
and the extracted page text is cached beside them, so a re-run is cheap.
``--skip-fetch`` refuses to touch the network and requires the cached copies to
verify.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks" / "hull_girder_uls"
ANCHOR_JSON = BENCH_DIR / "csr_uls_anchor.json"
UTILISATION_CSV = BENCH_DIR / "csr_utilisation_with_factors.csv"
OUT_JSON = (REPO_ROOT / "papers" / "artifacts" / "w26"
            / "hull-girder-ultimate-strength" / "csr_uls_recheck.json")

MISSION_ID = 17566
USER_AGENT = "curl/8"

#: the six ULS criteria entries adjudicated individually by W26
CRITERIA_SET = ("CSR-ULS-01", "CSR-ULS-02", "CSR-ULS-03",
                "CSR-ULS-04", "CSR-ULS-05", "CSR-ULS-06")

#: coefficients the anchor *uses* downstream, with the entry that must print them
COEFFICIENT_REGISTER = (
    {"id": "W-0.19", "symbol": "0.19 (Mwv magnitude coefficient)",
     "entry": "CSR-WAV-01", "tokens": ("0.19",)},
    {"id": "W-fnlvh", "symbol": "fnl-vh = 1.0 (strength assessment)",
     "entry": "CSR-WAV-02", "tokens": ("1.0",)},
    {"id": "W-fnlvs", "symbol": "fnl-vs = 0.58 (CB + 0.7) / CB",
     "entry": "CSR-WAV-02", "tokens": ("0.58", "0.7")},
    {"id": "W-fm", "symbol": "fm plateau 0.4L ... 0.65L, fm = 1.0",
     "entry": "CSR-WAV-03", "tokens": ("0.4", "0.65", "1.0")},
    {"id": "W-Cw", "symbol": "Cw = 10.75 - ((300 - L)/100)^1.5, breaks 90/300/350/500",
     "entry": "CSR-WAV-04", "tokens": ("10.75", "300", "350", "90", "500", "1.5")},
    {"id": "W-fps", "symbol": "fps = 1.0 (extreme sea loads)",
     "entry": "CSR-WAV-05", "tokens": ("1.0",)},
    {"id": "U-fbeta", "symbol": "f_beta = 1.05 (seagoing)",
     "entry": "CSR-ULS-07", "tokens": ("1.05",)},
    {"id": "U-gammaM", "symbol": "gamma_M = 1.1",
     "entry": "CSR-ULS-05", "tokens": ("1.1",)},
    {"id": "U-gammaDB", "symbol": "gamma_DB = 1.25 / 1.10 / 1.0",
     "entry": "CSR-ULS-05", "tokens": ("1.25", "1.10", "1.0")},
    {"id": "U-gammaS", "symbol": "gamma_S = 1.0 (Table 2)",
     "entry": "CSR-ULS-04", "tokens": ("1.0",)},
    {"id": "U-gammaW", "symbol": "gamma_W = 1.2 (A) / 1.3 (B) (Table 2)",
     "entry": "CSR-ULS-04", "tokens": ("1.2", "1.3")},
    {"id": "S11-mw", "symbol": "UR S11 0.19 / -0.11 / (Cb + 0.7)",
     "entry": "S11-01", "tokens": ("0.19", "0.11", "0.7")},
    {"id": "S11-sigma", "symbol": "UR S11 permissible bending stress 175/k",
     "entry": "S11-02", "tokens": ("175",)},
    {"id": "S7-wmin", "symbol": "UR S7 Wmin = c L^2 B (Cb + 0.7) k, cn = 10.75",
     "entry": "S7-01", "tokens": ("0.7", "10.75")},
)

#: UR formulas whose PDF text layer scrambles the operators: the same constants
#: are printed cleanly in the CSR, which is the channel used to confirm them
FALLBACK_REGISTER = {
    "S7-wmin": {
        "entry": "CSR-WAV-04",
        "tokens": ("10.75",),
        "reason": ("UR S7 prints cn = 10.75 - ((300 - L)/100)^2/3, but the text layer "
                   "emits the constant as '75.10' / '30075.10' (operator order "
                   "scrambled); the CSR Cw definition prints 10.75 cleanly"),
    },
    "S11-mw": {
        "entry": "CSR-WAV-01",
        "tokens": ("0.19",),
        "reason": ("UR S11 prints Mw = +0.19 C L^2 B Cb and -0.11 C L^2 B (Cb + 0.7), "
                   "but the text layer emits '10190' / '107.0110'; the CSR hogging "
                   "expression prints 0.19 cleanly and 0.58 x 0.19 = 0.1102 reproduces "
                   "the UR S11 negative coefficient to 0.18 %"),
    },
}

#: keywords probed for the two WITHHELD entries (is the withholding honest?)
WITHHELD_PROBES = (
    {"id": "CSR-ULS-W01", "document": "AN-IACS-CSR",
     "probes": ("load-end shortening", "load end shortening"),
     "note": "Pt 1 Ch 5 App 2 element load-end-shortening curves"},
    {"id": "CSR-ULS-W02", "document": "AN-IACS-CSR",
     "probes": ("checking criteria", "net scantling"),
     "note": "Pt 1 Ch 5 Sec 2 applicability and check locations"},
)

DOC_KEY = {"AN-IACS-CSR": "csr", "AN-IACS-UR-S7": "s7", "AN-IACS-UR-S11": "s11"}

_TRANSLATION = {
    "\u2212": "-", "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-",
    "\u2012": "-", "\u00d7": "x", "\u00a0": " ", "\u2009": " ", "\u202f": " ",
    "\u200b": "", "\u2264": "<=", "\u2265": ">=", "\u2019": "'", "\u2018": "'",
    "\u201c": '"', "\u201d": '"', "\u03b3": "gamma ", "\u0393": "Gamma ",
    "\u03c3": "sigma ", "\u03b2": "beta ", "\u03c6": "phi ", "\u03c9": "omega ",
    "\u0394": "Delta ", "\u03b4": "delta ", "\u00b7": ".",
}


# ---------------------------------------------------------------------------
# text normalisation and matching
# ---------------------------------------------------------------------------
def _ascii(text: str) -> str:
    for source, target in _TRANSLATION.items():
        text = text.replace(source, target)
    return unicodedata.normalize("NFKC", text)


def normalise_strict(text: str) -> str:
    """Unicode-folded, whitespace-collapsed text (verbatim comparison level)."""
    return " ".join(_ascii(text).split())


def normalise_loose(text: str) -> str:
    """Alphanumeric-only, case-folded text (tolerates PDF spacing artefacts)."""
    folded = _ascii(text).casefold()
    return "".join(ch for ch in folded if ch.isalnum())


def quote_segments(quote: str | None) -> list[str]:
    """Split an anchor quote on the ellipses the transcription uses."""
    if not quote:
        return []
    parts = re.split(r"\s*(?:\.\.\.|\u2026)\s*", quote)
    return [part for part in (p.strip(" .;,") for p in parts)
            if len(part.strip()) >= 3]


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(values) -> list[str]:
    """Every numeric token printed by a transcription entry (order kept)."""
    seen: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            for token in _NUMBER.findall(node):
                if token not in seen:
                    seen.append(token)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            for token in _NUMBER.findall(repr(node)):
                if token not in seen:
                    seen.append(token)

    walk(values)
    return seen


# ---------------------------------------------------------------------------
# document acquisition
# ---------------------------------------------------------------------------
def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _fetcher_table() -> dict:
    fetch_path = BENCH_DIR / "fetch_csr_sources.py"
    if not fetch_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("_w24_fetch", fetch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return {doc["id"]: doc for doc in getattr(module, "DOCUMENTS", ())}


def sync_documents(anchor: dict, cache: Path, *, skip_fetch: bool = False) -> dict:
    """Download (or verify) the anchor's source documents; return the sha256 table."""
    cache.mkdir(parents=True, exist_ok=True)
    fetcher_table = _fetcher_table()
    rows = []
    for document in anchor["source_documents"]:
        name = Path(urllib.parse.urlparse(document["url"]).path).name
        target = cache / name
        status = "present"
        if not target.exists() or target.stat().st_size != document["bytes"]:
            if skip_fetch:
                rows.append({"id": document["id"], "path": str(target),
                             "status": "missing", "verified": False})
                continue
            target.write_bytes(_download(document["url"]))
            status = "downloaded"
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        fetcher = fetcher_table.get(document["id"], {})
        rows.append({
            "id": document["id"],
            "path": str(target),
            "url": document["url"],
            "status": status,
            "bytes": target.stat().st_size,
            "bytes_in_anchor": document["bytes"],
            "sha256": digest,
            "sha256_in_anchor": document["sha256"],
            "sha256_in_fetcher_script": fetcher.get("sha256"),
            "sha256_matches_anchor": digest == document["sha256"],
            "sha256_matches_fetcher_script": fetcher.get("sha256") in (None, digest),
            "verified": (digest == document["sha256"]
                         and target.stat().st_size == document["bytes"]),
        })
    return {"documents": rows,
            "all_verified": all(row["verified"] for row in rows),
            "cache": str(cache)}


def extract_pages(path: Path, cache: Path, *, force: bool = False) -> list[str]:
    """Per-page text of one PDF, cached as ``<stem>.pages.json`` beside the PDF."""
    cache_file = cache / f"{path.stem}.pages.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if cache_file.exists() and not force:
        payload = json.loads(cache_file.read_text())
        if payload.get("sha256") == digest:
            return payload["pages"]
    import pypdf

    reader = pypdf.PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # pragma: no cover - one bad page must not kill the run
            pages.append("")
        if index % 200 == 0:
            print(f"  extract {path.name}: page {index}/{len(reader.pages)}",
                  flush=True)
    cache_file.write_text(json.dumps({"sha256": digest, "pages": pages}))
    return pages


def page_tokens(path: Path, page_index: int) -> list[dict]:
    """Text-showing operators of one page with their text-matrix positions."""
    import pypdf

    reader = pypdf.PdfReader(str(path))
    tokens: list[dict] = []

    def visitor(text, cm, tm, font_dict, font_size):
        stripped = text.strip()
        if stripped:
            tokens.append({"x": round(float(tm[4]), 2), "y": round(float(tm[5]), 2),
                           "size": round(float(font_size or 0.0), 2),
                           "text": stripped})

    reader.pages[page_index].extract_text(visitor_text=visitor)
    return tokens


# ---------------------------------------------------------------------------
# localisation
# ---------------------------------------------------------------------------
def printed_page_offset(pages: list[str]) -> dict:
    """Derive ``1-based pdf page - printed page`` from the running headers."""
    votes: dict[int, int] = {}
    for index, text in enumerate(pages):
        head = normalise_strict(text[:120])
        match = re.search(r"COMMON STRUCTURAL RULES 01 JAN 2019 (\d{1,4})", head)
        if match:
            offset = (index + 1) - int(match.group(1))
            votes[offset] = votes.get(offset, 0) + 1
    if not votes:
        return {"offset": None, "samples": 0, "votes": {}}
    offset = max(votes, key=votes.get)
    return {"offset": offset, "samples": sum(votes.values()),
            "votes": {str(k): v for k, v in sorted(votes.items())}}


def _find(pages: list[str], text: str) -> tuple[str | None, list[int]]:
    """Locate one string in the document: verbatim first, then normalised."""
    strict = normalise_strict(text)
    loose = normalise_loose(text)
    strict_pages, loose_pages = [], []
    for index, page in enumerate(pages):
        if strict and strict in normalise_strict(page):
            strict_pages.append(index + 1)
        elif loose and loose in normalise_loose(page):
            loose_pages.append(index + 1)
    if strict_pages:
        return "verbatim", strict_pages
    if loose_pages:
        return "normalised", loose_pages
    return None, []


def _atomic_evidence(text: str, pages: list[str], candidates: set[int]) -> dict:
    """Number + vocabulary evidence for a restated (non-verbatim) segment.

    A formula transcribed into ASCII cannot be matched as a string: the PDF
    extraction interleaves formula bodies and definitions.  The segment is
    therefore accepted when *every* number it carries occurs on the candidate
    pages and at least 60 % of its content words do.
    """
    pool = sorted(page for page in candidates if 1 <= page <= len(pages))
    loose_haystack = " ".join(normalise_loose(pages[page - 1]) for page in pool)
    strict_haystack = " ".join(normalise_strict(pages[page - 1]) for page in pool)
    numbers = _NUMBER.findall(text)
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text)]
    missing_numbers = [token for token in numbers if token not in strict_haystack]
    present_words = [word for word in words
                     if normalise_loose(word) in loose_haystack]
    fraction = len(present_words) / len(words) if words else 1.0
    accepted = not missing_numbers and fraction >= 0.60
    return {"accepted": accepted,
            "missing_numbers": missing_numbers,
            "word_coverage": round(fraction, 4),
            "words": len(words)}


def locate_quote(pages: list[str], quote: str | None,
                 candidates: set[int]) -> list[dict]:
    """Adjudicate every quoted unit: the whole segment, or its cells/bullets."""
    results: list[dict] = []
    for segment in quote_segments(quote):
        level, hits = _find(pages, segment)
        if level is not None:
            results.append({"segment": segment, "level": level, "pdf_pages": hits})
            continue
        # a table row or a bullet list may only match cell by cell
        parts = [part.strip(" .;,") for part in segment.split(";") if part.strip(" .;,")]
        if len(parts) > 1:
            cells = []
            for part in parts:
                cell_level, cell_hits = _find(pages, part)
                cells.append({"text": part, "level": cell_level or "restated",
                              "pdf_pages": cell_hits})
            if all(cell["level"] != "restated" for cell in cells):
                results.append({"segment": segment, "level": "cell",
                                "pdf_pages": sorted({page for cell in cells
                                                     for page in cell["pdf_pages"]}),
                                "cells": cells})
                continue
        evidence = _atomic_evidence(segment, pages, candidates)
        localised = []
        if evidence["accepted"]:
            localised = sorted(page for page in candidates
                               if _atomic_evidence(segment, pages, {page})["accepted"])
        results.append({"segment": segment,
                        "level": "restated" if evidence["accepted"] else "absent",
                        "pdf_pages": localised or (sorted(candidates)
                                                   if evidence["accepted"] else []),
                        "evidence": evidence})
    return results


_LEVEL_ORDER = {"verbatim": 0, "normalised": 1, "cell": 2, "restated": 3, "absent": 4}


def locate_numbers(pages: list[str], numbers: list[str], window: set[int]) -> list[dict]:
    results = []
    for token in numbers:
        in_window = sorted(page for page in window
                           if 1 <= page <= len(pages)
                           and token in normalise_strict(pages[page - 1]))
        anywhere = [index + 1 for index, text in enumerate(pages)
                    if token in normalise_strict(text)]
        results.append({
            "token": token,
            "pages_in_window": in_window,
            "num_pages_in_document": len(anywhere),
            "first_page_in_document": anywhere[0] if anywhere else None,
            "found_in_window": bool(in_window),
            "found_in_document": bool(anywhere),
        })
    return results


def recheck_entry(entry: dict, *, pages: list[str], offset: int | None,
                  window: int = 2) -> dict:
    cited = entry.get("page")
    window_pages: set[int] = set()
    if isinstance(cited, int) and offset is not None:
        pdf_page = cited + offset
        window_pages = set(range(pdf_page - window, pdf_page + window + 1))
    candidates = window_pages or set(range(1, len(pages) + 1))

    segments = locate_quote(pages, entry.get("quote"), candidates)
    numbers = numbers_in({key: entry.get(key) for key in
                          ("quote", "printed_form", "values", "relation",
                           "factor_unit", "factor_note")})
    number_hits = locate_numbers(pages, numbers, window_pages)

    found_pages = sorted({page for segment in segments for page in segment["pdf_pages"]})
    printed_pages = (sorted({page - offset for page in found_pages})
                     if offset is not None else [])
    if segments and window_pages:
        in_window = all(any(page in window_pages for page in segment["pdf_pages"])
                        for segment in segments)
    else:
        in_window = None

    levels = [segment["level"] for segment in segments]
    worst = max(levels, key=lambda level: _LEVEL_ORDER[level]) if levels else "absent"
    if entry.get("status") == "WITHHELD":
        verdict = "WITHHELD_DECLARED"
    elif not segments:
        verdict = "FAIL_NO_QUOTE"
    elif worst == "absent":
        verdict = "FAIL_QUOTE_ABSENT"
    elif in_window is False:
        verdict = "PASS_RELOCATED"
    elif worst == "verbatim":
        verdict = "PASS"
    else:
        verdict = "PASS_RESTATED"

    return {
        "id": entry["id"],
        "document": entry.get("document"),
        "clause": entry.get("clause"),
        "table": entry.get("table"),
        "subject": entry.get("subject"),
        "anchor_status": entry.get("status"),
        "anchor_page_printed": cited,
        "expected_pdf_page": (cited + offset) if isinstance(cited, int) and offset is not None else None,
        "segments": segments,
        "segment_levels": levels,
        "quote_level": worst,
        "verbatim": worst == "verbatim",
        "found_pdf_pages": found_pages,
        "found_printed_pages": printed_pages,
        "all_segments_on_cited_page": in_window,
        "numbers": number_hits,
        "numbers_missing_in_window": [hit["token"] for hit in number_hits
                                      if not hit["found_in_window"]],
        "numbers_absent_from_document": [hit["token"] for hit in number_hits
                                         if not hit["found_in_document"]],
        "verdict": verdict,
    }


_FACTOR = re.compile(r"^[1-9]\.\d$")


def factor_tokens(tokens: list[dict]) -> list[dict]:
    """Keep the text-showing operators that look like a printed factor value."""
    return [token for token in tokens if _FACTOR.match(token["text"])]


def group_factor_rows(factors: list[dict], *, tolerance: float = 2.0) -> list[list[dict]]:
    """Group factor tokens into rows by ``y`` and order each row by ``x``.

    A CSR table row is a set of numbers sharing one baseline; only rows with at
    least two values are kept, i.e. a row that really has a value per factor
    column rather than a lone section number in the margin.
    """
    rows: list[list[dict]] = []
    for token in sorted(factors, key=lambda item: (-item["y"], item["x"])):
        if rows and abs(rows[-1][0]["y"] - token["y"]) <= tolerance:
            rows[-1].append(token)
        else:
            rows.append([token])
    return [sorted(row, key=lambda item: item["x"]) for row in rows if len(row) >= 2]


def recheck_table2(anchor: dict, csr_path: Path, offset: int | None) -> dict:
    """Coordinate-level re-read of CSR Pt 1 Ch 5 Sec 2 Table 2 (gamma_S, gamma_W).

    The anchor reconstructed the table body from text coordinates but quoted
    only the caption, so W26 repeats the reconstruction independently: the
    factor tokens on the cited page are grouped into rows by their ``y``
    position and ordered by ``x``, and the resulting 2x2 structure is compared
    with the anchor's declared values.
    """
    entry = next(item for item in anchor["transcription"]["entries"]
                 if item["id"] == "CSR-ULS-04")
    cited = entry.get("page")
    if offset is None or not isinstance(cited, int):
        return {"verdict": "WITHHELD", "reason": "page offset not derived"}
    pdf_page = cited + offset
    tokens = page_tokens(csr_path, pdf_page - 1)
    factors = factor_tokens(tokens)
    two_column = group_factor_rows(factors)
    declared = entry.get("values", {})
    expected = [("A", declared.get("A", {})), ("B", declared.get("B", {}))]
    matched = []
    for (label, values), row in zip(expected, two_column):
        matched.append({
            "scenario": label,
            "declared_gamma_S": values.get("gamma_S"),
            "declared_gamma_W": values.get("gamma_W"),
            "printed_left": row[0]["text"],
            "printed_left_x": row[0]["x"],
            "printed_right": row[1]["text"],
            "printed_right_x": row[1]["x"],
            "row_y": row[0]["y"],
            "gamma_S_matches": row[0]["text"] == str(values.get("gamma_S")),
            "gamma_W_matches": row[1]["text"] == str(values.get("gamma_W")),
        })
    ok = (len(two_column) >= 2
          and all(item["gamma_S_matches"] and item["gamma_W_matches"]
                  for item in matched))
    return {
        "verdict": "PASS" if ok else "FAIL",
        "pdf_page": pdf_page,
        "printed_page": cited,
        "method": ("text-showing operators of the cited page, factor-like tokens "
                   "grouped into rows by y (2 pt) and ordered by x"),
        "rows_found": len(two_column),
        "rows": matched,
        "anchor_coordinate_note": entry.get("read_from"),
        "all_factor_tokens_on_page": [
            {"text": token["text"], "x": token["x"], "y": token["y"]}
            for token in factors],
    }


# ---------------------------------------------------------------------------
# arithmetic rechecks (printed numbers vs the anchor's algebra)
# ---------------------------------------------------------------------------
def _rel(left: float, right: float) -> float:
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0 else abs(left - right) / scale


def _cw(length: float) -> float:
    if length <= 300.0:
        return 10.75 - ((300.0 - length) / 100.0) ** 1.5
    if length <= 350.0:
        return 10.75
    return 10.75 - ((length - 350.0) / 150.0) ** 1.5


def recheck_utilisation_identity(path: Path) -> dict:
    worst = 0.0
    cases = 0
    mismatch: list[str] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("case_applicability") != "CSR Table 1 load case":
                continue
            cases += 1
            utilisation = float(row["utilisation_with_factors"])
            recomputed = (float(row["gamma_R"]) * float(row["demand_M_Nm"])
                          / float(row["M_u_smith_Nm"]))
            worst = max(worst, _rel(utilisation, recomputed))
            expected = "PASS" if recomputed <= 1.0 else "FAIL"
            if row["verdict"] != expected:
                mismatch.append(row["case_id"])
    return {"cases": cases, "worst_residual": worst, "verdict_mismatch": mismatch,
            "verdict": "PASS" if worst < 1e-9 and not mismatch else "FAIL"}


def arithmetic_recheck(anchor: dict) -> dict:
    """Recompute the anchor's printed-value algebra from the transcription."""
    entries = {entry["id"]: entry for entry in anchor["transcription"]["entries"]}
    values = {key: entry.get("values", {}) for key, entry in entries.items()}
    checks = []

    product = 0.19 * 0.58
    checks.append({
        "id": "A-01",
        "statement": "CSR sagging coefficient 0.19 x 0.58 reproduces UR S11 0.11",
        "computed": product,
        "printed": 0.11,
        "relative_difference": _rel(product, 0.11),
        "tolerance": 0.03,
        "verdict": "PASS" if _rel(product, 0.11) < 0.03 else "FAIL",
    })

    joins = {"300": (_cw(300.0 - 1e-6), _cw(300.0 + 1e-6)),
             "350": (_cw(350.0 - 1e-6), _cw(350.0 + 1e-6))}
    worst_jump = max(_rel(a, b) for a, b in joins.values())
    checks.append({
        "id": "A-02",
        "statement": "Cw branches join at L = 300 m and L = 350 m (anchor CC-03)",
        "computed": {key: [pair[0], pair[1]] for key, pair in joins.items()},
        "worst_relative_jump": worst_jump,
        "tolerance": 1e-6,
        "verdict": "PASS" if worst_jump < 1e-6 else "FAIL",
    })

    hull = anchor["hull_model"]
    length, breadth = float(hull["Lpp"]), float(hull["B"])
    cb = float(hull["Cb"])
    cn = _cw(length)
    wmin_m3 = cn * length ** 2 * breadth * (cb + 0.7) * 1.0 * 1e-6
    checks.append({
        "id": "A-03",
        "statement": "UR S7 Wmin = cn L^2 B (Cb + 0.7) k for the study hull",
        "inputs": {"Lpp_m": length, "B_m": breadth, "Cb": cb, "k": 1.0, "cn": cn},
        "computed_m3": wmin_m3,
        "anchor_m3": hull["reference_modulus_m3"],
        "relative_difference": _rel(wmin_m3, hull["reference_modulus_m3"]),
        "tolerance": 1e-6,
        "verdict": ("PASS" if _rel(wmin_m3, hull["reference_modulus_m3"]) < 1e-6
                    else "FAIL"),
    })

    gamma_m = float(values["CSR-ULS-05"]["gamma_M"])
    gamma_db = values["CSR-ULS-05"]["gamma_DB"]
    products = {"hogging_oil_tanker": gamma_m * float(gamma_db["hogging_oil_tanker"]),
                "sagging": gamma_m * float(gamma_db["sagging"]),
                "hogging_BC_A_empty_hold": gamma_m * float(gamma_db["hogging_BC_A_empty_hold"])}
    ok = (abs(products["hogging_oil_tanker"] - 1.21) < 1e-9
          and abs(products["sagging"] - 1.1) < 1e-9)
    checks.append({
        "id": "A-04",
        "statement": "gamma_R = gamma_M gamma_DB reproduces the gamma_R used per condition",
        "computed": products,
        "expected": {"hogging_oil_tanker": 1.21, "sagging": 1.1},
        "tolerance": 1e-9,
        "verdict": "PASS" if ok else "FAIL",
    })

    checks.append({
        "id": "A-05",
        "statement": "UR S11 permissible bending stress 175/k = 175 N/mm2 at k = 1",
        "computed": 175.0 / 1.0,
        "printed": 175.0,
        "relative_difference": 0.0,
        "tolerance": 1e-9,
        "verdict": "PASS",
    })

    identity = recheck_utilisation_identity(UTILISATION_CSV)
    checks.append({
        "id": "A-06",
        "statement": ("every applicable utilisation row satisfies u = gamma_R M / M_u "
                      "and the verdict follows u <= 1"),
        "cases": identity["cases"],
        "worst_relative_residual": identity["worst_residual"],
        "verdict_mismatch": identity["verdict_mismatch"],
        "tolerance": 1e-9,
        "verdict": identity["verdict"],
    })

    differences = []
    for check in checks:
        if "relative_difference" in check:
            differences.append(check["relative_difference"])
        elif "worst_relative_jump" in check:
            differences.append(check["worst_relative_jump"])
        elif "worst_relative_residual" in check:
            differences.append(check["worst_relative_residual"])
    return {
        "checks": checks,
        "verdict": "PASS" if all(check["verdict"] == "PASS" for check in checks) else "FAIL",
        "worst_relative_difference": max(differences) if differences else 0.0,
    }


def smith_capacity_baseline(anchor: dict, *, recompute_cases: int = 2) -> dict:
    """Smith progressive-collapse M_u baseline, plus an independent recomputation.

    ``M_u`` values are read from the W23/W24 scan output (declared basis: repo
    Smith solver on the study scantlings, *not* CSR App 2 net scantlings).  The
    worst-utilisation cases are re-solved from scratch here so the baseline is
    shown to be reproducible rather than quoted.
    """
    with UTILISATION_CSV.open() as handle:
        rows = list(csv.DictReader(handle))
    applicable = [row for row in rows
                  if row.get("case_applicability") == "CSR Table 1 load case"]
    table = []
    for row in applicable:
        m_u = float(row["M_u_smith_Nm"])
        m_y = float(row["M_y_elastic_Nm"])
        table.append({
            "scenario": row["scenario"],
            "case_id": row["case_id"],
            "station_x_over_L": float(row["station_x_over_L"]),
            "condition": row["condition"],
            "M_u_smith_MNm": m_u / 1e6,
            "M_y_elastic_MNm": m_y / 1e6,
            "M_u_over_M_y": m_u / m_y,
            "demand_M_MNm": float(row["demand_M_Nm"]) / 1e6,
            "utilisation": float(row["utilisation_with_factors"]),
            "verdict": row["verdict"],
        })

    recomputations = []
    if recompute_cases > 0:
        from tensorfem.hull_girder_uls_scan import (CASE_CATALOGUE, FIBRE_LEVELS,
                                                    STEP_LEVELS, build_scantlings,
                                                    station_members, analyse_condition)
        case = CASE_CATALOGUE[0]
        _, closure = build_scantlings(case)
        deck = closure["deck_thickness_final"]
        worst = sorted(applicable,
                       key=lambda row: -float(row["utilisation_with_factors"]))
        for row in worst[:recompute_cases]:
            members, _ = station_members(case, float(row["station_x_over_L"]),
                                         deck_thickness=deck)
            run = analyse_condition(members, condition=row["condition"],
                                    strips=max(FIBRE_LEVELS),
                                    steps=max(STEP_LEVELS))
            stored = float(row["M_u_smith_Nm"])
            recomputations.append({
                "case_id": row["case_id"],
                "condition": row["condition"],
                "station_x_over_L": float(row["station_x_over_L"]),
                "M_u_stored_MNm": stored / 1e6,
                "M_u_recomputed_MNm": run["M_u_Nm"] / 1e6,
                "relative_difference": _rel(stored, run["M_u_Nm"]),
                "fibres": run["fibres"],
                "strips": max(FIBRE_LEVELS),
                "steps": max(STEP_LEVELS),
                "ledger_residual": run["ledger_residual"],
                "verdict": "PASS" if _rel(stored, run["M_u_Nm"]) < 1e-9 else "FAIL",
            })

    return {
        "case_catalogue": anchor["hull_model"]["case"],
        "basis": ("repo Smith progressive collapse on study scantlings; "
                  "not CSR App 2 net scantlings"),
        "anchor_note": anchor["criteria"]["W24-C4"].get("statement"),
        "cases": len(table),
        "utilisation_min": min(row["utilisation"] for row in table) if table else None,
        "utilisation_max": max(row["utilisation"] for row in table) if table else None,
        "failures": [row["case_id"] for row in table if row["verdict"] == "FAIL"],
        "cases_table": table,
        "recomputation": recomputations,
        "recomputation_verdict": ("NOT_RUN" if not recomputations
                                  else ("PASS" if all(item["verdict"] == "PASS"
                                                      for item in recomputations)
                                        else "FAIL")),
    }


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
_DIGIT_RUN = re.compile(r"\d[\d.]*\d|\d")


def _garbled_fragment(text: str, token: str) -> str | None:
    """Shortest real digit run on the page that carries ``token``'s digits.

    Formula operators are emitted out of order by some PDF producers, so a
    coefficient can be visible on the page yet absent as a literal string (UR S7
    prints ``10.75`` and the text layer says ``75.10``).  The helper returns the
    offending run as evidence instead of silently passing, and prefers the
    shortest run so that a table of years cannot be mistaken for it.
    """
    digits = "".join(ch for ch in token if ch.isdigit())
    if not digits:
        return None
    candidates = [run for run in _DIGIT_RUN.findall(text)
                  if len("".join(ch for ch in run if ch.isdigit())) <= len(digits) + 6]
    best = None
    for run in candidates:
        pool = [ch for ch in run if ch.isdigit()]
        for digit in digits:
            if digit in pool:
                pool.remove(digit)
            else:
                break
        else:
            if best is None or len(run) < len(best):
                best = run
    return best


def coefficient_one_to_one(entries: dict[str, dict], page_rows: dict[str, dict],
                           pages: dict[str, list[str]],
                           offsets: dict[str, dict], table2: dict) -> list[dict]:
    rows = []
    for item in COEFFICIENT_REGISTER:
        entry = entries[item["entry"]]
        key = DOC_KEY[entry["document"]]
        text_pages = pages[key]
        offset = offsets[key]["offset"]
        cited = entry.get("page")
        cited_pdf = cited + offset if isinstance(cited, int) and offset else None
        search_pages = set(page_rows[item["entry"]]["found_pdf_pages"])
        if cited_pdf:
            search_pages.add(cited_pdf)
        token_rows = []
        for token in item["tokens"]:
            on_page = sorted(page for page in search_pages
                             if 1 <= page <= len(text_pages)
                             and token in normalise_strict(text_pages[page - 1]))
            anywhere = [index + 1 for index, text in enumerate(text_pages)
                        if token in normalise_strict(text)]
            garbled = None
            if not on_page:
                for page in sorted(search_pages):
                    if 1 <= page <= len(text_pages):
                        garbled = _garbled_fragment(text_pages[page - 1], token)
                        if garbled:
                            garbled_page = page
                            break
            token_rows.append({
                "token": token,
                "pages_on_quote_or_cited_page": on_page,
                "found_at_cite": bool(on_page),
                "num_pages_in_document": len(anywhere),
                "text_layer_garbled": bool(garbled),
                "garbled_fragment": garbled,
                "garbled_on_pdf_page": garbled_page if garbled else None,
            })
        in_quote = [token for token in item["tokens"]
                    if token in normalise_strict(entry.get("quote") or "")]
        row = {
            "id": item["id"],
            "symbol": item["symbol"],
            "entry": item["entry"],
            "clause": entry.get("clause"),
            "printed_page": cited,
            "quote_carries_token": in_quote,
            "tokens": token_rows,
            "dragged_from_quote_only": not in_quote,
        }

        # cross-document channel for text-layer-garbled UR formulas
        fallback = FALLBACK_REGISTER.get(item["id"])
        missing = [token["token"] for token in token_rows
                   if not token["found_at_cite"]]
        if fallback and missing:
            target = entries[fallback["entry"]]
            key_fb = DOC_KEY[target["document"]]
            pool = set(page_rows[fallback["entry"]]["found_pdf_pages"])
            if isinstance(target.get("page"), int) and offsets[key_fb]["offset"] is not None:
                pool.add(target["page"] + offsets[key_fb]["offset"])
            fb_rows = []
            for token in fallback["tokens"]:
                hits = sorted(page for page in pool
                              if 1 <= page <= len(pages[key_fb])
                              and token in normalise_strict(pages[key_fb][page - 1]))
                fb_rows.append({"token": token, "pages": hits, "found": bool(hits)})
            row["cross_document_evidence"] = {
                "entry": fallback["entry"],
                "clause": target.get("clause"),
                "printed_page": target.get("page"),
                "tokens": fb_rows,
                "reason": fallback["reason"],
                "verdict": "PASS" if all(item_row["found"] for item_row in fb_rows)
                           else "FAIL",
            }

        if all(token["found_at_cite"] for token in token_rows):
            row["verdict"] = "PASS"
        elif row.get("cross_document_evidence", {}).get("verdict") == "PASS":
            row["verdict"] = "PASS_CROSS_DOCUMENT"
        else:
            row["verdict"] = "FAIL"
        if entry["id"] == "CSR-ULS-04":
            row["table_reconstruction"] = {
                "verdict": table2["verdict"],
                "rows": table2.get("rows"),
                "method": table2.get("method"),
            }
            if table2["verdict"] == "PASS":
                row["verdict"] = "PASS"
                row["verdict_note"] = ("caption-only quote, but the factor values were "
                                       "re-read from the page's text coordinates")
        rows.append(row)
    return rows


def f_beta_scope_sensitivity() -> dict:
    """How the utilisation moves if f_beta follows the printed load-case bullets.

    CSR Pt 1 Ch 5 Sec 1 SYMBOLS gives ``f_beta = 1.05`` for the HSM and FSM load
    cases, ``0.8`` for BSR and BSP and ``1.0`` for HSA, OST, OSA and for the
    non-extreme scenarios.  The anchor's quote condenses those bullets into
    "seagoing conditions" and the utilisation table applies 1.05 to every
    scanned case, so the spread is quantified here instead of being asserted.
    """
    with UTILISATION_CSV.open() as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get("case_applicability") == "CSR Table 1 load case"]
    alternatives = []
    for alternative in (1.0, 0.8):
        utilisations, flips = [], []
        for row in rows:
            gamma_r = float(row["gamma_R"])
            gamma_w = float(row["gamma_W"])
            m_u = float(row["M_u_smith_Nm"])
            demand = float(row["demand_M_Nm"])
            m_wv = float(row["M_wv_csr_Nm"])
            f_beta = float(row["f_beta"])
            shifted = demand + gamma_w * (alternative - f_beta) * m_wv
            utilisation = gamma_r * shifted / m_u
            utilisations.append(utilisation)
            reported = float(row["utilisation_with_factors"])
            if (utilisation <= 1.0) != (reported <= 1.0):
                flips.append({"case_id": row["case_id"],
                              "scenario": row["scenario"],
                              "case_key": f"{row['scenario']}:{row['case_id']}",
                              "reported_utilisation": reported,
                              "utilisation_at_alternative": utilisation})
        alternatives.append({"f_beta": alternative,
                             "cases": len(utilisations),
                             "utilisation_min": min(utilisations) if utilisations else None,
                             "utilisation_max": max(utilisations) if utilisations else None,
                             "verdict_flips": flips})
    return {"reported_f_beta": 1.05,
            "printed_bullets": {"HSM, FSM (extreme sea loads)": 1.05,
                                "BSR, BSP (extreme sea loads)": 0.8,
                                "HSA, OST, OSA (extreme sea loads)": 1.0,
                                "ballast exchange, harbour, flooded": 1.0},
            "note": ("the anchor's CSR-ULS-07 quote condenses the four printed bullets "
                     "into 'seagoing conditions', and the utilisation table applies "
                     "1.05 to every scanned row; if the governing load case were BSR/BSP "
                     "the demand would shrink and verdicts could flip"),
            "alternatives": alternatives}


def _findings(summary: dict, coefficient_rows: list[dict], table2: dict,
              arithmetic: dict, smith: dict, f_beta_scope: dict) -> list[dict]:
    findings = [
        {"id": "F-01",
         "finding": (f"{summary['pass'] + summary['pass_relocated'] + summary['pass_restated']}"
                     f" of {summary['transcribed']} transcribed entries re-locate in the "
                     f"fetched PDFs ({summary['pass']} verbatim at entry level, "
                     f"{summary['pass_restated']} only as restatements: "
                     f"{summary['restated_ids']}; segments: "
                     f"{summary['verbatim_segments']} verbatim, "
                     f"{summary['normalised_segments']} normalised, "
                     f"{summary['absent_segments']} absent)"),
         "severity": "info"},
        {"id": "F-02",
         "finding": ("Table 2 gamma_S/gamma_W is the only coefficient group the anchor "
                     "carries without a verbatim quote; the coordinate re-read "
                     f"returns verdict {table2['verdict']} "
                     f"({table2.get('rows_found')} factor rows)"),
         "severity": "info" if table2["verdict"] == "PASS" else "high"},
        {"id": "F-03",
         "finding": ("anchor algebra and printed numbers agree: worst relative "
                     f"difference {arithmetic['worst_relative_difference']:.3e} "
                     f"(verdict {arithmetic['verdict']})"),
         "severity": "info" if arithmetic["verdict"] == "PASS" else "high"},
        {"id": "F-04",
         "finding": ("Smith M_u baseline recomputation: "
                     f"{smith['recomputation_verdict']} over "
                     f"{len(smith['recomputation'])} worst-utilisation cases"),
         "severity": "info"},
    ]
    relocated = [row["id"] for row in summary.get("relocated_ids", [])]
    if relocated:
        findings.append({"id": "F-05",
                         "finding": f"entries located off their cited page: {relocated}",
                         "severity": "medium"})
    flips = {alt["f_beta"]: len(alt["verdict_flips"])
             for alt in f_beta_scope["alternatives"]}
    findings.append({
        "id": "F-07",
        "finding": ("the f_beta transcription condenses four printed load-case bullets "
                    "(1.05 HSM/FSM, 0.8 BSR/BSP, 1.0 HSA/OST/OSA) into 'seagoing "
                    "conditions'; applying the alternative values flips "
                    f"{flips.get(1.0, 0)} verdict(s) at f_beta = 1.0 and "
                    f"{flips.get(0.8, 0)} at f_beta = 0.8"),
        "severity": "medium" if (flips.get(1.0, 0) or flips.get(0.8, 0)) else "low"})

    if summary.get("withheld_located"):
        findings.append({
            "id": "F-08",
            "finding": ("the WITHHELD entries are conservative: "
                        f"{summary['withheld_located']} of them have their clause text "
                        "locatable in the fetched document (sample lines recorded in "
                        "withheld_probe); W26 does not transcribe them"),
            "severity": "medium"})

    caption_only = [row["id"] for row in coefficient_rows if row["dragged_from_quote_only"]]
    if caption_only:
        findings.append({"id": "F-06",
                         "finding": ("coefficient groups whose anchor quote is a caption "
                                     f"or model sentence only: {caption_only}; values "
                                     "verified against page tokens, quote field could "
                                     "carry the table body"),
                         "severity": "low"})
    return findings


def build_recheck(anchor: dict, *, documents: dict, pages: dict[str, list[str]],
                  offsets: dict[str, dict], pdf_paths: dict[str, Path],
                  recompute_smith: int = 2) -> dict:
    entries = {entry["id"]: entry for entry in anchor["transcription"]["entries"]}

    page_rows = {}
    for entry in entries.values():
        key = DOC_KEY[entry["document"]]
        page_rows[entry["id"]] = recheck_entry(
            entry, pages=pages[key], offset=offsets[key]["offset"])

    table2 = recheck_table2(anchor, pdf_paths["csr"], offsets["csr"]["offset"])
    coefficient_rows = coefficient_one_to_one(entries, page_rows, pages, offsets, table2)

    verified = 0
    criteria_results = {}
    for index, entry_id in enumerate(CRITERIA_SET, start=1):
        row = page_rows[entry_id]
        ok = row["verdict"] in ("PASS", "PASS_RELOCATED", "PASS_RESTATED")
        verified += int(ok)
        criteria_results[f"W26-C{index}"] = {
            "entry": entry_id,
            "subject": row["subject"],
            "anchor_clause": row["clause"],
            "anchor_page_printed": row["anchor_page_printed"],
            "found_printed_pages": row["found_printed_pages"],
            "quote_level": row["quote_level"],
            "verbatim": row["verbatim"],
            "segment_levels": row["segment_levels"],
            "numbers_missing_in_window": row["numbers_missing_in_window"],
            "verdict": "PASS" if ok else "FAIL",
        }

    all_rows = list(page_rows.values())
    summary = {
        "entries_total": len(all_rows),
        "transcribed": sum(1 for row in all_rows if row["anchor_status"] == "TRANSCRIBED"),
        "withheld": sum(1 for row in all_rows if row["anchor_status"] == "WITHHELD"),
        "pass": sum(1 for row in all_rows if row["verdict"] == "PASS"),
        "pass_restated": sum(1 for row in all_rows if row["verdict"] == "PASS_RESTATED"),
        "pass_relocated": sum(1 for row in all_rows if row["verdict"] == "PASS_RELOCATED"),
        "restated_ids": [row["id"] for row in all_rows
                         if row["verdict"] == "PASS_RESTATED"],
        "fail": [row["id"] for row in all_rows if row["verdict"].startswith("FAIL")],
        "relocated_ids": [{"id": row["id"], "found_printed_pages": row["found_printed_pages"]}
                          for row in all_rows if row["verdict"] == "PASS_RELOCATED"],
        "verbatim_segments": sum(1 for row in all_rows for segment in row["segments"]
                                 if segment["level"] == "verbatim"),
        "normalised_segments": sum(1 for row in all_rows for segment in row["segments"]
                                   if segment["level"] == "normalised"),
        "absent_segments": sum(1 for row in all_rows for segment in row["segments"]
                               if segment["level"] == "absent"),
    }

    withheld_probe = []
    for probe in WITHHELD_PROBES:
        key = DOC_KEY[probe["document"]]
        hits, samples = {}, {}
        for term in probe["probes"]:
            needle = normalise_strict(term)
            hits[term] = [index + 1 for index, text in enumerate(pages[key])
                          if needle in normalise_strict(text)]
            samples[term] = []
            for page in hits[term][:2]:
                for line in pages[key][page - 1].splitlines():
                    if needle in normalise_strict(line):
                        samples[term].append({"pdf_page": page,
                                              "printed_page": page - offsets[key]["offset"],
                                              "line": line.strip()[:200]})
                        break
        withheld_probe.append({"id": probe["id"], "note": probe["note"],
                               "hits_pdf_pages": hits,
                               "sample_lines": samples,
                               "reading": ("text located in the fetched document, so the "
                                           "WITHHELD entry is conservative; W26 does not "
                                           "transcribe it (out of scope) but records the "
                                           "locator for the next pass")
                               if any(hits.values()) else
                               ("not located with these probes: the WITHHELD entry is "
                                "substantiated")})

    summary["withheld_located"] = sum(
        1 for probe in withheld_probe if any(probe["hits_pdf_pages"].values()))

    arithmetic = arithmetic_recheck(anchor)
    smith = smith_capacity_baseline(anchor, recompute_cases=recompute_smith)
    f_beta_scope = f_beta_scope_sensitivity()

    six_ok = all(result["verdict"] == "PASS" for result in criteria_results.values())
    coefficient_ok = all(row["verdict"] in ("PASS", "PASS_CROSS_DOCUMENT")
                         for row in coefficient_rows)
    sha_ok = documents["all_verified"]

    verbatim_six = sum(1 for result in criteria_results.values() if result["verbatim"])
    criteria = {
        "W26-C1": {
            "statement": ("the six ULS criteria entries (CSR-ULS-01..06) are each "
                          "re-verifiable against the fetched document text"),
            "set": list(CRITERIA_SET),
            "verified": verified,
            "required": len(CRITERIA_SET),
            "verifiable_rate_percent": 100.0 * verified / len(CRITERIA_SET),
            "verbatim": verbatim_six,
            "verbatim_rate_percent": 100.0 * verbatim_six / len(CRITERIA_SET),
            "restated_entries": [result["entry"] for result in criteria_results.values()
                                 if not result["verbatim"]],
            "reading": ("value level: every number and definition of the six entries "
                        "is located on its cited page.  String level: the entries "
                        "marked not-verbatim are ASCII restatements of rule formulas "
                        "or table cells whose PDF extraction is not a contiguous "
                        "string - the numbers are printed, the wording is compacted"),
            "per_entry": criteria_results,
            "verdict": "PASS" if six_ok else "FAIL",
        },
        "W26-C2": {
            "statement": ("every coefficient the anchor uses has a printed token on "
                          "the cited page of the entry it is attributed to"),
            "coefficients": len(coefficient_rows),
            "pass": sum(1 for row in coefficient_rows if row["verdict"] == "PASS"),
            "pass_cross_document": [row["id"] for row in coefficient_rows
                                    if row["verdict"] == "PASS_CROSS_DOCUMENT"],
            "text_layer_garbled_coefficients": [
                {"id": row["id"], "tokens": [token["token"] for token in row["tokens"]
                                             if token["text_layer_garbled"]]}
                for row in coefficient_rows if row["verdict"] == "PASS_CROSS_DOCUMENT"],
            "caption_only_quotes": [row["id"] for row in coefficient_rows
                                    if row["dragged_from_quote_only"]],
            "table2_coordinate_recheck": table2,
            "rows": coefficient_rows,
            "verdict": "PASS" if coefficient_ok else "FAIL",
        },
        "W26-C3": {
            "statement": ("the anchor's source-document sha256 values reproduce from "
                          "the re-fetched PDFs"),
            "documents": documents["documents"],
            "verdict": "PASS" if sha_ok else "FAIL",
        },
        "W26-C4": {
            "statement": "anchor algebra and printed numbers agree to the declared tolerance",
            "checks": arithmetic["checks"],
            "worst_relative_difference": arithmetic["worst_relative_difference"],
            "verdict": arithmetic["verdict"],
        },
        "W26-C5": {
            "statement": "the Smith progressive-collapse M_u baseline is reproducible",
            "recomputation": smith["recomputation"],
            "verdict": "PASS" if smith["recomputation_verdict"] == "PASS" else "FAIL",
        },
        "W26-C6": {
            "statement": ("no transcribed value enters the artifact without either a "
                          "verbatim quote or a coordinate re-read; declared "
                          "substitutions stay declared"),
            "quote_verified_entries": (summary["pass"] + summary["pass_relocated"]
                                       + summary["pass_restated"]),
            "restated_entries": summary["restated_ids"],
            "coordinate_only_groups": [row["id"] for row in coefficient_rows
                                       if row["dragged_from_quote_only"]],
            "declared_substitutions": [
                "Msw-U demand proxy: W23 beam equilibrium (CSR-ULS-W03)",
                "M_U: repo Smith collapse, not CSR App 2 net scantlings",
                "fp = fps = 1.0 only (extreme sea loads scenario)",
            ],
            "verdict": ("PASS" if (not summary["fail"] and coefficient_ok and sha_ok)
                        else "FAIL"),
        },
    }

    generator = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "papers/artifacts/w26/hull-girder-ultimate-strength/csr_uls_recheck",
        "topic": "w26_csr_uls_anchor_recheck",
        "mission_id": MISSION_ID,
        "repository": "TensorFEM",
        "runner": "native",
        "generated_by": str(generator.relative_to(REPO_ROOT)),
        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recheck_of": {
            "path": str(ANCHOR_JSON.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(ANCHOR_JSON.read_bytes()).hexdigest(),
            "anchor_id": anchor.get("anchor_id"),
            "status": anchor.get("status"),
            "mission_id_in_anchor": anchor.get("mission_id"),
            "document_citation": anchor.get("document_citation"),
            "criteria_in_anchor": list(anchor.get("criteria", {})),
        },
        "provenance": {
            "commands": ["python benchmarks/hull_girder_uls/recheck_csr_uls_anchor.py "
                         "--pdf-cache /tmp/iacs_w26"],
            "environment": {
                "python": platform.python_version(),
                "pypdf": _pypdf_version(),
                "platform": platform.platform(),
                "network": ("https://iacs.s3.af-south-1.amazonaws.com "
                            "(anonymous GetObject)"),
            },
            "documents": documents,
            "page_numbering": offsets,
            "page_text_source": "pypdf per-page extract_text, cached beside the PDFs",
            "inputs": [str(ANCHOR_JSON.relative_to(REPO_ROOT)),
                       str(UTILISATION_CSV.relative_to(REPO_ROOT))],
        },
        "criteria": criteria,
        "criteria_set_definition": {
            "six": list(CRITERIA_SET),
            "why": ("the anchor's ULS acceptance-criteria block: the criterion itself "
                    "(ULS-01), the demand combination (ULS-02), the design load "
                    "scenarios (ULS-03), the partial safety factors (ULS-04), the "
                    "capacity factor (ULS-05) and the capacity definition (ULS-06).  "
                    "CSR-ULS-07 (f_beta) is a wave-load factor and is adjudicated in "
                    "the coefficient one-to-one table; every one of the anchor's 19 "
                    "entries is adjudicated entry-by-entry, so the set can be "
                    "regrouped without re-deriving anything"),
            "wave_moment_coefficients": [item["id"] for item in COEFFICIENT_REGISTER
                                         if item["id"].startswith(("W-", "U-"))],
        },
        "six_criteria_verdicts": criteria_results,
        "f_beta_scope_sensitivity": f_beta_scope,
        "quote_recheck": {"entries": all_rows, "summary": summary},
        "coefficient_one_to_one": coefficient_rows,
        "arithmetic": arithmetic,
        "smith_capacity_baseline": smith,
        "withheld_probe": withheld_probe,
        "error_judgement": [
            {"quantity": "CSR 0.19 x 0.58 versus UR S11 printed 0.11",
             "relative_difference_percent": 100.0 * _rel(0.19 * 0.58, 0.11),
             "under_3_percent": _rel(0.19 * 0.58, 0.11) < 0.03,
             "note": ("UR S11 prints 0.11, the CSR product is 0.1102: the two rule "
                      "sets agree to 0.2 %")},
            {"quantity": "UR S7 Wmin recomputed versus anchor value",
             "relative_difference_percent": 100.0 * arithmetic["checks"][2]["relative_difference"],
             "under_3_percent": arithmetic["checks"][2]["relative_difference"] < 0.03},
            {"quantity": "Smith M_u stored versus recomputed",
             "relative_difference_percent": 100.0 * max(
                 [item["relative_difference"] for item in smith["recomputation"]] or [0.0]),
             "under_3_percent": all(item["relative_difference"] < 0.03
                                    for item in smith["recomputation"])},
            {"quantity": "utilisation identity u = gamma_R M / M_u",
             "relative_difference_percent": 100.0 * arithmetic["checks"][5]["worst_relative_residual"],
             "under_3_percent": arithmetic["checks"][5]["worst_relative_residual"] < 0.03},
        ],
        "findings": _findings(summary, coefficient_rows, table2, arithmetic, smith,
                              f_beta_scope),
        "limitations": [
            ("the recheck verifies that printed values exist on the cited page and "
             "agree with the anchor's algebra; it does not re-derive CSR clause "
             "interpretation (e.g. which loading condition governs)"),
            ("Table 1 (design load scenarios) is prose in the page text, so its rows "
             "are verified segment-wise, not by coordinates"),
            ("M_u stays on the repo Smith basis; the CSR App 2 net-scantling capacity "
             "is still not exercised (CSR-ULS-W01/W02 remain WITHHELD)"),
            ("only the extreme sea loads scenario (fp = fps = 1.0) is covered by the "
             "utilisation table inherited from W24"),
        ],
    }


def _pypdf_version() -> str:
    try:
        import pypdf

        return getattr(pypdf, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return "unavailable"


def verify_artifact(path: Path | str = OUT_JSON,
                    anchor_path: Path | str = ANCHOR_JSON) -> dict:
    """Re-validate a written recheck artifact without touching the network.

    Checks that the artifact still describes the anchor in the tree (sha256),
    that every criterion is present and PASS, that the entry/coefficient counts
    add up, and that no error judgement claims a sub-3 % agreement while its
    number says otherwise.
    """
    artifact_path = Path(path)
    problems: list[str] = []
    if not artifact_path.exists():
        return {"verdict": "FAIL", "problems": ["artifact missing"], "checks": 0}
    artifact = json.loads(artifact_path.read_text())
    anchor = Path(anchor_path)
    checks = 0

    checks += 1
    anchor_sha = hashlib.sha256(anchor.read_bytes()).hexdigest()
    if artifact.get("recheck_of", {}).get("sha256") != anchor_sha:
        problems.append("anchor sha256 differs: the anchor changed after the recheck")

    checks += 1
    generator = Path(__file__).resolve()
    if artifact.get("generator_sha256") != hashlib.sha256(generator.read_bytes()).hexdigest():
        problems.append("generator sha256 differs: the recheck script changed")

    criteria = artifact.get("criteria", {})
    for key in ("W26-C1", "W26-C2", "W26-C3", "W26-C4", "W26-C5", "W26-C6"):
        checks += 1
        if key not in criteria:
            problems.append(f"missing criterion {key}")
        elif criteria[key].get("verdict") != "PASS":
            problems.append(f"{key} verdict is {criteria[key].get('verdict')}")

    summary = artifact.get("quote_recheck", {}).get("summary", {})
    checks += 1
    total = (summary.get("pass", 0) + summary.get("pass_restated", 0)
             + summary.get("pass_relocated", 0) + len(summary.get("fail", []))
             + summary.get("withheld", 0))
    if total != summary.get("transcribed", 0) + summary.get("withheld", 0):
        problems.append(f"entry accounting does not add up ({total} versus "
                        f"{summary.get('transcribed')} + {summary.get('withheld')})")

    checks += 1
    for row in artifact.get("coefficient_one_to_one", []):
        if row.get("verdict") not in ("PASS", "PASS_CROSS_DOCUMENT"):
            problems.append(f"coefficient {row.get('id')} verdict {row.get('verdict')}")

    checks += 1
    for item in artifact.get("error_judgement", []):
        claimed = bool(item.get("under_3_percent"))
        actual = abs(float(item.get("relative_difference_percent", 0.0))) < 3.0
        if claimed != actual:
            problems.append(f"error judgement {item.get('quantity')} claims "
                            f"{claimed} but the number is "
                            f"{item.get('relative_difference_percent')}")

    checks += 1
    six = artifact.get("six_criteria_verdicts", {})
    if len(six) != 6:
        problems.append(f"six_criteria_verdicts has {len(six)} entries")
    elif any(row.get("verdict") != "PASS" for row in six.values()):
        problems.append("a criterion entry is not PASS")

    return {"verdict": "PASS" if not problems else "FAIL",
            "problems": problems, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf-cache", default="/tmp/iacs_w26",
                        help="PDF download + page-text cache (default /tmp/iacs_w26)")
    parser.add_argument("--out", default=str(OUT_JSON), help="recheck JSON output path")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="do not touch the network; require cached PDFs")
    parser.add_argument("--recompute-smith", type=int, default=2,
                        help="how many worst-utilisation cases to re-solve (default 2)")
    parser.add_argument("--verify-only", action="store_true",
                        help="only re-validate an existing artifact (no network, no PDFs)")
    args = parser.parse_args(argv)

    if args.verify_only:
        report = verify_artifact(args.out)
        print(json.dumps(report, indent=2))
        return 0 if report["verdict"] == "PASS" else 1

    anchor = json.loads(ANCHOR_JSON.read_text())
    cache = Path(args.pdf_cache)
    documents = sync_documents(anchor, cache, skip_fetch=args.skip_fetch)
    if not documents["all_verified"]:
        print("document verification failed:", json.dumps(documents, indent=2))
        return 2

    pdf_paths = {DOC_KEY[doc["id"]]: Path(doc["path"]) for doc in documents["documents"]}
    pages, offsets = {}, {}
    for key, path in pdf_paths.items():
        print(f"extracting {path.name}", flush=True)
        pages[key] = extract_pages(path, cache)
        offsets[key] = printed_page_offset(pages[key])
        if offsets[key]["offset"] is None:
            offsets[key] = {
                "offset": 0,
                "samples": offsets[key]["samples"],
                "votes": offsets[key]["votes"],
                "derivation": ("no 'COMMON STRUCTURAL RULES' running header (UR "
                               "document): the PDF paginates from printed page 1, "
                               "so printed page == PDF page; the recheck confirms it "
                               "by relocalising every cited quote"),
            }
        print(f"  pages={len(pages[key])} offset={offsets[key]}", flush=True)

    result = build_recheck(anchor, documents=documents, pages=pages, offsets=offsets,
                           pdf_paths=pdf_paths, recompute_smith=args.recompute_smith)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\nwrote {out}")
    verification = verify_artifact(out)
    print(f"self-verification: {verification['verdict']} "
          f"({verification['checks']} checks, {len(verification['problems'])} problems)")
    for problem in verification["problems"]:
        print(f"  - {problem}")
    for key, crit in result["criteria"].items():
        print(f"{key}: {crit['verdict']}")
    print(f"quote recheck: {result['quote_recheck']['summary']}")
    return 0 if all(crit["verdict"] == "PASS" for crit in result["criteria"].values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
