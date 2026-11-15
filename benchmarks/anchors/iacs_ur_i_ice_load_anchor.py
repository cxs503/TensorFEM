"""Clause-level anchor transcription of the IACS Polar Class design ice load.

What this script does
---------------------
1. **Transcription gate.** The source documents live in
   ``benchmarks/anchors/sources/iacs/`` as the *unaltered* IACS PDFs.  Every
   anchor entry is produced by locating a (start-marker, end-marker) pair in
   the normalised text of one PDF page and copying out the exact substring in
   between.  No quote and no coefficient is typed from memory: the quotes are
   the extracted text, and the numbers used downstream are parsed back out of
   those quotes by regular expression.
2. **Hash binding.** Each anchor carries ``sha256_file`` (the whole PDF) and
   ``sha256_quote`` (the UTF-8 bytes of the quote).  Re-running the script
   recomputes both; ``--check`` fails if anything drifts.
3. **Formula implementation cross-check.** :func:`iacs_bow_design_load`
   implements the bow-area design load of clause I2.3.2.1 (iii).  Every
   constant in that function is re-parsed from the transcribed quotes and
   asserted equal to the Python literal, so a typo in the code cannot survive
   a run.
4. **Emits** ``benchmarks/anchors/iacs-ur-i-ice-load.json``.

Honesty contract
----------------
* ``I2.3.1``--``I2.3.5`` of UR I2 are transcribed.  ISO 19906 is **not** in
  this work tree and no licensed copy is reachable, so the ISO 19906 entries
  are emitted with ``status = "WITHHELD"`` and an explicit list of the
  clause/table/page that would be needed; no ISO 19906 number is invented
  here.  The ISO pressure--area coefficients used by TensorFEM remain declared
  engineering values and are labelled as such everywhere they appear.
* The IACS documents are public (iacs.org.uk, free download); the transcription
  records the exact URL and the download date so the hash can be reproduced.

Run
---
    python benchmarks/anchors/iacs_ur_i_ice_load_anchor.py
    python benchmarks/anchors/iacs_ur_i_ice_load_anchor.py --check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

ANCHOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANCHOR_DIR.parents[1]
SOURCE_DIR = ANCHOR_DIR / "sources" / "iacs"
OUT_PATH = ANCHOR_DIR / "iacs-ur-i-ice-load.json"

TASK_KEY = "ice-coupled-6dof"
DOC_I2 = "UR-I2Rev5.pdf"
DOC_I1 = "UR-I1Rev2-Corr1.pdf"

# URL + retrieval date are part of the provenance chain: the sha256 can only be
# reproduced if the same bytes are fetched from the same public location.
PROVENANCE = {
    DOC_I2: {
        "document": "IACS Unified Requirement I2, Polar Class ships - structural requirements",
        "url": ("https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/"
                "2025/06/11113711/UR-I2Rev5.pdf"),
        "revision_as_printed": "IACS Req. 2006/Rev.5 2025",
        "retrieved_utc": "2026-11-15",
        "publisher": "International Association of Classification Societies (IACS)",
        "access": "public, free download",
    },
    DOC_I1: {
        "document": "IACS Unified Requirement I1, Polar Class ships - Polar Class descriptions",
        "url": ("https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/"
                "2022/02/12082733/ur-i1rev2-1.pdf"),
        "revision_as_printed": "IACS Req. 2007/Rev.2 2022 (Corr.1)",
        "retrieved_utc": "2026-11-15",
        "publisher": "International Association of Classification Societies (IACS)",
        "access": "public, free download",
    },
}

# ---------------------------------------------------------------------------
# anchor definitions: (id, document, clause, start marker, end marker)
# The markers are short literal fragments of the extracted page text; the quote
# is everything from the start marker through the end marker.
# ---------------------------------------------------------------------------
ANCHORS = [
    dict(
        anchor_id="IACS-UR-I2-I2.3.1(ii)",
        document=DOC_I2,
        clause="I2.3.1 (ii)",
        carries="definition of the design ice load: average pressure over a rectangular patch b x w",
        start="(ii) The design ice load is characterized",
        end="height (b) and width (w).",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.1(iv)",
        document=DOC_I2,
        clause="I2.3.1 (iv)",
        carries="fixed load patch aspect ratio for non-bow areas (AR = 3.6)",
        start="(iv) In other ice-strengthened areas",
        end="AR = 3.6.",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.2.1(iii)(a)",
        document=DOC_I2,
        clause="I2.3.2.1 (iii) (a)",
        carries="bow shape coefficient fai = min(fai1, fai2, fai3) and its constants",
        start="(a) Shape coefficient, fai, is to be taken as",
        end="fai,3 = 0.60",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.2.1(iii)(b)",
        document=DOC_I2,
        clause="I2.3.2.1 (iii) (b)",
        carries="bow glancing impact force Fi = fai CFC DUI^0.64 [MN] (displacement exponent 0.64)",
        start="(b) Force, Fi:",
        end="[MN]",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.2.1(iii)(c)",
        document=DOC_I2,
        clause="I2.3.2.1 (iii) (c)",
        carries="load patch aspect ratio ARi = 7.46 sin(beta'i) >= 1.3",
        start="(c) Load patch aspect ratio, ARi:",
        end=">= 1.3",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.2.1(iii)(d)",
        document=DOC_I2,
        clause="I2.3.2.1 (iii) (d)",
        carries="line load Qi = Fi^0.61 CFD / ARi^0.35 [MN/m]",
        start="(d) Line load, Qi:",
        end="[MN/m]",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.2.1(iii)(e)",
        document=DOC_I2,
        clause="I2.3.2.1 (iii) (e)",
        carries="design pressure Pi = Fi^0.22 CFD^2 ARi^0.3 [MPa] (pressure-area trend)",
        start="(e) Pressure, Pi:",
        end="[MPa]",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.2.2(i)(a)",
        document=DOC_I2,
        clause="I2.3.2.2 (i) (a)",
        carries="non-bow force FNonBow = 0.36 CFC DF [MN]",
        start="(a) Force, FNonBow:",
        end="0.36 · CFC · DF [MN]",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.2.2(i)(b)",
        document=DOC_I2,
        clause="I2.3.2.2 (i) (b)",
        carries="non-bow line load QNonBow = 0.639 FNonBow^0.61 CFD [MN/m]",
        start="(b) Line Load, QNonBow:",
        end="[MN/m]",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.2.2(i)-DF",
        document=DOC_I2,
        clause="I2.3.2.2 (i), definition of DF",
        carries="ship displacement factor DF = DUI^0.64 below CFDIS, linear above",
        start="DF = ship displacement factor",
        end="CFDIS0.64 + 0.10 · (DUI - CFDIS)",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.3(i)",
        document=DOC_I2,
        clause="I2.3.3 (i)",
        carries="bow design load patch: wBow = FBow/QBow, bBow = QBow/PBow",
        start="wBow = FBow / QBow [m]",
        end="bBow = QBow / PBow [m]",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.3(ii)",
        document=DOC_I2,
        clause="I2.3.3 (ii)",
        carries="non-bow design load patch: wNonBow = FNonBow/QNonBow, bNonBow = wNonBow/3.6",
        start="wNonBow = FNonBow / QNonBow [m]",
        end="bNonBow = wNonBow / 3.6 [m]",
    ),
    dict(
        anchor_id="IACS-UR-I2-I2.3.4(i)",
        document=DOC_I2,
        clause="I2.3.4 (i)",
        carries="average pressure over the design patch Pavg = F/(b w) [MPa]",
        start="Pavg = F / (b · w) [MPa]",
        end="appropriate for the hull area under consideration [m]",
    ),
    dict(
        anchor_id="IACS-UR-I2-Table3",
        document=DOC_I2,
        clause="I2.3.4 (ii), Table 3",
        carries="peak pressure factors PPFp, PPFt, PPFs (local pressure concentration)",
        start="Table 3 - Peak Pressure Factors",
        end="PPFs = 2.0 - 2.0 · Sw / w, if Sw < (0.5 · w)",
    ),
    dict(
        anchor_id="IACS-UR-I2-Table1",
        document=DOC_I2,
        clause="I2.3.2.1 (iii), Table 1",
        carries="class factors CFC, CFF, CFD, CFDIS, CFL for PC1..PC7",
        start="Table 1 - Class factors to be used in I2.3.2.1 (iii)",
        end="PC7 1.80 4.06 1.11 22 1.81",
    ),
    dict(
        anchor_id="IACS-UR-I2-Table2",
        document=DOC_I2,
        clause="I2.3.2.1 (iv), Table 2",
        carries="class factors CFCV, CFQV, CFPV for vertical-sided bows (PC6, PC7)",
        start="Table 2 - Class factors to be used in I2.3.2.1 (iv)",
        end="PC7 2.60 2.33 0.65",
    ),
    dict(
        anchor_id="IACS-UR-I2-Table4",
        document=DOC_I2,
        clause="I2.3.5, Table 4",
        carries="hull area factors AF per hull area and Polar Class",
        start="Table 4 - Hull Area Factors (AF)",
        end="Note to Table 4:",
    ),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def normalise(text: str) -> str:
    """Collapse all whitespace and fold the special glyphs to ASCII."""
    t = (text.replace("\u2032", "'").replace("\u2033", '"')
             .replace("\u2265", ">=").replace("\u2013", "-").replace("\u2014", "-")
             .replace("\u201c", '"').replace("\u201d", '"'))
    return re.sub(r"\s+", " ", t).strip()


def page_texts(path: Path) -> list:
    """Normalised text of every PDF page (index 0 == printed page 1)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:  # pragma: no cover
        from pypdf import PdfReader
        return [normalise(p.extract_text() or "") for p in PdfReader(str(path)).pages]
    doc = fitz.open(str(path))
    return [normalise(p.get_text()) for p in doc]


def locate(pages: list, start: str, end: str) -> dict:
    """Find ``start`` .. ``end`` on one page; return quote + locator."""
    hits = []
    for idx, text in enumerate(pages):
        i = text.find(start)
        if i < 0:
            continue
        j = text.find(end, i)
        if j < 0:
            continue
        hits.append((idx, i, j + len(end), text[i:j + len(end)]))
    if not hits:
        return {"status": "LOCATE_FAILED"}
    # keep the shortest match; report the multiplicity so ambiguity is visible
    hits.sort(key=lambda h: len(h[3]))
    status = "LOCATED" if len(hits) == 1 else "LOCATED_AMBIGUOUS"
    idx, i, j, quote = hits[0]
    printed = re.search(r"Page \d+ of \d+", pages[idx])
    return {
        "status": status,
        "pdf_page": idx + 1,
        "printed_page": printed.group(0) if printed else None,
        "candidate_pages": sorted({h[0] + 1 for h in hits}),
        "quote": quote,
        "sha256_quote": sha256_bytes(quote.encode("utf-8")),
    }


# ---------------------------------------------------------------------------
# numbers parsed back out of the quotes (never typed)
# ---------------------------------------------------------------------------
def _num(pattern: str, text: str, label: str) -> float:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"cannot parse {label!r} from quote: {text[:200]!r}")
    return float(m.group(1))


def parse_class_factors(quote: str) -> dict:
    """Parse the PC1..PC7 rows of Table 1 (CFC, CFF, CFD, CFDIS, CFL)."""
    out = {}
    for pc in range(1, 8):
        m = re.search(rf"PC{pc} ((?:\d+(?:\.\d+)? ){{4}}\d+(?:\.\d+)?)", quote)
        if not m:
            continue
        vals = [float(v) for v in m.group(1).split()]
        out[f"PC{pc}"] = dict(CFC=vals[0], CFF=vals[1], CFD=vals[2],
                              CFDIS=vals[3], CFL=vals[4])
    return out


def parse_hull_area_factors(quote: str) -> dict:
    """Parse the hull-area rows of Table 4 into {row label: [PC1..PC7]}.

    Table 4 uses "1.00*" (footnote) and "**" (strengthening not required) as
    cells, so a cell is recorded as a float when it parses and as the raw token
    otherwise -- never silently dropped.
    """
    markers = {
        "Bow (B)": "Bow (B) All B",
        "Bow Intermediate (BI) Icebelt": "Bow Intermediate (BI) Icebelt BIi",
        "Bow Intermediate (BI) Lower": "Lower BIl",
        "Bow Intermediate (BI) Bottom": "Bottom BIb",
        "Midbody (M) Icebelt": "Midbody (M) Icebelt Mi",
        "Midbody (M) Lower": "Lower Ml",
        "Midbody (M) Bottom": "Bottom Mb",
        "Stern (S) Icebelt": "Stern (S) Icebelt Si",
        "Stern (S) Lower": "Lower Sl",
        "Stern (S) Bottom": "Bottom Sb",
    }
    rows = {}
    for label, marker in markers.items():
        i = quote.find(marker)
        if i < 0:
            continue
        tokens = quote[i + len(marker):].split()[:7]
        if len(tokens) < 7:
            continue
        cells = []
        for tok in tokens:
            try:
                cells.append(float(tok))
            except ValueError:
                cells.append(tok)
        rows[label] = cells
    return rows


def parse_code_constants(quotes: dict) -> dict:
    """Re-parse every constant of the bow design-load chain from the quotes."""
    c = {}
    q_a = quotes["IACS-UR-I2-I2.3.2.1(iii)(a)"]
    q_b = quotes["IACS-UR-I2-I2.3.2.1(iii)(b)"]
    q_c = quotes["IACS-UR-I2-I2.3.2.1(iii)(c)"]
    q_d = quotes["IACS-UR-I2-I2.3.2.1(iii)(d)"]
    q_e = quotes["IACS-UR-I2-I2.3.2.1(iii)(e)"]
    q_nb_f = quotes["IACS-UR-I2-I2.3.2.2(i)(a)"]
    q_nb_q = quotes["IACS-UR-I2-I2.3.2.2(i)(b)"]
    q_nb_ar = quotes["IACS-UR-I2-I2.3.3(ii)"]
    q_ar_fixed = quotes["IACS-UR-I2-I2.3.1(iv)"]
    q_t3 = quotes["IACS-UR-I2-Table3"]
    c["fai1_scale"] = _num(r"\((0\.097) - 0\.68", q_a, "0.097")
    c["fai1_x_coeff"] = _num(r"\(0\.097 - (0\.68)", q_a, "0.68")
    c["fai1_offset"] = _num(r"x/ LUI - (0\.15)\)2", q_a, "0.15")
    c["fai1_beta_exponent"] = _num(r"\(β'i\)(0\.\d+)", q_a, "0.5")
    c["fai2_scale"] = _num(r"fai,2 = (1\.2)", q_a, "1.2")
    c["fai3_cap"] = _num(r"fai,3 = (0\.60)", q_a, "0.60")
    c["displacement_exponent"] = _num(r"DUI(0\.\d+) \[MN\]", q_b, "DUI exponent")
    c["ar_scale"] = _num(r"ARi = (7\.46)", q_c, "7.46")
    c["ar_min"] = _num(r">= (1\.3)", q_c, "1.3")
    c["line_load_force_exponent"] = _num(r"Qi = Fi ?(0\.\d+)", q_d, "Fi exponent")
    c["line_load_ar_exponent"] = _num(r"/ ARi(0\.\d+)", q_d, "ARi exponent")
    c["pressure_force_exponent"] = _num(r"Pi = Fi(0\.\d+)", q_e, "Fi exponent")
    c["pressure_cfd_exponent"] = _num(r"CFD(\d+) ·", q_e, "CFD exponent")
    c["pressure_ar_exponent"] = _num(r"ARi(0\.\d+)", q_e, "ARi exponent")
    c["nonbow_force_scale"] = _num(r"FNonBow = (0\.36)", q_nb_f, "0.36")
    c["nonbow_line_load_scale"] = _num(r"QNonBow = (0\.639)", q_nb_q, "0.639")
    c["nonbow_line_load_exponent"] = _num(r"FNonBow(0\.\d+)", q_nb_q, "FNonBow exponent")
    c["nonbow_aspect_ratio"] = _num(r"bNonBow = wNonBow / (3\.6)", q_nb_ar, "3.6")
    c["nonbow_AR_fixed"] = _num(r"AR = (3\.6)", q_ar_fixed, "3.6")
    c["ppf_plate_transverse_offset"] = _num(r"PPFp = \((1\.8) - s\) >= 1\.2", q_t3, "1.8")
    c["ppf_plate_transverse_floor"] = _num(r"PPFp = \(1\.8 - s\) >= (1\.2)", q_t3, "1.2")
    c["ppf_plate_longitudinal_offset"] = _num(r"PPFp = \((2\.2) - 1\.2", q_t3, "2.2")
    c["ppf_plate_longitudinal_slope"] = _num(r"PPFp = \(2\.2 - (1\.2)", q_t3, "1.2")
    c["ppf_plate_longitudinal_floor"] = _num(r"PPFp = \(2\.2 - 1\.2 · s\) >= (1\.5)", q_t3, "1.5")
    return c


# ---------------------------------------------------------------------------
# the transcribed design-load chain (bow area, I2.3.2.1 (iii))
# ---------------------------------------------------------------------------
CODE_CONSTANTS = {
    "fai1_scale": 0.097,
    "fai1_offset": 0.15,
    "fai1_x_coeff": 0.68,
    "fai1_beta_exponent": 0.5,
    "fai2_scale": 1.2,
    "fai3_cap": 0.60,
    "displacement_exponent": 0.64,
    "ar_scale": 7.46,
    "ar_min": 1.3,
    "line_load_force_exponent": 0.61,
    "line_load_ar_exponent": 0.35,
    "pressure_force_exponent": 0.22,
    "pressure_cfd_exponent": 2.0,
    "pressure_ar_exponent": 0.3,
}


def iacs_bow_design_load(*, displacement_kt: float, class_factors: dict,
                         sub_regions: list, pc: str = "PC6") -> dict:
    """IACS UR I2 bow-area design ice load for one Polar Class.

    Parameters
    ----------
    displacement_kt : ship displacement ``DUI`` in kt (>= 5 for the bow area).
    class_factors : the Table 1 row of the Polar Class (``CFC``, ``CFF``,
        ``CFD``, ``CFDIS``, ``CFL``).
    sub_regions : list of dicts with the bow geometry at the mid-length of each
        sub-region: ``x_over_L`` (x/LUI), ``alpha_deg`` (waterline angle),
        ``beta_prime_deg`` (normal frame angle).  Clause I2.3.2.1 (ii) divides
        the bow waterline length into four equal sub-regions.
    pc : label recorded in the result.

    Returns
    -------
    dict with the per-sub-region Fi/Qi/Pi/ARi/fai, the governing (maximum
    force) sub-region, the design patch ``bBow``, ``wBow`` and ``Pavg``.
    """
    c = CODE_CONSTANTS
    dui = max(float(displacement_kt), 5.0)
    cfc = float(class_factors["CFC"])
    cff = float(class_factors["CFF"])
    cfd = float(class_factors["CFD"])
    rows = []
    for sr in sub_regions:
        x = float(sr["x_over_L"])
        alpha = float(sr["alpha_deg"])
        beta = float(sr["beta_prime_deg"])
        fai1 = ((c["fai1_scale"] - c["fai1_x_coeff"] * (x - c["fai1_offset"]) ** 2)
                * alpha / beta ** c["fai1_beta_exponent"])
        fai2 = (c["fai2_scale"] * cff
                / (math.sin(math.radians(beta)) * cfc * dui ** c["displacement_exponent"]))
        fai = min(fai1, fai2, c["fai3_cap"])
        force = fai * cfc * dui ** c["displacement_exponent"]          # [MN]
        ar = max(c["ar_scale"] * math.sin(math.radians(beta)), c["ar_min"])
        line = force ** c["line_load_force_exponent"] * cfd / ar ** c["line_load_ar_exponent"]
        press = (force ** c["pressure_force_exponent"] * cfd ** c["pressure_cfd_exponent"]
                 * ar ** c["pressure_ar_exponent"])                    # [MPa]
        rows.append(dict(sub_region=len(rows) + 1, x_over_L=x, alpha_deg=alpha,
                         beta_prime_deg=beta, fai=fai, fai1=fai1, fai2=fai2,
                         F_MN=force, AR=ar, Q_MNm=line, P_MPa=press,
                         patch_width_m=force / line, patch_height_m=line / press,
                         patch_area_m2=(force / line) * (line / press)))
    gov = max(rows, key=lambda r: r["F_MN"])
    force, line, press = gov["F_MN"], gov["Q_MNm"], gov["P_MPa"]
    w, b = force / line, line / press
    return {
        "polar_class": pc,
        "displacement_kt": dui,
        "class_factors": dict(class_factors),
        "sub_regions": rows,
        "governing_sub_region": gov["sub_region"],
        "design_force_MN": force,
        "design_line_load_MNm": line,
        "design_pressure_MPa": press,
        "patch_width_m": w,
        "patch_height_m": b,
        "patch_area_m2": w * b,
        "Pavg_MPa": force / (b * w),
        "formula": "I2.3.2.1 (iii)(a)-(e) + I2.3.3 (i) + I2.3.4 (i)",
    }


def code_pressure_area_exponent() -> dict:
    """Pressure--area exponent *implied by* the IACS chain itself.

    Along the code's own family of bow design solutions the free variable is
    the shape coefficient ``fai`` (it depends on hull angles)::

        F  = fai CFC DUI^0.64
        AR = 7.46 sin(beta')                 (independent of fai)
        Q  = F^0.61 CFD / AR^0.35
        P  = F^0.22 CFD^2 AR^0.3
        A  = b w = F / P = F^(1-0.22) / (CFD^2 AR^0.3)

    so ``dlnP/dlnA = 0.22/(1-0.22) = +0.282`` at fixed AR and CFD.  This is an
    algebraic property of the transcribed clauses, not a fit, and its sign is
    opposite to the ISO 19906 global pressure--area law (``ex < 0``).
    """
    c = CODE_CONSTANTS
    e_p = c["pressure_force_exponent"]
    area_force_exponent = 1.0 - e_p
    return {
        "pressure_force_exponent": e_p,
        "area_force_exponent": area_force_exponent,
        "dlnp_dlnA": e_p / area_force_exponent,
        "caveat": ("at fixed AR and CFD; AR varies with the normal frame angle and CFD "
                   "with the Polar Class, so the full trend in a design space is not a "
                   "single power law"),
    }


# ---------------------------------------------------------------------------
# ISO 19906 gap record (no numbers are invented here)
# ---------------------------------------------------------------------------
ISO_ENTRIES = [
    {
        "anchor_id": "ISO-19906-global-pressure-area",
        "document": ("ISO 19906:2019 Petroleum and natural gas industries - Arctic "
                     "offshore structures"),
        "needed": ("Annex A, global ice pressure clause: the coefficients C_R and the "
                   "area-law exponent ex together with the A_0 normalisation and the "
                   "validity range in A"),
        "status": "WITHHELD",
        "reason": ("no licensed copy of ISO 19906 is reachable from this work tree and "
                   "the standard is not public; writing a recalled coefficient would be "
                   "fabrication"),
        "consequence": ("the ISO 19906 comparison is run at TensorFEM's *declared* "
                        "coefficients (C_R = 1.0 MPa, A_0 = 1 m2, ex = -0.28) and every "
                        "ISO-19906-derived number is tagged non_normative; the exponent is "
                        "swept so that no conclusion rests on that one value"),
    },
    {
        "anchor_id": "ISO-19906-local-pressure",
        "document": "ISO 19906:2019 Annex A, local ice pressure clause/table",
        "needed": ("local (high-pressure-zone) pressure--area parameters and the local area "
                   "definition"),
        "status": "WITHHELD",
        "reason": "same as above: document not reachable, no transcription possible",
        "consequence": "local pressure is not compared; only the global law family is used",
    },
    {
        "anchor_id": "ISO-19906-shape-factor",
        "document": "ISO 19906:2019 Annex A, pressure-area shape/correction factor clause",
        "needed": ("the shape factor clause that corrects pressure for non-circular or "
                   "non-square contact areas"),
        "status": "WITHHELD",
        "reason": "same as above: document not reachable",
        "consequence": ("the floe contact areas of this ensemble are rectangular (h x w), so a "
                        "shape factor of order one is the expected correction, but it cannot "
                        "be quantified from an anchored clause here"),
    },
]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build() -> dict:
    hashes = {}
    texts = {}
    for name in (DOC_I2, DOC_I1):
        path = SOURCE_DIR / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing; the IACS PDFs must be present for the transcription gate")
        hashes[name] = sha256_file(path)
        texts[name] = page_texts(path)

    anchors = []
    for spec in ANCHORS:
        loc = locate(texts[spec["document"]], spec["start"], spec["end"])
        entry = {k: v for k, v in spec.items() if k not in ("start", "end")}
        entry["sha256_file"] = hashes[spec["document"]]
        entry["document_url"] = PROVENANCE[spec["document"]]["url"]
        entry["retrieved_utc"] = PROVENANCE[spec["document"]]["retrieved_utc"]
        entry["document_revision"] = PROVENANCE[spec["document"]]["revision_as_printed"]
        entry.update(loc)
        entry["claim_eligible"] = loc["status"] in ("LOCATED", "LOCATED_AMBIGUOUS")
        anchors.append(entry)

    quotes = {a["anchor_id"]: a.get("quote", "") for a in anchors}
    located = all(a["claim_eligible"] for a in anchors)
    parsed = {}
    consistency = {}
    if located:
        parsed["bow_design_load_constants"] = parse_code_constants(quotes)
        parsed["class_factors_table1"] = parse_class_factors(quotes["IACS-UR-I2-Table1"])
        parsed["hull_area_factors_table4"] = parse_hull_area_factors(
            quotes["IACS-UR-I2-Table4"])
        for key, value in parsed["bow_design_load_constants"].items():
            if key in CODE_CONSTANTS:
                consistency[key] = {
                    "code_literal": CODE_CONSTANTS[key],
                    "transcribed": value,
                    "match": abs(CODE_CONSTANTS[key] - value) < 1e-12,
                }

    verdict = "PASS" if (located and consistency
                         and all(v["match"] for v in consistency.values())) else "FAIL"
    payload = {
        "schema_version": "1.0.0",
        "task_key": TASK_KEY,
        "topic": "iacs-ur-i-polar-class-design-ice-load",
        "experiment": ("clause-level transcription of the IACS Polar Class design ice load with "
                       "sha256 binding, plus the ISO 19906 gap record"),
        "repository": "TensorFEM",
        "branch": _branch(),
        "git_head": git_commit(),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": {
            "script": "benchmarks/anchors/iacs_ur_i_ice_load_anchor.py",
            "command": "python benchmarks/anchors/iacs_ur_i_ice_load_anchor.py",
            "python": platform.python_version(),
        },
        "sources": [
            dict(file=name, **PROVENANCE[name], sha256_file=hashes[name],
                 pages=len(texts[name]))
            for name in (DOC_I2, DOC_I1)
        ],
        "anchor_count": len(anchors),
        "anchors": anchors,
        "parsed_from_quotes": parsed,
        "code_vs_transcription_consistency": consistency,
        "all_anchor_values_from_transcription": bool(
            consistency) and all(v["match"] for v in consistency.values()),
        "code_pressure_area_exponent": code_pressure_area_exponent(),
        "iso_19906": {
            "status": "WITHHELD",
            "entries": ISO_ENTRIES,
            "declared_values_used_downstream": {
                "C_R_MPa": 1.0, "A_0_m2": 1.0, "ex": -0.28,
                "grade": "DECLARED_REFERENCE_NOT_NORMATIVE",
                "source": ("tensorfem.iso19906_ice_6dof.ISO19906PressureArea defaults; declared "
                           "engineering values of this study, not a transcription"),
            },
        },
        "verdict": verdict,
        "limitations": [
            ("IACS UR I2 is transcribed at Rev.5 (June 2025); another revision changes the class "
             "factors and would move every comparison below"),
            ("only the bow-area chain (I2.3.2.1 (iii)) is evaluated; the vertical-side PC6/PC7 "
             "chain (I2.3.2.1 (iv)) and the non-bow chain (I2.3.2.2) are transcribed but not "
             "evaluated here"),
            ("ISO 19906 stays WITHHELD: the ISO-side comparison is against declared coefficients "
             "and cannot claim compliance with the standard"),
            ("IACS design loads are scantling loads for a prescribed glancing bow impact, not a "
             "time-domain simulation output, so agreement/disagreement with a 6-DOF ensemble is a "
             "formulation-scope comparison, not a validation of either side"),
        ],
    }
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="re-extract and fail if the stored artefact differs")
    args = ap.parse_args(argv)

    payload = build()
    status = payload["verdict"]
    for a in payload["anchors"]:
        if not a["claim_eligible"]:
            print(f"  [MISS] {a['anchor_id']}: {a['status']}", file=sys.stderr)
    print(f"anchors: {payload['anchor_count']}  verdict: {status}")
    print(f"table1 classes: {sorted(payload['parsed_from_quotes'].get('class_factors_table1', {}))}")
    print(f"iso 19906: {payload['iso_19906']['status']} ({len(ISO_ENTRIES)} entries withheld)")
    if not payload["all_anchor_values_from_transcription"]:
        print("  [FAIL] code constants do not match the transcribed text", file=sys.stderr)

    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not OUT_PATH.is_file():
            print(f"  [FAIL] {OUT_PATH} missing", file=sys.stderr)
            return 1
        stored = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        fresh = json.loads(text)
        stored.pop("generated_at_utc", None)
        fresh.pop("generated_at_utc", None)
        if stored != fresh:
            print("  [FAIL] stored artefact differs from a fresh transcription", file=sys.stderr)
            return 1
        print("  [ok] stored artefact reproduces exactly (timestamps excluded)")
        return 0 if status == "PASS" else 1

    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())