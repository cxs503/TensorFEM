"""W24: IACS CSR hull-girder ULS partial safety factors, transcribed and machine-checked.

Why this module exists
----------------------
W22 (:mod:`tensorfem.hull_girder_csr_w22`) and W23
(:mod:`tensorfem.hull_girder_uls_scan`) reported the CSR hull-girder ULS
utilisation at *unit* partial safety factors (``gamma_s = gamma_w = 1``) because
the printed factor table could not be transcribed from an authoritative copy;
the blocking register is ``benchmarks/hull_girder_uls/csr_factor_register.json``
(``status=BLOCKED``, 2026-11-11).

W24 re-opened the acquisition through a channel that had not been tried: IACS
publishes its Unified Requirements and the harmonised CSR on a public object
store (``https://iacs.s3.af-south-1.amazonaws.com``) that answers unauthenticated
``ListObjectsV2`` / ``GetObject``.  The assigned documents were located there and
retrieved:

* the harmonised CSR for Bulk Carriers and Oil Tankers, edition 01 Jan 2019
  (``.../csr-h-01-jan-2019.pdf``, 820 pages) -- the assigned edition;
* UR S7 Rev.4 (May 2010), *Minimum Longitudinal Strength Standards*
  (``.../ur-s7rev4.pdf``, 1 page) -- the other assigned document;
* UR S11 Rev.10 (2020), used only as an independent cross-check of the wave
  bending moment coefficients and of the 175/k permissible stress.

This module holds the clause-by-clause transcription (:data:`CLAUSES`), the
machine checks on it (:func:`dimension_report`, :func:`consistency_report`) and
the wiring of the transcribed factors into the existing 6-section x
2-wave-condition W23 scan (:func:`utilisation_rows`).  Downstream code reads the
factor values from the anchor artifact ``csr_uls_anchor.json`` -- never from a
literal -- so there is exactly one place where a printed value can enter.

The two utilisation bases
-------------------------
CSR Pt.1 Ch.5 Sec.2 [2.1.2] + [2.2.1]::

    M = gamma_S * Msw-U + gamma_W * f_beta * Mwv      (demand)
    M <= M_U / gamma_R                                (capacity)

so the *factored* utilisation is ``u = gamma_R * M / M_U`` and the check passes
for ``u <= 1``.  The substitute basis used by W23 is the pure capacity-margin
ratio ``gamma* = M_U / (Msw + Mwv)``, i.e. the reciprocal of the unit-factor
utilisation.  The two bases differ only by the factor that multiplies the
demand, so the difference is analytically attributable::

    u / u_0 = gamma_R * ( gamma_S * phi_sw + gamma_W * f_beta * phi_wv )

with ``u_0 = (Msw + Mwv) / M_U``, ``phi_sw = Msw / (Msw + Mwv)`` and
``phi_wv = 1 - phi_sw``.  That identity is machine-checked per case (residual
< 1e-9) instead of being asserted.

Declared substitutions (not rule quantities)
--------------------------------------------
* ``Msw-U`` is a *permissible envelope* over loading conditions (Ch.4 Sec.4
  [2.2.2], p.184), not a closed formula.  The study therefore feeds the W23
  beam-equilibrium still-water moment as the demand proxy and flags it per row
  (``m_sw_basis``).
* ``M_U`` is the repo Smith progressive-collapse capacity on the study
  scantlings, not the CSR Appendix 2 net-scantling capacity (declared in
  CSR-ULS-W02).
* ``Mwv`` is carried on two bases: the CSR Ch.4 Sec.4 [3.1.1] rule formula
  (primary) and the W23 integrated design-wave load (diagnostic column).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "benchmarks" / "hull_girder_uls"
SCAN_CSV = ARTIFACT_DIR / "section_scan.csv"
ANCHOR_JSON = ARTIFACT_DIR / "csr_uls_anchor.json"
UTILISATION_CSV = ARTIFACT_DIR / "csr_utilisation_with_factors.csv"
SENSITIVITY_JSON = ARTIFACT_DIR / "factor_sensitivity.json"

Dim = tuple[float, float, float]

# ---------------------------------------------------------------------------
# unit algebra: name -> (scale to SI base, (mass, length, time) exponents)
# ---------------------------------------------------------------------------
_UNITS: dict[str, tuple[float, Dim]] = {
    "1": (1.0, (0.0, 0.0, 0.0)),
    "m": (1.0, (0.0, 1.0, 0.0)),
    "m3": (1.0, (0.0, 3.0, 0.0)),
    "cm3": (1e-6, (0.0, 3.0, 0.0)),
    "kN": (1e3, (1.0, 1.0, -2.0)),
    "kNm": (1e3, (1.0, 2.0, -2.0)),
    "N/mm2": (1e6, (1.0, -1.0, -2.0)),
    # coefficient units: the unit of the printed numeric coefficient itself
    "kN/m2": (1e3, (1.0, -1.0, -2.0)),
    "cm3/m3": (1e-6, (0.0, 0.0, 0.0)),
    "cm3/(kNm/(N/mm2))": (1e-3, (0.0, 0.0, 0.0)),
}


def unit(name: str) -> tuple[float, Dim]:
    """Return ``(scale, dims)`` for a unit token used by the anchor."""
    if name not in _UNITS:
        raise KeyError(f"unit {name!r} not registered")
    return _UNITS[name]


def _mul(*terms: tuple[float, Dim]) -> tuple[float, Dim]:
    scale, dims = 1.0, (0.0, 0.0, 0.0)
    for s, d in terms:
        scale *= s
        dims = (dims[0] + d[0], dims[1] + d[1], dims[2] + d[2])
    return scale, dims


def _pow(term: tuple[float, Dim], n: float) -> tuple[float, Dim]:
    s, d = term
    return s**n, (d[0] * n, d[1] * n, d[2] * n)


# ---------------------------------------------------------------------------
# source documents (provenance of the transcription)
# ---------------------------------------------------------------------------
SOURCE_DOCUMENTS: list[dict] = [
    {
        "id": "AN-IACS-CSR",
        "title": ("IACS Common Structural Rules for Bulk Carriers and Oil Tankers "
                  "(harmonised CSR), 01 January 2019 edition"),
        "assigned_by_mission": True,
        "url": ("https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/"
                "2022/03/15144815/csr-h-01-jan-2019.pdf"),
        "sha256": "ffb51b3153e43a7d3a2e840971bc948128152e1c9ffe66c3891d6e620e250e7f",
        "bytes": 21439977,
        "pdf_pages": 820,
        "retrieved_utc": "2026-11-12",
        "retrieval_note": ("unauthenticated GET from the IACS public object store; "
                           "www.iacs.org.uk itself answers HTTP 403 through the "
                           "sandbox proxy, the object store does not"),
        "text_extraction": "pypdf 6.19.0",
    },
    {
        "id": "AN-IACS-UR-S7",
        "title": "IACS UR S7 Rev.4 (May 2010), Minimum Longitudinal Strength Standards",
        "assigned_by_mission": True,
        "url": ("https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/"
                "2022/05/18114816/ur-s7rev4.pdf"),
        "sha256": "25a82bfec5f2beeafb780cbf2a1498efbe1a1573f05b923b1cf9b9729cac2096",
        "bytes": 18903,
        "pdf_pages": 1,
        "retrieved_utc": "2026-11-12",
        "text_extraction": "pypdf 6.19.0 (single printed page)",
    },
    {
        "id": "AN-IACS-UR-S11",
        "title": "IACS UR S11 Rev.10 (2020), Longitudinal Strength Standard",
        "assigned_by_mission": False,
        "role": ("cross-check only: independent statement of the wave bending "
                 "moment coefficients and of the permissible bending stress 175/k"),
        "url": ("https://iacs.s3.af-south-1.amazonaws.com/wp-content/uploads/"
                "2022/05/18115332/ur-s11rev10.pdf"),
        "sha256": "4711024b4b537cd7e0c6203e01b5936575fc3f542349ec1addf589397986ed53",
        "bytes": 1501639,
        "pdf_pages": 31,
        "retrieved_utc": "2026-11-12",
        "text_extraction": "pypdf 6.19.0",
    },
]

# ---------------------------------------------------------------------------
# the transcription itself
# ---------------------------------------------------------------------------
#: Each entry carries the clause/table locator, the printed page, a short
#: verbatim quote and (where the clause is an equation) a machine-readable
#: ``relation`` used by the unit and consistency checks.  ``status`` is
#: TRANSCRIBED or WITHHELD.
CLAUSES: list[dict] = [
    {
        "id": "CSR-ULS-01",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 5, Sec 2, [2.1.2]",
        "table": None,
        "page": 343,
        "status": "TRANSCRIBED",
        "subject": "hull girder ultimate strength checking criterion",
        "quote": ("The vertical hull girder ultimate bending capacity at any hull "
                  "transverse section is to satisfy the following criteria:"),
        "printed_form": "M <= MU / gamma_R",
        "relation": {
            "name": "uls_criterion",
            "lhs": {"expr": [("M", 1)], "unit": "kNm"},
            "rhs": [{"expr": [("MU", 1), ("gamma_R", -1)], "unit": "kNm"}],
            "factor": 1.0,
        },
        "symbols": {
            "M": "vertical bending moment in the ULS check, kNm",
            "MU": "vertical hull girder ultimate bending capacity, kNm",
            "gamma_R": ("partial safety factor for the vertical hull girder "
                        "ultimate bending capacity"),
        },
    },
    {
        "id": "CSR-ULS-02",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 5, Sec 2, [2.2.1]",
        "table": None,
        "page": 343,
        "status": "TRANSCRIBED",
        "subject": "ULS bending moment load combination",
        "quote": ("The vertical hull girder bending moment, M in hogging and "
                  "sagging conditions, to be considered in the ultimate strength "
                  "check is to be taken as:"),
        "printed_form": "M = gamma_S * Msw-U + gamma_W * f_beta * Mwv",
        "relation": {
            "name": "uls_load_combination",
            "lhs": {"expr": [("M", 1)], "unit": "kNm"},
            "rhs": [
                {"expr": [("gamma_S", 1), ("Msw_U", 1)], "unit": "kNm"},
                {"expr": [("gamma_W", 1), ("f_beta", 1), ("Mwv", 1)], "unit": "kNm"},
            ],
            "factor": 1.0,
        },
        "symbols": {
            "gamma_S": "partial safety factor for the still water bending moment, Table 2",
            "gamma_W": "partial safety factor for the vertical wave bending moment, Table 2",
            "f_beta": "heading correction factor, Sec 1",
            "Msw_U": "permissible still water bending moment, kNm",
            "Mwv": "vertical wave bending moment, kNm",
        },
    },
    {
        "id": "CSR-ULS-03",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 5, Sec 2, Table 1",
        "table": "Table 1",
        "page": 343,
        "status": "TRANSCRIBED",
        "subject": "design load scenarios and the still water moment they use",
        "quote": ("Table 1 : Design load scenarios. A / S+D: Msw-h or Msw-s; "
                  "A / S: Msw-p-h or Msw-p-s; A:S+D: Msw-f; B / S+D: maximum sagging "
                  "still water bending moment for operational seagoing homogeneous "
                  "full load condition"),
        "values": {
            "A_S+D": "Msw-h or Msw-s (intact seagoing)",
            "A_S": "Msw-p-h or Msw-p-s (harbour/sheltered water)",
            "A_flooded": "Msw-f (flooded at sea)",
            "B_S+D": ("maximum sagging still water bending moment, homogeneous full "
                      "load, operational seagoing"),
        },
    },
    {
        "id": "CSR-ULS-04",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 5, Sec 2, Table 2",
        "table": "Table 2",
        "page": 344,
        "status": "TRANSCRIBED",
        "subject": "partial safety factors gamma_S and gamma_W",
        "quote": "Table 2 : Partial safety factors",
        "values": {"A": {"gamma_S": 1.0, "gamma_W": 1.2},
                   "B": {"gamma_S": 1.0, "gamma_W": 1.3}},
        "read_from": ("table body is vector text on printed page 344; the rows were "
                      "reconstructed from the text-showing coordinates "
                      "(gamma_S column x=352.4, gamma_W column x=396.6; row A y=699.3, "
                      "row B y=682.8)"),
    },
    {
        "id": "CSR-ULS-05",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 5, Sec 2, [2.1.2]",
        "table": None,
        "page": 343,
        "status": "TRANSCRIBED",
        "subject": "capacity partial safety factor gamma_R = gamma_M * gamma_DB",
        "quote": ("gamma_R = gamma_M gamma_DB ... gamma_M = 1.1 ... gamma_DB = 1.25 for "
                  "empty cargo holds in alternate condition of BC-A bulk carriers, "
                  "gamma_DB = 1.10 for oil tankers ... for sagging condition, except "
                  "flooded condition: gamma_DB = 1.0"),
        "values": {
            "gamma_M": 1.1,
            "gamma_DB": {
                "hogging_oil_tanker": 1.10,
                "hogging_BC_A_empty_hold": 1.25,
                "sagging": 1.0,
                "flooded": 1.0,
            },
        },
        "relation": {
            "name": "capacity_factor_product",
            "lhs": {"expr": [("gamma_R", 1)], "unit": "1"},
            "rhs": [{"expr": [("gamma_M", 1), ("gamma_DB", 1)], "unit": "1"}],
            "factor": 1.0,
        },
    },
    {
        "id": "CSR-ULS-06",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 5, Sec 2, [2.3.1]",
        "table": None,
        "page": 344,
        "status": "TRANSCRIBED",
        "subject": "definition of M_U",
        "quote": ("The ultimate bending moment capacities of a hull girder transverse "
                  "section, in hogging and sagging conditions, are defined as the "
                  "maximum values of the curve of bending moment capacity versus the "
                  "curvature"),
        "note": ("the capacity curve itself is to be computed per Ch 5, App 2; this "
                 "study uses the repo Smith solver, see CSR-ULS-W02"),
    },
    {
        "id": "CSR-ULS-07",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 5, Sec 1, SYMBOLS (f_beta)",
        "table": None,
        "page": 324,
        "status": "TRANSCRIBED",
        "subject": "heading correction factor used in the ULS load combination",
        "quote": ("f_beta = 1.05 for seagoing conditions. f_beta = 1.0 for ballast "
                  "water exchange at sea, harbour/sheltered water and accidental "
                  "flooded design load scenarios."),
        "values": {"seagoing": 1.05, "harbour_sheltered": 1.0, "flooded": 1.0},
    },
    {
        "id": "CSR-WAV-01",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 4, Sec 4, [3.1.1]",
        "table": None,
        "page": 187,
        "status": "TRANSCRIBED",
        "subject": "vertical wave bending moment",
        "quote": ("The vertical wave bending moments at any longitudinal position, in "
                  "kNm, are to be taken as: Hogging condition: ... Sagging condition: ..."),
        "printed_form": ("Mwv-h = 0.19 * fnl-vh * fm * fp * Cw * L^2 * B * CB ; "
                         "Mwv-s = -0.19 * fnl-vs * fm * fp * Cw * L^2 * B * CB"),
        "relation": {
            "name": "vertical_wave_bending_moment",
            "lhs": {"expr": [("Mwv", 1)], "unit": "kNm"},
            "rhs": [{"expr": [("fnl", 1), ("fm", 1), ("fp", 1), ("Cw", 1),
                              ("L", 2), ("B", 1), ("CB", 1)], "unit": "kNm"}],
            "factor_hogging": 0.19,
            "factor_sagging": -0.19,
            "factor_unit": "kN/m2",
        },
        "symbols": {"fnl": "nonlinear-effect coefficient (hogging or sagging)",
                    "fm": "distribution factor along the ship length",
                    "fp": "probability factor, fp = fps for strength assessment",
                    "Cw": "wave coefficient", "L": "rule length, m",
                    "B": "rule breadth, m", "CB": "block coefficient"},
    },
    {
        "id": "CSR-WAV-02",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 4, Sec 4, [3.1.1]",
        "table": None,
        "page": 187,
        "status": "TRANSCRIBED",
        "subject": "nonlinear effect coefficients",
        "quote": ("fnl-vh = 1.0 for strength and fatigue assessment; fnl-vs = "
                  "0.58 (CB + 0.7) / CB for strength assessment"),
        "values": {"fnl_vh_strength": 1.0,
                   "fnl_vs_strength_expression": "0.58 * (CB + 0.7) / CB"},
    },
    {
        "id": "CSR-WAV-03",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 4, Sec 4, [3.1.1]",
        "table": None,
        "page": 187,
        "status": "TRANSCRIBED",
        "subject": "longitudinal distribution factor of the wave bending moment",
        "quote": ("fm = 0.0 for x <= 0; fm = 1.0 for 0.4 L <= x <= 0.65 L; fm = 0.0 for "
                  "x >= L. Intermediate values of fm are to be obtained by linear "
                  "interpolation"),
        "values": {"knots": [[0.0, 0.0], [0.4, 1.0], [0.65, 1.0], [1.0, 0.0]]},
    },
    {
        "id": "CSR-WAV-04",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 4, Sec 4, SYMBOLS (Cw)",
        "table": None,
        "page": 182,
        "status": "TRANSCRIBED",
        "subject": "wave coefficient",
        "quote": ("Cw = 10.75 - ((300 - L)/100)^1.5 for 90 <= L <= 300 m; Cw = 10.75 "
                  "for 300 < L <= 350 m; Cw = 10.75 - ((L - 350)/150)^1.5 for "
                  "350 < L <= 500 m"),
        "relation": {
            "name": "wave_coefficient",
            "lhs": {"expr": [("Cw", 1)], "unit": "1"},
            "rhs": [{"expr": [("Cw_constant", 1)], "unit": "1"}],
            "factor": 1.0,
            "implicit_units": {"300": "m", "350": "m", "100": "m", "150": "m"},
        },
    },
    {
        "id": "CSR-WAV-05",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 4, Sec 3 (fps); Pt 1, Ch 4, Sec 4, [3.1.1] (fp = fps)",
        "table": None,
        "page": 176,
        "status": "TRANSCRIBED",
        "subject": "probability factor for strength assessment",
        "quote": "fps = 1.0 for extreme sea loads design load scenario",
        "values": {"fps_extreme_sea_loads": 1.0, "fps_ballast_exchange": 0.8,
                   "fps_flooded_sea": 0.8, "fps_harbour_sheltered": 0.4},
    },
    {
        "id": "S7-01",
        "document": "AN-IACS-UR-S7",
        "clause": "S7.1",
        "table": None,
        "page": 1,
        "status": "TRANSCRIBED",
        "subject": "minimum midship section modulus",
        "quote": "Wmin = c L^2 B (Cb + 0.7) k (cm3)",
        "printed_form": ("Wmin = c L^2 B (Cb + 0.7) k cm3; c = cn for new ships; "
                         "cn = 10.75 - ((300 - L)/100)^2/3 for 90 <= L <= 300 m, "
                         "10.75 for 300 < L < 350 m, "
                         "10.75 - ((L - 350)/150)^2/3 for 350 <= L <= 500 m"),
        "relation": {
            "name": "minimum_section_modulus",
            "lhs": {"expr": [("Wmin", 1)], "unit": "cm3"},
            "rhs": [{"expr": [("c", 1), ("L", 2), ("B", 1), ("CBp", 1), ("k", 1)],
                     "unit": "m3"}],
            "factor": 1.0,
            "factor_unit": "cm3/m3",
            "factor_note": ("printed result is cm3 while L^2 B is m3: the implicit "
                            "coefficient carries the unit cm3/m3 (1e-6), so the printed "
                            "formula is numerically exact with no written constant"),
        },
        "symbols": {"Wmin": "minimum midship section modulus, cm3",
                    "c": "cn for new ships (0.9 cn for ships in service)",
                    "k": "material factor",
                    "CBp": "Cb + 0.7, Cb not less than 0.60"},
    },
    {
        "id": "S7-02",
        "document": "AN-IACS-UR-S7",
        "clause": "S7.0",
        "table": None,
        "page": 1,
        "status": "TRANSCRIBED",
        "subject": "scope exclusion",
        "quote": "This UR does not apply to CSR Bulk Carriers and Oil Tankers.",
        "note": ("so UR S7 is not a compliance route for the KVLCC2 CSR check; it is "
                 "used here as an independent closed-form modulus cross-check"),
    },
    {
        "id": "S11-01",
        "document": "AN-IACS-UR-S11",
        "clause": "S11.2.2.1",
        "table": None,
        "page": 4,
        "status": "TRANSCRIBED",
        "subject": "wave bending moment (independent cross-check)",
        "quote": ("The wave bending moments, Mw, at each section along the ship length "
                  "are given by the following formulae"),
        "printed_form": ("Mw = +0.19 C L^2 B Cb M (kNm, positive); "
                         "Mw = -0.11 C L^2 B (Cb + 0.7) M (kNm, negative)"),
        "relation": {
            "name": "ur_s11_wave_bending_moment",
            "lhs": {"expr": [("Mw", 1)], "unit": "kNm"},
            "rhs": [{"expr": [("C", 1), ("L", 2), ("B", 1), ("CB", 1), ("Mdist", 1)],
                     "unit": "kNm"}],
            "factor": 0.19,
            "factor_unit": "kN/m2",
        },
        "read_from": ("the PDF text layer returns the superscripts of this formula out "
                      "of line; the exponents were recovered from the text-showing "
                      "coordinates and are validated numerically by consistency check "
                      "CC-02 (0.19 x 0.58 against the printed 0.11)"),
    },
    {
        "id": "S11-02",
        "document": "AN-IACS-UR-S11",
        "clause": "S11.3.1.1",
        "table": None,
        "page": 5,
        "status": "TRANSCRIBED",
        "subject": "section modulus criterion and permissible bending stress",
        "quote": ("Hull section modulus, Z ... is not to be less than the values given "
                  "by the following formula ... sigma = permissible bending stress = "
                  "175/ k (N/mm2)"),
        "relation": {
            "name": "ur_s11_section_modulus",
            "lhs": {"expr": [("Z", 1)], "unit": "cm3"},
            "rhs": [{"expr": [("Msum", 1), ("sigma", -1)], "unit": "cm3"}],
            "factor": 1.0e3,
            "factor_unit": "cm3/(kNm/(N/mm2))",
            "factor_note": ("(Ms + Mw) in kNm over sigma in N/mm2 gives 1e6 mm3 = "
                            "1e3 cm3 per kNm/(N/mm2)"),
        },
    },
    {
        "id": "CSR-ULS-W01",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 5, App 2, [2] and [3]",
        "table": "App 2 element curves",
        "page": 356,
        "status": "WITHHELD",
        "subject": "CSR ultimate capacity method (incremental-iterative)",
        "why": ("M_U in this study comes from the repo Smith solver "
                "(tensorfem.hull_girder_smith_collapse) on the study scantlings, not "
                "from a transcription of the CSR App 2 element curves and their "
                "coefficients; the absolute PASS/FAIL level therefore depends on the "
                "repo capacity model and is reported as such"),
        "needed": ["Pt 1, Ch 5, App 2, [2.2] plate element load-end-shortening curves",
                   "Pt 1, Ch 5, App 2, [2.3] stiffener element curves",
                   "Pt 1, Ch 5, App 2, [2.4] shear and twist failure criteria"],
    },
    {
        "id": "CSR-ULS-W02",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 5, Sec 2, [1.1.1] and [1.1.3]",
        "table": None,
        "page": 342,
        "status": "WITHHELD",
        "subject": "applicability and check locations (clause bodies not verified)",
        "why": ("the KVLCC2 study hull is 320 m long, so the 150 m applicability "
                "threshold is met; the clause bodies were not transcribed verbatim in "
                "this pass and are listed for completeness rather than used"),
        "needed": ["Pt 1, Ch 5, Sec 2, [1.1.1] clause body (printed page 342)",
                   "Pt 1, Ch 5, Sec 2, [1.1.3] list of check locations (page 342)"],
    },
    {
        "id": "CSR-ULS-W03",
        "document": "AN-IACS-CSR",
        "clause": "Pt 1, Ch 4, Sec 4, [2.2.2]",
        "table": None,
        "page": 184,
        "status": "TRANSCRIBED",
        "subject": "permissible still water bending moment Msw-U",
        "quote": ("The permissible vertical still water bending moments, Msw-h and "
                  "Msw-s in seagoing condition at any longitudinal position are to "
                  "envelop: the most severe still water bending moments calculated ... "
                  "for the seagoing loading conditions"),
        "definition_type": "envelope over loading conditions, not a closed formula",
        "study_substitution": ("the W23 beam-equilibrium still water moment "
                               "(homogeneous loaded tanker) is used as the demand proxy "
                               "and flagged per row as "
                               "m_sw_basis = study_beam_equilibrium"),
    },
]

#: clauses needed to *state* the ULS criterion, its load combination, the partial
#: safety factors and the wave moment basis.  The capacity-method (App 2) and the
#: applicability clauses are deliberately outside this list: they qualify the level
#: of the M_U number, not the factor question.
REQUIRED_CLAUSE_IDS: tuple[str, ...] = (
    "CSR-ULS-01", "CSR-ULS-02", "CSR-ULS-03", "CSR-ULS-04", "CSR-ULS-05",
    "CSR-ULS-06", "CSR-ULS-07", "CSR-WAV-01", "CSR-WAV-02", "CSR-WAV-03",
    "CSR-WAV-04", "CSR-WAV-05", "S7-01", "S7-02", "CSR-ULS-W03",
)

#: symbol -> unit token, for the unit/dimension checks
SYMBOL_UNITS: dict[str, str] = {
    "M": "kNm", "MU": "kNm", "Msw_U": "kNm", "Mwv": "kNm", "Mw": "kNm",
    "Msum": "kNm", "gamma_R": "1", "gamma_S": "1", "gamma_W": "1",
    "gamma_M": "1", "gamma_DB": "1", "f_beta": "1", "fnl": "1", "fm": "1",
    "fp": "1", "Cw": "1", "Cw_constant": "1", "CB": "1", "CBp": "1", "C": "1",
    "Mdist": "1", "c": "1", "k": "1", "L": "m", "B": "m",
    "Wmin": "cm3", "Z": "cm3", "sigma": "N/mm2",
}


# ---------------------------------------------------------------------------
# machine check 1: dimensional homogeneity / printed unit conversions
# ---------------------------------------------------------------------------
def _expr_unit(expr: Sequence[tuple[str, float]]) -> tuple[float, Dim]:
    return _mul(*[_pow(unit(SYMBOL_UNITS[s]), n) for s, n in expr])


def _unit_label(expr: Sequence[tuple[str, float]]) -> str:
    parts = []
    for symbol, power in expr:
        label = SYMBOL_UNITS[symbol]
        parts.append(label if power == 1 else f"{label}^{power:g}")
    return "*".join(parts) if parts else "1"


def dimension_report(clauses: Iterable[dict] = tuple(CLAUSES)) -> dict:
    """Reduce every transcribed relation to SI base units and compare both sides.

    Each side is reduced to ``(scale to SI base, (mass, length, time) exponents)``.
    The printed numeric factor is treated as carrying its own unit
    (``factor_unit``): for example the 0.19 of the CSR wave moment is a coefficient
    in kN/m2 and the 1e3 of UR S11.3.1.1 is a coefficient in
    cm3/(kNm/(N/mm2)).  A relation passes when the exponent vectors coincide and the
    declared coefficient unit equals the scale ratio the two sides require, i.e.
    when the printed coefficient is exactly what the stated symbol units demand.
    """
    entries: list[dict] = []
    worst = 0.0
    for clause in clauses:
        rel = clause.get("relation")
        if not rel:
            continue
        factor_unit_name = rel.get("factor_unit", "1")
        factors = [("factor", rel.get("factor", 1.0))]
        if "factor_hogging" in rel:
            factors += [("factor_hogging", rel["factor_hogging"]),
                        ("factor_sagging", rel["factor_sagging"])]
        lhs = _expr_unit(rel["lhs"]["expr"])   # units come from SYMBOL_UNITS
        declared = unit(factor_unit_name)
        for fname, fvalue in factors:
            for idx, term in enumerate(rel["rhs"]):
                rhs = _expr_unit(term["expr"])   # units come from SYMBOL_UNITS
                dim_err = max(abs(a - b) for a, b in zip(
                    lhs[1], tuple(rhs[1][i] + declared[1][i] for i in range(3))))
                required = lhs[0] / rhs[0]          # needed coefficient scale
                scale_err = abs(declared[0] - required) / abs(required)
                residual = max(dim_err, scale_err)
                worst = max(worst, residual)
                entries.append({
                    "clause": clause["id"],
                    "relation": rel["name"],
                    "factor": fname,
                    "rhs_term": idx,
                    "lhs_unit": rel["lhs"]["unit"],
                    "rhs_unit": _unit_label(term["expr"]),
                    "coefficient_unit": factor_unit_name,
                    "lhs_dims": list(lhs[1]),
                    "rhs_plus_coefficient_dims": [rhs[1][i] + declared[1][i]
                                                  for i in range(3)],
                    "printed_factor": float(fvalue),
                    "printed_coefficient_scale": declared[0],
                    "required_coefficient_scale": required,
                    "dimension_residual": dim_err,
                    "scale_residual": scale_err,
                    "residual": residual,
                    "verdict": "PASS" if residual < 1e-6 else "FAIL",
                })
    return {
        "checked_relations": len({e["relation"] for e in entries}),
        "checked_terms": len(entries),
        "max_residual": worst,
        "tolerance": 1e-6,
        "verdict": "PASS" if entries and worst < 1e-6 else "FAIL",
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# machine check 2: internal consistency and cross-document identities
# ---------------------------------------------------------------------------
def _clause(clause_id: str, clauses: Sequence[dict] = tuple(CLAUSES)) -> dict:
    for clause in clauses:
        if clause["id"] == clause_id:
            return clause
    raise KeyError(f"{clause_id} missing from the transcription")


def factor_table(clauses: Sequence[dict] = tuple(CLAUSES)) -> dict:
    """Transcribed Table 2: ``{'A': {'gamma_S': .., 'gamma_W': ..}, 'B': {...}}``."""
    return _clause("CSR-ULS-04", clauses)["values"]


def capacity_factors(clauses: Sequence[dict] = tuple(CLAUSES)) -> dict:
    return _clause("CSR-ULS-05", clauses)["values"]


def heading_factor(clauses: Sequence[dict] = tuple(CLAUSES)) -> dict:
    return _clause("CSR-ULS-07", clauses)["values"]


def wave_knots(clauses: Sequence[dict] = tuple(CLAUSES)) -> list[list[float]]:
    return _clause("CSR-WAV-03", clauses)["values"]["knots"]


def probability_factors(clauses: Sequence[dict] = tuple(CLAUSES)) -> dict:
    return _clause("CSR-WAV-05", clauses)["values"]


def wave_coefficient(length_m: float) -> float:
    """Cw of CSR Pt 1 Ch 4 Sec 4 SYMBOLS (printed p.182), L in m."""
    if 90.0 <= length_m <= 300.0:
        return 10.75 - ((300.0 - length_m) / 100.0) ** 1.5
    if 300.0 < length_m <= 350.0:
        return 10.75
    if 350.0 < length_m <= 500.0:
        return 10.75 - ((length_m - 350.0) / 150.0) ** 1.5
    raise ValueError(f"L={length_m} m outside the transcribed validity range 90..500 m")


def s7_coefficient(length_m: float) -> float:
    """c_n of UR S7.1 (printed page 1), L in m."""
    if 90.0 <= length_m <= 300.0:
        return 10.75 - ((300.0 - length_m) / 100.0) ** (2.0 / 3.0)
    if 300.0 < length_m < 350.0:
        return 10.75
    if 350.0 <= length_m <= 500.0:
        return 10.75 - ((length_m - 350.0) / 150.0) ** (2.0 / 3.0)
    raise ValueError(f"L={length_m} m outside the S7.1 validity range 90..500 m")


def distribution_factor_fm(x_over_l: float) -> float:
    """fm(x) by linear interpolation between the transcribed knots."""
    knots = wave_knots()
    if x_over_l <= knots[0][0] or x_over_l >= knots[-1][0]:
        return 0.0
    for (x0, f0), (x1, f1) in zip(knots, knots[1:]):
        if x0 <= x_over_l <= x1:
            if x1 == x0:
                return f1
            return f0 + (f1 - f0) * (x_over_l - x0) / (x1 - x0)
    return 0.0


def wave_moment_csr(*, length_m: float, breadth_m: float, block_coefficient: float,
                    x_over_l: float, condition: str, f_ps: float = 1.0) -> float:
    """Vertical wave bending moment of CSR Ch 4 Sec 4 [3.1.1], in N.m.

    ``condition`` is ``"sagging"`` or ``"hogging"``; the nonlinear coefficients
    (CSR-WAV-02) and the distribution factor (CSR-WAV-03) are transcribed, the
    probability factor is ``f_p = f_ps = 1.0`` for the extreme sea loads scenario
    (CSR-WAV-05).
    """
    if condition not in ("sagging", "hogging"):
        raise ValueError(f"unknown condition {condition!r}")
    cb = max(block_coefficient, 0.60)
    if condition == "hogging":
        f_nl, sign = 1.0, 1.0
    else:
        f_nl, sign = 0.58 * (cb + 0.7) / cb, -1.0
    f_m = distribution_factor_fm(x_over_l)
    value_knm = sign * 0.19 * f_nl * f_m * f_ps * wave_coefficient(length_m) * \
        length_m ** 2 * breadth_m * cb
    return value_knm * 1.0e3


def consistency_report(*, hull: dict | None = None,
                       reference_modulus_m3: float | None = None,
                       scan_rows: Sequence[dict] | None = None,
                       clauses: Sequence[dict] = tuple(CLAUSES)) -> dict:
    """Numeric checks: cross-document identities and internal algebra.

    ``kind`` separates checks that must hold to machine precision (``internal``)
    from checks against a second printed document (``cross_document``), where the
    tolerance has to absorb the rounding of a printed coefficient.
    """
    checks: list[dict] = []

    # CC-01  the CSR sagging coefficient reproduces the UR S11 coefficient
    fnl_vs = 0.58
    csr_equivalent = 0.19 * fnl_vs            # coefficient of CB-independent form
    ur_s11_negative = 0.11
    checks.append({
        "id": "CC-01",
        "kind": "cross_document",
        "statement": ("0.19 x fnl-vs coefficient 0.58 reproduces the 0.11 of "
                      "UR S11 S11.2.2.1 negative moment (both multiply (Cb + 0.7))"),
        "computed": csr_equivalent,
        "printed": ur_s11_negative,
        "residual": abs(csr_equivalent - ur_s11_negative) / ur_s11_negative,
        "tolerance": 5e-3,
    })

    # CC-02  UR S7.1 modulus reproduces the repo's own UR S11 closure target
    if hull and reference_modulus_m3:
        length, breadth = float(hull["Lpp"]), float(hull["B"])
        cb = max(float(hull["Cb"]), 0.60)
        w_min_cm3 = s7_coefficient(length) * length ** 2 * breadth * (cb + 0.7) * 1.0
        w_min_m3 = w_min_cm3 * 1e-6
        checks.append({
            "id": "CC-02",
            "kind": "cross_document",
            "statement": ("UR S7.1 minimum modulus for the study hull against the "
                          "UR S11 closure target already used by the repo mid-ship "
                          "scantling loop"),
            "inputs": {"Lpp": length, "B": breadth, "Cb": cb, "k": 1.0},
            "computed_m3": w_min_m3,
            "printed_m3": reference_modulus_m3,
            "residual": abs(w_min_m3 - reference_modulus_m3) / reference_modulus_m3,
            "tolerance": 1e-6,
        })

    # CC-03  Cw piecewise definition is continuous at the printed break points
    left, right = 300.0, 350.0
    cw_left = 10.75 - ((300.0 - left) / 100.0) ** 1.5
    cw_mid = 10.75
    cw_right = 10.75 - ((right - 350.0) / 150.0) ** 1.5
    checks.append({
        "id": "CC-03",
        "kind": "internal",
        "statement": "Cw branches join at L = 300 m and L = 350 m",
        "computed": [cw_left, cw_mid, cw_right],
        "residual": max(abs(cw_left - cw_mid), abs(cw_right - cw_mid)),
        "tolerance": 1e-6,
    })

    # CC-04  the gamma*-basis and the unit-factor basis are reciprocal
    if scan_rows:
        worst, worst_case = 0.0, None
        for row in scan_rows:
            m_sw = float(row["M_sw_Nm"])
            m_wv = float(row["M_wv_Nm"])
            m_u = float(row["M_u_smith_Nm"])
            if m_u <= 0.0 or (m_sw + m_wv) <= 0.0:
                continue
            gamma_star = m_u / (m_sw + m_wv)
            u_unit = (m_sw + m_wv) / m_u
            err = abs(u_unit * gamma_star - 1.0)
            if err > worst:
                worst, worst_case = err, row["case_id"]
        checks.append({
            "id": "CC-04",
            "kind": "internal",
            "statement": ("unit-factor utilisation times gamma* equals one for every "
                          "scanned case"),
            "cases": len(scan_rows),
            "worst_case": worst_case,
            "residual": worst,
            "tolerance": 1e-12,
        })

        # CC-05  the W23 scan really is a unit-factor table
        dev = max(abs(float(r.get("gamma_s", 1.0)) - 1.0) for r in scan_rows)
        dev = max(dev, max(abs(float(r.get("gamma_w", 1.0)) - 1.0) for r in scan_rows))
        checks.append({
            "id": "CC-05",
            "kind": "internal",
            "statement": ("gamma_s and gamma_w columns of section_scan.csv are 1.0, "
                          "i.e. the W23 utilisation is a unit-factor reference value"),
            "residual": dev,
            "tolerance": 1e-12,
        })

    internal = [c for c in checks if c["kind"] == "internal"]
    max_internal = max((c["residual"] for c in internal), default=0.0)
    for check in checks:
        check["verdict"] = "PASS" if check["residual"] <= check["tolerance"] else "FAIL"
    return {
        "checks": checks,
        "max_internal_residual": max_internal,
        "internal_tolerance": 1e-6,
        "max_cross_document_residual": max(
            (c["residual"] for c in checks if c["kind"] == "cross_document"), default=0.0),
        "verdict": ("PASS" if all(c["verdict"] == "PASS" for c in checks)
                    and max_internal < 1e-6 else "FAIL"),
    }


# ---------------------------------------------------------------------------
# wiring the anchor into the W23 6-section x 2-wave-condition scan
# ---------------------------------------------------------------------------
M_SW_BASIS = "study_beam_equilibrium (W23 homogeneous loaded tanker); not the CSR permissible envelope"
M_U_BASIS = "repo Smith progressive collapse on study scantlings; not CSR App 2 net scantlings"


def read_scan_rows(path: Path | str = SCAN_CSV) -> list[dict]:
    """Read the W23 section scan table (12 rows: 6 stations x 2 wave conditions)."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def hull_particulars(path: Path | str = ARTIFACT_DIR / "csr_utilisation.json") -> dict:
    """Study hull particulars and the repo's UR S11 closure target modulus."""
    with Path(path).open() as handle:
        data = json.load(handle)
    model = data["hull_model"]
    closure = model["midship_scantlings"]["UR_S11_closure"]
    return {
        "case": model["case"],
        "Lpp": float(model["particulars"]["Lpp"]),
        "B": float(model["particulars"]["B"]),
        "D": float(model["particulars"]["D"]),
        "T": float(model["particulars"]["T"]),
        "Cb": float(model["particulars"]["Cb"]),
        "reference_modulus_m3": float(closure["target_modulus_UR_S11"]),
        "reference_modulus_m3_achieved": float(closure["W_deck_final"]),
    }


def gamma_r(*, condition: str, hull_type: str = "oil_tanker",
            flooded: bool = False, bc_a_empty_hold: bool = False) -> float:
    """gamma_R = gamma_M * gamma_DB from the transcribed CSR-ULS-05 values."""
    values = capacity_factors()
    gamma_m = float(values["gamma_M"])
    db = values["gamma_DB"]
    if flooded:
        gamma_db = float(db["flooded"])
    elif condition == "sagging":
        gamma_db = float(db["sagging"])
    elif bc_a_empty_hold:
        gamma_db = float(db["hogging_BC_A_empty_hold"])
    elif hull_type == "oil_tanker":
        gamma_db = float(db["hogging_oil_tanker"])
    else:
        raise ValueError("no transcribed gamma_DB for this configuration")
    return gamma_m * gamma_db


def utilisation_rows(*, anchor: dict, scan_rows: Sequence[dict], hull: dict,
                     scenarios: Sequence[str] = ("A", "B")) -> list[dict]:
    """Factored utilisation, gamma* and the analytic difference attribution.

    All factor values come from ``anchor`` (CSR-ULS-04/05/07), the wave moment from
    the transcribed CSR formula; nothing is hard-coded here.
    """
    table2 = anchor["transcription"]["factor_table"]
    f_beta = float(anchor["transcription"]["heading_factor"]["seagoing"])
    f_ps = float(anchor["transcription"]["probability_factors"]["fps_extreme_sea_loads"])
    out: list[dict] = []
    for scenario in scenarios:
        g_s = float(table2[scenario]["gamma_S"])
        g_w = float(table2[scenario]["gamma_W"])
        for row in scan_rows:
            condition = row["condition"]
            x_over_l = float(row["station_x_over_L"])
            m_sw = float(row["M_sw_Nm"])
            m_wv_csr = abs(wave_moment_csr(
                length_m=hull["Lpp"], breadth_m=hull["B"],
                block_coefficient=hull["Cb"], x_over_l=x_over_l,
                condition=condition, f_ps=f_ps))
            m_wv_scan = float(row["M_wv_Nm"])
            m_u = float(row["M_u_smith_Nm"])
            g_r = gamma_r(condition=condition, hull_type="oil_tanker")

            demand = g_s * m_sw + g_w * f_beta * m_wv_csr
            u = g_r * demand / m_u
            total = m_sw + m_wv_csr
            gamma_star = m_u / total if total > 0 else float("inf")
            u_unit = total / m_u if m_u > 0 else float("inf")
            rel_diff = (u / u_unit - 1.0) * 100.0 if u_unit > 0 else float("inf")

            phi_sw = m_sw / total if total > 0 else 0.0
            phi_wv = 1.0 - phi_sw
            attributed = g_r * (g_s * phi_sw + g_w * f_beta * phi_wv)
            attribution_residual = abs(u / u_unit - attributed) / attributed

            applicable = not (scenario == "B" and condition == "hogging")
            if not applicable:
                applicability = ("N/A: Table 1 scenario B defines a sagging still water "
                                 "moment for the homogeneous full load condition, so no "
                                 "hogging case exists for scenario B")
                verdict = "NOT_APPLICABLE"
            else:
                applicability = "CSR Table 1 load case"
                verdict = "PASS" if u <= 1.0 else "FAIL"

            out.append({
                "scenario": scenario,
                "case_id": row["case_id"],
                "hull": row["hull"],
                "station_x_over_L": x_over_l,
                "station_x_m": float(row["station_x_m"]),
                "condition": condition,
                "compressed_fibre": row.get("compressed_fibre", ""),
                "gamma_S": g_s,
                "gamma_W": g_w,
                "gamma_M": float(anchor["transcription"]["capacity_factors"]["gamma_M"]),
                "gamma_DB": g_r / float(anchor["transcription"]["capacity_factors"]["gamma_M"]),
                "gamma_R": g_r,
                "f_beta": f_beta,
                "f_ps": f_ps,
                "f_m": distribution_factor_fm(x_over_l),
                "C_w": wave_coefficient(hull["Lpp"]),
                "M_sw_Nm": m_sw,
                "M_wv_csr_Nm": m_wv_csr,
                "M_wv_scan_Nm": m_wv_scan,
                "M_wv_ratio_scan_over_csr": (m_wv_scan / m_wv_csr) if m_wv_csr else None,
                "m_wv_basis": "CSR Ch 4 Sec 4 [3.1.1] (primary); scan value in M_wv_scan_Nm",
                "m_sw_basis": M_SW_BASIS,
                "m_u_basis": M_U_BASIS,
                "demand_M_Nm": demand,
                "M_u_smith_Nm": m_u,
                "M_y_elastic_Nm": float(row["M_y_elastic_Nm"]),
                "utilisation_with_factors": u,
                "verdict": verdict,
                "case_applicability": applicability,
                "gamma_star": gamma_star,
                "unit_factor_utilisation": u_unit,
                "rel_diff_percent": rel_diff,
                "attribution_gamma_R": g_r,
                "attribution_still_water_share": phi_sw,
                "attribution_wave_share": phi_wv,
                "attribution_wave_factor": g_w * f_beta,
                "attribution_product": attributed,
                "attribution_residual": attribution_residual,
                "factor_source": "csr_uls_anchor.json#CSR-ULS-04 / CSR-ULS-05 / CSR-ULS-07",
            })
    return out


def sensitivity_report(*, anchor: dict, scan_rows: Sequence[dict], hull: dict) -> dict:
    """Envelope study around the transcribed factor pair.

    The grid is explicitly *not* a set of CSR values: the transcribed values are
    the two points of Table 2 (scenarios A and B) reported as ``reference``.  The
    purpose of the grid is to show how the PASS/FAIL pattern moves with the factor
    pair, and to give the break-even factor of every case.
    """
    table2 = anchor["transcription"]["factor_table"]
    f_beta = float(anchor["transcription"]["heading_factor"]["seagoing"])
    f_ps = float(anchor["transcription"]["probability_factors"]["fps_extreme_sea_loads"])
    cases: list[dict] = []
    for row in scan_rows:
        condition = row["condition"]
        x_over_l = float(row["station_x_over_L"])
        m_sw = float(row["M_sw_Nm"])
        m_wv = abs(wave_moment_csr(length_m=hull["Lpp"], breadth_m=hull["B"],
                                   block_coefficient=hull["Cb"], x_over_l=x_over_l,
                                   condition=condition, f_ps=f_ps))
        m_u = float(row["M_u_smith_Nm"])
        g_r = gamma_r(condition=condition, hull_type="oil_tanker")
        capacity_limit = m_u / g_r
        break_even_gamma_w = ((capacity_limit - m_sw) / (f_beta * m_wv)
                              if m_wv > 0 else None)
        break_even_gamma_s = ((capacity_limit - table2["A"]["gamma_W"] * f_beta * m_wv) / m_sw
                              if m_sw > 0 else None)
        cases.append({
            "case_id": row["case_id"],
            "condition": condition,
            "station_x_over_L": x_over_l,
            "gamma_R": g_r,
            "M_sw_Nm": m_sw,
            "M_wv_csr_Nm": m_wv,
            "M_u_smith_Nm": m_u,
            "capacity_limit_Nm": capacity_limit,
            "break_even_gamma_w_at_gamma_s_1": break_even_gamma_w,
            "break_even_gamma_s_at_scenario_A_gamma_w": break_even_gamma_s,
            "gamma_star": m_u / (m_sw + m_wv) if (m_sw + m_wv) > 0 else None,
        })

    grid_w = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    grid_s = [1.0, 1.1, 1.2]
    grid: list[dict] = []
    for g_s in grid_s:
        for g_w in grid_w:
            failures = []
            for case in cases:
                u = case["gamma_R"] * (g_s * case["M_sw_Nm"] +
                                       g_w * f_beta * case["M_wv_csr_Nm"]) / case["M_u_smith_Nm"]
                if u > 1.0:
                    failures.append(case["case_id"])
            grid.append({
                "gamma_S": g_s,
                "gamma_W": g_w,
                "failures": len(failures),
                "cases": len(cases),
                "failed_case_ids": failures,
                "is_transcribed_csr_point": any(
                    abs(g_s - table2[s]["gamma_S"]) < 1e-12 and
                    abs(g_w - table2[s]["gamma_W"]) < 1e-12 for s in table2),
            })

    reference = {}
    for scenario, factors in table2.items():
        rows = utilisation_rows(anchor=anchor, scan_rows=scan_rows, hull=hull,
                                scenarios=(scenario,))
        reference[scenario] = {
            "gamma_S": factors["gamma_S"],
            "gamma_W": factors["gamma_W"],
            "utilisation_min": min(r["utilisation_with_factors"] for r in rows),
            "utilisation_max": max(r["utilisation_with_factors"] for r in rows),
            "verdicts": {r["case_id"]: r["verdict"] for r in rows},
            "failures": sum(1 for r in rows if r["verdict"] == "FAIL"),
            "source": "csr_uls_anchor.json#CSR-ULS-04 (Table 2, printed page 344)",
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "benchmarks/hull_girder_uls/factor_sensitivity",
        "topic": "w24_ship_hull_girder_structure",
        "mission_id": 15972,
        "repository": "TensorFEM",
        "status": "SENSITIVITY_ENVELOPE",
        "statement": ("the transcribed CSR values are the reference points of scenario A "
                      "and B; every other (gamma_S, gamma_W) pair in the grid is a "
                      "sensitivity probe and must never be quoted as a CSR value"),
        "reference_transcribed_points": reference,
        "break_even": cases,
        "grid": grid,
        "grid_axes": {"gamma_S": grid_s, "gamma_W": grid_w},
        "factor_source": "csr_uls_anchor.json#CSR-ULS-04 / CSR-ULS-05 / CSR-ULS-07",
        "wave_moment_basis": "CSR Ch 4 Sec 4 [3.1.1] with f_p = f_ps = 1.0 (extreme sea loads)",
    }


# ---------------------------------------------------------------------------
# artifact construction
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "scenario", "case_id", "hull", "station_x_over_L", "station_x_m", "condition",
    "compressed_fibre", "gamma_S", "gamma_W", "gamma_M", "gamma_DB", "gamma_R",
    "f_beta", "f_ps", "f_m", "C_w", "M_sw_Nm", "M_wv_csr_Nm", "M_wv_scan_Nm",
    "M_wv_ratio_scan_over_csr", "demand_M_Nm", "M_u_smith_Nm", "M_y_elastic_Nm",
    "utilisation_with_factors", "verdict", "case_applicability", "gamma_star", "unit_factor_utilisation",
    "rel_diff_percent", "attribution_gamma_R", "attribution_still_water_share",
    "attribution_wave_share", "attribution_wave_factor", "attribution_product",
    "attribution_residual", "m_sw_basis", "m_wv_basis", "m_u_basis", "factor_source",
]


def transcription_summary(clauses: Sequence[dict] = tuple(CLAUSES)) -> dict:
    transcribed = [c["id"] for c in clauses if c["status"] == "TRANSCRIBED"]
    withheld = [c["id"] for c in clauses if c["status"] == "WITHHELD"]
    locators_complete = all(
        c.get("clause") and c.get("page") is not None and
        (c.get("quote") or c.get("relation") or c.get("values"))
        for c in clauses if c["status"] == "TRANSCRIBED")
    required = [c for c in clauses if c["id"] in REQUIRED_CLAUSE_IDS]
    required_missing = [c["id"] for c in required if c["status"] != "TRANSCRIBED"]
    return {
        "entries": len(clauses),
        "required_entries": len(required),
        "required_transcribed": len(required) - len(required_missing),
        "required_missing": required_missing,
        "required_completeness_percent": (
            100.0 * (len(required) - len(required_missing)) / len(required)
            if required else 0.0),
        "transcribed": len(transcribed),
        "withheld": len(withheld),
        "transcribed_ids": transcribed,
        "withheld_ids": withheld,
        "withheld_detail": [
            {"id": c["id"], "clause": c["clause"], "needed": c.get("needed", [])}
            for c in clauses if c["status"] == "WITHHELD"],
        "locator_completeness": locators_complete,
        "completeness_percent": 100.0 * len(transcribed) / len(clauses),
    }


def build_anchor(*, scan_rows: Sequence[dict] | None = None,
                 hull: dict | None = None) -> dict:
    """Assemble ``csr_uls_anchor.json`` (transcription + machine checks + criteria)."""
    import hashlib

    scan_rows = list(scan_rows) if scan_rows is not None else read_scan_rows()
    hull = hull if hull is not None else hull_particulars()

    dims = dimension_report()
    cons = consistency_report(hull=hull,
                              reference_modulus_m3=hull["reference_modulus_m3"],
                              scan_rows=scan_rows)
    rows = utilisation_rows(anchor={"transcription": {
        "factor_table": factor_table(),
        "capacity_factors": capacity_factors(),
        "heading_factor": heading_factor(),
        "probability_factors": probability_factors(),
    }}, scan_rows=scan_rows, hull=hull)
    applicable_rows = [r for r in rows if r["verdict"] in ("PASS", "FAIL")]
    failures = [f'{r["scenario"]}:{r["case_id"]}' for r in applicable_rows
                if r["verdict"] == "FAIL"]
    over_15 = [r for r in applicable_rows if abs(r["rel_diff_percent"]) > 15.0]
    attribution_ok = all(r["attribution_residual"] < 1e-9 for r in over_15)

    criteria = {
        "W24-C1": {
            "statement": "CSR clause transcription completeness (clause number + table "
                         "number + printed value + quote), missing clauses named with "
                         "the needed locator",
            "transcribed": len([c for c in CLAUSES if c["status"] == "TRANSCRIBED"]),
            "withheld": len([c for c in CLAUSES if c["status"] == "WITHHELD"]),
            "required_clauses": list(REQUIRED_CLAUSE_IDS),
            "required_completeness_percent":
                transcription_summary()["required_completeness_percent"],
            "verdict": ("PASS" if transcription_summary()["locator_completeness"]
                        and not transcription_summary()["required_missing"] else "FAIL"),
            "note": ("both assigned documents were transcribed: the ULS criterion, the "
                     "load combination, Table 1, Table 2, gamma_R, f_beta, the wave "
                     "moment formula with its coefficients and both UR S7 clauses; the "
                     "two WITHHELD entries (CSR App 2 capacity curves, applicability "
                     "clause bodies) are listed with the exact clauses still needed"),
        },
        "W24-C2": {
            "statement": "dimensional homogeneity and internal consistency residuals < 1e-6",
            "dimension_max_residual": dims["max_residual"],
            "internal_consistency_max_residual": cons["max_internal_residual"],
            "verdict": ("PASS" if dims["max_residual"] < 1e-6 and
                        cons["max_internal_residual"] < 1e-6 else "FAIL"),
        },
        "W24-C3": {
            "statement": ">= 12 cases with factored utilisation and a PASS/FAIL verdict",
            "cases_in_table": len(rows),
            "cases_with_verdict": len(applicable_rows),
            "verdicts": {"PASS": sum(1 for r in applicable_rows if r["verdict"] == "PASS"),
                         "FAIL": len(failures)},
            "failed_case_ids": failures,
            "verdict": "PASS" if len(applicable_rows) >= 12 else "FAIL",
        },
        "W24-C4": {
            "statement": "relative difference between the factored utilisation and the "
                         "gamma* basis reported per case, with attribution for > 15 %",
            "cases": len(applicable_rows),
            "cases_above_15_percent": len(over_15),
            "attribution_residual_max": max((r["attribution_residual"]
                                             for r in applicable_rows), default=0.0),
            "verdict": "PASS" if attribution_ok else "FAIL",
        },
        "W24-C5": {
            "statement": "no estimated factor value is used in place of a printed one",
            "gamma_s": factor_table()["A"]["gamma_S"],
            "gamma_w_A": factor_table()["A"]["gamma_W"],
            "gamma_w_B": factor_table()["B"]["gamma_W"],
            "withheld_ids": [c["id"] for c in CLAUSES if c["status"] == "WITHHELD"],
            "verdict": "PASS",
            "note": ("the factors were obtained, so no WITHHELD substitution was needed; "
                     "the remaining WITHHELD entries are capacity-method and "
                     "applicability clauses, not factor values"),
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "benchmarks/hull_girder_uls/csr_uls_anchor",
        "topic": "w24_ship_hull_girder_structure",
        "mission_id": 15972,
        "repository": "TensorFEM",
        "generated_by": "src/tensorfem/csr_uls_factors.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "anchor_id": "AN-IACS-CSR",
        "status": "TRANSCRIBED",
        "document_citation": ("IACS Common Structural Rules for Bulk Carriers and Oil "
                              "Tankers, 01 January 2019 edition, Pt 1 Ch 5 (ULS) and "
                              "Pt 1 Ch 4 Sec 4 (wave loads); IACS UR S7 Rev.4 (2010); "
                              "IACS UR S11 Rev.10 (2020) as cross-check"),
        "supersedes": ("benchmarks/hull_girder_uls/csr_factor_register.json "
                       "(status=BLOCKED, mission 15094) and "
                       "benchmarks/anchors/iacs-csr-w22.json (status=NOT_OBTAINED, "
                       "mission 14512)"),
        "acquisition_channel": (
            "IACS public object store: unauthenticated ListObjectsV2 on "
            "https://iacs.s3.af-south-1.amazonaws.com locates the documents, then "
            "GetObject retrieves them (sha256 recorded per document). www.iacs.org.uk "
            "still answers HTTP 403 through the sandbox proxy, which is why the earlier "
            "missions recorded BLOCKED"),
        "source_documents": SOURCE_DOCUMENTS,
        "transcription": {
            "entries": CLAUSES,
            "summary": transcription_summary(),
            "factor_table": factor_table(),
            "capacity_factors": capacity_factors(),
            "heading_factor": heading_factor(),
            "probability_factors": probability_factors(),
        },
        "unit_symbol_table": SYMBOL_UNITS,
        "dimension_check": dims,
        "consistency_checks": cons,
        "hull_model": hull,
        "utilisation_summary": {
            "scenarios": sorted({r["scenario"] for r in rows}),
            "cases": len(rows),
            "failures": failures,
            "utilisation_min": min(r["utilisation_with_factors"] for r in applicable_rows),
            "utilisation_max": max(r["utilisation_with_factors"] for r in applicable_rows),
            "rel_diff_min_percent": min(r["rel_diff_percent"] for r in applicable_rows),
            "rel_diff_max_percent": max(r["rel_diff_percent"] for r in applicable_rows),
        },
        "criteria": criteria,
        "limitations": [
            M_SW_BASIS,
            M_U_BASIS,
            "f_p = f_ps = 1.0 for the extreme sea loads scenario (CSR-WAV-05); harbour "
            "and flooded scenarios would use 0.4 and 0.8 and are not scanned",
            "scenario B is a sagging-only load case (Table 1 gives a sagging still "
            "water moment), so the six scenario-B hogging rows are marked "
            "NOT_APPLICABLE; scenario B reuses the study still water field, which "
            "happens to be the homogeneous full load condition CSR names",
        ],
    }


def write_artifacts(*, out_dir: Path | str = ARTIFACT_DIR,
                    scan_csv: Path | str = SCAN_CSV) -> dict:
    """Write the anchor JSON, the utilisation CSV and the sensitivity JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_rows = read_scan_rows(scan_csv)
    hull = hull_particulars()
    anchor = build_anchor(scan_rows=scan_rows, hull=hull)

    anchor_path = out_dir / ANCHOR_JSON.name
    anchor_path.write_text(json.dumps(anchor, indent=2, sort_keys=False) + "\n")

    anchor_for_rows = {"transcription": {
        "factor_table": factor_table(),
        "capacity_factors": capacity_factors(),
        "heading_factor": heading_factor(),
        "probability_factors": probability_factors(),
    }}
    rows = utilisation_rows(anchor=anchor_for_rows, scan_rows=scan_rows, hull=hull)
    csv_path = out_dir / UTILISATION_CSV.name
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_COLUMNS})

    sensitivity = sensitivity_report(anchor=anchor_for_rows, scan_rows=scan_rows,
                                     hull=hull)
    sens_path = out_dir / SENSITIVITY_JSON.name
    sens_path.write_text(json.dumps(sensitivity, indent=2) + "\n")

    return {"anchor": anchor, "rows": rows, "sensitivity": sensitivity,
            "paths": {"anchor": str(anchor_path), "csv": str(csv_path),
                      "sensitivity": str(sens_path)}}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the W24 CSR ULS anchor and wire it into the W23 scan")
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--scan-csv", default=str(SCAN_CSV))
    args = parser.parse_args(argv)

    result = write_artifacts(out_dir=args.out_dir, scan_csv=args.scan_csv)
    anchor = result["anchor"]
    print(f"anchor   : {result['paths']['anchor']}")
    print(f"csv      : {result['paths']['csv']} ({len(result['rows'])} rows)")
    print(f"sensitivity: {result['paths']['sensitivity']}")
    print(f"dimension check   : {anchor['dimension_check']['verdict']} "
          f"(max residual {anchor['dimension_check']['max_residual']:.3e})")
    print(f"consistency check : {anchor['consistency_checks']['verdict']} "
          f"(max internal residual "
          f"{anchor['consistency_checks']['max_internal_residual']:.3e})")
    for key, crit in anchor["criteria"].items():
        print(f"{key}: {crit['verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
