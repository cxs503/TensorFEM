"""W22 anchor register: IACS CSR Pt.1 Ch.5 ULS / yield criteria for the hull girder.

Why this script exists
----------------------
The W22 mission requires the IACS CSR partial-safety-factor table to be
transcribed **verbatim** from an authorised copy of the rule text before any
utilization number can be called a CSR compliance value.  This environment does
not hold such a copy (see ``acquisition_attempts`` below, which are the actual
commands and their actual outcomes).  The register therefore records every
factor the criterion needs, with the clause locator *as assigned by the mission
brief*, and marks each one ``BLOCKED`` instead of filling in a remembered
number.

Discipline (inherited from ``benchmarks/anchors/sources``):

* an entry may only reach ``VERBATIM`` by copying characters from an authorised
  copy of the source document (quote, table/figure/page locator, printed value
  strings, no re-rounding);
* values recalled from memory, quoted in secondary literature, or read back
  from this repository's own code are **not** anchors and stay
  ``UNCITED_CANDIDATE`` with ``usable_for_compliance_claim = false``.

Run ``python benchmarks/anchors/iacs_csr_w22_anchor.py`` to (re)write
``benchmarks/anchors/iacs-csr-w22.json``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DOCUMENT = {
    "citation_as_assigned": (
        "IACS Common Structural Rules for Bulk Carriers and Oil Tankers (2019), "
        "Pt.1 Ch.5; IACS Unified Requirement S7 (2010) Minimum longitudinal "
        "strength standard"
    ),
    "subject_quantity": (
        "hull-girder ultimate limit state gamma_s M_sw + gamma_w M_wv <= M_u and "
        "yield criterion sigma = M/Z <= sigma_y/gamma_R at the midship section"
    ),
    "why_it_matters": (
        "without the printed gamma_s / gamma_w / gamma_R table the utilization "
        "computed here is a unit-factor reference value, not a CSR compliance "
        "statement; the study therefore also reports the admissible factor "
        "M_u / (M_sw + M_wv)"
    ),
}

#: One row per factor / clause the criterion needs.  ``clause_as_assigned`` is
#: the locator supplied by the mission brief and is *not* verified against the
#: rule text in this environment.
FACTOR_REGISTER = [
    {
        "id": "gamma_s",
        "symbol": "gamma_s",
        "role": "partial safety factor on the still-water bending moment M_sw",
        "clause_as_assigned": "IACS CSR BC&OT (2019) Pt.1 Ch.5, hull-girder ULS load combination",
        "expected_printed_form": "gamma_s M_sw + gamma_w M_wv <= M_u (or an equivalent table of gamma values)",
    },
    {
        "id": "gamma_w",
        "symbol": "gamma_w",
        "role": "partial safety factor on the vertical wave bending moment M_wv",
        "clause_as_assigned": "IACS CSR BC&OT (2019) Pt.1 Ch.5, hull-girder ULS load combination",
        "expected_printed_form": "same combination expression as gamma_s",
    },
    {
        "id": "gamma_R",
        "symbol": "gamma_R",
        "role": "resistance factor in the yield criterion sigma = M/Z <= sigma_y/gamma_R",
        "clause_as_assigned": "IACS CSR BC&OT (2019) Pt.1 Ch.5, yield / permissible-stress check",
        "expected_printed_form": "gamma_R with the associated material factor gamma_M if printed",
    },
    {
        "id": "M_u_definition",
        "symbol": "M_u",
        "role": "definition of the ultimate vertical bending moment and how the rule expects it to be obtained",
        "clause_as_assigned": "IACS CSR BC&OT (2019) Pt.1 Ch.5, ultimate strength; IACS UR S7 (2010) minimum longitudinal strength standard",
        "expected_printed_form": "M_u >= gamma_s M_sw + gamma_w M_wv with the capacity model the rule prescribes",
    },
    {
        "id": "M_wv_formula",
        "symbol": "M_wv",
        "role": "rule vertical wave bending moment (hogging and sagging) and its distribution factors",
        "clause_as_assigned": "IACS CSR BC&OT (2019) Pt.1 Ch.5 wave-load formula (equivalently IACS UR S11)",
        "expected_printed_form": "closed-form expression in L, B, Cb and the wave coefficient C with f_nl/f_m/f_h",
    },
    {
        "id": "W_min_formula",
        "symbol": "W_min",
        "role": "minimum midship section modulus used to close the scantlings before the ULS check",
        "clause_as_assigned": "IACS UR S7 (2010) minimum longitudinal strength standard",
        "expected_printed_form": "W_min = C L^2 B (Cb + 0.7) with its unit convention",
    },
]

ACQUISITION_ATTEMPTS = [
    {
        "channel": "local filesystem corpus",
        "action": "find / -maxdepth 6 -iname '*csr*' -o -iname '*iacs*' (excluding /proc, /sys and this worktree)",
        "result": (
            "no rule text; hits are the repository's own hull-girder code "
            "(src/tensorfem/hull_girder_csr_chain.py, docs/hull-girder-csr-bending-chain.md, "
            "docs/assets/hull-girder-csr/) and unrelated csrc directories"
        ),
    },
    {
        "channel": "local PDF corpus",
        "action": "find / -maxdepth 5 -iname '*.pdf'",
        "result": "no IACS publication among the PDFs found (tooling manuals and unrelated papers only)",
    },
    {
        "channel": "repository / platform corpus",
        "action": "grep -rIl -i 'common structural rules' /root /opt /srv /data (excluding this worktree)",
        "result": "no match: no verbatim text, table extract or page locator for CSR Pt.1 Ch.5 or UR S7",
    },
    {
        "channel": "network",
        "action": (
            "urllib GET https://iacs.org.uk/ , https://iacs.org.uk/resolutions/unified-requirements , "
            "https://www.iacs.org.uk/download/10936 (timeout 6 s); direct curl to external hosts is "
            "blocked by the execution sandbox policy"
        ),
        "result": (
            "HTTP 403 Forbidden from the sandbox proxy for every IACS URL; no retrievable full text. "
            "IACS rule text is a controlled publication, so even a successful anonymous fetch would "
            "not be an authorised copy for verbatim transcription"
        ),
    },
    {
        "channel": "operator-supplied copy",
        "action": "checked benchmarks/anchors/sources/ for an operator-deposited transcription",
        "result": "only sname_tr_bulletin_3-37_4-7.transcription.json (different topic); no CSR deposition",
    },
]

REPO_CANDIDATES = [
    {
        "key": "tensorfem_ur_s11_wave_moment",
        "provenance": "TensorFEM:src/tensorfem/hull_girder_csr_chain.py:rule_wave_bending_moment",
        "claimed_standard": "IACS UR S11 / CSR wave bending moment",
        "claimed_expression": "M_wh = 0.19 C L^2 B Cb [kN.m]; M_ws = 0.11 C L^2 B (Cb + 0.7) [kN.m]",
        "citation_quote": None,
        "table_or_clause": None,
        "page": None,
        "printed_values": None,
        "status": "UNCITED_CANDIDATE",
        "usable_for_compliance_claim": False,
        "note": (
            "the expression is implemented in this repository and used to build the wave-moment "
            "envelope of the W22 study, but it is not a transcription of the rule text: no clause, "
            "page or printed value string has been checked against a copy of UR S11"
        ),
    },
    {
        "key": "tensorfem_ur_s11_minimum_modulus",
        "provenance": "TensorFEM:src/tensorfem/hull_girder_csr_chain.py:ur_s11_minimum_section_modulus",
        "claimed_standard": "IACS UR S11 / UR S7 minimum midship section modulus",
        "claimed_expression": "W_min = C L^2 B (Cb + 0.7) [cm^3]",
        "citation_quote": None,
        "table_or_clause": None,
        "page": None,
        "printed_values": None,
        "status": "UNCITED_CANDIDATE",
        "usable_for_compliance_claim": False,
        "note": "used only to close the deck scantlings; not verified against the rule text",
    },
    {
        "key": "tensorfem_csr_gamma_m_and_permissible_stress",
        "provenance": "TensorFEM:src/tensorfem/hull_girder_csr_chain.py:CSR_GAMMA_M, CSR_PERMISSIBLE_STRESS",
        "claimed_standard": "CSR material factor gamma_M = 1.43 and permissible hull-girder stress 220 MPa",
        "citation_quote": None,
        "table_or_clause": None,
        "page": None,
        "printed_values": None,
        "status": "UNCITED_CANDIDATE",
        "usable_for_compliance_claim": False,
        "note": (
            "module constants attributed to CSR in the repository; the W22 study does NOT use them "
            "as CSR factor values"
        ),
    },
]


def build_register() -> dict:
    rows = []
    for entry in FACTOR_REGISTER:
        row = dict(entry)
        row.update({
            "citation_quote": None,
            "table_or_clause": None,
            "page": None,
            "printed_values": None,
            "status": "BLOCKED",
            "missing_pillars": [
                "authorised copy of the rule text not available in this environment",
                "citation_quote",
                "table_or_clause",
                "page",
                "printed_values",
            ],
            "usable_for_compliance_claim": False,
            "consequence": (
                "utilization is reported at unit factors and as the admissible factor "
                "M_u/(M_sw+M_wv); no CSR compliance claim is made"
            ),
        })
        rows.append(row)
    verbatim = sum(1 for row in rows if row["status"] == "VERBATIM")
    return {
        "schema_version": "1.0.0",
        "artifact": "anchors/iacs-csr-w22",
        "topic": "w22_hull_girder",
        "mission_id": 14512,
        "repository": "TensorFEM",
        "generated_by": "benchmarks/anchors/iacs_csr_w22_anchor.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "document": DOCUMENT,
        "acquisition_status": "NOT_OBTAINED",
        "acquisition_attempts": ACQUISITION_ATTEMPTS,
        "factor_register": rows,
        "completeness": {
            "required_records": len(rows),
            "verbatim_records": verbatim,
            "complete_rate": verbatim / len(rows),
            "criterion_C1_threshold": 1.0,
            "criterion_C1_verdict": "PASS" if verbatim == len(rows) else "FAIL",
        },
        "repo_candidates": REPO_CANDIDATES,
        "consequences_for_the_study": [
            "M_sw and M_wv are computed (homogeneous-loaded still water + UR S11 closed form implemented in this repository) and tabled in N.m.",
            "M_u comes from the Smith incremental collapse engine, which does not depend on the rule text.",
            "the ULS inequality is evaluated at unit factors (gamma_s = gamma_w = 1) and the admissible factor M_u/(M_sw+M_wv) is reported so that any factor table can be applied without this file inventing numbers.",
            "the yield criterion is reported as sigma and as sigma_y/sigma (the admissible gamma_R).",
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "benchmarks" / "anchors" / "iacs-csr-w22.json"
    register = build_register()
    out.write_text(json.dumps(register, indent=2) + "\n")
    print(f"wrote {out} ({register['completeness']['complete_rate']:.0%} verbatim)")


if __name__ == "__main__":
    main()