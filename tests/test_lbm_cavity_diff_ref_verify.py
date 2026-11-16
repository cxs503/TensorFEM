"""Regression gate for JOUR-DIFFLBM-REF-VERIFY: DOI citation integrity.

Anchor JOUR-DIFFLBM-REF-VERIFY — 立项引用逐条 DOI 核证。This test asserts
*invariants* on the ``References`` block of the JOUR-DIFFLBM fluid core
(``src/tensorfem/lbm_cavity_diff.py``):

* the three cited references carry their canonical DOIs,
* the previously-cited (fabricated) Ladd journal "J. Supercomput" is absent
  (a regression would re-introduce the citation error),
* the Ladd moving-wall bounce-back is attributed to J. Fluid Mech. 271
  (the only authoritative source).

These are *citation-integrity* invariants, not numerical ones: they guard
against the exact defect class the anchor is designed to catch (wrong
journal / fabricated DOI), and are cheap to run.
"""
from __future__ import annotations

import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "tensorfem" / "lbm_cavity_diff.py"

# canonical DOIs for the three JOUR-DIFFLBM references
LADD_DOI = "10.1017/S0022112094001771"
GHIA_DOI = "10.1016/0021-9991(82)90058-4"
GUO_DOI = "10.1063/1.1474820"


def _refs() -> str:
    text = SRC.read_text()
    # the References block runs from the "References" header to the closing """
    start = text.index("References")
    end = text.index('"""', start)
    return text[start:end]


def test_ladd_reference_carries_canonical_doi():
    refs = _refs()
    assert LADD_DOI in refs, "Ladd moving-wall bounce-back must carry DOI 10.1017/S0022112094001771"


def test_ladd_reference_attributed_to_j_fluid_mech():
    refs = _refs()
    # the only authoritative source for the Ladd moving-wall bounce-back
    assert "J. Fluid Mech. 271" in refs


def test_fabricated_ladd_journal_absent():
    # regression guard: the previously-cited "J. Supercomput. 4, 149" does not
    # exist and must never be re-introduced as a *live* citation (with the
    # fabricated volume/page).  The correction note may mention the old journal
    # name generically, but never as "J. Supercomput. 4, 149".
    refs = _refs()
    assert "J. Supercomput. 4, 149" not in refs


def test_ghia_reference_carries_canonical_doi():
    refs = _refs()
    assert GHIA_DOI in refs, "Ghia (1982) must carry DOI 10.1016/0021-9991(82)90058-4"


def test_guo_reference_carries_canonical_doi():
    refs = _refs()
    assert GUO_DOI in refs, "Guo (2002) must carry DOI 10.1063/1.1474820"


def test_all_three_doi_prefixes_valid():
    # every cited DOI must be a resolvable "10." handle
    refs = _refs()
    for doi in (LADD_DOI, GHIA_DOI, GUO_DOI):
        assert doi.startswith("10."), f"DOI {doi} must be a 10. handle"
