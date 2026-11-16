"""Citation-integrity gate for JOUR-DIFFLBM-REF-VERIFY (registry-based).

Anchor ``JOUR-DIFFLBM-REF-VERIFY`` — 立项引用逐条 DOI 核证。The first delivery
guarded the DOIs with exact-substring assertions, which a *wrong* citation
passes as long as the strings are present: the Guo-Zheng-Shi (2002) entry
carried the title of Phys. Rev. E 65, 046308 with the venue/volume/pages/DOI
of Phys. Fluids 14, 2007-2010 (10.1063/1.1471914) and an unregistered handle
(10.1063/1.1474820), and the guard was green.

These tests assert the *invariants* the acceptance criterion needs:

* every reference of the two JOUR-DIFFLBM fluid modules matches its registry
  record field by field (title / authors / year / venue / volume / pages / DOI),
* the registry record itself matches the Crossref evidence,
* the withdrawn handle stays unregistered and can never pass a check again,
* punctuation/spacing variants of a bad citation are rejected too (the
  loophole that let the substring guard be bypassed),
* the docstring block equals the registry-rendered block (``__references__``).

No network is used: the Crossref evidence is baked into the registry
(:data:`tensorfem.lbm_references.CROSSREF_EVIDENCE`).
"""
from __future__ import annotations

import pathlib

import pytest

from tensorfem import lbm_references as R

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCES = (
    "src/tensorfem/lbm_cavity_diff.py",
    "src/tensorfem/lbm_channel2d.py",
)

# ---------------------------------------------------------------------------
# the exact citation m22758 shipped (used as the negative control)
# ---------------------------------------------------------------------------
M22758_GUO_CITATION = (
    "Guo, Z., Zheng, C., Shi, B. (2002). Discrete lattice effects on the "
    "forcing term in the lattice Boltzmann method. Phys. Fluids 14, 2007-2010. "
    "DOI: 10.1063/1.1474820 (forcing).  [REF-VERIFY: verified.]"
)


def _entries(rel: str) -> list[str]:
    return R.parse_reference_entries((REPO / rel).read_text(encoding="utf-8"))


@pytest.mark.parametrize("source", SOURCES)
def test_reference_block_has_entries(source: str) -> None:
    # guard against a silently empty parse (which would make every other
    # assertion below vacuous)
    assert len(_entries(source)) >= 3


@pytest.mark.parametrize("source", SOURCES)
def test_every_reference_verifies_against_the_registry(source: str) -> None:
    checks = R.audit_source(REPO / source)
    assert checks, "no reference entry parsed"
    for check in checks:
        assert check.matched_key is not None, check.problems
        assert check.fields, f"{check.matched_key}: no field was compared"
        assert all(check.fields.values()), (
            f"{check.matched_key}: {check.problems}")
        assert check.chimera_of is None, check.problems
        assert check.doi_registered is True, check.problems
        assert check.ok, check.problems


@pytest.mark.parametrize("source", SOURCES)
def test_docstring_matches_registry_rendered_block(source: str) -> None:
    module = __import__("tensorfem." + pathlib.Path(source).stem,
                        fromlist=["__references__"])
    rendered = R.parse_reference_entries(
        "References\n----------\n" + module.__references__)
    assert len(rendered) == len(_entries(source))
    for line in module.__references__.splitlines():
        check = R.check_citation(line.split("] ", 1)[1])
        assert check.ok, check.problems


def test_registry_matches_crossref_evidence() -> None:
    for ref in R.REFS:
        ev = R.evidence_for(ref.doi)
        assert ev is not None, f"{ref.key}: no recorded Crossref evidence"
        fields = R.evidence_matches_registry(ref.doi, ev)
        assert fields.pop("registered") is True, ref.key
        assert all(fields.values()), f"{ref.key}: {fields}"


def test_withdrawn_doi_is_not_registered() -> None:
    doi = "10.1063/1.1474820"
    assert doi in R.WITHDRAWN_DOIS
    ev = R.evidence_for(doi)
    assert ev is not None and int(ev["http"]) == 404
    # and it is not a registry citation any more
    assert all(R.normalise_doi(r.doi) != doi for r in R.REFS)


def test_m22758_chimera_citation_is_rejected() -> None:
    """The citation m22758 graded 'verified' must fail — twice over."""
    check = R.check_citation(M22758_GUO_CITATION)
    assert not check.ok
    assert check.matched_key == "guo2002_forcing"      # right title
    assert check.chimera_of == "guo2002_extrapolation"  # someone else's venue
    assert check.doi_registered is False                # 404
    assert check.fields["venue"] is False
    assert check.fields["doi"] is False


@pytest.mark.parametrize("variant", [
    "J. Supercomput. 4, 149-168",
    "J. Supercomput 4, 149",
    "J. Supercomput. 4:149",
    "j supercomput 4 149",
    "J--Supercomput, 4, 149",
])
def test_substring_loophole_variants_are_rejected(variant: str) -> None:
    """The old guard tested one literal string; every variant of the same
    fabricated citation must be rejected by the registry check."""
    docstring = (
        '"""Tiny module.\n\nReferences\n----------\n'
        f"[1] Ladd, A.J. (1994). moving-wall bounce-back. {variant}.\n"
        '"""\n'
    )
    checks = R.audit_text(docstring)
    assert checks and not checks[0].ok


def test_corrected_ladd_citation_passes() -> None:
    entry = ("Ladd, A.J.C. (1994). Numerical simulations of particulate "
             "suspensions via a discretized Boltzmann equation. Part 1. "
             "Theoretical foundation. J. Fluid Mech. 271, 285-309. "
             "DOI: 10.1017/S0022112094001771")
    assert R.check_citation(entry).ok


def test_ghia_citation_with_parenthesised_doi_parses() -> None:
    entry = ("Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for "
             "incompressible flow using the Navier-Stokes equations and a "
             "multigrid method. J. Comput. Phys. 48, 387-411. "
             "DOI: 10.1016/0021-9991(82)90058-4")
    check = R.check_citation(entry)
    assert check.doi == "10.1016/0021-9991(82)90058-4"
    assert check.ok, check.problems


def test_doi_handle_structure() -> None:
    for ref in R.REFS:
        assert R.doi_handle_ok(ref.doi), ref.doi
    assert not R.doi_handle_ok("10.1063/.")
    assert not R.doi_handle_ok("not-a-doi")


def test_year_and_author_mismatch_is_caught() -> None:
    """A record with the right DOI but the wrong year must not pass."""
    entry = ("Guo, Z., Zheng, C., Shi, B. (1999). Discrete lattice effects on "
             "the forcing term in the lattice Boltzmann method. "
             "Phys. Rev. E 65, 046308. DOI: 10.1103/PhysRevE.65.046308")
    check = R.check_citation(entry)
    assert check.fields["year"] is False
    assert not check.ok