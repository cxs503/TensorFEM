"""Canonical bibliographic registry for the JOUR-DIFFLBM citation set.

Anchor: ``JOUR-DIFFLBM-REF-VERIFY`` (总账 acceptance: 立项引用逐条 DOI 核证 ——
题名/作者/年/DOI 一致；不一致者改引或降级为"未核证").

Why a module and not a paragraph of prose
-----------------------------------------
The first delivery of this anchor (m22758) put the DOIs in a docstring and
guarded them with an exact-substring test.  A string guard cannot tell a
correct DOI from a *registered-looking* one: the citation of Guo-Zheng-Shi
(2002) carried the title of Phys. Rev. E **65**, 046308 while its venue,
volume, pages and DOI belonged to a completely different paper (Phys. Fluids
**14**, 2007-2010 = 10.1063/1.1471914).  Both strings were "present", so the
test was green and the citation was still wrong.

This module replaces prose by executable data:

* :data:`REFS` — every reference of the JOUR-DIFFLBM fluid core, with the
  full bibliographic tuple (authors / year / title / venue / volume / pages /
  DOI) and the *external evidence* the tuple was checked against (Crossref,
  fetched live, see :data:`CROSSREF_FETCHED`).
* :data:`WITHDRAWN_DOIS` — DOIs that must never be cited again, with the
  reason they were withdrawn (Crossref HTTP 404 / superseded).
* :func:`check_citation` — field-by-field verification of a *free-text*
  citation against the registry, including **cross-contamination detection**:
  a citation whose title resolves to one record while its venue/volume/pages
  resolve to a different record is reported as a chimera instead of passing.
* :func:`audit_source` — parse the ``References`` block out of a Python
  source file and verify every entry, returning machine-readable findings.
* :func:`resolve_doi_live` — optional live Crossref resolution (network), used
  by the benchmark entry point; the registry checks themselves are offline.

Everything here is pure Python + NumPy-free stdlib so the citation gate can
run in the smallest possible environment (no FEM, no LBM, no network).
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = [
    "Reference",
    "RefCheck",
    "REFS",
    "REFS_BY_KEY",
    "WITHDRAWN_DOIS",
    "CROSSREF_FETCHED",
    "JOUR_DIFFLBM_CORE_REFS",
    "normalise",
    "normalise_doi",
    "doi_handle_ok",
    "parse_reference_entries",
    "check_citation",
    "audit_source",
    "audit_text",
    "render_reference_block",
    "resolve_doi_live",
]

#: date of the live Crossref resolution recorded in :data:`CROSSREF_EVIDENCE`
CROSSREF_FETCHED = "2026-11-16"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Reference:
    """One verified bibliography entry.

    ``status`` is the verification verdict of the *acceptance criterion*:

    * ``verified``   — title/authors/year/venue/volume/pages/DOI all agree
      with the external record returned by Crossref for ``doi``.
    * ``corrected``  — the citation was wrong in an earlier revision of this
      repository and the tuple here is the corrected one (the note records
      what was wrong).
    * ``unregistered`` — the DOI does not resolve; the record is kept only so
      that the withdrawn handle can be recognised and refused.
    """

    key: str
    authors: tuple[tuple[str, str], ...]   # (family, given-initials)
    year: int
    title: str
    venue: str                             # full container title
    venue_abbrev: str                      # form used in the docstrings
    volume: str
    pages: str
    doi: str
    status: str = "verified"
    note: str = ""
    used_by: tuple[str, ...] = ()          # code that relies on this entry
    #: raw external evidence (Crossref message fields), filled from
    #: :data:`CROSSREF_EVIDENCE`
    evidence: Mapping[str, object] = field(default_factory=dict)

    @property
    def first_author(self) -> str:
        return self.authors[0][0]

    @property
    def cites_ok(self) -> bool:
        return self.status in ("verified", "corrected")

    def bib_line(self) -> str:
        """Single-line rendering used inside module docstrings."""
        auth = "; ".join(f"{fam}, {giv}" for fam, giv in self.authors)
        return (
            f"{auth} ({self.year}). {self.title}. "
            f"{self.venue_abbrev} {self.volume}, {self.pages}. "
            f"DOI: {self.doi}"
        )


#: Crossref records captured live on :data:`CROSSREF_FETCHED`.  Only the
#: fields the verification actually compares are kept; the HTTP status is part
#: of the evidence because a 404 is what proves a DOI was never registered.
CROSSREF_EVIDENCE: dict[str, dict[str, object]] = {
    "10.1017/S0022112094001771": {
        "http": 200,
        "title": "Numerical simulations of particulate suspensions via a "
                 "discretized Boltzmann equation. Part 1. Theoretical "
                 "foundation",
        "authors": [["Ladd", "Anthony J. C."]],
        "container": "Journal of Fluid Mechanics",
        "volume": "271",
        "page": "285-309",
        "year": 1994,
    },
    "10.1016/0021-9991(82)90058-4": {
        "http": 200,
        "title": "High-Re solutions for incompressible flow using the "
                 "Navier-Stokes equations and a multigrid method",
        "authors": [["Ghia", "U"], ["Ghia", "K.N"], ["Shin", "C.T"]],
        "container": "Journal of Computational Physics",
        "volume": "48",
        "page": "387-411",
        "year": 1982,
    },
    "10.1103/PhysRevE.65.046308": {
        "http": 200,
        "title": "Discrete lattice effects on the forcing term in the "
                 "lattice Boltzmann method",
        "authors": [["Guo", "Zhaoli"], ["Zheng", "Chuguang"], ["Shi", "Baochang"]],
        "container": "Physical Review E",
        "volume": "65",
        "page": "046308",
        "year": 2002,
    },
    "10.1063/1.1471914": {
        "http": 200,
        "title": "An extrapolation method for boundary conditions in lattice "
                 "Boltzmann method",
        "authors": [["Guo", "Zhaoli"], ["Zheng", "Chuguang"], ["Shi", "Baochang"]],
        "container": "Physics of Fluids",
        "volume": "14",
        "page": "2007-2010",
        "year": 2002,
    },
    "10.1017/S0962492902000077": {
        "http": 200,
        "title": "The immersed boundary method",
        "authors": [["Peskin", "Charles S."]],
        "container": "Acta Numerica",
        "volume": "11",
        "page": "479-517",
        "year": 2002,
    },
    "10.1007/3-540-34596-5_15": {
        "http": 200,
        "title": "Proposal for Numerical Benchmarking of Fluid-Structure "
                 "Interaction between an Elastic Object and Laminar "
                 "Incompressible Flow",
        "authors": [["Turek", "Stefan"], ["Hron", "Jaroslav"]],
        "container": "Lecture Notes in Computational Science and Engineering",
        "volume": None,      # book series volume 53 is not in the Crossref record
        "page": "371-385",
        "year": None,        # book chapter record carries no issued year
    },
    "10.1063/1.1474820": {
        "http": 404,
        "error": "Resource not found (Crossref AND doi.org). The handle is "
                 "not registered by any publisher.",
    },
}

REFS: tuple[Reference, ...] = (
    Reference(
        key="ladd1994",
        authors=(("Ladd", "A.J.C."),),
        year=1994,
        title="Numerical simulations of particulate suspensions via a "
              "discretized Boltzmann equation. Part 1. Theoretical foundation",
        venue="Journal of Fluid Mechanics",
        venue_abbrev="J. Fluid Mech.",
        volume="271",
        pages="285-309",
        doi="10.1017/S0022112094001771",
        status="corrected",
        note="moving-wall (Ladd) half-way bounce-back. An earlier revision of "
             "the JOUR-DIFFLBM core cited 'J. Supercomput. 4, 149' — that "
             "journal/volume/page combination does not exist and the citation "
             "was replaced by the authoritative J. Fluid Mech. record.",
        used_by=("src/tensorfem/lbm_cavity_diff.py:_stream (lid closure)",
                 "src/tensorfem/lbm_channel2d.py:_wall_links (wall closure)"),
    ),
    Reference(
        key="ghia1982",
        authors=(("Ghia", "U."), ("Ghia", "K.N."), ("Shin", "C.T.")),
        year=1982,
        title="High-Re solutions for incompressible flow using the "
              "Navier-Stokes equations and a multigrid method",
        venue="Journal of Computational Physics",
        venue_abbrev="J. Comput. Phys.",
        volume="48",
        pages="387-411",
        doi="10.1016/0021-9991(82)90058-4",
        status="verified",
        note="lid-driven cavity benchmark tables (Re = 100 centre-line "
             "velocities) used by examples/lbm_cavity_diff_validation.py.",
        used_by=("examples/lbm_cavity_diff_validation.py:GHIA_*",),
    ),
    Reference(
        key="guo2002_forcing",
        authors=(("Guo", "Z."), ("Zheng", "C."), ("Shi", "B.")),
        year=2002,
        title="Discrete lattice effects on the forcing term in the lattice "
              "Boltzmann method",
        venue="Physical Review E",
        venue_abbrev="Phys. Rev. E",
        volume="65",
        pages="046308",
        doi="10.1103/PhysRevE.65.046308",
        status="corrected",
        note="the discrete-lattice-effect-corrected forcing scheme. An earlier "
             "revision stitched this record's title/authors/year onto the "
             "venue/volume/pages/DOI of guo2002_extrapolation "
             "(Phys. Fluids 14, 2007-2010, 10.1063/1.1471914) and quoted the "
             "unregistered handle 10.1063/1.1474820 — a chimera citation.",
        used_by=("src/tensorfem/lbm_channel2d.py:_collide (body_force branch)",
                 "src/tensorfem/lbm_forcing.py (Guo source term + benchmark)"),
    ),
    Reference(
        key="guo2002_extrapolation",
        authors=(("Guo", "Z."), ("Zheng", "C."), ("Shi", "B.")),
        year=2002,
        title="An extrapolation method for boundary conditions in lattice "
              "Boltzmann method",
        venue="Physics of Fluids",
        venue_abbrev="Phys. Fluids",
        volume="14",
        pages="2007-2010",
        doi="10.1063/1.1471914",
        status="verified",
        note="velocity/pressure extrapolation boundary conditions. Registered "
             "here because it is the paper the withdrawn chimera citation "
             "actually pointed at; cited only where a boundary closure is "
             "implemented.",
        used_by=(),
    ),
    Reference(
        key="peskin2002",
        authors=(("Peskin", "C.S."),),
        year=2002,
        title="The immersed boundary method",
        venue="Acta Numerica",
        venue_abbrev="Acta Numer.",
        volume="11",
        pages="479-517",
        doi="10.1017/S0962492902000077",
        status="verified",
        note="Peskin 4-point delta kernel used for Lagrangian marker exchange.",
        used_by=("src/tensorfem/lbm_channel2d.py:peskin_delta",),
    ),
    Reference(
        key="turek2006",
        authors=(("Turek", "S."), ("Hron", "J.")),
        year=2006,
        title="Proposal for numerical benchmarking of fluid-structure "
              "interaction between an elastic object and laminar "
              "incompressible flow",
        venue="Lecture Notes in Computational Science and Engineering",
        venue_abbrev="LNCS",
        volume="53",
        pages="371-385",
        doi="10.1007/3-540-34596-5_15",
        status="verified",
        note="FSI benchmark 2D-2 / FSI-2. Crossref returns the chapter title "
             "(case differs) and pages 371-385 but no series volume/year "
             "fields, so volume 53 / year 2006 come from the book "
             "(LNCSE 53) and are not part of the machine comparison.",
        used_by=("src/tensorfem/lbm_channel2d.py (FSI-2 benchmark face)",),
    ),
)

REFS_BY_KEY: dict[str, Reference] = {r.key: r for r in REFS}

#: DOIs that must never appear in a live citation again, with the verdict.
WITHDRAWN_DOIS: dict[str, str] = {
    "10.1063/1.1474820": (
        "NOT REGISTERED (Crossref HTTP 404, "
        f"checked {CROSSREF_FETCHED}). The handle was used in m22758 for "
        "'Guo, Zheng, Shi (2002), Discrete lattice effects on the forcing "
        "term..., Phys. Fluids 14, 2007-2010': the title belongs to "
        "10.1103/PhysRevE.65.046308 (Phys. Rev. E 65, 046308) and the venue/"
        "volume/pages belong to 10.1063/1.1471914 (Phys. Fluids 14, "
        "2007-2010)."
    ),
}

#: references of the JOUR-DIFFLBM fluid core, in citation order
JOUR_DIFFLBM_CORE_REFS: tuple[str, ...] = (
    "ladd1994",
    "ghia1982",
    "guo2002_forcing",
)


# ---------------------------------------------------------------------------
# normalisation helpers — the *whole point* is that punctuation and spacing
# variants of the same (wrong) citation must compare equal, so a string guard
# cannot be bypassed by writing "J. Supercomput 4:149".
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"[^0-9a-z]+")


def normalise(text: str) -> str:
    """Lowercase, strip accents-free punctuation, collapse whitespace."""
    return _PUNCT.sub(" ", text.lower()).strip()


def normalise_doi(doi: str) -> str:
    """Canonical DOI form: lower case, no ``doi:``/URL prefix, no trailing dot."""
    d = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                   "doi:"):
        if d.lower().startswith(prefix):
            d = d[len(prefix):]
    return d.strip().rstrip(".,;)").lower()


def doi_handle_ok(doi: str) -> bool:
    """Structural check only (a ``10.NNNN/…`` handle); says nothing about
    registration — that is what :func:`resolve_doi_live` is for."""
    return bool(re.fullmatch(r"10\.\d{4,9}/\S+", normalise_doi(doi)))


#: DOI -> Reference, keyed by the normalised handle
REFS_BY_DOI: dict[str, Reference] = {normalise_doi(r.doi): r for r in REFS}
#: DOI -> Crossref evidence, keyed by the normalised handle
EVIDENCE_BY_DOI: dict[str, dict[str, object]] = {
    normalise_doi(d): e for d, e in CROSSREF_EVIDENCE.items()}


def evidence_for(doi: str) -> dict[str, object] | None:
    """Crossref evidence for ``doi`` (registry-baked), or ``None``."""
    return EVIDENCE_BY_DOI.get(normalise_doi(doi))


_MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()


def _venue_forms(ref: Reference) -> tuple[str, ...]:
    return (normalise(ref.venue), normalise(ref.venue_abbrev))


def _author_present(entry_norm: str, family: str) -> bool:
    return normalise(family) in entry_norm


# ---------------------------------------------------------------------------
# free-text citation checking
# ---------------------------------------------------------------------------
@dataclass
class RefCheck:
    """Result of verifying one free-text citation against the registry."""

    text: str
    matched_key: str | None = None
    doi: str | None = None
    doi_registered: bool | None = None
    fields: dict[str, bool] = field(default_factory=dict)
    offender: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    chimera_of: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.matched_key is not None
            and self.chimera_of is None
            and self.doi_registered is not False
            and all(self.fields.values())
            and not self.problems
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "text": " ".join(self.text.split())[:400],
            "matched_key": self.matched_key,
            "doi": self.doi,
            "doi_registered": self.doi_registered,
            "fields": dict(self.fields),
            "chimera_of": self.chimera_of,
            "problems": list(self.problems),
            "ok": self.ok,
        }


def _find_doi(entry: str) -> str | None:
    # DOIs may themselves contain parentheses (e.g. 10.1016/0021-9991(82)90058-4),
    # so take the whole non-space run and only strip trailing sentence
    # punctuation that cannot end a handle.
    m = re.search(r"10\.\d{4,9}/\S+", entry)
    return normalise_doi(m.group(0).rstrip(".,;")) if m else None


def _identify(entry: str) -> Reference | None:
    """Which registry entry is this *title* about?  Title is the primary key
    because a chimera citation keeps the right title and the wrong venue."""
    norm = normalise(entry)
    best: tuple[int, Reference] | None = None
    for ref in REFS:
        # score by the length of the longest matching leading title fragment
        words = normalise(ref.title).split()
        score = 0
        for start in (0,):
            run = 0
            for w in words[start:]:
                if normalise(w) and normalise(w) in norm:
                    run += 1
                else:
                    break
            score = max(score, run)
        # a distinctive tail phrase also identifies the paper
        tail = " ".join(normalise(ref.title).split()[-4:])
        if tail and tail in norm:
            score = max(score, len(words))
        if best is None or score > best[0]:
            best = (score, ref)
    if best is None or best[0] < 4:
        return None
    return best[1]


def check_citation(entry: str, live: Mapping[str, object] | None = None
                   ) -> RefCheck:
    """Verify one free-text citation entry field by field.

    Parameters
    ----------
    entry
        A citation as it appears in a docstring, e.g.
        ``"Guo, Z., Zheng, C., Shi, B. (2002). ... Phys. Fluids 14, 2007-2010.
        DOI: 10.1063/1.1474820"``.
    live
        Optional ``{doi: crossref message}`` mapping (from
        :func:`resolve_doi_live`) that overrides the baked evidence, so the
        same checker can be driven by a live Crossref run.
    """
    text = " ".join(entry.split())
    norm = normalise(text)
    res = RefCheck(text=text)
    ref = _identify(text)
    if ref is None:
        res.problems.append("citation does not match any registered reference "
                            "title (unverified / not in registry)")
        res.matched_key = None
        res.doi = _find_doi(text)
        return res
    res.matched_key = ref.key

    # --- DOI ---------------------------------------------------------------
    doi = _find_doi(text)
    res.doi = doi
    if doi is None:
        res.problems.append("no DOI found in entry")
        res.doi_registered = None
    else:
        ev = None
        if live:
            ev = next((v for k, v in live.items()
                       if normalise_doi(k) == doi), None)
        if ev is None:
            ev = evidence_for(doi)
        if doi in {normalise_doi(k) for k in WITHDRAWN_DOIS}:
            res.doi_registered = False
            res.problems.append(
                f"DOI {doi} is withdrawn: {WITHDRAWN_DOIS[doi]}")
        elif ev is None:
            res.doi_registered = None
            res.problems.append(f"DOI {doi} has no Crossref evidence in the "
                                "registry (cannot certify registration)")
        elif int(ev.get("http", 0)) != 200:
            res.doi_registered = False
            res.problems.append(f"DOI {doi} returned HTTP {ev.get('http')}")
        else:
            res.doi_registered = True
            ev_doi = normalise_doi(str(ev.get("doi") or doi))
            if ev_doi != doi:
                res.problems.append(
                    f"Crossref resolves {doi} to {ev_doi} (redirect/mismatch)")

    # --- per-field agreement with the identified record --------------------
    res.fields["year"] = str(ref.year) in text
    res.fields["title"] = all(
        w in norm for w in normalise(ref.title).split()[-6:])
    res.fields["authors"] = all(_author_present(norm, fam)
                                for fam, _ in ref.authors)
    venue_ok = any(v and v in norm for v in _venue_forms(ref))
    res.fields["venue"] = venue_ok
    res.fields["volume"] = bool(re.search(
        rf"(?<![\d.]){re.escape(ref.volume)}(?![\d.])", text))
    res.fields["pages"] = normalise(ref.pages).replace(" ", "") in \
        norm.replace(" ", "") or ref.pages in text
    expected_doi = normalise_doi(ref.doi)
    res.fields["doi"] = (doi == expected_doi) if doi else False

    for name, good in res.fields.items():
        if not good:
            res.problems.append(f"field '{name}' does not match the reference "
                                f"record {ref.key}")

    # --- cross-contamination (chimera) detection ---------------------------
    # If the entry's venue/volume/pages agree with a *different* registry
    # record than the one implied by its title, the citation mixes two papers.
    if not (venue_ok and res.fields["volume"] and res.fields["pages"]):
        for other in REFS:
            if other.key == ref.key:
                continue
            other_venue = any(v and v in norm for v in _venue_forms(other))
            other_vol = bool(re.search(
                rf"(?<![\d.]){re.escape(other.volume)}(?![\d.])", text))
            other_pages = other.pages in text
            if other_venue and other_vol and other_pages:
                res.chimera_of = other.key
                res.problems.append(
                    "chimera citation: title/authors belong to "
                    f"{ref.key} but venue/volume/pages belong to {other.key} "
                    f"({other.venue_abbrev} {other.volume}, {other.pages}, "
                    f"DOI {other.doi})")
                break
    return res


# ---------------------------------------------------------------------------
# reference-block parsing / source auditing
# ---------------------------------------------------------------------------
_ENTRY_RE = re.compile(r"(?m)^\[(\d+)\]\s")


def parse_reference_entries(text: str) -> list[str]:
    """Extract the numbered ``References`` entries of a module docstring.

    Robust to punctuation/indentation variants: it only relies on the
    ``[n] `` markers at line starts and on the section header.
    """
    idx = text.find("References")
    if idx < 0:
        return []
    section = text[idx:]
    # stop at the end of the docstring / at a trailing horizontal rule
    for stop in ('\n"""', "\n'''"):
        j = section.find(stop)
        if j > 0:
            section = section[:j]
    marks = list(_ENTRY_RE.finditer(section))
    entries = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(section)
        entries.append(section[m.end():end].strip())
    return entries


def audit_text(text: str, live: Mapping[str, object] | None = None
               ) -> list[RefCheck]:
    """Audit every numbered reference of a docstring / markdown block."""
    return [check_citation(e, live=live) for e in parse_reference_entries(text)]


def audit_source(path: str | "object", live: Mapping[str, object] | None = None
                 ) -> list[RefCheck]:
    """:func:`audit_text` applied to a file (str or :class:`pathlib.Path`)."""
    from pathlib import Path

    return audit_text(Path(path).read_text(encoding="utf-8"), live=live)


def render_reference_block(keys: Iterable[str]) -> str:
    """Render registry entries as a docstring-style numbered block.

    Production modules expose ``__references__ = render_reference_block(...)``
    so the citation text is *generated from* the verified registry instead of
    being retyped (and a test can then assert the docstring agrees with it).
    """
    lines = []
    for n, key in enumerate(keys, start=1):
        ref = REFS_BY_KEY[key]
        lines.append(f"[{n}] {ref.bib_line()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# optional live resolution (network) — used by the benchmark entry point
# ---------------------------------------------------------------------------
CROSSREF_URL = "https://api.crossref.org/works/"


def resolve_doi_live(doi: str, timeout: float = 15.0) -> dict[str, object]:
    """Resolve one DOI through Crossref.  Returns ``{"http": 200, ...}`` on
    success and ``{"http": <code or None>, "error": ...}`` otherwise; never
    raises for a missing DOI (a 404 is a *result*, not an exception)."""
    url = CROSSREF_URL + urllib.parse.quote(normalise_doi(doi))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            msg = json.load(fh).get("message", {})
    except urllib.error.HTTPError as exc:
        return {"http": exc.code, "error": str(exc)}
    except Exception as exc:                       # noqa: BLE001 - offline box
        return {"http": None, "error": repr(exc)}
    authors = [[a.get("family", ""), a.get("given", "")]
               for a in msg.get("author", [])]
    years = (msg.get("issued", {}) or {}).get("date-parts") or [[None]]
    return {
        "http": 200,
        "doi": msg.get("DOI"),
        "title": (msg.get("title") or [""])[0],
        "authors": authors,
        "container": (msg.get("container-title") or [""])[0],
        "volume": msg.get("volume"),
        "page": msg.get("page"),
        "year": years[0][0],
    }


def evidence_matches_registry(doi: str, ev: Mapping[str, object]) -> dict[str, bool]:
    """Compare a live Crossref record against the registry tuple, field by
    field.  Returns ``{field: bool}`` (a missing registry DOI yields all
    ``False`` with an extra ``'registered'`` key)."""
    nd = normalise_doi(doi)
    ref = next((r for r in REFS if normalise_doi(r.doi) == nd), None)
    if ref is None:
        return {"registered": False}
    out = {
        "registered": int(ev.get("http", 0) or 0) == 200,
        "title": normalise(str(ev.get("title", ""))) == normalise(ref.title),
    }
    # fields the Crossref record does not carry (None) cannot be certified by
    # it and are skipped rather than counted as a mismatch
    if ev.get("year") is not None:
        out["year"] = str(ev.get("year")) == str(ref.year)
    if ev.get("volume") is not None:
        out["volume"] = str(ev.get("volume")) == str(ref.volume)
    if ev.get("page") is not None:
        out["pages"] = str(ev.get("page") or "").strip() == ref.pages
    out["venue"] = normalise(str(ev.get("container", ""))) == normalise(ref.venue)
    fam = {normalise(f) for f, _ in ref.authors}
    ev_fam = {normalise(a[0]) for a in (ev.get("authors") or []) if a}
    out["authors"] = bool(fam) and fam == ev_fam
    return out


def audit_summary(checks: Sequence[RefCheck]) -> dict[str, object]:
    """Machine-readable summary used by the artifacts / reports."""
    return {
        "n_entries": len(checks),
        "n_ok": sum(1 for c in checks if c.ok),
        "passed": bool(checks) and all(c.ok for c in checks),
        "entries": [c.as_dict() for c in checks],
        "withdrawn_dois": dict(WITHDRAWN_DOIS),
        "crossref_fetched": CROSSREF_FETCHED,
        "registry_dois": {r.key: r.doi for r in REFS},
    }