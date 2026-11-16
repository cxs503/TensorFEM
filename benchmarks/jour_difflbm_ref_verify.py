#!/usr/bin/env python3
"""JOUR-DIFFLBM-REF-VERIFY — citation gate + code-backed forcing benchmark.

Entry point of the anchor ``JOUR-DIFFLBM-REF-VERIFY``
(acceptance: 立项引用逐条 DOI 核证 —— 题名/作者/年/DOI 一致；不一致者改引或
降级为"未核证").

Two legs, both machine-checkable:

1. **Citation leg** — every numbered reference of the two JOUR-DIFFLBM fluid
   modules (``src/tensorfem/lbm_cavity_diff.py``, ``src/tensorfem/lbm_channel2d.py``)
   is parsed out of its docstring and verified field by field (title / authors
   / year / venue / volume / pages / DOI) against
   :mod:`tensorfem.lbm_references`.  Chimeric citations — right title, someone
   else's venue/volume/pages/DOI — are detected and reported, not tolerated.
   ``--online`` additionally resolves every registry DOI through Crossref
   (a 404 is a *result*: the withdrawn handle must stay unregistered).

2. **Forcing leg** — the reference of the forcing scheme must be code-backed:
   :func:`tensorfem.lbm_forcing.poiseuille_benchmark` drives (a) a reference
   D2Q9 implementation with the Guo source term and with the uncorrected
   scheme, and (b) the shipping solver
   :class:`tensorfem.lbm_channel2d.ChannelLBM`, against the analytic
   force-driven Poiseuille profile.  ``ref`` vs ``got`` numbers, gate < 3 %.

Exit status is the gate: 0 only when the citation audit passes and the
benchmark gate passes (``--no-bench`` runs the citation leg alone).

Usage
-----
    PYTHONPATH=src python benchmarks/jour_difflbm_ref_verify.py
    PYTHONPATH=src python benchmarks/jour_difflbm_ref_verify.py --online
    PYTHONPATH=src python benchmarks/jour_difflbm_ref_verify.py --quick
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorfem import lbm_references as refs                     # noqa: E402
from tensorfem.lbm_forcing import (                             # noqa: E402
    benchmark_passed,
    poiseuille_benchmark,
)

#: sources whose reference blocks belong to this anchor
SOURCES = (
    "src/tensorfem/lbm_cavity_diff.py",
    "src/tensorfem/lbm_channel2d.py",
)


def citation_leg(online: bool, cached_live: dict | None = None) -> dict:
    """Audit every reference of :data:`SOURCES`; optionally go to Crossref."""
    live: dict[str, dict] = dict(cached_live or {})
    if online:
        for ref in refs.REFS:
            live[ref.doi] = refs.resolve_doi_live(ref.doi)
        for doi in refs.WITHDRAWN_DOIS:
            live[doi] = refs.resolve_doi_live(doi)

    sources = {}
    all_ok = True
    for src in SOURCES:
        checks = refs.audit_source(REPO / src, live=live or None)
        summary = refs.audit_summary(checks)
        summary["file"] = src
        summary["rendered_registry_block"] = refs.render_reference_block(
            tuple(c.matched_key for c in checks if c.matched_key))
        # docstring must agree with the registry-rendered block
        module = __import__(
            "tensorfem." + pathlib.Path(src).stem, fromlist=["__references__"])
        summary["module_references_in_sync"] = all(
            refs.check_citation(line.split("] ", 1)[1]).ok
            for line in module.__references__.splitlines())
        sources[src] = summary
        all_ok = all_ok and summary["passed"] and \
            summary["module_references_in_sync"]

    evidence = {}
    for ref in refs.REFS:
        ev = live.get(ref.doi) or refs.evidence_for(ref.doi) or {}
        evidence[ref.key] = {
            "doi": ref.doi,
            "status": ref.status,
            "source": "crossref-live" if ref.doi in live else
                      f"crossref-baked:{refs.CROSSREF_FETCHED}",
            "http": ev.get("http"),
            "fields": refs.evidence_matches_registry(ref.doi, ev),
        }
        if not all(evidence[ref.key]["fields"].values()):
            all_ok = False

    # withdrawn handles must NOT resolve
    withdrawn = {}
    for doi, reason in refs.WITHDRAWN_DOIS.items():
        ev = live.get(doi)
        stays_dead = True if ev is None else int(ev.get("http") or 0) != 200
        withdrawn[doi] = {
            "reason": reason,
            "http": None if ev is None else ev.get("http"),
            "still_unregistered": stays_dead,
            "probe": "not probed (offline)" if ev is None else "crossref",
        }
        all_ok = all_ok and stays_dead

    return {
        "anchor_slug": "JOUR-DIFFLBM-REF-VERIFY",
        "mode": "online" if online else "offline(registry evidence baked "
                                        f"from Crossref {refs.CROSSREF_FETCHED})",
        "sources": sources,
        "registry_evidence": evidence,
        "withdrawn_dois": withdrawn,
        "passed": bool(all_ok),
    }


def forcing_leg(quick: bool = False) -> dict:
    """Guo forcing benchmark on the reference and production solvers."""
    result = poiseuille_benchmark(
        resolutions=(10, 18) if quick else (10, 18, 34),
        guo_only_resolutions=() if quick else (66,),
        tau_sweep=(0.05, 0.2) if quick else (0.05, 0.2, 0.5),
        include_production=not quick,
    )
    result["gate"]["passed"] = benchmark_passed(result)
    result["reference_registry_key"] = "guo2002_forcing"
    result["reference_status"] = refs.REFS_BY_KEY["guo2002_forcing"].status
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--online", action="store_true",
                    help="resolve every DOI through Crossref (network)")
    ap.add_argument("--quick", action="store_true",
                    help="reduced grid ladder / viscosity sweep")
    ap.add_argument("--no-bench", action="store_true",
                    help="citation leg only")
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "artifacts" / "JOUR-DIFFLBM-REF-VERIFY")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cit = citation_leg(args.online)
    (args.out / "ref_verify.json").write_text(
        json.dumps(cit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[citation] mode={cit['mode']}")
    for src, summary in cit["sources"].items():
        print(f"  {src}: {summary['n_ok']}/{summary['n_entries']} entries ok, "
              f"registry-in-sync={summary['module_references_in_sync']}")
        for entry in summary["entries"]:
            flag = "OK  " if entry["ok"] else "FAIL"
            print(f"    {flag} {entry['matched_key']} doi={entry['doi']} "
                  f"registered={entry['doi_registered']}")
            for problem in entry["problems"]:
                print(f"         ! {problem}")
    for doi, info in cit["withdrawn_dois"].items():
        print(f"  withdrawn {doi}: still_unregistered="
              f"{info['still_unregistered']} ({info['probe']})")
    for key, ev in cit["registry_evidence"].items():
        bad = [k for k, v in ev["fields"].items() if not v]
        print(f"  registry {key}: http={ev['http']} "
              f"source={ev['source']} mismatched={bad or 'none'}")

    passed = cit["passed"]
    if not args.no_bench:
        bench = forcing_leg(args.quick)
        (args.out / "forcing_poiseuille.json").write_text(
            json.dumps(bench, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print("[forcing] " + bench["reference"])
        for row in bench["rows"]:
            print(f"  ny={row['ny']:3d} {row['solver']:22s} "
                  f"{row['scheme']:5s} ref={row['u_max_ref']:.6f} "
                  f"got={row['u_max_got']:.6f} "
                  f"err={100 * row['rel_err_u_max']:7.4f}% "
                  f"L2={100 * row['rel_l2_profile']:7.4f}%")
        order = bench.get("convergence_order") or {}
        if order:
            print(f"  observed order (guo): {order['mean_order']:.3f} "
                  f"@{order['resolutions']}")
        for entry in bench["tau_sweep"]:
            print(f"  tau={entry['tau']:.2f} naive_amplitude_ratio="
                  f"{entry['naive_amplitude_ratio']:.4f} "
                  f"(1+1/2tau={entry['amplitude_law']:.4f}) "
                  f"guo_amplitude_ratio={entry['guo_amplitude_ratio']:.4f}")
        print(f"  gate passed={bench['gate']['passed']}")
        passed = passed and bench["gate"]["passed"]

    print(f"[gate] {'PASS' if passed else 'FAIL'} "
          f"({time.time() - t0:.1f} s, artifacts in {args.out})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())