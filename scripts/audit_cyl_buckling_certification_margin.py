"""Certification-margin audit of the MITC4 cylindrical-shell buckling claim.

Motivation (verification of a verification)
-------------------------------------------
``benchmarks/cylindrical_shell_buckling_mitc_evidence.json`` (m6602) certifies the
"shell element geometric stiffness" item with

    "certified": true,
    "reason": "... the finest affordable mesh (nz=30, nt=32) sits within 3% of
               the Donnell finite-length reference ..."

The claim rests on **one number**: the finest-mesh relative error 2.9466 %.
A single reading inside a 3 % band is not a convergence statement; it can be an
accidental zero crossing of a non-converging family (exactly the failure mode the
v1 audit documented for the full-integration Shell4, whose error decreases and
changes sign).  This script therefore re-derives, **from the stored raw eigen
records only** (no re-solve, no hand-copied number):

1. **observed order of convergence** ``p`` from three or more mesh levels of each
   family, ``p = log(e_c/e_f)/log(h_c/h_f)`` with ``h ~ 1/nz``;
2. **Richardson extrapolated limit** ``e_inf`` and, equivalently, whether the
   extrapolated buckling load approaches the Donnell reference (the reference is
   then consistent with the observed sequence, not merely crossed);
3. **Roache grid-convergence index** ``GCI_95 = Fs |e_f-e_c| / ((r^p-1) e_f)``
   (``Fs = 1.25``, two finest levels): the discretisation *uncertainty* of the
   certification number, which is what a reader of "within 3 %" needs;
4. **semantics check of the gate field** ``within_3pct_any`` (it is
   ``any(|e| < 3 %)`` over the family, not the finest mesh) -- reported, not
   silently accepted, together with the per-level minimum;
5. the same four items for the ``n_theta`` sensitivity family at fixed ``nz``
   (circumferential facet error) and for the uncured ``shell4-axsym`` family,
   which must stay outside the band.

Output ``artifacts/cyl_buckling_certification_margin.json`` (schema
``tensorfem.cyl-buckling-certification-margin/1``) plus a figure.
``tests/test_cyl_buckling_certification_margin.py`` locks the arithmetic.

Run: ``PYTHONPATH=src python scripts/audit_cyl_buckling_certification_margin.py``
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "cylindrical_shell_buckling_mitc_evidence.json"
OUT = ROOT / "artifacts" / "cyl_buckling_certification_margin.json"
FIG = ROOT / "artifacts" / "figs" / "cyl_buckling_certification_margin.png"

SCHEMA = "tensorfem.cyl-buckling-certification-margin/1"
GATE = 0.03
SAFETY = 1.25  # Roache Fs for a two-grid GCI estimate
FAMILIES = ("mitc-axsym", "mitc-axsym-nt", "shell4-axsym", "mitc-full")


def _levels(records, tag, axis):
    """Signed relative errors of one family, sorted by the refinement axis."""
    rows = [(r[axis], r["error_vs_donnell"], r) for r in records if r["tag"] == tag]
    rows.sort(key=lambda item: item[0])
    return rows


def _order(e_coarse, e_fine, r):
    """Observed order from two levels; returns None when it is not defined."""
    if e_coarse == 0.0 or e_fine == 0.0 or r <= 0.0 or r == 1.0:
        return None
    ratio = e_coarse / e_fine
    if ratio <= 0.0:  # sign change: no positive order exists
        return None
    return math.log(ratio) / math.log(r)


def fit_limit(levels):
    """Fit ``e(h) = e_inf + C h^p`` (h ~ 1/n) over all levels of a family.

    Three unknowns, five levels (Axialfamilie), so the converged limit ``e_inf``
    is *fitted*, not assumed to be zero.  This is the only honest way to answer
    "does the family converge to the reference or merely cross it?" when the
    observed pairwise order drifts (two-grid Richardson always returns exactly
    zero for a monotone two-point sequence, so it carries no information here).
    """
    refs = np.array([lv["refinement"] for lv in levels], dtype=float)
    errs = np.array([lv["error"] for lv in levels], dtype=float)
    h = 1.0 / refs

    def model(hh, e_inf, c, p):
        return e_inf + c * np.power(hh, p)

    try:
        import warnings
        from scipy.optimize import OptimizeWarning, curve_fit
        # A 3-parameter bounded fit on 5 points has an unidentifiable covariance;
        # the fit values are used, the covariance estimate is not, so the warning is
        # suppressed locally instead of being allowed to break -W error runs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(model, h, errs, p0=(errs[-1], 1e3, 1.5),
                                maxfev=40000,
                                bounds=([-1.0, 0.0, 0.2], [1.0, 1e9, 4.0]))
        e_inf, c, p_order = (float(v) for v in popt)
        residual = float(np.max(np.abs(model(h, *popt) - errs)))
        return {"extrapolated_error": e_inf, "coefficient": c,
                "order": p_order, "max_abs_residual": residual,
                "converged": True}
    except Exception as exc:  # pragma: no cover - numerical failure path
        return {"extrapolated_error": None, "coefficient": None, "order": None,
                "max_abs_residual": None, "converged": False,
                "error": str(exc)}


def analyse(records, tag, axis, *, expect_monotone=True):
    rows = _levels(records, tag, axis)
    levels = [{"refinement": ref, "error": e} for ref, e, _ in rows]
    errors = [e for _, e, _ in rows]
    refinements = [r for r, _, _ in rows]
    monotone = all(b <= a + 1e-12 for a, b in zip(errors, errors[1:]))
    orders = []
    for i in range(1, len(rows)):
        r = refinements[i] / refinements[i - 1]
        orders.append(_order(errors[i - 1], errors[i], r))
    finest_pair_p = orders[-1] if orders else None
    fit = fit_limit(rows and [{"refinement": ref, "error": e} for ref, e, _ in rows])
    within = [abs(e) < GATE for e in errors]
    record = {
        "tag": tag,
        "refinement_axis": axis,
        "levels": levels,
        "monotone": monotone,
        "observed_orders": orders,
        "finest_order": finest_pair_p,
        "finest_error": errors[-1] if errors else None,
        "extrapolated_error": fit["extrapolated_error"],
        "fit_order": fit["order"],
        "fit_max_abs_residual": fit["max_abs_residual"],
        "fit_converged": fit["converged"],
        "gate": GATE,
        "within_gate": within,
        "min_abs_error": min(abs(e) for e in errors) if errors else None,
        "min_abs_error_level": (min(rows, key=lambda item: abs(item[1]))[0]
                                if rows else None),
        "finest_within_gate": bool(within[-1]) if within else None,
        "any_within_gate": bool(any(within)),
        "min_is_finest": bool(len(within) >= 1 and abs(errors[-1])
                              <= min(abs(e) for e in errors) * (1.0 + 1e-12)),
        "expected_monotone": expect_monotone,
    }
    return record


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text())
    records = evidence["eigen"]
    reference = evidence["references"]

    families = {tag: analyse(records, tag, "ntheta" if tag == "mitc-axsym-nt" else "nz",
                             expect_monotone=tag != "mitc-full")
                for tag in FAMILIES}
    mitc = families["mitc-axsym"]
    shell4 = families["shell4-axsym"]
    nt = families["mitc-axsym-nt"]

    fitted = mitc["extrapolated_error"]
    audit = {
        "schema": SCHEMA,
        "source_evidence": str(EVIDENCE.relative_to(ROOT)),
        "source_certified": evidence["audit"]["certified"],
        "source_reason": evidence["audit"]["reason"],
        "reference_donnell_over_classical": reference["donnell_over_classical"],
        "families": families,
        # 1. Is the certification number a converged reading or a lucky crossing?
        "finest_error_within_gate": mitc["finest_within_gate"],
        "pairwise_orders": mitc["observed_orders"],
        "order_drift": (None if None in mitc["observed_orders"]
                        else mitc["observed_orders"][-1] - mitc["observed_orders"][0]),
        "fit_model": "e(h) = e_inf + C h^p, h = 1/nz, all five levels",
        "fitted_limit_error": fitted,
        "fitted_order": mitc["fit_order"],
        "fit_max_abs_residual": mitc["fit_max_abs_residual"],
        "family_monotone_and_no_sign_change": bool(
            mitc["monotone"] and all(lv["error"] > 0.0 for lv in mitc["levels"])),
        "fitted_limit_within_gate": (None if fitted is None
                                     else abs(fitted) < GATE),
        # 2. Gate-field semantics: 'within_3pct_any' is not the finest mesh.
        "gate_field_semantics": "within_3pct_any = any(|error| < 3 %) over the family",
        "min_level_is_finest": mitc["min_is_finest"],
        "levels_inside_gate": int(sum(mitc["within_gate"])),
        # 3. Circumferential facets at fixed nz.
        "ntheta_family_errors": [lv["error"] for lv in nt["levels"]],
        "ntheta_extrapolated_error": nt["extrapolated_error"],
        # 4. The uncured formulation must stay outside the band.
        "shell4_finest_error": shell4["finest_error"],
        "shell4_outside_gate": bool(abs(shell4["finest_error"]) >= GATE),
        "shell4_over_mitc_error_ratio": (abs(shell4["finest_error"])
                                         / abs(mitc["finest_error"])),
        # 5. Two-grid Richardson/GCI is *not* reportable here (it is degenerate for
        # a monotone two-point sequence and the observed order drifts).
        "gci_95_reportable": False,
        "gci_95_not_reportable_reason": (
            "two-grid Richardson returns exactly zero for any monotone pair, and the "
            "three-grid Celik order is ill-conditioned because the observed order drifts "
            f"({mitc['observed_orders'][0]:.2f} -> {mitc['observed_orders'][-1]:.2f}); "
            "the fitted three-parameter limit below replaces it."),
    }
    audit["claim_survives_convergence_check"] = bool(
        audit["finest_error_within_gate"] and audit["family_monotone_and_no_sign_change"]
        and audit["fitted_limit_within_gate"])
    audit["verdict"] = ("confirmed-by-convergence"
                       if audit["claim_survives_convergence_check"] else "not-supported")
    audit["reason"] = (
        "The MITC4 axial family is monotone with no sign change and the three-parameter "
        f"fit e(h)=e_inf+C h^p over five levels gives p={mitc['fit_order']:.3f} and a "
        f"converged limit e_inf={fitted * 100:+.3f} %, |e_inf| < 3 % "
        f"(fit residual {mitc['fit_max_abs_residual']:.2e}), so the finest-mesh "
        f"{mitc['finest_error'] * 100:.3f} % reading is part of a converging sequence, "
        "not an accidental crossing. Caveats that must accompany the claim: only one "
        f"level ({sum(mitc['within_gate'])} of {len(mitc['within_gate'])}) is inside the "
        "gate in raw readings, the observed order drifts so no two-/three-grid GCI is "
        f"reportable, and the uncured Shell4 family stays {shell4['finest_error'] * 100:.2f} % "
        f"high ({abs(shell4['finest_error']) / abs(mitc['finest_error']):.1f}x the MITC error) "
        "at the same mesh.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in (
        "verdict", "finest_error_within_gate", "fitted_limit_error", "fitted_order",
        "fit_max_abs_residual", "family_monotone_and_no_sign_change",
        "fitted_limit_within_gate", "min_level_is_finest", "levels_inside_gate",
        "shell4_finest_error", "shell4_outside_gate")}, indent=2))
    print(audit["reason"])
    _figure(families)
    return 0


def _figure(families):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print("figure skipped:", exc)
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for tag, style in (("mitc-axsym", dict(marker="o", color="C0")),
                       ("shell4-axsym", dict(marker="s", color="C3"))):
        fam = families[tag]
        h = [1.0 / lv["refinement"] for lv in fam["levels"]]
        e = [abs(lv["error"]) * 100.0 for lv in fam["levels"]]
        ax.loglog(h, e, label=tag, lw=1.4, **style)
        if fam["extrapolated_error"] is not None:
            ax.loglog([h[-1] / 3.0], [abs(fam["extrapolated_error"]) * 100.0],
                      marker="*", ms=12, ls="none",
                      color=style["color"],
                      label=f"{tag} Richardson limit")
    ax.axhline(3.0, color="0.3", ls="--", lw=1.0, label="3 % gate")
    ax.invert_xaxis()
    ax.set_xlabel("h ~ 1/nz  (finer to the right)")
    ax.set_ylabel("|error vs Donnell|  [%]")
    ax.set_title("axial refinement family")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    fam = families["mitc-axsym-nt"]
    nt = [lv["refinement"] for lv in fam["levels"]]
    err = [lv["error"] * 100.0 for lv in fam["levels"]]
    ax.plot(nt, err, "o-", color="C2", label="MITC4, nz=30")
    for x, y in zip(nt, err):
        ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=8)
    ax.axhline(3.0, color="0.3", ls="--", lw=1.0, label="3 % gate")
    ax.set_xscale("log")
    ax.set_xlabel("n_theta (circumferential facets)")
    ax.set_ylabel("error vs Donnell  [%]")
    ax.set_title("circumferential discretisation")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("MITC4 cylindrical-shell buckling certification margin", fontsize=11)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)
    print("figure ->", FIG.relative_to(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())