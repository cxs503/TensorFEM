#!/usr/bin/env python3
"""Audit figures for m19999 (CASE-FEM-CANTILEVER-TIP).

Data sources (all machine-read, nothing hand-typed):
  * docs/audit_m19999/audited_case/case_fem_cantilever_tip_evidence.json  (mission artifact)
  * artifacts/m19999_audit/rerun_evidence.json                            (auditor rerun, if present)
  * artifacts/m19999_audit/recheck_analytic.json                          (auditor re-derivation)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
AUD = HERE / "audited_case/case_fem_cantilever_tip_evidence.json"
RERUN = HERE.parents[1] / "artifacts/m19999_audit/rerun_evidence.json"
REFCHK = HERE.parents[1] / "artifacts/m19999_audit/verify_reference_mesh.json"
REMAIN = HERE.parents[1] / "artifacts/m19999_audit/verify_remaining_levels.json"
RECHECK = HERE.parents[1] / "artifacts/m19999_audit/recheck_analytic.json"
FIG = HERE / "figs"
FIG.mkdir(exist_ok=True)


def _rows(doc, key):
    return [r for r in doc["studies"][key] if isinstance(r, dict)]


def solid_series(doc):
    rows = [(r["dofs"], r["relative_error_vs_euler_bernoulli"] * 100,
             f"{r['axial_elements']}x{r['cross_elements']}x{r['cross_elements']}")
            for r in _rows(doc, "solid_ladder")]
    rows.append((doc["levels"]["solid_reference"]["dofs"],
                 doc["levels"]["solid_reference"]["relative_error_vs_euler_bernoulli"] * 100,
                 "ref 32x4x4"))
    rows.sort()
    return rows


def beam_series(doc):
    return sorted((r["dofs"], r["relative_error_vs_euler_bernoulli"] * 100)
                  for r in _rows(doc, "beam_eb_sweep"))


def main():
    aud = json.loads(AUD.read_text())
    ref = aud["verdict"]["reference_value_m"]
    tol = aud["verdict"]["tolerance_rel"]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    s = solid_series(aud)
    ax.plot([r[0] for r in s], [r[1] for r in s], "o-", color="crimson",
            label="HEX20 solid (audited evidence)")
    for d, e, lab in s:
        ax.annotate(lab, (d, e), textcoords="offset points", xytext=(4, -10), fontsize=7)
    b = beam_series(aud)
    ax.plot([r[0] for r in b], [r[1] for r in b], "s--", color="royalblue",
            label="EB frame (machine exact)")
    pts = []
    if REFCHK.exists():
        r = json.loads(REFCHK.read_text())["reference_rows"]["solid_reference_32x4x4"]
        pts.append((r["dofs"], (r["audit_m"] - ref) / ref * 100.0))
    if REMAIN.exists():
        rows = json.loads(REMAIN.read_text())["rows"]
        for k, v in rows.items():
            if k.startswith("L/") or "audit_m" not in v or "dofs" not in v:
                continue
            pts.append((v["dofs"], (v["audit_m"] - ref) / ref * 100.0))
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "x", color="black", ms=8,
                label="auditor re-solve on main (bit-identical)")
    if RERUN.exists():
        rs = solid_series(json.loads(RERUN.read_text()))
        ax.plot([r[0] for r in rs], [r[1] for r in rs], "^", mfc="none", color="darkgreen",
                ms=8, label="auditor full-workload rerun")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.axhspan(-tol * 100, tol * 100, color="grey", alpha=0.12)
    ax.axhline(tol * 100, color="grey", ls="--", lw=0.9)
    ax.axhline(-tol * 100, color="grey", ls="--", lw=0.9,
               label=f"±{tol*100:.0f}% case-pool tolerance")
    ax.set_xscale("log")
    ax.set_xlabel("degrees of freedom [-]")
    ax.set_ylabel("relative error vs $PL^3/3EI$ [%]")
    ax.set_title("CASE-FEM-CANTILEVER-TIP: h-convergence of the HEX20 reference study")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_convergence.png", dpi=150)

    # fig2: got vs reference, all levels, log-log tip deflection
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    allrows = _rows(aud, "solid_ladder") + [aud["levels"]["solid_reference"]]
    got = [lv["tip_deflection_m"] for lv in allrows]
    dofs = [lv["dofs"] for lv in allrows]
    ax2.loglog(dofs, got, "o", color="crimson", label="computed tip deflection")
    ax2.axhline(ref, color="black", ls="-", lw=1.2, label=f"analytic $PL^3/3EI$ = {ref:.6e} m")
    t = aud["reference"].get("timoshenko_m")
    if t:
        ax2.axhline(t, color="royalblue", ls=":", lw=1.2, label=f"Timoshenko = {t:.6e} m")
    ax2.axhspan(ref * (1 - tol), ref * (1 + tol), color="grey", alpha=0.12,
                label=f"±{tol*100:.0f}% band")
    if RECHECK.exists():
        rc = json.loads(RECHECK.read_text())["independent"]
        ax2.plot([min(dofs), max(dofs)], [rc["delta_eb_m"]] * 2, color="green", lw=0.8, ls="--",
                 label="auditor independent re-derivation")
    ax2.set_xlabel("degrees of freedom [-]")
    ax2.set_ylabel("tip deflection [m]")
    ax2.set_title("Computed vs analytic tip deflection (all model levels)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig(FIG / "fig2_got_vs_ref.png", dpi=150)
    print("wrote", FIG / "fig1_convergence.png", FIG / "fig2_got_vs_ref.png")


if __name__ == "__main__":
    main()