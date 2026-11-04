"""Recompute the audit verdict from the evidence JSON and draw the audit figure.

Reads ``benchmarks/cylindrical_shell_buckling_evidence.json`` (produced by
``scripts/audit_cylindrical_shell_buckling.py``), rebuilds the ``audit`` block
with the mesh families de-duplicated, and writes
``docs/assets/cylindrical-shell-buckling.png``.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def family(rows, **want):
    seen = {}
    for row in rows:
        if all(row[k] == v for k, v in want.items()):
            seen[(row["nz"], row["ntheta"])] = row
    return [seen[k] for k in sorted(seen, key=lambda k: (k[0], k[1]))]


def main() -> int:
    path = pathlib.Path("benchmarks/cylindrical_shell_buckling_evidence.json")
    data = json.loads(path.read_text())
    rows = data["eigen"]
    hoop = sorted(family(rows, nz=10), key=lambda r: r["ntheta"])
    axial = sorted(family(rows, ntheta=32), key=lambda r: r["nz"])
    errors = [r["error_vs_donnell"] for r in rows]
    data["audit"] = {
        "monotone_in_ntheta": all(b["error_vs_donnell"] <= a["error_vs_donnell"] + 1e-12
                                  for a, b in zip(hoop, hoop[1:])),
        "monotone_in_nz": all(b["error_vs_donnell"] <= a["error_vs_donnell"] + 1e-12
                              for a, b in zip(axial, axial[1:])),
        "sign_change": min(errors) < 0 < max(errors),
        "min_error": min(errors), "max_error": max(errors),
        "meshes_within_3pct": [f"nz={r['nz']},ntheta={r['ntheta']}" for r in rows
                               if abs(r["error_vs_donnell"]) < .03],
        "axisymmetry_lost_at": [f"nz={r['nz']},ntheta={r['ntheta']}" for r in rows
                                if not r["axisymmetric"]],
        "certified": False,
        "reason": ("flat-facet cylinder buckling is not certifiable within 3%: the error is "
                   "monotone in ntheta at fixed nz but it overshoots and changes sign, the "
                   "axial family is still 23% high at the finest affordable mesh, and the "
                   "critical mode stops being axisymmetric once ntheta >= 48"),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    nt = [r["ntheta"] for r in hoop]
    ax1.axhspan(-3, 3, color="tab:green", alpha=.12, label="±3% acceptance band")
    ax1.axhline(0, color="k", lw=.8)
    ax1.plot(nt, [100 * r["error_vs_donnell"] for r in hoop], "o-", color="tab:red",
             label="circumferential sweep, nz=10")
    ax1.plot([r["nz"] for r in axial], [100 * r["error_vs_donnell"] for r in axial], "s--",
             color="tab:blue", label="axial sweep, ntheta=32 (abscissa in nz)")
    for r in hoop:
        ax1.annotate(f"{r['ntheta']}", (r["ntheta"], 100 * r["error_vs_donnell"]),
                     textcoords="offset points", xytext=(3, 5), fontsize=7)
    ax1.set_xscale("log")
    ax1.set_xlabel("circumferential facets $n_\\theta$  (blue: axial facets $n_z$)")
    ax1.set_ylabel("$N_{cr}$ error vs Donnell finite-length reference  [%]")
    ax1.set_title("(a) eigenvalue error: over-stiff, then sign change")
    ax1.grid(alpha=.3, which="both")
    ax1.legend(fontsize=7, loc="upper right")

    labels = [f"$m$={r['m']}" for r in data["prescribed_energy"]]
    ax2.axhline(1.0, color="k", lw=.8, label="Donnell theory")
    ax2.bar(labels, [r["ratio"] for r in data["prescribed_energy"]], color="tab:orange",
            alpha=.8, label="prescribed-field energy, $n_\\theta$=16")
    ax2.axhline(data["hoop_energy"][0]["ratio_over_facet_area"], color="tab:blue", ls="--",
                label="hoop energy / facet-area ratio ($\\to 1/(1-\\nu^2)$, exact)")
    ax2.set_ylim(0.9, 1.85)
    ax2.set_xlabel("Donnell axial half-wave count $m$")
    ax2.set_ylabel("FEM energy / continuum energy")
    ax2.set_title("(b) energy audit: hoop term exact, bending-dominated field stiff")
    ax2.grid(alpha=.3, axis="y")
    ax2.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = pathlib.Path("docs/assets/cylindrical-shell-buckling.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print("wrote", out)
    print(json.dumps(data["audit"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())