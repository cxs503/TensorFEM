#!/usr/bin/env python3
"""Generate the TET10 mass-matrix evidence bundle and its figures.

Outputs (all regenerated from code, nothing typed in by hand):

* ``benchmarks/tet10_mass_audit.json`` -- every number quoted in
  ``docs/tet10-consistent-mass.md``,
* ``docs/assets/tet10-mass/{exactness,rod_convergence}.png``.

Run: ``python3 scripts/audit_tet10_mass.py`` (single process, ~1 min).
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.tet10_mass_audit import build_evidence  # noqa: E402

OUT = ROOT / "benchmarks" / "tet10_mass_audit.json"
FIGS = ROOT / "docs" / "assets" / "tet10-mass"


def figure_exactness(ev, path):
    rows = ev["rule_orders"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    orders = [r["order"] for r in rows]
    errs = [max(r["frobenius_relative"], 1e-18) for r in rows]
    ax.semilogy(orders, errs, "o-", label="Duffy tensor rule")
    ax.axhline(ev["keast"]["frobenius_relative"], color="crimson", ls="--",
               label="shipped 5-point Keast (degree 3)")
    ax.axhline(1e-15, color="grey", lw=0.8)
    ax.set_xlabel("Duffy points per collapsed direction")
    ax.set_ylabel(r"$\|M_h-M_{\rm exact}\|_F/\|M_{\rm exact}\|_F$")
    ax.set_title("TET10 consistent mass on an affine element")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_rod(ev, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for key, label, style in (
        ("rod_consistent_nu0", r"consistent, $\nu=0$ (pure axial)", "o-"),
        ("rod_hrz_nu0", r"HRZ lumped, $\nu=0$", "s--"),
        ("rod_consistent", r"consistent, $\nu=0.3$ (3-D coupling)", "^-"),
    ):
        rows = ev[key]
        ax.loglog([r["nex"] for r in rows],
                  [max(r["relative_errors"][0], 1e-16) for r in rows], style, label=label)
    ax.set_xlabel("elements along the axis")
    ax.set_ylabel(r"$|\omega_{h,1}/\omega_{\rm rod}-1|$")
    ax.set_title("fixed-free rod, first axial mode")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ev = build_evidence()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ev, indent=1, sort_keys=True) + "\n")
    FIGS.mkdir(parents=True, exist_ok=True)
    figure_exactness(ev, FIGS / "exactness.png")
    figure_rod(ev, FIGS / "rod_convergence.png")
    print(json.dumps({k: ev[k] for k in ("element", "lumping", "keast")}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
