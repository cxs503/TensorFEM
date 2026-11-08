#!/usr/bin/env python3
"""Generate the HEX20 mass-matrix evidence bundle and its figures.

Outputs (all regenerated from code, nothing typed in by hand):

* ``benchmarks/hex20_mass_audit.json`` -- every number quoted in
  ``docs/hex20-consistent-mass.md``,
* ``docs/assets/hex20-mass/*.png`` -- four figures.

Run: ``python3 scripts/audit_hex20_mass.py`` (single process, seconds).
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

from tensorfem.hex20_mass_audit import (  # noqa: E402
    build_evidence,
    constrained_rod_modes,
    distortion_scope,
)
from tensorfem.hex20 import hex20_lumped_mass, structured_hex20_mesh  # noqa: E402

ASSETS = ROOT / "docs" / "assets" / "hex20-mass"
BENCH = ROOT / "benchmarks"


def fig_rule_error(ev, path):
    rules = ev["rules"]
    orders = [r["order"] for r in rules]
    frob = [max(r["rel_frobenius"], 1e-16) for r in rules]
    total = [abs(r["total_mass_rel"]) + 1e-17 for r in rules]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar([o - 0.2 for o in orders], frob, width=0.4, label="mass-matrix error (Frobenius)")
    ax.bar([o + 0.2 for o in orders], total, width=0.4, label="total-mass error")
    ax.set_yscale("log")
    ax.set_xticks(orders)
    ax.set_xlabel("Gauss order per natural coordinate (27 points = order 3)")
    ax.set_ylabel("relative error  [-]")
    ax.set_title("HEX20 consistent mass: what the rule choice actually changes")
    ax.axhline(1e-12, color="0.6", lw=0.8, ls=":")
    ax.annotate(
        f"order 2: {rules[1]['rel_frobenius']*100:.1f} % matrix deficiency\n"
        "total mass still exact -> mass sum is a blind check",
        xy=(2.2, rules[1]["rel_frobenius"]),
        xytext=(2.45, 1.0),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_rod_convergence(ev, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cons = ev["rod_constrained"]
    nex = [c["nex"] for c in cons]
    for mode in (1, 2):
        err = [abs(c["rows"][mode - 1]["rel_error"]) for c in cons]
        ax.loglog(nex, err, "o-", label=f"1-D constrained rod, axial mode {mode}")
    full = ev["rod_full3d"]
    ax.loglog([f["nex"] for f in full], [abs(f["rows"][0]["rel_error"]) for f in full],
              "s--", label="full 3-D rod, first axial mode")
    ref = [abs(cons[0]["rows"][0]["rel_error"]) * (nex[0] / n) ** 4 for n in nex]
    ax.loglog(nex, ref, "k:", lw=1.0, label=r"$O(h^{4})$ reference")
    ax.set_xlabel("elements along the rod")
    ax.set_ylabel("relative frequency error  [-]")
    ax.set_title("Inertia-operator verification by eigenfrequency (mode 1, 2)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_lumping(ev, path):
    nodes, cells = structured_hex20_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    density = 7850.0
    hrz = hex20_lumped_mass(nodes[cells], density, scheme="hrz")[0][0::3].tolist()
    row = hex20_lumped_mass(nodes[cells], density, scheme="row")[0][0::3].tolist()
    idx = list(range(1, 21))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar([i - 0.2 for i in idx], hrz, width=0.4, label="HRZ (default)")
    ax.bar([i + 0.2 for i in idx], row, width=0.4, label="row sum (unusable)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("HEX20 local node (1-8 corners, 9-20 edge midpoints)")
    ax.set_ylabel("nodal mass per component  [kg]")
    ax.set_title("Lumping traps: row-sum gives negative corner masses")
    ax.annotate(
        f"corner = -rho*V/8 = {row[0]:.1f} kg",
        xy=(1.2, row[0]),
        xytext=(3.0, row[0] * 0.75),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_geometry_scope(ev, path):
    labels = ["straight\n(affine)", "affine\nshear", "non-planar\ncorner", "curved\nmid-edge"]
    vals = [
        max(ev["straight_element"]["mass_vs_reference_frobenius"], 1e-17),
        max(ev["distortion"]["affine_shear_rule3_reference_rel"], 1e-17),
        max(ev["distortion"]["nonplanar_rule3_reference_rel"], 1e-17),
        max(ev["curved"]["order3"]["rel_frobenius"], 1e-17),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.6, f"{v:.2e}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("order-3 rule vs high-order reference (Frobenius)")
    ax.set_title("Scope of the 'exact on straight HEX20' claim")
    ax.set_ylim(1e-17, 1e-2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    BENCH.mkdir(exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    ev = build_evidence()
    # extra curve for the figure (finer sweep of the constrained rod)
    ev["rod_constrained_fine"] = [
        {"nex": n, "rows": constrained_rod_modes(n)["rows"]} for n in (1, 2, 3, 4, 6, 8, 12, 16)
    ]
    ev["distortion_extra"] = distortion_scope()
    (BENCH / "hex20_mass_audit.json").write_text(json.dumps(ev, indent=2, sort_keys=True) + "\n")

    fig_rule_error(ev, ASSETS / "rule-error.png")
    fig_rod_convergence(ev, ASSETS / "rod-convergence.png")
    fig_lumping(ev, ASSETS / "lumping.png")
    fig_geometry_scope(ev, ASSETS / "geometry-scope.png")

    print("straight element vs NumPy oracle:",
          f"{ev['straight_element']['mass_vs_reference_frobenius']:.2e}")
    print("total mass error:", f"{ev['straight_element']['volume_rel_error']:.2e}")
    print("rotation energy errors:", [f"{v:.2e}" for v in ev["kinetic_energy"]["rotation_rel_error"]])
    print("rule 2 deficiency:", f"{ev['rules'][1]['rel_frobenius']*100:.2f} %")
    print("curved rule 3 residual:", f"{ev['curved']['order3']['rel_frobenius']*100:.3f} %")
    print("constrained rod errors:",
          [f"{r['nex']}:{abs(r['rows'][0]['rel_error'])*100:.2e}%" for r in ev["rod_constrained_fine"]])
    print("observed order:", [round(o, 2) for o in ev["rod_constrained_observed_order"]])
    print("free-free rigid ratio:", f"{ev['free_free']['rigid_max_over_max']:.2e}")
    print("wrote", BENCH / "hex20_mass_audit.json", "and 4 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())