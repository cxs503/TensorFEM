"""Saint-Venant torsion audit for the merged HEX8 solid path (m2863).

Runs the exact-oracle qualification in ``tensorfem.saint_venant_torsion`` over
three sections and three-to-four mesh levels, writes the evidence JSON and the
figures, and prints a compact table.

    OMP_NUM_THREADS=2 MPLBACKEND=Agg PYTHONPATH=src python scripts/audit_saint_venant_torsion.py
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.saint_venant_torsion import (  # noqa: E402
    AnnulusCircleSection,
    RectSection,
    default_sections,
    torsion_run,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MESH_LEVELS = [(8, 3), (12, 4), (16, 5), (24, 7)]


def oracle_checks(section) -> dict:
    """Analytic self-checks of the reference field itself."""
    out = {"section": section.name, "j": section.j, "tau_max": section.tau_max}
    out["torque_from_stress_over_GJ"] = section.torque_from_stress() / (
        section.g * section.theta * section.j)
    if section.name != "annulus":
        h = 0.3 * section.scale
        axis = torch.linspace(-h, h, 9, dtype=torch.float64)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        out["laplacian_residual_max"] = float(
            section.laplacian_residual(xx, yy).abs().max())
        out["torque_from_phi_over_GJ"] = section.torque_from_phi() / (
            section.g * section.theta * section.j)
    return out


def convergence_order(pairs):
    """Observed order from ``(h, error)`` pairs (log-log slope of the last two)."""
    if len(pairs) < 2:
        return float("nan")
    (h0, e0), (h1, e1) = pairs[-2], pairs[-1]
    if e0 <= 0.0 or e1 <= 0.0 or h0 <= 0.0 or h1 <= 0.0:
        return float("nan")
    return math.log(e0 / e1) / math.log(h0 / h1)


def plot_convergence(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=110)
    styles = {"rectangle": "o-", "annulus": "s--"}
    seen = {}
    labels = {}
    for row in rows:
        key = row["section"] + f"@{row['ratio']}"
        style = styles.get(row["section"], "^-")
        colour = plt.cm.viridis(0.15 + 0.55 * (row["n_section"] - 8) / 16.0)
        label = key if key not in labels else None
        labels[key] = True
        for ax, key_metric in ((axes[0], "rigidity_error"),
                               (axes[1], "shear_l2_error")):
            value = abs(row[key_metric])
            if value <= 0.0:
                continue
            ax.loglog(1.0 / row["n_section"], value, style, color=colour,
                      markersize=4, linewidth=1.0, label=label)
        seen[key] = True
    for ax, title in ((axes[0], "|T_FE/(G theta J) - 1|"),
                      (axes[1], "mid-span shear L2 error")):
        ax.set_xlabel("element size h / section width")
        ax.set_ylabel(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_fields(section, n, layers, path):
    """Recovered vs exact shear magnitude on the mid-span cross-section."""
    nodes, hexes, m, nz = __import__(
        "tensorfem.saint_venant_torsion",
        fromlist=["prism_mesh"]).prism_mesh(section, n=n, layers=layers, length=4.0)
    import torch as th
    from tensorfem.solid3d import elasticity_matrix_3d, hex8_stiffness
    from tensorfem.sparse_core import assemble_coo, solve_sparse_static

    ne = len(hexes)
    ndof = 3 * len(nodes)
    E = nodes.new_full((ne,), 2.0 * (1.0 + 0.3))
    nu = nodes.new_full((ne,), 0.3)
    D = elasticity_matrix_3d(E, nu)
    ke, B = hex8_stiffness(nodes[hexes], D)
    edofs = (3 * hexes[:, :, None] + th.arange(3)).reshape(ne, 24)
    K = assemble_coo(ke, edofs, ndof)
    theta = section.theta
    dirichlet = {}
    for k, sl in enumerate((th.arange(m), th.arange(ndof // 3 - m, ndof // 3))):
        z = 0.0 if k == 0 else 4.0
        xy = nodes[sl]
        ux, uy = -theta * z * xy[:, 1], theta * z * xy[:, 0]
        for pos, node in enumerate(sl.tolist()):
            dirichlet[3 * node] = float(ux[pos])
            dirichlet[3 * node + 1] = float(uy[pos])
    force = nodes.new_zeros(ndof)
    u = solve_sparse_static(K, force, dirichlet=dirichlet, rtol=1e-11).displacement
    strain = th.einsum("eij,ej->ei", B, u.flatten()[edofs])
    stress = th.einsum("eij,ej->ei", D, strain)
    centres = nodes[hexes].mean(1)
    band = (centres[:, 2] - 2.0).abs() <= 0.3
    cx, cy = centres[band, 0], centres[band, 1]
    got = th.hypot(stress[band, 5], stress[band, 4])
    exz, eyz = section.tau(cx, cy)
    want = th.hypot(exz, eyz)
    scale = float(want.max())
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), dpi=110)
    for ax, values, title in (
        (axes[0], got / scale, "recovered |tau| / exact max"),
        (axes[1], want / scale, "exact |tau| / exact max"),
        (axes[2], (got - want) / scale, "recovered - exact (fraction of max)"),
    ):
        lim = 1.0 if values is axes[0] or values is axes[1] else 1e-3
        art = ax.tricontourf(cx.numpy(), cy.numpy(), values.numpy(), levels=20,
                             cmap="viridis" if lim > 1e-2 else "coolwarm")
        fig.colorbar(art, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=7)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{section.name} n={n}: mid-span shear field", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--output", default="benchmarks/saint_venant_torsion.json")
    parser.add_argument("--assets", default="docs/assets/saint-venant")
    args = parser.parse_args()
    assets = ROOT / args.assets
    assets.mkdir(parents=True, exist_ok=True)
    oracle = [oracle_checks(section) for section in default_sections()]
    rows, by_section = [], {}
    for section in default_sections():
        widths = getattr(section, "a", 1.0)
        for n, layers in MESH_LEVELS:
            result = torsion_run(section, n=n, layers=layers, length=args.length)
            h = widths / n
            row = {
                "section": section.name,
                "ratio": round(getattr(section, "b", widths) / getattr(section, "a", widths), 3)
                if hasattr(section, "b") else 1.0,
                "n_section": n,
                "layers": layers,
                "elements": result.elements,
                "dofs": result.dofs,
                "h": h,
                "rigidity_ratio": result.torsional_rigidity_ratio,
                "rigidity_error": result.torsional_rigidity_ratio - 1.0,
                "shear_l2_error": result.shear_l2_error,
                "shear_max_error": result.shear_max_error,
                "shear_peak_ratio": result.shear_peak_ratio,
                "torque_equilibrium": abs(result.torque_end_a + result.torque_end_b)
                / result.torque_exact,
                "displacement_error": result.displacement_error,
            }
            rows.append(row)
            by_section.setdefault(section.name, []).append((h, abs(result.torsional_rigidity_ratio - 1.0),
                                                            result.shear_l2_error))
            print(f"{section.name:9s} n={n:2d} dofs={result.dofs:5d} "
                  f"rig={result.torsional_rigidity_ratio:.5f} "
                  f"l2={result.shear_l2_error*100:.3f}% "
                  f"max={result.shear_max_error*100:.3f}%")
    orders = {name: {"rigidity_order": convergence_order([(h, e) for h, e, _ in pairs]),
                     "shear_l2_order": convergence_order([(h, e) for h, _, e in pairs])}
              for name, pairs in by_section.items()}
    plot_convergence(rows, assets / "convergence.png")
    plot_fields(RectSection(1.0, 0.5), 16, 5, assets / "rectangle-field.png")
    plot_fields(AnnulusCircleSection(0.5, 1.0), 16, 5, assets / "annulus-field.png")
    payload = {"length": args.length, "mesh_levels": MESH_LEVELS,
               "oracle": oracle, "rows": rows, "orders": orders,
               "figures": ["convergence.png", "rectangle-field.png", "annulus-field.png"]}
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))
    print("orders:", json.dumps(orders))
    print("written:", target)


if __name__ == "__main__":
    main()