"""Evidence run for the Boussinesq half-space verification (TensorFEM).

Produces ``docs/halfspace_evidence.json`` and the four figures used in
``docs/elastic-halfspace-boussinesq-verification.md``:

* convergence of the patch-centre settlement against the closed form,
* the vertical-stress profile along the load axis against Newmark,
* the surface settlement profile against the Boussinesq integral,
* a sigma_zz contour of the truncated quarter model on the x-z slice.

Run: ``PYTHONPATH=src python3 examples/halfspace_verification.py``
"""
from __future__ import annotations

import json
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.elastic_halfspace import solve_halfspace_patch, settlement, vertical_stress

torch.set_num_threads(8)
ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    evidence = {"cases": [], "figure": []}
    runs = []
    for name, cfg in (("coarse", dict(n_fine=2, n_grow=10, nz_layers=10)),
                      ("medium", dict(n_fine=4, n_grow=12, nz_layers=14))):
        start = time.time()
        result = solve_halfspace_patch(rtol=1e-10, **cfg)
        record = result.summary()
        record.update(name=name, wall_time_s=round(time.time() - start, 1), config=cfg)
        evidence["cases"].append(record)
        runs.append(result)
        print(name, record["n_dofs"], record["wall_time_s"], record["w_center_rel_error"], flush=True)

    exact_center = runs[0].w_center_exact
    exact_corner = runs[0].w_corner_exact
    evidence["reference"] = {"w_center": exact_center, "w_corner": exact_corner}

    # --- domain truncation, at a fixed grading ratio so only the extent changes ---
    truncation = []
    for lateral, n_grow in ((4.0, 4), (8.0, 7), (20.0, 10)):
        result = solve_halfspace_patch(n_fine=2, n_grow=n_grow, nz_layers=10,
                                       growth=1.3, lateral=lateral, rtol=1e-10)
        truncation.append({"lateral_over_a": lateral, "w_center": result.w_center,
                           "rel_error": result.center_error, "n_grow": n_grow})
        print("lateral", lateral, result.w_center, result.center_error, flush=True)
    evidence["truncation"] = truncation

    # --- figure 1: settlement convergence ---
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    spacing = [run.finest_spacing for run in runs]
    errors = [run.center_error for run in runs]
    ax.loglog(spacing, errors, "o-", label="quarter-model centre settlement")
    ax.loglog(spacing, errors[-1] * (spacing[-1] / torch.tensor(spacing).numpy()) ** 2,
              "--", lw=0.8, color="grey", label=r"reference slope $h^2$")
    ax.set_xlabel("finest element size  $h/a$")
    ax.set_ylabel("relative settlement error")
    ax.set_title("Boussinesq patch settlement: convergence to the closed form")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "halfspace_settlement_convergence.png", dpi=150)
    evidence["figure"].append("halfspace_settlement_convergence.png")
    plt.close(fig)

    # --- figure 2: vertical stress profile ---
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    depths = torch.linspace(0.05, 18.0, 400)
    reference = [-vertical_stress(1.0, 2.0, 2.0, 0.0, 0.0, float(d)) for d in depths]
    ax.semilogx(depths, reference, "-", color="k", label="Newmark closed form (axis)")
    for run in runs:
        usable = [(d, fe) for d, fe in zip(run.depths, run.stress_fe) if d >= 0.1]
        ax.semilogx([d for d, _ in usable], [-v for _, v in usable], "o-",
                    ms=3, label=f"HEX8 {run.n_nodes} nodes")
    ax.set_xlabel("depth below the loaded surface  $z/a$")
    ax.set_ylabel(r"$|\sigma_z|/p$  on the load axis")
    ax.set_title("Vertical stress under a uniformly loaded patch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "halfspace_stress_profile.png", dpi=150)
    evidence["figure"].append("halfspace_stress_profile.png")
    plt.close(fig)

    # --- figure 3: surface settlement profile ---
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    grid = torch.linspace(0.0, 5.0, 300)
    ax.plot(grid, [settlement(1.0, 1.0, 0.3, 2.0, 2.0, float(x) + 1.0, 1.0) for x in grid],
            "-", color="k", label="Boussinesq closed form")
    for run in runs:
        ax.plot(run.profile_x, run.profile_w, "o--", ms=3,
                label=f"HEX8 {run.n_nodes} nodes")
    ax.axvline(2.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("distance from the patch centre  $x/a$  (patch edge at 2)")
    ax.set_ylabel("surface settlement  $w E/(p a)$")
    ax.set_title("Surface settlement of a $2a\\times 2a$ patch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "halfspace_settlement_profile.png", dpi=150)
    evidence["figure"].append("halfspace_settlement_profile.png")
    plt.close(fig)

    # --- figure 4: sigma_zz contour on the x-z slice ---
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = [point[0] for point in runs[-1].slice_xz]
    zs = [point[1] for point in runs[-1].slice_xz]
    values = [point[2] for point in runs[-1].slice_xz]
    contour = ax.tricontourf(xs, zs, values, levels=20, cmap="viridis")
    ax.invert_yaxis()
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(8.0, 0.0)
    ax.set_xlabel("$x/a$")
    ax.set_ylabel("$z/a$")
    ax.set_title(r"$\sigma_{zz}/p$ in the truncated quarter model (slice at $y=h_0/2$)")
    fig.colorbar(contour, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "halfspace_sigma_zz_field.png", dpi=150)
    evidence["figure"].append("halfspace_sigma_zz_field.png")
    plt.close(fig)

    (ROOT / "docs" / "halfspace_evidence.json").write_text(json.dumps(evidence, indent=1))
    return evidence


if __name__ == "__main__":
    payload = main()
    for case in payload["cases"]:
        print(f"{case['name']:7s} dofs={case['n_dofs']:6d} "
              f"centre_err={100 * case['w_center_rel_error']:.3f}% "
              f"corner_err={100 * case['w_corner_rel_error']:.3f}% "
              f"sigma_z_max_err={100 * case['sigma_z_max_rel_error']:.1f}% "
              f"balance={case['reaction_balance_rel']:.1e}")
    print("figures:", ", ".join(payload["figure"]))