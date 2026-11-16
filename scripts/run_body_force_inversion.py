"""Driver: sparse-observation body-force inversion, archived as JSON + figures.

Runs the anchor case (JOUR-DIFFLBM-L3-INVERSION, FEM instantiation) plus the
controls that decide whether the number may be trusted, and writes every figure
input to ``artifacts/body_force_inversion/``:

``case_acceptance.json``   full result of the <= 10 %-node recovery
``observability.json``     singular spectrum of the observation operator
``controls.json``          out-of-span truth and full-nodal negative control
``noise_sweep.json``       field error vs sensor noise at two damping levels
``summary.json``           the handful of scalars quoted in the doc/report
``figures/*.png``          convergence curve, field maps, noise curve

Every number quoted downstream is read from these files, never re-typed.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.body_force_inversion import (  # noqa: E402
    InversionOptions,
    ModalBasis,
    PlaneStressMesh,
    PlaneStressModel,
    run_body_force_inversion,
)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "body_force_inversion"
FIGS = OUT / "figures"

MESH = PlaneStressMesh(Lx=1.0, Ly=0.25, nx=20, ny=10)
TRUTH = [1.0, 0.5, -0.4, 0.3, 0.6, -0.7, 0.25, 0.4]  # order-2 basis, 4 modes/component
ACCEPTANCE_ALPHA = 1.0e-12


def _dump(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _field_grid(basis: ModalBasis, coeffs: list[float], n: int = 81):
    xs = torch.linspace(0.0, MESH.Lx, n, dtype=torch.float64)
    ys = torch.linspace(0.0, MESH.Ly, n, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    c = torch.tensor(coeffs, dtype=torch.float64)
    bx = basis.evaluate(c, 0, xx.reshape(-1), yy.reshape(-1)).reshape(n, n)
    by = basis.evaluate(c, 1, xx.reshape(-1), yy.reshape(-1)).reshape(n, n)
    return xx, yy, bx, by


def _boundary_dofs(mesh: PlaneStressMesh) -> torch.Tensor:
    ids = []
    for iy in range(mesh.ny + 1):
        for ix in range(mesh.nx + 1):
            if ix in (0, mesh.nx) or iy in (0, mesh.ny):
                n = mesh.node_index(ix, iy)
                ids.extend((2 * n, 2 * n + 1))
    return torch.tensor(sorted(ids), dtype=torch.long)


def _solve_dirichlet(model, u_exact, load=None):
    mesh = model.mesh
    K = model.stiffness()
    fixed = _boundary_dofs(mesh)
    free = torch.ones(mesh.n_dof, dtype=torch.bool)
    free[fixed] = False
    rhs = torch.zeros(mesh.n_dof, dtype=torch.float64) if load is None else load.clone()
    rhs[free] = rhs[free] - K[free][:, fixed] @ u_exact[fixed]
    u = torch.zeros(mesh.n_dof, dtype=torch.float64)
    u[fixed] = u_exact[fixed]
    u[free] = torch.linalg.solve(K[free][:, free], rhs[free])
    return u


def _patch_errors():
    """Constant-strain reproduction on the whole-boundary-Dirichlet patch setup."""
    out = {}
    for nu in (0.0, 0.3, 0.49):
        mesh = PlaneStressMesh(1.0, 0.25, 8, 4)
        model = PlaneStressModel(mesh=mesh, nu=nu)
        c = mesh.coords()
        eps = 1.0e-3
        states = {
            "uniaxial": torch.stack(
                [eps * c[:, 0], -nu * eps * c[:, 1]], dim=1
            ).reshape(-1),
            "pure_shear": torch.stack(
                [eps * c[:, 1], torch.zeros_like(c[:, 1])], dim=1
            ).reshape(-1),
        }
        if nu == 0.0:
            states["pure_bending"] = torch.stack(
                [-eps * c[:, 0] * c[:, 1], 0.5 * eps * c[:, 0] ** 2], dim=1
            ).reshape(-1)
        for name, u_ex in states.items():
            u = _solve_dirichlet(model, u_ex)
            out[f"nu{nu}_{name}"] = float((u - u_ex).norm() / u_ex.norm())
    return out


def _manufactured_errors():
    """Displacement error of the analytic-trigonometric case on three meshes."""
    nu, E, amp = 0.3, 210.0e9, 1.0e-4
    Lx, Ly = 1.0, 0.25
    cc = E / (1.0 - nu * nu)
    g = cc * (1.0 - nu) / 2.0
    kx, ky = math.pi / Lx, math.pi / Ly
    rows = []
    for nx, ny in ((16, 4), (32, 8), (64, 16)):
        mesh = PlaneStressMesh(Lx, Ly, nx, ny)
        model = PlaneStressModel(mesh=mesh, E=E, nu=nu)

        def bx(x, y):
            return amp * (cc * kx * kx + g * ky * ky) * torch.sin(kx * x) * torch.sin(ky * y)

        def by(x, y):
            return -amp * (g + cc * nu) * kx * ky * torch.cos(kx * x) * torch.cos(ky * y)

        load = model.load_matrix([(bx, 0), (by, 1)]).sum(dim=1)
        c = mesh.coords()
        u_ex = torch.zeros(mesh.n_dof, dtype=torch.float64)
        u_ex[0::2] = amp * torch.sin(kx * c[:, 0]) * torch.sin(ky * c[:, 1])
        u = _solve_dirichlet(model, u_ex, load)
        interior = torch.ones(mesh.n_dof, dtype=torch.bool)
        interior[_boundary_dofs(mesh)] = False
        rows.append(
            {
                "nx": nx,
                "ny": ny,
                "rel_l2_interior": float((u - u_ex)[interior].norm() / u_ex[interior].norm()),
            }
        )
    rates = []
    for a, b in zip(rows, rows[1:]):
        rates.append(math.log(a["rel_l2_interior"] / b["rel_l2_interior"]) / math.log(2.0))
    return {"rows": rows, "observed_rates": rates}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # -- anchor case ------------------------------------------------------- #
    acc = run_body_force_inversion(
        mesh=MESH,
        truth_coeffs=TRUTH,
        order=2,
        fraction=0.10,
        options=InversionOptions(alpha_rel=ACCEPTANCE_ALPHA),
    )
    _dump("case_acceptance.json", acc)

    # -- observability: sampling density vs modal stiffness ---------------- #
    obs_rows = []
    for frac in (0.10, 0.25, 1.00):
        r = run_body_force_inversion(
            mesh=MESH,
            truth_coeffs=TRUTH,
            order=2,
            fraction=frac,
            options=InversionOptions(alpha_rel=ACCEPTANCE_ALPHA),
        )
        obs_rows.append(
            {
                "fraction": r["observations"]["fraction"],
                "n_node_sampled": r["observations"]["n_node_sampled"],
                "condition_number": r["observability"]["condition_number"],
                "scaled_singular_values": r["observability"]["scaled_singular_values"],
                "field_rel_l2": r["field_rel_l2"],
            }
        )
    _dump("observability.json", {"mesh": acc["mesh"], "rows": obs_rows})

    # -- forward-model verification (patch tests + manufactured rate) ------- #
    patch = _patch_errors()
    manufactured = _manufactured_errors()
    _dump(
        "forward_verification.json",
        {
            "patch_relative_errors": patch,
            "patch_max": max(patch.values()),
            "manufactured": manufactured,
        },
    )

    # -- negative controls -------------------------------------------------- #
    def out_of_span(x, y):
        return 0.6 * torch.sin(5 * math.pi * x / MESH.Lx) * torch.sin(3 * math.pi * y / MESH.Ly)

    oos = run_body_force_inversion(
        mesh=MESH,
        truth_coeffs=TRUTH,
        truth_extra=out_of_span,
        order=2,
        fraction=0.10,
        options=InversionOptions(alpha_rel=ACCEPTANCE_ALPHA),
    )
    nodal_mesh = PlaneStressMesh(1.0, 0.25, 12, 6)
    basis_n = ModalBasis(mesh=nodal_mesh, order=2)
    c_true = torch.tensor(TRUTH, dtype=torch.float64)
    nodal_truth = list(basis_n.nodal_field(c_true, 0)) + list(basis_n.nodal_field(c_true, 1))
    nodal = run_body_force_inversion(
        mesh=nodal_mesh,
        truth_coeffs=[float(v) for v in nodal_truth],
        mode="nodal",
        fraction=0.10,
        options=InversionOptions(alpha_rel=1.0e-8),
    )
    _dump(
        "controls.json",
        {
            "out_of_span": {
                "field_rel_l2": oos["field_rel_l2"],
                "acceptance": oos["acceptance"],
                "iterations": oos["iterations"],
            },
            "full_nodal": {
                "field_rel_l2": nodal["field_rel_l2"],
                "null_space_dim": nodal["null_space_dim"],
                "n_unknown": nodal["basis"]["n_unknown"],
                "n_observation": nodal["observations"]["n_observation"],
                "acceptance": nodal["acceptance"],
            },
        },
    )

    # -- noise sweep -------------------------------------------------------- #
    sweep = []
    for noise in (0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02):
        for alpha in (1.0e-12, 1.0e-6):
            r = run_body_force_inversion(
                mesh=MESH,
                truth_coeffs=TRUTH,
                order=2,
                fraction=0.10,
                options=InversionOptions(alpha_rel=alpha, noise=noise),
            )
            sweep.append(
                {
                    "noise_relative": noise,
                    "alpha_rel": alpha,
                    "field_rel_l2": r["field_rel_l2"],
                    "iterations": r["iterations"],
                    "stop_reason": r["stop_reason"],
                }
            )
    _dump("noise_sweep.json", {"rows": sweep})

    # -- figures ------------------------------------------------------------ #
    _fig_convergence(acc)
    _fig_fields(acc)
    _fig_noise(sweep)

    summary = {
        "mesh": acc["mesh"],
        "n_unknown": acc["basis"]["n_unknown"],
        "observed_fraction": acc["observations"]["fraction"],
        "n_node_sampled": acc["observations"]["n_node_sampled"],
        "n_observation": acc["observations"]["n_observation"],
        "iterations": acc["iterations"],
        "adjoint_solves": acc["adjoint_solves"],
        "adjoint_budget": acc["adjoint_iteration_budget"],
        "stop_reason": acc["stop_reason"],
        "field_rel_l2": acc["field_rel_l2"],
        "relative_data_residual": acc["relative_data_residual"],
        "acceptance": acc["acceptance"],
        "condition_number": acc["observability"]["condition_number"],
        "condition_number_full_sampling": obs_rows[-1]["condition_number"],
        "out_of_span_field_rel_l2": oos["field_rel_l2"],
        "nodal_field_rel_l2": nodal["field_rel_l2"],
        "nodal_null_space_dim": nodal["null_space_dim"],
        "noise_1pct_field_rel_l2_undamped": next(
            row["field_rel_l2"]
            for row in sweep
            if row["noise_relative"] == 0.01 and row["alpha_rel"] == 1.0e-12
        ),
        "noise_1pct_field_rel_l2_damped": next(
            row["field_rel_l2"]
            for row in sweep
            if row["noise_relative"] == 0.01 and row["alpha_rel"] == 1.0e-6
        ),
        "patch_max_relative_error": max(patch.values()),
        "manufactured_rates": [round(r, 3) for r in manufactured["observed_rates"]],
        "manufactured_finest_error": manufactured["rows"][-1]["rel_l2_interior"],
        "wall_clock_s": round(time.time() - t0, 2),
    }
    _dump("summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _fig_convergence(res: dict) -> None:
    it = [entry["iteration"] for entry in res["convergence"]]
    field = [entry["field_rel_l2"] for entry in res["convergence"]]
    resid = [entry["relative_data_residual"] for entry in res["convergence"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(it, field, "o-", label=r"relative $L_2$ field error")
    ax.semilogy(it, resid, "s--", label="relative data residual")
    ax.axhline(0.05, color="crimson", lw=1.0, ls=":", label="5 % acceptance band")
    ax.set_xlabel("adjoint iteration (1 forward + 1 adjoint solve each)")
    ax.set_ylabel("relative error / residual")
    ax.set_title(
        f"Body-force inversion, {res['observations']['fraction']*100:.1f} % of nodes observed, "
        f"{res['basis']['n_unknown']} unknowns"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_convergence.png", dpi=150)
    plt.close(fig)


def _fig_fields(res: dict) -> None:
    basis = ModalBasis(mesh=MESH, order=2)
    rec = res["coefficients"]["recovered"]
    truth = res["coefficients"]["truth"]
    xx, yy, bx_t, by_t = _field_grid(basis, truth)
    _, _, bx_r, by_r = _field_grid(basis, rec)
    panels = [
        (bx_t, r"true $b_x$", "viridis"),
        (bx_r, r"recovered $b_x$", "viridis"),
        (by_t, r"true $b_y$", "magma"),
        (by_r, r"recovered $b_y$", "magma"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 4.6), sharex=True, sharey=True)
    for ax, (field, title, cmap) in zip(axes.reshape(-1), panels):
        im = ax.imshow(
            field,
            origin="lower",
            extent=[0.0, MESH.Lx, 0.0, MESH.Ly],
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes[1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.suptitle(
        f"Volume force field: truth vs recovery at "
        f"{res['observations']['fraction']*100:.1f} % node sampling "
        f"(rel. $L_2$ error {res['field_rel_l2']:.2e})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "fig_fields.png", dpi=150)
    plt.close(fig)


def _fig_noise(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for alpha in sorted({row["alpha_rel"] for row in rows}):
        sub = [row for row in rows if row["alpha_rel"] == alpha]
        noise = [max(row["noise_relative"], 1e-6) for row in sub]
        err = [row["field_rel_l2"] for row in sub]
        ax.loglog(noise, err, "o-", label=f"$\\alpha_{{rel}}$ = {alpha:g}")
    ax.axhline(0.05, color="crimson", lw=1.0, ls=":", label="5 % acceptance band")
    ax.set_xlabel("relative sensor noise on the sampled displacements")
    ax.set_ylabel(r"relative $L_2$ field error")
    ax.set_title("Noise robustness at 10 % node sampling (8 unknowns)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_noise.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())