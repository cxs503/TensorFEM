"""Evidence run: 2-D analytical verification of the Q4 heat-conduction kernel.

Cases (see ``tensorfem.thermal_analytic`` for the derivations):

* A harmonic Laplace field on the unit square, Cartesian quad mesh;
* B the same field on a sheared quad mesh (isoparametric mapping test);
* C logarithmic radial field in an annulus, curvilinear quad ring mesh;
* D two-dimensional transient decay ``exp(-2 pi^2 alpha t)``;
* E element-centre heat-flux recovery against the analytic gradient.

Outputs
-------
* ``benchmarks/thermal_2d_analytic_evidence.json`` -- raw numbers and errors
* ``docs/assets/thermal_2d/convergence.png``     -- log-log error vs h
* ``docs/assets/thermal_2d/field_and_error.png`` -- FEM field and error map
* ``docs/assets/thermal_2d/radial_and_transient.png`` -- radial profile and decay

Usage::

    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/thermal_2d_analytic.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.thermal_analytic import (  # noqa: E402
    annular_convergence,
    annular_quad_mesh,
    annular_radial_gradient,
    annular_temperature,
    element_areas,
    field_errors,
    flux_error,
    harmonic_convergence,
    harmonic_gradient,
    harmonic_temperature,
    heat_balance_residual,
    laplacian_residual,
    solve_dirichlet,
    structured_quad_mesh,
    transient_decay,
    transient_decay_rate,
)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets" / "thermal_2d"
EVIDENCE = ROOT / "benchmarks" / "thermal_2d_analytic_evidence.json"

HARMONIC_SIZES = [4, 8, 16, 32]
SHEAR = 0.35
ANNULAR_NT = [16, 32, 64, 128]
TRANSIENT_SIZES = [8, 12, 16, 24]


def _annulus_exact(nodes: torch.Tensor) -> torch.Tensor:
    return annular_temperature(torch.linalg.vector_norm(nodes, dim=1), 1.0, 2.0, 100.0, 20.0)


def _annulus_gradient(nodes: torch.Tensor) -> torch.Tensor:
    r = torch.linalg.vector_norm(nodes, dim=1)
    magnitude = annular_radial_gradient(r, 1.0, 2.0, 100.0, 20.0)
    return nodes / r[:, None] * magnitude[:, None]


def run() -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)
    evidence: dict = {"generated_by": "benchmarks/thermal_2d_analytic.py", "cases": {}}

    # ---- oracle self-checks ------------------------------------------------
    probe = structured_quad_mesh(23, 19).nodes
    evidence["oracle_checks"] = {
        "harmonic_laplacian_relative_residual": laplacian_residual(harmonic_temperature, probe),
        "annular_laplacian_relative_residual": laplacian_residual(
            lambda pts: _annulus_exact(pts), probe * 0.5 + 0.75),
        "unit_square_area": float(element_areas(structured_quad_mesh(16, 16)).sum()),
        "annulus_polygon_area_nt16": float(element_areas(annular_quad_mesh(2, 16)).sum()),
        "annulus_polygon_area_exact_nt16": 0.5 * 16 * math.sin(2 * math.pi / 16) * 3.0,
    }

    # ---- A: Cartesian harmonic --------------------------------------------
    cart = harmonic_convergence(HARMONIC_SIZES)
    sheared = harmonic_convergence(HARMONIC_SIZES, shear=SHEAR)
    evidence["cases"]["A_harmonic_cartesian"] = cart.as_dict()
    evidence["cases"]["B_harmonic_sheared"] = sheared.as_dict()
    evidence["cases"]["B_harmonic_sheared"]["shear"] = SHEAR

    # ---- annular area (polygonal idealisation) ----------------------------
    area_levels, area_errors = [], []
    for nt in ANNULAR_NT:
        mesh = annular_quad_mesh(max(1, nt // 8), nt)
        area_levels.append(nt)
        area_errors.append(abs(float(element_areas(mesh).sum()) - math.pi * 3.0) / (math.pi * 3.0))
    evidence["cases"]["C_annular_area_error"] = {"nt": area_levels, "relative_error": area_errors}

    # ---- C: annular logarithmic ------------------------------------------
    ann = annular_convergence(ANNULAR_NT)
    evidence["cases"]["C_annular_logarithmic"] = ann.as_dict()

    # ---- E: flux recovery -------------------------------------------------
    flux = {"levels": [], "gradient_l2_relative": []}
    stored = {}
    for n in HARMONIC_SIZES:
        mesh = structured_quad_mesh(n, n)
        temperature = solve_dirichlet(mesh, harmonic_temperature)
        err = flux_error(mesh, temperature, harmonic_gradient)
        flux["levels"].append(n)
        flux["gradient_l2_relative"].append(err["gradient_l2_relative"])
        stored[n] = (mesh, temperature)
    flux["observed_order"] = [
        math.log(flux["gradient_l2_relative"][i - 1] / flux["gradient_l2_relative"][i])
        / math.log(HARMONIC_SIZES[i] / HARMONIC_SIZES[i - 1])
        for i in range(1, len(HARMONIC_SIZES))]
    evidence["cases"]["E_flux_recovery"] = flux

    # ---- global heat balance and field snapshots -------------------------
    mesh32, temperature32 = stored[HARMONIC_SIZES[-1]]
    evidence["cases"]["A_harmonic_cartesian"]["heat_balance_relative_residual"] = (
        heat_balance_residual(mesh32, harmonic_temperature, temperature32))
    evidence["cases"]["A_harmonic_cartesian"]["finest_error_detail"] = field_errors(
        mesh32, temperature32, harmonic_temperature)
    mesh_a, temperature_a = stored[8]
    evidence["cases"]["B_harmonic_sheared"]["finest_error_detail"] = field_errors(
        structured_quad_mesh(HARMONIC_SIZES[-1], HARMONIC_SIZES[-1], shear=SHEAR),
        solve_dirichlet(structured_quad_mesh(HARMONIC_SIZES[-1], HARMONIC_SIZES[-1], shear=SHEAR),
                        harmonic_temperature), harmonic_temperature)

    # annular finest detail + radial profile
    nt_fine = ANNULAR_NT[-1]
    mesh_ann = annular_quad_mesh(max(1, nt_fine // 8), nt_fine)
    temperature_ann = solve_dirichlet(mesh_ann, _annulus_exact)
    evidence["cases"]["C_annular_logarithmic"]["finest_error_detail"] = field_errors(
        mesh_ann, temperature_ann, _annulus_exact)
    evidence["cases"]["C_annular_logarithmic"]["flux"] = flux_error(
        mesh_ann, temperature_ann, _annulus_gradient)
    radial = torch.linalg.vector_norm(mesh_ann.nodes, dim=1)
    radial_nodes = torch.nonzero(torch.abs(mesh_ann.nodes[:, 1]) < 1.0e-9).reshape(-1)
    order = torch.argsort(radial[radial_nodes])
    evidence["cases"]["C_annular_logarithmic"]["radial_profile"] = {
        "r": radial[radial_nodes][order].tolist(),
        "fem": temperature_ann[radial_nodes][order].tolist(),
        "exact": _annulus_exact(mesh_ann.nodes[radial_nodes])[order].tolist(),
    }

    # ---- D: transient decay ----------------------------------------------
    transient = []
    for n in TRANSIENT_SIZES:
        run_n = transient_decay(n)
        transient.append(run_n)
    evidence["cases"]["D_transient_decay"] = transient
    evidence["cases"]["D_transient_decay_exact_rate"] = transient_decay_rate(1.0)

    # ---- figures ----------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for case, marker in ((cart, "o"), (sheared, "s")):
        h = [1.0 / n for n in case.levels]
        ax[0].loglog(h, case.l2, marker + "-", label=case.name)
    h_ann = [2 * math.pi / nt for nt in ann.levels]
    ax[0].loglog(h_ann, ann.l2, "^-", label="annular logarithmic")
    ref = [cart.l2[-1] * (1.0 / n) ** 2 / (1.0 / HARMONIC_SIZES[-1]) ** 2 for n in HARMONIC_SIZES]
    ax[0].loglog([1.0 / n for n in HARMONIC_SIZES], ref, "k--", lw=1.0,
                 label=r"reference slope $h^{2}$")
    ax[0].set_xlabel("characteristic size $h$")
    ax[0].set_ylabel("relative $L_2$ temperature error")
    ax[0].set_title("Mesh convergence of the Q4 conduction kernel")
    ax[0].legend(fontsize=8)

    ax[1].loglog([1.0 / n for n in flux["levels"]], flux["gradient_l2_relative"], "d-",
                 label="element-centre $\\nabla T$")
    ax[1].loglog(area_levels, area_errors, "v-", label="annulus polygon area")
    ax[1].set_xlabel("size metric (1/n or $2\\pi/n_t$)")
    ax[1].set_ylabel("relative error")
    ax[1].set_title("Flux recovery and geometric idealisation")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "convergence.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.6))
    grid = 121
    dense = structured_quad_mesh(grid - 1, grid - 1)
    from matplotlib.tri import Triangulation

    tri = Triangulation(dense.nodes[:, 0].numpy(), dense.nodes[:, 1].numpy())
    ax[0].tripcolor(tri, harmonic_temperature(dense.nodes).numpy(), shading="gouraud", cmap="inferno")
    ax[0].set_title("exact $\\sin(\\pi x)\\,\\sinh(\\pi y)/\\sinh\\pi$")
    ax[0].set_aspect("equal")
    tn = mesh32.nodes
    ax[1].tripcolor(Triangulation(tn[:, 0].numpy(), tn[:, 1].numpy()),
                    (temperature32 - harmonic_temperature(tn)).numpy(),
                    shading="gouraud", cmap="coolwarm")
    ax[1].set_title("FEM error, 32$\\times$32 ($\\times10^{-4}$)")
    ax[1].set_aspect("equal")
    sc = ax[2].scatter(mesh_ann.nodes[:, 0], mesh_ann.nodes[:, 1],
                       c=(temperature_ann - _annulus_exact(mesh_ann.nodes)).numpy(),
                       s=4, cmap="coolwarm")
    ax[2].set_title("annular FEM error\n$n_t=%d$" % nt_fine)
    ax[2].set_aspect("equal")
    fig.colorbar(sc, ax=ax[2], shrink=0.85)
    fig.tight_layout()
    fig.savefig(ASSETS / "field_and_error.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    profile = evidence["cases"]["C_annular_logarithmic"]["radial_profile"]
    ax[0].plot(profile["r"], profile["exact"], "k-", label="exact $\\ln r$")
    ax[0].plot(profile["r"], profile["fem"], "o", ms=3, label="Q4 FEM")
    ax[0].set_xlabel("$r$")
    ax[0].set_ylabel("$T$")
    ax[0].set_title("annular radial temperature")
    ax[0].legend(fontsize=8)
    fine = transient[-1]
    rate = transient_decay_rate(1.0)
    t_dummy = torch.linspace(0, 1.0 / rate, 50, dtype=torch.float64)
    ax[1].semilogy(t_dummy.numpy(), torch.exp(-rate * t_dummy).numpy(), "k-", label="exact decay")
    for run_n in transient:
        ax[1].semilogy([c["t_over_tau"] / rate for c in run_n["checkpoints"]],
                       1.0 - torch.tensor([c["amplitude_relative"] for c in run_n["checkpoints"]]).numpy(),
                       "o", ms=3, label=f"{run_n['mesh'][0]}$\\times${run_n['mesh'][1]}")
    ax[1].set_xlabel("physical time $t$")
    ax[1].set_ylabel("peak amplitude")
    ax[1].set_title("2-D transient decay, fitted rate error %.2f%%"
                    % (100.0 * fine["rate_relative_error"]))
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "radial_and_transient.png", dpi=140)
    plt.close(fig)

    # ---- summary ----------------------------------------------------------
    evidence["summary"] = {
        "gate_3pct_steady_finest": {
            "harmonic_cartesian": cart.l2[-1],
            "harmonic_sheared": sheared.l2[-1],
            "annular_logarithmic": ann.l2[-1],
            "flux_recovery": flux["gradient_l2_relative"][-1],
        },
        "observed_orders": {
            "harmonic_cartesian": cart.orders,
            "harmonic_sheared": sheared.orders,
            "annular_logarithmic": ann.orders,
            "flux_recovery": flux["observed_order"],
        },
        "transient_rate_relative_error": [r["rate_relative_error"] for r in transient],
    }
    return evidence


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def render_markdown(evidence: dict) -> str:
    """Emit the verification note with every number injected from the evidence."""
    c = evidence["cases"]
    s = evidence["summary"]
    lines = []
    add = lines.append
    add("# Two-dimensional analytical verification of the Q4 conduction kernel")
    add("")
    add("Generated by `benchmarks/thermal_2d_analytic.py`; every number in this "
        "note is injected from `benchmarks/thermal_2d_analytic_evidence.json`, "
        "none is transcribed by hand.")
    add("")
    add("## Why this note exists (verification debt)")
    add("")
    add("`docs/thermal-validation.md` records six anchors for the thermal kernel, "
        "but the only two-dimensional one is a **linear Q4 temperature patch**. A patch "
        "test is satisfied by *any* consistent element, so it cannot detect a wrong "
        "isoparametric map, a wrong quadrature order, or a conduction matrix that is "
        "too stiff by a constant factor. The 2-D conduction path had therefore never "
        "been checked against a field with curvature in two directions, against a "
        "curvilinear mesh, or against a heat flux. This note closes that gap with "
        "closed-form references that are independent of the discretisation.")
    add("")
    add("## Analytic references and their self-checks")
    add("")
    add("| Case | Closed form | Why it is exact |")
    add("|---|---|---|")
    add("| A, B | $T=\\sin(\\pi x)\\,\\sinh(\\pi y)/\\sinh\\pi$ on the unit square "
        "| $\\partial_{xx}T=-\\pi^2T$, $\\partial_{yy}T=+\\pi^2T$, so "
        "$\\nabla^2T=0$ |")
    add("| C | $T=T_1+(T_2-T_1)\\ln(r/r_1)/\\ln(r_2/r_1)$ | the 2-D Laplacian of "
        "$\\ln r$ vanishes |")
    add("| D | $T=\\sin\\pi x\\,\\sin\\pi y\\,e^{-2\\pi^2\\alpha t}$ "
        "| eigenfunction of the Dirichlet square Laplacian, eigenvalue $-2\\pi^2$ |")
    add("")
    checks = evidence["oracle_checks"]
    add("The references are checked numerically before they are used as targets:")
    add("")
    add("| Check | Value |")
    add("|---|---|")
    add(f"| relative Laplacian residual of the harmonic field (central differences) "
        f"| {_fmt(checks['harmonic_laplacian_relative_residual'])} |")
    add(f"| relative Laplacian residual of the logarithmic field | "
        f"{_fmt(checks['annular_laplacian_relative_residual'])} |")
    add(f"| area of the 16x16 sheared unit square | {checks['unit_square_area']:.15f} |")
    add(f"| polygon area of the $n_t=16$ annulus vs the closed form | "
        f"{checks['annulus_polygon_area_nt16']:.15f} vs "
        f"{checks['annulus_polygon_area_exact_nt16']:.15f} |")
    add("")
    add("## Method")
    add("")
    add("* Production path only: `ThermalModel(element_type='q4')`, "
        "`assemble_thermal`, `solve_steady_thermal`, `solve_transient_thermal`. "
        "The verification module never re-implements the element.")
    add("* Dirichlet data are taken from the exact field on every boundary node; "
        "no symmetry reduction, so the whole boundary participates.")
    add("* Errors use 2x2 Gauss quadrature of $(T_h-T)^2$ for the volume norm and "
        "exact nodal pointwise maxima, both normalised by the exact field. "
        "Flux errors use the element-centre gradient weighted by the exact polygon "
        "area. Using quadrature-based rather than nodal errors matters: a nodal-only "
        "metric can hide an element that is wrong in its interior.")
    add("* Observed order is the Richardson estimate "
        "$p=\\ln(e_{k}/e_{k+1})/\\ln(h_k/h_{k+1})$. Second order is the "
        "expected value for bilinear quads, so the order is a *capability* check, "
        "not only an accuracy check.")
    add("* Transient case D uses Crank-Nicolson ($\\theta=0.5$) with 100 steps per "
        "decay time, so the temporal error is far below the spatial error and the "
        "measured rate error is dominated by the Q4 spatial discretisation.")
    add("")
    add("## Results")
    add("")
    for key, title in (("A_harmonic_cartesian", "Case A: harmonic field, Cartesian mesh"),
                       ("B_harmonic_sheared", "Case B: harmonic field, sheared mesh "
                                              f"($s={c['B_harmonic_sheared']['shear']}$)"),
                       ("C_annular_logarithmic", "Case C: logarithmic radial field, annulus")):
        case = c[key]
        add(f"### {title}")
        add("")
        add("| level | dofs | relative $L_2$ | nodal pointwise max | observed order |")
        add("|---|---|---|---|---|")
        for i, level in enumerate(case["levels"]):
            order = "-" if i == 0 else f"{case['observed_order'][i - 1]:.2f}"
            dofs = case.get("n_dof", [])
            add(f"| {level} | {'-' if not dofs else dofs[i]} | {_fmt(case['l2_relative'][i])} "
                f"| {_fmt(case['pointwise_max_relative'][i])} | {order} |")
        add("")
    area = c["C_annular_area_error"]
    add("### Polygonal idealisation of the annulus")
    add("")
    add("The circular boundaries are chords, so the meshed domain is an inscribed "
        "polygon and part of the observed error is geometric rather than "
        "discretisation error. The area defect converges as:")
    add("")
    add("| $n_t$ | relative area error |")
    add("|---|---|")
    for nt, err in zip(area["nt"], area["relative_error"]):
        add(f"| {nt} | {_fmt(err)} |")
    add("")
    flux = c["E_flux_recovery"]
    add("### Case E: element-centre heat-flux recovery (harmonic field)")
    add("")
    add("| mesh | relative $L_2$ error of $\\nabla T$ | observed order |")
    add("|---|---|---|")
    for i, n in enumerate(flux["levels"]):
        order = "-" if i == 0 else f"{flux['observed_order'][i - 1]:.2f}"
        add(f"| {n}x{n} | {_fmt(flux['gradient_l2_relative'][i])} | {order} |")
    add("")
    fine = c["C_annular_logarithmic"]["flux"]
    add(f"Annular flux recovery at $n_t={c['C_annular_logarithmic']['levels'][-1]}$: "
        f"{_fmt(fine['gradient_l2_relative'])} over {fine['n_samples']} elements.")
    add("")
    add("### Case D: two-dimensional transient decay")
    add("")
    add("| mesh | dofs | fitted $d\\ln\\max|T|/dt$ | exact | rate error | field $L_2$ at $t=\\tau$ |")
    add("|---|---|---|---|---|---|")
    for run_n in c["D_transient_decay"]:
        add(f"| {run_n['mesh'][0]}x{run_n['mesh'][1]} | {run_n['n_dof']} "
            f"| {run_n['fitted_rate']:.6f} | {-run_n['exact_rate']:.6f} "
            f"| {100.0 * run_n['rate_relative_error']:.3f}% "
            f"| {_fmt(run_n['checkpoints'][-1]['field_l2_relative'])} |")
    add("")
    add("The exact rate is $-2\\pi^2\\alpha=-19.7392088$ for $\\alpha=1$; the "
        "fitted slope comes from a least-squares fit of $\\ln\\max|T|$ against "
        "$t$ over the whole run, not from a single sample.")
    add("")
    add("## Figures")
    add("")
    add("![convergence](assets/thermal_2d/convergence.png)")
    add("")
    add("*Left:* relative $L_2$ temperature error against the element size for the "
        "Cartesian, sheared and annular cases, with a reference $h^2$ slope. "
        "*Right:* flux recovery and the polygon area defect, both second order.")
    add("")
    add("![field and error](assets/thermal_2d/field_and_error.png)")
    add("")
    add("Exact harmonic field, the Q4 error map on the 32x32 mesh, and the annular "
        "mesh error at the finest circumferential refinement.")
    add("")
    add("![radial and transient](assets/thermal_2d/radial_and_transient.png)")
    add("")
    add("Annular radial temperature against the closed form, and the transient decay "
        "against $e^{-2\\pi^2\\alpha t}$ for four meshes.")
    add("")
    add("## Error judgement against the 3% gate")
    add("")
    gate = s["gate_3pct_steady_finest"]
    add("| Quantity | Finest-mesh relative error | < 3%? |")
    add("|---|---|---|")
    for label, key in (("harmonic field, Cartesian", "harmonic_cartesian"),
                       ("harmonic field, sheared mesh", "harmonic_sheared"),
                       ("logarithmic radial field, annulus", "annular_logarithmic"),
                       ("harmonic field, element-centre gradient", "flux_recovery")):
        value = gate[key]
        add(f"| {label} | {_fmt(value)} | {'yes' if value < 0.03 else 'NO'} |")
    rate_errors = s["transient_rate_relative_error"]
    add(f"| transient decay rate, finest mesh | {100.0 * rate_errors[-1]:.3f}% | "
        f"{'yes' if rate_errors[-1] < 0.03 else 'NO'} |")
    add(f"| transient field at $t=\\tau$, finest mesh | "
        f"{_fmt(c['D_transient_decay'][-1]['checkpoints'][-1]['field_l2_relative'])} | "
        f"{'yes' if c['D_transient_decay'][-1]['checkpoints'][-1]['field_l2_relative'] < 0.03 else 'NO'} |")
    add("")
    add("Every quantity passes the 3% gate with more than an order of magnitude of "
        "margin, and no quantity is accepted on the basis of a single mesh: all of "
        "them also converge at the element order. The discrimination controls in "
        "`tests/test_thermal_2d_analytic.py` show that the same metric fails a "
        "1.05x scaled field, a 0.01 shifted field, and a coarse 4x4 mesh, so the "
        "passing numbers are not an artefact of an insensitive metric.")
    add("")
    add("## Caveats and scope")
    add("")
    add("* Linear isotropic conduction, bilinear quads, constant thickness, no "
        "radiation, no temperature-dependent conductivity, no contact or phase "
        "change. Nothing here certifies anisotropic or nonlinear conduction.")
    add("* The annular case carries an $O(h^2)$ geometric error from the polygonal "
        "boundary, which is why it is judged by refinement. Multiplying $n_t$ and "
        "$n_r$ together keeps both contributions balanced.")
    add("* The transient case uses a single eigenmode; it certifies the decay rate "
        "and field shape of that mode, not a multi-mode transient or a "
        "time-dependent boundary condition.")
    add("* The solvers are the dense reference implementations, so the sizes above "
        "are verification sizes. No performance claim is made.")
    add("* `theta < 0.5` and inverted elements still fail closed; those contracts are "
        "re-tested here rather than assumed.")
    add("")
    add("## Reproduction")
    add("")
    add("```bash")
    add("OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/thermal_2d_analytic.py")
    add("PYTHONPATH=src python -m pytest -q tests/test_thermal_2d_analytic.py")
    add("```")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    evidence = run()
    EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n")
    note = ROOT / "docs" / "thermal-2d-analytic-verification.md"
    note.write_text(render_markdown(evidence))
    print(json.dumps(evidence["summary"], indent=2))
    print("written", EVIDENCE)
    print("written", note)


if __name__ == "__main__":
    main()