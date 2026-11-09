"""Reproducible study driver for ``tensorfem.polar_thermal_stress``.

Runs the whole chain (verification anchors -> temperature field -> thermal
stress -> ice load -> limit state), writes every number to
``artifacts/polar_thermal_stress/results.json`` and the paper figures next to it.

    python scripts/polar_thermal_stress_study.py

Machine-readable, no hand-copied numbers: the report tables are emitted by this
script as ``report_tables.md`` and the figures are rendered from the same dict
that is dumped to JSON.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # 128-core shared host: launch overhead dominates tiny dense kernels

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tensorfem.polar_thermal_stress as pts  # noqa: E402
from tensorfem.polar_thermal_stress import (  # noqa: E402
    DTYPE,
    FacePatch,
    IceContactPatch,
    PolarEnvironment,
    PolarPanel,
    POLAR_STEELS,
    assemble_elasticity,
    assemble_pressure_load,
    assemble_thermal_load,
    clamped_plate_bending,
    clamped_plate_uniform_reference,
    faces_on_plane,
    isotropic_D,
    layered_slab_mesh,
    lumped_capacitance,
    patch_area,
    polar_thermal_limit_state_matrix,
    restrained_biaxial_stress,
    solve_dirichlet,
    solve_panel_stress,
    solve_panel_temperature,
    solve_steady_conduction,
    structured_hex_mesh,
    transient_conduction,
    wall_analytic_1d,
)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "polar_thermal_stress"
OUT.mkdir(parents=True, exist_ok=True)

STEEL = POLAR_STEELS["EH36"]
T_AIR_CASES = (-40.0, -30.0, -20.0, -10.0, 0.0)
ICE_PRESSURES = (0.5e6, 1.0e6, 2.0e6)
GAMMA_M = 1.15


# ---------------------------------------------------------------------------
# verification anchors
# ---------------------------------------------------------------------------

def verify_wall() -> dict:
    layers = [(0.003, 0.3), (0.025, 45.0)]
    h_out, h_in, T_air, T_in = 30.0, 8.0, -40.0, 5.0
    mesh = layered_slab_mesh([(0.003, 2), (0.025, 5)])
    k = torch.tensor([0.3] * 2 + [45.0] * 5, dtype=DTYPE)
    outer, po = faces_on_plane(mesh, 0, 0.0)
    inner, pi = faces_on_plane(mesh, 0, 0.028)
    T = solve_steady_conduction(mesh, k, [FacePatch.of(outer, po, h_out, T_air),
                                          FacePatch.of(inner, pi, h_in, T_in)])
    x = mesh.nodes[:, 0].tolist()
    ana = wall_analytic_1d(layers, T_air, h_out, T_in, h_in, x_samples=sorted(set(x)))
    # nodal comparison on the unique x stations
    fe = {}
    for xi, ti in zip(x, T.tolist()):
        fe.setdefault(round(xi, 9), []).append(ti)
    stations = sorted(fe)
    fe_mean = [sum(fe[s]) / len(fe[s]) for s in stations]
    err = max(abs(a - b) for a, b in zip(fe_mean, ana["temperature"]))
    # also the single-plate (bare steel, no coating) closed form
    bare = wall_analytic_1d([(0.025, 45.0)], -40.0, 60.0, 5.0, 500.0,
                            x_samples=[0.0, 0.025])
    return {
        "layers": [{"thickness_m": t, "k_W_mK": kk} for t, kk in layers],
        "h_outer_W_m2K": h_out,
        "h_inner_W_m2K": h_in,
        "T_air_C": T_air,
        "T_inner_C": T_in,
        "heat_flux_outward_W_m2": ana["heat_flux_outward"],
        "resistance_m2K_W": ana["resistance"],
        "x_stations_m": stations,
        "T_fem_C": fe_mean,
        "T_analytic_C": ana["temperature"],
        "max_abs_error_K": err,
        "max_rel_error": err / (T_in - T_air),
        "criterion": 0.01,
        "bare_steel_wall": bare,
    }


def verify_biaxial() -> dict:
    """Biaxially restrained uniform dT against E alpha dT / (1 - nu)."""
    mesh = structured_hex_mesh(2, 2, 2, 0.025, 0.1, 0.1)
    rows = []
    for dT in (-60.0, -40.0, -20.0):
        T_hold, T_body = 20.0, 20.0 + dT
        E = float(STEEL.E(torch.tensor([T_body]))[0])
        alpha = float(STEEL.alpha(torch.tensor([T_body]))[0])
        K = assemble_elasticity(mesh, E, STEEL.nu)
        f = assemble_thermal_load(mesh, E, alpha,
                                  torch.full((mesh.n_elements,), T_body, dtype=DTYPE),
                                  T_hold, STEEL.nu)
        X = mesh.nodes
        fixed = []
        for i in range(mesh.n_nodes):
            fixed += [3 * i + 1, 3 * i + 2]
        fixed += [3 * i for i in range(mesh.n_nodes) if abs(float(X[i, 0])) < 1e-12]
        u = solve_dirichlet(K, f, torch.tensor(fixed), torch.zeros(len(fixed), dtype=DTYPE))
        gd = pts._hex8_gauss_data(mesh)
        B = pts._hex8_B(gd)
        D = isotropic_D(E, STEEL.nu)
        ue = u.reshape(mesh.n_nodes, 3)[mesh.elements].reshape(mesh.n_elements, 24)
        eps = torch.einsum("egki,ei->egk", B, ue)
        sc = alpha * (T_body - T_hold)
        eps_th = torch.zeros_like(eps)
        eps_th[:, :, 0] = sc
        eps_th[:, :, 1] = sc
        eps_th[:, :, 2] = sc
        sigma = torch.einsum("kl,egl->egk", D, eps - eps_th).mean(dim=1)
        analytic = restrained_biaxial_stress(E, alpha, dT, STEEL.nu)
        fem = float(sigma[0, 1])
        rows.append({
            "dT_K": dT, "E_Pa": E, "alpha_per_K": alpha,
            "sigma_fem_Pa": fem, "sigma_analytic_Pa": analytic,
            "rel_error": abs(abs(fem) - abs(analytic)) / abs(analytic),
            "sigma_xx_Pa": float(sigma[0, 0]),
        })
    return {"rows": rows, "criterion": 0.02,
            "max_rel_error": max(r["rel_error"] for r in rows)}


def verify_material() -> dict:
    out = {}
    for name, steel in POLAR_STEELS.items():
        tab = pts.MATERIAL_ANCHORS[name]
        out[name] = {
            "E_ref_GPa": steel.E_ref / 1e9,
            "dE_dT_GPa_per_K": steel.dE_dT / 1e9,
            "alpha_ref_per_K": steel.alpha_ref,
            "dalpha_dT_per_K2": steel.dalpha_dT,
            "yield_ref_MPa": steel.yield_ref / 1e6,
            "dyield_dT_MPa_per_K": steel.dyield_dT / 1e6,
            "nu": steel.nu, "k_W_mK": steel.conductivity,
            "anchor_residual": steel.anchor_residual,
            "measured_residual": pts.material_anchor_residuals(steel),
            "anchor_temperature_C": list(tab["temperature_C"]),
            "anchor_E_GPa": list(tab["E_GPa"]),
            "anchor_alpha_1e6": list(tab["alpha_1e-6_per_K"]),
            "anchor_yield_MPa": list(tab["yield_MPa"]),
            "provenance": steel.source,
            "model_E_GPa": [steel.E(t) / 1e9 for t in tab["temperature_C"]],
            "model_alpha_1e6": [steel.alpha(t) * 1e6 for t in tab["temperature_C"]],
            "model_yield_MPa": [steel.yield_strength(t) / 1e6 for t in tab["temperature_C"]],
        }
    return {"materials": out, "criterion": 0.05,
            "max_rel_error": max(max(m["measured_residual"].values()) for m in out.values())}


def verify_transient() -> dict:
    thickness = 0.002
    mesh = structured_hex_mesh(1, 2, 2, thickness, 0.05, 0.05)
    outer, po = faces_on_plane(mesh, 0, 0.0)
    patch = FacePatch.of(outer, po, 20.0, -40.0)
    rho_cp = STEEL.density * STEEL.specific_heat
    tau = STEEL.density * STEEL.specific_heat * thickness / 20.0
    sol = transient_conduction(mesh, STEEL.conductivity, rho_cp, [patch],
                              T0=20.0, dt=20.0, n_steps=100)
    ana = lumped_capacitance(sol.times.tolist(), 20.0, -40.0, tau)
    fe = sol.temperatures.mean(dim=1)
    err = float((fe - ana).abs().max())
    long_run = transient_conduction(mesh, STEEL.conductivity, rho_cp, [patch],
                                    T0=20.0, dt=50.0, n_steps=400)
    steady = solve_steady_conduction(mesh, STEEL.conductivity, [patch])
    bi = 20.0 * thickness / STEEL.conductivity
    return {
        "thickness_m": thickness, "h_W_m2K": 20.0, "tau_s": tau, "biot": bi,
        "times_s": sol.times.tolist(),
        "T_fem_C": fe.tolist(),
        "T_lumped_C": ana.tolist(),
        "max_abs_error_K": err,
        "max_rel_error_of_span": err / 60.0,
        "long_run_final_C": float(long_run.final.mean()),
        "steady_C": float(steady.mean()),
        "criterion": 0.02,
    }


def verify_plate() -> dict:
    E, nu, t, p, L = STEEL.E(20.0), STEEL.nu, 0.025, 0.125e6, 1.2
    ref = clamped_plate_uniform_reference(L, L, t, p, E, nu)
    rows = []
    for n in (24, 48, 72):
        res = clamped_plate_bending(L, L, t, E, nu, p, n_y=n, n_z=n)
        rows.append({
            "n_per_side": n, "h_m": res.h,
            "sigma_max_MPa": res.sigma_max / 1e6,
            "w_max_mm": res.w_max * 1e3,
            "sigma_rel_error": abs(res.sigma_max - ref["sigma_max_Pa"]) / ref["sigma_max_Pa"],
            "w_rel_error": abs(res.w_max - ref["w_max_m"]) / ref["w_max_m"],
            "total_load_N": res.total_load,
        })
    return {"reference": {k: v for k, v in ref.items() if k != "source"},
            "reference_source": ref["source"], "criterion": 0.02,
            "rows": rows,
            "max_sigma_rel_error": max(r["sigma_rel_error"] for r in rows),
            "max_w_rel_error": max(r["w_rel_error"] for r in rows)}


def verify_pressure_reaction() -> dict:
    panel = PolarPanel(n_thickness=3, n_y=8, n_z=12)
    mesh = panel.mesh()
    ice = IceContactPatch(length=0.3, width=0.3, pressure=3.0e6)
    patch = pts.ice_patch_faces(mesh, panel, ice)
    f = assemble_pressure_load(mesh, patch, ice.pressure)
    reaction = float(-f.reshape(-1, 3)[:, 0].sum())
    area = patch_area(mesh, patch.nodes)
    return {
        "pressure_MPa": ice.pressure / 1e6,
        "nominal_patch_area_m2": ice.area,
        "discretised_loaded_area_m2": area,
        "reaction_N": reaction,
        "p_times_area_N": ice.pressure * area,
        "rel_error": abs(reaction - ice.pressure * area) / (ice.pressure * area),
        "criterion": 1e-9,
    }


def hex8_bending_limitation() -> dict:
    """Quantify the solid-hex8 bending deficit that motivates the plate route."""
    E, nu, P, L, t, b = STEEL.E(20.0), STEEL.nu, 10.0, 1.2, 0.025, 0.1
    EI = E * b * t ** 3 / 12.0
    exact = -P * L ** 3 / (3.0 * EI)
    rows = []
    for nx, ny, nz in ((2, 2, 12), (4, 3, 24), (6, 4, 48)):
        mesh = structured_hex_mesh(nx, ny, nz, t, b, L)
        X = mesh.nodes
        nodes = torch.nonzero(X[:, 2] <= 1e-12, as_tuple=False).reshape(-1)
        fixed = (3 * nodes[:, None] + torch.arange(3)).reshape(-1)
        K = assemble_elasticity(mesh, E, nu)
        tip = torch.nonzero(X[:, 2] >= L - 1e-12, as_tuple=False).reshape(-1)
        f = torch.zeros(3 * mesh.n_nodes, dtype=DTYPE)
        f[3 * tip] = -P / len(tip)
        u = solve_dirichlet(K, f, fixed, torch.zeros(fixed.numel(), dtype=DTYPE))
        w = float(u.reshape(-1, 3)[tip, 0].mean())
        rows.append({"mesh": [nx, ny, nz], "w_fem_m": w, "w_beam_m": exact,
                     "ratio": w / exact})
    return {"cantilever_L_m": L, "thickness_m": t, "width_m": b, "load_N": P,
            "euler_bernoulli_tip_deflection_m": exact, "rows": rows,
            "note": ("solid hex8 with full integration is a poor bending "
                     "discretisation at hull-plate aspect ratios; the pressure "
                     "bending stress is therefore taken from the verified "
                     "Kirchhoff plate solver, not from the solid")}


# ---------------------------------------------------------------------------
# main chain
# ---------------------------------------------------------------------------

def main() -> dict:
    t_start = time.time()
    panel = PolarPanel()
    results = {
        "meta": {
            "module": "tensorfem.polar_thermal_stress",
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "threads": torch.get_num_threads(),
            "gamma_M": GAMMA_M,
            "T_air_cases_C": list(T_AIR_CASES),
            "ice_pressures_MPa": [p / 1e6 for p in ICE_PRESSURES],
            "patch_m": [0.3, 0.3],
        },
        "panel": panel.summary(),
        "verification": {},
        "temperature_fields": [],
    }
    results["verification"]["wall_1d"] = verify_wall()
    results["verification"]["biaxial_stress"] = verify_biaxial()
    results["verification"]["material"] = verify_material()
    results["verification"]["transient"] = verify_transient()
    results["verification"]["plate_bending"] = verify_plate()
    results["verification"]["pressure_reaction"] = verify_pressure_reaction()
    results["verification"]["hex8_bending_limitation"] = hex8_bending_limitation()

    # temperature fields and thermal stress maps
    fields = {}
    for T_air in T_AIR_CASES:
        env = PolarEnvironment(T_air=T_air)
        field = solve_panel_temperature(panel, env, STEEL)
        sol = solve_panel_stress(field.mesh, field.temperature, STEEL, env.T_frame, panel=panel)
        g = field.temperature.reshape(panel.n_thickness + 1, panel.n_y + 1, panel.n_z + 1)
        outer_layer = sol.von_mises.reshape(panel.n_thickness, panel.n_y, panel.n_z)[0]
        fields[T_air] = (field, sol, g, outer_layer)
        results["temperature_fields"].append({
            "T_air_C": T_air,
            "h_outer_conv_W_m2K": env.h_outer_conv,
            "h_rad_outer_W_m2K": field.h_rad_outer,
            "h_outer_total_W_m2K": field.h_outer_total,
            "h_inner_total_W_m2K": field.h_inner_total,
            "picard_iterations": field.picard_iterations,
            "picard_delta_C": field.picard_delta_C,
            **field.surface_temperatures(),
            "through_thickness_profile": field.through_thickness_profile(),
            "sigma_thermal_outer_max_MPa": float(outer_layer.max()) / 1e6,
            "sigma_thermal_outer_mean_MPa": float(outer_layer.mean()) / 1e6,
            "digest": field.digest(),
            "mesh": {"nodes": field.mesh.n_nodes, "elements": field.mesh.n_elements},
        })

    # limit-state matrix
    matrix = polar_thermal_limit_state_matrix(
        panel, STEEL, T_air_values=T_AIR_CASES, ice_pressures=ICE_PRESSURES,
        env_base=PolarEnvironment(), gamma_M=GAMMA_M)
    results["limit_state"] = {
        "rows": [r.to_dict() for r in matrix["rows"]],
        "per_pressure": matrix["per_pressure"],
        "monotonic_margin_in_T_air": matrix["monotonic_margin_in_T_air"],
        "all_pass": matrix["all_pass"],
        "governing": matrix["governing"].to_dict(),
    }

    # ice-side plate response over the pressure range (design curve)
    plate_curve = []
    ice = IceContactPatch(length=0.3, width=0.3)
    for p in (0.25e6, 0.5e6, 0.75e6, 1.0e6, 1.5e6, 2.0e6, 2.5e6):
        T_crit = -34.0
        res = clamped_plate_bending(panel.frame_spacing, panel.stringer_spacing,
                                    panel.thickness, float(STEEL.E(T_crit)), STEEL.nu,
                                    p, patch=ice.box(panel), n_y=24, n_z=48)
        plate_curve.append({"pressure_MPa": p / 1e6,
                           "sigma_bending_MPa": res.sigma_max / 1e6,
                           "w_mm": res.w_max * 1e3,
                           "total_load_N": res.total_load})
    results["ice_plate_curve"] = plate_curve

    # restrained vs free bracket for the coldest case
    env_cold = PolarEnvironment(T_air=-40.0)
    clamped = pts.assess_polar_panel(panel, env_cold, STEEL, ice_pressure=1.0e6,
                                     boundary="clamped", verify_superposition=True)
    free = pts.assess_polar_panel(panel, env_cold, STEEL, ice_pressure=1.0e6,
                                  boundary="free", verify_superposition=False)
    results["restraint_bracket"] = {
        "clamped": clamped.to_dict(), "free": free.to_dict(),
        "ratio_free_to_clamped_thermal": free.sigma_thermal_max / clamped.sigma_thermal_max,
    }
    results["superposition_rel_error"] = clamped.superposition_rel_error
    results["runtime_s"] = time.time() - t_start

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    make_figures(results, fields, panel, matrix)
    (OUT / "report_tables.md").write_text(render_tables(results))
    return results


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True})
    return plt


def make_figures(results, fields, panel, matrix) -> None:
    plt = _plt()

    # 1. 1-D wall verification
    wall = results["verification"]["wall_1d"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(wall["x_stations_m"], wall["T_analytic_C"], "k-", lw=1.4,
            label="analytic (Robin both ends)")
    ax.plot(wall["x_stations_m"], wall["T_fem_C"], "o", ms=4, mfc="none", label="hex8 FEM")
    ax.set_xlabel("depth from outer surface x [m]")
    ax.set_ylabel("temperature [degC]")
    ax.set_title(f"1-D composite wall: max |error| = {wall['max_abs_error_K']:.2e} K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig1_wall_verification.png")
    plt.close(fig)

    # 2. material model vs anchors
    mat = results["verification"]["material"]["materials"]["EH36"]
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.0))
    axs[0].plot(mat["anchor_temperature_C"], mat["anchor_E_GPa"], "ks", ms=4, label="anchor table")
    axs[0].plot(mat["anchor_temperature_C"], mat["model_E_GPa"], "-", label="E(T) model")
    axs[0].set_ylabel("E [GPa]")
    axs[1].plot(mat["anchor_temperature_C"], mat["anchor_alpha_1e6"], "ks", ms=4)
    axs[1].plot(mat["anchor_temperature_C"], mat["model_alpha_1e6"], "-")
    axs[1].set_ylabel("alpha [1e-6/K]")
    axs[2].plot(mat["anchor_temperature_C"], mat["anchor_yield_MPa"], "ks", ms=4)
    axs[2].plot(mat["anchor_temperature_C"], mat["model_yield_MPa"], "-")
    axs[2].set_ylabel("sigma_y [MPa]")
    for a in axs:
        a.set_xlabel("temperature [degC]")
    axs[0].legend()
    fig.suptitle(f"EH36 material model vs transcribed anchors (max residual "
                 f"{max(mat['measured_residual'].values()) * 100:.3f} %)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_material_model.png")
    plt.close(fig)

    # 3. outer-surface temperature and through-thickness profiles
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for T_air in T_AIR_CASES:
        prof = fields[T_air][0].through_thickness_profile()
        axs[0].plot([v * 1e3 for v in prof["x"]], prof["T"], "o-", ms=3,
                    label=f"T_air = {T_air:.0f} degC")
    axs[0].set_xlabel("depth from outer surface [mm]")
    axs[0].set_ylabel("temperature at bay centre [degC]")
    axs[0].legend(fontsize=7)
    axs[0].set_title("through-thickness profile (mid-bay)")
    field_cold, _, g_cold, _ = fields[-40.0]
    yy = field_cold.mesh.nodes.reshape(g_cold.shape + (3,))[0, :, :, 1]
    zz = field_cold.mesh.nodes.reshape(g_cold.shape + (3,))[0, :, :, 2]
    mesh = axs[1].pcolormesh(yy, zz, g_cold[0], shading="nearest", cmap="viridis")
    fig.colorbar(mesh, ax=axs[1], label="T [degC]")
    axs[1].set_xlabel("y along frame spacing [m]")
    axs[1].set_ylabel("z along stringer spacing [m]")
    axs[1].set_title("outer-surface T(x=0,y,z), T_air = -40 degC")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_temperature_field.png")
    plt.close(fig)

    # 4. thermal stress field
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for ax, T_air in zip(axs, (-40.0, -20.0, 0.0)):
        _, sol, g, outer_layer = fields[T_air]
        vm = outer_layer.numpy() / 1e6
        im = ax.imshow(vm, origin="lower", cmap="magma", aspect="auto",
                       extent=[0, panel.stringer_spacing, 0, panel.frame_spacing])
        fig.colorbar(im, ax=ax, label="von Mises [MPa]")
        ax.set_title(f"sigma_thermal, T_air = {T_air:.0f} degC\nmax {vm.max():.1f} MPa")
        ax.set_xlabel("z [m]")
        ax.set_ylabel("y [m]")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_thermal_stress_field.png")
    plt.close(fig)

    # 5. plate bending verification + ice pressure response
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
    plate = results["verification"]["plate_bending"]
    ns = [r["n_per_side"] for r in plate["rows"]]
    axs[0].plot(ns, [r["sigma_max_MPa"] for r in plate["rows"]], "o-", label="FD Kirchhoff")
    axs[0].axhline(plate["reference"]["sigma_max_Pa"] / 1e6, ls="--", c="k",
                   label="Timoshenko Table 36")
    axs[0].set_xlabel("grid points per side")
    axs[0].set_ylabel("sigma_max [MPa]")
    axs[0].set_title("clamped square, uniform pressure")
    axs[0].legend(fontsize=7)
    curve = results["ice_plate_curve"]
    axs[1].plot([c["pressure_MPa"] for c in curve], [c["sigma_bending_MPa"] for c in curve],
                "o-", label="bending stress")
    axs[1].plot([c["pressure_MPa"] for c in curve], [c["w_mm"] * 100 for c in curve],
                "s--", label="w x 100 [mm]")
    axs[1].set_xlabel("ice pressure on 0.3 x 0.3 m patch [MPa]")
    axs[1].set_ylabel("MPa / (mm x100)")
    axs[1].set_title("ice-load plate response (clamped bay)")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_plate_bending.png")
    plt.close(fig)

    # 6. limit state
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for key, series in matrix["per_pressure"].items():
        axs[0].plot(series["T_air_C"], series["margin"], "o-", label=f"P_ice = {key}")
        axs[1].plot(series["T_air_C"], series["utilization"], "o-", label=f"P_ice = {key}")
    axs[0].axhline(0.0, c="k", lw=0.8)
    axs[1].axhline(1.0, c="k", ls="--", lw=0.8, label="allowable limit")
    axs[0].set_xlabel("air temperature [degC]")
    axs[0].set_ylabel("margin = 1 - utilisation")
    axs[0].set_title("safety margin")
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel("air temperature [degC]")
    axs[1].set_ylabel("utilisation eta")
    axs[1].set_title(f"eta = sigma_total / (sigma_y/gamma_M), gamma_M = {GAMMA_M}")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_limit_state.png")
    plt.close(fig)

    # 7. transient verification
    tr = results["verification"]["transient"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(tr["times_s"], tr["T_lumped_C"], "k-", label="lumped capacitance exp(-t/tau)")
    ax.plot(tr["times_s"], tr["T_fem_C"], "o", ms=3, mfc="none", label="backward-Euler FEM")
    ax.axhline(tr["steady_C"], c="r", ls=":", label="steady solution")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("temperature [degC]")
    ax.set_title(f"transient conduction: max error {tr['max_abs_error_K']:.3f} K, "
                 f"Bi = {tr['biot']:.1e}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_transient.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# tables (programmatic, zero hand-copying)
# ---------------------------------------------------------------------------

def render_tables(results: dict) -> str:
    L = []
    v = results["verification"]
    L.append("# Polar hull thermal-stress / ice-load chain -- generated tables\n")
    L.append("## Verification anchors\n")
    L.append("| anchor | quantity | value | criterion | status |")
    L.append("|---|---|---|---|---|")
    w = v["wall_1d"]
    L.append(f"| 1-D composite wall | max nodal error | {w['max_abs_error_K']:.3e} K "
             f"({w['max_rel_error'] * 100:.2e} % of {w['T_inner_C'] - w['T_air_C']:.0f} K span) "
             f"| <1 % | {'PASS' if w['max_rel_error'] < 0.01 else 'FAIL'} |")
    b = v["biaxial_stress"]
    L.append(f"| restrained biaxial | max rel. error vs E a dT/(1-nu) | "
             f"{b['max_rel_error'] * 100:.2e} % | <2 % | "
             f"{'PASS' if b['max_rel_error'] < 0.02 else 'FAIL'} |")
    m = v["material"]
    L.append(f"| material E/alpha/sigma_y | max residual vs anchor table | "
             f"{m['max_rel_error'] * 100:.4f} % | <5 % | "
             f"{'PASS' if m['max_rel_error'] < 0.05 else 'FAIL'} |")
    t = v["transient"]
    L.append(f"| transient conduction | max error vs exp(-t/tau) | {t['max_abs_error_K']:.3f} K "
             f"({t['max_rel_error_of_span'] * 100:.2f} %) | <2 % | "
             f"{'PASS' if t['max_rel_error_of_span'] < 0.02 else 'FAIL'} |")
    p = v["plate_bending"]
    L.append(f"| Kirchhoff plate vs Timoshenko | sigma_max at n=72 | "
             f"{p['rows'][-1]['sigma_max_MPa']:.2f} MPa, err "
             f"{p['rows'][-1]['sigma_rel_error'] * 100:.2f} % | <2 % | "
             f"{'PASS' if p['max_sigma_rel_error'] < 0.02 else 'FAIL'} |")
    L.append(f"| Kirchhoff plate vs Timoshenko | w_max at n=72 | "
             f"{p['rows'][-1]['w_max_mm']:.4f} mm, err "
             f"{p['rows'][-1]['w_rel_error'] * 100:.2f} % | <2 % | "
             f"{'PASS' if p['max_w_rel_error'] < 0.02 else 'FAIL'} |")
    r = v["pressure_reaction"]
    L.append(f"| pressure face load | reaction vs p*A | {r['reaction_N']:.1f} N vs "
             f"{r['p_times_area_N']:.1f} N (rel {r['rel_error']:.1e}) | <1e-9 | "
             f"{'PASS' if r['rel_error'] < 1e-9 else 'FAIL'} |")
    L.append(f"| FEM superposition | relative error of combined vs tensor sum | "
             f"{results['superposition_rel_error']:.2e} | <1e-8 | "
             f"{'PASS' if results['superposition_rel_error'] < 1e-8 else 'FAIL'} |\n")

    L.append("## Temperature field by air temperature\n")
    L.append("| T_air [degC] | h_conv [W/m2K] | h_rad [W/m2K] | T_outer,min [degC] | "
             "T_outer,mean [degC] | T_inner,mean [degC] | dT through thickness [K] | "
             "sigma_th,max [MPa] |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f in results["temperature_fields"]:
        L.append(f"| {f['T_air_C']:.0f} | {f['h_outer_conv_W_m2K']:.2f} | "
                 f"{f['h_rad_outer_W_m2K']:.2f} | {f['outer_min_C']:.2f} | "
                 f"{f['outer_mean_C']:.2f} | {f['inner_mean_C']:.2f} | "
                 f"{f['through_thickness_gradient_C']:.3f} | "
                 f"{f['sigma_thermal_outer_max_MPa']:.1f} |")
    L.append("")
    L.append("## Limit state (gamma_M = %.2f)\n" % results["meta"]["gamma_M"])
    L.append("| T_air [degC] | P_ice [MPa] | sigma_thermal [MPa] | sigma_ice [MPa] | "
             "sigma_total [MPa] | sigma_y(T) [MPa] | eta | margin | SF | status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in results["limit_state"]["rows"]:
        L.append(f"| {row['T_air']:.0f} | {row['ice_pressure_MPa']:.2f} | "
                 f"{row['sigma_thermal_max_MPa']:.1f} | {row['sigma_ice_max_MPa']:.1f} | "
                 f"{row['sigma_total_max_MPa']:.1f} | {row['sigma_yield_at_critical_MPa']:.1f} | "
                 f"{row['utilization']:.3f} | {row['margin']:+.3f} | "
                 f"{row['safety_factor']:.2f} | {row['status']} |")
    L.append("")
    mono = results['limit_state']['monotonic_margin_in_T_air']
    allp = results['limit_state']['all_pass']
    L.append(f"Monotone margin in T_air: **{mono}**; all PASS: **{allp}**\n")
    L.append("## Restraint bracket (T_air = -40 degC, P_ice = 1 MPa)\n")
    rb = results["restraint_bracket"]
    L.append("| boundary | sigma_thermal [MPa] | sigma_total [MPa] | margin |")
    L.append("|---|---|---|---|")
    for key in ("clamped", "free"):
        d = rb[key]
        L.append(f"| {key} | {d['sigma_thermal_max_MPa']:.1f} | "
                 f"{d['sigma_total_max_MPa']:.1f} | {d['margin']:+.3f} |")
    L.append("")
    L.append("## Solid hex8 bending limitation (cantilever, tip load)\n")
    lim = v["hex8_bending_limitation"]
    L.append("| mesh (nx,ny,nz) | w_FEM [m] | w_Euler-Bernoulli [m] | ratio |")
    L.append("|---|---|---|---|")
    for row in lim["rows"]:
        L.append(f"| {row['mesh']} | {row['w_fem_m']:.3e} | {row['w_beam_m']:.3e} | "
                 f"{row['ratio']:.3f} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    res = main()
    print(json.dumps({k: res[k] for k in ("runtime_s", "superposition_rel_error")}, indent=2))
    print(f"artifacts written to {OUT}")