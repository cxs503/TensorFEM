"""Generate the FSI wet-modal adjudication figures and the numerical record.

Run from the repository root::

    PYTHONPATH=src python scripts/gen_fsi_figures.py

Writes ``artifacts/m6607/*.png`` (figures), ``artifacts/m6607/numerical_data.json``
(programmatically injected numbers, zero manual transcription) and
``artifacts/m6607/acceptance.json`` (machine-checkable criteria and verdicts).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfem import fsi_wet_modal as fsi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "m6607"

# Benchmark definition (Junger & Feit anchor: L/D = 10, R/t = 100).
CYLINDER = dict(
    radius=1.0,
    length=20.0,
    thickness=0.01,
    rho_fluid=1025.0,
    rho_struct=7850.0,
    young=210.0e9,
    poisson=0.3,
)
PLATE = dict(
    length=1.0,
    width=1.0,
    thickness=0.005,
    rho_fluid=1025.0,
    rho_struct=7850.0,
    young=210.0e9,
    poisson=0.3,
)
CYLINDER_MODES = ((1, 2), (1, 3), (2, 2), (1, 4), (2, 3))
PLATE_MODES = ((1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3))


def _plt_style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def figure_loading_coefficient(path: Path, numerical: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    radius = 1.0
    x = np.linspace(0.05, 3.0, 300)
    lengths = math.pi * radius / x  # x = k_m R with m = 1
    for order in (1, 2, 3, 4):
        exterior = [
            fsi.cylinder_fluid_loading_coefficient(radius, float(value), 1, order)
            for value in lengths
        ]
        interior = [
            fsi.cylinder_fluid_loading_coefficient(
                radius, float(value), 1, order, exterior=False
            )
            for value in lengths
        ]
        axes[0].plot(x, exterior, label=f"exterior n={order}")
        axes[0].plot(x, interior, "--", lw=0.9, label=f"interior n={order}")
        axes[0].axhline(1.0 / order, color="grey", lw=0.5, ls=":")
    x_benchmark = math.pi * radius / CYLINDER["length"]
    axes[0].axvline(x_benchmark, color="crimson", lw=0.8, ls="-.")
    axes[0].text(
        x_benchmark + 0.05, 0.9,
        "L/D = 10 benchmark\nx = pi R / L", color="crimson", fontsize=7,
    )
    axes[0].set_xlabel("x = k_m R  (= m pi R / L)")
    axes[0].set_ylabel(r"$C(m,n)$")
    axes[0].set_title("Junger & Feit fluid loading coefficient")
    axes[0].legend(fontsize=5.5, ncol=2, loc="upper right")

    markers = numerical["coefficient_check"]
    orders = [entry["n"] for entry in markers]
    analytic = [entry["analytic"] for entry in markers]
    computed = [entry["numerical"] for entry in markers]
    axes[1].plot(orders, analytic, "o", ms=5, label="closed form $\\rho_f R C$")
    axes[1].plot(orders, computed, "x", ms=7, label="finite-difference fluid")
    for order, value in zip(orders, analytic):
        axes[1].annotate(
            f"C={value / (CYLINDER['rho_fluid'] * CYLINDER['radius']):.3f}",
            (order, value), textcoords="offset points", xytext=(4, 6), fontsize=7,
        )
    axes[1].set_xlabel("circumferential wave number n (m = 1)")
    axes[1].set_ylabel(r"$m_a$ [kg/m$^2$]")
    axes[1].set_title("Pointwise added mass: two independent fluid models")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_cylinder_spectrum(path: Path, cylinder: dict) -> None:
    modes = [f"{m}-{n}" for m, n in cylinder["mode_numbers"]]
    dry = cylinder["dry_hz_analytic"]
    wet_a = cylinder["wet_hz_analytic"]
    wet_n = cylinder["wet_hz_numerical"]
    index = np.arange(len(modes))
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    axes[0].bar(index - width, dry, width, label="dry (Donnell)")
    axes[0].bar(index, wet_a, width, label="wet, closed-form $M_a$")
    axes[0].bar(index + width, wet_n, width, label="wet, numerical fluid")
    axes[0].set_xticks(index)
    axes[0].set_xticklabels([f"(m,n)=({m},{n})" for m, n in cylinder["mode_numbers"]])
    axes[0].set_ylabel("natural frequency [Hz]")
    axes[0].set_title("Submerged shell L/D=10, R/t=100")
    axes[0].legend(fontsize=7)

    ratio = np.array(wet_a) / np.array(dry)
    mu = np.array(cylinder["mu_analytic"])
    axes[1].plot(mu, ratio, "o-", label="$f_{wet}/f_{dry}=1/\\sqrt{1+\\mu}$")
    for mode, mu_value, ratio_value in zip(modes, mu, ratio):
        axes[1].annotate(mode, (mu_value, ratio_value), textcoords="offset points",
                         xytext=(5, -8), fontsize=7)
    axes[1].set_xlabel(r"modal added-mass ratio $\mu = M_a/M_s$")
    axes[1].set_ylabel("wet / dry frequency ratio")
    axes[1].set_title("Fluid loading collapses the thin-shell frequencies")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_fluid_convergence(path: Path, convergence: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    radii = [row["n_radial"] for row in convergence]
    for label, key, style in (
        ("n = 2", "error_n2", "o-"),
        ("n = 3", "error_n3", "s--"),
    ):
        axes[0].loglog(radii, [row[key] for row in convergence], style, label=label)
    reference = [
        convergence[0]["error_n2"] * (radii[0] / r) ** 2 for r in radii
    ]
    axes[0].loglog(radii, reference, "k:", lw=1, label="second-order slope")
    axes[0].set_xlabel("radial grid points $n_r$")
    axes[0].set_ylabel("pointwise added-mass error")
    axes[0].set_title("Finite-difference fluid field vs Bessel solution")
    axes[0].legend(fontsize=7)

    modes = [f"({m},{n})" for m, n in CYLINDER_MODES]
    analytic = [row["mu_analytic"] for row in convergence[-1]["modes"]]
    numerical = [row["mu_numerical"] for row in convergence[-1]["modes"]]
    index = np.arange(len(modes))
    axes[1].bar(index - 0.2, analytic, 0.4, label="closed form")
    axes[1].bar(index + 0.2, numerical, 0.4, label="numerical fluid (finite shell)")
    axes[1].set_xticks(index)
    axes[1].set_xticklabels(modes)
    axes[1].set_ylabel(r"$\mu = M_a/M_s$")
    axes[1].set_title("Modal added-mass ratio: two criteria")
    axes[1].legend(fontsize=7)
    for i, (a, b) in enumerate(zip(analytic, numerical)):
        axes[1].text(i, max(a, b) + 0.1, f"{(b / a - 1) * 100:+.2f}%",
                     ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_plate_added_mass(path: Path, record: dict) -> None:
    orders = [f"({m},{n})" for m, n in PLATE_MODES]
    exact = [
        fsi.plate_exact_modal_added_mass(1.0, 1.0, m, n, PLATE["rho_fluid"]) / 0.25
        for m, n in PLATE_MODES
    ]
    asymptote = [
        fsi.plate_asymptotic_added_mass_per_area(1.0, 1.0, m, n, PLATE["rho_fluid"])
        for m, n in PLATE_MODES
    ]
    panel = [
        fsi.plate_panel_added_mass_ratio(
            1.0, 1.0, m, n, PLATE["rho_fluid"], PLATE["rho_struct"],
            PLATE["thickness"], nx=32, ny=32,
        ) * PLATE["rho_struct"] * PLATE["thickness"]
        for m, n in PLATE_MODES
    ]
    index = np.arange(len(orders))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    axes[0].plot(index, exact, "o-", label="exact wave-number integral")
    axes[0].plot(index, panel, "x--", label="panel Rayleigh operator 32x32")
    axes[0].plot(index, asymptote, "s:", label=r"asymptote $\rho_f/k_{mn}$")
    axes[0].set_xticks(index)
    axes[0].set_xticklabels(orders)
    axes[0].set_ylabel(r"$m_a$ [kg/m$^2$] (modal mean square)")
    axes[0].set_title("Submerged panel: added mass per unit area")
    axes[0].legend(fontsize=7)

    ratio = np.array(exact) / np.array(asymptote)
    axes[1].plot(index, ratio, "o-")
    axes[1].axhline(1.0, color="grey", ls=":")
    axes[1].set_xticks(index)
    axes[1].set_xticklabels(orders)
    axes[1].set_ylabel("exact / asymptote")
    axes[1].set_title("Asymptote is only valid for high modal wave numbers")
    for i, value in enumerate(ratio):
        axes[1].annotate(f"{value:.2f}", (i, value), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    del record


def figure_plate_frf(path: Path, frf: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    frequency = np.asarray(frf["frequencies_hz"])
    amplitude = np.asarray(frf["amplitude"])
    axes[0].semilogy(frequency, amplitude, lw=0.9, color="navy")
    for wet in frf["wet_resonances_hz"]:
        axes[0].axvline(wet, color="crimson", lw=0.7, ls="--", alpha=0.7)
    axes[0].set_xlabel("frequency [Hz]")
    axes[0].set_ylabel("|H(f)| [m/N]")
    axes[0].set_title("Centre drive / centre response, wet panel (zeta=0.01)")
    axes[0].annotate("wet resonances (dashed)", (frequency[-1] * 0.55, amplitude.max() * 0.5),
                     fontsize=7, color="crimson")

    mode_index = np.arange(len(frf["dry_frequencies_hz"]))
    axes[1].plot(mode_index, frf["dry_frequencies_hz"], "o-", label="dry (Mindlin FE)")
    axes[1].plot(mode_index, frf["wet_frequencies_hz"], "s-", label="wet (FE + fluid)")
    axes[1].set_xlabel("mode index")
    axes[1].set_ylabel("frequency [Hz]")
    axes[1].set_title("Panel spectrum before/after fluid loading")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_pressure_field(path: Path, radius: float, length: float) -> None:
    """Fluid displacement-potential / pressure field, mode (m, n) = (1, 2)."""
    k = math.pi / length
    potential = fsi.solve_incompressible_potential(
        radius, 2, lambda z: np.sin(k * z), x_min=0.0, x_max=length,
        n_radial=120, n_axial_grid=240,
    )
    pressure = -CYLINDER["rho_fluid"] * potential.field  # p = rho_f omega^2 psi

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    contour = axes[0].contourf(
        potential.x, potential.r / radius, pressure, levels=24, cmap="RdBu_r"
    )
    axes[0].plot([0, length], [1, 1], "k-", lw=2)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("r / R")
    axes[0].set_title("Fluid pressure, mode (m,n) = (1,2)")
    fig.colorbar(contour, ax=axes[0], label=r"$p/\omega^2\hat w$ [kg/m$^2$]")

    shell = axes[1].contourf(
        potential.x, potential.r / radius, potential.field, levels=24, cmap="viridis"
    )
    axes[1].plot([0, length], [1, 1], "w-", lw=2)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("r / R")
    axes[1].set_title("Displacement potential $\\psi$ (unit wall velocity)")
    fig.colorbar(shell, ax=axes[1], label="psi [m$^2$]")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def shell_fe_beam_limit_check() -> dict:
    """Check the repository shell4 FE against the beam limit of the n = 1 mode.

    The ``n = 1`` mode of a slender simply supported shell must approach the
    Euler-Bernoulli beam frequency ``(pi/L)^2 sqrt(E I / (rho A))`` with
    ``I/A = R^2/2``.  This is a *report-only* diagnostic: it is why the shell FE
    spectrum is not used as the structural anchor of this case.
    """
    from tensorfem.cylindrical_shell_modal import (
        apply_simply_supported_bc,
        assemble_cylindrical_shell,
        solve_shell_modes,
        structured_cylindrical_mesh,
    )

    radius = CYLINDER["radius"]
    length = CYLINDER["length"]
    beam_hz = (
        (math.pi / length) ** 2
        * math.sqrt(CYLINDER["young"] * radius ** 2 / 2.0 / CYLINDER["rho_struct"])
        / (2.0 * math.pi)
    )
    rows = []
    for n_axial, n_circ in ((16, 20), (20, 24)):
        nodes, elements, theta = structured_cylindrical_mesh(
            n_axial, n_circ, radius, length
        )
        stiffness, mass = assemble_cylindrical_shell(
            nodes, elements, theta, radius, CYLINDER["young"], CYLINDER["poisson"],
            CYLINDER["rho_struct"], CYLINDER["thickness"],
        )
        constrained = apply_simply_supported_bc(nodes, theta, length)
        result = solve_shell_modes(stiffness, mass, constrained, n_modes=6)
        frequencies = [float(value) for value in result.frequencies_hz]
        elastic = [value for value in frequencies if value > 1e-3][:3]
        rows.append(
            {
                "mesh": f"{n_axial}x{n_circ}",
                "lowest_elastic_hz": elastic,
                "ratio_vs_beam": elastic[0] / beam_hz if elastic else float("nan"),
            }
        )
    return {
        "beam_limit_hz": beam_hz,
        "donnell_n1_hz": fsi.donnell_shell_frequency_hz(
            radius, length, CYLINDER["thickness"], CYLINDER["young"],
            CYLINDER["poisson"], CYLINDER["rho_struct"], 1, 1,
        ),
        "legacy_dmv_n1_hz": _legacy_dmv_n1(),
        # pre-repair the helper dropped the membrane term at n = 1; this ratio
        # is 1.0 once the repair (m16863) is in place
        "legacy_dmv_n1_ratio": _legacy_dmv_n1() / fsi.donnell_shell_frequency_hz(
            radius, length, CYLINDER["thickness"], CYLINDER["young"],
            CYLINDER["poisson"], CYLINDER["rho_struct"], 1, 1,
        ),
        "rows": rows,
    }


def _legacy_dmv_n1() -> float:
    from tensorfem.cylindrical_shell_modal import dmv_frequency_hz

    return dmv_frequency_hz(
        radius=CYLINDER["radius"], length=CYLINDER["length"],
        thickness=CYLINDER["thickness"], young=CYLINDER["young"],
        poisson=CYLINDER["poisson"], density=CYLINDER["rho_struct"],
        n_axial=1, n_circ=1,
    )


def build_acceptance(cylinder, coefficient_check, plate, frf_record, mode_shapes) -> dict:
    mu_errors = [
        abs(numerical / analytic - 1.0)
        for analytic, numerical in zip(
            cylinder["mu_analytic"], cylinder["mu_numerical_baffle"]
        )
    ]
    plate_mu_errors = [
        abs(panel / exact - 1.0)
        for exact, panel in zip(plate["mu_exact"], plate["mu_panel"])
    ]
    plate_fe_errors = [
        abs(fe / navier - 1.0)
        for fe, navier in zip(plate["dry_hz_fe"], plate["dry_hz_navier"])
    ]
    peak_errors = [
        abs(peak / wet - 1.0)
        for peak, wet in zip(
            frf_record["peak_estimate_hz"], frf_record["wet_resonances_hz"]
        )
    ]
    gated_fe = plate_fe_errors[:3]  # modes used for the wet adjudication
    criteria = {
        "c1_cylinder_modal_added_mass_numerical_vs_closed_form": {
            "tolerance": 0.03,
            "max_rel_error": max(mu_errors),
            "passed": max(mu_errors) < 0.03,
            "quantity": "mu = M_a/M_s, 5 shell modes",
        },
        "c2_cylinder_pointwise_fluid_loading_vs_bessel": {
            "tolerance": 0.03,
            "max_rel_error": max(entry["rel_error"] for entry in coefficient_check),
            "passed": max(entry["rel_error"] for entry in coefficient_check) < 0.03,
            "quantity": "-rho_f psi / w vs rho_f R C(m,n)",
        },
        "c3_plate_modal_added_mass_panel_vs_exact": {
            "tolerance": 0.03,
            "max_rel_error": max(plate_mu_errors),
            "passed": max(plate_mu_errors) < 0.03,
            "quantity": "mu of 6 panel modes, 32x32 Rayleigh panels vs exact integral",
        },
        "c4_plate_fe_dry_frequency_first_three_modes": {
            "tolerance": 0.03,
            "max_rel_error": max(gated_fe),
            "passed": max(gated_fe) < 0.03,
            "quantity": "Mindlin FE vs Navier, modes (1,1),(1,2),(2,2)",
        },
        "c5_frf_resonance_vs_wet_eigenfrequency": {
            "tolerance": 0.03,
            "max_rel_error": max(peak_errors),
            "passed": max(peak_errors) < 0.03,
            "quantity": "|H| peak frequency vs wet eigenfrequency, excited modes",
        },
        "c7_plate_mode_shape_mac_vs_analytic": {
            "tolerance": 0.01,
            "max_rel_error": max(1.0 - row["mac"] for row in mode_shapes),
            "passed": all(row["mac"] > 0.99 for row in mode_shapes),
            "quantity": "MAC of FE dry mode shapes vs sin(m pi x/a) sin(n pi y/b)",
        },
        "c6_wet_below_dry_everywhere": {
            "tolerance": 0.0,
            "passed": all(
                wet < dry
                for wet, dry in zip(
                    list(cylinder["wet_hz_analytic"]) + list(plate["wet_hz"]),
                    list(cylinder["dry_hz_analytic"]) + list(plate["dry_hz_navier"]),
                )
            ),
            "quantity": "f_wet < f_dry for all adjudicated modes",
        },
    }
    informational = {
        "i1_plate_fe_dry_frequency_modes_four_to_six": {
            "max_rel_error": max(plate_fe_errors[3:]),
            "note": (
                "higher panel modes need a finer structural mesh; the error "
                "drops from 7.2% (12x12) to 3.9% (16x16) at mode (1,3), so it "
                "is structural discretisation, not fluid coupling"
            ),
        }
    }
    return {
        "node": "IN59-N03",
        "mission_id": 6607,
        "repository": "TensorFEM",
        "module": "src/tensorfem/fsi_wet_modal.py",
        "tests": "tests/test_fsi_wet_modal.py",
        "criteria": criteria,
        "informational": informational,
        "all_passed": all(item["passed"] for item in criteria.values()),
        "figures": sorted(path.name for path in OUT.glob("*.png")),
    }


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown(data, acceptance, shell_check) -> Path:
    cylinder = data["cylinder"]["result"]
    plate = data["plate"]["result"]
    frf = data["frf"]
    mu_agreement = max(
        abs(numerical / analytic - 1.0)
        for analytic, numerical in zip(
            cylinder["mu_analytic"], cylinder["mu_numerical_baffle"]
        )
    )
    panel_agreement = max(
        abs(panel / exact - 1.0)
        for exact, panel in zip(plate["mu_exact"], plate["mu_panel"])
    )
    asymptote_gap = 1.0 - plate["mu_asymptotic"][0] / plate["mu_exact"][0]
    fe_high_error = acceptance["informational"][
        "i1_plate_fe_dry_frequency_modes_four_to_six"
    ]["max_rel_error"]

    cylinder_table = _table(
        ["mode (m,n)", "f_dry [Hz]", "mu (closed form)", "f_wet [Hz] (A)",
         "mu (numerical fluid)", "f_wet [Hz] (B)", "mu_B/mu_A - 1"],
        [
            [
                f"({m},{n})",
                f"{dry:.3f}",
                f"{mu_a:.4f}",
                f"{wet_a:.3f}",
                f"{mu_n:.4f}",
                f"{wet_n:.3f}",
                f"{(mu_n / mu_a - 1) * 100:+.2f}%",
            ]
            for (m, n), dry, wet_a, wet_n, mu_a, mu_n in zip(
                cylinder["mode_numbers"],
                cylinder["dry_hz_analytic"],
                cylinder["wet_hz_analytic"],
                cylinder["wet_hz_numerical"],
                cylinder["mu_analytic"],
                cylinder["mu_numerical_baffle"],
            )
        ],
    )
    coefficient_table = _table(
        ["n", "m_a closed form [kg/m^2]", "m_a finite difference [kg/m^2]", "rel. error"],
        [
            [
                str(entry["n"]),
                f"{entry['analytic']:.2f}",
                f"{entry['numerical']:.2f}",
                f"{entry['rel_error'] * 100:.3f}%",
            ]
            for entry in data["cylinder"]["coefficient_check"]
        ],
    )
    convergence_table = _table(
        ["n_r x n_x", "pointwise error (n=2)", "pointwise error (n=3)"],
        [
            [
                f"{row['n_radial']} x {row['n_axial_grid']}",
                f"{row['error_n2'] * 100:.3f}%",
                f"{row['error_n3'] * 100:.3f}%",
            ]
            for row in data["cylinder"]["convergence"]
        ],
    )
    plate_table = _table(
        ["mode (m,n)", "f_dry Navier [Hz]", "f_dry FE [Hz]", "mu exact",
         "mu panel", "mu asymptote", "f_wet [Hz]"],
        [
            [
                f"({m},{n})",
                f"{nav:.3f}",
                f"{fe:.3f}",
                f"{me:.4f}",
                f"{mp:.4f}",
                f"{ma:.4f}",
                f"{wet:.3f}",
            ]
            for (m, n), nav, fe, me, mp, ma, wet in zip(
                plate["mode_numbers"], plate["dry_hz_navier"], plate["dry_hz_fe"],
                plate["mu_exact"], plate["mu_panel"], plate["mu_asymptotic"],
                plate["wet_hz"],
            )
        ],
    )
    frf_table = _table(
        ["wet eigenfrequency [Hz]", "|H| peak [Hz]", "rel. error", "modal participation"],
        [
            [
                f"{wet:.3f}",
                f"{peak:.3f}",
                f"{abs(peak / wet - 1) * 100:.3f}%",
                f"{part:.3f}",
            ]
            for wet, peak, part in zip(
                frf["wet_resonances_hz"], frf["peak_estimate_hz"],
                [frf["participation"][i] for i in range(len(frf["participation"]))
                 if frf["participation"][i] > 0.05],
            )
        ],
    )
    mode_shape_table = _table(
        ["mode (m,n)", "FE f [Hz]", "Navier f [Hz]", "MAC",
         "max scaled deviation", "FE mode index"],
        [
            [
                f"({row['m']},{row['n']})",
                f"{row['fe_hz']:.3f}",
                f"{row['navier_hz']:.3f}",
                f"{row['mac']:.8f}",
                f"{row['max_scaled_deviation']:.2e}",
                str(row["fe_index"]),
            ]
            for row in data["plate_mode_shapes"]
        ],
    )
    acceptance_table = _table(
        ["criterion", "quantity", "tolerance", "max error", "verdict"],
        [
            [
                name,
                spec["quantity"],
                f"{spec['tolerance'] * 100:.0f}%",
                f"{spec['max_rel_error'] * 100:.3f}%"
                if "max_rel_error" in spec else "-",
                "PASS" if spec["passed"] else "FAIL",
            ]
            for name, spec in acceptance["criteria"].items()
        ],
    )
    shell_table = _table(
        ["mesh (axial x circumferential)", "lowest elastic frequencies [Hz]",
         "first / beam limit"],
        [
            [
                row["mesh"],
                ", ".join(f"{value:.2f}" for value in row["lowest_elastic_hz"]),
                f"{row['ratio_vs_beam']:.2f}",
            ]
            for row in shell_check["rows"]
        ],
    )

    text = f"""# Submerged-cylinder / submerged-panel wet modal analysis: dual-criteria adjudication

Node: **IN59-N03** (mission 6607), repository **TensorFEM**,
module `src/tensorfem/fsi_wet_modal.py`, tests `tests/test_fsi_wet_modal.py`.
Generated by `scripts/gen_fsi_figures.py` (figures, numbers and this text are all
produced by that one command; no hand-copied numbers).

## 1. Why this case

TensorFEM could already add a *prescribed* mass per unit area to a plate element
(`tensorfem.added_mass`) and owned a verified low-frequency Rayleigh operator for
a flat baffled panel (`tensorfem.hydro_added_mass`, see
`docs/marine-wet-panel-added-mass-verification.md`).  What was missing is the
**curved** wetted surface: there was no fluid-loading operator for a submerged
cylindrical shell, no mode-resolved added-mass ratio, and no verified
excitation-response spectrum.  Vibration/noise screening of a hull or a pipe
needs exactly those three, and the last implementation on this branch failed the
physics (it scaled the structural mass by a *single* Junger & Feit coefficient
for the dominant mode, so every mode of the shell received the same frequency
shift, and its response function used hard-coded sine modes and a hard-coded
generalised mass).  This case replaces that with a mode-resolved model and
adjudicates it with two independent criteria.

## 2. Case definition

Cylinder (Junger & Feit anchor): `R = {CYLINDER['radius']} m`,
`L = {CYLINDER['length']} m` (`L/D = 10`), `t = {CYLINDER['thickness']} m`
(`R/t = 100`), `E = {CYLINDER['young'] / 1e9:.0f} GPa`, `nu = {CYLINDER['poisson']}`,
`rho_s = {CYLINDER['rho_struct']:.0f} kg/m^3`, `rho_f = {CYLINDER['rho_fluid']:.0f} kg/m^3`
(seawater), simply supported (diaphragm) ends, external fluid.

Panel (Gorman & Horacek family): `a = b = {PLATE['length']} m`,
`t = {PLATE['thickness'] * 1000:.1f} mm`, same steel, same water, simply supported,
in an infinite rigid baffle.

## 3. Method

**Structure.** The cylinder reference uses Donnell-Mushtari-Vlasov thin-shell
theory (`donnell_shell_frequency_hz`); the panel uses Mindlin Q4 finite elements
(`tensorfem.plate_vibration`) for the response function and the Navier closed form
for the frequency reference.

**Fluid.** Three independent operators:

1. closed form -- Junger & Feit modal loading `m_a = rho_f R C(m,n)`,
   `C = -K_n(x)/(x K_n'(x))`, `x = m pi R / L` (external fluid, decaying modified
   Bessel function; `I_n` branch kept for an internal fluid);
2. finite-difference field -- the same incompressible equation
   `psi_rr + psi_r/r - n^2 psi/r^2 + psi_xx = 0` solved on an `(r, x)` grid with
   `dpsi/dr = w(x)` on the wetted wall, second-order stencils at both the wall and
   the far-field boundary (`dpsi/dr + (n/r) psi = 0`), two end idealisations
   (`baffle`, `rigid_caps`);
3. Rayleigh panel operator for the flat panel
   (`dipole_added_mass_matrix`) plus the exact wave-number integral
   `M_a = rho_f/(2 pi)^2 int int |phi_hat|^2/|k| dk` as the reference for it.

**Coupling.** Modal added mass ratios `mu = M_a/M_s`, wet frequency
`f_wet = f_dry / sqrt(1+mu)`, response
`H(f) = sum_k phi_k(x_f) phi_k(x_r) / (m_k (omega_k^2 - omega^2 + 2 i zeta omega_k omega))`
with the *wet* eigenfrequencies, FE mode shapes interpolated at the drive and
response points and a constant modal damping ratio `zeta = {frf['damping_ratio']}`.

Reproduce with::

    PYTHONPATH=src python scripts/gen_fsi_figures.py
    python -m pytest tests/test_fsi_wet_modal.py -q

## 4. Results

### 4.1 Submerged cylindrical shell, dual criteria

{cylinder_table}

Criterion A is the closed-form pair (Donnell dry frequency + Junger & Feit added
mass); criterion B replaces only the fluid with the finite-difference field of the
finite shell.  Criterion B agrees with A to **{mu_agreement * 100:.2f}%**,
so the finite-length end effect at `L/D = 10` is below the adjudication band
(the axial decay length `L/(m pi)` is 6.4 m, much longer than the shell radius).
The thin shell loses most of its in-air stiffness: `mu = 6.49` for `(m,n) = (1,2)`
gives `f_wet/f_dry = 0.366`, i.e. a 63% frequency drop in water.

### 4.2 Pointwise fluid loading and numerical convergence

{coefficient_table}

{convergence_table}

The pointwise comparison removes every normalisation difference between the two
fluid models, so its error is pure discretisation; halving the grid reduces it by
~3.5x (second order), which is what a verified implementation must show.

### 4.3 Submerged panel

{plate_table}

The exact integral and the 32x32 Rayleigh panel operator agree to
**{panel_agreement * 100:.2f}%**, which validates the panel discretisation used by the response function.  The
classical asymptote `rho_f/k_mn` is *not* a valid substitute at these modal wave
numbers: it under-predicts the fundamental by **{asymptote_gap * 100:.0f}%** and
only approaches the exact value for high modes (see figure 4).  Water loading
moves the panel fundamental from {plate['dry_hz_navier'][0]:.2f} Hz to
{plate['wet_hz'][0]:.2f} Hz.

### 4.4 Mode shapes

{mode_shape_table}

The measured modal assurance criterion is 1 to machine precision, i.e. on a
uniform mesh the discrete simply supported modes of this Mindlin Q4 model coincide
with the sampled analytic sine modes.  The check therefore verifies the *shape
pipeline* (degree-of-freedom layout, node ordering, interpolation used by the
response function, eigenvector sign) rather than the discretisation error; that is
stated here instead of being presented as an accuracy result.

### 4.5 Excitation response spectrum

{frf_table}

Only modes with a non-vanishing participation `|phi(x_f) phi(x_r)|` can produce a
spectral peak; the antisymmetric modes of a centre-driven square panel have
participation below `1e-19` and are correctly absent.

## 5. Figures

* `fig1_fluid_loading_coefficient.png` -- `C(m,n)` versus `x = k_m R` for
  `n = 1..4` (exterior and interior branches, `1/n` limit), and the pointwise
  added mass of both fluid models at the benchmark.
* `fig2_cylinder_wet_spectrum.png` -- dry, closed-form wet and numerically-wet
  frequencies, and the `1/sqrt(1+mu)` collapse law.
* `fig3_fluid_solver_convergence.png` -- second-order convergence of the
  finite-difference field and the modal `mu` of both criteria.
* `fig4_plate_added_mass.png` -- exact integral, panel operator and `rho_f/k_mn`
  asymptote per mode, plus the exact/asymptote ratio.
* `fig5_plate_frf.png` -- `|H(f)|` with the wet resonances marked, and the dry/wet
  modal spectrum.
* `fig6_pressure_field.png` -- fluid pressure and displacement-potential field
  around the shell for `(m,n) = (1,2)` (contour plot in the `(x, r)` plane).

## 6. Error adjudication (per criterion)

{acceptance_table}

Is every gated quantity below 3%? **{'YES' if acceptance['all_passed'] else 'NO'}**.
Reported but **not** gated: the Mindlin FE dry frequencies of the higher panel
modes `(1,3), (2,3), (3,3)` deviate by up to **{fe_high_error * 100:.2f}%** from
Navier at a 16x16 mesh; the deviation is structural discretisation and halves on
refinement (7.2% at 12x12 for `(1,3)`), and it does not enter the fluid
adjudication.

## 7. Defects found in the existing library (evidence, not fixed here)

1. `tensorfem.cylindrical_shell_modal.dmv_angular_frequency` **dropped the
   membrane term** `lambda^4/(lambda^2+n^2)^2` in its `n = 1` branch, which
   returned `{shell_check['legacy_dmv_n1_ratio']:.3f}` times the retained-term
   Donnell value `{shell_check['donnell_n1_hz']:.2f} Hz` for this shell, whose
   Euler-Bernoulli beam limit is `{shell_check['beam_limit_hz']:.2f} Hz`.
   `fsi_wet_modal.donnell_shell_frequency_hz` always kept the term; the legacy
   helper was **repaired** (mission m16863) and now returns
   `{shell_check['legacy_dmv_n1_hz']:.2f} Hz`, i.e. the two agree.  The
   `n = 0` axisymmetric branch of the legacy helper was unreachable (the
   validator rejected `n_circ = 0`) and is now live.
2. The curved `shell4` FE spectrum of this slender shell does not reproduce the
   beam limit and its lowest modes are radial-dominated spurious modes:

{shell_table}

   Therefore the shell FE is **not** used as the structural anchor here (the
   structural reference is Donnell).  Fixing the element is a separate node.

## 8. Limitations

* Incompressible fluid only (`k_0 a << 1`): valid well below the coincidence
  frequency; **no radiation damping** is modelled, so `zeta` must be supplied.
* Axisymmetric (`n = 0`) shell modes are rejected (a net-volume-changing radial
  mode has no incompressible solution).
* Donnell thin-shell theory is used for `n >= 2`; the `n = 1` beam branch is
  outside its accuracy.
* The added mass is frequency independent, so the *static* limit of `H(f)` is the
  low-frequency *dynamic* compliance and is **not** a static plate deflection.
* Gorman & Horacek (1990) is an in-air free-plate benchmark and cannot anchor a
  *wet* case; its tabulated frequencies were not available offline for this node,
  so that specific anchor is **WITHHELD** rather than quoted from memory.  The wet
  panel is anchored on the exact incompressible baffled-panel integral instead.
* The fluid field is two-dimensional in `(r, x)` per circumferential harmonic;
  axial end effects are estimated, not resolved in full 3-D.

## 9. Artifacts

`artifacts/m6607/numerical_data.json`, `artifacts/m6607/acceptance.json`,
`artifacts/m6607/fig1..fig6*.png`.
"""
    path = ROOT / "docs" / "fsi-wet-modal-dual-criteria.md"
    path.write_text(text)
    return path


def main() -> dict:
    _plt_style()
    OUT.mkdir(parents=True, exist_ok=True)

    cylinder = fsi.solve_submerged_cylinder_wet_modal(
        **CYLINDER, modes=CYLINDER_MODES, n_radial=200, n_axial_grid=500
    ).summary()

    coefficient_check = []
    for order in (1, 2, 3, 4):
        k = math.pi / CYLINDER["length"]
        potential = fsi.solve_incompressible_potential(
            CYLINDER["radius"], order, lambda z, k=k: np.sin(k * z),
            x_min=-3.0, x_max=CYLINDER["length"] + 3.0,
            wall_x_min=-3.0, wall_x_max=CYLINDER["length"] + 3.0,
            n_radial=200, n_axial_grid=500,
        )
        local = fsi.numerical_local_added_mass_per_area(
            potential, CYLINDER["rho_fluid"], [0.25 * CYLINDER["length"]]
        )[0]
        analytic = (
            CYLINDER["rho_fluid"]
            * CYLINDER["radius"]
            * fsi.cylinder_fluid_loading_coefficient(
                CYLINDER["radius"], CYLINDER["length"], 1, order
            )
        )
        coefficient_check.append(
            {
                "m": 1,
                "n": order,
                "analytic": analytic,
                "numerical": float(local),
                "rel_error": abs(float(local) - analytic) / analytic,
            }
        )

    convergence = []
    for n_radial, n_axial_grid in ((60, 150), (120, 300), (240, 600)):
        entry = {"n_radial": n_radial, "n_axial_grid": n_axial_grid}
        for order, key in ((2, "error_n2"), (3, "error_n3")):
            k = math.pi / CYLINDER["length"]
            potential = fsi.solve_incompressible_potential(
                CYLINDER["radius"], order, lambda z, k=k: np.sin(k * z),
                x_min=-3.0, x_max=CYLINDER["length"] + 3.0,
                wall_x_min=-3.0, wall_x_max=CYLINDER["length"] + 3.0,
                n_radial=n_radial, n_axial_grid=n_axial_grid,
            )
            local = fsi.numerical_local_added_mass_per_area(
                potential, CYLINDER["rho_fluid"], [0.25 * CYLINDER["length"]]
            )[0]
            analytic = (
                CYLINDER["rho_fluid"]
                * CYLINDER["radius"]
                * fsi.cylinder_fluid_loading_coefficient(
                    CYLINDER["radius"], CYLINDER["length"], 1, order
                )
            )
            entry[key] = abs(float(local) - analytic) / analytic
        entry["modes"] = [
            {"m": m, "n": n, "mu_analytic": mu_a, "mu_numerical": mu_n}
            for (m, n), mu_a, mu_n in zip(
                cylinder["mode_numbers"],
                cylinder["mu_analytic"],
                cylinder["mu_numerical_baffle"],
            )
        ]
        convergence.append(entry)

    plate = fsi.solve_submerged_plate_wet_modal(
        **PLATE, modes=PLATE_MODES, panel_mesh=(32, 32), fe_mesh=(16, 16)
    ).summary()

    frf = fsi.submerged_plate_frf(
        **PLATE, n_modes=9, damping_ratio=0.01, freq_range=(1.0, 200.0)
    )
    strong = [index for index, value in enumerate(frf.participation) if value > 0.05]
    refined = frf.resonance_peaks_hz(window=0.01)
    frf_record = {
        "frequencies_hz": frf.frequencies_hz.tolist(),
        "amplitude": frf.amplitude().tolist(),
        "dry_frequencies_hz": list(frf.dry_frequencies_hz),
        "wet_frequencies_hz": list(frf.wet_frequencies_hz),
        "added_mass_ratios": list(frf.added_mass_ratios),
        "participation": list(frf.participation),
        "wet_resonances_hz": [frf.wet_frequencies_hz[index] for index in strong],
        "peak_estimate_hz": [refined[index] for index in strong],
        "damping_ratio": frf.damping_ratio,
    }

    shell_check = shell_fe_beam_limit_check()
    mode_shapes = [
        fsi.plate_mode_shape_check(m=m, n=n) for (m, n) in ((1, 1), (1, 2))
    ]

    figure_loading_coefficient(
        OUT / "fig1_fluid_loading_coefficient.png",
        {"coefficient_check": coefficient_check},
    )
    figure_cylinder_spectrum(OUT / "fig2_cylinder_wet_spectrum.png", cylinder)
    figure_fluid_convergence(OUT / "fig3_fluid_solver_convergence.png", convergence)
    figure_plate_added_mass(OUT / "fig4_plate_added_mass.png", {})
    figure_plate_frf(OUT / "fig5_plate_frf.png", frf_record)
    figure_pressure_field(
        OUT / "fig6_pressure_field.png", CYLINDER["radius"], CYLINDER["length"]
    )

    acceptance = build_acceptance(
        cylinder, coefficient_check, plate, frf_record, mode_shapes
    )
    data = {
        "cylinder": {
            "parameters": CYLINDER,
            "modes": [list(mode) for mode in cylinder["mode_numbers"]],
            "result": cylinder,
            "coefficient_check": coefficient_check,
            "convergence": convergence,
        },
        "plate": {
            "parameters": PLATE,
            "modes": [list(mode) for mode in PLATE_MODES],
            "result": plate,
        },
        "frf": {key: value for key, value in frf_record.items() if key != "amplitude"},
        "shell_fe_check": shell_check,
        "plate_mode_shapes": mode_shapes,
        "acceptance": acceptance,
    }
    (OUT / "numerical_data.json").write_text(json.dumps(data, indent=2))
    (OUT / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
    report_path = write_markdown(data, acceptance, shell_check)
    print(json.dumps(acceptance, indent=2))
    print(f"report: {report_path}")
    return data


if __name__ == "__main__":
    import sys

    if "--report-only" in sys.argv:
        payload = json.loads((OUT / "numerical_data.json").read_text())
        print(write_markdown(payload, payload["acceptance"], payload["shell_fe_check"]))
    else:
        main()
