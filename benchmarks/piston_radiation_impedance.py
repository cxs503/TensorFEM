"""Finite-frequency (finite-``ka``) baffled-piston radiation impedance benchmark.

Continues ``benchmarks/wet_panel_added_mass.py`` past its documented ``k -> 0``
scope: the full divergent-spherical-wave kernel ``exp(-j k r)/(2 pi r)`` is
discretised and its radiation impedance compared with Rayleigh's exact piston
impedance ``rho c pi a^2 [R1(2ka) + j X1(2ka)]``.  Evidence is written to
``artifacts/piston_radiation_impedance/`` (machine-readable JSON + PNGs).

Anchors
-------
A1  Piston impedance ``Z(ka)``: both the radiation resistance ``Re Z`` and the
    radiation reactance ``Im Z`` must match the closed form to < 3 % for
    ``ka <= 2`` at >= 64 panels/side, and for ``ka = 3`` at >= 96 panels/side
    (the reactance is near its first zero there, so the *relative* error is a
    harsh metric -- the scale-normalised error ``|dZ| / (rho c pi a^2)`` is
    reported alongside).
A2  ``k -> 0`` consistency with ``hydro_added_mass``: the finite-``k`` operator
    must collapse onto the in-phase dipole operator (element-wise) and
    ``Re(Z/(j w))`` must reproduce the anchored added mass ``(8/3) rho a^3``.
A3  Panel application: modal radiation impedance of a simply-supported ship
    side-shell strake, giving the radiation (added) damping ratio that the
    in-phase module explicitly does not model.  Cross-check: the low-``k``
    reactance must reproduce ``mu * m_struct`` from
    ``hydro_added_mass.panel_added_mass_ratio`` (independent code path).
A4  (informative) The finite-``ka`` added mass ``Re(Z/(j w))`` collapses as the
    frequency rises -- the in-phase added mass is not usable above ``ka ~ 1``.

Usage
-----
    PYTHONPATH=src python benchmarks/piston_radiation_impedance.py
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.hydro_added_mass import (
    dipole_added_mass_matrix,
    disc_panels,
    panel_added_mass_ratio,
    piston_radiation_mass,
    sine_mode_values,
)
from tensorfem.hydro_radiation import (
    baffled_kernel_matrix,
    effective_added_mass,
    modal_radiation_impedance,
    piston_impedance_exact,
    planar_radiation_impedance,
)

torch.set_num_threads(4)

OUT_DIR = os.path.join("artifacts", "piston_radiation_impedance")
ASSET_DIR = os.path.join("docs", "assets", "piston-radiation-impedance")

RADIUS = 0.5  # piston radius [m]
RHO_WATER = 1025.0  # sea water [kg/m^3]
SOUND_SPEED = 1500.0  # [m/s]

KA_VALUES = (0.1, 0.5, 1.0, 2.0, 3.0)
N_CELLS = (16, 32, 64, 96)
TOL = 3.0  # percent, repository acceptance threshold

# Ship side-shell strake between frames (same panel as the in-phase benchmark)
PANEL = dict(
    lx=1.2, ly=0.8, thickness=0.008, rho_steel=7850.0, dry_frequency=44.397, wet_frequency=16.250
)


def piston_rows() -> list[dict]:
    """A1/A4: piston impedance error table over (ka, panels per side)."""
    rows = []
    panels = {n: disc_panels(RADIUS, n) for n in N_CELLS}
    for ka in KA_VALUES:
        k = ka / RADIUS
        exact = piston_impedance_exact(RADIUS, k, RHO_WATER, SOUND_SPEED)
        scale = RHO_WATER * SOUND_SPEED * math.pi * RADIUS**2  # rho c A
        for n in N_CELLS:
            centres, areas = panels[n]
            z = planar_radiation_impedance(centres, areas, k, RHO_WATER, SOUND_SPEED)
            omega = k * SOUND_SPEED
            added_mass = effective_added_mass(z, omega)
            rows.append(
                {
                    "ka": ka,
                    "n_cells": n,
                    "n_panels": int(areas.numel()),
                    "wavelength_over_radius": (2.0 * math.pi / k) / RADIUS,
                    "exact_re": exact.real,
                    "exact_im": exact.imag,
                    "fe_re": z.real,
                    "fe_im": z.imag,
                    "re_error_pct": (z.real - exact.real) / exact.real * 100.0,
                    "im_error_pct": (z.imag - exact.imag) / exact.imag * 100.0,
                    "re_error_of_scale_pct": (z.real - exact.real) / scale * 100.0,
                    "im_error_of_scale_pct": (z.imag - exact.imag) / scale * 100.0,
                    "abs_error_of_scale_pct": abs(z - exact) / scale * 100.0,
                    "added_mass": added_mass,
                    "added_mass_rel_to_lowfreq": added_mass
                    / piston_radiation_mass(RADIUS, RHO_WATER),
                }
            )
    return rows


def gate_summary(rows: list[dict]) -> list[dict]:
    """A1 gate: per-ka finest admissible mesh and pass/fail on both components."""
    gates = []
    for ka in KA_VALUES:
        required_cells = 96 if ka > 2.0 else 64
        candidate = [r for r in rows if r["ka"] == ka and r["n_cells"] >= required_cells]
        worst = max(candidate, key=lambda r: max(abs(r["re_error_pct"]), abs(r["im_error_pct"])))
        coarsest = min(candidate, key=lambda r: r["n_cells"])
        gates.append(
            {
                "ka": ka,
                "required_cells": required_cells,
                "re_error_pct": coarsest["re_error_pct"],
                "im_error_pct": coarsest["im_error_pct"],
                "abs_error_of_scale_pct": coarsest["abs_error_of_scale_pct"],
                "worst_component_error_pct": max(
                    abs(coarsest["re_error_pct"]), abs(coarsest["im_error_pct"])
                ),
                "worst_over_all_admissible_pct": max(
                    abs(worst["re_error_pct"]), abs(worst["im_error_pct"])
                ),
                "passed": max(abs(coarsest["re_error_pct"]), abs(coarsest["im_error_pct"])) < TOL,
            }
        )
    return gates


def lowk_consistency() -> dict:
    """A2: finite-k operator -> in-phase dipole operator as k -> 0."""
    centres, areas = disc_panels(RADIUS, 32)
    in_phase = dipole_added_mass_matrix(centres, areas, RHO_WATER)
    k = 1.0e-6
    finite = baffled_kernel_matrix(centres, areas, k, RHO_WATER, SOUND_SPEED)
    denom = float(in_phase.abs().sum())
    elementwise = float((finite.real - in_phase).abs().sum()) / denom
    symmetry = float((finite - finite.transpose(0, 1)).abs().max())
    z = planar_radiation_impedance(centres, areas, k, RHO_WATER, SOUND_SPEED)
    added_mass = effective_added_mass(z, k * SOUND_SPEED)
    exact_mass = piston_radiation_mass(RADIUS, RHO_WATER)
    return {
        "k": k,
        "elementwise_rel_deviation": elementwise,
        "max_asymmetry": symmetry,
        "added_mass_from_impedance": added_mass,
        "raileigh_8_3_rho_a3": exact_mass,
        "added_mass_error_pct": (added_mass - exact_mass) / exact_mass * 100.0,
        "in_phase_operator_sum": float(in_phase.sum()),
        "elementwise_passed": elementwise < 1.0e-5,
        "added_mass_passed": abs(added_mass - exact_mass) / exact_mass * 100.0 < TOL,
    }


def panel_application() -> dict:
    """A3: modal radiation damping of the wet strake, and the low-k cross-check."""
    lx, ly = PANEL["lx"], PANEL["ly"]
    thickness, rho_steel = PANEL["thickness"], PANEL["rho_steel"]
    nx, ny = 60, 40
    dx, dy = lx / nx, ly / ny
    xs = dx * (torch.arange(nx, dtype=torch.float64) + 0.5)
    ys = dy * (torch.arange(ny, dtype=torch.float64) + 0.5)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    centres = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    areas = torch.full((centres.shape[0],), dx * dy, dtype=torch.float64)
    phi = sine_mode_values(centres, lx, ly, 1, 1)

    # Low-k cross-check against the independent in-phase code path.
    in_phase = panel_added_mass_ratio(lx, ly, thickness, rho_steel, RHO_WATER)
    m_struct = in_phase["structural_generalised_mass"]
    k_small = 1.0e-6
    z_small = modal_radiation_impedance(centres, areas, phi, k_small, RHO_WATER, SOUND_SPEED)
    reactance_mass = z_small.imag / (k_small * SOUND_SPEED)

    # Radiation damping at the *wet* frequency (16.25 Hz measured in the
    # in-phase benchmark), plus the half-power bandwidth it implies.
    frequency = PANEL["wet_frequency"]
    omega = 2.0 * math.pi * frequency
    k = omega / SOUND_SPEED
    z = modal_radiation_impedance(centres, areas, phi, k, RHO_WATER, SOUND_SPEED)
    resistance = z.real
    added_mass = z.imag / omega
    wet_modal_mass = m_struct + added_mass
    zeta = resistance / (2.0 * wet_modal_mass * omega)
    return {
        "n_panels": int(areas.numel()),
        "structural_generalised_mass": m_struct,
        "in_phase_mu": in_phase["mu"],
        "in_phase_wet_frequency_ratio": in_phase["wet_frequency_ratio"],
        "lowk_reactance_mass": reactance_mass,
        "lowk_reactance_error_vs_in_phase_pct": (reactance_mass - in_phase["mu"] * m_struct)
        / (in_phase["mu"] * m_struct)
        * 100.0,
        "frequency_hz": frequency,
        "ka_equivalent": k * 0.5 * math.sqrt(lx * ly / math.pi),
        "radiation_resistance": resistance,
        "radiation_added_mass": added_mass,
        "wet_modal_mass": wet_modal_mass,
        "radiation_damping_ratio": zeta,
        "half_power_bandwidth_hz": 2.0 * zeta * frequency,
    }


def plot_impedance_vs_ka(rows: list[dict], path: str) -> None:
    """R/R0 and X/X0 over a dense ka sweep (exact curves) with FE points at 64."""
    ka_dense = [0.005 * i for i in range(1, 1201)]  # 0.005 .. 6
    exact = [piston_impedance_exact(RADIUS, ka / RADIUS, RHO_WATER, SOUND_SPEED) for ka in ka_dense]
    scale = RHO_WATER * SOUND_SPEED * math.pi * RADIUS**2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(ka_dense, [z.real / scale for z in exact], "-", lw=2, label="exact  $R_1(2ka)$")
    axes[1].plot(ka_dense, [z.imag / scale for z in exact], "-", lw=2, label="exact  $X_1(2ka)$")
    fe = [r for r in rows if r["n_cells"] == 64]
    fe_ka = sorted({r["ka"] for r in fe})
    axes[0].plot(
        fe_ka,
        [next(r["fe_re"] for r in fe if r["ka"] == ka) / scale for ka in fe_ka],
        "o",
        ms=6,
        label="FE 64 panels/side",
    )
    axes[1].plot(
        fe_ka,
        [next(r["fe_im"] for r in fe if r["ka"] == ka) / scale for ka in fe_ka],
        "o",
        ms=6,
        label="FE 64 panels/side",
    )
    axes[0].set_xlabel("$ka$")
    axes[0].set_ylabel("$R/(\\rho c \\pi a^2)$")
    axes[0].set_title("radiation resistance")
    axes[1].set_xlabel("$ka$")
    axes[1].set_ylabel("$X/(\\rho c \\pi a^2)$")
    axes[1].set_title("radiation reactance")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_convergence(rows: list[dict], path: str) -> None:
    """Relative error vs panels per side, with the 3 % acceptance line."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ka in KA_VALUES:
        sel = [r for r in rows if r["ka"] == ka]
        axes[0].plot(
            [r["n_panels"] for r in sel], [r["re_error_pct"] for r in sel], "o-", label=f"$ka={ka}$"
        )
        axes[1].plot(
            [r["n_panels"] for r in sel], [r["im_error_pct"] for r in sel], "o-", label=f"$ka={ka}$"
        )
    for ax, title in zip(axes, ("resistance", "reactance")):
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axhline(TOL, color="r", ls="--", lw=1, label="3 % gate")
        ax.axhline(-TOL, color="r", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("panels per side")
        ax.set_ylabel("relative error [%]")
        ax.set_title(f"piston {title} vs Rayleigh closed form")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_damping_field(path: str) -> None:
    """Pressure field radiated by the wet panel mode (1,1) at its wet frequency."""
    lx, ly = PANEL["lx"], PANEL["ly"]
    nx, ny = 60, 40
    dx, dy = lx / nx, ly / ny
    xs = dx * (torch.arange(nx, dtype=torch.float64) + 0.5)
    ys = dy * (torch.arange(ny, dtype=torch.float64) + 0.5)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    centres = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    areas = torch.full((centres.shape[0],), dx * dy, dtype=torch.float64)
    phi = sine_mode_values(centres, lx, ly, 1, 1)
    frequency = PANEL["wet_frequency"]
    omega = 2.0 * math.pi * frequency
    k = omega / SOUND_SPEED
    # Surface pressure for unit normal velocity: p = j w rho sum_j phi_j A_j G.
    src = phi * areas
    pressure = torch.zeros(centres.shape[0], dtype=torch.complex128)
    for start in range(0, centres.shape[0], 256):
        stop = min(start + 256, centres.shape[0])
        dxs = centres[start:stop, None, 0] - centres[None, :, 0]
        dys = centres[start:stop, None, 1] - centres[None, :, 1]
        dist = torch.sqrt(dxs * dxs + dys * dys)
        phase = torch.cos(k * dist) - 1j * torch.sin(k * dist)
        block = src[None, :] * phase / dist
        rows = torch.arange(stop - start)
        block[rows, rows + start] = 0.0
        pressure[start:stop] = block.sum(dim=1)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    field = torch.abs(pressure).reshape(ny, nx)
    mesh = ax.pcolormesh(xs.numpy(), ys.numpy(), field.numpy(), shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="|p| per unit velocity [Pa s/m]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"baffled pressure of mode (1,1) at {frequency:.1f} Hz")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ASSET_DIR, exist_ok=True)

    rows = piston_rows()
    gates = gate_summary(rows)
    lowk = lowk_consistency()
    panel = panel_application()

    fig1 = os.path.join(OUT_DIR, "impedance_vs_ka.png")
    fig2 = os.path.join(OUT_DIR, "mesh_convergence.png")
    fig3 = os.path.join(OUT_DIR, "panel_mode_pressure.png")
    plot_impedance_vs_ka(rows, fig1)
    plot_convergence(rows, fig2)
    plot_damping_field(fig3)
    for src in (fig1, fig2, fig3):
        dest = os.path.join(ASSET_DIR, os.path.basename(src))
        with open(src, "rb") as fh, open(dest, "wb") as out:
            out.write(fh.read())

    passed = (
        all(g["passed"] for g in gates) and lowk["elementwise_passed"] and lowk["added_mass_passed"]
    )
    report = {
        "benchmark": "piston_radiation_impedance",
        "reference": "Rayleigh (1945) Sec. 302: Z = rho c pi a^2 [R1(2ka)+j X1(2ka)]",
        "piston": {"radius": RADIUS, "rho_water": RHO_WATER, "sound_speed": SOUND_SPEED},
        "tolerance_pct": TOL,
        "anchors": {
            "A1_piston_impedance": {"rows": rows, "gates": gates},
            "A2_low_k_consistency": lowk,
            "A3_panel_application": panel,
            "A4_note": "added_mass column of A1 rows shows the collapse above ka~1",
        },
        "figures": [os.path.basename(p) for p in (fig1, fig2, fig3)],
        "passed": bool(passed),
    }
    json_path = os.path.join(OUT_DIR, "validation_report.json")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(ASSET_DIR, "validation_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps({"gates": gates, "lowk": lowk, "panel": panel}, indent=2))
    print("passed:", passed)
    return report


if __name__ == "__main__":
    main()
