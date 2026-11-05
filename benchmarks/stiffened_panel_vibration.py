"""Verification study for ``tensorfem.stiffened_panel_vibration``.

Produces machine-readable evidence
(``benchmarks/stiffened_panel_vibration_evidence.json``) and the report figures
under ``docs/assets/``.  Every reference number is the closed-form first-order
shear deformation (Mindlin) Navier eigenvalue of the *same* orthotropic
rigidity; nothing is copied from the finite element output.

Run:
    OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/stiffened_panel_vibration.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.plate import PlateRigidity, q4_mindlin_stiffness, structured_square_mesh  # noqa: E402
from tensorfem.plate_vibration import kirchhoff_angular_frequency, plate_rigidity  # noqa: E402
from tensorfem.stiffened_panel_vibration import (  # noqa: E402
    StiffenerGeometry,
    fsdt_navier_angular_frequency,
    fsdt_navier_spectrum,
    panel_mode_table,
    smeared_panel_rigidity,
    smearing_wavelength_warning,
    solve_panel_vibration,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
EVIDENCE = ROOT / "benchmarks" / "stiffened_panel_vibration_evidence.json"

GATE = 0.03
STEEL = dict(young=200.0e9, poisson=0.3, density=7850.0)


def _hz(omega: float) -> float:
    return omega / (2.0 * math.pi)


def _element_identity() -> dict:
    """Orthotropic entry point must reproduce the isotropic matrix exactly."""
    nodes, elements = structured_square_mesh(2, 1.0)
    xy = nodes[elements[0]]
    payload = {}
    for treatment in ("sri", "mitc"):
        built_in = q4_mindlin_stiffness(xy, STEEL["young"], STEEL["poisson"], 0.01,
                                        shear_treatment=treatment)
        via_rigidity = q4_mindlin_stiffness(
            xy, shear_treatment=treatment,
            rigidity=PlateRigidity.isotropic(STEEL["young"], STEEL["poisson"], 0.01),
        )
        payload[treatment] = float((built_in - via_rigidity).abs().max() / built_in.abs().max())
    return payload


def _shear_bias_table() -> list[dict]:
    """FSDT Navier vs Kirchhoff Navier for the isotropic plate, per a/h."""
    rows = []
    for thickness in (0.01, 0.02, 0.05, 0.1):
        rigidity = PlateRigidity.isotropic(STEEL["young"], STEEL["poisson"], thickness)
        fsdt = _hz(fsdt_navier_angular_frequency(
            rigidity=rigidity, density=STEEL["density"], thickness=thickness,
            length=1.0, width=1.0, half_waves_x=1, half_waves_y=1))
        kirchhoff = _hz(kirchhoff_angular_frequency(
            length=1.0, width=1.0, young=STEEL["young"], poisson=STEEL["poisson"],
            thickness=thickness, density=STEEL["density"],
            half_waves_x=1, half_waves_y=1))
        rows.append({
            "thickness_m": thickness,
            "a_over_h": 1.0 / thickness,
            "kirchhoff_hz": kirchhoff,
            "fsdt_hz": fsdt,
            "relative_bias": (fsdt - kirchhoff) / kirchhoff,
        })
    return rows


def _orthotropic_case() -> dict:
    """Panel A: synthetic orthotropic rigidity with dx/dy = 6."""
    thickness = 0.01
    rigid = plate_rigidity(STEEL["young"], STEEL["poisson"], thickness)
    shear = 5.0 / 6.0 * STEEL["young"] / (2.0 * (1.0 + STEEL["poisson"])) * thickness
    rigidity = PlateRigidity(
        dx=6.0 * rigid, dy=rigid,
        dxy=STEEL["poisson"] * math.sqrt(6.0) * rigid,
        dt=rigid * (1.0 - STEEL["poisson"]) / 2.0, sx=shear, sy=shear,
    )
    length, width = 1.2, 0.8
    reference = fsdt_navier_spectrum(
        rigidity=rigidity, density=STEEL["density"], thickness=thickness,
        length=length, width=width, max_half_waves_x=3, max_half_waves_y=3,
    )
    levels = []
    finest = None
    for nx, ny in ((8, 6), (12, 9), (16, 12)):
        result = solve_panel_vibration(
            nx, ny, rigidity=rigidity, length=length, width=width,
            thickness=thickness, density=STEEL["density"], modes=4,
        )
        rows = panel_mode_table(
            result, rigidity=rigidity, length=length, width=width,
            thickness=thickness, density=STEEL["density"],
        )
        level = {
            "elements": [nx, ny],
            "modes": [
                {
                    "label": [row.half_waves_x, row.half_waves_y],
                    "overlap": row.overlap,
                    "fe_hz": row.frequency_fe_hz,
                    "reference_hz": row.frequency_reference_hz,
                    "relative_error": row.relative_error,
                }
                for row in rows
            ],
        }
        levels.append(level)
        finest = level
    first = [level["modes"][0]["relative_error"] for level in levels]
    order = math.log(first[0] / first[-1]) / math.log(16.0 / 8.0)
    return {
        "rigidity": {
            "dx": rigidity.dx, "dy": rigidity.dy, "dxy": rigidity.dxy,
            "dt": rigidity.dt, "sx": rigidity.sx, "sy": rigidity.sy,
            "navier_twist": rigidity.navier_twist,
        },
        "length_m": length, "width_m": width, "thickness_m": thickness,
        "reference_spectrum": [
            {"label": [mode.half_waves_x, mode.half_waves_y],
             "reference_hz": mode.frequency_hz}
            for mode in reference[:6]
        ],
        "levels": levels,
        "fundamental_convergence_order": order,
        "worst_error_finest": max(mode["relative_error"] for mode in finest["modes"]),
        "gate_pass": max(mode["relative_error"] for mode in finest["modes"]) < GATE,
    }


def _ship_panel_case() -> dict:
    """Panel B: deck grillage 3.2 m x 2.4 m, 12 mm plating, 200x12 flats at 0.8 m."""
    thickness = 0.012
    longitudinal = StiffenerGeometry(depth=0.2, web_thickness=0.012, spacing=0.8)
    transverse = StiffenerGeometry(depth=0.2, web_thickness=0.012, spacing=0.8)
    length, width = 3.2, 2.4
    rigidity = smeared_panel_rigidity(
        young=STEEL["young"], poisson=STEEL["poisson"], thickness=thickness,
        longitudinal=longitudinal, transverse=transverse,
    )
    rigidity_twist = smeared_panel_rigidity(
        young=STEEL["young"], poisson=STEEL["poisson"], thickness=thickness,
        longitudinal=longitudinal, transverse=transverse, twist_stiffeners=True,
    )
    result = solve_panel_vibration(
        20, 16, rigidity=rigidity, length=length, width=width,
        thickness=thickness, density=STEEL["density"], modes=4,
    )
    rows = panel_mode_table(
        result, rigidity=rigidity, length=length, width=width,
        thickness=thickness, density=STEEL["density"],
    )
    warning = smearing_wavelength_warning(
        rigidity=rigidity, density=STEEL["density"], thickness=thickness,
        length=length, width=width, stiffener_spacing=longitudinal.spacing,
        max_half_waves_x=3, max_half_waves_y=3,
    )
    f1_twist = _hz(fsdt_navier_angular_frequency(
        rigidity=rigidity_twist, density=STEEL["density"], thickness=thickness,
        length=length, width=width))
    f1_plain = _hz(fsdt_navier_angular_frequency(
        rigidity=rigidity, density=STEEL["density"], thickness=thickness,
        length=length, width=width))
    f1_thick = _hz(fsdt_navier_angular_frequency(
        rigidity=smeared_panel_rigidity(
            young=STEEL["young"], poisson=STEEL["poisson"], thickness=thickness + 0.002,
            longitudinal=longitudinal, transverse=transverse),
        density=STEEL["density"], thickness=thickness + 0.002, length=length, width=width))
    isotropic_reference = _hz(fsdt_navier_angular_frequency(
        rigidity=PlateRigidity.isotropic(STEEL["young"], STEEL["poisson"], thickness),
        density=STEEL["density"], thickness=thickness, length=length, width=width))
    return {
        "length_m": length, "width_m": width, "thickness_m": thickness,
        "stiffener": {"depth_m": longitudinal.depth,
                      "web_thickness_m": longitudinal.web_thickness,
                      "spacing_m": longitudinal.spacing},
        "rigidity": {
            "dx": rigidity.dx, "dy": rigidity.dy, "dxy": rigidity.dxy,
            "dt": rigidity.dt, "sx": rigidity.sx, "sy": rigidity.sy,
            "dx_over_plate_rigidity": rigidity.dx / plate_rigidity(
                STEEL["young"], STEEL["poisson"], thickness),
        },
        "modes": [
            {
                "label": [row.half_waves_x, row.half_waves_y],
                "overlap": row.overlap,
                "fe_hz": row.frequency_fe_hz,
                "reference_hz": row.frequency_reference_hz,
                "relative_error": row.relative_error,
            }
            for row in rows
        ],
        "elements": [20, 16],
        "worst_error": max(row.relative_error for row in rows),
        "gate_pass": max(row.relative_error for row in rows) < GATE,
        "unsmeared_modes": [[mode.half_waves_x, mode.half_waves_y] for mode in warning],
        "sensitivity": {
            "f1_plain_hz": f1_plain,
            "f1_with_stiffener_torsion_hz": f1_twist,
            "stiffener_torsion_shift": (f1_twist - f1_plain) / f1_plain,
            "f1_thickness_14mm_hz": f1_thick,
            "thickness_shift": (f1_thick - f1_plain) / f1_plain,
            "f1_unstiffened_plate_hz": isotropic_reference,
            "stiffening_gain": (f1_plain - isotropic_reference) / isotropic_reference,
        },
    }


def _figure_mode_shape(orthotropic: dict) -> Path:
    """FE (1,2) mode shape against the analytic sine product."""
    thickness = orthotropic["thickness_m"]
    data = orthotropic["rigidity"]
    rigid = PlateRigidity(dx=data["dx"], dy=data["dy"], dxy=data["dxy"],
                          dt=data["dt"], sx=data["sx"], sy=data["sy"])
    length, width = orthotropic["length_m"], orthotropic["width_m"]
    nx, ny = 16, 12
    result = solve_panel_vibration(
        nx, ny, rigidity=rigid, length=length, width=width,
        thickness=thickness, density=STEEL["density"], modes=4,
    )
    labels = panel_mode_table(
        result, rigidity=rigid, length=length, width=width,
        thickness=thickness, density=STEEL["density"],
    )
    matches = [index for index, row in enumerate(labels)
               if (row.half_waves_x, row.half_waves_y) == (1, 2)]
    if not matches:  # pragma: no cover - the case always contains (1,2)
        raise RuntimeError("mode (1,2) not found")
    target = matches[0]
    row = labels[target]
    x = torch.linspace(0.0, length, nx + 1)
    y = torch.linspace(0.0, width, ny + 1)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    analytic = torch.sin(math.pi * xx.double() / length) * torch.sin(
        2.0 * math.pi * yy.double() / width)
    analytic = analytic / analytic.abs().max()
    w = result.modes[0::3, target].reshape(ny + 1, nx + 1)
    if float(w.double().flatten() @ analytic.flatten()) < 0.0:
        w = -w  # the eigenvector sign is arbitrary; align before differencing
    w = w / w.abs().max()
    residual = float((w.double() - analytic).abs().max() / analytic.abs().max())

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    mesh0 = axes[0].pcolormesh(xx, yy, w, cmap="RdBu_r", shading="auto")
    figure.colorbar(mesh0, ax=axes[0], shrink=0.85)
    axes[0].set_title(f"FE mode (1,2), {row.frequency_fe_hz:.2f} Hz")
    mesh1 = axes[1].pcolormesh(xx, yy, analytic.float(), cmap="RdBu_r", shading="auto")
    figure.colorbar(mesh1, ax=axes[1], shrink=0.85)
    axes[1].set_title(f"analytic sine, {row.frequency_reference_hz:.2f} Hz")
    axes[2].pcolormesh(xx, yy, (w.double() - analytic).float(), cmap="PuOr_r", shading="auto")
    axes[2].set_title(f"difference, max {residual * 100:.2f}%")
    for axis in axes:
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
    figure.suptitle("Panel A orthotropic mode (1,2): finite element vs closed-form Navier shape")
    figure.tight_layout()
    path = ASSETS / "stiffened-panel-mode-shape.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    orthotropic["mode12_shape_residual"] = residual
    return path


def _figure_convergence(orthotropic: dict) -> Path:
    levels = orthotropic["levels"]
    sizes = [1.0 / level["elements"][0] for level in levels]
    errors = [100.0 * level["modes"][0]["relative_error"] for level in levels]
    worst = [100.0 * max(mode["relative_error"] for mode in level["modes"]) for level in levels]
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    axis.loglog(sizes, errors, "o-", label="fundamental (1,1)")
    axis.loglog(sizes, worst, "s--", label="worst of first 4 modes")
    reference = [errors[0] * (sizes[0] / size) ** 2.0 for size in sizes]
    axis.loglog(sizes, reference, "k:", label=r"slope 2 reference ($h^2$)")
    axis.axhline(GATE * 100.0, color="crimson", linestyle="-.", label="3% gate")
    axis.set_yticks([0.1, 0.3, 1.0, 3.0, 10.0])
    axis.set_yticklabels(["0.1%", "0.3%", "1%", "3%", "10%"])
    axis.set_xlabel("element size h / panel length")
    axis.set_ylabel("relative error vs FSDT Navier")
    axis.set_title("Panel A orthotropic convergence (Q4 Mindlin, SRI)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = ASSETS / "stiffened-panel-convergence.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def _figure_spectrum(orthotropic: dict, ship: dict) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for axis, case, title in (
        (axes[0], {"modes": orthotropic["levels"][-1]["modes"]},
         "Panel A: orthotropic dx/dy = 6 (16x12 mesh)"),
        (axes[1], ship, "Panel B: deck grillage 200x12 flat @0.8 m"),
    ):
        modes = case["modes"]
        labels = [f"({mode['label'][0]},{mode['label'][1]})" for mode in modes]
        positions = range(len(modes))
        axis.bar([p - 0.19 for p in positions], [mode["reference_hz"] for mode in modes],
                 width=0.38, label="FSDT Navier (closed form)")
        axis.bar([p + 0.19 for p in positions], [mode["fe_hz"] for mode in modes],
                 width=0.38, label="FE (Q4 Mindlin)")
        for p, mode in zip(positions, modes):
            axis.annotate(f"{mode['relative_error'] * 100:.2f}%",
                          (p, max(mode["fe_hz"], mode["reference_hz"])), ha="center",
                          va="bottom", fontsize=7)
        axis.set_xticks(list(positions))
        axis.set_xticklabels(labels)
        axis.set_ylabel("frequency [Hz]")
        axis.set_title(title)
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("Simply supported stiffened panel: FE frequencies against the closed-form reference")
    figure.tight_layout()
    path = ASSETS / "stiffened-panel-spectrum.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def _figure_ship_mode(ship: dict) -> Path:
    data = ship["rigidity"]
    rigid = PlateRigidity(dx=data["dx"], dy=data["dy"], dxy=data["dxy"],
                          dt=data["dt"], sx=data["sx"], sy=data["sy"])
    length, width = ship["length_m"], ship["width_m"]
    thickness = ship["thickness_m"]
    nx, ny = 16, 12
    result = solve_panel_vibration(
        nx, ny, rigidity=rigid, length=length, width=width,
        thickness=thickness, density=STEEL["density"], modes=2,
    )
    w = result.modes[0::3, 0].reshape(ny + 1, nx + 1)
    w = w / w.abs().max()
    x = torch.linspace(0.0, length, nx + 1)
    y = torch.linspace(0.0, width, ny + 1)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    mesh0 = axis.pcolormesh(xx, yy, w, cmap="RdBu_r", shading="auto")
    figure.colorbar(mesh0, ax=axis, shrink=0.85)
    axis.set_xlabel("x [m] (spans two frame bays)")
    axis.set_ylabel("y [m]")
    axis.set_title(f"Deck grillage fundamental mode, FE (1,1) = {float(result.frequencies_hz[0]):.1f} Hz")
    for station in (0.8, 1.6, 2.4):  # longitudinal lines on the panel
        axis.axvline(station, color="0.35", lw=0.6, ls=":")
    for station in (0.8, 1.6):
        axis.axhline(station, color="0.35", lw=0.6, ls=":")
    figure.tight_layout()
    path = ASSETS / "stiffened-panel-ship-mode1.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def main() -> None:
    torch.set_grad_enabled(False)
    ASSETS.mkdir(parents=True, exist_ok=True)
    identity = _element_identity()
    shear = _shear_bias_table()
    orthotropic = _orthotropic_case()
    ship = _ship_panel_case()
    figures = {
        "mode_shape": _figure_mode_shape(orthotropic),
        "convergence": _figure_convergence(orthotropic),
        "spectrum": _figure_spectrum(orthotropic, ship),
        "ship_mode1": _figure_ship_mode(ship),
    }
    payload = {
        "module": "tensorfem.stiffened_panel_vibration",
        "gate": GATE,
        "element_identity_max_relative_deviation": identity,
        "isotropic_fsdt_vs_kirchhoff": shear,
        "panel_a_orthotropic": orthotropic,
        "panel_b_ship_grillage": ship,
        "figures": {name: str(path.relative_to(ROOT)) for name, path in figures.items()},
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "gate_pass_orthotropic": orthotropic["gate_pass"],
        "gate_pass_ship": ship["gate_pass"],
        "worst_orthotropic": orthotropic["worst_error_finest"],
        "worst_ship": ship["worst_error"],
        "convergence_order": orthotropic["fundamental_convergence_order"],
        "mode12_shape_residual": orthotropic.get("mode12_shape_residual"),
    }, indent=2))
    print("evidence:", EVIDENCE.relative_to(ROOT))


if __name__ == "__main__":
    main()