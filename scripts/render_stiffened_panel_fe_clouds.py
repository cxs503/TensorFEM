"""Render filled FE displacement and von-Mises contour maps."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _plot_contour(x, y, values, *, label, title, path, cell_centred=False):
    actual_min = float(values.min())
    actual_max = float(values.max())
    mean = float(values.mean())
    is_constant = actual_max - actual_min <= max(abs(mean), 1.0) * 1e-10

    if cell_centred:
        # Extend element-centre values to the physical boundary so the contour
        # covers the complete Q4 domain rather than stopping at cell centres.
        dx = x[0, 1] - x[0, 0]
        dy = y[1, 0] - y[0, 0]
        x_axis = np.r_[x[0, 0] - dx / 2, x[0], x[0, -1] + dx / 2]
        y_axis = np.r_[y[0, 0] - dy / 2, y[:, 0], y[-1, 0] + dy / 2]
        x, y = np.meshgrid(x_axis, y_axis)
        values = np.pad(values, 1, mode="edge")

    if is_constant:
        # Do not turn round-off noise into false spatial gradients.  Preserve
        # the actual extrema in the caption while rendering the physical
        # constant field at its mean value.
        plotted_values = np.full_like(values, mean)
        pad = max(abs(mean) * 0.01, 1e-12)
        levels = np.linspace(mean - pad, mean + pad, 33)
    else:
        plotted_values = values
        levels = 33

    fig, ax = plt.subplots(figsize=(9, 4.2))
    cloud = ax.contourf(x, y, plotted_values, levels=levels, cmap="turbo")
    fig.colorbar(cloud, ax=ax, label=label)
    ax.set(xlabel="x (m)", ylabel="y (m)", title=title)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    fig.text(
        0.5,
        0.015,
        f"field range: {actual_min:.9g} to {actual_max:.9g} {label.split()[-1]}",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    case = report["cases"][-1]
    field = case["field"]
    nx, ny = case["mesh"]["nx"], case["mesh"]["ny"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nodes = field["nodes"]
    node_x = np.array([row["x"] for row in nodes]).reshape(ny + 1, nx + 1)
    node_y = np.array([row["y"] for row in nodes]).reshape(ny + 1, nx + 1)
    ux_mm = np.array([row["ux"] for row in nodes]).reshape(ny + 1, nx + 1) * 1e3
    _plot_contour(
        node_x,
        node_y,
        ux_mm,
        label="u_x (mm)",
        title=f"Stiffened panel Q4 {nx}x{ny}: axial displacement",
        path=args.output_dir / "displacement.png",
    )

    stress = field["element_stress"]
    stress_x = np.array([row["x"] for row in stress]).reshape(ny, nx)
    stress_y = np.array([row["y"] for row in stress]).reshape(ny, nx)
    vm_mpa = np.array([row["von_mises"] for row in stress]).reshape(ny, nx) * 1e-6
    _plot_contour(
        stress_x,
        stress_y,
        vm_mpa,
        label="von Mises (MPa)",
        title=f"Stiffened panel Q4 {nx}x{ny}: von Mises stress",
        path=args.output_dir / "von_mises.png",
        cell_centred=True,
    )


if __name__ == "__main__":
    main()
