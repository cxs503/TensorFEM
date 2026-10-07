"""Render filled displacement and stress contours from hull-girder FE JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _plot(x, y, values, *, title, label, path, cell_centred=False):
    if cell_centred:
        dx = x[0, 1] - x[0, 0]
        dy = y[1, 0] - y[0, 0]
        x_axis = np.r_[x[0, 0] - dx / 2, x[0], x[0, -1] + dx / 2]
        y_axis = np.r_[y[0, 0] - dy / 2, y[:, 0], y[-1, 0] + dy / 2]
        x, y = np.meshgrid(x_axis, y_axis)
        values = np.pad(values, 1, mode="edge")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    cloud = ax.contourf(x, y, values, levels=33, cmap="turbo")
    fig.colorbar(cloud, ax=ax, label=label)
    ax.set(xlabel="x (m)", ylabel="y (m)", title=title)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    fig.text(
        0.5,
        0.015,
        f"field range: {values.min():.7g} to {values.max():.7g} {label.split()[-1]}",
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

    data = json.loads(args.report.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in data["cases"]:
        field = case["field"]
        nx, ny = case["mesh"]["nx"], case["mesh"]["ny"]

        nodes = field["nodes"]
        node_x = np.array([row["x"] for row in nodes]).reshape(ny + 1, nx + 1)
        node_y = np.array([row["y"] for row in nodes]).reshape(ny + 1, nx + 1)
        uy_mm = np.array([row["uy"] for row in nodes]).reshape(ny + 1, nx + 1) * 1e3
        _plot(
            node_x,
            node_y,
            uy_mm,
            title=f"Hull girder Q4 {nx}x{ny}: transverse displacement",
            label="u_y (mm)",
            path=args.output_dir / f"hull_girder_{nx}x{ny}_displacement.png",
        )

        stress = field["element_stress"]
        stress_x = np.array([row["x"] for row in stress]).reshape(ny, nx)
        stress_y = np.array([row["y"] for row in stress]).reshape(ny, nx)
        vm_mpa = np.array([row["von_mises"] for row in stress]).reshape(ny, nx) * 1e-6
        _plot(
            stress_x,
            stress_y,
            vm_mpa,
            title=f"Hull girder Q4 {nx}x{ny}: von Mises stress",
            label="von Mises (MPa)",
            path=args.output_dir / f"hull_girder_{nx}x{ny}_von_mises.png",
            cell_centred=True,
        )


if __name__ == "__main__":
    main()
