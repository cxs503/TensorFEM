"""Generate the HEX20 distortion/curvature qualification evidence.

Outputs
-------
``benchmarks/hex20_distortion_audit.json``
    Machine-readable evidence: amplitude/curvature sweeps, rank audits, exact
    volume comparison, data hash.
``benchmarks/hex20_distortion_audit.md``
    Markdown tables rendered from the same dictionary (no hand-typed numbers).
``docs/assets/hex20_distortion/hex20_distortion.png``
    Two-panel figure: patch residual versus distortion amplitude and versus
    curvature, correct kernel vs the single-Jacobian bug control.

Everything here is a handful of 60-DOF element problems, so the script runs in
seconds and is safe to run under load.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfem.element_rank_audit import audit_element, rigid_body_basis_solid3d  # noqa: E402
from tensorfem.hex20 import structured_hex20_mesh  # noqa: E402
from tensorfem.hex20_distortion_audit import (  # noqa: E402
    boundary_nodes,
    cylindrical_bend,
    element_stiffness,
    element_stiffness_reduced,
    element_volume,
    energy_identity_residual,
    gauss_rule,
    graded_shear,
    patch_residual,
    warp,
)
from tensorfem.solid3d import elasticity_matrix_3d  # noqa: E402

D = torch.float64
GRAD = torch.tensor(
    [[0.012, 0.007, -0.003], [-0.002, -0.009, 0.005], [0.004, 0.006, 0.015]], dtype=D
)
OFFSET = torch.tensor([0.1, -0.2, 0.05], dtype=D)
AMPLITUDES = (0.0, 0.1, 0.2, 0.3, 0.4)
RADII = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 20.0)


def _mesh(kind, amplitude=0.0, radius=0.0, nx=2, ny=2, nz=2):
    nodes, cells = structured_hex20_mesh(2.0, 1.0, 1.0, nx, ny, nz, dtype=D)
    ref = nodes.clone()
    if kind == "warp":
        nodes = warp(nodes, amplitude)
    elif kind == "graded_shear":
        nodes = graded_shear(nodes, amplitude)
    elif kind == "bent":
        nodes = cylindrical_bend(nodes, radius, axis_x=0.0)
    return nodes, cells, boundary_nodes(ref, 2.0, 1.0, 1.0)


def sweeps():
    out = {}
    for kind in ("warp", "graded_shear"):
        rows = []
        for a in AMPLITUDES:
            nodes, cells, bnd = _mesh(kind, amplitude=a)
            good = patch_residual(nodes, cells, bnd, GRAD, OFFSET)
            bad = patch_residual(nodes, cells, bnd, GRAD, OFFSET, center_jacobian=True)
            rows.append(
                {
                    "amplitude": float(a),
                    "displacement_relative_error": good["displacement_relative_error"],
                    "centroid_stress_relative_error": good["centroid_stress_relative_error"],
                    "energy_identity_relative_error": energy_identity_residual(nodes, cells, GRAD),
                    "control_displacement_relative_error": bad["displacement_relative_error"],
                }
            )
        out[kind] = rows
    rows = []
    for r in RADII:
        nodes, cells, bnd = _mesh("bent", radius=r)
        good = patch_residual(nodes, cells, bnd, GRAD, OFFSET)
        bad = patch_residual(nodes, cells, bnd, GRAD, OFFSET, center_jacobian=True)
        vol = float(element_volume(nodes[cells.long()]).sum())
        exact = 2.0 * (1.0 + 1.0 / (2.0 * r))
        rows.append(
            {
                "radius": float(r),
                "curvature": 1.0 / r,
                "displacement_relative_error": good["displacement_relative_error"],
                "centroid_stress_relative_error": good["centroid_stress_relative_error"],
                "energy_identity_relative_error": energy_identity_residual(nodes, cells, GRAD),
                "control_displacement_relative_error": bad["displacement_relative_error"],
                "volume": vol,
                "analytic_volume": exact,
                "volume_relative_error": abs(vol - exact) / exact,
            }
        )
        nodes4, cells4, _ = _mesh("bent", radius=r, nx=4, ny=2, nz=2)
        vol4 = float(element_volume(nodes4[cells4.long()]).sum())
        rows[-1]["volume_relative_error_nx4"] = abs(vol4 - exact) / exact
    out["bent"] = rows
    return out


def ranks():
    rows = []
    for label, kind, amp, rad in (
        ("regular cube", "regular", 0.0, 0.0),
        ("warp a=0.3", "warp", 0.3, 0.0),
        ("graded shear a=0.4", "graded_shear", 0.4, 0.0),
        ("bent R=2", "bent", 0.0, 2.0),
        ("bent R=1", "bent", 0.0, 1.0),
    ):
        nodes, cells, _ = _mesh(kind, amp, rad, nx=1, ny=1, nz=1)
        x = nodes[cells.long()]
        d = elasticity_matrix_3d(x.new_full((len(x),), 70e9), x.new_full((len(x),), 0.27))
        kf, _ = element_stiffness(x, d)
        full = audit_element(f"hex20 3x3x3 ({label})", kf[0], rigid_body_basis_solid3d(x[0]))
        kr, _ = element_stiffness_reduced(x, d)
        red = audit_element(f"hex20 2x2x2 ({label})", kr[0], rigid_body_basis_solid3d(x[0]))
        rows.append(
            {
                "geometry": label,
                "full_null_dim": int(full.null_dim),
                "full_spurious": int(full.spurious),
                "full_first_positive_ratio": float(full.first_positive_ratio),
                "reduced_null_dim": int(red.null_dim),
                "reduced_spurious": int(red.spurious),
                "reduced_first_positive_ratio": float(red.first_positive_ratio),
            }
        )
    return rows


def markdown(data):
    lines = [
        "### Distortion sweep (serendipity brick, 2x2x2 elements, 27 free nodes)",
        "",
        "| warp a | displacement | centroid stress | energy identity | control (single Jacobian) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in data["sweeps"]["warp"]:
        lines.append(
            f"| {r['amplitude']:.2f} | {r['displacement_relative_error']:.3e} | "
            f"{r['centroid_stress_relative_error']:.3e} | {r['energy_identity_relative_error']:.3e} | "
            f"{r['control_displacement_relative_error']:.3e} |"
        )
    lines += [
        "",
        "| graded shear a | displacement | centroid stress | energy identity | control (single Jacobian) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in data["sweeps"]["graded_shear"]:
        lines.append(
            f"| {r['amplitude']:.2f} | {r['displacement_relative_error']:.3e} | "
            f"{r['centroid_stress_relative_error']:.3e} | {r['energy_identity_relative_error']:.3e} | "
            f"{r['control_displacement_relative_error']:.3e} |"
        )
    lines += [
        "",
        "| bend radius R | displacement | centroid stress | volume error (2 elem) | volume error (4 elem) | control |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in data["sweeps"]["bent"]:
        lines.append(
            f"| {r['radius']:.1f} | {r['displacement_relative_error']:.3e} | "
            f"{r['centroid_stress_relative_error']:.3e} | {r['volume_relative_error']:.3%} | "
            f"{r['volume_relative_error_nx4']:.3%} | {r['control_displacement_relative_error']:.3e} |"
        )
    lines += ["", "| geometry | 3x3x3 null / spurious | 2x2x2 null / spurious |", "|---|---:|---:|"]
    for r in data["ranks"]:
        lines.append(
            f"| {r['geometry']} | {r['full_null_dim']} / {r['full_spurious']} | "
            f"{r['reduced_null_dim']} / {r['reduced_spurious']} |"
        )
    return "\n".join(lines) + "\n"


def figure(data, path: pathlib.Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    for kind, style in (("warp", "o-"), ("graded_shear", "s-")):
        rows = data["sweeps"][kind]
        ax.semilogy(
            [r["amplitude"] for r in rows],
            [r["displacement_relative_error"] for r in rows],
            style,
            label=f"correct kernel, {kind.replace('_', ' ')}",
        )
        ax.semilogy(
            [r["amplitude"] for r in rows],
            [r["control_displacement_relative_error"] for r in rows],
            style,
            mfc="none",
            label=f"single-Jacobian control, {kind.replace('_', ' ')}",
        )
    ax.axhline(1e-11, color="k", ls=":", lw=1, label="roundoff gate 1e-11")
    ax.set_xlabel("distortion amplitude a")
    ax.set_ylabel("relative patch-test residual")
    ax.set_title("(a) warped / skewed bricks")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[1]
    rows = data["sweeps"]["bent"]
    xr = [r["curvature"] for r in rows]
    ax.loglog(xr, [r["displacement_relative_error"] for r in rows], "o-", label="correct kernel")
    ax.loglog(
        xr,
        [r["control_displacement_relative_error"] for r in rows],
        "s--",
        label="single-Jacobian control",
    )
    ax.axhline(1e-11, color="k", ls=":", lw=1, label="roundoff gate 1e-11")
    ax.set_xlabel("curvature 1/R of the bent sector")
    ax.set_ylabel("relative patch-test residual")
    ax.set_title("(b) genuinely curved bricks")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    data = {
        "element": "HEX20 serendipity brick, 60 dof",
        "production_integration": "3x3x3 Gauss (5/9, 8/9, 5/9)",
        "gauss_points_full": len(gauss_rule()),
        "field": "u(x) = A x + b with A and b fixed in this script",
        "linear_field_gradient": GRAD.tolist(),
        "linear_field_offset": OFFSET.tolist(),
        "sweeps": sweeps(),
        "ranks": ranks(),
    }
    data["data_sha256"] = hashlib.sha256(
        json.dumps(data, sort_keys=True, default=float).encode()
    ).hexdigest()
    out = root / "benchmarks" / "hex20_distortion_audit.json"
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (root / "benchmarks" / "hex20_distortion_audit.md").write_text(markdown(data))
    fig_dir = root / "docs" / "assets" / "hex20_distortion"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure(data, fig_dir / "hex20_distortion.png")
    print(markdown(data))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
