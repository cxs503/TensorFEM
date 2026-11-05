"""Audit script for curved shell buckling qualification.

This script runs the cylindrical shell buckling qualification and generates
the evidence files.
"""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from tensorfem.shell4_curved_buckling import (
    simply_supported_cylindrical_shell_buckling,
    run_cylindrical_shell_buckling_qualification,
)


def run_audit():
    """Run the curved shell buckling audit."""
    # Run the qualification
    report = run_cylindrical_shell_buckling_qualification()

    # Generate convergence plot
    fig, ax = plt.subplots(figsize=(8, 6))
    meshes = [row["mesh"] for row in report["meshes"]]
    errors = [row["relative_error"] for row in report["meshes"]]
    ax.plot(meshes, errors, "o-", label="FE relative error")
    ax.axhline(y=report["tolerance"], color="r", linestyle="--", label=f"Tolerance ({report['tolerance']*100}%)")
    ax.set_xlabel("Mesh resolution")
    ax.set_ylabel("Relative error")
    ax.set_title("Cylindrical Shell Buckling Convergence")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig("docs/assets/shell4-curved-buckling/curved_buckling_convergence.png", dpi=150)
    plt.close(fig)

    # Generate mode shape plot
    case = dict(
        length=1.0,
        radius=0.5,
        nx=16,
        ntheta=16,
        young=210e9,
        poisson=0.3,
        thickness=0.01,
    )
    result = simply_supported_cylindrical_shell_buckling(**case)
    nodes, elements = None, None  # We need to regenerate the mesh for plotting
    # Regenerate mesh for plotting
    from tensorfem.shell4_curved_buckling import cylindrical_shell_mesh
    nodes, elements = cylindrical_shell_mesh(case["length"], case["radius"], 16, 16)

    # Plot the first mode shape
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "3d"})
    # Plot the undeformed shell
    for conn in elements:
        node_coords = nodes[conn].numpy()
        ax.plot(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], "k-", alpha=0.3)
    # Plot the deformed shell (scaled)
    scale = 0.1
    deformed_nodes = nodes.numpy() + result.modes[:, 0][:, None] * scale
    for conn in elements:
        node_coords = deformed_nodes[conn]
        ax.plot(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], "r-", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("First Buckling Mode Shape")
    plt.tight_layout()
    fig.savefig("docs/assets/shell4-curved-buckling/curved_buckling_mode.png", dpi=150)
    plt.close(fig)

    # Save evidence JSON
    evidence = {
        "schema": "tensorfem.shell4-curved-buckling/audit/1.0",
        "report": report,
        "figures": [
            "curved_buckling_convergence.png",
            "curved_buckling_mode.png",
        ],
    }
    Path("docs/assets/shell4-curved-buckling/evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )

    print("Audit completed successfully.")
    print(f"Report: {json.dumps(report, indent=2)}")
    return report


if __name__ == "__main__":
    run_audit()
