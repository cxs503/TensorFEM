"""Generate the element rank-deficiency audit evidence (JSON + figures).

Run from the repository root::

    PYTHONPATH=src python3 scripts/audit_element_rank.py

Outputs
-------
``docs/benchmarks/tutorials/element-rank-audit.json``
    Every scalar quoted in ``docs/element-rank-deficiency-audit.md``.
``docs/assets/element_rank_element_nullspace.png``
    Spurious zero-energy count per element formulation.
``docs/assets/element_rank_free_free_modes.png``
    Free-free square-plate spectrum with the SRI rule (4 zero modes) and the
    assumed-shear ``mitc`` rule (3 zero modes); first elastic frequency converges
    to the literature value.
``docs/assets/element_rank_mesh_scaling.png``
    Softest elastic eigenvalue of the free plate mesh against refinement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.element_rank_audit import (  # noqa: E402
    audit_library,
    audit_plate_mesh_scan,
    audit_shell_mesh_scan,
    summary_table,
)
from tensorfem.plate import q4_mindlin_stiffness  # noqa: E402
from tensorfem.plate_vibration import (  # noqa: E402
    kirchhoff_angular_frequency,
    q4_plate_mass_matrix,
    structured_rectangular_mesh,
)

YOUNG = 200.0e9
POISSON = 0.3
THICKNESS = 0.01
DENSITY = 7800.0
SIDE = 1.0
# Classical free-free square-plate frequency parameter lambda = omega a^2 sqrt(rho h / D).
LEISSA_FREE_FREE = 13.49


def plate_mesh(n: int, treatment: str):
    nodes, cells = structured_rectangular_mesh(n, n, SIDE, SIDE)
    ndof = 3 * len(nodes)
    k = torch.zeros((ndof, ndof))
    m = torch.zeros((ndof, ndof))
    for cell in cells.tolist():
        ke = q4_mindlin_stiffness(
            nodes[cell], YOUNG, POISSON, THICKNESS, shear_treatment=treatment
        )
        me = q4_plate_mass_matrix(nodes[cell], DENSITY, THICKNESS)
        idx = torch.tensor([3 * i + a for i in cell for a in range(3)])
        k[idx[:, None], idx] = k[idx[:, None], idx] + ke
        m[idx[:, None], idx] = m[idx[:, None], idx] + me
    return nodes, k, m


def simply_supported_dofs(nodes):
    fixed = []
    for i, (x, y) in enumerate(zip(nodes[:, 0].tolist(), nodes[:, 1].tolist())):
        on_x = abs(x) < 1e-10 or abs(x - SIDE) < 1e-10
        on_y = abs(y) < 1e-10 or abs(y - SIDE) < 1e-10
        if on_x or on_y:
            fixed.append(3 * i)
        if on_x:
            fixed.append(3 * i + 2)
        if on_y:
            fixed.append(3 * i + 1)
    return sorted(set(fixed))


def eigenvalues(k, m, free):
    kr = k[free][:, free]
    mr = m[free][:, free]
    lower = torch.linalg.cholesky(mr)
    inverse = torch.linalg.inv(lower)
    a = inverse @ kr @ inverse.transpose(0, 1)
    a = 0.5 * (a + a.transpose(0, 1))
    return torch.linalg.eigvalsh(a).clamp(min=0.0).sqrt()


def modal_study(sizes=(4, 6, 8, 10, 12)):
    reference = kirchhoff_angular_frequency(
        length=SIDE, width=SIDE, density=DENSITY, young=YOUNG, poisson=POISSON,
        thickness=THICKNESS,
    )
    rigidity = YOUNG * THICKNESS**3 / (12.0 * (1.0 - POISSON**2))
    free_free_reference = LEISSA_FREE_FREE * math.sqrt(rigidity / (DENSITY * THICKNESS * SIDE**4))
    rows = []
    for n in sizes:
        for treatment in ("sri", "mitc"):
            nodes, k, m = plate_mesh(n, treatment)
            all_dofs = torch.arange(k.shape[0])
            omega_free = eigenvalues(k, m, all_dofs)
            fixed = simply_supported_dofs(nodes)
            free = torch.tensor([i for i in range(k.shape[0]) if i not in set(fixed)])
            omega_ss = eigenvalues(k, m, free)
            rows.append(
                {
                    "n": n,
                    "shear_treatment": treatment,
                    "dofs": int(k.shape[0]),
                    "free_free_zero_modes": int((omega_free[:8] < 1.0).sum()),
                    "free_free_lowest": [float(v) for v in omega_free[:6].tolist()],
                    "free_free_omega1": float(omega_free[int((omega_free[:8] < 1.0).sum())]),
                    "simply_supported_omega11": float(omega_ss[0]),
                    "simply_supported_error_percent": 100.0 * (float(omega_ss[0]) - reference) / reference,
                }
            )
    return {
        "reference_simply_supported_omega11": reference,
        "reference_free_free_omega1_literature": free_free_reference,
        "reference_note": (
            "Simply supported value is the exact Navier frequency from "
            "tensorfem.plate_vibration.kirchhoff_angular_frequency. The free-free value uses the "
            "classical Leissa frequency parameter 13.49 for a square plate (thin-plate theory)."
        ),
        "rows": rows,
    }


def figure_elements(records, path: Path):
    labels = [r.element for r in records]
    expected = [r.expected_rigid_dim for r in records]
    spurious = [r.spurious for r in records]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    positions = range(len(records))
    ax.bar([p - 0.2 for p in positions], expected, width=0.4, label="expected null dim (analytic)", color="#4c72b0")
    ax.bar([p + 0.2 for p in positions], spurious, width=0.4, label="spurious zero-energy", color="#c44e52")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("zero eigenvalues of the free element stiffness")
    ax.set_title("Rank-deficiency audit of the free element stiffness (no boundary conditions)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_free_free(modal, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for treatment, colour in (("sri", "#c44e52"), ("mitc", "#4c72b0")):
        rows = [r for r in modal["rows"] if r["shear_treatment"] == treatment]
        lows = rows[-1]["free_free_lowest"]
        ax[0].plot(range(len(lows)), [max(v, 1e-3) for v in lows], "o-", color=colour,
                   label=f"{treatment.upper()} (n={rows[-1]['n']})")
        hs = [SIDE / r["n"] for r in rows]
        ax[1].plot(hs, [r["simply_supported_error_percent"] for r in rows], "s-", color=colour,
                   label=f"{treatment.upper()}")
    ax[0].set_yscale("log")
    ax[0].axhline(modal["reference_free_free_omega1_literature"], color="k", ls="--", lw=1,
                  label="free-free omega1, Leissa 13.49")
    ax[0].set_xlabel("mode index")
    ax[0].set_ylabel("angular frequency [rad/s] (log)")
    ax[0].set_title("Free-free square plate: 4 vs 3 zero modes")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("element size h")
    ax[1].set_ylabel("simply supported omega11 error [%]")
    ax[1].set_title("Accuracy is unaffected by the shear field")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_scaling(plate_scan, shell_scan_sri, shell_scan_mitc, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy([1.0 / r.n for r in plate_scan], [r.first_positive_ratio for r in plate_scan], "o-",
                   color="#4c72b0", label="q4 Mindlin plate SRI")
    ax[0].semilogy([1.0 / r.n for r in shell_scan_sri], [r.first_positive_ratio for r in shell_scan_sri], "s--",
                   color="#c44e52", label="shell4 facet, SRI plate")
    ax[0].semilogy([1.0 / r.n for r in shell_scan_mitc], [r.first_positive_ratio for r in shell_scan_mitc], "^--",
                   color="#55a868", label="shell4 facet, MITC plate")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("element size h")
    ax[0].set_ylabel("lambda_softest / lambda_max")
    ax[0].set_title("Softest elastic mode of the free mesh")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].plot([r.n for r in plate_scan], [r.spurious for r in plate_scan], "o-", color="#4c72b0",
               label="q4 Mindlin plate SRI")
    ax[1].plot([r.n for r in shell_scan_sri], [r.spurious for r in shell_scan_sri], "s--", color="#c44e52",
               label="shell4 facet, SRI plate")
    ax[1].plot([r.n for r in shell_scan_mitc], [r.spurious for r in shell_scan_mitc], "^--", color="#55a868",
               label="shell4 facet, MITC plate")
    ax[1].set_xlabel("mesh divisions per side n")
    ax[1].set_ylabel("spurious zero-energy modes")
    ax[1].set_title("Spurious modes surviving in the free mesh")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_hex20(geometry, integration, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    curved = [r for r in geometry if r["family"] == "curvature" and "arc" in r["element"]]
    control = [r for r in geometry if r["family"] == "curvature" and "straight" in r["element"]]
    angles = [float(r["element"].split()[3]) for r in curved]
    ax[0].semilogy(angles, [r["first_positive_ratio"] for r in curved], "o-", color="#4c72b0",
                   label="curved HEX20 (mid-edge on arc)")
    if control:
        ax[0].semilogy(angles, [control[0]["first_positive_ratio"]] * len(angles), "s--", color="#55a868",
                       label="straight control (30 deg chord width)")
    ax[0].set_xlabel("arc angle per element [deg]")
    ax[0].set_ylabel("lambda_softest / lambda_max")
    ax[0].set_title("Softest elastic mode of the free HEX20 (63 DOF) ")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    labels, spurious = [], []
    for row in integration:
        labels.append(row["element"].replace("HEX20 ", "").replace(" integration", ""))
        spurious.append(row["spurious"])
    positions = range(len(labels))
    ax[1].bar(list(positions), spurious, width=0.55, color="#c44e52")
    ax[1].set_xticks(list(positions))
    ax[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax[1].set_ylabel("spurious zero-energy modes")
    ax[1].set_title("Full (3x3x3) vs reduced (2x2x2) HEX20 integration")
    ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    evidence_dir = root / "docs" / "benchmarks" / "tutorials"
    asset_dir = root / "docs" / "assets"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    elements = audit_library(young=YOUNG, poisson=POISSON, thickness=THICKNESS)
    plate_scan = audit_plate_mesh_scan(young=YOUNG, poisson=POISSON, thickness=THICKNESS)
    plate_scan_mitc = audit_plate_mesh_scan(young=YOUNG, poisson=POISSON, thickness=THICKNESS, treatment="mitc")
    shell_scan_sri = audit_shell_mesh_scan(formulation="sri", young=YOUNG, poisson=POISSON, thickness=THICKNESS)
    shell_scan_mitc = audit_shell_mesh_scan(
        formulation="sri", young=YOUNG, poisson=POISSON, thickness=THICKNESS, plate_shear="mitc"
    )
    modal = modal_study()

    from tensorfem.element_rank_audit import (
        audit_hex20_curved_mesh,
        audit_hex20_geometry_sweep,
        audit_hex20_integration_variants,
        probe_hex20_jacobian_guard,
    )

    hex20_geometry = audit_hex20_geometry_sweep(young=YOUNG, poisson=POISSON)
    hex20_integration = audit_hex20_integration_variants(young=YOUNG, poisson=POISSON)
    hex20_mesh = audit_hex20_curved_mesh(young=YOUNG, poisson=POISSON)
    hex20_guard = probe_hex20_jacobian_guard(young=YOUNG, poisson=POISSON)

    payload = {
        "description": "Zero-energy (rank-deficiency) audit of the TensorFEM element library",
        "parameters": {
            "young": YOUNG, "poisson": POISSON, "thickness": THICKNESS, "density": DENSITY,
            "side": SIDE, "rtol": 1e-10,
        },
        "element_audit": [r.as_dict() for r in elements],
        "plate_mesh_scan_sri": [r.as_dict() for r in plate_scan],
        "plate_mesh_scan_mitc": [r.as_dict() for r in plate_scan_mitc],
        "shell_mesh_scan_plate_sri": [r.as_dict() for r in shell_scan_sri],
        "shell_mesh_scan_plate_mitc": [r.as_dict() for r in shell_scan_mitc],
        "modal_study": modal,
        "hex20_geometry_sweep": hex20_geometry,
        "hex20_integration_variants": hex20_integration,
        "hex20_curved_mesh": [r.as_dict() for r in hex20_mesh],
        "hex20_jacobian_guard": hex20_guard,
    }
    path = evidence_dir / "element-rank-audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    figure_elements(elements, asset_dir / "element_rank_element_nullspace.png")
    figure_free_free(modal, asset_dir / "element_rank_free_free_modes.png")
    figure_scaling(plate_scan, shell_scan_sri, shell_scan_mitc, asset_dir / "element_rank_mesh_scaling.png")
    figure_hex20(hex20_geometry, hex20_integration, asset_dir / "element_rank_hex20_geometry.png")

    print(summary_table(elements))
    print("--- plate free-mesh scan (SRI)")
    print(summary_table(plate_scan))
    print("--- plate free-mesh scan (MITC)")
    print(summary_table(plate_scan_mitc))
    print("--- shell free-mesh scan (SRI plate)")
    print(summary_table(shell_scan_sri))
    print("--- shell free-mesh scan (MITC plate)")
    print(summary_table(shell_scan_mitc))
    print("--- modal study")
    for row in modal["rows"]:
        print(row)
    print("--- HEX20 geometry sweep")
    for row in hex20_geometry:
        print(f"{row['element']:<42s} null={row['null_dim']} spurious={row['spurious']} "
              f"lambda1_ratio={row['first_positive_ratio']:.3e} " 
              f"jac_min={row['jacobian_min']:.4e} jac_max={row['jacobian_max']:.4e}")
    print("--- HEX20 integration variants")
    for row in hex20_integration:
        print(f"{row['element']:<42s} null={row['null_dim']} spurious={row['spurious']} "
              f"lambda1_ratio={row['first_positive_ratio']:.3e}")
    print("--- HEX20 curved arc mesh (free-free)")
    for row in hex20_mesh:
        print(f"{row.element} sectors={row.n} dofs={row.dofs} null={row.null_dim} spurious={row.spurious} "
              f"lambda1_ratio={row.first_positive_ratio:.3e}")
    print("--- HEX20 Jacobian guard")
    for row in hex20_guard:
        print(row)
    print(f"evidence written to {path}")


if __name__ == "__main__":
    main()