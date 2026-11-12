#!/usr/bin/env python3
"""Structural dynamics modal analysis verification: beam / plate / cylindrical shell."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.tensorfem.modal import cantilever_beam_modes
from src.tensorfem.plate_vibration import solve_plate_vibration
from src.tensorfem.cylindrical_shell_modal import (
    cylindrical_shell_modal_analysis,
    dmv_frequency_hz,
    NASA_TN_D3939_FREQS,
)


def run_cantilever_beam():
    L, b, h = 1.0, 0.05, 0.02
    E, rho = 210e9, 7800.0
    A, I = b * h, b * h**3 / 12.0
    EI, rho_a = E * I, rho * A
    beta_roots = [1.875104068711961, 4.694091132974174, 7.854757438237612]
    analytical = [b**2 * math.sqrt(EI / (rho_a * L**4)) / (2 * math.pi) for b in beta_roots]

    results = []
    for n in [10, 20, 40, 80]:
        fem = cantilever_beam_modes(L, n, EI, rho_a, modes=3)
        errors = [abs(f - a) / a * 100 for f, a in zip(fem.frequencies_hz.tolist(), analytical)]
        results.append({"n_elements": n, "frequencies_hz": fem.frequencies_hz.tolist(),
                        "relative_errors_pct": errors, "max_error_pct": max(errors)})

    max_err = results[-1]["max_error_pct"]
    return {
        "description": "Cantilever beam, modes 1-3, Euler-Bernoulli + consistent mass",
        "parameters": {"L": L, "b": b, "h": h, "E": E, "rho": rho},
        "analytical_frequencies_hz": analytical,
        "mesh_convergence": results,
        "max_error_pct_finest": max_err,
        "verdict": "PASS" if max_err < 3.0 else "FAIL",
    }


def run_simply_supported_plate():
    a, b, t = 1.0, 0.8, 0.01
    E, rho, nu = 70e9, 2700.0, 0.3
    D = E * t**3 / (12.0 * (1 - nu**2))

    modes = []
    for m in range(1, 5):
        for n in range(1, 5):
            omega = math.pi**2 * math.sqrt(D / (rho * t)) * ((m/a)**2 + (n/b)**2)
            modes.append((omega / (2 * math.pi), m, n))
    modes.sort()
    analytical = [f for f, _, _ in modes[:3]]
    mode_labels = [(m, n) for _, m, n in modes[:3]]

    results = []
    for nx, ny in [(6, 5), (8, 6), (10, 8)]:
        plate = solve_plate_vibration(nx, ny, length=a, width=b,
                                      young=E, poisson=nu, thickness=t, density=rho,
                                      boundary="simple", modes=4)
        fem = plate.frequencies_hz.tolist()[:3]
        errors = [abs(f - a) / a * 100 for f, a in zip(fem, analytical)]
        results.append({"n_elements_x": nx, "n_elements_y": ny,
                        "n_elements_total": nx * ny,
                        "frequencies_hz": fem, "relative_errors_pct": errors,
                        "max_error_pct": max(errors)})

    max_err = results[-1]["max_error_pct"]
    return {
        "description": "Simply supported plate, modes 1-3, Mindlin Q4 + consistent mass",
        "parameters": {"a": a, "b": b, "t": t, "E": E, "rho": rho, "nu": nu},
        "mode_indices": mode_labels,
        "analytical_frequencies_hz": analytical,
        "mesh_convergence": results,
        "max_error_pct_finest": max_err,
        "verdict": "PASS" if max_err < 10.0 else "FAIL",
        "note": "Mindlin Q4 converges O(h^2) for thin plates; 10x8 mesh gives ~7% error",
    }


def run_cylindrical_shell():
    R, L, t = 0.5, 1.0, 0.005
    E, rho, nu = 210e9, 7800.0, 0.3

    geometry = dict(radius=R, length=L, thickness=t, young=E, poisson=nu, density=rho)

    # Euler-Bernoulli beam limit: the n = 1 mode of a cylinder is beam bending,
    # omega = (m pi/L)^2 sqrt(E I/(rho A)) with I/A = R^2/2.  Donnell is only
    # accurate for n >= 2, so this is the bracket for the n = 1 row.
    beam_hz = (math.pi / L) ** 2 * math.sqrt(E * R ** 2 / 2.0 / rho) / (2 * math.pi)

    # Donnell-Mushtari-Vlasov oracle over the (m, n) grid.  n = 0 is the
    # axisymmetric breathing mode and is admissible since m16863; before that
    # repair the validator rejected it and the n = 1 branch dropped the membrane
    # term (17.33 Hz against a 2881.62 Hz beam limit, see
    # docs/fem-shell-modal-anchor-audit.md).
    oracle = []
    for m in (1, 2, 3):
        for n in range(0, 9):
            oracle.append((dmv_frequency_hz(n_axial=m, n_circ=n, **geometry), m, n))
    oracle.sort()
    oracle_min_hz, oracle_min_mode = oracle[0][0], [oracle[0][1], oracle[0][2]]

    def classify(mesh_axial, mesh_circ, result):
        """Split the FE spectrum into physical modes and rotation-only mechanisms.

        A shell mode carries transverse displacement.  The pre-existing shell4
        spectrum of this cylinder contains DOFs that store only rotation: their
        radial participation is ~1e-25 and their frequencies are mesh-dependent
        (75.85 Hz at 4x8, 39.17 Hz at 8x8, 0.001 Hz at 8x16), so a "lowest
        frequency" quote that does not discard them is not a shell mode.
        """
        modes = result.modes.numpy()
        freqs = result.frequencies_hz.numpy()
        rows = []
        for index in range(freqs.size):
            phi = modes[:, index]
            total = float((phi ** 2).sum())
            radial = float((phi.reshape(mesh_axial + 1, mesh_circ, 6)[:, :, 2] ** 2).sum())
            rotation = float((phi.reshape(-1, 6)[:, 3:6] ** 2).sum())
            rows.append({
                "frequency_hz": float(freqs[index]),
                "radial_participation": radial / total if total else 0.0,
                "rotation_participation": rotation / total if total else 0.0,
                "physical": (radial / total if total else 0.0) > 1e-6,
            })
        return rows

    ladder = []
    for n_axial_mesh, n_circ_mesh in ((4, 8), (8, 8), (8, 16)):
        fem = cylindrical_shell_modal_analysis(
            radius=R, length=L, thickness=t, young=E, poisson=nu, density=rho,
            n_axial_mesh=n_axial_mesh, n_circ_mesh=n_circ_mesh, n_modes=20,
        )
        rows = classify(n_axial_mesh, n_circ_mesh, fem)
        physical = [row["frequency_hz"] for row in rows if row["physical"]]
        mechanisms = [row["frequency_hz"] for row in rows if not row["physical"]]
        ladder.append({
            "mesh": f"{n_axial_mesh}x{n_circ_mesh}",
            "n_dof": fem.n_dof,
            "n_free_dof": fem.n_free_dof,
            "n_modes_requested": 20,
            "n_mechanism_modes": len(mechanisms),
            "lowest_mechanism_hz": mechanisms[0] if mechanisms else None,
            "lowest_physical_hz": physical[0] if physical else None,
            "first_physical_hz": physical[:4],
            "first_six_hz": [row["frequency_hz"] for row in rows[:6]],
            "mechanism_free_first_ten": not any(
                not row["physical"] for row in rows[:10]
            ),
            "n_mechanism_first_ten": sum(1 for row in rows[:10] if not row["physical"]),
        })

    # The anchor's own mesh (4x8) is the one whose spectrum a reader sees first;
    # coarser-to-finer only adds mechanisms, it never adds physical shell modes,
    # so the honest comparison uses the first mesh that has one at all.
    graded = next((row for row in ladder if row["lowest_physical_hz"] is not None), None)
    matched_error_pct = (
        abs(graded["lowest_physical_hz"] - oracle_min_hz) / oracle_min_hz * 100.0
        if graded is not None else 999.0
    )
    graded_mesh = graded["mesh"] if graded is not None else "none"

    return {
        "description": "Cylindrical shell, Shell4 element + consistent mass",
        "parameters": {"R": R, "L": L, "t": t, "E": E, "rho": rho, "nu": nu, "R_over_t": R / t},
        "nasa_experimental_freqs_hz": NASA_TN_D3939_FREQS,
        "nasa_note": "NASA TN D-3939: clamped BCs; this uses simply supported (diaphragm)",
        "oracle": {
            "theory": "Donnell-Mushtari-Vlasov, simply supported, membrane term retained for all n",
            "grid": "m = 1..3, n = 0..8",
            "lowest_hz": oracle_min_hz,
            "lowest_mode_mn": oracle_min_mode,
            "first_six": [[f, m, n] for f, m, n in oracle[:6]],
            "beam_limit_n1_hz": beam_hz,
        },
        "fem_ladder": ladder,
        "lowest_physical_vs_oracle_pct": matched_error_pct,
        "lowest_physical_mesh": graded_mesh,
        "verdict": "FAIL",
        "note": (
            "The FE spectrum of this cylinder is dominated by rotation-only DOFs: at "
            "the anchor's original 4x8 mesh the three lowest 'frequencies' (75.85, "
            "75.85, 79.84 Hz) have radial participation ~1e-25 and 100% rotation "
            "participation, i.e. they are mechanisms, not shell modes; their frequency "
            "is mesh dependent (39.17 Hz at 8x8, 0.001 Hz at 8x16). The first physical "
            "mode is 100.35 Hz (4x8) / 98.87 Hz (8x8) against a Donnell minimum of "
            f"{oracle_min_hz:.2f} Hz, so the FE evidence does not converge to the "
            "oracle. The anchor verdict is FAIL: the shell element and/or its "
            "simply-supported constraint set needs repair before this anchor can "
            "support a PASS."
        ),
    }


ANCHOR_DIR = Path(__file__).resolve().parent

# Figures named by ``gates_verified.json`` (``artifacts``) plus the two per-anchor
# convergence panels the README tables quote.  Every one of them is a pure
# function of the numbers in ``result.json``, which is what makes
# ``--from-dump`` able to rebuild them without re-running an eigensolve.
FIGURE_CAPTIONS = {
    "beam_convergence.png": "Cantilever beam: FE modes 1-3 over the mesh ladder vs Euler-Bernoulli",
    "plate_convergence.png": "Simply supported plate: FE modes over the mesh ladder vs Kirchhoff-Navier",
    "shell_fem_vs_dmv.png": "Cylindrical shell: lowest FE physical modes vs the Donnell-Mushtari-Vlasov spectrum",
    "shell_modal_audit.png": "Shell modal anchor: Donnell oracle vs FE spectrum, rotation-only mechanisms marked",
}
CERTIFICATE_FIGURE = "shell_modal_audit.png"


def render_figures(beam, plate, shell, out_dir):
    """Render the anchor figures from recorded numbers only (no solver call).

    ``--from-dump`` and the live run share this function, so a rebuilt figure and
    a freshly produced one cannot diverge: both read the same ``result.json``
    fields.  The certificate names ``shell_modal_audit.png``; the other three are
    the panels the README tables summarise.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    written = []

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(3):
        n = [r["n_elements"] for r in beam["mesh_convergence"]]
        f = [r["frequencies_hz"][i] for r in beam["mesh_convergence"]]
        ax.plot(n, f, 'o-', color=colors[i], label=f'FE mode {i+1}')
        ax.axhline(y=beam["analytical_frequencies_hz"][i], color=colors[i], ls='--', alpha=0.5)
    ax.set_xlabel('Elements'); ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Cantilever Beam (dashed = Euler-Bernoulli)'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); ax.figure.savefig(out_dir / "beam_convergence.png", dpi=150)
    plt.close(fig); written.append(out_dir / "beam_convergence.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(3):
        n = [r["n_elements_total"] for r in plate["mesh_convergence"]]
        f = [r["frequencies_hz"][i] for r in plate["mesh_convergence"]]
        ax.plot(n, f, 'o-', color=colors[i], label=f'FE mode {plate["mode_indices"][i]}')
        ax.axhline(y=plate["analytical_frequencies_hz"][i], color=colors[i], ls='--', alpha=0.5)
    ax.set_xlabel('Elements'); ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Simply Supported Plate (dashed = Kirchhoff-Navier)'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "plate_convergence.png", dpi=150)
    plt.close(fig); written.append(out_dir / "plate_convergence.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    fem = shell["fem_ladder"][0]["first_physical_hz"]
    dmv = [row[0] for row in shell["oracle"]["first_six"]][:len(fem)]
    x = range(1, len(fem) + 1)
    ax.plot(x, dmv, 's--', color='red', label='DMV (SS)', alpha=0.7)
    ax.plot(x, fem, 'o-', color='blue', label=f'FEM ({shell["fem_ladder"][0]["mesh"]})')
    ax.set_xlabel('Mode index'); ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Cylindrical Shell: FEM vs DMV'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "shell_fem_vs_dmv.png", dpi=150)
    plt.close(fig); written.append(out_dir / "shell_fem_vs_dmv.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    oracle_n = [row[2] for row in shell["oracle"]["first_six"]]
    oracle_f = [row[0] for row in shell["oracle"]["first_six"]]
    ax.plot(oracle_n, oracle_f, 's--', color='black', label='Donnell oracle (m=1)')
    ax.axhline(shell["oracle"]["beam_limit_n1_hz"], color='grey', ls=':',
               label='Euler-Bernoulli n=1 bracket')
    for entry in shell["fem_ladder"]:
        phys = list(entry["first_physical_hz"])
        slot = -1 - 0.4 * shell["fem_ladder"].index(entry)
        ax.scatter([slot] * len(phys), phys, marker='o', label=f'FE physical {entry["mesh"]}')
        if entry["lowest_mechanism_hz"] is not None:
            ax.scatter([slot], [entry["lowest_mechanism_hz"]], marker='x', color='red')
    ax.set_yscale('log')
    ax.set_xlabel('circumferential wavenumber n (oracle) / mesh slot (FE)')
    ax.set_ylabel('frequency (Hz)')
    ax.set_title('Shell modal anchor: Donnell oracle vs FE (x = rotation-only mechanism)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "shell_modal_audit.png", dpi=150)
    plt.close(fig); written.append(out_dir / "shell_modal_audit.png")

    return written


def write_figure_provenance(result_path, out_dir, figures):
    """Stamp every figure with the sha256 of the ``result.json`` it was built from.

    Without this stamp a rebuilt figure is indistinguishable from a figure some
    other (possibly stale) run produced; with it, ``tests/test_shell_modal_anchor_certificate.py``
    can prove the PNG on disk is the render of the JSON on disk.
    """
    result_path, out_dir = Path(result_path), Path(out_dir)
    payload = {
        "schema": "tensorfem.figure-provenance/1",
        "anchor": "fem_shell_modal",
        "source": result_path.name,
        "source_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "renderer": "benchmarks/verified/fem_shell_modal/run_verification.py --from-dump",
        "solver_executed": False,
        "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in figures},
    }
    path = out_dir / "figure_provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def rebuild_from_dump(result_path, out_dir):
    """``--from-dump``: rebuild the figures of the committed result.json, zero solve."""
    result_path, out_dir = Path(result_path), Path(out_dir)
    payload = json.loads(result_path.read_text())
    anchors = payload["anchors"]
    figures = render_figures(anchors["beam"], anchors["plate"], anchors["shell"], out_dir)
    provenance = write_figure_provenance(result_path, out_dir, figures)
    for path in figures:
        print(f"wrote {path}")
    print(f"wrote {provenance} (source sha256 {json.loads(provenance.read_text())['source_sha256'][:16]}...)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dump", action="store_true",
                        help="rebuild the figures from the committed result.json; no solver is run")
    parser.add_argument("--result", default=str(ANCHOR_DIR / "result.json"))
    parser.add_argument("--out-dir", default=str(ANCHOR_DIR))
    args = parser.parse_args(argv)
    if args.from_dump:
        return rebuild_from_dump(args.result, args.out_dir)

    print("TensorFEM Modal Analysis Verification (IN28-E1)")
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print(f"Python: {sys.version.split()[0]}, PyTorch: {torch.__version__}\n")

    beam = run_cantilever_beam()
    plate = run_simply_supported_plate()
    shell = run_cylindrical_shell()


    gates = [
        {
            "id": "G1.beam", "quantity": "cantilever beam modes 1-3, max relative error [%]",
            "value": beam["max_error_pct_finest"], "operator": "<=", "threshold": 3.0,
            "passed": beam["max_error_pct_finest"] <= 3.0,
            "reference": "beta_n roots of cos(beta)cosh(beta) = -1, Euler-Bernoulli",
        },
        {
            "id": "G2.plate", "quantity": "SS plate modes 1-3, max relative error [%]",
            "value": plate["max_error_pct_finest"], "operator": "<=", "threshold": 10.0,
            "passed": plate["max_error_pct_finest"] <= 10.0,
            "reference": "Kirchhoff Navier series, thin plate",
        },
        {
            "id": "G3.shell_lowest_physical_vs_donnell",
            "quantity": "lowest FE shell mode (radial participation > 1e-6) vs Donnell minimum [%]"
                        " at mesh " + str(shell["lowest_physical_mesh"]),
            "value": shell["lowest_physical_vs_oracle_pct"], "operator": "<=", "threshold": 15.0,
            "passed": (
                shell["lowest_physical_vs_oracle_pct"] == shell["lowest_physical_vs_oracle_pct"]
                and shell["lowest_physical_vs_oracle_pct"] <= 15.0
            ),
            "reference": (
                "Donnell-Mushtari-Vlasov over m=1..3, n=0..8: "
                f"{shell['oracle']['lowest_hz']:.2f} Hz at (m,n)="
                f"({shell['oracle']['lowest_mode_mn'][0]},{shell['oracle']['lowest_mode_mn'][1]})"
            ),
        },
        {
            "id": "G4.shell_mechanism_free_low_spectrum",
            "quantity": "rotation-only modes inside the first ten FE modes (must be zero)",
            "value": float(shell["fem_ladder"][0]["n_mechanism_first_ten"]),
            "operator": "==", "threshold": 0.0,
            "passed": shell["fem_ladder"][0]["mechanism_free_first_ten"],
            "reference": "a shell mode must carry transverse displacement; radial participation > 1e-6",
        },
    ]
    all_pass = all(gate["passed"] for gate in gates)

    result = {
        "benchmark": "IN28-E1 Structural Dynamics Modal Analysis",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "torch": torch.__version__, "machine": os.uname().machine},
        "anchors": {"beam": beam, "plate": plate, "shell": shell},
        "overall_verdict": "PASS" if all_pass else "FAIL",
        "gates": gates,
    }

    out = Path(__file__).parent
    envelope = {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": "fem_shell_modal",
        "passed": all_pass,
        "gates": gates,
        "failed_gates": [gate["id"] for gate in gates if not gate["passed"]],
        "artifacts": {"result": "result.json", "figure": "shell_modal_audit.png"},
        "limitations": [
            "the cylinder FE spectrum contains rotation-only mechanisms; the shell "
            "element (or its SS constraint set) is not yet certifiable",
            "Donnell theory is a thin-shell approximation: n = 1 is bracketed by the "
            "Euler-Bernoulli beam limit, not gated",
            "G2 plate error is 6.84% on a 10x8 mesh: discretisation, not physics",
        ],
    }
    with open(out / "result.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)
    with open(out / "gates_verified.json", 'w') as f:
        json.dump(envelope, f, indent=2)

    figures = render_figures(beam, plate, shell, out)
    provenance = write_figure_provenance(out / "result.json", out, figures)
    print(f"\nfigures: {', '.join(p.name for p in figures)}")
    print(f"provenance: {provenance.name} (result.json sha256 {json.loads(provenance.read_text())['source_sha256'][:16]}...)")

    print(f"\nOverall: {result['overall_verdict']}")
    for g in result["gates"]:
        print(f"  {g['id']}: {'PASS' if g['passed'] else 'FAIL'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
