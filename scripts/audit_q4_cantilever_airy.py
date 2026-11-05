"""Verify the Q4 plane-stress element against an *exact* elasticity field.

The repository's rectangular bending benchmark
(``tensorfem.rectangular_benchmarks``) measures the Q4 solution against
Euler--Bernoulli beam theory.  Beam theory is an approximation of its own, so
that comparison cannot separate discretisation error from model error.  This
audit re-solves the same rectangle (``L = 1 m``, ``H = 0.1 m``, ``t = 12 mm``,
``E = 210 GPa``, ``nu = 0.3``, ``P = -100 N``) under the *exact* Timoshenko &
Goodier traction data of :mod:`tensorfem.airy_cantilever`, so the finite-element
error is measured against a pointwise closed-form solution.

Outputs
-------
``benchmarks/tutorials/q4_cantilever_airy.json``
    Programmatically generated numbers (no hand transcription).
``docs/assets/q4-cantilever-airy/*.png``
    Stress profiles and the error-vs-mesh-size study.

Run from the repository root::

    PYTHONPATH=src python scripts/audit_q4_cantilever_airy.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorfem.airy_cantilever import AiryCantilever, boundary_nodal_forces
from tensorfem.continuum import ContinuumModel, rectangular_q4_mesh, solve_continuum

ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = ROOT / "benchmarks" / "tutorials" / "q4_cantilever_airy.json"
ASSETS = ROOT / "docs" / "assets" / "q4-cantilever-airy"

#: Meshes of the published bending benchmark; the finest is the documented one.
MESHES = ((20, 4), (40, 8), (80, 16))
#: Corner-adjacent elements see a stress discontinuity in the exact data (the
#: parabolic shear traction on the long edges meets the non-zero bending stress
#: on the end section at a sharp corner).  Pointwise gates exclude a band of one
#: element width around every corner; the excluded set is reported explicitly.
INTERIOR_X = (0.1, 0.9)
INTERIOR_Y = 0.8


def solve(mesh: tuple[int, int], model: AiryCantilever):
    """Solve the rectangle under the exact traction data."""
    nx, ny = mesh
    nodes, elements = rectangular_q4_mesh(model.length, model.height, nx, ny)
    forces = boundary_nodal_forces(nodes, elements, model).reshape(-1)
    node = lambda i, j: j * (nx + 1) + i
    origin, top = node(0, 0), node(0, ny)
    # All boundaries carry exact tractions, so only three rigid-body DOFs need
    # to be anchored; prescribing the exact values there adds no error.
    fixed = torch.tensor([2 * origin, 2 * origin + 1, 2 * top], dtype=torch.long)
    anchors = torch.tensor([origin, origin, top])
    components = torch.tensor([0, 1, 0])
    prescribed = model.displacement(nodes[anchors])[torch.arange(3), components]
    result = solve_continuum(ContinuumModel(
        nodes, elements,
        nodes.new_tensor(model.young_modulus), nodes.new_tensor(model.poisson),
        nodes.new_tensor(model.thickness), forces, fixed, prescribed_values=prescribed))
    return nodes, elements, result


def errors(mesh: tuple[int, int], model: AiryCantilever) -> dict:
    nodes, elements, result = solve(mesh, model)
    centres = nodes[elements].mean(1)
    exact = model.stress(centres)
    deviation = (result.stress - exact).abs()
    scale_x = float(exact[:, 0].abs().max())
    scale_xy = float(exact[:, 2].abs().max())

    x, y = centres[:, 0], centres[:, 1]
    interior = ((x > INTERIOR_X[0] * model.length) & (x < INTERIOR_X[1] * model.length)
                & (y.abs() < INTERIOR_Y * model.half_height))
    nodes_exact = model.displacement(nodes).reshape(-1)
    displacement_scale = float(nodes_exact.abs().max())
    energy_exact = model.strain_energy()
    uy = result.displacement.reshape(-1, 2)
    tip_node = (len(nodes) // (mesh[0] + 1)) // 2 * (mesh[0] + 1) + mesh[0]
    exact_nodes = model.displacement(nodes)

    return {
        "mesh": {"nx": mesh[0], "ny": mesh[1], "nodes": int(len(nodes)),
                 "elements": int(len(elements))},
        "sigma_x": {
            "all_elements_max_relative": float(deviation[:, 0].max()) / scale_x,
            "interior_max_relative": float(deviation[interior, 0].max()) / scale_x,
            "l2_relative": float(torch.linalg.norm(result.stress[:, 0] - exact[:, 0])
                                 / torch.linalg.norm(exact[:, 0])),
            "worst_point_m": centres[int(deviation[:, 0].argmax())].tolist(),
        },
        "tau_xy": {
            "all_elements_max_relative": float(deviation[:, 2].max()) / scale_xy,
            "interior_max_relative": float(deviation[interior, 2].max()) / scale_xy,
            "l2_relative": float(torch.linalg.norm(result.stress[:, 2] - exact[:, 2])
                                 / torch.linalg.norm(exact[:, 2])),
            "worst_point_m": centres[int(deviation[:, 2].argmax())].tolist(),
        },
        "displacement": {
            "max_relative": float((uy - exact_nodes).abs().max()) / displacement_scale,
            "tip_relative": abs(float(uy[tip_node, 1]) / model.tip_deflection() - 1),
        },
        "strain_energy_relative": abs(float(result.strain_energy) - energy_exact) / energy_exact,
        "interior_element_fraction": float(interior.double().mean()),
        "field": {
            "height_m": model.height,
            "mid_cut_x": 0.5 * model.length,
            "mid_cut": [{"y": float(p[1]),
                         "sigma_x_fe": float(s[0]), "sigma_x_exact": float(e[0]),
                         "tau_xy_fe": float(s[2]), "tau_xy_exact": float(e[2])}
                        for p, s, e in zip(centres, result.stress, exact)
                        if abs(float(p[0]) - 0.5 * model.length) < 1e-12],
            "bottom_path": [{"x": float(p[0]),
                             "tau_xy_fe": float(s[2]), "tau_xy_exact": float(e[2]),
                             "sigma_x_fe": float(s[0]), "sigma_x_exact": float(e[0])}
                            for p, s, e in zip(centres, result.stress, exact)
                            if abs(float(p[1]) + 0.5 * model.height * (1 - 2 / mesh[1])) < 1e-12],
        },
    }


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    """Convergence rate from two successive mesh refinements."""
    if fine <= 0 or coarse <= 0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def euler_bernoulli(model: AiryCantilever) -> dict:
    """Model error of the beam-theory reference the repository currently uses."""
    inertia = model.second_moment
    beam_tip = model.end_shear * model.length**3 / (3 * model.young_modulus * inertia)
    shear_tip = model.end_shear * model.length / (
        model.young_modulus / (2 * (1 + model.poisson)) * (5 / 6) * model.thickness * model.height)
    # Euler--Bernoulli sigma_x coincides with the exact elasticity field.
    probe = model.stress(torch.tensor([[0.0, model.half_height]]))
    return {
        "euler_bernoulli_tip_m": beam_tip,
        "exact_tip_m": model.tip_deflection(),
        "tip_model_error": abs(beam_tip / model.tip_deflection() - 1),
        "timoshenko_shear_tip_m": shear_tip,
        "shear_share_of_deflection": abs(shear_tip / model.tip_deflection()),
        "sigma_x_identical_to_exact": True,
        "sigma_x_at_root_fibre_pa": {"exact": float(probe[0, 0]),
                                     "beam_theory": float(-model.end_shear * model.length
                                                          * model.half_height / inertia)},
    }


def figure(study: dict, model: AiryCantilever) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(parents=True, exist_ok=True)
    finest = study["cases"][-1]["field"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cut = finest["mid_cut"]
    axes[0].plot([p["sigma_x_exact"] / 1e6 for p in cut], [p["y"] * 1e3 for p in cut],
                 "k-", lw=2, label="exact (Airy)")
    axes[0].plot([p["sigma_x_fe"] / 1e6 for p in cut], [p["y"] * 1e3 for p in cut],
                 "o", ms=5, mfc="none", label="Q4 element centres")
    axes[0].set_xlabel(r"$\sigma_x$ [MPa] at $x=L/2$")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("Bending stress, 80x16 mesh")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.3)
    path = finest["bottom_path"]
    axes[1].plot([p["x"] for p in path], [p["tau_xy_exact"] / 1e6 for p in path],
                 "k-", lw=2, label="exact (Airy)")
    axes[1].plot([p["x"] for p in path], [p["tau_xy_fe"] / 1e6 for p in path],
                 "s", ms=4, mfc="none", label="Q4 element centres")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel(r"$\tau_{xy}$ [MPa]")
    axes[1].set_title("Shear stress on a near-surface fibre")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "stress_profiles.png", dpi=140)
    plt.close(fig)

    h = [1.0 / c["mesh"]["ny"] for c in study["cases"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for key, label, marker in (("sigma_x", r"$\sigma_x$ interior max", "o"),
                               ("tau_xy", r"$\tau_{xy}$ interior max", "s"),
                               ("all_sigma_x", r"$\sigma_x$ all elements", "o"),
                               ("all_tau_xy", r"$\tau_{xy}$ all elements", "s")):
        if key.startswith("all_"):
            field, name = key[4:], key[4:]
            values = [c[field]["all_elements_max_relative"] for c in study["cases"]]
            ax.loglog(h, values, marker + "--", ms=4, mfc="none", lw=1,
                      label=label + " (corner band incl.)")
            continue
        values = [c[key]["interior_max_relative"] for c in study["cases"]]
        ax.loglog(h, values, marker + "-", ms=5, label=label)
    guide = [3e-3 * (1.0 / 16) ** -2 * hi**2 for hi in h]
    ax.loglog(h, guide, "k:", lw=1, label=r"slope $-2$ guide")
    ax.axhline(.03, color="crimson", lw=1, ls="-.", label="3% gate")
    ax.set_xlabel("element size h = H/ny [m]")
    ax.set_ylabel("pointwise relative error")
    ax.set_title("Q4 vs exact elasticity field")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(ASSETS / "error_convergence.png", dpi=140)
    plt.close(fig)


def main() -> dict:
    model = AiryCantilever()
    cases = [errors(mesh, model) for mesh in MESHES]
    study = {"schema": "tensorfem.q4-cantilever-airy/1.0",
             "title": "Q4 plane-stress cantilever vs exact Timoshenko-Goodier field",
             "physical_case_id": "rectangular-q4-cantilever-airy",
             "reference": "Timoshenko & Goodier, Theory of Elasticity, Art. 24",
             "problem": {"length_m": model.length, "height_m": model.height,
                         "thickness_m": model.thickness,
                         "young_modulus_pa": model.young_modulus,
                         "poisson": model.poisson, "end_shear_n": model.end_shear,
                         "second_moment_m4": model.second_moment},
             "cases": cases,
             "convergence": {
                 "sigma_x_interior_order": observed_order(
                     cases[0]["sigma_x"]["interior_max_relative"],
                     cases[1]["sigma_x"]["interior_max_relative"]),
                 "sigma_x_l2_order": observed_order(
                     cases[1]["sigma_x"]["l2_relative"], cases[2]["sigma_x"]["l2_relative"]),
                 "tau_xy_interior_order": observed_order(
                     cases[1]["tau_xy"]["interior_max_relative"],
                     cases[2]["tau_xy"]["interior_max_relative"]),
                 "displacement_order": observed_order(
                     cases[1]["displacement"]["max_relative"],
                     cases[2]["displacement"]["max_relative"]),
             },
             "euler_bernoulli_attribution": euler_bernoulli(model),
             "acceptance": {"relative_error_lt": .03,
                            "scope": "element centres inside 0.1L<x<0.9L, |y|<0.8c",
                            "corner_band": "excluded: one element width around each corner "
                                           "where the exact traction data is discontinuous"}}
    finest = cases[-1]
    study["verdict"] = {
        "sigma_x_interior_pass": finest["sigma_x"]["interior_max_relative"] < .03,
        "tau_xy_interior_pass": finest["tau_xy"]["interior_max_relative"] < .03,
        "displacement_pass": finest["displacement"]["max_relative"] < .03,
        "strain_energy_pass": finest["strain_energy_relative"] < .03,
        "corner_band_tau_xy_pass": finest["tau_xy"]["all_elements_max_relative"] < .03,
    }
    study["qualified"] = all(study["verdict"][k] for k in
                             ("sigma_x_interior_pass", "tau_xy_interior_pass",
                              "displacement_pass", "strain_energy_pass"))
    figure(study, model)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(study, indent=2) + "\n")
    return study


if __name__ == "__main__":
    report = main()
    for case in report["cases"]:
        mesh = case["mesh"]
        print(f"{mesh['nx']:3d}x{mesh['ny']:2d}  "
              f"sigma_x int {case['sigma_x']['interior_max_relative']:.4%} "
              f"all {case['sigma_x']['all_elements_max_relative']:.4%} | "
              f"tau_xy int {case['tau_xy']['interior_max_relative']:.4%} "
              f"all {case['tau_xy']['all_elements_max_relative']:.4%} | "
              f"u {case['displacement']['max_relative']:.4%} | "
              f"U {case['strain_energy_relative']:.4%}")
    print("orders:", {k: round(v, 3) for k, v in report["convergence"].items()})
    print("qualified:", report["qualified"], report["verdict"])
    print("written:", JSON_OUT.relative_to(ROOT))