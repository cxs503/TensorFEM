#!/usr/bin/env python3
"""P19-N01 rung L3: free vibration of a simply supported beam and plate.

Physics
-------
1. **Simply supported beam** ``L = 4 m``, ``b = h = 50 mm``, steel
   ``E = 200 GPa``, ``rho = 7850 kg/m^3`` (``EI = 104.17 kN m^2``,
   ``rho A = 19.625 kg/m``).  Euler--Bernoulli theory:

       omega_n = (n pi / L)^2 sqrt(EI / (rho A)),   f_n = omega_n / 2 pi.

   First five: 7.15, 28.61, 64.38, 114.4, 178.8 Hz.

2. **Simply supported rectangular plate** ``a x b = 1.0 x 0.75 m``,
   ``t = 10 mm``, ``E = 210 GPa``, ``nu = 0.3``, ``rho = 7850 kg/m^3``.
   Navier eigenfrequencies

       omega_mn = pi^2 sqrt(D / (rho t)) [ (m/a)^2 + (n/b)^2 ].

   ``a/b = 4/3`` removes all degeneracies up to the fifth mode, so the FE and
   analytic orderings can be matched one-to-one without ambiguity:
   (1,1) 68.29, (2,1) 142.05, (1,2) 199.42, (3,1) 264.98, (2,2) 273.17 Hz.

Model
-----
Beam: 2-node cubic Hermite element with the consistent mass of
:func:`tensorfem.modal.uniform_beam_matrices` (no rotary inertia), pinned ends
(``w = 0``, rotations free), reduced symmetric eigenproblem
(:func:`tensorfem.modal.solve_modes`).
Plate: BFS ``C1`` bicubic element with the consistent mass of
:mod:`tensorfem.plate_bfs` (``rho t int N_i N_j``, 4x4 Gauss), hard simple
support, shift-invert Lanczos on the sparse generalised problem.
Meshes ``N = 10, 20, 40`` in each direction (plate ``n_y = 3N/4``).

Acceptance (campaign thresholds)
--------------------------------
* five beam and five plate frequencies: relative error ``<= 3 %``;
* h-refinement: error strictly decreasing, observed order ``>= 2`` (beam) and
  ``>= 1.5`` (plate).

Note on the directory: ``benchmarks/verified/fem_modal/result.json`` is the
legacy cantilever rung produced by ``benchmarks/structural_ladder.py`` and is
kept untouched.  This mission's modal ladder writes ``result_p19.json`` and
``gates_verified.json`` next to it.

Usage::

    PYTHONPATH=src python benchmarks/verified/fem_modal/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from _harness import Gate, gate_manifest, pyplot, utc_now, write_json  # noqa: E402

from tensorfem import plate_bfs as pb  # noqa: E402
from tensorfem.modal import simply_supported_beam_modes  # noqa: E402
from tensorfem.plate_bfs import ss_beam_angular_frequencies  # noqa: E402

MISSION = "5418"
NODE = "P19-N01"
ANCHOR = "fem_modal_p19"

LEVELS = (10, 20, 40)
MODES = 5
THRESHOLD = 3.0e-2  # 3 % on every modal data point

BEAM_LENGTH = 4.0
BEAM_WIDTH = 0.05
BEAM_HEIGHT = 0.05
BEAM_YOUNG = 200.0e9
BEAM_DENSITY = 7850.0
BEAM_EI = BEAM_YOUNG * BEAM_WIDTH * BEAM_HEIGHT**3 / 12.0
BEAM_RHO_A = BEAM_DENSITY * BEAM_WIDTH * BEAM_HEIGHT

PLATE_A = 1.0
PLATE_B = 0.75
PLATE_T = 0.01
PLATE_YOUNG = 210.0e9
PLATE_POISSON = 0.3
PLATE_DENSITY = 7850.0
PLATE_RIGIDITY = pb.plate_rigidity(PLATE_YOUNG, PLATE_POISSON, PLATE_T)


def beam_level(elements: int) -> dict:
    result = simply_supported_beam_modes(
        BEAM_LENGTH, elements, BEAM_EI, BEAM_RHO_A, modes=MODES
    )
    frequencies = result.frequencies_hz.numpy()
    exact = ss_beam_angular_frequencies(
        BEAM_LENGTH, BEAM_EI, BEAM_RHO_A, modes=MODES
    ) / (2.0 * np.pi)
    errors = np.abs(frequencies - exact) / exact
    return {
        "elements": elements,
        "h_m": BEAM_LENGTH / elements,
        "frequencies_hz": frequencies.tolist(),
        "exact_hz": exact.tolist(),
        "relative_errors": errors.tolist(),
        "max_relative_error": float(errors.max()),
        "modes": result.modes.numpy(),
    }


def plate_level(n_x: int) -> dict:
    n_y = max(1, int(round(n_x * PLATE_B / PLATE_A)))
    plate = pb.assemble_bfs_plate(
        n_x, n_y, PLATE_A, PLATE_B, PLATE_YOUNG, PLATE_POISSON, PLATE_T,
        density=PLATE_DENSITY,
    )
    fixed = pb.bfs_boundary_dofs(n_x, n_y, "ssss")
    omega, vectors = pb.solve_bfs_mode_shapes(plate, fixed, modes=MODES)
    exact_pairs = pb.ssss_plate_angular_frequencies(
        PLATE_A, PLATE_B, PLATE_YOUNG, PLATE_POISSON, PLATE_T, PLATE_DENSITY,
        m_max=5, n_max=5,
    )[:MODES]
    exact = np.array([pair[2] for pair in exact_pairs])
    errors = np.abs(omega - exact) / exact
    return {
        "n_x": n_x,
        "n_y": n_y,
        "h_m": PLATE_A / n_x,
        "n_dof": plate.n_dof,
        "frequencies_hz": (omega / (2.0 * np.pi)).tolist(),
        "exact_hz": (exact / (2.0 * np.pi)).tolist(),
        "mode_indices": [[pair[0], pair[1]] for pair in exact_pairs],
        "relative_errors": errors.tolist(),
        "max_relative_error": float(errors.max()),
        "vectors": vectors,
        "plate": plate,
    }


def observed_order(h: list[float], errors: list[float]) -> float:
    h = np.asarray(h, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if np.any(errors <= 0.0) or not np.all(np.isfinite(errors)):
        return float("nan")
    return float(np.polyfit(np.log(h), np.log(errors), 1)[0])


def monotone_decreasing(values: list[float]) -> bool:
    return all(b < a for a, b in zip(values, values[1:]))


def main() -> dict:
    beam = [beam_level(n) for n in LEVELS]
    plate = [plate_level(n) for n in LEVELS]

    beam_h = [level["h_m"] for level in beam]
    plate_h = [level["h_m"] for level in plate]
    beam_orders = [
        observed_order(beam_h, [level["relative_errors"][mode] for level in beam])
        for mode in range(MODES)
    ]
    plate_orders = [
        observed_order(plate_h, [level["relative_errors"][mode] for level in plate])
        for mode in range(MODES)
    ]
    beam_finest = beam[-1]
    plate_finest = plate[-1]

    gates = [
        Gate(
            id="L3.beam_frequencies",
            quantity="worst relative error of beam modes 1-5 (N=40)",
            reference="omega_n = (n pi/L)^2 sqrt(EI/(rho A))",
            value=beam_finest["max_relative_error"],
            threshold=THRESHOLD,
        ),
        Gate(
            id="L3.plate_frequencies",
            quantity="worst relative error of plate modes (1,1)..(2,2) (N=40)",
            reference="omega_mn = pi^2 sqrt(D/(rho t)) [(m/a)^2+(n/b)^2]",
            value=plate_finest["max_relative_error"],
            threshold=THRESHOLD,
        ),
        Gate(
            id="L3.beam_order",
            quantity="minimum observed order over the five beam mode errors",
            reference="log-log slope over N=10,20,40",
            value=float(np.nanmin(beam_orders)),
            threshold=2.0,
            operator=">=",
        ),
        Gate(
            id="L3.plate_order",
            quantity="minimum observed order over the plate mode errors",
            reference="log-log slope over N=10,20,40",
            value=float(np.nanmin(plate_orders)),
            threshold=1.5,
            operator=">=",
        ),
        Gate(
            id="L3.monotone_beam",
            quantity="beam frequency errors decrease monotonically with refinement",
            reference="strict decrease at every mode",
            value=1.0 if all(
                monotone_decreasing([level["relative_errors"][mode] for level in beam])
                for mode in range(MODES)
            ) else 0.0,
            threshold=1.0,
            operator=">=",
        ),
        Gate(
            id="L3.monotone_plate",
            quantity="plate frequency errors decrease monotonically with refinement",
            reference="strict decrease at every mode",
            value=1.0 if all(
                monotone_decreasing([level["relative_errors"][mode] for level in plate])
                for mode in range(MODES)
            ) else 0.0,
            threshold=1.0,
            operator=">=",
        ),
    ]

    data_points = [
        {
            "id": f"L3.beam_mode_{mode + 1}",
            "fe_hz": beam_finest["frequencies_hz"][mode],
            "reference": beam_finest["exact_hz"][mode],
            "reference_formula": "f_n = (n pi/L)^2 sqrt(EI/(rho A)) / (2 pi)",
            "relative_error": beam_finest["relative_errors"][mode],
            "threshold": THRESHOLD,
            "passed": beam_finest["relative_errors"][mode] <= THRESHOLD,
        }
        for mode in range(MODES)
    ] + [
        {
            "id": "L3.plate_mode_%d%d" % tuple(plate_finest["mode_indices"][mode]),
            "fe_hz": plate_finest["frequencies_hz"][mode],
            "reference": plate_finest["exact_hz"][mode],
            "reference_formula": "f_mn = pi^2 sqrt(D/(rho t))[(m/a)^2+(n/b)^2]/(2 pi)",
            "relative_error": plate_finest["relative_errors"][mode],
            "threshold": THRESHOLD,
            "passed": plate_finest["relative_errors"][mode] <= THRESHOLD,
        }
        for mode in range(MODES)
    ]

    def strip(level: dict) -> dict:
        return {
            key: value
            for key, value in level.items()
            if key not in ("modes", "vectors", "plate")
        }

    result = {
        "case": "Free vibration: simply supported beam modes 1-5 and rectangular SS plate modes (1,1)..(2,2)",
        "mission": MISSION,
        "node": NODE,
        "generated_by": "benchmarks/verified/fem_modal/run.py",
        "generated_at": utc_now(),
        "model": {
            "beam": "2-node cubic Hermite, consistent mass (tensorfem.modal), no rotary inertia, pinned ends",
            "plate": "BFS C1 bicubic, consistent mass rho t N_i N_j (4x4 Gauss), hard simple support",
            "solver": {
                "beam": "symmetric reduction with the Cholesky factor of M (tensorfem.modal.solve_modes)",
                "plate": "scipy shift-invert Lanczos (sigma=0) on the sparse generalised problem",
            },
        },
        "parameters": {
            "beam": {
                "length_m": BEAM_LENGTH,
                "width_m": BEAM_WIDTH,
                "height_m": BEAM_HEIGHT,
                "young_pa": BEAM_YOUNG,
                "density_kg_m3": BEAM_DENSITY,
                "EI_N_m2": BEAM_EI,
                "rho_A_kg_per_m": BEAM_RHO_A,
                "slenderness_L_over_h": BEAM_LENGTH / BEAM_HEIGHT,
            },
            "plate": {
                "a_m": PLATE_A,
                "b_m": PLATE_B,
                "thickness_m": PLATE_T,
                "young_pa": PLATE_YOUNG,
                "poisson": PLATE_POISSON,
                "density_kg_m3": PLATE_DENSITY,
                "rigidity_n_m": PLATE_RIGIDITY,
                "aspect_ratio": PLATE_A / PLATE_B,
                "a_over_t": PLATE_A / PLATE_T,
            },
            "modes": MODES,
            "levels": list(LEVELS),
        },
        "beam_levels": [strip(level) for level in beam],
        "plate_levels": [strip(level) for level in plate],
        "convergence": {
            "beam_h_m": beam_h,
            "plate_h_m": plate_h,
            "beam_orders_per_mode": beam_orders,
            "plate_orders_per_mode": plate_orders,
            "beam_order_min": float(np.nanmin(beam_orders)),
            "plate_order_min": float(np.nanmin(plate_orders)),
        },
        "data_points": data_points,
        "gates": [gate.to_dict() for gate in gates],
        "passed": all(gate.passed for gate in gates),
    }

    write_json(HERE / "result_p19.json", result)
    write_json(
        HERE / "gates_verified.json",
        gate_manifest(
            anchor=ANCHOR,
            title="Beam and plate modal analysis vs closed-form eigenfrequencies (L3 rung)",
            mission=MISSION,
            criteria={
                "frequency_relative_error_max": THRESHOLD,
                "beam_convergence_order_min": 2.0,
                "plate_convergence_order_min": 1.5,
                "levels": list(LEVELS),
                "modes": MODES,
            },
            gates=gates,
            artifacts={
                "result": "benchmarks/verified/fem_modal/result_p19.json",
                "figures": ["beam_modes.png", "plate_modes.png", "modal_convergence.png"],
                "legacy_result_untouched": "benchmarks/verified/fem_modal/result.json",
            },
            limitations=[
                "Euler-Bernoulli beam (no rotary inertia, no shear): the L/h = 80 "
                "beam is slender enough for the first five modes, but mode 5 of a "
                "stubby beam would need Timoshenko kinematics.",
                "Kirchhoff plate: a/t = 100 puts the Mindlin shear correction at "
                "~1e-4 relative, below the 3 % gate but not modelled here.",
                "Simply supported edges only -- clamped and free plate modes are "
                "not covered by this rung (the CCCC static case is in "
                "benchmarks/verified/fem_plate_static).",
                "FE consistent-mass models are stiffer in frequency than the exact "
                "spectrum only in the limit h -> 0; the observed errors approach "
                "zero from above, which the monotonicity gate records.",
            ],
        ),
    )
    make_figures(beam, plate)
    return result


def make_figures(beam: list[dict], plate: list[dict]) -> None:
    plt = pyplot()
    # beam mode shapes
    figure, axes = plt.subplots(1, 3, figsize=(9.4, 2.9))
    x = np.linspace(0.0, BEAM_LENGTH, 401)
    for mode, axis in enumerate(axes):
        axis.plot(
            x,
            np.sin((mode + 1) * np.pi * x / BEAM_LENGTH),
            "k-",
            lw=1.4,
            label="sin(n pi x/L)",
        )
        nodes = np.linspace(0.0, BEAM_LENGTH, beam[-1]["elements"] + 1)
        vector = beam[-1]["modes"][:, mode]
        axis.plot(nodes, vector[0::2] / np.abs(vector[0::2]).max(), "o", ms=2.5, fillstyle="none", label="FE N=40")
        axis.set_title(f"beam mode {mode + 1}: {beam[-1]['frequencies_hz'][mode]:.2f} Hz")
        axis.set_xlabel("x [m]")
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(HERE / "beam_modes.png", bbox_inches="tight")
    plt.close(figure)

    # plate mode shapes of the finest level
    finest = plate[-1]
    grid = np.linspace(0.0, PLATE_A, 41)
    ys = np.linspace(0.0, PLATE_B, 31)
    points = np.array([[gx, gy] for gx in grid for gy in ys])
    figure, axes = plt.subplots(1, 3, figsize=(9.4, 2.8))
    for axis, mode in zip(axes, range(min(3, MODES))):
        field = pb.bfs_evaluate(finest["plate"], finest["vectors"][:, mode], points)["w"]
        image = axis.contourf(grid, ys, field.reshape(len(grid), len(ys)).T, levels=16, cmap="RdBu_r")
        axis.set_title(
            "plate mode (%d,%d): %.1f Hz"
            % (*finest["mode_indices"][mode], finest["frequencies_hz"][mode]),
            fontsize=8,
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(False)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(HERE / "plate_modes.png", bbox_inches="tight")
    plt.close(figure)

    # convergence
    figure, axes = plt.subplots(1, 1, figsize=(4.8, 3.6))
    axes.loglog(
        [level["h_m"] for level in beam],
        [level["max_relative_error"] for level in beam],
        "o-",
        label="beam (worst of 5 modes)",
    )
    axes.loglog(
        [level["h_m"] for level in plate],
        [level["max_relative_error"] for level in plate],
        "s-",
        label="plate (worst of 5 modes)",
    )
    h = np.array([level["h_m"] for level in beam])
    axes.loglog(h, h**4 * beam[-1]["max_relative_error"] / h[-1] ** 4, "k--", lw=1.0, label="h^4")
    axes.set_xlabel("h [m]")
    axes.set_ylabel("max relative frequency error")
    axes.set_title("modal refinement")
    axes.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(HERE / "modal_convergence.png", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    payload = main()
    print(json.dumps({"passed": payload["passed"], "convergence": payload["convergence"]}, indent=2))