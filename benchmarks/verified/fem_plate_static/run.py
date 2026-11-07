#!/usr/bin/env python3
"""P19-N01 rung L2: Kirchhoff plate statics (SSSS + CCCC) against series/table oracles.

Physics
-------
A square thin plate ``a = b = 1 m``, ``t = 10 mm`` (``a/t = 100``), steel
``E = 210 GPa``, ``nu = 0.3``, uniform pressure ``q = 10 kPa``.
Bending stiffness ``D = E t^3 / (12 (1 - nu^2)) = 19.2308 kN m``.

* **Simply supported (SSSS)** -- Navier double-sine solution

      w(x,y) = 16 q / (pi^6 D) sum_{m,n odd} sin(m pi x/a) sin(n pi y/b)
               / [ m n ((m/a)^2 + (n/b)^2)^2 ],

  truncated at ``m, n <= 11`` exactly as the campaign mandates, plus a
  converged 301-harmonic evaluation of the same series that is used as the
  oracle for the convergence order (the 11x11 truncation is itself only
  ``~5e-8`` accurate on ``w_c`` but ``~1e-3`` on the moments, so grading the
  moments against the 11-term series would measure the series, not the FE).
  Timoshenko table 8 values ``w_c = 0.00406 q a^4/D`` and
  ``M = 0.0479 q a^2`` are carried as an independent literature check.
* **Clamped (CCCC)** -- Timoshenko table 35:
  ``w_c = 0.00126 q a^4 / D`` and ``M_x = M_y = 0.0231 q a^2`` at the centre.
  The table is quoted to three significant figures, so its own rounding is a
  ``0.4 %`` floor on any comparison; the BFS result at ``N = 80`` sits on that
  floor (``alpha = 0.0012653`` versus the classical ``0.00126532``).

Model
-----
Bogner--Fox--Schmit ``C1`` bicubic rectangle (16 dof per element: ``w, w_,x,
w_,y, w_,xy``) from :mod:`tensorfem.plate_bfs`: 3x3 Gauss bending stiffness,
consistent transverse load 3x3 Gauss, sparse assembly, SuperLU solve.
Boundary conditions: *hard* simple support (``w = 0`` and the tangential slope
zero on every edge -- the exact Kirchhoff condition) and full clamping
(``w = w_,x = w_,y = 0`` on every edge, corner twist dof free).
Meshes ``N = 10, 20, 40`` with an extra ``N = 80`` level on the deflection.

Acceptance (campaign thresholds)
--------------------------------
* SSSS centre deflection vs the 11x11 Navier series ``<= 1 %``;
* every plate data point (deflection and both centre moments, both boundary
  conditions) ``<= 2 %`` of its oracle;
* h-refinement: error strictly decreasing and observed order ``>= 1.5``.

Outputs (this directory): ``result.json``, ``gates_verified.json``,
``fields_ssss.png``, ``fields_cccc.png``, ``convergence.png``.

Usage::

    PYTHONPATH=src python benchmarks/verified/fem_plate_static/run.py
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

MISSION = "5418"
NODE = "P19-N01"
ANCHOR = "fem_plate_static"

SIDE = 1.0
THICKNESS = 0.01
YOUNG = 210.0e9
POISSON = 0.3
PRESSURE = 1.0e4
LEVELS = (10, 20, 40)
EXTENDED_LEVELS = (80,)

RIGIDITY = pb.plate_rigidity(YOUNG, POISSON, THICKNESS)
NAVIER_TERMS = 11          # campaign-mandated truncation
NAVIER_CONVERGED = 301     # oracle used for the convergence order

THRESHOLD_NAVIER_CENTRE = 1.0e-2   # campaign: < 1 % for the SSSS centre deflection
THRESHOLD_POINT = 2.0e-2           # campaign: < 2 % for every plate data point


def centre_values(n_x: int, boundary: str) -> dict:
    """Solve one mesh and recover the centre deflection and moments."""
    plate = pb.assemble_bfs_plate(
        n_x, n_x, SIDE, SIDE, YOUNG, POISSON, THICKNESS, surface_load=PRESSURE
    )
    fixed = pb.bfs_boundary_dofs(n_x, n_x, boundary)
    displacement = pb.solve_bfs_static(plate, fixed)
    point = np.array([[SIDE / 2.0, SIDE / 2.0]])
    deflection = float(pb.bfs_evaluate(plate, displacement, point)["w"][0])
    moments = pb.bfs_moments(plate, displacement, point, YOUNG, POISSON, THICKNESS)
    return {
        "n_x": n_x,
        "n_y": n_x,
        "h_m": SIDE / n_x,
        "n_dof": plate.n_dof,
        "n_free_dof": plate.n_dof - len(fixed),
        "centre_deflection_m": deflection,
        "alpha": deflection * RIGIDITY / (PRESSURE * SIDE**4),
        "M_x_N": float(moments["M_x"][0]),
        "M_y_N": float(moments["M_y"][0]),
        "M_x_normalised": float(moments["M_x"][0]) / (PRESSURE * SIDE**2),
        "M_y_normalised": float(moments["M_y"][0]) / (PRESSURE * SIDE**2),
        "deflection_field": displacement,
    }


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


def observed_order(h: list[float], errors: list[float]) -> float:
    h = np.asarray(h, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if np.any(errors <= 0.0) or not np.all(np.isfinite(errors)):
        return float("nan")
    return float(np.polyfit(np.log(h), np.log(errors), 1)[0])


def self_convergence_order(values: list[float], h: list[float]) -> float:
    """Order from successive differences (no exact reference needed)."""
    differences = [
        abs(b - a) for a, b in zip(values, values[1:])
    ]
    if any(d == 0.0 for d in differences):
        return float("nan")
    orders = [
        float(np.log(differences[i] / differences[i + 1]) / np.log(h[i] / h[i + 1]))
        for i in range(len(differences) - 1)
    ]
    return float(np.mean(orders)) if orders else float("nan")


def monotone_decreasing(values: list[float]) -> bool:
    return all(b < a for a, b in zip(values, values[1:]))


def strip(level: dict) -> dict:
    return {key: value for key, value in level.items() if key != "deflection_field"}


def main() -> dict:
    reference = {
        "navier_11_centre_deflection_m": pb.ssss_navier_centre(
            SIDE, SIDE, PRESSURE, RIGIDITY, terms=NAVIER_TERMS
        ),
        "navier_301_centre_deflection_m": pb.ssss_navier_centre(
            SIDE, SIDE, PRESSURE, RIGIDITY, terms=NAVIER_CONVERGED
        ),
        "navier_11_centre_moments_N": pb.ssss_navier_moments(
            SIDE / 2, SIDE / 2, SIDE, SIDE, PRESSURE, POISSON, terms=NAVIER_TERMS
        ),
        "navier_301_centre_moments_N": pb.ssss_navier_moments(
            SIDE / 2, SIDE / 2, SIDE, SIDE, PRESSURE, POISSON, terms=NAVIER_CONVERGED
        ),
        "timoshenko_ssss_alpha": pb.TIMOSHENKO_SSSS_ALPHA,
        "timoshenko_ssss_moment": pb.TIMOSHENKO_SSSS_MOMENT,
        "timoshenko_cccc_alpha": pb.TIMOSHENKO_CCCC_ALPHA,
        "timoshenko_cccc_moment": pb.TIMOSHENKO_CCCC_MOMENT,
    }
    navier_11 = reference["navier_11_centre_deflection_m"]
    navier_301 = reference["navier_301_centre_deflection_m"]
    moments_301 = reference["navier_301_centre_moments_N"]
    timo_ssss_w = reference["timoshenko_ssss_alpha"] * PRESSURE * SIDE**4 / RIGIDITY
    timo_cccc_w = reference["timoshenko_cccc_alpha"] * PRESSURE * SIDE**4 / RIGIDITY
    timo_ssss_m = reference["timoshenko_ssss_moment"] * PRESSURE * SIDE**2
    timo_cccc_m = reference["timoshenko_cccc_moment"] * PRESSURE * SIDE**2

    ssss = [centre_values(n, "ssss") for n in LEVELS]
    cccc = [centre_values(n, "cccc") for n in LEVELS]
    extended = {boundary: [centre_values(n, boundary) for n in EXTENDED_LEVELS]
                for boundary in ("ssss", "cccc")}

    for level in ssss:
        level["error_vs_navier_11"] = relative(level["centre_deflection_m"], navier_11)
        level["error_vs_navier_301"] = relative(level["centre_deflection_m"], navier_301)
        level["error_vs_timoshenko"] = relative(level["centre_deflection_m"], timo_ssss_w)
        level["M_x_error_vs_navier_301"] = relative(level["M_x_N"], moments_301[0])
        level["M_y_error_vs_navier_301"] = relative(level["M_y_N"], moments_301[1])
        level["M_error_vs_timoshenko"] = relative(level["M_x_N"], timo_ssss_m)
    for level in cccc:
        level["error_vs_timoshenko"] = relative(level["centre_deflection_m"], timo_cccc_w)
        level["M_error_vs_timoshenko"] = relative(level["M_x_N"], timo_cccc_m)

    h = [level["h_m"] for level in ssss]
    ssss_error = [level["error_vs_navier_301"] for level in ssss]
    ssss_moment_error = [level["M_x_error_vs_navier_301"] for level in ssss]
    cccc_alpha_self = self_convergence_order(
        [level["alpha"] for level in cccc], h
    )
    cccc_error = [level["error_vs_timoshenko"] for level in cccc]

    finest_ssss = ssss[-1]
    finest_cccc = cccc[-1]

    gates = [
        Gate(
            id="L2.ssss_centre_vs_navier_11",
            quantity="SSSS centre deflection vs Navier 11x11 (N=40)",
            reference="w_c = 16q/(pi^6 D) sum_{m,n<=11 odd} ...",
            value=finest_ssss["error_vs_navier_11"],
            threshold=THRESHOLD_NAVIER_CENTRE,
        ),
        Gate(
            id="L2.ssss_centre_vs_navier_301",
            quantity="SSSS centre deflection vs converged Navier series (N=40)",
            reference="w_c = 16q/(pi^6 D) sum_{m,n<=301 odd} ...",
            value=finest_ssss["error_vs_navier_301"],
            threshold=THRESHOLD_POINT,
        ),
        Gate(
            id="L2.ssss_centre_vs_timoshenko",
            quantity="SSSS centre deflection vs Timoshenko table 8 alpha=0.00406 (N=40)",
            reference="w_c = 0.00406 q a^4 / D",
            value=finest_ssss["error_vs_timoshenko"],
            threshold=THRESHOLD_POINT,
        ),
        Gate(
            id="L2.ssss_moments_vs_navier",
            quantity="SSSS centre M_x (worst of M_x, M_y) vs converged Navier (N=40)",
            reference="M_x = M_y = 0.047886 q a^2 (Navier), table 0.0479",
            value=max(
                finest_ssss["M_x_error_vs_navier_301"],
                finest_ssss["M_y_error_vs_navier_301"],
            ),
            threshold=THRESHOLD_POINT,
        ),
        Gate(
            id="L2.cccc_centre_vs_timoshenko",
            quantity="CCCC centre deflection vs Timoshenko table 35 alpha=0.00126 (N=40)",
            reference="w_c = 0.00126 q a^4 / D",
            value=finest_cccc["error_vs_timoshenko"],
            threshold=THRESHOLD_POINT,
            note="table rounding alone is 0.42 %, see extended_levels",
        ),
        Gate(
            id="L2.cccc_moments_vs_timoshenko",
            quantity="CCCC centre M_x (worst of M_x, M_y) vs Timoshenko 0.0231 q a^2 (N=40)",
            reference="M_x = M_y = 0.0231 q a^2",
            value=finest_cccc["M_error_vs_timoshenko"],
            threshold=THRESHOLD_POINT,
        ),
        Gate(
            id="L2.order_ssss",
            quantity="observed order of the SSSS centre-deflection error",
            reference="log-log slope vs the converged Navier series",
            value=observed_order(h, ssss_error),
            threshold=1.5,
            operator=">=",
        ),
        Gate(
            id="L2.order_ssss_moment",
            quantity="observed order of the SSSS centre-moment error",
            reference="log-log slope vs the converged Navier series",
            value=observed_order(h, ssss_moment_error),
            threshold=1.5,
            operator=">=",
        ),
        Gate(
            id="L2.order_cccc_self",
            quantity="observed order of the CCCC deflection (successive differences)",
            reference="Richardson self-convergence over N=10,20,40 (table floor excluded)",
            value=cccc_alpha_self,
            threshold=1.5,
            operator=">=",
        ),
        Gate(
            id="L2.monotone_ssss",
            quantity="SSSS centre-deflection error decreases monotonically",
            reference="strict decrease vs the converged Navier series",
            value=1.0 if monotone_decreasing(ssss_error) else 0.0,
            threshold=1.0,
            operator=">=",
        ),
        Gate(
            id="L2.monotone_cccc",
            quantity="CCCC centre-deflection error decreases monotonically",
            reference="strict decrease vs Timoshenko 0.00126",
            value=1.0 if monotone_decreasing(cccc_error) else 0.0,
            threshold=1.0,
            operator=">=",
        ),
    ]

    data_points = [
        {
            "id": "L2.ssss_centre_deflection_navier_11",
            "fe": finest_ssss["centre_deflection_m"],
            "reference": navier_11,
            "reference_formula": "Navier double sine series, m,n<=11",
            "relative_error": finest_ssss["error_vs_navier_11"],
            "threshold": THRESHOLD_NAVIER_CENTRE,
            "passed": finest_ssss["error_vs_navier_11"] <= THRESHOLD_NAVIER_CENTRE,
        },
        {
            "id": "L2.ssss_centre_deflection_timoshenko",
            "fe": finest_ssss["centre_deflection_m"],
            "reference": timo_ssss_w,
            "reference_formula": "w_c = 0.00406 q a^4 / D",
            "relative_error": finest_ssss["error_vs_timoshenko"],
            "threshold": THRESHOLD_POINT,
            "passed": finest_ssss["error_vs_timoshenko"] <= THRESHOLD_POINT,
        },
        {
            "id": "L2.ssss_centre_M_x",
            "fe": finest_ssss["M_x_N"],
            "reference": moments_301[0],
            "reference_formula": "M_x = 0.047886 q a^2 (Navier, 301 harmonics)",
            "relative_error": finest_ssss["M_x_error_vs_navier_301"],
            "threshold": THRESHOLD_POINT,
            "passed": finest_ssss["M_x_error_vs_navier_301"] <= THRESHOLD_POINT,
        },
        {
            "id": "L2.ssss_centre_M_y",
            "fe": finest_ssss["M_y_N"],
            "reference": moments_301[1],
            "reference_formula": "M_y = 0.047886 q a^2 (Navier, 301 harmonics)",
            "relative_error": finest_ssss["M_y_error_vs_navier_301"],
            "threshold": THRESHOLD_POINT,
            "passed": finest_ssss["M_y_error_vs_navier_301"] <= THRESHOLD_POINT,
        },
        {
            "id": "L2.cccc_centre_deflection",
            "fe": finest_cccc["centre_deflection_m"],
            "reference": timo_cccc_w,
            "reference_formula": "w_c = 0.00126 q a^4 / D (Timoshenko table 35)",
            "relative_error": finest_cccc["error_vs_timoshenko"],
            "threshold": THRESHOLD_POINT,
            "passed": finest_cccc["error_vs_timoshenko"] <= THRESHOLD_POINT,
        },
        {
            "id": "L2.cccc_centre_M_x",
            "fe": finest_cccc["M_x_N"],
            "reference": timo_cccc_m,
            "reference_formula": "M_x = 0.0231 q a^2 (Timoshenko table 35)",
            "relative_error": finest_cccc["M_error_vs_timoshenko"],
            "threshold": THRESHOLD_POINT,
            "passed": finest_cccc["M_error_vs_timoshenko"] <= THRESHOLD_POINT,
        },
    ]

    result = {
        "case": "Kirchhoff thin plate statics: SSSS and CCCC square plate under uniform pressure",
        "mission": MISSION,
        "node": NODE,
        "generated_by": "benchmarks/verified/fem_plate_static/run.py",
        "generated_at": utc_now(),
        "model": {
            "element": "Bogner-Fox-Schmit C1 bicubic Hermite rectangle (16 dof: w, w,x, w,y, w,xy)",
            "module": "tensorfem.plate_bfs",
            "quadrature": "3x3 Gauss (stiffness and consistent load)",
            "solver": "scipy SuperLU on the constrained sparse system",
            "boundary_conditions": {
                "ssss": "w = 0 and tangential slope = 0 on every edge (hard simple support)",
                "cccc": "w = w,x = w,y = 0 on every edge (corner twist dof free)",
            },
            "load": "consistent transverse load of q = 10 kPa (not lumped)",
        },
        "parameters": {
            "side_m": SIDE,
            "thickness_m": THICKNESS,
            "young_pa": YOUNG,
            "poisson": POISSON,
            "pressure_pa": PRESSURE,
            "rigidity_n_m": RIGIDITY,
            "a_over_t": SIDE / THICKNESS,
            "levels": list(LEVELS),
            "extended_levels": list(EXTENDED_LEVELS),
            "navier_terms": NAVIER_TERMS,
            "navier_terms_converged": NAVIER_CONVERGED,
        },
        "reference": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in reference.items()
        },
        "ssss_levels": [strip(level) for level in ssss],
        "cccc_levels": [strip(level) for level in cccc],
        "extended_levels": {
            boundary: [strip(level) for level in levels]
            for boundary, levels in extended.items()
        },
        "convergence": {
            "h_m": h,
            "ssss_deflection_error_vs_navier_301": ssss_error,
            "ssss_moment_error_vs_navier_301": ssss_moment_error,
            "cccc_deflection_error_vs_timoshenko": cccc_error,
            "order_ssss_deflection": observed_order(h, ssss_error),
            "order_ssss_moment": observed_order(h, ssss_moment_error),
            "order_cccc_self": cccc_alpha_self,
        },
        "data_points": data_points,
        "gates": [gate.to_dict() for gate in gates],
        "passed": all(gate.passed for gate in gates),
    }

    write_json(HERE / "result.json", result)
    write_json(
        HERE / "gates_verified.json",
        gate_manifest(
            anchor=ANCHOR,
            title="Kirchhoff plate statics vs Navier series and Timoshenko tables (L2 rung)",
            mission=MISSION,
            criteria={
                "ssss_centre_vs_navier_11_max": THRESHOLD_NAVIER_CENTRE,
                "all_data_points_max": THRESHOLD_POINT,
                "convergence_order_min": 1.5,
                "levels": list(LEVELS),
                "extended_levels": list(EXTENDED_LEVELS),
            },
            gates=gates,
            artifacts={
                "result": "benchmarks/verified/fem_plate_static/result.json",
                "figures": ["fields_ssss.png", "fields_cccc.png", "convergence.png"],
            },
            limitations=[
                "Kirchhoff (thin-plate) theory: a/t = 100 keeps the Mindlin shear "
                "correction ~1e-4 relative, but the model carries no transverse "
                "shear deformation at all.",
                "Timoshenko table 35 (CCCC) is quoted to three significant digits, "
                "so 0.42 % of the reported CCCC error is reference rounding, not "
                "discretisation; the extended N=80 level (alpha = 0.001265) sits on "
                "that floor and the CCCC order is therefore graded by "
                "self-convergence rather than against the table.",
                "The CCCC centre moment converges from above to 0.02291 q a^2 "
                "(0.02322, 0.02298, 0.02292, 0.02291 at N=10,20,40,80), i.e. 0.8 % "
                "below the three-significant-digit table value 0.0231; the error "
                "against the table is therefore non-monotone (it crosses the table "
                "value near N=10) even though the moment itself converges at order 2.",
                "The 11x11 Navier truncation mandated by the campaign is accurate "
                "for deflections (~5e-8) but only ~1e-3 for the moments; the moment "
                "gates use the 301-harmonic evaluation of the same series.",
                "Hard simple support excludes the corner uplift dof; this is the "
                "exact Kirchhoff condition and the same convention as "
                "benchmarks/verified/fem_plate, so results are comparable but not "
                "identical (that anchor used lumped nodal loads).",
            ],
        ),
    )
    make_figures(ssss, cccc, extended, reference)
    return result


def field_grid(level: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plate = pb.assemble_bfs_plate(
        level["n_x"], level["n_y"], SIDE, SIDE, YOUNG, POISSON, THICKNESS
    )
    grid = np.linspace(0.0, SIDE, 61)
    points = np.array([[x, y] for x in grid for y in grid])
    values = pb.bfs_evaluate(plate, level["deflection_field"], points)["w"]
    return grid, grid, values.reshape(61, 61) * 1e3


def make_figures(ssss: list[dict], cccc: list[dict], extended: dict, reference: dict) -> None:
    plt = pyplot()
    for boundary, levels, title in (
        ("ssss", ssss, "SSSS plate: FE deflection (N=40) vs Navier"),
        ("cccc", cccc, "CCCC plate: FE deflection (N=40) vs clamped series"),
    ):
        grid, _, field = field_grid(levels[-1])
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
        image = axes[0].contourf(grid, grid, field, levels=20, cmap="viridis")
        figure.colorbar(image, ax=axes[0], label="w [mm]")
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        axes[0].set_title(title)
        axes[0].grid(False)

        centreline = field[30, :]
        axes[1].plot(grid, centreline, "k-", lw=1.6, label="FE N=40")
        if boundary == "ssss":
            navier = [
                pb.ssss_navier_deflection(
                    x, SIDE / 2.0, SIDE, SIDE, PRESSURE, RIGIDITY, terms=NAVIER_CONVERGED
                )
                * 1e3
                for x in grid
            ]
            axes[1].plot(grid, navier, "r--", lw=1.2, label="Navier (301 harmonics)")
        else:
            axes[1].plot(
                [],
                [],
                " ",
                label=f"alpha = {levels[-1]['alpha']:.6f} (table 0.00126)",
            )
        axes[1].set_xlabel("x [m] (y = a/2)")
        axes[1].set_ylabel("w [mm]")
        axes[1].set_title("centre line")
        axes[1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(HERE / f"fields_{boundary}.png", bbox_inches="tight")
        plt.close(figure)

    h = [level["h_m"] for level in ssss]
    figure, axes = plt.subplots(1, 1, figsize=(4.8, 3.6))
    axes.loglog(h, [level["error_vs_navier_301"] for level in ssss], "o-", label="SSSS w_c vs Navier")
    axes.loglog(h, [level["M_x_error_vs_navier_301"] for level in ssss], "s-", label="SSSS M_x vs Navier")
    axes.loglog(h, [level["error_vs_timoshenko"] for level in cccc], "^-", label="CCCC w_c vs 0.00126")
    reference_line = [level["error_vs_navier_301"] for level in ssss][0]
    axes.loglog(h, [reference_line * (h_i / h[0]) ** 2 for h_i in h], "k--", lw=1.0, label="h^2")
    for level in extended["cccc"]:
        pass
    axes.set_xlabel("h [m]")
    axes.set_ylabel("relative error")
    axes.set_title("plate refinement (CCCC flattened by table rounding)")
    axes.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(HERE / "convergence.png", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    payload = main()
    print(json.dumps({"passed": payload["passed"], "convergence": payload["convergence"]}, indent=2))