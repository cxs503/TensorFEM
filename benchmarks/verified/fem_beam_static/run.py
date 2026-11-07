#!/usr/bin/env python3
"""P19-N01 rung L1: Euler--Bernoulli beam statics against closed-form solutions.

Physics
-------
Two linear-elastic beam cases with textbook exact solutions:

1. **Simply supported beam, uniform load** ``q = 1 N/m``, ``L = 10 m``,
   ``EI = 1000 N m^2``.  Euler--Bernoulli theory gives

       w(x) = q x (L^3 - 2 L x^2 + x^3) / (24 EI),      w(L/2) = 5 q L^4 / (384 EI).

   Five sections ``x = 0.9, 2.4, 3.9, 5.4, 6.9 m`` are compared.  The sections
   are deliberately *off-node* for all three meshes (h = 1, 0.5, 0.25 m); for
   this element the nodal values are exact, so grading the interpolation at
   interior points is what actually exercises the shape functions.

2. **Cantilever, tip point load** ``P = 0.1 N``, ``L = 10 m``, ``EI = 1000 N m^2``:

       w(L) = P L^3 / (3 EI) = 3.333e-2 m,    theta(L) = P L^2 / (2 EI) = 5.0e-3 rad.

Model
-----
Two-node cubic Hermite beam element (4 dof per element: ``w, w_,x``) with the
stiffness of :func:`tensorfem.modal.beam_bending_stiffness` and the consistent
nodal load ``q*[le/2, le^2/12, le/2, -le^2/12]``.  Meshes of ``N = 10, 20, 40``
elements; for the simply supported beam only ``w`` at the two supports is
constrained (rotations free), for the cantilever ``w`` and ``w_,x`` at the
clamp.  Dense LU on the free block (the models are tiny).

Acceptance (campaign thresholds)
--------------------------------
* five section deflections: relative error ``<= 0.5 %``;
* cantilever tip deflection and rotation: ``<= 0.5 %``;
* h-refinement: error strictly decreasing and observed order ``>= 2``
  (measured on the ``L2`` relative error of the deflection field and on the
  section errors; the expected rate is 4 for this element).

Outputs (this directory): ``result.json``, ``gates_verified.json``,
``deflection_profile.png``, ``convergence.png``.

Usage::

    PYTHONPATH=src python benchmarks/verified/fem_beam_static/run.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from _harness import Gate, gate_manifest, pyplot, utc_now, write_json  # noqa: E402

from tensorfem.modal import beam_bending_stiffness  # noqa: E402
from tensorfem.plate_bfs import (  # noqa: E402
    cantilever_tip_load_deflection,
    simply_supported_uniform_load_deflection,
)

MISSION = "5418"
NODE = "P19-N01"
ANCHOR = "fem_beam_static"

LENGTH = 10.0
EI = 1000.0
UNIFORM_LOAD = 1.0
TIP_LOAD = 0.1
SECTIONS = (0.9, 2.4, 3.9, 5.4, 6.9)
LEVELS = (10, 20, 40)
THRESHOLD = 5.0e-3  # 0.5 % on every beam data point


# ---------------------------------------------------------------------------
# Finite element machinery
# ---------------------------------------------------------------------------

def beam_mesh(elements: int) -> tuple[np.ndarray, float]:
    """Assembled 4-dof-per-node Hermite stiffness of a uniform beam."""
    length_element = LENGTH / elements
    stiffness = beam_bending_stiffness(EI, length_element).numpy()
    ndof = 2 * (elements + 1)
    matrix = np.zeros((ndof, ndof))
    for element in range(elements):
        dofs = np.arange(2 * element, 2 * element + 4)
        matrix[np.ix_(dofs, dofs)] += stiffness
    return matrix, length_element


def uniform_load_vector(elements: int, length_element: float) -> np.ndarray:
    """Consistent nodal load of a uniform transverse load ``q``."""
    q = UNIFORM_LOAD
    element = q * np.array(
        [
            length_element / 2.0,
            length_element**2 / 12.0,
            length_element / 2.0,
            -length_element**2 / 12.0,
        ]
    )
    vector = np.zeros(2 * (elements + 1))
    for e in range(elements):
        vector[2 * e : 2 * e + 4] += element
    return vector


def solve(elements: int, fixed: tuple[int, ...], load: np.ndarray) -> np.ndarray:
    stiffness, _ = beam_mesh(elements)
    free = np.setdiff1d(np.arange(stiffness.shape[0]), np.array(fixed))
    displacement = np.zeros(stiffness.shape[0])
    displacement[free] = np.linalg.solve(
        stiffness[np.ix_(free, free)], load[free]
    )
    return displacement


def hermite(displacement: np.ndarray, x: float, elements: int) -> tuple[float, float]:
    """Deflection and slope of the FE field at ``x`` (Hermite interpolation)."""
    length_element = LENGTH / elements
    element = min(int(x / length_element), elements - 1)
    s = (x - element * length_element) / length_element
    shape = np.array(
        [
            1.0 - 3.0 * s**2 + 2.0 * s**3,
            length_element * (s - 2.0 * s**2 + s**3),
            3.0 * s**2 - 2.0 * s**3,
            length_element * (-(s**2) + s**3),
        ]
    )
    slope = np.array(
        [
            (-6.0 * s + 6.0 * s**2) / length_element,
            1.0 - 4.0 * s + 3.0 * s**2,
            (6.0 * s - 6.0 * s**2) / length_element,
            -2.0 * s + 3.0 * s**2,
        ]
    )
    local = displacement[2 * element : 2 * element + 4]
    return float(shape @ local), float(slope @ local)


def field_l2_error(displacement: np.ndarray, elements: int, analytic) -> float:
    """Relative ``L2`` error of the deflection field, 4-point Gauss per element."""
    nodes, weights = np.polynomial.legendre.leggauss(4)
    length_element = LENGTH / elements
    error = 0.0
    norm = 0.0
    for element in range(elements):
        for node, weight in zip(nodes, weights):
            s = 0.5 * (node + 1.0)
            x = (element + s) * length_element
            value, _ = hermite(displacement, min(x, LENGTH), elements)
            exact = analytic(x)
            error += 0.5 * weight * length_element * (value - exact) ** 2
            norm += 0.5 * weight * length_element * exact**2
    return float(math.sqrt(error / norm))


def beam_analytic(x: float) -> float:
    return simply_supported_uniform_load_deflection(x, LENGTH, UNIFORM_LOAD, EI)


def observed_order(h: list[float], errors: list[float]) -> float:
    """Least-squares slope of ``log(error)`` against ``log(h)``."""
    h = np.asarray(h)
    errors = np.asarray(errors)
    if np.any(errors <= 0.0):
        return float("nan")
    return float(np.polyfit(np.log(h), np.log(errors), 1)[0])


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

def simply_supported_level(elements: int) -> dict:
    displacement = solve(elements, fixed=(0, 2 * elements), load=uniform_load_vector(
        elements, LENGTH / elements
    ))
    points = []
    for x in SECTIONS:
        value, _ = hermite(displacement, x, elements)
        exact = beam_analytic(x)
        points.append(
            {
                "x_m": x,
                "w_fe_m": value,
                "w_exact_m": exact,
                "relative_error": abs(value - exact) / abs(exact),
            }
        )
    return {
        "elements": elements,
        "dof": 2 * (elements + 1),
        "h_m": LENGTH / elements,
        "sections": points,
        "max_section_relative_error": max(p["relative_error"] for p in points),
        "field_l2_relative_error": field_l2_error(displacement, elements, beam_analytic),
        "midspan_fe_m": hermite(displacement, LENGTH / 2.0, elements)[0],
        "midspan_exact_m": beam_analytic(LENGTH / 2.0),
    }


def cantilever_level(elements: int) -> dict:
    load = np.zeros(2 * (elements + 1))
    load[-2] = TIP_LOAD
    displacement = solve(elements, fixed=(0, 1), load=load)
    tip, rotation = hermite(displacement, LENGTH, elements)
    exact_tip = cantilever_tip_load_deflection(LENGTH, TIP_LOAD, EI)
    exact_rotation = TIP_LOAD * LENGTH**2 / (2.0 * EI)
    return {
        "elements": elements,
        "dof": 2 * (elements + 1),
        "h_m": LENGTH / elements,
        "tip_deflection_fe_m": tip,
        "tip_deflection_exact_m": exact_tip,
        "tip_deflection_relative_error": abs(tip - exact_tip) / exact_tip,
        "tip_rotation_fe_rad": rotation,
        "tip_rotation_exact_rad": exact_rotation,
        "tip_rotation_relative_error": abs(rotation - exact_rotation) / exact_rotation,
    }


def monotone_decreasing(values: list[float]) -> bool:
    return all(b < a for a, b in zip(values, values[1:]))


def main() -> dict:
    ss_levels = [simply_supported_level(n) for n in LEVELS]
    cantilever_levels = [cantilever_level(n) for n in LEVELS]

    h = [level["h_m"] for level in ss_levels]
    section_error_curve = [level["max_section_relative_error"] for level in ss_levels]
    l2_curve = [level["field_l2_relative_error"] for level in ss_levels]

    finest = ss_levels[-1]
    tip = cantilever_levels[-1]

    gates = [
        Gate(
            id="L1.sections",
            quantity="max relative error of the 5 section deflections (N=40)",
            reference="w(x)=q x (L^3-2Lx^2+x^3)/(24EI)",
            value=finest["max_section_relative_error"],
            threshold=THRESHOLD,
            note="off-node sections x=0.9,2.4,3.9,5.4,6.9 m",
        ),
        Gate(
            id="L1.cantilever_tip",
            quantity="cantilever tip deflection relative error (N=40)",
            reference="w_max=P L^3/(3EI)",
            value=tip["tip_deflection_relative_error"],
            threshold=THRESHOLD,
        ),
        Gate(
            id="L1.cantilever_rotation",
            quantity="cantilever tip rotation relative error (N=40)",
            reference="theta_max=P L^2/(2EI)",
            value=tip["tip_rotation_relative_error"],
            threshold=THRESHOLD,
        ),
        Gate(
            id="L1.order_l2",
            quantity="observed convergence order of the L2 deflection error",
            reference="log-log slope over N=10,20,40",
            value=observed_order(h, l2_curve),
            threshold=2.0,
            operator=">=",
        ),
        Gate(
            id="L1.order_sections",
            quantity="observed convergence order of the section errors",
            reference="log-log slope over N=10,20,40",
            value=observed_order(h, section_error_curve),
            threshold=2.0,
            operator=">=",
        ),
        Gate(
            id="L1.monotone",
            quantity="L2 error decreases monotonically over N=10,20,40",
            reference="strict decrease of field L2 error",
            value=1.0 if monotone_decreasing(l2_curve) else 0.0,
            threshold=1.0,
            operator=">=",
        ),
    ]

    data_points = [
        {
            "id": f"L1.section_{index + 1}",
            "x_m": point["x_m"],
            "fe": point["w_fe_m"],
            "reference": point["w_exact_m"],
            "reference_formula": "q x (L^3-2Lx^2+x^3)/(24EI)",
            "relative_error": point["relative_error"],
            "threshold": THRESHOLD,
            "passed": point["relative_error"] <= THRESHOLD,
        }
        for index, point in enumerate(finest["sections"])
    ] + [
        {
            "id": "L1.cantilever_tip_deflection",
            "fe": tip["tip_deflection_fe_m"],
            "reference": tip["tip_deflection_exact_m"],
            "reference_formula": "P L^3/(3EI)",
            "relative_error": tip["tip_deflection_relative_error"],
            "threshold": THRESHOLD,
            "passed": tip["tip_deflection_relative_error"] <= THRESHOLD,
        },
        {
            "id": "L1.cantilever_tip_rotation",
            "fe": tip["tip_rotation_fe_rad"],
            "reference": tip["tip_rotation_exact_rad"],
            "reference_formula": "P L^2/(2EI)",
            "relative_error": tip["tip_rotation_relative_error"],
            "threshold": THRESHOLD,
            "passed": tip["tip_rotation_relative_error"] <= THRESHOLD,
        },
    ]

    result = {
        "case": "Euler-Bernoulli beam statics: simply supported uniform load + cantilever tip load",
        "mission": MISSION,
        "node": NODE,
        "generated_by": "benchmarks/verified/fem_beam_static/run.py",
        "generated_at": utc_now(),
        "model": {
            "element": "2-node cubic Hermite (w, w,x), no shear deformation",
            "stiffness": "tensorfem.modal.beam_bending_stiffness",
            "consistent_load": "q*[le/2, le^2/12, le/2, -le^2/12]",
            "solver": "numpy dense LU on the free dofs",
            "boundary_conditions": {
                "simply_supported": "w=0 at x=0 and x=L (rotations free)",
                "cantilever": "w=0, w,x=0 at x=0",
            },
        },
        "parameters": {
            "length_m": LENGTH,
            "EI_N_m2": EI,
            "uniform_load_N_per_m": UNIFORM_LOAD,
            "tip_load_N": TIP_LOAD,
            "sections_m": list(SECTIONS),
            "levels_elements": list(LEVELS),
            "midspan_exact_m": beam_analytic(LENGTH / 2.0),
            "cantilever_tip_exact_m": cantilever_tip_load_deflection(LENGTH, TIP_LOAD, EI),
        },
        "simply_supported_levels": ss_levels,
        "cantilever_levels": cantilever_levels,
        "convergence": {
            "h_m": h,
            "section_error": section_error_curve,
            "field_l2_error": l2_curve,
            "section_order": observed_order(h, section_error_curve),
            "field_l2_order": observed_order(h, l2_curve),
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
            title="Euler-Bernoulli beam statics vs closed form (L1 of the P19 ladder)",
            mission=MISSION,
            criteria={
                "section_relative_error_max": THRESHOLD,
                "cantilever_tip_relative_error_max": THRESHOLD,
                "convergence_order_min": 2.0,
                "levels_elements": list(LEVELS),
                "sections_m": list(SECTIONS),
            },
            gates=gates,
            artifacts={
                "result": "benchmarks/verified/fem_beam_static/result.json",
                "figures": ["deflection_profile.png", "convergence.png"],
            },
            limitations=[
                "Nodal displacements of this Hermite element are exact; the graded "
                "interior sections measure the interpolation quality, which is why "
                "the observed order is 4 rather than 2.",
                "Euler-Bernoulli kinematics (no transverse shear, no warp) -- the "
                "cantilever tip load gives w/L = 3.3e-3, i.e. the geometric "
                "linearisation holds but shear deformation would matter for a "
                "stubby beam.",
                "Both cases are 1-D idealisations of the physical beam; they grade "
                "the element, not a 3-D solid discretisation.",
            ],
        ),
    )
    make_figures(ss_levels, cantilever_levels)
    return result


def make_figures(ss_levels: list[dict], cantilever_levels: list[dict]) -> None:
    plt = pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    x = np.linspace(0.0, LENGTH, 401)
    axes[0].plot(x, [beam_analytic(v) * 1e3 for v in x], "k-", lw=1.6, label="analytic (EB)")
    for level, style in zip(ss_levels, ("o", "s", "^")):
        displacement = solve(
            level["elements"],
            fixed=(0, 2 * level["elements"]),
            load=uniform_load_vector(level["elements"], LENGTH / level["elements"]),
        )
        xs = np.linspace(0.0, LENGTH, 4 * level["elements"] + 1)
        axes[0].plot(
            xs,
            [hermite(displacement, min(v, LENGTH), level["elements"])[0] * 1e3 for v in xs],
            style,
            ms=3.0,
            fillstyle="none",
            label=f"FE N={level['elements']}",
        )
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("w [mm]")
    axes[0].set_title("SS beam, q = 1 N/m (downward)")
    axes[0].legend(fontsize=7)

    h = [level["h_m"] for level in ss_levels]
    axes[1].loglog(h, [level["field_l2_relative_error"] for level in ss_levels], "o-", label="L2 field error")
    axes[1].loglog(h, [level["max_section_relative_error"] for level in ss_levels], "s-", label="max section error")
    reference = h[0] ** 4
    axes[1].loglog(h, [reference * (h_i / h[0]) ** 4 for h_i in h], "k--", lw=1.0, label="h^4")
    axes[1].set_xlabel("h [m]")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("h-refinement (beam)")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(HERE / "deflection_profile.png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 1, figsize=(4.6, 3.4))
    axes.semilogy(
        [level["elements"] for level in cantilever_levels],
        [level["tip_deflection_relative_error"] for level in cantilever_levels],
        "o-",
    )
    axes.set_xlabel("elements")
    axes.set_ylabel("tip deflection relative error")
    axes.set_title("Cantilever tip load: round-off-limited")
    axes.set_ylim(1e-17, 1e-12)
    figure.tight_layout()
    figure.savefig(HERE / "convergence.png", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    payload = main()
    print(json.dumps({k: payload[k] for k in ("passed", "convergence")}, indent=2))