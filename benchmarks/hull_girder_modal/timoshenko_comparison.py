"""Anchor 1 and 4 of the hull-girder modal benchmark: Timoshenko frequencies.

For each benchmark girder the dry (in-vacuo) simply supported bending
frequencies of modes ``n = 1..4`` are computed along four independent paths and
compared:

* ``f_EB``      -- Euler--Bernoulli closed form (no shear, no rotary inertia);
* ``f_T1``      -- the first-order Timoshenko correction used by the task
                   specification, ``f_EB * sqrt(1 - (n pi/L)^2 EI/(kappa G A))``;
* ``f_exact``   -- the exact Hu--Timoshenko dispersion relation
                   (``timoshenko_beam_modal``), the physical reference;
* ``f_FEM(N)``  -- the finite element model of the same girder on ``N`` equal
                   two-node Timoshenko elements (``N in {20, 40, 80}``).

Grid convergence of the finite element frequencies is quantified with the
Roache grid-convergence index, which also yields the observed order of accuracy
that the benchmark gate requires to be at least two.

Every number printed by :func:`main` is dumped into the ``timoshenko`` block of
``benchmarks/hull_girder_modal/result.json`` by ``make_result.py``.
"""

from __future__ import annotations

import math

from tensorfem.hull_girder_modal import (
    BoxGirderSection,
    exact_hull_girder_angular_frequencies,
    gci,
    timoshenko_corrected_angular_frequencies,
)
from tensorfem.timoshenko_beam_modal import (
    euler_bernoulli_simply_supported_angular_frequencies,
    simply_supported_timoshenko_modes,
)

#: Benchmark girders.  ``girder_g`` is the *gate* case: a slender box girder in
#: which the first-order correction of the task specification stays inside its
#: validity range over modes 1..4.  ``hull_s`` carries ship-like proportions
#: (L/D = 12.9) and is reported to document where that closed form runs out.
GIRDERS: dict[str, dict] = {
    "girder_g": {
        "length": 200.0,
        "breadth": 16.0,
        "depth": 4.0,
        "wall": 0.020,
        "draft": 2.4,
        "role": "gate case (slender box girder, L/D = 50)",
    },
    "hull_s": {
        "length": 180.0,
        "breadth": 24.0,
        "depth": 14.0,
        "wall": 0.022,
        "draft": 8.4,
        "role": "ship-proportioned girder (L/D = 12.9), applicability study",
    },
}

MODES = 4
MESHES = (20, 40, 80)


def build_section(spec: dict) -> BoxGirderSection:
    return BoxGirderSection(
        length=spec["length"],
        breadth=spec["breadth"],
        depth=spec["depth"],
        wall=spec["wall"],
        draft=spec["draft"],
    )


def analyse_case(spec: dict) -> dict:
    """All dry-mode anchor numbers for one girder."""
    section = build_section(spec)
    section.validate()
    beam = section.to_timoshenko_section()

    f_eb = euler_bernoulli_simply_supported_angular_frequencies(beam, MODES) / (2.0 * math.pi)
    f_exact = exact_hull_girder_angular_frequencies(section, MODES) / (2.0 * math.pi)

    # First-order Timoshenko correction of the specification; it is only
    # defined while 1 - (n pi/L)^2 EI/(kappa G A) > 0.
    validity = [
        (n * math.pi / section.length) ** 2 * section.kappa_shear_ratio
        for n in range(1, MODES + 1)
    ]
    try:
        f_first = timoshenko_corrected_angular_frequencies(section, MODES) / (2.0 * math.pi)
        f_first_list: list[float | None] = [float(v) for v in f_first]
    except ValueError:
        f_first_list = []
        for n in range(1, MODES + 1):
            try:
                value = timoshenko_corrected_angular_frequencies(section, n)
                f_first_list.append(float(value[-1]) / (2.0 * math.pi))
            except ValueError:
                f_first_list.append(None)

    fem: dict[int, list[float]] = {}
    for elements in MESHES:
        result = simply_supported_timoshenko_modes(beam, elements, MODES)
        fem[elements] = [float(v) for v in result.frequencies_hz]

    def rel(a: float, b: float) -> float:
        return abs(a - b) / abs(b) * 100.0

    per_mode = []
    for index in range(MODES):
        n = index + 1
        reference = float(f_exact[index])
        entry = {
            "mode": n,
            "f_euler_bernoulli_hz": float(f_eb[index]),
            "f_timoshenko_first_order_hz": f_first_list[index],
            "f_timoshenko_exact_hz": reference,
            "f_fem_hz": {str(mesh): fem[mesh][index] for mesh in MESHES},
            "validity_index": validity[index],
            "rel_error_first_order_pct": (
                None if f_first_list[index] is None else rel(f_first_list[index], reference)
            ),
            "rel_error_euler_bernoulli_pct": rel(float(f_eb[index]), reference),
            "rel_error_fem_pct": {str(mesh): rel(fem[mesh][index], reference) for mesh in MESHES},
            "shear_reduction_pct": (1.0 - reference / float(f_eb[index])) * 100.0,
        }
        per_mode.append(entry)

    convergence = []
    for index in range(MODES):
        series = [fem[mesh][index] for mesh in MESHES]  # coarse..fine order
        study = gci(series[2], series[1], series[0], refinement=2.0)
        convergence.append(
            {
                "mode": index + 1,
                "observed_order": study["observed_order"],
                "extrapolated_hz": study["extrapolated"],
                "gci_fine_pct": study["gci_fine"] * 100.0,
                "rel_error_fine_pct": study["rel_error_fine"] * 100.0,
                "rel_error_vs_exact_pct": rel(series[2], float(f_exact[index])),
                "convergence_ratio": study["convergence_ratio"],
            }
        )

    audit = section.geometry_audit()
    return {
        "role": spec["role"],
        "geometry": {
            "length_m": section.length,
            "breadth_m": section.breadth,
            "depth_m": section.depth,
            "wall_m": section.wall,
            "draft_m": section.section_draft,
            "slenderness_L_over_D": section.slenderness,
            "block_coefficient_area_over_BD": section.block_coefficient,
        },
        "section_properties": {
            "area_m2": section.area,
            "inertia_m4": section.inertia,
            "shear_area_m2": section.shear_area,
            "effective_shear_coefficient": section.effective_shear_coefficient,
            "mass_per_length_kg_m": section.mass_per_length,
            "bending_stiffness_Nm2": section.bending_stiffness,
            "shear_stiffness_N": section.shear_stiffness,
            "EI_over_kappa_GA_m2": section.kappa_shear_ratio,
            "inertia_over_area_m2": section.inertia / section.area,
        },
        "geometry_audit": audit,
        "modes": per_mode,
        "grid_convergence": convergence,
    }


def compute() -> dict:
    return {name: analyse_case(spec) for name, spec in GIRDERS.items()}


def main() -> int:  # pragma: no cover - reporting helper
    import json

    data = compute()
    for name, case in data.items():
        print(f"== {name}: {case['role']}")
        print(
            f"   L/B/D/t = {case['geometry']['length_m']}/{case['geometry']['breadth_m']}"
            f"/{case['geometry']['depth_m']}/{case['geometry']['wall_m']} m"
            f"  (L/D = {case['geometry']['slenderness_L_over_D']:.1f},"
            f" EI/kGA = {case['section_properties']['EI_over_kappa_GA_m2']:.1f} m2)"
        )
        print(
            f"   geometry audit: area rel err {case['geometry_audit']['area_rel_error']:.2e},"
            f" inertia rel err {case['geometry_audit']['inertia_rel_error']:.2e}"
        )
        print("   mode   f_EB     f_T1st    f_exact   f_FEM80   err_T1st%  err_FEM80%  p(GCI)")
        for mode, conv in zip(case["modes"], case["grid_convergence"]):
            first = mode["f_timoshenko_first_order_hz"]
            first_txt = "     n/a" if first is None else f"{first:9.5f}"
            err_first = mode["rel_error_first_order_pct"]
            err_txt = "     n/a" if err_first is None else f"{err_first:8.3f}"
            print(
                f"   {mode['mode']:>4} {mode['f_euler_bernoulli_hz']:9.5f} {first_txt}"
                f" {mode['f_timoshenko_exact_hz']:9.5f}"
                f" {mode['f_fem_hz']['80']:9.5f} {err_txt}"
                f" {mode['rel_error_fem_pct']['80']:10.4f} {conv['observed_order']:8.2f}"
            )
    print(json.dumps({"cases": list(data)}, indent=None))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())