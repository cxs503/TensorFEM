"""Anchor 2 and 3 of the hull-girder modal benchmark: wet modes and added mass.

Part A -- section added mass (Lewis conformal mapping)
    The two-dimensional heave added mass per unit length of the immersed section
    (doubled about the free surface, i.e. the rigid-lid convention of hull
    vibration practice) is computed exactly for the Lewis family

        z(theta) = M (zeta + a1/zeta + a3/zeta^3),  zeta = exp(i theta),

    and reported as the nondimensional coefficient
    ``c = m33 / (rho b^2)`` together with the added-mass ratio against the
    displaced water mass, ``kappa_33 = m33 / (rho_w B T)``.  The task
    specification writes ``kappa_33 ~ (pi B)/(4 T) * f(B/T)``; the two agree
    when ``f = c/pi``, so the table produced here *is* that correction factor,
    and it shows that within a strip model it can depend only on ``T/B``.

Part B -- wet/dry frequency ratio
    For a freely floating girder the vibrating mass per unit length is the
    displaced mass (structure + payload).  With a uniform added mass the
    generalised eigenproblem gives exactly

        f_wet / f_wet,dry = 1 / sqrt(1 + kappa),   kappa = m33 / m_total.

    The finite element model assembles the added mass with the same shape
    functions, which reproduces the scalar law to the extent that the
    structural rotary inertia is not shared by the added mass (a ~1e-4 effect,
    measured here rather than assumed).

Part C -- non-uniform added mass (the scalar reduction's real test)
    With hull-like lines (beam and draft tapering towards the ends) the added
    mass varies by more than a factor of two along the girder, and the scalar
    reduction ``f_wet = f_dry/sqrt(1+kappa_mean)`` is only an average.  The
    finite element wet frequency is compared with that scalar prediction; the
    deviation is the error of the reduction, not of the finite element model.
"""

from __future__ import annotations

import math

from tensorfem.hull_girder_modal import (
    BoxGirderSection,
    ballast_mass_per_length,
    beam_simply_supported_modes,
    floating_mass_per_length,
    lewis_fit,
    lewis_heave_added_mass,
    lewis_heave_coefficient,
    variable_added_mass_modes,
    wet_dry_frequency_ratio,
    wet_simply_supported_modes,
)

MODES = 4
MESHES = (20, 40, 80)
GATE_GIRDER = {"length": 200.0, "breadth": 16.0, "depth": 4.0, "wall": 0.020, "draft": 2.4}
TAPER_GIRDER = {
    "length": 200.0,
    "breadth": 16.0,
    "depth": 4.0,
    "wall": 0.020,
    "draft": 2.4,
    "beam_taper": 0.65,  # beam reduction at the ends, fraction of B
    "draft_taper": 0.35,  # draft reduction at the ends, fraction of T
}


def build(spec: dict) -> BoxGirderSection:
    return BoxGirderSection(
        length=spec["length"],
        breadth=spec["breadth"],
        depth=spec["depth"],
        wall=spec["wall"],
        draft=spec["draft"],
    )


# ---------------------------------------------------------------------------
# Part A: section added mass
# ---------------------------------------------------------------------------
def lewis_exact_checks() -> list[dict]:
    """Exact limits of the Lewis family (regression anchors of the implementation)."""
    cases = [
        ("circle_b1", 1.0, 1.0, math.pi),
        ("circle_b2.5", 2.5, 2.5, math.pi * 2.5**2),
        ("ellipse_b2_h1", 2.0, 1.0, math.pi * 2.0**2),
        ("flat_plate_b5", 5.0, 0.05, math.pi * 5.0**2),
    ]
    out = []
    for name, half_b, half_h, exact in cases:
        section = lewis_fit(half_b, half_h, area=math.pi * half_b * half_h)
        value = section.heave_added_mass(1.0)
        out.append(
            {
                "name": name,
                "half_breadth_m": half_b,
                "half_draft_m": half_h,
                "added_mass_per_rho": value,
                "exact_per_rho": exact,
                "rel_error": abs(value - exact) / exact,
            }
        )
    return out


def section_study(ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)) -> list[dict]:
    """Added-mass coefficient versus draft/half-breadth (a box section family)."""
    out = []
    for ratio in ratios:
        half_b = 1.0
        half_h = ratio
        coefficient = lewis_heave_coefficient(ratio)
        # kappa against the displaced water mass: m33 / (rho * B * T) with
        # B = 2 b and T = h  ->  c * b / (2 h).
        kappa = coefficient * half_b / (2.0 * half_h)
        form_factor = coefficient / math.pi
        out.append(
            {
                "half_draft_over_half_breadth": ratio,
                "draft_over_breadth": 0.5 * ratio,
                "added_mass_coefficient_c": coefficient,
                "kappa_33_vs_displaced_mass": kappa,
                "mission_formula_piB_over_4T": math.pi * 2.0 * half_b / (4.0 * half_h),
                "form_factor_f_equals_c_over_pi": form_factor,
            }
        )
    return out


def hull_added_mass(section: BoxGirderSection) -> dict:
    """Lewis added mass of the gate girder's section."""
    target_area = 2.0 * section.breadth * section.section_draft
    value, lewis = lewis_heave_added_mass(
        section.half_breadth, section.section_draft, section.water_density, area=target_area
    )
    total = floating_mass_per_length(section)
    return {
        "added_mass_per_length_kg_m": value,
        "half_breadth_m": section.half_breadth,
        "half_draft_m": section.section_draft,
        "lewis_M": lewis.M,
        "lewis_a1": lewis.a1,
        "lewis_a3": lewis.a3,
        "lewis_area_ratio_vs_doubled_box": lewis.area_ratio,
        "lewis_enclosed_area_m2": lewis.area(),
        "structural_mass_per_length_kg_m": section.mass_per_length,
        "displaced_mass_per_length_kg_m": section.displaced_mass_per_length,
        "floating_mass_per_length_kg_m": total,
        "ballast_mass_per_length_kg_m": ballast_mass_per_length(section),
        "kappa_vs_displaced_mass": value / section.displaced_mass_per_length,
        "kappa_vs_structural_mass": value / section.mass_per_length,
        "kappa_vs_floating_mass": value / total,
        "added_mass_coefficient_c": value / (section.water_density * section.half_breadth**2),
    }


# ---------------------------------------------------------------------------
# Part B: uniform wet/dry ratio in the finite element model
# ---------------------------------------------------------------------------
def uniform_wet_dry(section: BoxGirderSection, added: float) -> dict:
    total = floating_mass_per_length(section)
    ballast = ballast_mass_per_length(section)
    kappa = added / total
    closed = wet_dry_frequency_ratio(kappa)
    rows = []
    for elements in MESHES:
        dry = beam_simply_supported_modes(section, elements, (ballast,), MODES)
        wet = wet_simply_supported_modes(section, elements, added, ballast, MODES)
        ratios = [float(a / b) for a, b in zip(wet.frequencies_hz, dry.frequencies_hz)]
        rows.append(
            {
                "elements": elements,
                "dry_hz": [float(v) for v in dry.frequencies_hz],
                "wet_hz": [float(v) for v in wet.frequencies_hz],
                "ratio": ratios,
                "max_rel_error_pct": max(abs(r - closed) / closed * 100.0 for r in ratios),
            }
        )
    # bare steel (no payload): the girder would have to be ballasted to float,
    # but the ratio is still a useful diagnostic of the mass bookkeeping.
    bare = wet_simply_supported_modes(section, MESHES[-1], added, 0.0, MODES)
    bare_dry = beam_simply_supported_modes(section, MESHES[-1], (), MODES)
    bare_ratio = [float(a / b) for a, b in zip(bare.frequencies_hz, bare_dry.frequencies_hz)]
    bare_kappa = added / section.mass_per_length
    return {
        "kappa": kappa,
        "closed_form_ratio": closed,
        "meshes": rows,
        "bare_steel": {
            "kappa": bare_kappa,
            "closed_form_ratio": wet_dry_frequency_ratio(bare_kappa),
            "fem_ratio_finest": bare_ratio,
        },
    }


# ---------------------------------------------------------------------------
# Part C: non-uniform added mass along the girder
# ---------------------------------------------------------------------------
def taper_profile(spec: dict):
    """Hull-like beam and draft distributions ``B(x), T(x)``."""

    def beam(x: float) -> float:
        eta = abs(2.0 * x / spec["length"] - 1.0)
        return spec["breadth"] * (1.0 - spec["beam_taper"] * eta**3)

    def draft(x: float) -> float:
        eta = abs(2.0 * x / spec["length"] - 1.0)
        return spec["draft"] * (1.0 - spec["draft_taper"] * eta**3)

    return beam, draft


def tapered_wet_dry(spec: dict, rho_water: float = 1025.0) -> dict:
    section = build(spec)
    beam_fn, draft_fn = taper_profile(spec)
    ratio_cache: dict[float, float] = {}

    def local_added(x: float) -> float:
        b = 0.5 * beam_fn(x)
        t = draft_fn(x)
        ratio = t / b
        if ratio not in ratio_cache:
            ratio_cache[ratio] = lewis_heave_coefficient(ratio)
        return rho_water * ratio_cache[ratio] * b**2

    # Mean added mass over the length, and the mass that vibrates with it.
    samples = 4001
    length = section.length
    mean_added = sum(local_added(i * length / (samples - 1)) for i in range(samples)) / samples
    total = floating_mass_per_length(section)
    ballast = ballast_mass_per_length(section)
    kappa_mean = mean_added / total
    scalar_ratio = wet_dry_frequency_ratio(kappa_mean)

    rows = []
    for elements in MESHES:
        dry = beam_simply_supported_modes(section, elements, (ballast,), MODES)
        wet = variable_added_mass_modes(section, elements, local_added, ballast, MODES)
        ratios = [float(a / b) for a, b in zip(wet.frequencies_hz, dry.frequencies_hz)]
        rows.append(
            {
                "elements": elements,
                "dry_hz": [float(v) for v in dry.frequencies_hz],
                "wet_hz": [float(v) for v in wet.frequencies_hz],
                "ratio": ratios,
                "rel_error_vs_scalar_pct": [
                    abs(r - scalar_ratio) / scalar_ratio * 100.0 for r in ratios
                ],
            }
        )
    ends = {
        "added_mass_at_ends_kg_m": local_added(0.0),
        "added_mass_amidships_kg_m": local_added(0.5 * length),
        "moment_ratio": mean_added / total,
    }
    return {
        "kappa_mean": kappa_mean,
        "mean_added_mass_kg_m": mean_added,
        "scalar_ratio": scalar_ratio,
        "meshes": rows,
        "profile": ends,
    }


def compute() -> dict:
    section = build(GATE_GIRDER)
    added = hull_added_mass(section)["added_mass_per_length_kg_m"]
    return {
        "section": {
            "gate_girder": hull_added_mass(section),
            "exact_limits": lewis_exact_checks(),
            "coefficient_table": section_study(),
        },
        "uniform": uniform_wet_dry(section, added),
        "tapered": tapered_wet_dry(TAPER_GIRDER),
    }


def main() -> int:  # pragma: no cover - reporting helper
    data = compute()
    sec = data["section"]["gate_girder"]
    print("== Lewis section (gate girder)")
    print(
        f"   b = {sec['half_breadth_m']:.2f} m, h = {sec['half_draft_m']:.2f} m,"
        f" Lewis a1 = {sec['lewis_a1']:.4f}, a3 = {sec['lewis_a3']:.4f},"
        f" area ratio {sec['lewis_area_ratio_vs_doubled_box']:.4f}"
    )
    print(
        f"   m33 = {sec['added_mass_per_length_kg_m']:.1f} kg/m   "
        f"kappa(displaced) = {sec['kappa_vs_displaced_mass']:.4f}   "
        f"kappa(steel) = {sec['kappa_vs_structural_mass']:.2f}"
    )
    print("   exact Lewis limits:")
    for row in data["section"]["exact_limits"]:
        print(f"     {row['name']:<14} rel err {row['rel_error']:.2e}")
    print("   T/B   c=m33/(rho b^2)   kappa_33   piB/4T    f=c/pi")
    for row in data["section"]["coefficient_table"]:
        print(
            f"   {row['draft_over_breadth']:4.2f} {row['added_mass_coefficient_c']:14.5f}"
            f" {row['kappa_33_vs_displaced_mass']:11.4f}"
            f" {row['mission_formula_piB_over_4T']:9.4f} {row['form_factor_f_equals_c_over_pi']:8.4f}"
        )
    uni = data["uniform"]
    print(f"== uniform wet/dry: kappa = {uni['kappa']:.4f}, closed form = {uni['closed_form_ratio']:.6f}")
    for row in uni["meshes"]:
        print(
            f"   N={row['elements']:>3} ratio = "
            + ", ".join(f"{r:.6f}" for r in row["ratio"])
            + f"   max err {row['max_rel_error_pct']:.4f} %"
        )
    tap = data["tapered"]
    print(
        f"== tapered added mass: kappa_mean = {tap['kappa_mean']:.4f},"
        f" scalar ratio = {tap['scalar_ratio']:.6f}"
        f" (m33: ends {tap['profile']['added_mass_at_ends_kg_m']:.0f},"
        f" amidships {tap['profile']['added_mass_amidships_kg_m']:.0f} kg/m)"
    )
    for row in tap["meshes"]:
        print(
            f"   N={row['elements']:>3} ratio = "
            + ", ".join(f"{r:.6f}" for r in row["ratio"])
            + "   dev vs scalar % "
            + ", ".join(f"{v:.2f}" for v in row["rel_error_vs_scalar_pct"])
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())