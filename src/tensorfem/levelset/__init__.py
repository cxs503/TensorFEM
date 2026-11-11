"""Level-set crack propagation and bending fracture (2-D plates, ice sheets).

Modules
-------
``level_set``
    signed-distance field, Sussman re-initialisation, upwind / semi-Lagrangian
    advection, velocity extension, marching-squares zero contour, path metrics.
``crack_criteria``
    maximum tangential stress and maximum energy release rate kink criteria,
    stress-intensity / energy release rate conversions, bending-Griffith relation
    from the Kirchhoff/plane-stress analogy, tip kinetic law.
``plate_crack``
    Mindlin plate on a Winkler foundation with an explicit (node-duplicated)
    through crack, virtual-crack-extension energy release rate, exact
    Kelvin-function patch-load moment anchors.
``crack_path``
    mixed-mode path tracking with the level set against an independently
    integrated reference path.
``ice_plate_fracture``
    floating ice sheet parameters, analytic fracture-load anchors and the
    level-set driven crack-growth loop.

See ``docs/levelset-ice-plate-fracture.md`` for the verification report.
"""
from .crack_criteria import (  # noqa: F401
    bending_energy_release_rate,
    bending_moment_intensity,
    critical_crack_length,
    energy_release_rate,
    growth_velocity,
    hoop_stress_factor,
    inclined_crack_sifs,
    kink_angle_max_energy,
    kink_angle_max_hoop,
    propagation_direction,
)
from .ice_plate_fracture import (  # noqa: F401
    IceSheet,
    analytical_fracture_load,
    crack_span_from_level_set,
    energy_release_rate_ice,
    equivalent_flaw_size,
    level_set_crack_growth,
    run_ice_plate_fracture_benchmark,
    strength_first_crack_load,
)
from .level_set import (  # noqa: F401
    LevelSetField,
    circle_level_set,
    crack_level_set,
    path_length,
    plane_level_set,
    polyline_deviation,
    resample_polyline,
)
from .plate_crack import (  # noqa: F401
    energy_release_rate_vce,
    graded_axis,
    patch_max_moment,
    patch_max_moment_nevel,
    patch_moment_sum,
    plate_rigidity,
    rect_mindlin_stiffness,
    solve_cracked_plate,
    winkler_length,
)

__all__ = [
    "LevelSetField", "circle_level_set", "crack_level_set", "plane_level_set",
    "resample_polyline", "polyline_deviation", "path_length",
    "kink_angle_max_hoop", "kink_angle_max_energy", "hoop_stress_factor",
    "inclined_crack_sifs", "energy_release_rate", "bending_energy_release_rate",
    "bending_moment_intensity", "critical_crack_length", "growth_velocity",
    "propagation_direction",
    "solve_cracked_plate", "energy_release_rate_vce", "rect_mindlin_stiffness",
    "graded_axis", "patch_max_moment", "patch_moment_sum",
    "patch_max_moment_nevel", "plate_rigidity", "winkler_length",
    "IceSheet", "analytical_fracture_load", "strength_first_crack_load",
    "equivalent_flaw_size", "energy_release_rate_ice",
    "level_set_crack_growth", "crack_span_from_level_set",
    "run_ice_plate_fracture_benchmark",
]
