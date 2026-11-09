"""Detail fatigue primitives: weld-toe stress concentration and S-N life.

The package owns the *detail* end of the structural fatigue chain (as opposed to
the global beam/shell stress chain): parametric weld geometry, a plane finite
element model of the weld-toe notch, IIW hot-spot extrapolation and DNV-RP-C203
double-slope S-N curves.
"""
from __future__ import annotations

from .weld_toe import (
    ENVIRONMENTS,
    LINEAR_STATIONS,
    POISSON_STEEL,
    QUADRATIC_STATIONS,
    YOUNG_STEEL_PA,
    HotSpotResult,
    SNEnvironment,
    TwoSlopeSNCurve,
    WeldDetail,
    WeldToeSolution,
    anchor_stress_concentration,
    detail_fatigue_assessment,
    fatigue_life,
    hot_spot_stress,
    maddox_stress_concentration,
    mesh_convergence,
    nii_stress_concentration,
    solve_weld_detail,
)

__all__ = [
    "ENVIRONMENTS",
    "LINEAR_STATIONS",
    "POISSON_STEEL",
    "QUADRATIC_STATIONS",
    "YOUNG_STEEL_PA",
    "HotSpotResult",
    "SNEnvironment",
    "TwoSlopeSNCurve",
    "WeldDetail",
    "WeldToeSolution",
    "anchor_stress_concentration",
    "detail_fatigue_assessment",
    "fatigue_life",
    "hot_spot_stress",
    "maddox_stress_concentration",
    "mesh_convergence",
    "nii_stress_concentration",
    "solve_weld_detail",
]