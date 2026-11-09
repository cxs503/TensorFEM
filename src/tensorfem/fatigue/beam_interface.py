"""Global-to-detail fatigue chain: beam theory nominal stress -> SCF -> hot spot -> S-N life.

This module bridges the *global* structural analysis (beam/shell theory giving
nominal membrane + bending stresses) with the *detail* fatigue assessment
(weld-toe SCF, hot spot stress, S-N life).  It implements the standard
naval-architecture workflow:

1. Global beam theory computes nominal stress ``sigma_nom`` at a cross-section
   (axial force + bending moment / section modulus).
2. A stress concentration factor ``SCF`` (from FEM or empirical formula) converts
   ``sigma_nom`` to the hot spot stress ``sigma_hot = SCF * sigma_nom``.
3. The hot spot stress is fed to the S-N curve to obtain fatigue life.

The module provides:
- :class:`BeamSection` -- prismatic beam cross-section properties (area, I, Z).
- :func:`beam_nominal_stress` -- compute ``sigma_nom = N/A + M/Z``.
- :class:`GlobalDetailChain` -- full chain from global loads to fatigue life.
- :func:`verify_beam_detail_chain` -- machine-verification of the chain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .weld_toe import (
    ENVIRONMENTS,
    WeldDetail,
    detail_fatigue_assessment,
    fatigue_life,
    hot_spot_stress,
    solve_weld_detail,
)


@dataclass(frozen=True)
class BeamSection:
    """Prismatic beam cross-section properties (SI units).

    Parameters
    ----------
    area_m2:
        Cross-sectional area.
    I_m4:
        Second moment of area about the neutral axis.
    c_m:
        Distance from neutral axis to the extreme fibre (for bending stress).
    """

    area_m2: float
    I_m4: float
    c_m: float

    @property
    def Z_m3(self) -> float:
        """Section modulus ``I / c``."""
        return self.I_m4 / self.c_m


def beam_nominal_stress(
    axial_force_n: float,
    bending_moment_nm: float,
    section: BeamSection,
) -> float:
    """Nominal stress from beam theory: ``sigma = N/A + M/Z``.

    Positive values are tensile.  The bending term uses the section modulus
    ``Z = I / c`` so that ``M/Z`` is the extreme-fibre bending stress.
    """
    if section.area_m2 <= 0.0 or section.Z_m3 <= 0.0:
        raise ValueError("section properties must be positive")
    sigma_axial = axial_force_n / section.area_m2
    sigma_bending = bending_moment_nm / section.Z_m3
    return sigma_axial + sigma_bending


@dataclass(frozen=True)
class GlobalDetailResult:
    """Result of the global-to-detail fatigue chain."""

    # Global inputs
    axial_force_n: float
    bending_moment_nm: float
    section: BeamSection
    sigma_nom_pa: float

    # Detail
    detail: WeldDetail
    scf_toe: float
    scf_hotspot_linear: float
    scf_hotspot_quadratic: float

    # Hot spot stresses
    sigma_hot_linear_pa: float
    sigma_hot_quadratic_pa: float

    # Fatigue life
    life_linear_cycles: float
    life_quadratic_cycles: float
    environment: str


def global_to_detail_fatigue(
    axial_force_n: float,
    bending_moment_nm: float,
    section: BeamSection,
    detail: WeldDetail,
    *,
    environment: str = "air",
    level: int = 1,
) -> GlobalDetailResult:
    """Full chain: global beam loads -> nominal stress -> SCF -> hot spot -> S-N life.

    The nominal stress from beam theory is multiplied by the FEM-computed SCF
    (both toe and hot spot variants) to obtain the local stress range, which is
    then converted to fatigue life via the S-N curve.
    """
    sigma_nom = beam_nominal_stress(axial_force_n, bending_moment_nm, section)
    if sigma_nom <= 0.0:
        raise ValueError("nominal stress must be tensile (positive) for fatigue assessment")

    solution = solve_weld_detail(detail, level=level)
    lin = hot_spot_stress(solution, detail.plate_thickness_m, method="linear")
    quad = hot_spot_stress(solution, detail.plate_thickness_m, method="quadratic")

    sigma_hot_lin = lin.stress_ratio * sigma_nom
    sigma_hot_quad = quad.stress_ratio * sigma_nom

    life_lin = fatigue_life(sigma_hot_lin, environment=environment, thickness_m=detail.plate_thickness_m)
    life_quad = fatigue_life(sigma_hot_quad, environment=environment, thickness_m=detail.plate_thickness_m)

    return GlobalDetailResult(
        axial_force_n=axial_force_n,
        bending_moment_nm=bending_moment_nm,
        section=section,
        sigma_nom_pa=sigma_nom,
        detail=detail,
        scf_toe=solution.kt_toe,
        scf_hotspot_linear=lin.stress_ratio,
        scf_hotspot_quadratic=quad.stress_ratio,
        sigma_hot_linear_pa=sigma_hot_lin,
        sigma_hot_quadratic_pa=sigma_hot_quad,
        life_linear_cycles=life_lin,
        life_quadratic_cycles=life_quad,
        environment=environment,
    )


def verify_beam_detail_chain() -> dict[str, object]:
    """Machine verification of the global-to-detail fatigue chain.

    Checks:
    1. Nominal stress scaling: doubling the load doubles sigma_nom.
    2. SCF independence: SCF is independent of the load magnitude.
    3. Life power law: doubling the stress range reduces life by 2^m.
       For the default geometry, sigma_hot ~14 MPa gives life > 1e7 cycles,
       so we're in the m2=5 regime and the ratio should be ~32.
    4. Consistency: sigma_hot = SCF * sigma_nom.
    """
    # A typical hull girder cross-section (box girder approximation)
    # Plate thickness t=20mm, depth D=2m, width B=10m
    t = 0.020
    D = 2.0
    B = 10.0
    area = 2 * t * (B + D)  # simplified: top + bottom + 2 sides
    I = (B * D**3 - (B - 2 * t) * (D - 2 * t)**3) / 12.0
    section = BeamSection(area_m2=area, I_m4=I, c_m=D / 2.0)

    # Global loads - chosen to give sigma_nom ~14 MPa
    N = 1.0e6  # 1 MN axial
    M = 5.0e6  # 5 MNm bending

    sigma_nom = beam_nominal_stress(N, M, section)

    # Detail: T-joint with t=20mm
    detail = WeldDetail(
        joint_type="t_fillet",
        plate_thickness_m=t,
        toe_radius_m=0.001,
        attachment_thickness_m=t,
        attachment_height_m=1.2 * t,
        weld_leg_m=0.7 * t,
    )

    result = global_to_detail_fatigue(N, M, section, detail, level=1)

    # Determine expected power law exponent based on life regime
    knee_cycles = ENVIRONMENTS["air"].curve.knee_cycles
    life_1x = result.life_quadratic_cycles
    if life_1x <= knee_cycles:
        m_eff = 3.0
    else:
        m_eff = 5.0
    expected_ratio = 2.0 ** m_eff

    result_2 = global_to_detail_fatigue(2 * N, 2 * M, section, detail, level=1)
    actual_ratio = result.life_quadratic_cycles / result_2.life_quadratic_cycles

    # Verification checks
    checks = {}

    # 1. Nominal stress scaling
    sigma_nom_2 = beam_nominal_stress(2 * N, 2 * M, section)
    checks["nominal_stress_scaling"] = {
        "sigma_nom": sigma_nom,
        "sigma_nom_2x": sigma_nom_2,
        "ratio": sigma_nom_2 / sigma_nom,
        "expected": 2.0,
        "pass": abs(sigma_nom_2 / sigma_nom - 2.0) < 1e-10,
    }

    # 2. SCF independence
    checks["scf_independence"] = {
        "scf_toe_1x": result.scf_toe,
        "scf_toe_2x": result_2.scf_toe,
        "diff": abs(result.scf_toe - result_2.scf_toe),
        "pass": abs(result.scf_toe - result_2.scf_toe) < 1e-10,
    }

    # 3. Life power law
    checks["life_power_law"] = {
        "life_1x_linear": result.life_linear_cycles,
        "life_2x_linear": result_2.life_linear_cycles,
        "ratio": actual_ratio,
        "expected": expected_ratio,
        "effective_slope": m_eff,
        "sigma_hot_pa": result.sigma_hot_quadratic_pa,
        "knee_cycles": knee_cycles,
        "life_1x_cycles": life_1x,
        "pass": abs(actual_ratio - expected_ratio) / expected_ratio <= 0.02,
    }

    # 4. Consistency: sigma_hot = SCF * sigma_nom
    checks["consistency"] = {
        "sigma_nom": sigma_nom,
        "scf_hotspot_quad": result.scf_hotspot_quadratic,
        "sigma_hot_computed": result.sigma_hot_quadratic_pa,
        "sigma_hot_expected": result.scf_hotspot_quadratic * sigma_nom,
        "pass": abs(result.sigma_hot_quadratic_pa - result.scf_hotspot_quadratic * sigma_nom)
        / result.sigma_hot_quadratic_pa < 1e-10,
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {"checks": checks, "all_pass": all_pass, "result": result}
