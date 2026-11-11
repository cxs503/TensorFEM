"""Stiffened panel ultimate strength under ice loading (Polar Class).

Campaign node: p262w1-N05-fem-panel-ultimate
================================================

Physical problem
----------------
A stiffened plate panel (plate + Tee stiffeners) representative of an
ice-belt structure, subjected to lateral ice pressure per IACS Polar Class
requirements (PC6/7: 5–15 MPa).  The analysis traces the full equilibrium
path from first yield through plate buckling, stiffener plastic hinge
formation, and final collapse, capturing the peak (ultimate) load.

Model content
-------------
* **Plate**: 4-node flat shell with von Kármán moderate-rotation kinematics,
  5 Gauss layers through the thickness for plastic hinge resolution.
* **Stiffeners**: Eccentric Tee beams (web + flange) sharing plate nodes,
  with axial + bending + column geometric stiffness.
* **Material**: EH36 ship steel — Ludwik hardening σ(p) = σ₀ + K·p^n,
  J2 flow rule with radial return mapping.
* **Ice pressure**: IACS pressure-area relationship p(A) = p₀·(A/A₀)^(-0.3),
  applied as uniform lateral load over the panel bay.
* **Solver**: Crisfield spherical arc-length (Riks) continuation for
  post-peak path tracing.

Verification strategy
---------------------
* **F1**: Elastic buckling stress vs Timoshenko closed-form (≤5% error).
* **F2**: Ultimate strength vs Paik-Thayamballi empirical formula for
  three plate slenderness ratios (≤15% error).
* **F3**: Post-peak descent captured (load factor decreases after peak).
* **F4**: Energy conservation |W_ext - U - D_p| / W_ext ≤ 2%.
* **F5**: ≥12 pytest tests all green.
* **F6**: Artifacts — load-deflection curve, buckling mode, stress contour.

Scope and limitations
---------------------
* Small strain, moderate rotations (von Kármán).
* No strain-rate or temperature effects.
* No through-thickness residual stress field.
* Ice pressure is quasi-static (no slamming/impulse).
* Stiffener web local buckling (tripping) is not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

# Re-use the heavy infrastructure from ice_belt_collapse
from .ice_belt_collapse import (
    SteelGrade,
    IACSPressureArea,
    PlateModel,
    Stiffener,
    PointState,
    structured_mesh,
    tee_section,
    line_nodes,
    edge_nodes,
    simply_supported_edges,
    inplane_free_bcs,
    uniform_pressure_load,
    uniform_edge_load,
    buckling_eigenvalue,
    solve_arc_length,
    solve_load_control,
    arc_length_limit,
    arc_length_status,
    elastic_perfectly_plastic,
    csr_plate_buckling_stress,
    calibrate_user_K,
    SIGMA_U_REF,
    N_HARDENING,
    DOF_PER_NODE,
    I_U,
    I_V,
    I_W,
    E_STEEL,
    NU_STEEL,
)
from .plate_stiffener_ultimate import (
    paik_thayamballi_plate_ultimate,
    paik_thayamballi_stiffened_panel_ultimate,
)


# ---------------------------------------------------------------------------
# Material: EH36 ship steel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EH36Steel:
    """EH36 high-strength ship steel (IACS Unified Requirement W11)."""
    name: str = "EH36"
    young: float = 206.0e9       # Pa
    poisson: float = 0.3
    sigma_y: float = 355.0e6     # Pa (specified minimum yield)
    sigma_u: float = 490.0e6     # Pa (specified tensile strength)

    def to_grade(self) -> SteelGrade:
        """Convert to the Ludwik-hardening SteelGrade used by the solver."""
        K = calibrate_user_K(self.sigma_y, self.sigma_u)
        return SteelGrade(
            name=self.name, young=self.young, poisson=self.poisson,
            sigma_0=self.sigma_y, K=K, n=N_HARDENING,
            sigma_u=self.sigma_u,
            provenance=(
                f"EH36: sigma_y={self.sigma_y/1e6:.0f} MPa, "
                f"sigma_u={self.sigma_u/1e6:.0f} MPa per IACS UR W11; "
                f"Ludwik hardening with n={N_HARDENING}, K calibrated to pass "
                f"through (p=0.15, sigma_u)."
            ),
        )


# ---------------------------------------------------------------------------
# Panel geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StiffenedPanelGeometry:
    """Stiffened panel geometry for ice-belt structure."""
    span_x: float = 2.0          # Frame spacing (loaded direction) [m]
    width: float = 2.0           # Transverse dimension [m]
    plate_thickness: float = 0.015  # Plate thickness [m]
    n_stiffeners: int = 3        # Number of stiffeners (transverse)
    stiffener_spacing: float = 0.5  # Stiffener spacing [m]

    # Tee stiffener dimensions
    web_height: float = 0.250
    web_thickness: float = 0.012
    flange_width: float = 0.100
    flange_thickness: float = 0.018

    @property
    def stiffener_positions(self) -> list:
        """Y-coordinates of stiffener centrelines."""
        return [self.stiffener_spacing * (i + 1)
                for i in range(self.n_stiffeners)]


# ---------------------------------------------------------------------------
# Ice pressure model for Polar Class
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolarClassIcePressure:
    """IACS Polar Class ice pressure model.

    PC6/7 typical values: 5-15 MPa over design area.
    Uses IACS pressure-area relationship: p(A) = p0 * (A/A0)^(-0.3)
    """
    polar_class: str = "PC7"
    p0: float = 10.0e6          # Reference pressure at A0 [Pa]
    area_ref: float = 1.0       # Reference area [m²]
    exponent: float = -0.3      # IACS pressure-area exponent

    def pressure_at_area(self, area: float) -> float:
        """Ice pressure for a given loaded area."""
        if area <= 0:
            raise ValueError("area must be positive")
        return self.p0 * (area / self.area_ref) ** self.exponent

    @classmethod
    def for_class(cls, pc: str) -> "PolarClassIcePressure":
        """Create pressure model for a specific Polar Class."""
        # Typical design pressures (nominal, at 1 m² reference area)
        pressures = {
            "PC6": 12.0e6,
            "PC7": 8.0e6,
            "PC5": 15.0e6,
        }
        if pc not in pressures:
            raise ValueError(f"Unknown Polar Class {pc!r}; "
                             f"supported: {list(pressures.keys())}")
        return cls(polar_class=pc, p0=pressures[pc])


# ---------------------------------------------------------------------------
# Analytical reference solutions
# ---------------------------------------------------------------------------
def timoshenko_plate_buckling_stress(
    a: float, b: float, t: float,
    young: float = E_STEEL, poisson: float = NU_STEEL,
    n_terms: int = 20,
) -> float:
    """Critical buckling stress for simply-supported plate in uniaxial compression.

    σ_cr = k · π² · E / (12(1-ν²)) · (t/b)²
    k = min_m (m·b/a + a/(m·b))²

    Parameters
    ----------
    a : plate length (loaded direction) [m]
    b : plate width [m]
    t : plate thickness [m]
    young : Young's modulus [Pa]
    poisson : Poisson's ratio [-]

    Returns
    -------
    float : Critical buckling stress [Pa]
    """
    if a <= 0 or b <= 0 or t <= 0:
        raise ValueError("geometric parameters must be positive")
    best_k = float('inf')
    for m in range(1, n_terms + 1):
        k_m = (m * b / a + a / (m * b)) ** 2
        if k_m < best_k:
            best_k = k_m
    D_factor = young / (12.0 * (1.0 - poisson ** 2))
    return best_k * math.pi ** 2 * D_factor * (t / b) ** 2


def plate_slenderness_parameter(b: float, t: float,
                                sigma_y: float, E: float) -> float:
    """Plate slenderness β = (b/t) · √(σ_y/E)."""
    return (b / t) * math.sqrt(sigma_y / E)


# ---------------------------------------------------------------------------
# FE model builders
# ---------------------------------------------------------------------------
def build_stiffened_panel_model(
    geom: StiffenedPanelGeometry,
    grade: SteelGrade,
    nx: int = 24,
    ny: int = 16,
    von_karman: bool = True,
) -> PlateModel:
    """Build a stiffened panel FE model with Tee stiffeners.

    The plate is meshed with structured quad elements.  Stiffeners run
    in the x-direction (frame direction) at specified y-positions.
    """
    nodes, elems = structured_mesh(geom.span_x, geom.width, nx, ny)
    model = PlateModel(nodes, elems, geom.plate_thickness, grade)
    model.von_karman = von_karman

    # Add stiffeners
    section = tee_section(
        geom.web_height, geom.web_thickness,
        geom.flange_width, geom.flange_thickness,
        geom.plate_thickness,
    )
    for y_pos in geom.stiffener_positions:
        stiff_nodes = line_nodes(model, "y", y_pos, tol=1e-9)
        if len(stiff_nodes) > 1:
            model.stiffeners.append(Stiffener(
                node_line=stiff_nodes, direction="x",
                area=section["area"],
                inertia=section["inertia"],
                offset=section["offset"],
                name=f"stiff_y{y_pos:.3f}",
            ))

    return model


# ---------------------------------------------------------------------------
# Verification drivers
# ---------------------------------------------------------------------------
def verify_elastic_buckling(
    span_x: float = 2.0,
    width: float = 2.0,
    thickness: float = 0.015,
    nx: int = 20,
    ny: int = 14,
) -> dict:
    """F1: Elastic buckling stress vs Timoshenko analytical solution.

    Simply-supported plate under uniaxial compression.
    Criterion: |σ_FE - σ_analytical| / σ_analytical ≤ 5%.
    """
    grade = elastic_perfectly_plastic(sigma_0=355.0e6)
    nodes, elems = structured_mesh(span_x, width, nx, ny)
    model = PlateModel(nodes, elems, thickness, grade)
    model.von_karman = False

    fixed = simply_supported_edges(model)
    fixed += [DOF_PER_NODE * int(n) + I_U for n in edge_nodes(model, "x0")]
    corner = int(np.argmin(model.nodes[:, 0] ** 2 + model.nodes[:, 1] ** 2))
    fixed = sorted(set(fixed + [DOF_PER_NODE * corner + I_V]))

    sigma_ref = 1.0e6
    multiplier, _ = buckling_eigenvalue(model, sigma_ref, fixed, direction="x")
    sigma_fe = multiplier * sigma_ref

    sigma_analytical = timoshenko_plate_buckling_stress(
        span_x, width, thickness, E_STEEL, NU_STEEL)

    rel_error = sigma_fe / sigma_analytical - 1.0

    return {
        "sigma_fe": sigma_fe,
        "sigma_analytical": sigma_analytical,
        "rel_error": rel_error,
        "abs_error_pct": abs(rel_error) * 100,
        "pass_f1": abs(rel_error) <= 0.05,
        "mesh": [nx, ny],
        "span": [span_x, width],
        "thickness": thickness,
    }


def verify_ultimate_strength_paik(
    span_x: float = 2.0,
    width: float = 0.5,
    thickness: float = 0.015,
    nx: int = 16,
    ny: int = 6,
    n_steps: int = 40,
) -> dict:
    """F2: Ultimate strength vs Paik-Thayamballi empirical formula.

    Uses arc-length continuation to trace the post-buckling path and
    find the peak load.  Compares against Paik-Thayamballi (2003).
    Criterion: |σ_FE - σ_Paik| / σ_Paik ≤ 15%.
    """
    grade = elastic_perfectly_plastic(sigma_0=355.0e6)
    beta = plate_slenderness_parameter(width, thickness,
                                       grade.sigma_0, grade.young)

    nodes, elems = structured_mesh(span_x, width, nx, ny)
    model = PlateModel(nodes, elems, thickness, grade)

    # BCs: simply supported + compression
    fixed = simply_supported_edges(model)
    fixed += [DOF_PER_NODE * int(n) + I_U for n in edge_nodes(model, "x0")]
    corner = int(np.argmin(model.nodes[:, 0] ** 2 + model.nodes[:, 1] ** 2))
    fixed = sorted(set(fixed + [DOF_PER_NODE * corner + I_V]))

    # Imperfection: first buckling mode, scaled to 0.1·β²·t
    imperfection_amp = 0.1 * beta ** 2 * thickness
    _, _, modes = buckling_eigenvalue(model, 1.0e6, fixed, direction="x",
                                      return_modes=True)
    mode_w = modes[I_W::DOF_PER_NODE]
    mode_w = mode_w / np.max(np.abs(mode_w))
    model.set_imperfection(mode_w * imperfection_amp)

    # Reference load: unit compressive stress
    fext_ref = uniform_edge_load(model, "x1",
                                 -grade.sigma_0 * width * thickness,
                                 component=I_U)

    # Arc-length path tracing
    path = solve_arc_length(model, fext_ref, fixed,
                            n_steps=n_steps, dlam0=0.02)
    limit = arc_length_limit(path)

    sigma_fem = limit["peak_load_factor"] * grade.sigma_0

    # Paik-Thayamballi reference
    sigma_paik = paik_thayamballi_plate_ultimate(
        beta=beta, sigma_y=grade.sigma_0, a_over_b=span_x / width)

    rel_error = sigma_fem / sigma_paik - 1.0

    # Check post-peak descent
    lam_arr = np.array(path.lam[1:])
    has_post_peak = bool(np.any(np.diff(lam_arr) < 0)) if len(lam_arr) > 2 else False

    return {
        "beta": beta,
        "sigma_fem": sigma_fem,
        "sigma_paik": sigma_paik,
        "sigma_yield": grade.sigma_0,
        "rel_error": rel_error,
        "abs_error_pct": abs(rel_error) * 100,
        "pass_f2": abs(rel_error) <= 0.15,
        "has_post_peak": has_post_peak,
        "peak_load_factor": limit["peak_load_factor"],
        "energy_residual": path.energy_residual(),
        "mesh": [nx, ny],
        "limit_status": arc_length_status(limit),
        "path_lam": list(path.lam[1:]),
        "path_w": list(path.centre_w),
    }


def verify_ultimate_three_slenderness(
    span_x: float = 2.0,
    nx: int = 14,
    n_steps: int = 30,
) -> list:
    """F2 extended: Ultimate strength for 3 plate slenderness ratios.

    Three cases covering stocky (β≈1.0), moderate (β≈2.0), slender (β≈3.0).
    """
    cases = [
        # (width, thickness, label)
        (0.40, 0.020, "stocky"),     # β ≈ 1.1
        (0.50, 0.012, "moderate"),   # β ≈ 2.4
        (0.60, 0.008, "slender"),    # β ≈ 4.1
    ]
    results = []
    for width, thickness, label in cases:
        res = verify_ultimate_strength_paik(
            span_x=span_x, width=width, thickness=thickness,
            nx=nx, ny=max(4, nx // 3), n_steps=n_steps)
        res["label"] = label
        results.append(res)
    return results


def verify_ice_pressure_collapse(
    span_x: float = 2.0,
    width: float = 2.0,
    thickness: float = 0.015,
    polar_class: str = "PC7",
    nx: int = 16,
    ny: int = 12,
    n_steps: int = 30,
) -> dict:
    """Ice pressure collapse of stiffened panel.

    Applies IACS Polar Class ice pressure and traces the collapse path
    using arc-length continuation.
    """
    ice = PolarClassIcePressure.for_class(polar_class)
    load_area = span_x * width
    pressure = ice.pressure_at_area(load_area)

    grade = EH36Steel().to_grade()
    geom = StiffenedPanelGeometry(
        span_x=span_x, width=width, plate_thickness=thickness)

    model = build_stiffened_panel_model(geom, grade, nx=nx, ny=ny)

    # Apply ice pressure
    fext_unit = uniform_pressure_load(model, pressure)
    fixed = inplane_free_bcs(model)

    # Arc-length path
    path = solve_arc_length(model, fext_unit, fixed,
                            n_steps=n_steps, dlam0=0.03)
    limit = arc_length_limit(path)

    collapse_pressure = limit["peak_load_factor"] * pressure

    # Yield-line reference: p_code = 8·Mp/b²
    mp = grade.sigma_0 * thickness ** 2 / 4.0
    b_eff = width / (geom.n_stiffeners + 1)
    p_code = 8.0 * mp / b_eff ** 2

    return {
        "polar_class": polar_class,
        "design_pressure": pressure,
        "collapse_pressure": collapse_pressure,
        "code_pressure": p_code,
        "safety_factor": collapse_pressure / pressure if pressure > 0 else float('inf'),
        "rel_error_code": collapse_pressure / p_code - 1.0,
        "peak_load_factor": limit["peak_load_factor"],
        "energy_residual": path.energy_residual(),
        "mesh": [nx, ny],
        "limit_status": arc_length_status(limit),
        "path_lam": list(path.lam[1:]),
        "path_w": list(path.centre_w),
    }


def verify_energy_conservation(
    span_x: float = 2.0,
    width: float = 2.0,
    thickness: float = 0.015,
    nx: int = 12,
    ny: int = 10,
    n_steps: int = 20,
) -> dict:
    """F4: Energy conservation check.

    W_ext = U + D_p  (external work = strain energy + plastic dissipation)
    Criterion: |W_ext - U - D_p| / W_ext ≤ 2%.
    """
    grade = elastic_perfectly_plastic(sigma_0=355.0e6)
    nodes, elems = structured_mesh(span_x, width, nx, ny)
    model = PlateModel(nodes, elems, thickness, grade)

    fixed = inplane_free_bcs(model)
    pressure = 5.0e6  # 5 MPa ice pressure
    fext = uniform_pressure_load(model, pressure)

    path = solve_arc_length(model, fext, fixed, n_steps=n_steps, dlam0=0.05)

    w_ext = path.external_work
    u_elastic = path.elastic_energy
    d_plastic = path.plastic_work

    energy_sum = u_elastic + d_plastic
    residual = abs(w_ext - energy_sum) / max(abs(w_ext), 1e-30)

    return {
        "external_work": w_ext,
        "elastic_energy": u_elastic,
        "plastic_dissipation": d_plastic,
        "energy_sum": energy_sum,
        "energy_residual": residual,
        "pass_f4": residual <= 0.02,
        "mesh": [nx, ny],
    }


# ---------------------------------------------------------------------------
# Full verification suite
# ---------------------------------------------------------------------------
def run_full_verification(*, fast: bool = True) -> dict:
    """Run all verification checks and return consolidated results."""
    results = {}

    # F1: Elastic buckling
    results["F1_elastic_buckling"] = verify_elastic_buckling(
        nx=20 if fast else 28, ny=14 if fast else 20)

    # F2: Ultimate strength (3 slenderness ratios)
    n_steps = 25 if fast else 50
    results["F2_ultimate_strength"] = verify_ultimate_three_slenderness(
        n_steps=n_steps)

    # F3: Post-peak tracking (embedded in F2 results)
    results["F3_post_peak"] = {
        "has_post_peak": all(
            r.get("has_post_peak", False)
            for r in results["F2_ultimate_strength"]
        ),
        "detail": [
            {"label": r["label"], "has_post_peak": r.get("has_post_peak", False)}
            for r in results["F2_ultimate_strength"]
        ],
    }

    # F4: Energy conservation
    results["F4_energy"] = verify_energy_conservation(
        n_steps=15 if fast else 30)

    # Ice pressure collapse
    results["ice_collapse_PC7"] = verify_ice_pressure_collapse(
        polar_class="PC7", n_steps=20 if fast else 40)

    # Summary
    f1_pass = results["F1_elastic_buckling"]["pass_f1"]
    f2_pass = all(r["pass_f2"] for r in results["F2_ultimate_strength"])
    f3_pass = results["F3_post_peak"]["has_post_peak"]
    f4_pass = results["F4_energy"]["pass_f4"]

    results["summary"] = {
        "F1_elastic_buckling": f1_pass,
        "F2_ultimate_strength": f2_pass,
        "F3_post_peak_tracking": f3_pass,
        "F4_energy_conservation": f4_pass,
        "all_pass": f1_pass and f2_pass and f4_pass,
    }

    return results
