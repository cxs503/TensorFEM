"""Structural fatigue detail verification: weld toe K_t, hot-spot stress, S-N fatigue.

This module implements the analytical-experimental closed loop for structural
fatigue assessment of welded joints:

1. **Weld toe geometry** → parametric T-joint / butt-joint models with
   toe radius r = 0.5–2 mm.
2. **K_t analytical anchors**:
   - T-joint: Nii (1974) formula
   - Butt joint: Maddox (1975) formula
3. **Hot-spot stress extrapolation**:
   - Linear: 0.4t / 0.9t (IIW)
   - Quadratic: 0.4t / 0.9t / 1.4t (IIW)
4. **S-N curve**: DNV-RP-C203 dual-slope (m1=3, m2=5)
5. **Mesh convergence**: GCI-based verification

References:
  - Nii et al. (1974): Stress concentration at welded T-joints.
  - Maddox (1975): Fatigue strength of welded joints, IIW Doc. XIII-845-75.
  - DNV-RP-C203 (2021): Fatigue design of offshore steel structures.
  - IIW (2016): Recommendations for Fatigue Design of Welded Joints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .weld_fatigue_ice import WeldGeometry, toe_stress_concentration

__all__ = [
    "nii_t_joint_kt",
    "maddox_butt_joint_kt",
    "HotSpotExtrapolation",
    "hotspot_extrapolation_iiw",
    "DNVDualSlopeSN",
    "MeshConvergenceResult",
    "mesh_convergence_study",
    "FatigueDetailVerification",
    "run_fatigue_detail_verification",
]


# ---------------------------------------------------------------------------
# Analytical K_t formulas
# ---------------------------------------------------------------------------

def nii_t_joint_kt(
    plate_thickness_m: float,
    weld_leg_m: float,
    toe_radius_m: float,
) -> float:
    """Nii (1974) formula for T-joint stress concentration factor.

    For non-load-carrying fillet welds (T-joints):

        K_t = 1 + 1.24 × (a/t)^0.5 × (t/r)^0.25

    where:
        a = weld leg size (m)
        t = plate thickness (m)
        r = weld toe radius (m)

    Valid range: 0.3 ≤ a/t ≤ 1.0, 5 ≤ t/r ≤ 100.

    Parameters
    ----------
    plate_thickness_m : float
        Plate thickness t (m).
    weld_leg_m : float
        Weld leg size a (m).
    toe_radius_m : float
        Weld toe radius r (m).

    Returns
    -------
    float
        Stress concentration factor K_t.
    """
    t = float(plate_thickness_m)
    a = float(weld_leg_m)
    r = float(toe_radius_m)

    if t <= 0 or a <= 0 or r <= 0:
        raise ValueError("All dimensions must be positive")

    a_over_t = a / t
    t_over_r = t / r

    # Nii formula
    kt = 1.0 + 1.24 * math.sqrt(a_over_t) * (t_over_r ** 0.25)

    return kt


def maddox_butt_joint_kt(
    plate_thickness_m: float,
    reinforcement_height_m: float,
    toe_radius_m: float,
) -> float:
    """Maddox (1975) formula for butt joint stress concentration factor.

    For full-penetration butt welds with cap reinforcement:

        K_t = 1 + 0.18 × (h_r/t) × (t/r)^0.5

    where:
        h_r = reinforcement height (m)
        t = plate thickness (m)
        r = weld toe radius (m)

    Valid range: 0 ≤ h_r/t ≤ 0.25, 5 ≤ t/r ≤ 100.

    Parameters
    ----------
    plate_thickness_m : float
        Plate thickness t (m).
    reinforcement_height_m : float
        Reinforcement height h_r (m).
    toe_radius_m : float
        Weld toe radius r (m).

    Returns
    -------
    float
        Stress concentration factor K_t.
    """
    t = float(plate_thickness_m)
    h_r = float(reinforcement_height_m)
    r = float(toe_radius_m)

    if t <= 0 or r <= 0:
        raise ValueError("Plate thickness and toe radius must be positive")
    if h_r < 0:
        raise ValueError("Reinforcement height must be non-negative")

    h_over_t = h_r / t
    t_over_r = t / r

    # Maddox formula
    kt = 1.0 + 0.18 * h_over_t * math.sqrt(t_over_r)

    return kt


# ---------------------------------------------------------------------------
# Hot-spot stress extrapolation methods
# ---------------------------------------------------------------------------

@dataclass
class HotSpotExtrapolation:
    """Hot-spot stress extrapolation results.

    Attributes
    ----------
    sigma_hot_linear : float
        Hot-spot stress from linear extrapolation (0.4t, 0.9t).
    sigma_hot_quadratic : float
        Hot-spot stress from quadratic extrapolation (0.4t, 0.9t, 1.4t).
    sigma_04t : float
        Surface stress at 0.4t from weld toe.
    sigma_09t : float
        Surface stress at 0.9t from weld toe.
    sigma_14t : float
        Surface stress at 1.4t from weld toe.
    method : str
        Extrapolation method used.
    """
    sigma_hot_linear: float
    sigma_hot_quadratic: float
    sigma_04t: float
    sigma_09t: float
    sigma_14t: float
    method: str = "iiw_linear_quadratic"


def hotspot_extrapolation_iiw(
    surface_distances_m: Sequence[float],
    surface_stresses_pa: Sequence[float],
    plate_thickness_m: float,
) -> HotSpotExtrapolation:
    """IIW hot-spot stress extrapolation using linear and quadratic methods.

    Linear extrapolation (0.4t, 0.9t):
        σ_hot = (0.9t × σ(0.4t) - 0.4t × σ(0.9t)) / (0.9t - 0.4t)

    Quadratic extrapolation (0.4t, 0.9t, 1.4t):
        σ_hot = Lagrange interpolation at x=0

    Parameters
    ----------
    surface_distances_m : sequence of float
        Distances from weld toe along plate surface (m).
    surface_stresses_pa : sequence of float
        Surface stresses at each distance (Pa).
    plate_thickness_m : float
        Plate thickness t (m).

    Returns
    -------
    HotSpotExtrapolation
    """
    t = float(plate_thickness_m)
    x = np.asarray(surface_distances_m, dtype=float)
    y = np.asarray(surface_stresses_pa, dtype=float)

    if x.shape != y.shape or x.size < 3:
        raise ValueError("Need at least 3 matching distance/stress points")

    # Interpolate at 0.4t, 0.9t, 1.4t
    x_04t = 0.4 * t
    x_09t = 0.9 * t
    x_14t = 1.4 * t

    sigma_04t = float(np.interp(x_04t, x, y))
    sigma_09t = float(np.interp(x_09t, x, y))
    sigma_14t = float(np.interp(x_14t, x, y))

    # Linear extrapolation: σ_hot = (0.9t × σ(0.4t) - 0.4t × σ(0.9t)) / (0.9t - 0.4t)
    span = x_09t - x_04t
    sigma_hot_linear = (x_09t * sigma_04t - x_04t * sigma_09t) / span

    # Quadratic extrapolation: Lagrange at x=0
    # L_0(0) = (0 - x_09t)(0 - x_14t) / ((x_04t - x_09t)(x_04t - x_14t))
    # L_1(0) = (0 - x_04t)(0 - x_14t) / ((x_09t - x_04t)(x_09t - x_14t))
    # L_2(0) = (0 - x_04t)(0 - x_09t) / ((x_14t - x_04t)(x_14t - x_09t))
    L0 = (0 - x_09t) * (0 - x_14t) / ((x_04t - x_09t) * (x_04t - x_14t))
    L1 = (0 - x_04t) * (0 - x_14t) / ((x_09t - x_04t) * (x_09t - x_14t))
    L2 = (0 - x_04t) * (0 - x_09t) / ((x_14t - x_04t) * (x_14t - x_09t))

    sigma_hot_quadratic = L0 * sigma_04t + L1 * sigma_09t + L2 * sigma_14t

    return HotSpotExtrapolation(
        sigma_hot_linear=sigma_hot_linear,
        sigma_hot_quadratic=sigma_hot_quadratic,
        sigma_04t=sigma_04t,
        sigma_09t=sigma_09t,
        sigma_14t=sigma_14t,
        method="iiw_linear_quadratic",
    )


# ---------------------------------------------------------------------------
# DNV-RP-C203 dual-slope S-N curve
# ---------------------------------------------------------------------------

@dataclass
class DNVDualSlopeSN:
    """DNV-RP-C203 dual-slope S-N curve (Curve D in air).

    The D curve has two slopes:
      - Below knee (N ≤ 1e7): m1 = 3, log10(a1) = 12.164
      - Above knee (N > 1e7): m2 = 5, log10(a2) = 15.606

    The knee point is at N = 1e7 cycles, corresponding to stress range:
        Δσ_knee = (a1 / 1e7)^(1/m1) = 10^((12.164 - 7) / 3) = 52.64 MPa

    (The value was printed as 169.3 MPa in the first release; that is an
    arithmetic slip -- ``10**((12.164 - 7)/3) = 52.642``, see
    ``docs/fatigue-sn-curve-consistency-audit.md``.)

    Attributes
    ----------
    m1 : float
        First slope (high stress, low cycle).
    m2 : float
        Second slope (low stress, high cycle).
    log_a1 : float
        First intercept log10(a1).
    log_a2 : float
        Second intercept log10(a2).
    n_knee : float
        Knee point cycles.
    """
    m1: float = 3.0
    m2: float = 5.0
    log_a1: float = 12.164
    log_a2: float = 15.606
    n_knee: float = 1e7

    @property
    def stress_knee_pa(self) -> float:
        """Stress range at knee point (Pa)."""
        # Δσ_knee = (a1 / N_knee)^(1/m1)
        a1 = 10 ** self.log_a1
        stress_knee_mpa = (a1 / self.n_knee) ** (1.0 / self.m1)
        return stress_knee_mpa * 1e6  # Convert MPa to Pa

    def cycles_to_failure(self, stress_range_pa: float) -> float:
        """Compute cycles to failure for given stress range.

        Parameters
        ----------
        stress_range_pa : float
            Stress range Δσ (Pa).

        Returns
        -------
        float
            Number of cycles to failure N.
        """
        if stress_range_pa <= 0:
            raise ValueError("Stress range must be positive")

        stress_mpa = stress_range_pa / 1e6
        stress_knee_mpa = self.stress_knee_pa / 1e6

        if stress_mpa >= stress_knee_mpa:
            # Use first slope (m1)
            a1 = 10 ** self.log_a1
            n = a1 / (stress_mpa ** self.m1)
        else:
            # Use second slope (m2)
            a2 = 10 ** self.log_a2
            n = a2 / (stress_mpa ** self.m2)

        return n

    def stress_at_cycles(self, cycles: float) -> float:
        """Compute stress range for given cycles to failure.

        Parameters
        ----------
        cycles : float
            Number of cycles N.

        Returns
        -------
        float
            Stress range Δσ (Pa).
        """
        if cycles <= 0:
            raise ValueError("Cycles must be positive")

        if cycles <= self.n_knee:
            # Use first slope (m1)
            a1 = 10 ** self.log_a1
            stress_mpa = (a1 / cycles) ** (1.0 / self.m1)
        else:
            # Use second slope (m2)
            a2 = 10 ** self.log_a2
            stress_mpa = (a2 / cycles) ** (1.0 / self.m2)

        return stress_mpa * 1e6  # Convert MPa to Pa


# ---------------------------------------------------------------------------
# Mesh convergence study
# ---------------------------------------------------------------------------

@dataclass
class MeshConvergenceResult:
    """Mesh convergence study result.

    Attributes
    ----------
    mesh_levels : list of dict
        Mesh parameters and computed K_t at each level.
    scf_change_percent : float
        Percentage change in K_t between finest two meshes.
    gci_percent : float
        Grid Convergence Index (GCI) percentage.
    converged : bool
        Whether convergence criterion is met (≤2% change, ≤3% GCI).
    """
    mesh_levels: list
    scf_change_percent: float
    gci_percent: float
    converged: bool


def mesh_convergence_study(
    compute_kt_func,
    mesh_parameters: Sequence[dict],
) -> MeshConvergenceResult:
    """Perform mesh convergence study using GCI method.

    Parameters
    ----------
    compute_kt_func : callable
        Function that takes mesh parameters dict and returns K_t.
    mesh_parameters : sequence of dict
        List of mesh parameter sets (coarse to fine).

    Returns
    -------
    MeshConvergenceResult
    """
    if len(mesh_parameters) < 3:
        raise ValueError("Need at least 3 mesh levels for GCI analysis")

    # Compute K_t at each mesh level
    mesh_levels = []
    for params in mesh_parameters:
        kt = compute_kt_func(params)
        mesh_levels.append({
            "params": params,
            "kt": kt,
        })

    # Extract K_t values
    kt_values = [level["kt"] for level in mesh_levels]

    # Compute percentage change between finest two meshes
    kt_fine = kt_values[-1]
    kt_medium = kt_values[-2]
    scf_change_percent = abs(kt_fine - kt_medium) / kt_fine * 100

    # Compute GCI (Grid Convergence Index)
    # GCI = F_s * |(kt_fine - kt_medium) / kt_fine| * 100%
    # where F_s is a safety factor (typically 1.25 for 3 grids)
    # and the refinement ratio r = h_coarse / h_fine
    # For simplicity, assume uniform refinement ratio of 1.5
    r_refinement = 1.5
    p = 2.0  # Order of accuracy (linear elements)
    F_s = 1.25  # Safety factor

    epsilon = abs(kt_fine - kt_medium) / kt_fine
    gci_percent = F_s * epsilon / (r_refinement ** p - 1) * 100

    # Check convergence criterion
    converged = scf_change_percent <= 2.0 and gci_percent <= 3.0

    return MeshConvergenceResult(
        mesh_levels=mesh_levels,
        scf_change_percent=scf_change_percent,
        gci_percent=gci_percent,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Fatigue detail verification report
# ---------------------------------------------------------------------------

@dataclass
class FatigueDetailVerification:
    """Complete fatigue detail verification report.

    Attributes
    ----------
    t_joint_kt_nii : dict
        T-joint K_t comparison (FE vs Nii).
    butt_joint_kt_maddox : dict
        Butt joint K_t comparison (FE vs Maddox).
    hotspot_extrapolation : dict
        Hot-spot stress extrapolation results.
    sn_curve_verification : dict
        S-N curve dual-slope verification.
    mesh_convergence : dict
        Mesh convergence study results.
    fatigue_life : dict
        Fatigue life prediction results.
    all_passed : bool
        Whether all criteria passed.
    summary : str
        Summary string.
    """
    t_joint_kt_nii: dict
    butt_joint_kt_maddox: dict
    hotspot_extrapolation: dict
    sn_curve_verification: dict
    mesh_convergence: dict
    fatigue_life: dict
    all_passed: bool
    summary: str


def sn_curve_verification_report(sn_curve: "DNVDualSlopeSN | None" = None) -> dict:
    """Machine-checkable dual-slope verification of a DNV curve-D style S-N line.

    Three checks, each with an *explicit branch guard* so that a stress range can
    never be used to "verify" a slope branch it does not belong to:

    * m1 check -- ``ds`` and ``2 ds`` both above the knee, ``N(ds)/N(2ds) = 2**m1``;
    * knee check -- the knee stress maps back onto ``n_knee`` cycles;
    * m2 check -- ``ds`` and ``ds/2`` both **strictly below** the knee,
      ``N(ds/2)/N(ds) = 2**m2``.

    The first release hard-coded 100 MPa / 50 MPa for the m2 check.  Both are
    above the 52.64 MPa knee of curve D, so the "m2" check actually measured
    ``N(50 MPa, m=5) / N(100 MPa, m=3) = 8.85`` against an expectation of 32
    (72.3 % error) and returned ``passed=False`` for a curve that is correct.
    """
    sn_curve = DNVDualSlopeSN() if sn_curve is None else sn_curve
    stress_knee = sn_curve.stress_knee_pa

    # Test 1: m1 slope, both ranges above the knee.
    ds1 = 1.6 * stress_knee
    ds2 = 2.0 * ds1
    n1 = sn_curve.cycles_to_failure(ds1)
    n2 = sn_curve.cycles_to_failure(ds2)
    expected_ratio = 2.0 ** sn_curve.m1
    actual_ratio = n1 / n2
    ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio

    # Test 2: knee point round-trip.
    n_at_knee = sn_curve.cycles_to_failure(stress_knee)
    knee_error = abs(n_at_knee - sn_curve.n_knee) / sn_curve.n_knee * 100.0

    # Test 3: m2 slope, both ranges strictly below the knee (branch-guarded).
    ds3 = 0.45 * stress_knee
    ds4 = 0.5 * ds3
    n3 = sn_curve.cycles_to_failure(ds3)
    n4 = sn_curve.cycles_to_failure(ds4)
    expected_ratio_m5 = 2.0 ** sn_curve.m2
    actual_ratio_m5 = n4 / n3
    ratio_error_m5 = abs(actual_ratio_m5 - expected_ratio_m5) / expected_ratio_m5
    branch_guard = bool(ds1 > stress_knee and ds2 > stress_knee
                        and ds3 < stress_knee and ds4 < stress_knee)

    return {
        "m1": sn_curve.m1,
        "m2": sn_curve.m2,
        "log_a1": sn_curve.log_a1,
        "log_a2": sn_curve.log_a2,
        "stress_knee_pa": stress_knee,
        "n_at_knee": n_at_knee,
        "knee_error_percent": knee_error,
        "ds1_pa": ds1,
        "n1": n1,
        "ds2_pa": ds2,
        "n2": n2,
        "ratio_m3_actual": actual_ratio,
        "ratio_m3_expected": expected_ratio,
        "ratio_m3_error_percent": ratio_error * 100.0,
        "ds3_pa": ds3,
        "n3": n3,
        "ds4_pa": ds4,
        "n4": n4,
        "ratio_m5_actual": actual_ratio_m5,
        "ratio_m5_expected": expected_ratio_m5,
        "ratio_m5_error_percent": ratio_error_m5 * 100.0,
        "branch_guard": branch_guard,
        "passed": bool(branch_guard and ratio_error <= 1e-6
                       and knee_error <= 1e-6 and ratio_error_m5 <= 1e-6),
        "criterion": ("S-N dual-slope: branch-guarded ratio error <= 1e-6, "
                      "knee error <= 1e-6 %"),
    }


def run_fatigue_detail_verification(
    plate_thickness_m: float = 0.015,
    weld_leg_m: float = 0.010,
    toe_radius_m: float = 1.0e-3,
    reinforcement_height_m: float = 1.5e-3,
    stress_range_pa: float = 100e6,
    n_xi: int = 48,
    n_eta: int = 28,
) -> FatigueDetailVerification:
    """Run complete fatigue detail verification.

    Parameters
    ----------
    plate_thickness_m : float
        Plate thickness t (m).
    weld_leg_m : float
        Weld leg size a (m) for T-joint.
    toe_radius_m : float
        Weld toe radius r (m).
    reinforcement_height_m : float
        Reinforcement height h_r (m) for butt joint.
    stress_range_pa : float
        Applied stress range for fatigue life (Pa).
    n_xi, n_eta : int
        Mesh density parameters for FE submodel.

    Returns
    -------
    FatigueDetailVerification
    """
    t = plate_thickness_m
    a = weld_leg_m
    r = toe_radius_m
    h_r = reinforcement_height_m

    # ---- (a) T-joint K_t vs Nii formula ----
    kt_nii = nii_t_joint_kt(t, a, r)

    # FE K_t using notch submodel
    # For T-joint, use fillet weld geometry
    # Equivalent notch depth: d = r * sqrt(1 + 1.6*cos(theta))
    # For flank angle 50°, cos(50°) ≈ 0.643, so d ≈ r * sqrt(2.03) ≈ 1.42*r
    flank_angle_deg = 50.0
    theta = math.radians(flank_angle_deg)
    notch_depth = r * math.sqrt(1.0 + 1.6 * math.cos(theta))

    t_joint_weld_geom = WeldGeometry(
        weld_type="fillet",
        plate_thickness_m=t,
        reinforcement_m=a,  # weld leg size
        toe_radius_m=r,
        flank_angle_deg=flank_angle_deg,
        fillet_notch_fraction=0.15,
    )

    t_joint_submodel = toe_stress_concentration(
        t_joint_weld_geom, n_xi=n_xi, n_eta=n_eta
    )
    kt_fe_t_joint = t_joint_submodel.kt_toe

    t_joint_error = abs(kt_fe_t_joint - kt_nii) / kt_nii * 100
    t_joint_passed = t_joint_error <= 3.0

    t_joint_kt_nii = {
        "plate_thickness_m": t,
        "weld_leg_m": a,
        "toe_radius_m": r,
        "flank_angle_deg": flank_angle_deg,
        "notch_depth_m": notch_depth,
        "kt_nii": kt_nii,
        "kt_fe": kt_fe_t_joint,
        "error_percent": t_joint_error,
        "passed": t_joint_passed,
        "criterion": "T-joint K_t error ≤ 3% vs Nii formula",
    }

    # ---- (b) Butt joint K_t vs Maddox formula ----
    kt_maddox = maddox_butt_joint_kt(t, h_r, r)

    # FE K_t using notch submodel
    # For butt joint, use butt weld geometry
    # Equivalent notch depth for butt joint: d ≈ h_r (reinforcement height)
    butt_joint_weld_geom = WeldGeometry(
        weld_type="butt",
        plate_thickness_m=t,
        reinforcement_m=h_r,
        toe_radius_m=r,
        flank_angle_deg=30.0,  # Typical butt joint flank angle
        fillet_notch_fraction=0.15,
    )

    butt_joint_submodel = toe_stress_concentration(
        butt_joint_weld_geom, n_xi=n_xi, n_eta=n_eta
    )
    kt_fe_butt_joint = butt_joint_submodel.kt_toe

    butt_joint_error = abs(kt_fe_butt_joint - kt_maddox) / kt_maddox * 100
    butt_joint_passed = butt_joint_error <= 3.0

    butt_joint_kt_maddox = {
        "plate_thickness_m": t,
        "reinforcement_height_m": h_r,
        "toe_radius_m": r,
        "kt_maddox": kt_maddox,
        "kt_fe": kt_fe_butt_joint,
        "error_percent": butt_joint_error,
        "passed": butt_joint_passed,
        "criterion": "Butt joint K_t error ≤ 3% vs Maddox formula",
    }

    # ---- (c) Hot-spot stress extrapolation ----
    # Use surface path from FE submodel
    x_path = np.array(t_joint_submodel.surface_path_m)
    y_path = np.array(t_joint_submodel.surface_stress_ratio)

    hs_result = hotspot_extrapolation_iiw(x_path, y_path, t)

    # Mesh convergence study
    mesh_levels = [
        {"n_xi": 24, "n_eta": 14},
        {"n_xi": 36, "n_eta": 20},
        {"n_xi": 48, "n_eta": 28},
    ]

    def compute_kt_from_mesh(params):
        submodel = toe_stress_concentration(
            t_joint_weld_geom,
            n_xi=params["n_xi"],
            n_eta=params["n_eta"]
        )
        return submodel.kt_toe

    mesh_conv = mesh_convergence_study(compute_kt_from_mesh, mesh_levels)

    hotspot_extrapolation = {
        "sigma_hot_linear": hs_result.sigma_hot_linear,
        "sigma_hot_quadratic": hs_result.sigma_hot_quadratic,
        "sigma_04t": hs_result.sigma_04t,
        "sigma_09t": hs_result.sigma_09t,
        "sigma_14t": hs_result.sigma_14t,
        "linear_vs_quadratic_diff_percent": abs(
            hs_result.sigma_hot_linear - hs_result.sigma_hot_quadratic
        ) / hs_result.sigma_hot_linear * 100,
        "mesh_convergence": {
            "scf_change_percent": mesh_conv.scf_change_percent,
            "gci_percent": mesh_conv.gci_percent,
            "converged": mesh_conv.converged,
        },
        "passed": mesh_conv.scf_change_percent <= 2.0,
        "criterion": "Hot-spot stress mesh convergence ≤ 2%",
    }

    # ---- (d) S-N curve dual-slope verification ----
    sn_curve = DNVDualSlopeSN()
    sn_curve_verification = sn_curve_verification_report(sn_curve)
    stress_knee = sn_curve_verification["stress_knee_pa"]

    # ---- Fatigue life prediction ----
    # Use hot-spot stress for fatigue life
    sigma_hot = hs_result.sigma_hot_linear * stress_range_pa
    n_fatigue = sn_curve.cycles_to_failure(sigma_hot)

    # Reference life (using nominal stress without SCF).
    #
    # The first release hard-coded a passing flag here with the comment "computed
    # correctly if the S-N curve is valid" -- a gate that cannot fail.  It is now
    # a real check: the life is recomputed from the *knee-anchored* power law
    # (an independent route through the same published constants, which exposes a
    # wrong second-slope intercept or a flipped branch) and compared.
    def _knee_anchored_life(ds_pa: float) -> float:
        s_mpa = ds_pa / 1e6
        knee_stress_mpa = stress_knee / 1e6
        n_low = 10.0 ** sn_curve.log_a1 * s_mpa ** (-sn_curve.m1)
        if n_low <= sn_curve.n_knee:
            return n_low
        return sn_curve.n_knee * (knee_stress_mpa / s_mpa) ** sn_curve.m2

    n_reference = sn_curve.cycles_to_failure(stress_range_pa)
    _independent_life = _knee_anchored_life(sigma_hot)
    _life_error = abs(n_fatigue / _independent_life - 1.0)

    # Life reduction due to stress concentration
    life_reduction = n_reference / n_fatigue

    fatigue_life = {
        "stress_range_nominal_pa": stress_range_pa,
        "stress_range_hotspot_pa": sigma_hot,
        "scf_hotspot": hs_result.sigma_hot_linear,
        "cycles_to_failure": n_fatigue,
        "cycles_reference": n_reference,
        "life_reduction_factor": life_reduction,
        "slope_active": sn_curve.m1 if sigma_hot >= stress_knee else sn_curve.m2,
        "independent_life": _independent_life,
        "life_relative_error": _life_error,
        "passed": bool(_life_error <= 1e-9 and math.isfinite(n_fatigue) and n_fatigue > 0.0),
        "criterion": ("Life recomputed from the knee-anchored power law agrees "
                      "with the curve object to <= 1e-9 relative"),
    }

    # ---- Overall pass/fail ----
    criteria_passed = [
        t_joint_passed,
        butt_joint_passed,
        hotspot_extrapolation["passed"],
        sn_curve_verification["passed"],
        fatigue_life["passed"],
    ]

    all_passed = all(criteria_passed)

    summary = (
        f"Fatigue detail verification: {sum(criteria_passed)}/5 criteria passed. "
        f"(a) T-joint K_t: {'PASS' if t_joint_passed else 'FAIL'}, "
        f"(b) Butt joint K_t: {'PASS' if butt_joint_passed else 'FAIL'}, "
        f"(c) Hot-spot convergence: {'PASS' if hotspot_extrapolation['passed'] else 'FAIL'}, "
        f"(d) S-N dual-slope: {'PASS' if sn_curve_verification['passed'] else 'FAIL'}, "
        f"(e) Fatigue life: {'PASS' if fatigue_life['passed'] else 'FAIL'}."
    )

    return FatigueDetailVerification(
        t_joint_kt_nii=t_joint_kt_nii,
        butt_joint_kt_maddox=butt_joint_kt_maddox,
        hotspot_extrapolation=hotspot_extrapolation,
        sn_curve_verification=sn_curve_verification,
        mesh_convergence=mesh_conv,
        fatigue_life=fatigue_life,
        all_passed=all_passed,
        summary=summary,
    )
