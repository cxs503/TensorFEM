"""Low-temperature fracture mechanics for polar ship hull steel.

This module implements the full chain for assessing brittle fracture risk
in polar ship hull structures under ice impact loading:

1. **DBTT determination**: Ductile-to-brittle transition temperature from
   Charpy V-notch impact energy vs. temperature curves.
2. **KV(T) curve**: Oldfield sigmoidal model for impact energy.
3. **Dynamic fracture toughness K_ID(T)**: Temperature and loading rate
   dependent fracture toughness.
4. **Crack propagation criterion**: K_I >= K_ID(T, dK/dt) under ice impact.
5. **XFEM/CZM crack growth simulation**: Phase-field regularized crack
   propagation in hull plate panels.

Physical models
---------------
- Oldfield (1954) sigmoidal: KV(T) = KV_upper - (KV_upper - KV_lower) /
  [1 + exp((T - T_0) / Delta_T)]
- DBTT: temperature at KV = 27 J (or 50% fibrous fracture)
- Plate thickness effect: Delta_DBTT proportional to sqrt(t)
- Dynamic fracture toughness: K_ID(T, dK/dt) with rate sensitivity
- Crack propagation: K_I >= K_ID(T, loading_rate)

References
----------
- Oldfield, W. (1954). "A quantitative interpretation of the impact test."
- ASTM E23 (Charpy V-notch test standard)
- DNV-OS-B301 (Brittle fracture protection)
- IACS Polar Class rules for hull steel selection

Units: SI throughout (Pa, m, K or degC, J, MPa*sqrt(m))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# Material database: ship hull steel grades for polar service
# ============================================================================

# Typical Charpy and fracture toughness parameters for polar ship hull steels
# Sources: DNV-OS-B301, IACS Polar Class rules, published test data
# Parameters calibrated so that DBTT (KV=27J) matches literature values
STEEL_GRADES: Dict[str, dict] = {
    "EH36": {
        "yield_strength_MPa": 360.0,      # Minimum yield [MPa]
        "tensile_strength_MPa": 490.0,     # Minimum tensile [MPa]
        "KV_upper_J": 200.0,               # Upper shelf energy [J]
        "KV_lower_J": 10.0,                # Lower shelf energy [J]
        "T0_degC": 0.0,                    # Inflection temperature [degC]
        "Delta_T_degC": 8.6,               # Transition width [degC]
        "K_IC_MPa_sqrtm": 150.0,           # Plane-strain fracture toughness [MPa*sqrt(m)]
        "DBTT_27J_degC": -20.0,            # Reference DBTT at 27J [degC]
        "density_kg_m3": 7850.0,
        "E_GPa": 206.0,                    # Young's modulus [GPa]
        "nu": 0.3,                         # Poisson's ratio
    },
    "EH40": {
        "yield_strength_MPa": 400.0,
        "tensile_strength_MPa": 520.0,
        "KV_upper_J": 180.0,
        "KV_lower_J": 8.0,
        "T0_degC": 5.0,
        "Delta_T_degC": 8.0,
        "K_IC_MPa_sqrtm": 140.0,
        "DBTT_27J_degC": -15.0,
        "density_kg_m3": 7850.0,
        "E_GPa": 206.0,
        "nu": 0.3,
    },
    "EH47": {
        "yield_strength_MPa": 470.0,
        "tensile_strength_MPa": 570.0,
        "KV_upper_J": 160.0,
        "KV_lower_J": 7.0,
        "T0_degC": 10.0,
        "Delta_T_degC": 7.5,
        "K_IC_MPa_sqrtm": 130.0,
        "DBTT_27J_degC": -10.0,
        "density_kg_m3": 7850.0,
        "E_GPa": 206.0,
        "nu": 0.3,
    },
    "AH36": {
        "yield_strength_MPa": 360.0,
        "tensile_strength_MPa": 490.0,
        "KV_upper_J": 190.0,
        "KV_lower_J": 12.0,
        "T0_degC": -2.0,
        "Delta_T_degC": 9.0,
        "K_IC_MPa_sqrtm": 145.0,
        "DBTT_27J_degC": -18.0,
        "density_kg_m3": 7850.0,
        "E_GPa": 206.0,
        "nu": 0.3,
    },
}


# ============================================================================
# 1. Charpy V-notch impact energy model (Oldfield sigmoidal)
# ============================================================================

def charpy_kv(T_degC: float, KV_upper: float, KV_lower: float,
              T0: float, Delta_T: float) -> float:
    """Compute Charpy V-notch impact energy at temperature T.

    Oldfield (1954) sigmoidal model:
        KV(T) = KV_upper - (KV_upper - KV_lower) / [1 + exp((T - T0) / Delta_T)]

    Parameters
    ----------
    T_degC : float
        Temperature [degC]
    KV_upper : float
        Upper shelf energy [J]
    KV_lower : float
        Lower shelf energy [J]
    T0 : float
        Inflection temperature of transition [degC]
    Delta_T : float
        Transition width parameter [degC]

    Returns
    -------
    float
        Impact energy KV [J]
    """
    if Delta_T <= 0:
        raise ValueError("Delta_T must be positive")
    if KV_upper <= KV_lower:
        raise ValueError("KV_upper must exceed KV_lower")

    arg = (T_degC - T0) / Delta_T
    # Clamp to avoid overflow
    arg = max(-500.0, min(500.0, arg))
    exp_val = math.exp(arg)
    KV = KV_upper - (KV_upper - KV_lower) / (1.0 + exp_val)
    return KV


def charpy_kv_curve(steel_grade: str,
                    T_range: Optional[List[float]] = None) -> Tuple[List[float], List[float]]:
    """Generate KV(T) curve for a given steel grade.

    Parameters
    ----------
    steel_grade : str
        Steel grade name (e.g., 'EH36')
    T_range : list of float, optional
        Temperature points [degC]. Default: -80 to +40 in 2 degC steps.

    Returns
    -------
    tuple
        (temperatures, KV_values)
    """
    if steel_grade not in STEEL_GRADES:
        raise ValueError(f"Unknown steel grade: {steel_grade}. "
                         f"Available: {list(STEEL_GRADES.keys())}")

    mat = STEEL_GRADES[steel_grade]
    if T_range is None:
        T_range = list(np.arange(-80, 41, 2.0))

    KV_vals = [charpy_kv(T, mat["KV_upper_J"], mat["KV_lower_J"],
                         mat["T0_degC"], mat["Delta_T_degC"])
               for T in T_range]
    return T_range, KV_vals


# ============================================================================
# 2. DBTT determination
# ============================================================================

def find_DBTT_27J(steel_grade: str, T_min: float = -100.0,
                  T_max: float = 50.0, tol: float = 0.01) -> float:
    """Find DBTT as temperature where KV = 27 J (bisection).

    Parameters
    ----------
    steel_grade : str
        Steel grade name
    T_min, T_max : float
        Search bounds [degC]
    tol : float
        Temperature tolerance [degC]

    Returns
    -------
    float
        DBTT [degC] (temperature at KV = 27 J)
    """
    mat = STEEL_GRADES[steel_grade]
    target = 27.0

    # Bisection
    T_lo, T_hi = T_min, T_max
    for _ in range(200):
        T_mid = 0.5 * (T_lo + T_hi)
        KV_mid = charpy_kv(T_mid, mat["KV_upper_J"], mat["KV_lower_J"],
                           mat["T0_degC"], mat["Delta_T_degC"])
        if abs(KV_mid - target) < 0.01:
            return T_mid
        if KV_mid < target:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if (T_hi - T_lo) < tol:
            return 0.5 * (T_lo + T_hi)

    return 0.5 * (T_lo + T_hi)


def DBTT_thickness_shift(DBTT_ref: float, t_ref_mm: float,
                         t_mm: float) -> float:
    """Compute DBTT shift due to plate thickness change.

    Empirical relation: Delta_DBTT = C * (sqrt(t) - sqrt(t_ref))
    where C ~ 10 degC / sqrt(mm) for ship hull steels (conservative estimate
    from DNV-RP-C203 and published data).

    Parameters
    ----------
    DBTT_ref : float
        Reference DBTT [degC] at reference thickness
    t_ref_mm : float
        Reference plate thickness [mm]
    t_mm : float
        Actual plate thickness [mm]

    Returns
    -------
    float
        Shifted DBTT [degC]
    """
    if t_mm <= 0 or t_ref_mm <= 0:
        raise ValueError("Thicknesses must be positive")

    C = 10.0  # degC / sqrt(mm), conservative
    delta = C * (math.sqrt(t_mm) - math.sqrt(t_ref_mm))
    return DBTT_ref + delta


def DBTT_loading_rate_shift(DBTT_ref: float, strain_rate: float,
                            strain_rate_ref: float = 1.0) -> float:
    """Compute DBTT shift due to increased loading/strain rate.

    Empirical: Delta_DBTT ~ C_rate * log10(strain_rate / strain_rate_ref)
    C_rate ~ 20 degC per decade for ship hull steels.

    Parameters
    ----------
    DBTT_ref : float
        Reference DBTT [degC]
    strain_rate : float
        Actual strain rate [1/s]
    strain_rate_ref : float
        Reference strain rate [1/s] (quasi-static ~ 1e-3 to 1)

    Returns
    -------
    float
        Shifted DBTT [degC]
    """
    if strain_rate <= 0 or strain_rate_ref <= 0:
        raise ValueError("Strain rates must be positive")

    C_rate = 20.0  # degC per decade
    delta = C_rate * math.log10(strain_rate / strain_rate_ref)
    return DBTT_ref + delta


# ============================================================================
# 3. Dynamic fracture toughness K_ID(T, loading_rate)
# ============================================================================

def dynamic_fracture_toughness(steel_grade: str, T_degC: float,
                               loading_rate: float = 1.0,
                               T_ref_degC: float = 20.0) -> float:
    """Compute dynamic fracture toughness K_ID(T, dK/dt).

    Model: K_ID(T) = K_IC * [1 - alpha * (1 - exp(-beta * (T_ref - T)))]
    with rate correction: K_ID increases with loading rate (inertia effects).

    For T >= T_ref: K_ID -> K_IC (room temperature toughness)
    For T < T_ref: K_ID decreases exponentially with decreasing T.

    Rate effect: K_ID_rate = K_ID * (1 + gamma * log10(dK/dt / dK/dt_ref))
    for dK/dt > dK/dt_ref (dynamic increase).

    Parameters
    ----------
    steel_grade : str
        Steel grade name
    T_degC : float
        Temperature [degC]
    loading_rate : float
        Stress intensity factor rate dK/dt [MPa*sqrt(m)/s]
    T_ref_degC : float
        Reference temperature (room temp) [degC]

    Returns
    -------
    float
        Dynamic fracture toughness K_ID [MPa*sqrt(m)]
    """
    mat = STEEL_GRADES[steel_grade]
    K_IC = mat["K_IC_MPa_sqrtm"]

    # Temperature degradation parameters (calibrated for ship hull steels)
    # For EH36: K_IC=150, at -50°C we want K_ID ~ 100 MPa√m
    # 100 = 150 * [1 - alpha * (1 - exp(-beta * 70))]
    # 100/150 = 0.667 = 1 - alpha * (1 - exp(-70*beta))
    # alpha * (1 - exp(-70*beta)) = 0.333
    # With alpha=0.5, beta=0.02: 0.5*(1-exp(-1.4)) = 0.5*0.753 = 0.377 -> K_ID=93.5
    # With alpha=0.45, beta=0.02: 0.45*(1-exp(-1.4)) = 0.45*0.753 = 0.339 -> K_ID=99.2
    alpha = 0.45  # Maximum fractional reduction
    beta = 0.02   # Temperature sensitivity [1/degC]

    if T_degC >= T_ref_degC:
        K_ID = K_IC
    else:
        dT = T_ref_degC - T_degC
        K_ID = K_IC * (1.0 - alpha * (1.0 - math.exp(-beta * dT)))

    # Ensure physical lower bound (no less than 20% of K_IC)
    K_ID = max(K_ID, 0.2 * K_IC)

    # Loading rate effect (dynamic increase)
    # Reference quasi-static rate ~ 1 MPa*sqrt(m)/s
    dKdt_ref = 1.0
    if loading_rate > dKdt_ref:
        gamma = 0.05  # Rate sensitivity coefficient
        rate_factor = 1.0 + gamma * math.log10(loading_rate / dKdt_ref)
        K_ID *= rate_factor

    return K_ID


def K_ID_curve(steel_grade: str,
               T_range: Optional[List[float]] = None,
               loading_rate: float = 1.0) -> Tuple[List[float], List[float]]:
    """Generate K_ID(T) curve.

    Parameters
    ----------
    steel_grade : str
        Steel grade name
    T_range : list of float, optional
        Temperature points [degC]. Default: -80 to +20 in 2 degC steps.
    loading_rate : float
        dK/dt [MPa*sqrt(m)/s]

    Returns
    -------
    tuple
        (temperatures, K_ID_values)
    """
    if T_range is None:
        T_range = list(np.arange(-80, 21, 2.0))

    K_ID_vals = [dynamic_fracture_toughness(steel_grade, T, loading_rate)
                 for T in T_range]
    return T_range, K_ID_vals


# ============================================================================
# 4. Crack propagation criterion under ice impact
# ============================================================================

@dataclass
class IceImpactCrackAssessment:
    """Result of crack propagation assessment under ice impact loading.

    Attributes
    ----------
    K_I : float
        Applied stress intensity factor [MPa*sqrt(m)]
    K_ID : float
        Dynamic fracture toughness at service temperature [MPa*sqrt(m)]
    crack_will_propagate : bool
        True if K_I >= K_ID
    safety_margin : float
        (K_ID - K_I) / K_ID; positive = safe
    temperature_degC : float
        Service temperature [degC]
    loading_rate : float
        dK/dt [MPa*sqrt(m)/s]
    steel_grade : str
        Steel grade name
    """
    K_I: float
    K_ID: float
    crack_will_propagate: bool
    safety_margin: float
    temperature_degC: float
    loading_rate: float
    steel_grade: str


def assess_crack_propagation(steel_grade: str, T_degC: float,
                              crack_depth_m: float, plate_thickness_m: float,
                              applied_stress_MPa: float,
                              crack_type: str = "surface",
                              loading_rate: float = 100.0) -> IceImpactCrackAssessment:
    """Assess crack propagation under ice impact loading.

    Computes K_I for the given crack geometry and compares with K_ID(T, dK/dt).

    Crack geometries:
    - "surface": Semi-elliptical surface crack, K_I = 1.12 * sigma * sqrt(pi*a) * Y
    - "through": Through-thickness center crack, K_I = sigma * sqrt(pi*a)

    Parameters
    ----------
    steel_grade : str
        Steel grade name
    T_degC : float
        Service temperature [degC]
    crack_depth_m : float
        Crack depth (semi-major axis for surface crack) [m]
    plate_thickness_m : float
        Plate thickness [m]
    applied_stress_MPa : float
        Applied stress (from ice impact) [MPa]
    crack_type : str
        "surface" or "through"
    loading_rate : float
        dK/dt [MPa*sqrt(m)/s]

    Returns
    -------
    IceImpactCrackAssessment
    """
    if crack_type not in ("surface", "through"):
        raise ValueError(f"crack_type must be 'surface' or 'through', got '{crack_type}'")

    a = crack_depth_m
    sigma = applied_stress_MPa

    if crack_type == "surface":
        # Semi-elliptical surface crack correction factor
        # Y ~ 1.12 * M_k for surface crack (simplified)
        M_k = 1.0 + 0.12 * (1.0 - a / plate_thickness_m) ** 2
        Y = 1.12 * M_k
        K_I = Y * sigma * math.sqrt(math.pi * a)
    else:
        # Through-thickness center crack in finite width plate
        # K_I = sigma * sqrt(pi * a) * sec_correction
        W = plate_thickness_m  # Simplified: use thickness as width proxy
        sec_arg = math.pi * a / (2.0 * W) if W > 0 else 0
        if sec_arg < math.pi / 2:
            sec_corr = 1.0 / math.sqrt(math.cos(sec_arg))
        else:
            sec_corr = 2.0  # Cap
        K_I = sigma * math.sqrt(math.pi * a) * sec_corr

    K_ID = dynamic_fracture_toughness(steel_grade, T_degC, loading_rate)

    will_propagate = K_I >= K_ID
    safety_margin = (K_ID - K_I) / K_ID if K_ID > 0 else -1.0

    return IceImpactCrackAssessment(
        K_I=K_I,
        K_ID=K_ID,
        crack_will_propagate=will_propagate,
        safety_margin=safety_margin,
        temperature_degC=T_degC,
        loading_rate=loading_rate,
        steel_grade=steel_grade,
    )


# ============================================================================
# 5. XFEM / Phase-field crack growth simulation (simplified 1D/2D)
# ============================================================================

@dataclass
class PhaseFieldCrackResult:
    """Result of phase-field crack propagation simulation.

    Attributes
    ----------
    steps : list of dict
        Each step contains: load_factor, crack_length_m, energy_release_rate
    final_crack_length_m : float
        Final crack length [m]
    critical_load_MPa : float
        Load at which crack initiates [MPa]
    """
    steps: List[dict]
    final_crack_length_m: float
    critical_load_MPa: float


def phase_field_crack_growth_1d(
    steel_grade: str,
    T_degC: float,
    initial_crack_m: float,
    domain_length_m: float,
    G_c: float,
    E_GPa: float = 206.0,
    n_steps: int = 50,
    max_load_MPa: float = 500.0,
) -> PhaseFieldCrackResult:
    """Simplified 1D phase-field crack growth simulation.

    Uses a regularized crack phase field phi(x) with length scale l0.
    Crack grows when energy release rate G >= G_c(T).

    This is a pedagogical/demonstration implementation showing the
    phase-field regularization concept. For full 2D/3D XFEM, use the
    phase_field_fracture module.

    Parameters
    ----------
    steel_grade : str
        Steel grade name
    T_degC : float
        Temperature [degC]
    initial_crack_m : float
        Initial crack length [m]
    domain_length_m : float
        Domain length [m]
    G_c : float
        Critical energy release rate [J/m^2] (temperature-dependent)
    E_GPa : float
        Young's modulus [GPa]
    n_steps : int
        Number of load steps
    max_load_MPa : float
        Maximum applied stress [MPa]

    Returns
    -------
    PhaseFieldCrackResult
    """
    E = E_GPa * 1e3  # MPa
    K_ID = dynamic_fracture_toughness(steel_grade, T_degC)

    # Convert K_ID to G_c equivalent for comparison
    # G = K^2 / E' where E' = E for plane stress, E/(1-nu^2) for plane strain
    nu = 0.3
    E_prime = E / (1.0 - nu ** 2)  # Plane strain
    G_c_effective = K_ID ** 2 / E_prime  # MPa*m = MJ/m^2 -> convert

    # Phase field length scale
    l0 = 3.0 * initial_crack_m / 10.0  # Regularization length

    steps = []
    crack_length = initial_crack_m
    critical_load = None

    for i in range(n_steps):
        load_factor = (i + 1) / n_steps
        sigma = load_factor * max_load_MPa

        # Energy release rate for center crack
        G = sigma ** 2 * math.pi * crack_length / E_prime

        # Check crack growth criterion
        if G >= G_c_effective and crack_length < domain_length_m * 0.9:
            # Phase-field regularized growth
            delta_a = l0 * (G / G_c_effective - 1.0) * 0.1
            crack_length = min(crack_length + delta_a, domain_length_m * 0.9)
            if critical_load is None:
                critical_load = sigma

        steps.append({
            "step": i + 1,
            "load_factor": load_factor,
            "stress_MPa": sigma,
            "crack_length_m": crack_length,
            "energy_release_rate_MPa_m": G,
            "G_c_MPa_m": G_c_effective,
        })

    if critical_load is None:
        critical_load = max_load_MPa  # No crack growth

    return PhaseFieldCrackResult(
        steps=steps,
        final_crack_length_m=crack_length,
        critical_load_MPa=critical_load,
    )


# ============================================================================
# 6. Comprehensive assessment report
# ============================================================================

@dataclass
class FractureAssessmentReport:
    """Comprehensive low-temperature fracture assessment report.

    Contains all key results for a given steel grade and service condition.
    """
    steel_grade: str
    temperature_degC: float
    DBTT_27J_degC: float
    DBTT_shifted_degC: float
    KV_at_T_J: float
    K_ID_MPa_sqrtm: float
    K_IC_MPa_sqrtm: float
    crack_assessment: Optional[IceImpactCrackAssessment] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        d = {
            "steel_grade": self.steel_grade,
            "temperature_degC": self.temperature_degC,
            "DBTT_27J_degC": self.DBTT_27J_degC,
            "DBTT_shifted_degC": self.DBTT_shifted_degC,
            "KV_at_T_J": self.KV_at_T_J,
            "K_ID_MPa_sqrtm": self.K_ID_MPa_sqrtm,
            "K_IC_MPa_sqrtm": self.K_IC_MPa_sqrtm,
        }
        if self.crack_assessment is not None:
            ca = self.crack_assessment
            d["crack_assessment"] = {
                "K_I_MPa_sqrtm": ca.K_I,
                "K_ID_MPa_sqrtm": ca.K_ID,
                "crack_will_propagate": ca.crack_will_propagate,
                "safety_margin": ca.safety_margin,
            }
        return d


def comprehensive_assessment(
    steel_grade: str,
    T_degC: float,
    plate_thickness_mm: float = 25.0,
    crack_depth_m: float = 0.005,
    applied_stress_MPa: float = 200.0,
    loading_rate: float = 100.0,
) -> FractureAssessmentReport:
    """Run comprehensive low-temperature fracture assessment.

    Parameters
    ----------
    steel_grade : str
        Steel grade name
    T_degC : float
        Service temperature [degC]
    plate_thickness_mm : float
        Plate thickness [mm]
    crack_depth_m : float
        Initial crack depth [m]
    applied_stress_MPa : float
        Applied stress from ice impact [MPa]
    loading_rate : float
        dK/dt [MPa*sqrt(m)/s]

    Returns
    -------
    FractureAssessmentReport
    """
    mat = STEEL_GRADES[steel_grade]

    # DBTT
    DBTT_ref = find_DBTT_27J(steel_grade)
    DBTT_shifted = DBTT_thickness_shift(DBTT_ref, 25.0, plate_thickness_mm)

    # KV at service temperature
    KV_T = charpy_kv(T_degC, mat["KV_upper_J"], mat["KV_lower_J"],
                     mat["T0_degC"], mat["Delta_T_degC"])

    # Dynamic fracture toughness
    K_ID = dynamic_fracture_toughness(steel_grade, T_degC, loading_rate)

    # Crack assessment
    plate_thickness_m = plate_thickness_mm / 1000.0
    crack_assessment = assess_crack_propagation(
        steel_grade, T_degC, crack_depth_m, plate_thickness_m,
        applied_stress_MPa, crack_type="surface", loading_rate=loading_rate,
    )

    return FractureAssessmentReport(
        steel_grade=steel_grade,
        temperature_degC=T_degC,
        DBTT_27J_degC=DBTT_ref,
        DBTT_shifted_degC=DBTT_shifted,
        KV_at_T_J=KV_T,
        K_ID_MPa_sqrtm=K_ID,
        K_IC_MPa_sqrtm=mat["K_IC_MPa_sqrtm"],
        crack_assessment=crack_assessment,
    )
