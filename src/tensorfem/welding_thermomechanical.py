"""Welding thermo-elasto-plastic verification: Rosenthal, Satoh, butt weld.

This module provides verification cases for welding thermomechanical analysis:
(a) Rosenthal moving point/line heat source temperature field
(b) Satoh 1D thermo-elasto-plastic residual stress
(c) Butt weld residual stress and buckling deformation

References:
- Rosenthal, D. "Mathematical theory of heat distribution during welding
  and cutting" Weld. J. 20 (1941) 220s-234s
- Satoh, T. "Effects of thermal cycles on residual stresses in welding"
  J. Japan Weld. Soc. 23 (1954)
- Masubuchi, K. "Analysis of Welded Structures" Pergamon (1980)
- Kou, S. "Welding Metallurgy" Wiley (2003)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from typing import Tuple, Optional

Tensor = torch.Tensor
_DTYPE = torch.float64


# ============================================================================
# (a) Rosenthal Moving Heat Source
# ============================================================================

def rosenthal_point_source(x: float, y: float, z: float,
                           velocity: float, power: float,
                           conductivity: float, diffusivity: float,
                           T_initial: float = 20.0) -> float:
    """Rosenthal 3D moving point heat source temperature field.

    Quasi-steady state solution for a point source moving at velocity v
    along the x-axis on a semi-infinite body:

    T - T0 = (Q / (2*pi*k*R)) * exp(-v*(x + R)/(2*alpha))

    where:
    - Q = power (W)
    - k = thermal conductivity (W/(m·K))
    - alpha = thermal diffusivity (m²/s)
    - v = travel velocity (m/s)
    - R = sqrt(x² + y² + z²) distance from source
    - T0 = initial temperature

    Valid for: quasi-steady state, no boundaries, no phase change.
    """
    R = math.sqrt(x**2 + y**2 + z**2)
    if R < 1e-10:
        return float('inf')  # singularity at source

    T = T_initial + (power / (2.0 * math.pi * conductivity * R)) * \
        math.exp(-velocity * (x + R) / (2.0 * diffusivity))
    return T


def rosenthal_line_source(x: float, y: float,
                          velocity: float, power_per_length: float,
                          conductivity: float, diffusivity: float,
                          thickness: float,
                          T_initial: float = 20.0) -> float:
    """Rosenthal 2D moving line heat source (thin plate).

    Quasi-steady state for a line source through plate thickness h:

    T - T0 = (Q/(2*pi*k*h)) * K0(v*r/(2*alpha)) * exp(-v*x/(2*alpha))

    For large r, K0(z) ≈ sqrt(pi/(2z)) * exp(-z), giving:
    T - T0 ≈ (Q/(2*pi*k*h)) * sqrt(pi*alpha/(v*r)) * exp(-v*(x+r)/(2*alpha))

    Simplified (far field):
    T - T0 = (Q/(2*pi*k*h*r)) * exp(-v*(x+r)/(2*alpha))

    where r = sqrt(x² + y²)
    """
    r = math.sqrt(x**2 + y**2)
    if r < 1e-10:
        return float('inf')

    # Use the simplified far-field form (valid for v*r/(2*alpha) > 1)
    T = T_initial + (power_per_length / (2.0 * math.pi * conductivity * thickness * r)) * \
        math.exp(-velocity * (x + r) / (2.0 * diffusivity))
    return T


def rosenthal_temperature_field_3d(grid_x: Tensor, grid_y: Tensor, grid_z: Tensor,
                                   velocity: float, power: float,
                                   conductivity: float, diffusivity: float,
                                   T_initial: float = 20.0) -> Tensor:
    """Compute Rosenthal temperature field on a 3D grid.

    Returns temperature tensor of same shape as input grids.
    """
    shape = grid_x.shape
    T = torch.zeros(shape, dtype=_DTYPE)

    for idx in range(grid_x.numel()):
        xi = grid_x.flatten()[idx].item()
        yi = grid_y.flatten()[idx].item()
        zi = grid_z.flatten()[idx].item()
        T.flatten()[idx] = rosenthal_point_source(xi, yi, zi, velocity, power,
                                                   conductivity, diffusivity, T_initial)
    return T


@dataclass
class RosenthalResult:
    """Rosenthal moving heat source result."""
    peak_temperature: float
    peak_location: Tuple[float, float, float]
    cooling_rate_at_500C: Optional[float]  # K/s at 500°C on centerline
    fusion_zone_width: float  # approximate width where T > T_liquidus


def rosenthal_peak_temperature(power: float, velocity: float,
                               conductivity: float, diffusivity: float,
                               T_initial: float = 20.0,
                               reference_distance: float = 0.005) -> float:
    """Peak temperature at a reference distance behind the arc.

    The peak temperature on the centerline (y=z=0) at distance x behind
    the source (x < 0 in the wake):
    T_peak = T0 + Q/(2*pi*k*|x|) * exp(-v*(x+|x|)/(2*alpha))
           = T0 + Q/(2*pi*k*|x|) * exp(0)  (since x < 0, x + |x| = 0)
           = T0 + Q/(2*pi*k*|x|)
    """
    return T_initial + power / (2.0 * math.pi * conductivity * reference_distance)


# ============================================================================
# (b) Satoh 1D Thermo-Elasto-Plastic Residual Stress
# ============================================================================

@dataclass
class SatohResult:
    """Satoh bar thermo-elasto-plastic result."""
    positions: Tensor       # x positions along bar
    peak_temperatures: Tensor  # maximum temperature reached at each point
    residual_stress: Tensor    # final residual stress
    residual_plastic_strain: Tensor  # final plastic strain
    max_residual_stress: float
    yield_stress: float


def satoh_1d_residual_stress(n_points: int = 101,
                             peak_temp_center: float = 500.0,
                             width: float = 0.05,
                             young: float = 210e9,
                             yield_stress: float = 300e6,
                             alpha_thermal: float = 1.2e-5,
                             sigma_y_temp_dependence: float = 0.5e6  # Pa/K
                             ) -> SatohResult:
    """Satoh's 1D thermo-elasto-plastic residual stress analysis.

    Consider a long bar (1D, constrained at both ends) subjected to
    a thermal cycle. The temperature distribution is:
    T(x) = T_max * exp(-x²/(2*sigma²))

    Physics:
    - During heating: thermal expansion is constrained, causing compressive
      stress. When |sigma| > sigma_y(T), plastic compression occurs.
    - During cooling: the plastically compressed material contracts less
      than surrounding material, leading to tensile residual stress.

    Sign convention: positive = tension, negative = compression.

    The incremental analysis:
    1. Heating: eps_total = 0 (constrained), eps_mech = -eps_thermal
       sigma = E * (eps_mech - eps_plastic)
       When sigma < -sigma_y(T), plastic compression accumulates
    2. Cooling: reverse process, tensile residual stress develops
    """
    # Spatial domain
    L = 10.0 * width  # total bar length
    x = torch.linspace(-L/2, L/2, n_points, dtype=_DTYPE)

    # Temperature history: Gaussian peak
    T_peak = peak_temp_center * torch.exp(-x**2 / (2.0 * width**2))

    # Material properties
    E = young
    sigma_y0 = yield_stress
    alpha = alpha_thermal

    # Incremental analysis
    n_increments = 200
    sigma = torch.zeros(n_points, dtype=_DTYPE)
    ep = torch.zeros(n_points, dtype=_DTYPE)  # plastic strain (positive = tension)

    # Heating phase
    for i in range(n_increments):
        frac = (i + 1) / n_increments
        T_current = frac * T_peak

        # Thermal strain (positive when heated)
        eps_thermal = alpha * T_current

        # Total strain = 0 (fully constrained)
        # Mechanical strain = eps_total - eps_thermal = -eps_thermal
        # Elastic strain = eps_mech - eps_plastic = -eps_thermal - eps_plastic
        eps_elastic = -eps_thermal - ep

        # Trial stress
        sigma_trial = E * eps_elastic

        # Temperature-dependent yield stress
        sigma_y_T = torch.clamp(sigma_y0 - sigma_y_temp_dependence * T_current,
                                min=0.1 * sigma_y0)

        # Yield check: f = |sigma| - sigma_y
        f = torch.abs(sigma_trial) - sigma_y_T
        yielded = f > 0

        # Return mapping
        if yielded.any():
            # Plastic flow direction: sign of trial stress
            direction = torch.sign(sigma_trial)
            # Plastic multiplier
            dEp = f[yielded] / E
            ep[yielded] += dEp * direction[yielded]
            # Updated stress = yield stress with appropriate sign
            sigma[yielded] = sigma_y_T[yielded] * direction[yielded]
        else:
            sigma = sigma_trial.clone()

        sigma[~yielded] = sigma_trial[~yielded]

    # Cooling phase
    for i in range(n_increments):
        frac = 1.0 - (i + 1) / n_increments
        T_current = frac * T_peak

        eps_thermal = alpha * T_current
        eps_elastic = -eps_thermal - ep

        sigma_trial = E * eps_elastic

        sigma_y_T = torch.clamp(sigma_y0 - sigma_y_temp_dependence * T_current,
                                min=0.1 * sigma_y0)

        f = torch.abs(sigma_trial) - sigma_y_T
        yielded = f > 0

        if yielded.any():
            direction = torch.sign(sigma_trial)
            dEp = f[yielded] / E
            ep[yielded] += dEp * direction[yielded]
            sigma[yielded] = sigma_y_T[yielded] * direction[yielded]

        sigma[~yielded] = sigma_trial[~yielded]

    # At end of cooling, T=0, so eps_thermal=0
    # sigma = E * (-ep) = -E * ep
    # The plastic strain ep accumulated during heating is negative (compression)
    # So sigma = -E * (negative) = positive (tension) in heated zone

    # Enforce equilibrium: integral(sigma) dx = 0
    dx = x[1] - x[0]
    L_total = x[-1] - x[0]
    mean_sigma = torch.sum(sigma) * dx / L_total
    sigma_res = sigma - mean_sigma

    return SatohResult(
        positions=x,
        peak_temperatures=T_peak,
        residual_stress=sigma_res,
        residual_plastic_strain=ep,
        max_residual_stress=float(torch.max(torch.abs(sigma_res))),
        yield_stress=yield_stress,
    )


def satoh_analytical_residual_stress(x: Tensor, peak_temp_center: float,
                                     width: float, young: float,
                                     yield_stress: float,
                                     alpha_thermal: float) -> Tensor:
    """Analytical approximation for Satoh residual stress.

    For a Gaussian temperature peak T(x) = T_max * exp(-x²/(2w²)):

    If E*alpha*T_max > sigma_y (yielding occurs):
    - In the yielded zone: plastic strain accumulates during heating
    - After cooling: sigma_res = -E * ep (tensile where ep < 0)

    Simplified result:
    The maximum tensile residual stress is approximately sigma_y
    (in the fully yielded zone at the center).
    """
    E = young
    sigma_y = yield_stress
    alpha = alpha_thermal

    T_peak = peak_temp_center * torch.exp(-x**2 / (2.0 * width**2))

    # Thermal stress if elastic
    sigma_thermal = E * alpha * T_peak

    # Where thermal stress exceeds yield, plastic strain accumulates
    # After cooling, residual stress = -E * ep_plastic
    # ep_plastic ≈ (sigma_thermal - sigma_y) / E in yielded zone
    # sigma_res ≈ sigma_y - sigma_thermal + sigma_thermal = sigma_y (capped)

    # More precisely: in yielded zone, sigma_res ≈ sigma_y (tensile)
    # In elastic zone, sigma_res = E*alpha*T_peak (but must satisfy equilibrium)

    # Capped at yield
    sigma_capped = torch.clamp(sigma_thermal, max=sigma_y)

    # Equilibrium correction
    L = x[-1] - x[0]
    dx = x[1] - x[0]
    sigma_mean = torch.sum(sigma_capped) * dx / L

    sigma_res = sigma_capped - sigma_mean

    return sigma_res


# ============================================================================
# (c) Butt Weld Residual Stress and Buckling
# ============================================================================

@dataclass
class ButtWeldResult:
    """Butt weld residual stress and buckling result."""
    transverse_positions: Tensor  # y positions (perpendicular to weld)
    longitudinal_stress: Tensor   # sigma_xx residual stress
    transverse_stress: Tensor     # sigma_yy residual stress
    peak_tensile_stress: float
    peak_compressive_stress: float
    buckling_displacement: float  # out-of-plane displacement if buckled
    critical_weld_length: float   # length at which buckling occurs
    is_buckled: bool


def butt_weld_residual_stress_plate(width: float, thickness: float,
                                    weld_speed: float, power: float,
                                    conductivity: float, diffusivity: float,
                                    young: float, yield_stress: float,
                                    alpha_thermal: float,
                                    n_points: int = 51) -> ButtWeldResult:
    """Residual stress in a butt-welded plate (simplified analytical model).

    For a butt weld along the centerline of a plate:
    - The weld zone heats up and cools down
    - Tensile residual stress develops in/near the weld (yield-level)
    - Compressive residual stress develops away from the weld
    - Equilibrium: integral of stress through width = 0

    Model (based on Masubuchi, 1980):
    - Tensile zone: sigma_t = sigma_y at center, decaying Gaussian
    - Compressive zone: uniform compression to satisfy equilibrium
    """
    nu = 0.3

    # Transverse positions
    y = torch.linspace(-width/2, width/2, n_points, dtype=_DTYPE)

    # Tensile zone half-width (typical for welding)
    b_t = 0.1 * width

    # Tensile stress distribution: Gaussian peak at yield level
    sigma_tensile = yield_stress * torch.exp(-y**2 / (2.0 * b_t**2))

    # Compressive stress from equilibrium
    # integral(sigma_tensile + sigma_c) dy = 0
    # sigma_c = -integral(sigma_tensile) / width
    dy = y[1] - y[0]
    integral_tensile = torch.sum(sigma_tensile) * dy
    sigma_compressive = -integral_tensile / width

    # Total longitudinal stress
    sigma_xx = sigma_tensile + sigma_compressive

    # Transverse residual stress (smaller, self-equilibrating)
    sigma_yy = 0.3 * yield_stress * (y**2 / (width/2)**2 - 1.0/3.0) * \
               torch.exp(-y**2 / (4.0 * b_t**2))

    # Buckling assessment
    # Critical stress for plate buckling (simply supported edges)
    k_buckling = 4.0  # simply supported
    sigma_cr = k_buckling * math.pi**2 * young / (12.0 * (1.0 - nu**2)) * \
               (thickness / width)**2

    # Check if compressive stress exceeds critical
    max_compressive = float(torch.min(sigma_xx))
    is_buckled = abs(max_compressive) > sigma_cr

    # Buckling displacement (if buckled, simplified estimate)
    if is_buckled:
        excess = abs(max_compressive) / sigma_cr - 1.0
        buckling_disp = thickness * math.sqrt(max(excess, 0)) * 0.5
    else:
        buckling_disp = 0.0

    # Critical weld length for buckling
    if abs(sigma_compressive) > 0:
        b_cr = thickness * math.sqrt(k_buckling * math.pi**2 * young /
                                     (12.0 * (1.0 - nu**2) * abs(sigma_compressive)))
    else:
        b_cr = float('inf')

    return ButtWeldResult(
        transverse_positions=y,
        longitudinal_stress=sigma_xx,
        transverse_stress=sigma_yy,
        peak_tensile_stress=float(torch.max(sigma_xx)),
        peak_compressive_stress=float(torch.min(sigma_xx)),
        buckling_displacement=buckling_disp,
        critical_weld_length=b_cr,
        is_buckled=is_buckled,
    )


def butt_weld_thermal_cycle(y: float, weld_speed: float, power: float,
                            conductivity: float, diffusivity: float,
                            thickness: float,
                            T_initial: float = 20.0) -> Tuple[Tensor, Tensor]:
    """Compute thermal cycle at a point during butt welding.

    Uses Rosenthal line source for thin plate.

    Returns: (time_array, temperature_array)
    """
    # Time array
    t_total = 100.0  # seconds
    n_times = 200
    t = torch.linspace(0.001, t_total, n_times, dtype=_DTYPE)

    # Distance from weld centerline
    r = abs(y)

    # Temperature at each time step
    T = torch.zeros(n_times, dtype=_DTYPE)
    for i, ti in enumerate(t):
        x_rel = -weld_speed * ti  # relative position (weld moving away)
        r_current = math.sqrt(x_rel**2 + y**2)
        if r_current > 1e-6:
            T[i] = T_initial + (power / (2.0 * math.pi * conductivity * thickness * r_current)) * \
                   math.exp(-weld_speed * (x_rel + r_current) / (2.0 * diffusivity))
        else:
            T[i] = T_initial + 2000.0  # near source, cap at reasonable value

    return t, T
