"""Nonlinear structural verification: hyperelasticity, plastic hinge, Taylor bar.

This module provides verification cases for geometric/material nonlinearity:
(a) Neo-Hookean hyperelastic uniaxial/biaxial tension vs analytical solutions
(b) Plastic hinge: simply supported beam plastic limit load
(c) Taylor bar impact: final mushroom head diameter and length

References:
- Ogden, "Non-linear Elastic Deformations" (1984)
- Hodge, "Plastic Analysis of Structures" (1959)
- Taylor, "The use of flat-ended projectiles for determining dynamic yield
  stress I. Theoretical considerations" Proc. R. Soc. A 194 (1948)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

Tensor = torch.Tensor
_DTYPE = torch.float64


# ============================================================================
# (a) Neo-Hookean Hyperelastic Verification
# ============================================================================

def neo_hookean_uniaxial_stress(stretch: float, young: float, poisson: float
                                ) -> tuple[float, float, float]:
    """Analytical Neo-Hookean uniaxial stress (compressible).

    For uniaxial tension with stretch lambda_1 = stretch:
    - Lateral stretches lambda_2 = lambda_3 determined from sigma_22 = sigma_33 = 0
    - Cauchy stress: sigma_11 = (mu/J)*(l1^2 - 1) + (lam*logJ/J)
    - Nominal (1st PK) stress: P_11 = J * sigma_11 / l1

    Returns: (cauchy_stress, nominal_stress, lateral_stretch)
    """
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    l1 = stretch
    # Solve for lateral stretch l2 from sigma_22 = 0
    # sigma_22 = (mu/J)*(l2^2 - 1) + (lam*logJ/J) = 0
    # => mu*(l2^2 - 1) + lam*logJ = 0
    # Newton iteration
    l2 = 1.0 / math.sqrt(l1) if l1 > 0 else 1.0  # incompressible initial guess

    for iteration in range(200):
        J = l1 * l2 * l2
        if J <= 0:
            l2 = 0.99  # reset
            continue
        logJ = math.log(J)

        # f(l2) = mu*(l2^2 - 1) + lam*logJ = 0
        f_val = mu * (l2**2 - 1.0) + lam * logJ
        # df/dl2 = 2*mu*l2 + lam * 2/l2
        df_val = 2.0 * mu * l2 + lam * 2.0 / l2

        if abs(df_val) < 1e-30:
            break

        l2_new = l2 - f_val / df_val
        if l2_new <= 0:
            l2_new = l2 * 0.5  # keep positive

        if abs(l2_new - l2) < 1e-14 * max(1.0, abs(l2)):
            l2 = l2_new
            break
        l2 = l2_new

    J = l1 * l2 * l2
    logJ = math.log(J)

    # Cauchy stress: sigma = (mu/J)*(B - I) + (lam*logJ/J)*I
    sigma_11 = (mu / J) * (l1**2 - 1.0) + (lam * logJ / J)

    # Nominal (1st Piola-Kirchhoff) stress: P = J * sigma * F^{-T}
    # P_11 = J * sigma_11 / l1
    P_11 = J * sigma_11 / l1

    return sigma_11, P_11, l2


def neo_hookean_biaxial_stress(stretch: float, young: float, poisson: float
                               ) -> tuple[float, float, float]:
    """Analytical Neo-Hookean equi-biaxial stress (compressible).

    For equi-biaxial tension: lambda_1 = lambda_2 = stretch
    By symmetry: lambda_3 determined from sigma_33 = 0.

    Returns: (cauchy_stress_11, nominal_stress_11, thickness_stretch)
    """
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    l1 = stretch
    # Solve for l3 from sigma_33 = 0
    # sigma_33 = (mu/J)*(l3^2 - 1) + (lam*logJ/J) = 0
    # => mu*(l3^2 - 1) + lam*logJ = 0
    l3 = 1.0 / (l1 * l1) if l1 > 0 else 1.0  # incompressible initial guess

    for iteration in range(200):
        J = l1 * l1 * l3
        if J <= 0:
            l3 = 0.99
            continue
        logJ = math.log(J)

        f_val = mu * (l3**2 - 1.0) + lam * logJ
        df_val = 2.0 * mu * l3 + lam * 2.0 / l3

        if abs(df_val) < 1e-30:
            break

        l3_new = l3 - f_val / df_val
        if l3_new <= 0:
            l3_new = l3 * 0.5

        if abs(l3_new - l3) < 1e-14 * max(1.0, abs(l3)):
            l3 = l3_new
            break
        l3 = l3_new

    J = l1 * l1 * l3
    logJ = math.log(J)

    # Cauchy stress
    sigma_11 = (mu / J) * (l1**2 - 1.0) + (lam * logJ / J)

    P_11 = J * sigma_11 / l1

    return sigma_11, P_11, l3


def neo_hookean_uniaxial_incompressible(stretch: float, mu: float) -> float:
    """Incompressible Neo-Hookean uniaxial Cauchy stress (exact closed form).

    sigma_11 = mu * (lambda^2 - 1/lambda)
    """
    return mu * (stretch**2 - 1.0 / stretch)


def neo_hookean_biaxial_incompressible(stretch: float, mu: float) -> float:
    """Incompressible Neo-Hookean equi-biaxial Cauchy stress (exact closed form).

    sigma_11 = mu * (lambda^2 - 1/lambda^4)
    """
    return mu * (stretch**2 - 1.0 / stretch**4)


# ============================================================================
# (b) Plastic Hinge - Simply Supported Beam
# ============================================================================

@dataclass
class PlasticHingeResult:
    """Plastic limit analysis of a simply supported beam."""
    span: float           # L
    yield_stress: float   # sigma_y
    width: float          # b
    height: float         # h
    plastic_limit_load: float   # P_pl (analytical)
    elastic_limit_load: float   # P_y (first yield)
    shape_factor: float         # Z/S (plastic/elastic section modulus ratio)
    plastic_moment: float       # M_pl
    yield_moment: float         # M_y


def plastic_hinge_analysis(span: float, yield_stress: float,
                           width: float, height: float) -> PlasticHingeResult:
    """Plastic limit analysis of simply supported beam with central point load.

    Analytical solution:
    - Elastic section modulus: S = b*h^2/6
    - Plastic section modulus: Z = b*h^2/4
    - Shape factor: Z/S = 1.5 (rectangular section)
    - Yield moment: M_y = sigma_y * S
    - Plastic moment: M_pl = sigma_y * Z
    - Elastic limit load (first yield): P_y = 4*M_y/L
    - Plastic limit load: P_pl = 4*M_pl/L

    The plastic hinge forms at midspan when M = M_pl.
    """
    S = width * height**2 / 6.0  # elastic section modulus
    Z = width * height**2 / 4.0  # plastic section modulus
    shape_factor = Z / S  # = 1.5 for rectangular

    M_y = yield_stress * S
    M_pl = yield_stress * Z

    # Simply supported beam, central point load P:
    # M_max = P*L/4 at midspan
    # First yield: P_y = 4*M_y/L
    # Plastic collapse: P_pl = 4*M_pl/L
    P_y = 4.0 * M_y / span
    P_pl = 4.0 * M_pl / span

    return PlasticHingeResult(
        span=span,
        yield_stress=yield_stress,
        width=width,
        height=height,
        plastic_limit_load=P_pl,
        elastic_limit_load=P_y,
        shape_factor=shape_factor,
        plastic_moment=M_pl,
        yield_moment=M_y,
    )


# ============================================================================
# (c) Taylor Bar Impact
# ============================================================================

@dataclass
class TaylorBarResult:
    """Taylor bar impact results."""
    initial_velocity: float   # V_0
    initial_length: float     # L_0
    initial_diameter: float   # D_0
    dynamic_yield: float      # sigma_y_dyn (Y0)
    final_length: float       # L_f
    final_diameter: float     # D_f
    final_length_ratio: float     # L_f/L_0
    final_diameter_ratio: float   # D_f/D_0
    plastic_zone_length: float    # length of deformed region


def taylor_bar_analytical(initial_velocity: float, initial_length: float,
                          initial_diameter: float, density: float,
                          dynamic_yield: float) -> TaylorBarResult:
    """Taylor bar impact analytical solution.

    Based on Taylor (1948) theory for a rigid-perfectly-plastic bar
    impacting a rigid anvil.

    For rigid-perfectly-plastic material:
    - Final length: L_f/L_0 = exp(-rho*V_0^2/(2*sigma_y))
    - Final diameter from volume conservation: D_f/D_0 = sqrt(L_0/L_f)

    References:
    - Taylor, G.I. (1948) Proc. R. Soc. A 194, 289-306
    - Wilkins, M.L. & Guinan, R.W. (1973) Int. J. Eng. Sci. 11, 599-610
    """
    rho = density
    Y0 = dynamic_yield
    V0 = initial_velocity
    L0 = initial_length
    D0 = initial_diameter

    # Taylor's parameter
    alpha = rho * V0**2 / (2.0 * Y0)

    if alpha > 10:
        raise ValueError(f"Taylor bar parameter alpha={alpha:.2f} too large")

    # Final length ratio
    Lf_L0 = math.exp(-alpha)
    Lf = L0 * Lf_L0

    # Final mushroom diameter from volume conservation
    Df_D0 = math.sqrt(1.0 / Lf_L0)  # = exp(alpha/2)
    Df = D0 * Df_D0

    # Plastic zone length
    plastic_zone = L0 - Lf

    return TaylorBarResult(
        initial_velocity=V0,
        initial_length=L0,
        initial_diameter=D0,
        dynamic_yield=Y0,
        final_length=Lf,
        final_diameter=Df,
        final_length_ratio=Lf_L0,
        final_diameter_ratio=Df_D0,
        plastic_zone_length=plastic_zone,
    )


def taylor_bar_numerical(initial_velocity: float, initial_length: float,
                         initial_diameter: float, density: float,
                         dynamic_yield: float, young: float,
                         n_elements: int = 100,
                         n_steps: int = 10000) -> TaylorBarResult:
    """Taylor bar impact using 1D Lagrangian FEM with J2 plasticity.

    This solves the 1D Taylor bar problem using:
    - Lagrangian mesh along the bar axis
    - 1D J2 plasticity (uniaxial stress) with radial return
    - Central difference time integration (explicit)
    """
    L0 = initial_length
    D0 = initial_diameter
    V0 = initial_velocity
    rho = density
    Y0 = dynamic_yield
    E = young
    nu = 0.3

    # 1D mesh
    dx = L0 / n_elements
    area = math.pi * D0**2 / 4.0

    # Nodal positions (Lagrangian - initial = reference)
    x = torch.linspace(0, L0, n_elements + 1, dtype=_DTYPE)
    x_ref = x.clone()

    # Lumped mass
    m_node = rho * area * dx
    mass = torch.full((n_elements + 1,), m_node, dtype=_DTYPE)
    mass[0] = m_node / 2.0
    mass[-1] = m_node / 2.0

    # Initial conditions: uniform velocity -V0 (toward rigid wall at x=0)
    velocity = torch.full((n_elements + 1,), -V0, dtype=_DTYPE)
    displacement = torch.zeros(n_elements + 1, dtype=_DTYPE)

    # Element state
    stress = torch.zeros(n_elements, dtype=_DTYPE)
    ep = torch.zeros(n_elements, dtype=_DTYPE)  # plastic strain

    # Time step (CFL)
    c_wave = math.sqrt(E / rho)
    dt = 0.8 * dx / c_wave

    # Simulation
    t_total = 3.0 * L0 / max(V0, 1.0)
    n_steps_actual = min(n_steps, int(t_total / dt) + 1)

    for step in range(n_steps_actual):
        # Compute element strains (engineering)
        strain = torch.zeros(n_elements, dtype=_DTYPE)
        for e in range(n_elements):
            current_length = dx + displacement[e + 1] - displacement[e]
            strain[e] = (current_length - dx) / dx

        # Update stress using 1D J2 (uniaxial)
        for e in range(n_elements):
            # Trial stress: sigma = E * (eps - eps_p)
            sigma_trial = E * (strain[e] - ep[e])

            # Yield check (1D von Mises = |sigma|)
            f_trial = abs(sigma_trial) - Y0

            if f_trial > 0:
                # Return mapping
                direction = 1.0 if sigma_trial > 0 else -1.0
                dEp = f_trial / E
                ep[e] = ep[e] + dEp * direction
                stress[e] = Y0 * direction
            else:
                stress[e] = sigma_trial

        # Internal forces
        force = torch.zeros(n_elements + 1, dtype=_DTYPE)
        for e in range(n_elements):
            f_int = stress[e] * area
            force[e] -= f_int
            force[e + 1] += f_int

        # Update velocity
        acceleration = force / mass
        velocity += acceleration * dt

        # Impact BC: node 0 fixed (cannot go negative)
        if velocity[0] < 0:
            velocity[0] = 0.0

        # Update displacement
        displacement += velocity * dt

        # Check convergence
        if step > 100 and torch.max(torch.abs(velocity)).item() < 0.01 * V0:
            break

    # Final dimensions
    final_length = L0 + displacement[-1].item() - displacement[0].item()

    # Ensure positive length
    if final_length <= 0:
        final_length = L0 * 0.5  # fallback

    # Volume conservation for diameter
    total_volume = area * L0
    final_area = total_volume / final_length
    final_diameter = math.sqrt(4.0 * final_area / math.pi)

    return TaylorBarResult(
        initial_velocity=V0,
        initial_length=L0,
        initial_diameter=D0,
        dynamic_yield=Y0,
        final_length=final_length,
        final_diameter=final_diameter,
        final_length_ratio=final_length / L0,
        final_diameter_ratio=final_diameter / D0,
        plastic_zone_length=float(torch.sum(ep > 1e-6).item()) * dx,
    )
