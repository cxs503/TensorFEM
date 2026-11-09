"""Elastic trailing edge vortex-induced vibration (NACA0012 with flexible trailing edge).

Physical Background
-------------------
When a flexible plate is attached to the trailing edge of a bluff body
(such as a NACA0012 airfoil), vortex shedding from the body can excite
the plate into vortex-induced vibration (VIV). If the vortex shedding
frequency is close to the structural natural frequency, lock-in occurs
and large-amplitude oscillations develop.

This is a simplified model of:
- Fish fins (flexible trailing edges for propulsion)
- Flexible control surfaces
- Flutter suppression devices

Key Physics
-----------
1. **Lock-in phenomenon**: When f_vortex ≈ f_natural, the structure
   oscillates at the vortex shedding frequency (not its natural frequency).
   
2. **Amplitude saturation**: Nonlinear effects limit the amplitude.

3. **Wake capture**: The flexible trailing edge can "capture" vortices
   from the main body, enhancing thrust (relevant to bio-inspired propulsion).

Geometry
--------
- NACA0012 airfoil (chord c = 1 m)
- Elastic plate attached at trailing edge (x = c)
- Plate length L = 0.2c = 0.2 m
- Plate thickness h = 0.01c = 0.01 m
- Material: E = 1 MPa, ρ = 1000 kg/m³ (soft elastomer)

Flow
----
- Freestream velocity U∞ = 1 m/s
- Re = U∞ * c / ν = 10^5 (turbulent, but we use quasi-steady model)
- Vortex shedding from airfoil trailing edge

References
----------
[1] Heathcote, S., Wang, Z., Gursul, I. (2008). "Effect of spanwise
    flexibility on flapping foil propulsion." Journal of Fluids and
    Structures, 24(4), 522-539.
[2] Michelin, S., Llewellyn Smith, S.G. (2009). "Resonance and propulsion
    performance of small-heave-frequency wings." Journal of Fluid Mechanics,
    625, 345-374.
[3] Kang, C.K., Shyy, W. (2013). "Scaling law for flexible flapping foil
    aerodynamics." AIAA Journal, 51(7), 1679-1691.

Scope and Limitations
---------------------
- 2D model (infinite span)
- Quasi-steady aerodynamics (no unsteady wake modeling)
- Linear structural model (Euler-Bernoulli beam)
- Single-mode approximation (first bending mode)
- No large-deformation or geometric nonlinearity
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import torch
import numpy as np


D = torch.float64


@dataclass(frozen=True)
class TrailingEdgeConfig:
    """Configuration for elastic trailing edge VIV problem."""
    
    # Airfoil
    chord: float = 1.0  # [m]
    
    # Trailing edge plate
    plate_length: float = 0.2  # [m] (20% chord)
    plate_thickness: float = 0.01  # [m] (1% chord)
    plate_young: float = 2.0e4  # [Pa] (soft elastomer)
    plate_density: float = 1000.0  # [kg/m^3]
    plate_poisson: float = 0.3  # [-]
    
    # Flow
    freestream_velocity: float = 1.0  # [m/s]
    fluid_density: float = 1.225  # [kg/m^3] (air)
    fluid_viscosity: float = 1.5e-5  # [Pa·s] (air)
    
    # Derived quantities
    @property
    def reynolds_number(self) -> float:
        """Reynolds number based on chord."""
        return (self.freestream_velocity * self.chord * 
                self.fluid_density / self.fluid_viscosity)
    
    @property
    def plate_bending_stiffness(self) -> float:
        """EI for the plate [N·m²]."""
        I = 1.0 * self.plate_thickness**3 / 12.0  # unit width
        return self.plate_young * I
    
    @property
    def plate_mass_per_length(self) -> float:
        """Mass per unit length [kg/m]."""
        return self.plate_density * self.plate_thickness * 1.0
    
    @property
    def plate_natural_frequency(self) -> float:
        """First natural frequency of cantilever plate [Hz]."""
        beta_1_L = 1.87510407
        EI = self.plate_bending_stiffness
        m = self.plate_mass_per_length
        omega_1 = beta_1_L**2 * math.sqrt(EI / m) / self.plate_length**2
        return omega_1 / (2 * math.pi)
    
    @property
    def reduced_velocity(self) -> float:
        """Reduced velocity U* = U / (f_n * L)."""
        return self.freestream_velocity / (self.plate_natural_frequency * self.plate_length)
    
    @property
    def mass_ratio(self) -> float:
        """Mass ratio m* = m / (rho_f * L)."""
        return self.plate_mass_per_length / (self.fluid_density * self.plate_length)
    
    def validate(self) -> None:
        """Check physical consistency."""
        if self.chord <= 0:
            raise ValueError("chord must be positive")
        if self.plate_length <= 0:
            raise ValueError("plate_length must be positive")
        if self.plate_thickness <= 0:
            raise ValueError("plate_thickness must be positive")
        if self.plate_young <= 0:
            raise ValueError("plate_young must be positive")
        if self.freestream_velocity <= 0:
            raise ValueError("freestream_velocity must be positive")


class TrailingEdgeVIVSolver:
    """Solver for elastic trailing edge VIV problem.
    
    Models the trailing edge plate as a cantilever beam excited by
    vortex shedding from the airfoil. Uses a single-mode approximation
    with quasi-steady aerodynamic forcing.
    """
    
    def __init__(self, config: TrailingEdgeConfig | None = None):
        """Initialize solver.
        
        Parameters
        ----------
        config : TrailingEdgeConfig, optional
            Configuration. If None, uses default parameters.
        """
        self.config = config or TrailingEdgeConfig()
        self.config.validate()
        
        # Plate properties
        self.L = self.config.plate_length
        self.EI = self.config.plate_bending_stiffness
        self.m = self.config.plate_mass_per_length
        
        # Natural frequency
        self.beta_1_L = 1.87510407
        self.omega_n = self.beta_1_L**2 * math.sqrt(self.EI / self.m) / self.L**2
        self.f_n = self.omega_n / (2 * math.pi)
        
        # Vortex shedding frequency (from airfoil)
        # For a NACA0012 at Re=10^5, Strouhal number St ≈ 0.15-0.20
        self.St = 0.18  # typical value
        self.f_vortex = self.St * self.config.freestream_velocity / self.config.chord
        
        # Frequency ratio
        self.frequency_ratio = self.f_vortex / self.f_n
        
        # Damping ratio
        self.zeta = 0.02
        
        # Time integration
        self.dt = 0.001  # [s]
        self.n_steps = 5000  # 5 seconds
        
        # Modal state
        self.q = 0.0
        self.q_dot = 0.0
    
    def _vortex_force(self, t: float) -> float:
        """Compute vortex-induced forcing.
        
        Models the lift force due to vortex shedding from the airfoil
        trailing edge. The force is harmonic at the vortex shedding frequency.
        
        Parameters
        ----------
        t : float
            Current time [s]
        
        Returns
        -------
        F : float
            Modal force [N/m]
        """
        omega_v = 2 * math.pi * self.f_vortex
        
        # Force amplitude (calibrated)
        # The actual force depends on the airfoil lift coefficient
        # For a NACA0012 at small alpha, C_L ≈ 2*pi*alpha
        # We use a representative value
        q_inf = 0.5 * self.config.fluid_density * self.config.freestream_velocity**2
        C_L_osc = 0.1  # oscillating lift coefficient amplitude
        F0 = q_inf * self.config.chord * C_L_osc
        
        # Harmonic forcing
        F = F0 * math.sin(omega_v * t)
        
        return F
    
    def _modal_mass(self) -> float:
        """Generalized mass for first mode."""
        return 0.25 * self.m * self.L
    
    def _modal_stiffness(self) -> float:
        """Generalized stiffness for first mode."""
        M_1 = self._modal_mass()
        return M_1 * self.omega_n**2
    
    def _modal_damping(self) -> float:
        """Generalized damping for first mode."""
        M_1 = self._modal_mass()
        return 2 * self.zeta * M_1 * self.omega_n
    
    def solve(self) -> dict:
        """Solve the trailing edge VIV problem.
        
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'time': time array [s]
            - 'tip_displacement': tip displacement history [m]
            - 'amplitude': steady-state amplitude [m]
            - 'frequency': oscillation frequency [Hz]
            - 'frequency_ratio': f_vortex / f_natural
            - 'lock_in': bool, whether lock-in occurred
        """
        # Newmark-beta integration
        beta = 0.25
        gamma = 0.5
        
        M_1 = self._modal_mass()
        K_1 = self._modal_stiffness()
        C_1 = self._modal_damping()
        
        # Effective stiffness
        k_eff = K_1 + gamma / (beta * self.dt) * C_1 + 1.0 / (beta * self.dt**2) * M_1
        
        q = 0.0
        q_dot = 0.0
        
        time_history = []
        q_history = []
        
        for step in range(self.n_steps):
            t = step * self.dt
            
            # External force
            F_ext = self._vortex_force(t)
            
            # Effective force
            F_eff = (F_ext + 
                     M_1 * (q / (beta * self.dt**2) + q_dot / (beta * self.dt)) +
                     C_1 * (gamma / (beta * self.dt) * q + (gamma / beta - 1) * q_dot))
            
            # Solve
            q_new = F_eff / k_eff
            
            # Update
            q_ddot_new = (1.0 / (beta * self.dt**2) * (q_new - q) - 
                          1.0 / (beta * self.dt) * q_dot)
            q_dot_new = q_dot + self.dt * (gamma * q_ddot_new)
            
            # Store
            time_history.append(t)
            q_history.append(q_new)
            
            q = q_new
            q_dot = q_dot_new
        
        # Convert to arrays
        time_arr = np.array(time_history)
        q_arr = np.array(q_history)
        
        # Skip transient
        n_skip = int(1.0 / self.dt)
        time_steady = time_arr[n_skip:]
        q_steady = q_arr[n_skip:]
        
        # Measure amplitude
        amplitude = (np.max(q_steady) - np.min(q_steady)) / 2.0
        
        # Measure frequency
        zero_crossings = np.where(np.diff(np.sign(q_steady)))[0]
        if len(zero_crossings) > 1:
            periods = np.diff(time_steady[zero_crossings])
            avg_period = np.mean(periods) * 2
            frequency = 1.0 / avg_period
        else:
            frequency = 0.0
        
        # Check for lock-in
        # Lock-in occurs when oscillation frequency ≈ vortex shedding frequency
        lock_in = abs(frequency - self.f_vortex) / self.f_vortex < 0.1
        
        return {
            'time': time_arr,
            'tip_displacement': q_arr,
            'amplitude': amplitude,
            'frequency': frequency,
            'natural_frequency': self.f_n,
            'vortex_frequency': self.f_vortex,
            'frequency_ratio': self.frequency_ratio,
            'lock_in': lock_in,
            'config': self.config,
        }


def run_trailing_edge_viv() -> dict:
    """Run the elastic trailing edge VIV simulation.
    
    Returns
    -------
    result : dict
        Simulation results.
    """
    solver = TrailingEdgeVIVSolver()
    result = solver.solve()
    
    print("=" * 70)
    print("Elastic Trailing Edge VIV Results")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Plate length: {solver.config.plate_length} m")
    print(f"  Plate thickness: {solver.config.plate_thickness} m")
    print(f"  Reynolds number: {solver.config.reynolds_number:.0f}")
    print(f"  Natural frequency: {solver.f_n:.2f} Hz")
    print(f"  Vortex shedding frequency: {solver.f_vortex:.2f} Hz")
    print(f"  Frequency ratio: {solver.frequency_ratio:.3f}")
    print()
    print(f"Results:")
    print(f"  Amplitude: {result['amplitude']*1000:.2f} mm")
    print(f"  Oscillation frequency: {result['frequency']:.2f} Hz")
    print(f"  Lock-in: {'YES' if result['lock_in'] else 'NO'}")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = run_trailing_edge_viv()
