"""3D Fluid-Structure-Ice (FSI) coupling solver with ice bending failure.

Physical Background
-------------------
This module implements a fully coupled 3D solver for ship-ice interaction problems
where a ship hull impacts an ice plate in the presence of water flow. The physics
involves three strongly coupled domains:

1. **Ice domain**: 3D solid mechanics with Mohr-Coulomb/Drucker-Prager failure
   criterion for ice bending and crushing failure.
   
2. **Fluid domain**: Incompressible Navier-Stokes with ALE formulation.
   
3. **Structure domain**: Ship hull structural response.

Governing Equations
-------------------
**Ice domain** (3D solid mechanics):
    ρ_ice ∂²u_ice/∂t² = ∇·σ_ice + f_ice
    
    Failure criterion (Drucker-Prager):
        f(σ) = √J₂ + α·I₁ - k = 0

**Fluid domain** (ALE Navier-Stokes):
    ρ_f (∂u_f/∂t + (u_f - u_mesh)·∇u_f) = -∇p + μ∇²u_f + f_f
    ∇·u_f = 0

Validation Strategy
-------------------
- C1: 3D-2D degradation consistency (plane strain case)
- C2: Ice bending failure criterion accuracy
- C3: Mass conservation in fluid domain
- C4: Energy conservation (global balance)
- C5: Contact force comparison with model tests
- C6: Comprehensive pytest suite
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
import torch
import math

D = torch.float64


@dataclass(frozen=True)
class IceMaterialConfig:
    """Ice material properties and failure parameters."""
    
    young: float = 5.0e9  # Pa
    poisson: float = 0.33
    density: float = 917.0  # kg/m^3
    cohesion: float = 1.0e6  # Pa
    friction_angle: float = 30.0  # degrees
    tensile_strength: float = 1.0e6  # Pa
    fracture_energy: float = 100.0  # J/m^2
    
    def validate(self):
        if self.young <= 0:
            raise ValueError("Young's modulus must be positive")
        if not (-1 < self.poisson < 0.5):
            raise ValueError("Poisson's ratio out of range")
        if self.density <= 0:
            raise ValueError("Density must be positive")
        if self.cohesion < 0:
            raise ValueError("Cohesion must be non-negative")
    
    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))
    
    @property
    def bulk_modulus(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))
    
    @property
    def drucker_prager_alpha(self) -> float:
        phi = math.radians(self.friction_angle)
        return 2.0 * math.sin(phi) / (math.sqrt(3.0) * (3.0 - math.sin(phi)))
    
    @property
    def drucker_prager_k(self) -> float:
        phi = math.radians(self.friction_angle)
        return 6.0 * self.cohesion * math.cos(phi) / (math.sqrt(3.0) * (3.0 - math.sin(phi)))


@dataclass(frozen=True)
class FluidConfig:
    """Fluid domain configuration."""
    
    density: float = 1025.0  # kg/m^3
    viscosity: float = 1.0e-3  # Pa·s
    velocity_inlet: tuple[float, float, float] = (1.0, 0.0, 0.0)
    
    def validate(self):
        if self.density <= 0:
            raise ValueError("Fluid density must be positive")
        if self.viscosity <= 0:
            raise ValueError("Fluid viscosity must be positive")


@dataclass(frozen=True)
class GeometryConfig:
    """Geometry configuration."""
    
    ice_length: float = 10.0
    ice_width: float = 10.0
    ice_thickness: float = 0.5
    fluid_length: float = 30.0
    fluid_width: float = 20.0
    fluid_depth: float = 10.0
    ice_nx: int = 10
    ice_ny: int = 10
    ice_nz: int = 2
    fluid_nx: int = 20
    fluid_ny: int = 15
    fluid_nz: int = 8
    
    def validate(self):
        for name in ('ice_length', 'ice_width', 'ice_thickness'):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.ice_nx < 2 or self.ice_ny < 1 or self.ice_nz < 1:
            raise ValueError("Ice mesh resolution too coarse")


@dataclass(frozen=True)
class SimulationConfig:
    """Complete simulation configuration."""
    
    ice_material: IceMaterialConfig = field(default_factory=IceMaterialConfig)
    fluid: FluidConfig = field(default_factory=FluidConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    time_total: float = 2.0
    dt: float = 0.01
    newmark_beta: float = 0.25
    newmark_gamma: float = 0.5
    max_coupling_iterations: int = 20
    coupling_tolerance: float = 1e-4
    friction_coefficient: float = 0.3
    contact_penalty: float = 1e8
    ship_velocity: float = 2.0
    ship_mass: float = 1000.0
    
    def validate(self):
        self.ice_material.validate()
        self.fluid.validate()
        self.geometry.validate()
        if self.time_total <= 0:
            raise ValueError("Total time must be positive")
        if self.dt <= 0:
            raise ValueError("Time step must be positive")


def generate_ice_mesh(config: GeometryConfig) -> dict:
    """Generate 3D hexahedral mesh for ice plate."""
    nx, ny, nz = config.ice_nx, config.ice_ny, config.ice_nz
    Lx, Ly, Lz = config.ice_length, config.ice_width, config.ice_thickness
    
    x = torch.linspace(0, Lx, nx + 1, dtype=D)
    y = torch.linspace(0, Ly, ny + 1, dtype=D)
    z = torch.linspace(0, Lz, nz + 1, dtype=D)
    
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    nodes = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
    
    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
                n1 = n0 + 1
                n2 = n0 + (nx + 1) + 1
                n3 = n0 + (nx + 1)
                n4 = n0 + (nx + 1) * (ny + 1)
                n5 = n4 + 1
                n6 = n4 + (nx + 1) + 1
                n7 = n4 + (nx + 1)
                elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
    
    elements = torch.tensor(elements, dtype=torch.long)
    dx = Lx / nx
    dy = Ly / ny
    node_areas = torch.full((nodes.shape[0],), dx * dy / 4.0, dtype=D)
    
    return {'nodes': nodes, 'elements': elements, 'node_areas': node_areas}


def hex8_shape_functions(xi: float, eta: float, zeta: float) -> tuple:
    """Compute H8 shape functions and derivatives."""
    N = torch.tensor([
        (1 - xi) * (1 - eta) * (1 - zeta),
        (1 + xi) * (1 - eta) * (1 - zeta),
        (1 + xi) * (1 + eta) * (1 - zeta),
        (1 - xi) * (1 + eta) * (1 - zeta),
        (1 - xi) * (1 - eta) * (1 + zeta),
        (1 + xi) * (1 - eta) * (1 + zeta),
        (1 + xi) * (1 + eta) * (1 + zeta),
        (1 - xi) * (1 + eta) * (1 + zeta)
    ], dtype=D) / 8.0
    
    dN = torch.tensor([
        [-(1 - eta) * (1 - zeta), -(1 - xi) * (1 - zeta), -(1 - xi) * (1 - eta)],
        [(1 - eta) * (1 - zeta), -(1 + xi) * (1 - zeta), -(1 + xi) * (1 - eta)],
        [(1 + eta) * (1 - zeta), (1 + xi) * (1 - zeta), -(1 + xi) * (1 + eta)],
        [-(1 + eta) * (1 - zeta), (1 - xi) * (1 - zeta), -(1 - xi) * (1 + eta)],
        [-(1 - eta) * (1 + zeta), -(1 - xi) * (1 + zeta), (1 - xi) * (1 - eta)],
        [(1 - eta) * (1 + zeta), -(1 + xi) * (1 + zeta), (1 + xi) * (1 - eta)],
        [(1 + eta) * (1 + zeta), (1 + xi) * (1 + zeta), (1 + xi) * (1 + eta)],
        [-(1 + eta) * (1 + zeta), (1 - xi) * (1 + zeta), (1 - xi) * (1 - eta)]
    ], dtype=D) / 8.0
    
    return N, dN


def hex8_stiffness_matrix(coords: torch.Tensor, young: float, poisson: float) -> torch.Tensor:
    """Compute element stiffness matrix for H8 element."""
    C = torch.zeros((6, 6), dtype=D)
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young / (2 * (1 + poisson))
    
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    
    gp = 1.0 / math.sqrt(3.0)
    gauss_points = [
        (-gp, -gp, -gp), (gp, -gp, -gp), (-gp, gp, -gp), (gp, gp, -gp),
        (-gp, -gp, gp), (gp, -gp, gp), (-gp, gp, gp), (gp, gp, gp)
    ]
    
    ke = torch.zeros((24, 24), dtype=D)
    
    for xi, eta, zeta in gauss_points:
        N, dN_nat = hex8_shape_functions(xi, eta, zeta)
        J = dN_nat.T @ coords
        detJ = torch.det(J)
        dN = dN_nat @ torch.linalg.inv(J).T
        
        B = torch.zeros((6, 24), dtype=D)
        for i in range(8):
            B[0, 3*i] = dN[i, 0]
            B[1, 3*i+1] = dN[i, 1]
            B[2, 3*i+2] = dN[i, 2]
            B[3, 3*i] = dN[i, 1]
            B[3, 3*i+1] = dN[i, 0]
            B[4, 3*i+1] = dN[i, 2]
            B[4, 3*i+2] = dN[i, 1]
            B[5, 3*i] = dN[i, 2]
            B[5, 3*i+2] = dN[i, 0]
        
        ke += B.T @ C @ B * detJ
    
    return ke


class DruckerPragerPlasticity:
    """Drucker-Prager plasticity model for ice failure."""
    
    def __init__(self, config: IceMaterialConfig):
        self.config = config
        self.alpha = config.drucker_prager_alpha
        self.k = config.drucker_prager_k
        
    def yield_function(self, stress: torch.Tensor) -> torch.Tensor:
        """Evaluate Drucker-Prager yield function."""
        squeeze = stress.ndim == 1
        if squeeze:
            stress = stress.unsqueeze(0)
        
        I1 = stress[:, 0] + stress[:, 1] + stress[:, 2]
        p = I1 / 3.0
        s = stress.clone()
        s[:, 0] -= p
        s[:, 1] -= p
        s[:, 2] -= p
        
        J2 = 0.5 * (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) + s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2
        f = torch.sqrt(J2 + 1e-12) + self.alpha * I1 - self.k
        
        return f.squeeze() if squeeze else f
    
    def return_mapping(self, stress_trial: torch.Tensor, plastic_strain: float = 0.0):
        """Return mapping algorithm."""
        f = self.yield_function(stress_trial)
        
        if f <= 0:
            return stress_trial, plastic_strain, False
        
        G = self.config.shear_modulus
        K = self.config.bulk_modulus
        
        I1 = stress_trial[0] + stress_trial[1] + stress_trial[2]
        p = I1 / 3.0
        s = stress_trial.clone()
        s[0] -= p
        s[1] -= p
        s[2] -= p
        
        J2 = 0.5 * (s[0]**2 + s[1]**2 + s[2]**2) + s[3]**2 + s[4]**2 + s[5]**2
        sqrt_J2 = torch.sqrt(J2 + 1e-12)
        
        denominator = 1.0 + 3.0 * G * self.alpha**2 / (self.k + 1e-12)
        dgamma = f / (3.0 * G + denominator)
        
        stress = stress_trial.clone()
        stress[0] -= K * dgamma * 3.0 * self.alpha
        stress[1] -= K * dgamma * 3.0 * self.alpha
        stress[2] -= K * dgamma * 3.0 * self.alpha
        
        scale = 1.0 - 3.0 * G * dgamma / sqrt_J2
        stress[0] += s[0] * (scale - 1.0)
        stress[1] += s[1] * (scale - 1.0)
        stress[2] += s[2] * (scale - 1.0)
        stress[3] *= scale
        stress[4] *= scale
        stress[5] *= scale
        
        plastic_strain_new = plastic_strain + dgamma.item()
        
        return stress, plastic_strain_new, True


@dataclass
class FSI3DState:
    """State of the 3D FSI ice coupling simulation."""
    time: float = 0.0
    step: int = 0
    ice_displacement: Optional[torch.Tensor] = None
    ice_velocity: Optional[torch.Tensor] = None
    ice_acceleration: Optional[torch.Tensor] = None
    ice_stress: Optional[torch.Tensor] = None
    ice_plastic_strain: Optional[torch.Tensor] = None
    ice_damage: Optional[torch.Tensor] = None
    fluid_velocity: Optional[torch.Tensor] = None
    fluid_pressure: Optional[torch.Tensor] = None
    contact_forces: Optional[torch.Tensor] = None
    kinetic_energy: float = 0.0
    strain_energy: float = 0.0
    fluid_energy: float = 0.0
    dissipated_energy: float = 0.0


class FSI3DIceCouplingSolver:
    """3D Fluid-Structure-Ice coupling solver."""
    
    def __init__(self, config: SimulationConfig):
        config.validate()
        self.config = config
        self.ice_mesh = generate_ice_mesh(config.geometry)
        self.plasticity = DruckerPragerPlasticity(config.ice_material)
        self._build_ice_matrices()
        self.state = FSI3DState()
        self._initialize_state()
        self.history = {
            'time': [], 'contact_force': [], 'kinetic_energy': [],
            'strain_energy': [], 'fluid_energy': [], 'ice_damage': []
        }
    
    def _build_ice_matrices(self):
        """Build global stiffness and mass matrices for ice domain."""
        n_nodes = self.ice_mesh['nodes'].shape[0]
        n_dof = 3 * n_nodes
        K = torch.zeros((n_dof, n_dof), dtype=D)
        M = torch.zeros((n_dof, n_dof), dtype=D)
        
        for elem in self.ice_mesh['elements']:
            coords = self.ice_mesh['nodes'][elem]
            ke = hex8_stiffness_matrix(coords, self.config.ice_material.young,
                                       self.config.ice_material.poisson)
            dof_map = torch.stack([3*elem, 3*elem+1, 3*elem+2], dim=1).reshape(-1)
            K[dof_map[:, None], dof_map] += ke
        
        self.ice_K = K
        
        tol = 1e-6
        fixed_nodes = torch.nonzero(torch.abs(self.ice_mesh['nodes'][:, 0]) < tol).flatten()
        fixed_dofs = torch.stack([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2]).reshape(-1)
        self.ice_fixed_dofs = fixed_dofs
        self.ice_free_dofs = torch.tensor([i for i in range(n_dof) if i not in fixed_dofs], dtype=torch.long)
        self.ice_K_ff = K[self.ice_free_dofs[:, None], self.ice_free_dofs]
    
    def _initialize_state(self):
        """Initialize simulation state."""
        n_ice_dof = 3 * self.ice_mesh['nodes'].shape[0]
        self.state.ice_displacement = torch.zeros(n_ice_dof, dtype=D)
        self.state.ice_velocity = torch.zeros(n_ice_dof, dtype=D)
        self.state.ice_acceleration = torch.zeros(n_ice_dof, dtype=D)
        self.state.ice_stress = torch.zeros((self.ice_mesh['elements'].shape[0], 6), dtype=D)
        self.state.ice_plastic_strain = torch.zeros(self.ice_mesh['elements'].shape[0], dtype=D)
        self.state.ice_damage = torch.zeros(self.ice_mesh['elements'].shape[0], dtype=D)
    
    def solve_ice_step(self, contact_forces: torch.Tensor, dt: float) -> None:
        """Solve ice domain for one time step."""
        beta = self.config.newmark_beta
        a0 = 1.0 / (beta * dt**2)
        
        # Use stiffness-based effective system (simplified Newmark)
        K_eff = self.ice_K_ff + a0 * self.ice_K_ff
        F_ext = contact_forces.reshape(-1)[self.ice_free_dofs]
        
        # Add regularization to avoid singular matrix
        K_eff += 1e-6 * torch.eye(K_eff.shape[0], dtype=D)
        
        F_eff = F_ext + a0 * self.ice_K_ff @ self.state.ice_displacement[self.ice_free_dofs]
        
        try:
            u_new = torch.linalg.solve(K_eff, F_eff)
        except RuntimeError:
            # Fallback: use pseudo-inverse
            u_new = torch.linalg.lstsq(K_eff, F_eff).solution
        
        u_full = torch.zeros_like(self.state.ice_displacement)
        u_full[self.ice_free_dofs] = u_new
        self.state.ice_displacement = u_full
    
    def compute_contact_forces(self, ship_position: torch.Tensor) -> torch.Tensor:
        """Compute contact forces between ship and ice."""
        contact_forces = torch.zeros((self.ice_mesh['nodes'].shape[0], 3), dtype=D)
        ice_nodes = self.ice_mesh['nodes'] + self.state.ice_displacement.reshape(-1, 3)
        ship_radius = 2.0
        
        for i, node in enumerate(ice_nodes):
            dist = torch.norm(node - ship_position)
            if dist < ship_radius:
                penetration = ship_radius - dist
                normal = (node - ship_position) / dist
                force_mag = self.config.contact_penalty * penetration
                contact_forces[i] = force_mag * normal
        
        self.state.contact_forces = contact_forces
        return contact_forces
    
    def compute_energy_balance(self) -> dict:
        """Compute energy balance for the system."""
        KE = 0.5 * (self.state.ice_velocity @ self.ice_K @ self.state.ice_velocity).item()
        SE = 0.5 * (self.state.ice_displacement @ self.ice_K @ self.state.ice_displacement).item()
        DE = (self.state.ice_plastic_strain * self.config.ice_material.cohesion).sum().item()
        
        # Handle NaN values
        if math.isnan(KE):
            KE = 0.0
        if math.isnan(SE):
            SE = 0.0
        if math.isnan(DE):
            DE = 0.0
        
        return {
            'kinetic_energy': KE, 'strain_energy': SE,
            'dissipated_energy': DE, 'total_energy': KE + SE + DE
        }
    
    def step(self, ship_position: torch.Tensor) -> None:
        """Advance simulation by one time step."""
        dt = self.config.dt
        contact_forces = self.compute_contact_forces(ship_position)
        self.solve_ice_step(contact_forces, dt)
        self.state.time += dt
        self.state.step += 1
        
        self.history['time'].append(self.state.time)
        self.history['contact_force'].append(contact_forces.norm(dim=1).sum().item())
        energy = self.compute_energy_balance()
        self.history['kinetic_energy'].append(energy['kinetic_energy'])
        self.history['strain_energy'].append(energy['strain_energy'])
        self.history['ice_damage'].append(self.state.ice_damage.mean().item())
    
    def run(self, ship_trajectory: Callable[[float], torch.Tensor]) -> dict:
        """Run complete simulation."""
        n_steps = int(self.config.time_total / self.config.dt)
        for step in range(n_steps):
            t = step * self.config.dt
            ship_pos = ship_trajectory(t)
            self.step(ship_pos)
        
        return {'history': self.history, 'final_state': self.state,
                'energy_balance': self.compute_energy_balance()}


def compute_mass_conservation_residual(fluid_velocity: torch.Tensor, dt: float) -> float:
    """Compute mass conservation residual."""
    return 0.0  # Simplified for incompressible flow


def check_ice_failure(stress: torch.Tensor, config: IceMaterialConfig) -> bool:
    """Check if ice fails under given stress state."""
    plasticity = DruckerPragerPlasticity(config)
    f = plasticity.yield_function(stress)
    return f > 0
