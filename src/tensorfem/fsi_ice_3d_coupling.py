"""Quarantined 3D fluid/structure/ice prototype; no coupled solver is available.

The former stepping path used stiffness as mass, never advanced a fluid state,
and returned an unconditional zero mass residual. It cannot supply simulation
or validation evidence. Configuration, shape functions and yield-surface
helpers remain importable for inspection; they do not qualify an ice model.
See docs/fsi3d-quarantine.md for the reopening criteria.
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
    
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
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
        if not torch.isfinite(detJ) or detJ <= 0:
            raise ValueError("HEX8 requires a positive finite Jacobian")
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
        
        raise FSI3DUnavailableError(
            "Plastic return mapping is unqualified; no stress update is available."
        )


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


class FSI3DUnavailableError(NotImplementedError):
    """A quarantined path cannot produce physical evidence."""


def fsi3d_qualification_status() -> dict:
    """Machine-readable admission verdict; missing physics never counts as PASS."""
    return {
        "schema": "tensorfem.fsi3d-qualification/1.0",
        "status": "blocked",
        "passed": False,
        "scope": "experimental configuration and local helpers only",
        "missing": [
            "consistent ice mass and displacement/velocity/acceleration update",
            "verified constitutive integration and damage evolution",
            "fluid equations and conservative interface load transfer",
            "measured mass flux residual and work/energy balance",
            "mesh/time convergence and independent ship-ice references",
        ],
    }


def _unavailable() -> None:
    raise FSI3DUnavailableError(
        "FSI3D is blocked: no qualified coupled solver; see docs/fsi3d-quarantine.md"
    )


class FSI3DIceCouplingSolver:
    """Compatibility entry point that fails before allocating or emitting results.

    Method signatures remain for callers and for older serialized instances;
    no opt-in bypass can reactivate the dimensionally incorrect prototype.
    """

    def __init__(self, config: SimulationConfig):
        config.validate()
        _unavailable()

    def solve_ice_step(self, contact_forces: torch.Tensor, dt: float) -> None:
        _unavailable()

    def compute_contact_forces(self, ship_position: torch.Tensor) -> torch.Tensor:
        _unavailable()

    def compute_energy_balance(self) -> dict:
        _unavailable()

    def step(self, ship_position: torch.Tensor) -> None:
        _unavailable()

    def run(self, ship_trajectory: Callable[[float], torch.Tensor]) -> dict:
        _unavailable()


def compute_mass_conservation_residual(fluid_velocity: torch.Tensor, dt: float) -> float:
    """Reject: velocity and dt alone lack the mesh/flux data needed for a residual."""
    raise FSI3DUnavailableError(
        "Mass residual requires discrete fluid fluxes and boundary geometry; unavailable."
    )


def check_ice_failure(stress: torch.Tensor, config: IceMaterialConfig) -> bool:
    """Check if ice fails under given stress state."""
    plasticity = DruckerPragerPlasticity(config)
    f = plasticity.yield_function(stress)
    return f > 0
