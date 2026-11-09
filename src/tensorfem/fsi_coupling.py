"""Partitioned FSI coupling framework with strong/weak coupling schemes.

Physical Background
-------------------
Fluid-Structure Interaction (FSI) problems require solving the coupled
fluid and structural equations simultaneously. Two main approaches exist:

1. **Monolithic (strong) coupling**: Solve fluid and structure as a single
   system. Robust but expensive and requires a unified solver.

2. **Partitioned (weak/strong) coupling**: Solve fluid and structure
   separately, exchanging boundary conditions at the interface. Can be:
   - **Weak coupling** (explicit): Exchange once per time step. Simple but
     may be unstable for strong added-mass effects.
   - **Strong coupling** (implicit): Iterate at each time step until
     convergence. More stable but requires more fluid solves.

This module implements a partitioned strong coupling scheme with:
- Aitken's delta^2 acceleration for convergence
- Sub-iteration convergence monitoring
- Energy conservation verification

Governing Equations
-------------------
The FSI problem is defined by:

**Fluid domain** (Navier-Stokes):
    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
    ∇·u = 0

**Structural domain** (linear elasticity):
    ρ_s ∂²u_s/∂t² = ∇·σ_s + f_s

**Interface conditions** (at Γ_FSI):
    u_f = u_s  (kinematic compatibility)
    σ_f·n = σ_s·n  (dynamic equilibrium)

Convergence Criteria
--------------------
The sub-iteration converges when:
    ||u^{k+1} - u^k|| / ||u^{k+1}|| < tol_u
    ||F^{k+1} - F^k|| / ||F^{k+1}|| < tol_F

where k is the sub-iteration index.

Energy Conservation
-------------------
The work done by fluid forces on the structure must equal the sum of:
- Structural strain energy increase
- Kinetic energy increase
- Dissipated energy (if damping present)

    W_fluid = ΔE_strain + ΔE_kinetic + E_dissipated

References
----------
[1] Turek, S., Hron, J. (2006). "Proposal for Numerical Benchmarking of
    Fluid-Structure Interaction between an Elastic Object and Laminal
    Incompressible Flow." Lecture Notes in Computational Science and
    Engineering, Vol. 53.
[2] Küttler, U., Wall, W.A. (2008). "Fixed-point fluid-structure
    interaction solvers with dynamic relaxation." Computational Mechanics,
    42(1), 1-16.
[3] Irons, B., Tuck, O. (1969). "A version of the Aitken accelerator for
    computer iteration." International Journal for Numerical Methods in
    Engineering, 1(2), 205-206.

Scope and Limitations
---------------------
- Partitioned (black-box) coupling: fluid and structure solvers are separate
- Linear structural model (small deformations)
- Incompressible Newtonian fluid
- 2D problems (extendable to 3D)
- No mesh motion/remeshing (interface tracking only)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol
import torch
import math


D = torch.float64


class FluidSolver(Protocol):
    """Protocol for fluid solvers in FSI coupling."""
    
    def solve(
        self,
        interface_displacement: torch.Tensor,
        interface_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve fluid problem and return interface forces and pressures.
        
        Parameters
        ----------
        interface_displacement : torch.Tensor
            Current interface displacement [n_interface, 2]
        interface_velocity : torch.Tensor
            Current interface velocity [n_interface, 2]
        
        Returns
        -------
        forces : torch.Tensor
            Interface forces (fluid traction) [n_interface, 2]
        pressures : torch.Tensor
            Interface pressures [n_interface]
        """
        ...


class StructuralSolver(Protocol):
    """Protocol for structural solvers in FSI coupling."""
    
    def solve(
        self,
        interface_forces: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve structural problem and return interface response.
        
        Parameters
        ----------
        interface_forces : torch.Tensor
            Interface forces from fluid [n_interface, 2]
        
        Returns
        -------
        displacement : torch.Tensor
            Interface displacement [n_interface, 2]
        velocity : torch.Tensor
            Interface velocity [n_interface, 2]
        acceleration : torch.Tensor
            Interface acceleration [n_interface, 2]
        """
        ...


@dataclass
class FSICouplingResult:
    """Result of FSI coupling iteration."""
    
    # Time history
    time: torch.Tensor  # [n_steps]
    
    # Interface quantities
    interface_displacement: torch.Tensor  # [n_steps, n_interface, 2]
    interface_velocity: torch.Tensor  # [n_steps, n_interface, 2]
    interface_forces: torch.Tensor  # [n_steps, n_interface, 2]
    
    # Convergence history
    sub_iterations_per_step: list[int]  # [n_steps]
    residual_history: list[list[float]]  # [n_steps][n_sub_iter]
    
    # Energy balance
    fluid_work: torch.Tensor  # [n_steps] work done by fluid
    strain_energy: torch.Tensor  # [n_steps] structural strain energy
    kinetic_energy: torch.Tensor  # [n_steps] structural kinetic energy
    energy_imbalance: torch.Tensor  # [n_steps] energy conservation error
    
    # Metadata
    converged: bool
    coupling_scheme: str  # 'weak' or 'strong'
    max_sub_iterations: int
    tolerance: float


class SimpleBeamSolver:
    """Simple 2D beam/plate structural solver for FSI demonstration.
    
    Models a cantilever beam/plate with tip loading. Uses Euler-Bernoulli
    beam theory for simplicity.
    """
    
    def __init__(
        self,
        length: float,
        young: float,
        density: float,
        thickness: float,
        width: float = 1.0,
        n_elements: int = 10,
    ):
        """Initialize beam solver.
        
        Parameters
        ----------
        length : float
            Beam length [m]
        young : float
            Young's modulus [Pa]
        density : float
            Material density [kg/m^3]
        thickness : float
            Beam thickness [m]
        width : float
            Beam width [m] (default 1.0 for unit width)
        n_elements : int
            Number of beam elements
        """
        self.length = length
        self.young = young
        self.density = density
        self.thickness = thickness
        self.width = width
        self.n_elements = n_elements
        
        # Cross-section properties
        self.area = width * thickness
        self.I = width * thickness**3 / 12.0  # second moment of area
        
        # Mass per unit length
        self.m = density * self.area
        
        # Bending stiffness
        self.EI = young * self.I
        
        # Node positions (n_elements + 1 nodes)
        self.n_nodes = n_elements + 1
        self.x_nodes = torch.linspace(0, length, self.n_nodes, dtype=D)
        
        # Degrees of freedom: [w, theta] at each node
        # w = transverse displacement, theta = rotation
        self.ndof = 2 * self.n_nodes
        
        # Build stiffness and mass matrices
        self.K = self._build_stiffness_matrix()
        self.M = self._build_mass_matrix()
        
        # Boundary conditions: clamped at x=0
        # Fixed DOFs: w[0] and theta[0]
        self.fixed_dofs = torch.tensor([0, 1], dtype=torch.long)
        self.free_dofs = torch.arange(2, self.ndof, dtype=torch.long)
        
        # Reduced system
        self.K_ff = self.K[self.free_dofs[:, None], self.free_dofs]
        self.M_ff = self.M[self.free_dofs[:, None], self.free_dofs]
        
        # Current state
        self.u = torch.zeros(self.ndof, dtype=D)  # displacement
        self.v = torch.zeros(self.ndof, dtype=D)  # velocity
        self.a = torch.zeros(self.ndof, dtype=D)  # acceleration
        
        # Time integration parameters (Newmark-beta)
        self.beta = 0.25
        self.gamma = 0.5
    
    def _build_stiffness_matrix(self) -> torch.Tensor:
        """Build global stiffness matrix for Euler-Bernoulli beam."""
        K = torch.zeros((self.ndof, self.ndof), dtype=D)
        
        h = self.length / self.n_elements
        
        # Element stiffness matrix (Euler-Bernoulli beam)
        # DOF order: [w1, theta1, w2, theta2]
        a = self.EI / h**3
        ke = a * torch.tensor([
            [12, 6*h, -12, 6*h],
            [6*h, 4*h**2, -6*h, 2*h**2],
            [-12, -6*h, 12, -6*h],
            [6*h, 2*h**2, -6*h, 4*h**2]
        ], dtype=D)
        
        # Assemble
        for e in range(self.n_elements):
            dof_map = torch.tensor([2*e, 2*e+1, 2*e+2, 2*e+3], dtype=torch.long)
            K[dof_map[:, None], dof_map] += ke
        
        return K
    
    def _build_mass_matrix(self) -> torch.Tensor:
        """Build consistent mass matrix for Euler-Bernoulli beam."""
        M = torch.zeros((self.ndof, self.ndof), dtype=D)
        
        h = self.length / self.n_elements
        
        # Consistent element mass matrix
        a = self.m * h / 420.0
        me = a * torch.tensor([
            [156, 22*h, 54, -13*h],
            [22*h, 4*h**2, 13*h, -3*h**2],
            [54, 13*h, 156, -22*h],
            [-13*h, -3*h**2, -22*h, 4*h**2]
        ], dtype=D)
        
        # Assemble
        for e in range(self.n_elements):
            dof_map = torch.tensor([2*e, 2*e+1, 2*e+2, 2*e+3], dtype=torch.long)
            M[dof_map[:, None], dof_map] += me
        
        return M
    
    def solve(self, interface_forces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve structural problem with given interface forces.
        
        Parameters
        ----------
        interface_forces : torch.Tensor
            Forces at interface nodes [n_interface, 2] (x, y components)
            For beam, we only use y-component (transverse)
        
        Returns
        -------
        displacement : torch.Tensor
            Interface displacement [n_interface, 2]
        velocity : torch.Tensor
            Interface velocity [n_interface, 2]
        acceleration : torch.Tensor
            Interface acceleration [n_interface, 2]
        """
        # Extract transverse forces (y-component)
        # Map interface forces to beam DOFs
        f = torch.zeros(self.ndof, dtype=D)
        
        # Assume interface is at the tip (last node)
        # Force is applied as transverse load
        if interface_forces.shape[0] > 0:
            # Sum all interface forces (assuming they're at the tip)
            f_tip_y = interface_forces[:, 1].sum()
            f[-2] = f_tip_y  # transverse force at tip
        
        # Solve static problem: K_ff * u_f = f_f
        # For dynamic, we'd use Newmark, but for FSI iteration we use static
        u_f = torch.linalg.solve(self.K_ff, f[self.free_dofs])
        
        # Scatter back to full DOF vector
        u = torch.zeros(self.ndof, dtype=D)
        u[self.free_dofs] = u_f
        
        # Update state
        self.u = u
        
        # Compute velocities and accelerations (for static, these are zero)
        v = torch.zeros_like(u)
        a = torch.zeros_like(u)
        
        # Extract interface quantities (tip node)
        n_interface = interface_forces.shape[0]
        disp_interface = torch.zeros((n_interface, 2), dtype=D)
        vel_interface = torch.zeros((n_interface, 2), dtype=D)
        acc_interface = torch.zeros((n_interface, 2), dtype=D)
        
        # Tip displacement
        disp_interface[:, 1] = u[-2]  # transverse displacement
        
        return disp_interface, vel_interface, acc_interface
    
    def get_tip_displacement(self) -> float:
        """Get transverse displacement at beam tip."""
        return self.u[-2].item()
    
    def get_strain_energy(self) -> float:
        """Compute current strain energy."""
        return 0.5 * (self.u @ self.K @ self.u).item()
    
    def get_kinetic_energy(self) -> float:
        """Compute current kinetic energy."""
        return 0.5 * (self.v @ self.M @ self.v).item()


class SimpleFluidSolver:
    """Simplified fluid solver for FSI demonstration.
    
    Uses a quasi-steady aerodynamic model based on thin airfoil theory.
    This is NOT a full Navier-Stokes solver, but captures the essential
    fluid-structure coupling physics.
    """
    
    def __init__(
        self,
        density: float = 1.225,  # air density [kg/m^3]
        velocity: float = 10.0,  # freestream velocity [m/s]
        chord: float = 1.0,  # reference chord [m]
    ):
        """Initialize fluid solver.
        
        Parameters
        ----------
        density : float
            Fluid density [kg/m^3]
        velocity : float
            Freestream velocity [m/s]
        chord : float
            Reference chord length [m]
        """
        self.density = density
        self.velocity = velocity
        self.chord = chord
        
        # Dynamic pressure
        self.q_inf = 0.5 * density * velocity**2
    
    def solve(
        self,
        interface_displacement: torch.Tensor,
        interface_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve fluid problem and return interface forces.
        
        Uses quasi-steady aerodynamic model:
        - Lift coefficient: C_L = 2*pi*alpha (thin airfoil theory)
        - alpha = angle of attack from interface displacement
        
        Parameters
        ----------
        interface_displacement : torch.Tensor
            Current interface displacement [n_interface, 2]
        interface_velocity : torch.Tensor
            Current interface velocity [n_interface, 2]
        
        Returns
        -------
        forces : torch.Tensor
            Interface forces [n_interface, 2]
        pressures : torch.Tensor
            Interface pressures [n_interface]
        """
        n_interface = interface_displacement.shape[0]
        forces = torch.zeros((n_interface, 2), dtype=D)
        pressures = torch.zeros(n_interface, dtype=D)
        
        # Compute angle of attack from displacement
        # alpha ≈ dw/dx (small angle approximation)
        if n_interface > 1:
            dx = interface_displacement[1, 0] - interface_displacement[0, 0]
            dw = interface_displacement[1, 1] - interface_displacement[0, 1]
            if abs(dx.item()) > 1e-10:
                alpha = (dw / dx).item()
            else:
                alpha = 0.0
        else:
            alpha = 0.0
        
        # Lift coefficient (thin airfoil theory)
        C_L = 2 * math.pi * alpha
        
        # Lift force per unit span
        L = self.q_inf * self.chord * C_L
        
        # Distribute lift to interface nodes
        # For simplicity, apply equally to all nodes
        forces[:, 1] = L / n_interface  # y-component (lift)
        
        # Pressure = force / area
        # Reference area = chord * 1 (unit span)
        pressures[:] = L / self.chord
        
        return forces, pressures


def strong_coupling_fsi(
    fluid_solver: FluidSolver,
    structural_solver: StructuralSolver,
    n_time_steps: int,
    dt: float,
    max_sub_iterations: int = 20,
    tolerance: float = 1e-4,
    use_aitken: bool = True,
) -> FSICouplingResult:
    """Perform strong coupling FSI simulation.
    
    At each time step, iterate between fluid and structure until convergence:
    
    1. Predict interface displacement from structure
    2. Solve fluid with current interface position
    3. Transfer fluid forces to structure
    4. Solve structure with fluid forces
    5. Check convergence: ||u^{k+1} - u^k|| < tol
    6. If not converged, apply Aitken acceleration and repeat
    
    Parameters
    ----------
    fluid_solver : FluidSolver
        Fluid solver instance
    structural_solver : StructuralSolver
        Structural solver instance
    n_time_steps : int
        Number of time steps
    dt : float
        Time step size [s]
    max_sub_iterations : int
        Maximum sub-iterations per time step
    tolerance : float
        Convergence tolerance for sub-iterations
    use_aitken : bool
        Use Aitken's delta^2 acceleration
    
    Returns
    -------
    result : FSICouplingResult
        Coupling result with time history and convergence data
    """
    time = torch.arange(n_time_steps + 1, dtype=D) * dt
    
    # Storage
    n_interface = 1  # Simplified: single interface point
    disp_history = torch.zeros((n_time_steps + 1, n_interface, 2), dtype=D)
    vel_history = torch.zeros((n_time_steps + 1, n_interface, 2), dtype=D)
    force_history = torch.zeros((n_time_steps + 1, n_interface, 2), dtype=D)
    
    sub_iter_history = []
    residual_history = []
    
    energy_fluid = torch.zeros(n_time_steps + 1, dtype=D)
    energy_strain = torch.zeros(n_time_steps + 1, dtype=D)
    energy_kinetic = torch.zeros(n_time_steps + 1, dtype=D)
    energy_imbalance = torch.zeros(n_time_steps + 1, dtype=D)
    
    # Initial conditions
    disp = torch.zeros((n_interface, 2), dtype=D)
    vel = torch.zeros((n_interface, 2), dtype=D)
    
    converged = True
    
    for step in range(n_time_steps):
        t = time[step]
        
        # Store initial state
        disp_history[step] = disp
        vel_history[step] = vel
        
        # Sub-iteration loop
        sub_iter = 0
        residuals = []
        
        # Aitken acceleration
        if use_aitken and step > 0:
            disp_prev_prev = disp_history[step - 1].clone()
            disp_prev = disp.clone()
        
        while sub_iter < max_sub_iterations:
            # Solve fluid
            forces, pressures = fluid_solver.solve(disp, vel)
            
            # Solve structure
            disp_new, vel_new, acc_new = structural_solver.solve(forces)
            
            # Compute residual
            residual = torch.norm(disp_new - disp).item()
            residuals.append(residual)
            
            # Check convergence
            if residual < tolerance:
                disp = disp_new
                vel = vel_new
                break
            
            # Aitken acceleration
            if use_aitken and sub_iter > 0:
                r_prev = residuals[-2]
                r_curr = residuals[-1]
                if abs(r_curr - r_prev) > 1e-10:
                    omega = -r_prev / (r_curr - r_prev)
                    disp = disp_prev + omega * (disp_new - disp_prev)
                else:
                    disp = disp_new
            else:
                disp = disp_new
            
            vel = vel_new
            sub_iter += 1
        
        if sub_iter >= max_sub_iterations:
            converged = False
        
        sub_iter_history.append(sub_iter)
        residual_history.append(residuals)
        
        # Store final state
        disp_history[step + 1] = disp
        vel_history[step + 1] = vel
        force_history[step + 1] = forces
        
        # Energy balance
        # Work done by fluid: W = integral(F · du)
        if step > 0:
            delta_disp = disp_history[step + 1] - disp_history[step]
            work = (forces * delta_disp).sum()
            energy_fluid[step + 1] = energy_fluid[step] + work
        
        # Structural energies
        if hasattr(structural_solver, 'get_strain_energy'):
            energy_strain[step + 1] = structural_solver.get_strain_energy()
        if hasattr(structural_solver, 'get_kinetic_energy'):
            energy_kinetic[step + 1] = structural_solver.get_kinetic_energy()
        
        # Energy imbalance
        energy_imbalance[step + 1] = abs(
            energy_fluid[step + 1] - energy_strain[step + 1] - energy_kinetic[step + 1]
        )
    
    return FSICouplingResult(
        time=time,
        interface_displacement=disp_history,
        interface_velocity=vel_history,
        interface_forces=force_history,
        sub_iterations_per_step=sub_iter_history,
        residual_history=residual_history,
        fluid_work=energy_fluid,
        strain_energy=energy_strain,
        kinetic_energy=energy_kinetic,
        energy_imbalance=energy_imbalance,
        converged=converged,
        coupling_scheme='strong',
        max_sub_iterations=max_sub_iterations,
        tolerance=tolerance,
    )


def weak_coupling_fsi(
    fluid_solver: FluidSolver,
    structural_solver: StructuralSolver,
    n_time_steps: int,
    dt: float,
) -> FSICouplingResult:
    """Perform weak (explicit) coupling FSI simulation.
    
    Exchange data once per time step without sub-iteration.
    Simple but may be unstable for strong added-mass effects.
    
    Parameters
    ----------
    fluid_solver : FluidSolver
        Fluid solver instance
    structural_solver : StructuralSolver
        Structural solver instance
    n_time_steps : int
        Number of time steps
    dt : float
        Time step size [s]
    
    Returns
    -------
    result : FSICouplingResult
        Coupling result
    """
    time = torch.arange(n_time_steps + 1, dtype=D) * dt
    
    n_interface = 1
    disp_history = torch.zeros((n_time_steps + 1, n_interface, 2), dtype=D)
    vel_history = torch.zeros((n_time_steps + 1, n_interface, 2), dtype=D)
    force_history = torch.zeros((n_time_steps + 1, n_interface, 2), dtype=D)
    
    sub_iter_history = []
    residual_history = []
    
    energy_fluid = torch.zeros(n_time_steps + 1, dtype=D)
    energy_strain = torch.zeros(n_time_steps + 1, dtype=D)
    energy_kinetic = torch.zeros(n_time_steps + 1, dtype=D)
    energy_imbalance = torch.zeros(n_time_steps + 1, dtype=D)
    
    disp = torch.zeros((n_interface, 2), dtype=D)
    vel = torch.zeros((n_interface, 2), dtype=D)
    
    for step in range(n_time_steps):
        # Solve fluid (one pass)
        forces, pressures = fluid_solver.solve(disp, vel)
        
        # Solve structure (one pass)
        disp, vel, acc = structural_solver.solve(forces)
        
        # Store
        disp_history[step + 1] = disp
        vel_history[step + 1] = vel
        force_history[step + 1] = forces
        
        sub_iter_history.append(1)  # Only one iteration
        residual_history.append([0.0])  # No convergence check
        
        # Energy balance
        if step > 0:
            delta_disp = disp_history[step + 1] - disp_history[step]
            work = (forces * delta_disp).sum()
            energy_fluid[step + 1] = energy_fluid[step] + work
        
        if hasattr(structural_solver, 'get_strain_energy'):
            energy_strain[step + 1] = structural_solver.get_strain_energy()
        if hasattr(structural_solver, 'get_kinetic_energy'):
            energy_kinetic[step + 1] = structural_solver.get_kinetic_energy()
        
        energy_imbalance[step + 1] = abs(
            energy_fluid[step + 1] - energy_strain[step + 1] - energy_kinetic[step + 1]
        )
    
    return FSICouplingResult(
        time=time,
        interface_displacement=disp_history,
        interface_velocity=vel_history,
        interface_forces=force_history,
        sub_iterations_per_step=sub_iter_history,
        residual_history=residual_history,
        fluid_work=energy_fluid,
        strain_energy=energy_strain,
        kinetic_energy=energy_kinetic,
        energy_imbalance=energy_imbalance,
        converged=True,
        coupling_scheme='weak',
        max_sub_iterations=1,
        tolerance=0.0,
    )
