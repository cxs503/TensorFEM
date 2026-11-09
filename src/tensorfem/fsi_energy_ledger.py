"""FSI Energy Conservation Ledger.

Physical Background
-------------------
In a closed FSI system, the total energy must be conserved:

    E_total = E_fluid_kinetic + E_struct_kinetic + E_struct_strain + E_dissipated

The work done by fluid on the structure equals the change in structural energy
plus dissipation:

    W_fluid→struct = ΔE_struct_kinetic + ΔE_struct_strain + E_dissipated

For the fluid domain:
    dE_fluid/dt = W_inlet - W_outlet - W_viscous - W_fluid→struct

The energy ledger tracks all terms and verifies the balance to within
a specified tolerance (default 5%).

References
----------
[1] Küttler, U., Wall, W.A. (2008). "Fixed-point fluid-structure
    interaction solvers with dynamic relaxation."
[2] Degroote, J., Vierendeels, J. (2012). "Energy conservation in
    partitioned FSI."
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnergyLedgerEntry:
    """Single time-step entry in the energy ledger."""
    time: float
    fluid_kinetic_energy: float = 0.0
    struct_kinetic_energy: float = 0.0
    struct_strain_energy: float = 0.0
    viscous_dissipation: float = 0.0
    work_fluid_to_struct: float = 0.0
    work_inlet: float = 0.0
    work_outlet: float = 0.0
    cumulative_imbalance: float = 0.0


@dataclass
class EnergyLedger:
    """Energy conservation ledger for FSI simulations.
    
    Tracks all energy terms and verifies conservation.
    
    Parameters
    ----------
    tolerance : float
        Maximum allowed relative energy imbalance (default 0.05 = 5%)
    """
    tolerance: float = 0.05
    entries: list = field(default_factory=list)
    
    def record(self, entry: EnergyLedgerEntry):
        """Record an energy entry."""
        self.entries.append(entry)
    
    def compute_imbalance(self) -> np.ndarray:
        """Compute energy imbalance at each time step.
        
        For a driven system: W(t) = ΔE_kin(t) + ΔE_strain(t) + E_diss(t)
        
        The imbalance is:
            Δ = W(t) - (ΔE_kin(t) + ΔE_strain(t) + E_diss(t))
        
        Returns
        -------
        imbalance : np.ndarray
            Energy imbalance at each time step
        """
        n = len(self.entries)
        if n < 2:
            return np.array([0.0])
        
        imbalance = np.zeros(n)
        for i in range(1, n):
            e0 = self.entries[0]
            ei = self.entries[i]
            
            # Change in structural energy
            delta_E_kin = ei.struct_kinetic_energy - e0.struct_kinetic_energy
            delta_E_strain = ei.struct_strain_energy - e0.struct_strain_energy
            E_diss = ei.viscous_dissipation - e0.viscous_dissipation
            
            # Cumulative work done on structure
            W_total = ei.work_fluid_to_struct - e0.work_fluid_to_struct
            
            # Imbalance: work should equal energy change + dissipation
            expected = delta_E_kin + delta_E_strain + E_diss
            imbalance[i] = W_total - expected
        
        return imbalance
    
    def relative_imbalance(self) -> np.ndarray:
        """Compute relative energy imbalance.
        
        Returns
        -------
        rel_imbalance : np.ndarray
            |imbalance| / max(|W_total|, reference)
        """
        imbalance = self.compute_imbalance()
        n = len(self.entries)
        
        rel = np.zeros(n)
        for i in range(n):
            ei = self.entries[i]
            e0 = self.entries[0]
            # Reference scale: total work done
            W_ref = max(abs(ei.work_fluid_to_struct - e0.work_fluid_to_struct), 1e-14)
            rel[i] = abs(imbalance[i]) / W_ref
        
        return rel
    
    def check_conservation(self) -> dict:
        """Check if energy conservation is satisfied.
        
        Returns
        -------
        result : dict
            Dictionary with:
            - 'passed': bool
            - 'max_relative_imbalance': float
            - 'mean_relative_imbalance': float
            - 'final_imbalance': float
            - 'tolerance': float
        """
        rel = self.relative_imbalance()
        
        if len(rel) < 2:
            return {
                'passed': True,
                'max_relative_imbalance': 0.0,
                'mean_relative_imbalance': 0.0,
                'final_imbalance': 0.0,
                'tolerance': self.tolerance,
            }
        
        # Skip initial transient
        rel_steady = rel[max(1, len(rel) // 4):]
        
        max_rel = np.max(rel_steady)
        mean_rel = np.mean(rel_steady)
        final_rel = rel[-1]
        
        return {
            'passed': max_rel < self.tolerance,
            'max_relative_imbalance': float(max_rel),
            'mean_relative_imbalance': float(mean_rel),
            'final_imbalance': float(final_rel),
            'tolerance': self.tolerance,
        }
    
    def summary(self) -> str:
        """Generate a summary string."""
        result = self.check_conservation()
        status = "PASS" if result['passed'] else "FAIL"
        return (
            f"Energy Conservation: {status}\n"
            f"  Max relative imbalance: {result['max_relative_imbalance']*100:.3f}%\n"
            f"  Mean relative imbalance: {result['mean_relative_imbalance']*100:.3f}%\n"
            f"  Tolerance: {result['tolerance']*100:.1f}%"
        )


# ---------------------------------------------------------------------------
# Modal energy computation for Turek-Hron FSI-2
# ---------------------------------------------------------------------------

def compute_modal_energies(
    q: np.ndarray,
    q_dot: np.ndarray,
    M: float,
    K: float,
) -> dict:
    """Compute modal energies for a single-DOF system.
    
    Parameters
    ----------
    q : np.ndarray
        Modal displacement history
    q_dot : np.ndarray
        Modal velocity history
    M : float
        Modal mass
    K : float
        Modal stiffness
    
    Returns
    -------
    energies : dict
        Dictionary with kinetic, strain, and total energy arrays
    """
    E_kinetic = 0.5 * M * q_dot ** 2
    E_strain = 0.5 * K * q ** 2
    E_total = E_kinetic + E_strain
    
    return {
        'kinetic': E_kinetic,
        'strain': E_strain,
        'total': E_total,
    }


def compute_work_from_force(
    force: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    """Compute cumulative work done by a force.
    
    W(t) = ∫₀ᵗ F(τ) · v(τ) dτ ≈ Σ F_i · (q_{i+1} - q_i)
    
    Parameters
    ----------
    force : np.ndarray
        Force history [N]
    displacement : np.ndarray
        Displacement history [N]
    
    Returns
    -------
    work : np.ndarray
        Cumulative work [N]
    """
    dq = np.diff(displacement)
    # Trapezoidal rule
    f_avg = 0.5 * (force[:-1] + force[1:])
    dW = f_avg * dq
    
    work = np.zeros(len(force))
    work[1:] = np.cumsum(dW)
    
    return work


def compute_viscous_dissipation(
    q_dot: np.ndarray,
    C: float,
    dt: float,
) -> np.ndarray:
    """Compute cumulative viscous dissipation.
    
    E_diss(t) = ∫₀ᵗ C · v²(τ) dτ
    
    Parameters
    ----------
    q_dot : np.ndarray
        Velocity history
    C : float
        Damping coefficient
    dt : float
        Time step
    
    Returns
    -------
    dissipation : np.ndarray
        Cumulative dissipation
    """
    power = C * q_dot ** 2
    dissipation = np.zeros(len(q_dot))
    dissipation[1:] = np.cumsum(0.5 * (power[:-1] + power[1:]) * dt)
    
    return dissipation


# ---------------------------------------------------------------------------
# Full energy ledger for Turek-Hron FSI-2
# ---------------------------------------------------------------------------

def build_turek_hron_energy_ledger(
    time: np.ndarray,
    q: np.ndarray,
    q_dot: np.ndarray,
    F_ext: np.ndarray,
    M: float,
    K: float,
    C: float,
    tolerance: float = 0.05,
) -> EnergyLedger:
    """Build energy ledger for Turek-Hron FSI-2 simulation.
    
    For a driven damped oscillator:
        M*q'' + C*q' + K*q = F(t)
    
    Energy balance:
        W(t) = ΔE_kin(t) + ΔE_strain(t) + E_diss(t)
    
    where W(t) = ∫F·dq is the work done by external force.
    
    Parameters
    ----------
    time : np.ndarray
        Time array
    q : np.ndarray
        Modal displacement
    q_dot : np.ndarray
        Modal velocity
    F_ext : np.ndarray
        External (fluid) force
    M : float
        Modal mass
    K : float
        Modal stiffness
    C : float
        Modal damping
    tolerance : float
        Energy conservation tolerance
    
    Returns
    -------
    ledger : EnergyLedger
        Populated energy ledger
    """
    ledger = EnergyLedger(tolerance=tolerance)
    dt = time[1] - time[0] if len(time) > 1 else 0.001
    
    # Compute energies
    energies = compute_modal_energies(q, q_dot, M, K)
    work = compute_work_from_force(F_ext, q)
    dissipation = compute_viscous_dissipation(q_dot, C, dt)
    
    for i in range(len(time)):
        entry = EnergyLedgerEntry(
            time=time[i],
            fluid_kinetic_energy=0.0,  # Not tracked in modal model
            struct_kinetic_energy=energies['kinetic'][i],
            struct_strain_energy=energies['strain'][i],
            viscous_dissipation=dissipation[i],
            work_fluid_to_struct=work[i],
        )
        ledger.record(entry)
    
    return ledger


def verify_energy_conservation_undamped() -> dict:
    """Verify energy conservation for undamped free vibration.
    
    For an undamped system with no external force, total energy should be constant.
    
    Returns
    -------
    result : dict
        Conservation check result
    """
    # Undamped free vibration: q(t) = A*sin(omega*t)
    omega = 2 * np.pi * 3.0  # 3 Hz
    A = 0.05  # 5 cm amplitude
    t = np.linspace(0, 2, 200)
    q = A * np.sin(omega * t)
    q_dot = A * omega * np.cos(omega * t)
    
    M = 1.0
    K = omega ** 2  # omega = sqrt(K/M)
    C = 0.0  # No damping
    
    ledger = build_turek_hron_energy_ledger(
        t, q, q_dot, np.zeros_like(t), M, K, C, tolerance=0.01
    )
    
    # Check that total energy is constant
    E_total = np.array([e.struct_kinetic_energy + e.struct_strain_energy 
                       for e in ledger.entries])
    E_mean = np.mean(E_total)
    E_var = np.std(E_total) / E_mean if E_mean > 0 else 0.0
    
    return {
        'passed': E_var < 0.01,
        'energy_variation': E_var,
        'mean_energy': E_mean,
    }
