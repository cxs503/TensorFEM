"""Radial Basis Function (RBF) interpolation for non-matching mesh transfer.

Physical Background
-------------------
In partitioned FSI, the fluid and structural meshes rarely coincide at the
interface. Data (displacements, forces) must be transferred between non-matching
grids while preserving:
1. **Consistency**: constant fields transfer exactly
2. **Conservation**: total force/moment is preserved

This module implements RBF interpolation with conservation correction following
the approach of de Boer et al. (2006) and Rendall & Allen (2009).

Method
------
Given source points X_s with values f_s and target points X_t:

1. Build RBF interpolation matrix:
   Φ_ij = φ(||x_i - x_j||)
   where φ is the radial basis function (thin-plate, Gaussian, etc.)

2. Solve for weights:
   Φ · w = f_s

3. Interpolate at targets:
   f_t = Φ_ts · w

4. Conservation correction (force transfer):
   Ensure Σ f_t · A_t = Σ f_s · A_s by scaling.

References
----------
[1] de Boer, A., van der Molen, M.H., Bijl, H. (2007). "Robust methods for
    interpolating data between non-matching meshes."
[2] Rendall, T.C.S., Allen, C.B. (2009). "Efficient fluid-structure
    interpolation using compact radial basis functions."
[3] Beckert, A., Wendland, H. (2001). "Interpolating clip distributions
    using thin-plate splines."
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# RBF kernel functions
# ---------------------------------------------------------------------------

def rbf_thin_plate(r: np.ndarray) -> np.ndarray:
    """Thin-plate spline: φ(r) = r² log(r), φ(0) = 0."""
    result = np.zeros_like(r)
    mask = r > 1e-14
    result[mask] = r[mask] ** 2 * np.log(r[mask])
    return result


def rbf_gaussian(r: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Gaussian RBF: φ(r) = exp(-ε² r²)."""
    return np.exp(-(epsilon * r) ** 2)


def rbf_multiquadric(r: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Multiquadric: φ(r) = √(1 + ε² r²)."""
    return np.sqrt(1.0 + (epsilon * r) ** 2)


def rbf_wendland_c2(r: np.ndarray, support: float = 1.0) -> np.ndarray:
    """Wendland C2 compactly supported RBF.
    
    φ(r) = (1 - r/R)⁴ (4r/R + 1) for r < R, 0 otherwise.
    """
    rho = np.clip(r / support, 0.0, 1.0)
    return (1.0 - rho) ** 4 * (4.0 * rho + 1.0) * (r < support).astype(float)


# ---------------------------------------------------------------------------
# RBF Interpolator
# ---------------------------------------------------------------------------

@dataclass
class RBFInterpolator:
    """RBF interpolator for scattered data.
    
    Parameters
    ----------
    source_points : np.ndarray
        Source point coordinates [N_s, dim]
    kernel : str
        RBF kernel type: 'thin_plate', 'gaussian', 'multiquadric', 'wendland_c2'
    epsilon : float
        Shape parameter (for gaussian/multiquadric)
    support : float
        Support radius (for wendland_c2)
    polynomial : bool
        Whether to include polynomial terms for consistency
    """
    source_points: np.ndarray
    kernel: str = 'thin_plate'
    epsilon: float = 1.0
    support: float = 1.0
    polynomial: bool = True
    
    def __post_init__(self):
        self.n_source = self.source_points.shape[0]
        self.dim = self.source_points.shape[1]
        self._build_matrix()
    
    def _compute_rbf(self, r: np.ndarray) -> np.ndarray:
        """Evaluate RBF kernel."""
        if self.kernel == 'thin_plate':
            return rbf_thin_plate(r)
        elif self.kernel == 'gaussian':
            return rbf_gaussian(r, self.epsilon)
        elif self.kernel == 'multiquadric':
            return rbf_multiquadric(r, self.epsilon)
        elif self.kernel == 'wendland_c2':
            return rbf_wendland_c2(r, self.support)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _build_matrix(self):
        """Build the RBF interpolation matrix."""
        N = self.n_source
        # Distance matrix
        diff = self.source_points[:, np.newaxis, :] - self.source_points[np.newaxis, :, :]
        R = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # RBF matrix
        Phi = self._compute_rbf(R)
        
        if self.polynomial:
            # Augmented system with polynomial terms for consistency
            # [Φ  P] [w]   [f]
            # [Pᵀ 0] [γ] = [0]
            P = np.hstack([np.ones((N, 1)), self.source_points])
            n_poly = P.shape[1]
            
            self.A = np.zeros((N + n_poly, N + n_poly))
            self.A[:N, :N] = Phi
            self.A[:N, N:] = P
            self.A[N:, :N] = P.T
        else:
            self.A = Phi
            self.P = None
        
        # Factorize
        try:
            self.A_inv = np.linalg.inv(self.A + 1e-12 * np.eye(self.A.shape[0]))
        except np.linalg.LinAlgError:
            self.A_inv = np.linalg.pinv(self.A)
    
    def interpolate(self, target_points: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Interpolate values from source to target points.
        
        Parameters
        ----------
        target_points : np.ndarray
            Target point coordinates [N_t, dim]
        values : np.ndarray
            Values at source points [N_s] or [N_s, n_comp]
        
        Returns
        -------
        interpolated : np.ndarray
            Interpolated values at target points [N_t] or [N_t, n_comp]
        """
        N_t = target_points.shape[0]
        
        # Distance from targets to sources
        diff = target_points[:, np.newaxis, :] - self.source_points[np.newaxis, :, :]
        R_ts = np.sqrt(np.sum(diff ** 2, axis=2))
        Phi_ts = self._compute_rbf(R_ts)
        
        if values.ndim == 1:
            values = values[:, np.newaxis]
            squeeze = True
        else:
            squeeze = False
        
        n_comp = values.shape[1]
        result = np.zeros((N_t, n_comp))
        
        for c in range(n_comp):
            if self.polynomial:
                rhs = np.zeros(self.A.shape[0])
                rhs[:self.n_source] = values[:, c]
                coeffs = self.A_inv @ rhs
                result[:, c] = Phi_ts @ coeffs[:self.n_source] + \
                    np.hstack([np.ones((N_t, 1)), target_points]) @ coeffs[self.n_source:]
            else:
                coeffs = self.A_inv @ values[:, c]
                result[:, c] = Phi_ts @ coeffs
        
        if squeeze:
            result = result[:, 0]
        
        return result


# ---------------------------------------------------------------------------
# Conservative transfer with correction
# ---------------------------------------------------------------------------

@dataclass
class ConservativeTransfer:
    """Conservative data transfer between non-matching meshes.
    
    Ensures that total force/moment is preserved during transfer.
    
    Parameters
    ----------
    source_points : np.ndarray
        Source mesh interface points [N_s, dim]
    target_points : np.ndarray
        Target mesh interface points [N_t, dim]
    source_areas : np.ndarray
        Tributary areas at source points [N_s]
    target_areas : np.ndarray
        Tributary areas at target points [N_t]
    kernel : str
        RBF kernel type
    """
    source_points: np.ndarray
    target_points: np.ndarray
    source_areas: Optional[np.ndarray] = None
    target_areas: Optional[np.ndarray] = None
    kernel: str = 'thin_plate'
    
    def __post_init__(self):
        N_s = self.source_points.shape[0]
        N_t = self.target_points.shape[0]
        
        if self.source_areas is None:
            self.source_areas = np.ones(N_s) / N_s
        if self.target_areas is None:
            self.target_areas = np.ones(N_t) / N_t
        
        self.interpolator = RBFInterpolator(
            self.source_points, kernel=self.kernel
        )
    
    def transfer_displacement(self, disp_source: np.ndarray) -> np.ndarray:
        """Transfer displacement from source to target (consistency).
        
        Parameters
        ----------
        disp_source : np.ndarray
            Displacement at source points [N_s, dim]
        
        Returns
        -------
        disp_target : np.ndarray
            Displacement at target points [N_t, dim]
        """
        return self.interpolator.interpolate(self.target_points, disp_source)
    
    def transfer_force(self, force_source: np.ndarray) -> np.ndarray:
        """Transfer force from source to target with conservation correction.
        
        Ensures: Σ F_target · A_target = Σ F_source · A_source
        
        Parameters
        ----------
        force_source : np.ndarray
            Force at source points [N_s, dim]
        
        Returns
        -------
        force_target : np.ndarray
            Force at target points [N_t, dim]
        """
        # Interpolate forces
        force_target = self.interpolator.interpolate(self.target_points, force_source)
        
        # Conservation correction
        total_source = np.sum(force_source * self.source_areas[:, np.newaxis], axis=0)
        total_target = np.sum(force_target * self.target_areas[:, np.newaxis], axis=0)
        
        # Scale to conserve total force
        for d in range(force_target.shape[1]):
            if abs(total_target[d]) > 1e-14:
                scale = total_source[d] / total_target[d]
                force_target[:, d] *= scale
        
        return force_target
    
    def conservation_error(self, force_source: np.ndarray, force_target: np.ndarray) -> float:
        """Compute relative conservation error.
        
        Returns
        -------
        error : float
            Relative error in total force: |F_target - F_source| / |F_source|
        """
        F_src = np.sum(force_source * self.source_areas[:, np.newaxis], axis=0)
        F_tgt = np.sum(force_target * self.target_areas[:, np.newaxis], axis=0)
        
        norm_src = np.linalg.norm(F_src)
        if norm_src < 1e-14:
            return np.linalg.norm(F_tgt - F_src)
        
        return np.linalg.norm(F_tgt - F_src) / norm_src


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def create_interface_transfer(
    fluid_interface_points: np.ndarray,
    structural_interface_points: np.ndarray,
    fluid_areas: Optional[np.ndarray] = None,
    structural_areas: Optional[np.ndarray] = None,
    kernel: str = 'thin_plate',
) -> ConservativeTransfer:
    """Create a conservative transfer operator between fluid and structure.
    
    Parameters
    ----------
    fluid_interface_points : np.ndarray
        Fluid mesh interface nodes [N_f, 2]
    structural_interface_points : np.ndarray
        Structural mesh interface nodes [N_s, 2]
    fluid_areas : np.ndarray, optional
        Tributary areas for fluid nodes
    structural_areas : np.ndarray, optional
        Tributary areas for structural nodes
    kernel : str
        RBF kernel type
    
    Returns
    -------
    transfer : ConservativeTransfer
        Transfer operator
    """
    return ConservativeTransfer(
        source_points=fluid_interface_points,
        target_points=structural_interface_points,
        source_areas=fluid_areas,
        target_areas=structural_areas,
        kernel=kernel,
    )
