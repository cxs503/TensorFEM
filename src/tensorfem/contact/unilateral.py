"""Unilateral (one-sided) contact solver with active-set method.

This module solves the unilateral contact problem where a structure rests on
a Winkler-type elastic foundation that can only push (no tension / no adhesion).
The contact conditions are:

    y_i ≥ 0          (gap is non-negative: no interpenetration)
    R_i ≥ 0          (contact reaction is compressive only)
    R_i · y_i = 0    (complementarity: either in contact with reaction, or
                      separated with zero reaction)

where y_i is the gap at contact node i and R_i is the contact pressure.

Sign convention:
- Displacement u > 0 means moving TOWARD the foundation (downward/into contact).
- Gap y = g₀ - C @ u: positive means separated, negative means penetrating.
- Contact reaction R ≥ 0 (compressive, pushing structure away from foundation).
- Equilibrium: K u = f - Cᵀ R (reaction opposes penetration)

For a beam/plate on a tensionless Winkler foundation:
- The foundation provides upward reaction only when the structure deflects
  downward (into the foundation).
- When the structure lifts off, the foundation provides zero reaction.

The active-set method iteratively determines which nodes are in contact
(active set) and which are separated (inactive set). At convergence:
- Active nodes: y_i = 0, R_i ≥ 0
- Inactive nodes: y_i > 0, R_i = 0

Theory
------
The linear complementarity problem (LCP) is:

    Find (y, R) such that:
        K u = f - Cᵀ R      (equilibrium: reaction opposes penetration)
        y = g₀ - C u ≥ 0    (non-penetration)
        R ≥ 0                (compression only)
        Rᵀ y = 0            (complementarity)

where K is the structural stiffness, f is the external load, C is the
contact operator (extracting penetration), and g₀ is the initial gap.

References
----------
[1] Wriggers, P., *Computational Contact Mechanics*, 2nd ed., Springer, 2006.
[2] Laursen, T.A., *Computational Contact and Impact Mechanics*, Springer, 2002.
[3] Cottle, R.W. et al., *The Linear Complementarity Problem*, SIAM, 2009.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class UnilateralContactResult:
    """Result of unilateral contact solve.

    Attributes
    ----------
    displacement : (ndof,) tensor
        Nodal displacements.
    contact_force : (ncontact,) tensor
        Contact reactions (positive = compressive).
    gap : (ncontact,) tensor
        Gap values at contact nodes (positive = open/separated, zero = closed).
    active_set : (ncontact,) bool tensor
        True where contact is active (gap = 0, reaction ≥ 0).
    complementarity_residual : float
        Max |R_i · y_i| over all contact nodes (should be ≈ 0).
    iterations : int
        Number of active-set iterations.
    converged : bool
        Whether the active-set method converged.
    structural_energy : float
        Strain energy 0.5 * uᵀ K u.
    """
    displacement: torch.Tensor
    contact_force: torch.Tensor
    gap: torch.Tensor
    active_set: torch.Tensor
    complementarity_residual: float
    iterations: int
    converged: bool
    structural_energy: float


def complementarity_residual(
    R: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute the complementarity residual R_i · y_i for each contact node.

    For a valid solution, R_i ≥ 0, y_i ≥ 0, and R_i · y_i = 0 for all i.

    Parameters
    ----------
    R : (n,) tensor of contact reactions.
    y : (n,) tensor of gap values.

    Returns
    -------
    residual : (n,) tensor of R_i * y_i products.
    """
    return R * y


def solve_unilateral_contact(
    K: torch.Tensor,
    f: torch.Tensor,
    C: torch.Tensor,
    g0: torch.Tensor,
    *,
    method: str = "active_set",
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> UnilateralContactResult:
    """Solve unilateral contact problem using active-set method.

    The problem is:
        K u = f - Cᵀ R      (equilibrium: reaction opposes penetration)
        y = g₀ - C u ≥ 0    (non-penetration: gap must be non-negative)
        R ≥ 0                (compression only)
        Rᵀ y = 0            (complementarity)

    Sign convention:
    - u > 0 means displacement TOWARD foundation (into contact)
    - y = g₀ - C u: positive gap = separated, negative = penetrating
    - R > 0 means compressive reaction (pushing away from foundation)
    - Equilibrium: K u = f - Cᵀ R (reaction reduces penetration)

    Parameters
    ----------
    K : (ndof, ndof) symmetric positive-definite stiffness matrix.
    f : (ndof,) external force vector (positive = toward foundation).
    C : (ncontact, ndof) contact operator (maps displacement to penetration).
    g0 : (ncontact,) initial gap (positive = initially open).
    method : "active_set" (exact).
    tolerance : convergence tolerance for active-set changes.
    max_iterations : maximum number of active-set iterations.

    Returns
    -------
    result : UnilateralContactResult
    """
    if method not in ("active_set",):
        raise ValueError(f"Unknown method: {method}")

    ndof = K.shape[0]
    ncontact = C.shape[0]
    dtype = K.dtype
    device = K.device

    # Precompute free response (no contact)
    u_free = torch.linalg.solve(K, f)

    # Initialize active set: contacts that would penetrate in free solution
    # Penetration = C @ u - g0 > 0 means penetrating (gap < 0)
    penetration_free = C @ u_free - g0
    active = penetration_free > tolerance

    R = torch.zeros(ncontact, dtype=dtype, device=device)
    u = u_free.clone()

    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        active_idx = torch.nonzero(active, as_tuple=False).flatten()

        if active_idx.numel() == 0:
            # No active contacts: free solution
            u = u_free
            R = torch.zeros(ncontact, dtype=dtype, device=device)
        else:
            # Active set method: solve KKT system
            # On active set: gap = 0, i.e., g₀ - C_a u = 0 → C_a u = g0_a
            # Equilibrium: K u = f - Cᵀ R → K u + Cᵀ R = f
            # KKT system:
            # [K    C_aᵀ] [u]     [f    ]
            # [C_a   0  ] [R_a] = [g0_a ]
            na = active_idx.numel()
            C_a = C[active_idx]
            g0_a = g0[active_idx]

            Z = torch.zeros((na, na), dtype=dtype, device=device)
            A = torch.cat([
                torch.cat([K, C_a.T], dim=1),
                torch.cat([C_a, Z], dim=1),
            ], dim=0)
            rhs = torch.cat([f, g0_a])

            sol = torch.linalg.solve(A, rhs)
            u = sol[:ndof]
            R_a = sol[ndof:]

            R = torch.zeros(ncontact, dtype=dtype, device=device)
            R[active_idx] = R_a

        # Compute gaps: y = g₀ - C u (positive = separated)
        y = g0 - C @ u

        # Update active set based on KKT conditions
        # Active if: penetrating (y < 0) OR (was active and reaction > 0)
        new_active = (y < -tolerance) | (active & (R > tolerance))

        # Deactivate if reaction becomes negative (tension)
        for idx in active_idx:
            if R[idx] < -tolerance:
                new_active[idx] = False

        if torch.equal(new_active, active):
            converged = True
            break

        active = new_active

    # Final gap and complementarity check
    y = g0 - C @ u
    # Ensure non-negative for output
    y_final = torch.clamp(y, min=0.0)
    R_final = torch.clamp(R, min=0.0)

    comp_res = complementarity_residual(R_final, y_final)
    max_comp_res = float(torch.max(torch.abs(comp_res)).item()) if comp_res.numel() > 0 else 0.0

    structural_energy = 0.5 * float((u @ K @ u).item())

    return UnilateralContactResult(
        displacement=u,
        contact_force=R_final,
        gap=y_final,
        active_set=active,
        complementarity_residual=max_comp_res,
        iterations=iteration,
        converged=converged,
        structural_energy=structural_energy,
    )


def solve_winkler_unilateral(
    K_struct: torch.Tensor,
    f: torch.Tensor,
    k_foundation: float,
    contact_dofs: torch.Tensor,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> UnilateralContactResult:
    """Solve beam/plate on tensionless Winkler foundation.

    The foundation provides upward reaction only where the structure deflects
    downward (into the foundation). The contact condition is:
        R_i = k * u_i when u_i > 0 (downward deflection)
        R_i = 0 when u_i < 0 (lifted off)

    Sign convention:
    - u > 0 at contact DOF means downward deflection (into foundation)
    - Foundation reacts with upward force R = k * u when u > 0
    - When u < 0 (lifted off), R = 0
    - Equilibrium: K u = f - R (foundation reaction opposes downward motion)

    Parameters
    ----------
    K_struct : (ndof, ndof) structural stiffness (without foundation).
    f : (ndof,) external force vector (positive = downward/into foundation).
    k_foundation : Winkler modulus (force/length³ or force/length²).
    contact_dofs : (ncontact,) indices of DOFs where foundation acts.
    tolerance : convergence tolerance.
    max_iterations : maximum iterations.

    Returns
    -------
    result : UnilateralContactResult
    """
    ndof = K_struct.shape[0]
    ncontact = contact_dofs.numel()
    dtype = K_struct.dtype
    device = K_struct.device

    # Build contact operator C: extracts displacements at contact DOFs
    # Penetration = C @ u (positive = into foundation)
    C = torch.zeros((ncontact, ndof), dtype=dtype, device=device)
    for i, dof in enumerate(contact_dofs):
        C[i, dof] = 1.0

    # Initial gap is zero (foundation surface at reference position)
    g0 = torch.zeros(ncontact, dtype=dtype, device=device)

    # Active-set iteration
    # Start with all contact DOFs where load pushes into foundation
    u_free = torch.linalg.solve(K_struct, f)
    penetration_free = C @ u_free
    active = penetration_free > tolerance

    u = torch.zeros(ndof, dtype=dtype, device=device)
    R = torch.zeros(ncontact, dtype=dtype, device=device)

    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        active_idx = torch.nonzero(active, as_tuple=False).flatten()

        if active_idx.numel() == 0:
            # No foundation contact: solve without foundation
            u = torch.linalg.solve(K_struct, f)
            R = torch.zeros(ncontact, dtype=dtype, device=device)
        else:
            # Build effective stiffness with foundation at active DOFs
            # K_eff u = f, where K_eff = K + k at active DOFs
            K_eff = K_struct.clone()
            for idx in active_idx:
                dof = contact_dofs[idx]
                K_eff[dof, dof] += k_foundation

            # Solve with foundation
            u = torch.linalg.solve(K_eff, f)

            # Compute reactions at active DOFs
            R = torch.zeros(ncontact, dtype=dtype, device=device)
            for idx in active_idx:
                dof = contact_dofs[idx]
                # Reaction = k * u for downward deflection (u > 0)
                R[idx] = k_foundation * u[dof]

        # Gap at contact DOFs: y = -u (positive = lifted off, negative = penetrating)
        y = torch.zeros(ncontact, dtype=dtype, device=device)
        for i, dof in enumerate(contact_dofs):
            y[i] = -u[dof]  # gap = -displacement (positive = separated)

        # Update active set
        # Active if: penetrating (u > 0, i.e., y < 0) OR (was active and reaction > 0)
        new_active = (y < -tolerance) | (active & (R > tolerance))

        # Deactivate if reaction becomes negative (tension)
        for idx in active_idx:
            if R[idx] < -tolerance:
                new_active[idx] = False

        if torch.equal(new_active, active):
            converged = True
            break

        active = new_active

    # Final quantities
    y_final = torch.zeros(ncontact, dtype=dtype, device=device)
    for i, dof in enumerate(contact_dofs):
        y_final[i] = -u[dof]  # gap = -displacement

    # Ensure proper signs for output
    gap_output = torch.clamp(y_final, min=0.0)  # positive = separated
    R_final = torch.clamp(R, min=0.0)  # positive = compressive

    comp_res = complementarity_residual(R_final, gap_output)
    max_comp_res = float(torch.max(torch.abs(comp_res)).item()) if comp_res.numel() > 0 else 0.0

    structural_energy = 0.5 * float((u @ K_struct @ u).item())

    return UnilateralContactResult(
        displacement=u,
        contact_force=R_final,
        gap=gap_output,
        active_set=active,
        complementarity_residual=max_comp_res,
        iterations=iteration,
        converged=converged,
        structural_energy=structural_energy,
    )
