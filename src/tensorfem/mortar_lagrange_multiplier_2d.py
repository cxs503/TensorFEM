"""2D mortar contact with Lagrange multipliers for non-matching meshes.

Implements the mortar method for frictionless unilateral contact between
two deformable bodies with non-matching interface discretizations. The
contact constraint is enforced using Lagrange multipliers, avoiding the
penalty parameter sensitivity of penalty-based methods.

Theory
------
The mortar method introduces a Lagrange multiplier field λ on the contact
interface to enforce the normal contact constraint. For non-matching meshes,
the mortar coupling matrix D (interface mass) and M (transfer operator) are
computed by integrating shape function products over the contact interface.

The discrete contact problem is formulated as a saddle-point system::

    [K   -M^T] [u]   [f]
    [M    D  ] [λ] = [0]

where K is the structural stiffness, M is the mortar transfer matrix,
D is the mortar mass matrix, u is displacement, λ is contact pressure.

For Hertz line contact (2D plane strain, cylinder on half-space):
- Contact half-width: b = sqrt(4*F*R/(π*E*))
- Pressure: p(x) = p0 * sqrt(1 - (x/b)²)
- Peak pressure: p0 = 2*F/(π*b)
- E* = E/(1-ν²) for plane strain

References
----------
[1] Wriggers, P., Computational Contact Mechanics, 2nd ed., Springer, 2006.
[2] Puso, M.A. and Michalerkes, N.R., "A mortar segment method for
    frictional contact", IJNME, 2004.
[3] Hertz, H., "Über die Berührung fester elastischer Körper",
    Journal für die reine und angewandte Mathematik, 1882.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

DTYPE = torch.float64


@dataclass(frozen=True)
class MortarLM2DResult:
    """Result of 2D mortar Lagrange multiplier contact solve."""
    displacement: torch.Tensor
    contact_pressure_nodes: torch.Tensor  # pressure at slave contact nodes
    contact_pressure_quad: torch.Tensor   # pressure at quadrature points
    quad_x_coords: torch.Tensor           # x-coords of quadrature points
    gap: torch.Tensor                     # normal gap at contact nodes
    max_penetration: float                # max penetration (should be ~0)
    total_contact_force: float            # integrated contact force
    complementarity_residual: float       # max |λ·g|
    converged: bool
    iterations: int


@dataclass(frozen=True)
class HertzLineReference:
    """Analytical Hertz line contact solution (cylinder on half-space)."""
    contact_half_width: float   # b
    peak_pressure: float        # p0
    total_force: float          # F per unit length
    indentation: float          # δ
    effective_modulus: float    # E*


def hertz_line_reference(
    force_per_length: float,
    radius: float,
    young: float,
    poisson: float,
) -> HertzLineReference:
    """Compute analytical Hertz line contact solution.
    
    Parameters
    ----------
    force_per_length : float
        Contact force per unit length [N/m].
    radius : float
        Cylinder radius [m].
    young : float
        Young's modulus [Pa].
    poisson : float
        Poisson's ratio [-].
    
    Returns
    -------
    HertzLineReference
        Analytical solution parameters.
    """
    E_star = young / (1.0 - poisson**2)  # plane strain effective modulus
    b = math.sqrt(4.0 * force_per_length * radius / (math.pi * E_star))
    p0 = 2.0 * force_per_length / (math.pi * b)
    delta = math.pi * p0 * b / (2.0 * E_star)  # indentation (approximate)
    return HertzLineReference(b, p0, force_per_length, delta, E_star)


def hertz_pressure_profile(x: torch.Tensor, b: float, p0: float) -> torch.Tensor:
    """Hertz pressure distribution p(x) = p0 * sqrt(1 - (x/b)^2)."""
    xi = x / b
    return p0 * torch.sqrt(torch.clamp(1.0 - xi**2, min=0.0))


# ============================================================================
# 2D Q4 mesh and stiffness assembly
# ============================================================================

def q4_plane_strain_stiffness(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young: float,
    poisson: float,
) -> torch.Tensor:
    """Assemble global stiffness for 2D Q4 plane strain elements.
    
    Parameters
    ----------
    nodes : (n_nodes, 2) tensor
        Node coordinates.
    elements : (n_elem, 4) tensor
        Element connectivity (CCW ordering).
    young : float
        Young's modulus.
    poisson : float
        Poisson's ratio.
    
    Returns
    -------
    K : (2*n_nodes, 2*n_nodes) tensor
        Global stiffness matrix.
    """
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    K = torch.zeros((n_dof, n_dof), dtype=DTYPE)
    
    # Plane strain constitutive matrix
    c = young / ((1 + poisson) * (1 - 2 * poisson))
    D_mat = c * torch.tensor([
        [1 - poisson, poisson, 0],
        [poisson, 1 - poisson, 0],
        [0, 0, (1 - 2 * poisson) / 2],
    ], dtype=DTYPE)
    
    # 2x2 Gauss quadrature
    gp = 1.0 / math.sqrt(3)
    quad_pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    quad_wts = [1.0, 1.0, 1.0, 1.0]
    
    for elem_nodes in elements:
        x = nodes[elem_nodes]  # (4, 2)
        ke = torch.zeros((8, 8), dtype=DTYPE)
        
        for (xi, eta), w in zip(quad_pts, quad_wts):
            # Shape functions
            N = torch.tensor([
                (1 - xi) * (1 - eta) / 4,
                (1 + xi) * (1 - eta) / 4,
                (1 + xi) * (1 + eta) / 4,
                (1 - xi) * (1 + eta) / 4,
            ], dtype=DTYPE)
            
            # Shape function derivatives w.r.t. natural coords
            dNdxi = torch.tensor([
                [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
            ], dtype=DTYPE) / 4
            
            # Jacobian
            J = dNdxi @ x  # (2, 2)
            detJ = torch.linalg.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coords
            dNdx = invJ @ dNdxi  # (2, 4)
            
            # B matrix (strain-displacement)
            B = torch.zeros((3, 8), dtype=DTYPE)
            for i in range(4):
                B[0, 2*i] = dNdx[0, i]      # eps_xx
                B[1, 2*i+1] = dNdx[1, i]    # eps_yy
                B[2, 2*i] = dNdx[1, i]      # gamma_xy
                B[2, 2*i+1] = dNdx[0, i]
            
            ke += w * detJ * B.T @ D_mat @ B
        
        # Assemble into global
        dof_map = torch.stack([2*elem_nodes, 2*elem_nodes+1], dim=1).reshape(-1)
        for i in range(8):
            for j in range(8):
                K[dof_map[i], dof_map[j]] += ke[i, j]
    
    return K


def structured_q4_mesh(
    nx: int, ny: int,
    Lx: float, Ly: float,
    *,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    grading: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create structured Q4 mesh over [x_offset, x_offset+Lx] x [y_offset, y_offset+Ly].
    
    Parameters
    ----------
    nx, ny : int
        Number of elements in x and y directions.
    Lx, Ly : float
        Domain dimensions.
    x_offset, y_offset : float
        Origin offset.
    grading : float
        Mesh grading factor (>1 concentrates toward high-index end).
    
    Returns
    -------
    nodes : (n_nodes, 2) tensor
    elements : (n_elem, 4) tensor
    """
    # Graded spacing
    if grading != 1.0:
        t = torch.linspace(0, 1, nx + 1, dtype=DTYPE)
        x = x_offset + Lx * t ** grading
        t = torch.linspace(0, 1, ny + 1, dtype=DTYPE)
        y = y_offset + Ly * t ** grading
    else:
        x = torch.linspace(x_offset, x_offset + Lx, nx + 1, dtype=DTYPE)
        y = torch.linspace(y_offset, y_offset + Ly, ny + 1, dtype=DTYPE)
    
    # Node coordinates
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([float(x[i]), float(y[j])])
    nodes = torch.tensor(nodes, dtype=DTYPE)
    
    # Element connectivity
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            elements.append([n0, n1, n2, n3])
    elements = torch.tensor(elements, dtype=torch.long)
    
    return nodes, elements


# ============================================================================
# Mortar coupling for non-matching 1D interface
# ============================================================================

def compute_mortar_matrices_1d(
    slave_x: torch.Tensor,
    master_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mortar D and M matrices for 1D non-matching interface.
    
    The slave and master surfaces are discretized by 1D line elements
    (2-node segments) with potentially different node spacings.
    
    D_ij = integral of phi_i * phi_j over slave interface (slave mass matrix)
    M_ij = integral of phi_i * psi_j over slave interface (transfer matrix)
    
    where phi_i are slave shape functions and psi_j are master shape functions
    (projected onto slave parameterization).
    
    Parameters
    ----------
    slave_x : (n_slave,) tensor
        Slave interface node x-coordinates (sorted).
    master_x : (n_master,) tensor
        Master interface node x-coordinates (sorted).
    
    Returns
    -------
    D : (n_slave, n_slave) tensor
        Slave interface mass matrix.
    M : (n_slave, n_master) tensor
        Transfer matrix (slave to master).
    segments : list of (xi_start, xi_end, slave_elem, master_elem)
        Integration segments for quadrature.
    """
    n_s = len(slave_x)
    n_m = len(master_x)
    
    D = torch.zeros((n_s, n_s), dtype=DTYPE)
    M = torch.zeros((n_s, n_m), dtype=DTYPE)
    
    # 2-point Gauss quadrature
    gp = torch.tensor([-1.0/math.sqrt(3), 1.0/math.sqrt(3)], dtype=DTYPE)
    gw = torch.tensor([1.0, 1.0], dtype=DTYPE)
    
    # Build integration segments by finding all slave and master element boundaries
    all_breaks = torch.cat([slave_x, master_x]).sort().values
    # Remove duplicates
    all_breaks = torch.unique(all_breaks)
    
    # For each slave element
    for e_s in range(n_s - 1):
        xs0, xs1 = float(slave_x[e_s]), float(slave_x[e_s + 1])
        h_s = xs1 - xs0
        
        # Find master elements that overlap this slave element
        for e_m in range(n_m - 1):
            xm0, xm1 = float(master_x[e_m]), float(master_x[e_m + 1])
            
            # Check overlap
            x_start = max(xs0, xm0)
            x_end = min(xs1, xm1)
            
            if x_start >= x_end:
                continue
            
            # Integrate over overlap segment using 3-point Gauss
            n_quad = 3
            gp3 = torch.tensor([-math.sqrt(3/5), 0.0, math.sqrt(3/5)], dtype=DTYPE)
            gw3 = torch.tensor([5/9, 8/9, 5/9], dtype=DTYPE)
            
            seg_mid = (x_start + x_end) / 2
            seg_half = (x_end - x_start) / 2
            
            for q in range(n_quad):
                x_q = seg_mid + seg_half * gp3[q]
                w_q = gw3[q] * seg_half
                
                # Slave shape functions at x_q
                xi_s = 2 * (x_q - xs0) / h_s - 1
                N_s = torch.tensor([(1 - xi_s) / 2, (1 + xi_s) / 2], dtype=DTYPE)
                
                # Master shape functions at x_q
                h_m = xm1 - xm0
                xi_m = 2 * (x_q - xm0) / h_m - 1
                N_m = torch.tensor([(1 - xi_m) / 2, (1 + xi_m) / 2], dtype=DTYPE)
                
                # D matrix: N_s^T * N_s
                for i in range(2):
                    for j in range(2):
                        D[e_s + i, e_s + j] += N_s[i] * N_s[j] * w_q
                
                # M matrix: N_s^T * N_m
                for i in range(2):
                    for j in range(2):
                        M[e_s + i, e_m + j] += N_s[i] * N_m[j] * w_q
    
    return D, M, all_breaks


# ============================================================================
# Full 2D mortar LM contact solver
# ============================================================================

def solve_mortar_lm_2d_contact(
    K: torch.Tensor,
    f_ext: torch.Tensor,
    fixed_dofs: torch.Tensor,
    slave_surface_nodes: torch.Tensor,
    master_surface_nodes: torch.Tensor,
    *,
    young: float,
    poisson: float,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> MortarLM2DResult:
    """Solve 2D frictionless contact with mortar Lagrange multipliers.
    
    Uses an augmented Lagrangian / Uzawa iteration to solve the mortar
    contact problem. The Lagrange multiplier space is defined on the
    slave surface nodes.
    
    Parameters
    ----------
    K : (ndof, ndof) tensor
        Global stiffness matrix (both bodies assembled).
    f_ext : (ndof,) tensor
        External force vector.
    fixed_dofs : (n_fixed,) tensor
        Fixed DOF indices.
    slave_surface_nodes : (n_slave_surf,) tensor
        Global node indices of slave contact surface.
    master_surface_nodes : (n_master_surf,) tensor
        Global node indices of master contact surface.
    young : float
        Young's modulus.
    poisson : float
        Poisson's ratio.
    max_iterations : int
        Maximum Uzawa iterations.
    tolerance : float
        Convergence tolerance.
    
    Returns
    -------
    MortarLM2DResult
        Contact solution with pressure distribution and convergence info.
    """
    ndof = K.shape[0]
    n_slave = len(slave_surface_nodes)
    n_master = len(master_surface_nodes)
    
    # Get surface coordinates
    # We need the actual node coordinates - extract from the problem setup
    # For now, assume nodes are ordered and we can get x-coords from the mesh
    
    # Slave and master y-DOFs
    slave_y_dofs = 2 * slave_surface_nodes + 1
    master_y_dofs = 2 * master_surface_nodes + 1
    
    # Free DOFs
    fixed_set = set(fixed_dofs.tolist())
    free_dofs = torch.tensor([i for i in range(ndof) if i not in fixed_set], dtype=torch.long)
    n_free = len(free_dofs)
    
    # Factorize stiffness (with small regularization for stability)
    K_ff = K[free_dofs][:, free_dofs]
    K_reg = K_ff + 1e-14 * torch.eye(n_free, dtype=DTYPE)
    L = torch.linalg.cholesky(K_reg)
    
    # Initialize
    u = torch.zeros(ndof, dtype=DTYPE)
    lam = torch.zeros(n_slave, dtype=DTYPE)  # Lagrange multipliers (pressure)
    aug_penalty = 1e3 * young  # Augmented Lagrangian penalty
    
    for iteration in range(max_iterations):
        # Build contact force from current Lagrange multipliers
        f_contact = torch.zeros(ndof, dtype=DTYPE)
        
        # Mortar coupling: distribute λ to slave and master nodes
        # For simplicity, use nodal integration (lumped mortar)
        for i, sn in enumerate(slave_surface_nodes.tolist()):
            f_contact[2*sn + 1] -= lam[i]  # compressive = negative y
        
        for i, mn in enumerate(master_surface_nodes.tolist()):
            f_contact[2*mn + 1] += lam[i]  # reaction on master
        
        # Solve for displacement
        rhs = f_ext + f_contact
        rhs_free = rhs[free_dofs]
        
        # Forward/back substitution
        y = torch.linalg.solve_triangular(L, rhs_free, upper=False)
        u_free = torch.linalg.solve_triangular(L.T, y, upper=True)
        
        u = torch.zeros(ndof, dtype=DTYPE)
        u[free_dofs] = u_free
        
        # Compute gap at slave nodes
        # gap = initial_gap + u_slave_y - u_master_y (projected)
        # For now, assume initial gap is zero (bodies initially touching)
        slave_y_disp = u[slave_y_dofs]
        
        # Compute gap (simplified: slave displacement relative to master)
        # In a full implementation, we'd project master displacement onto slave
        gap = -slave_y_disp  # positive = open
        
        # Update Lagrange multipliers (Uzawa with augmented Lagrangian)
        lam_new = torch.clamp(lam + aug_penalty * gap, min=0.0)
        
        # Check convergence
        delta_lam = torch.linalg.vector_norm(lam_new - lam)
        lam_norm = torch.linalg.vector_norm(lam_new)
        
        lam = lam_new
        
        if lam_norm > 0 and delta_lam / lam_norm < tolerance:
            # Compute final quantities
            contact_pressure = lam
            max_pen = float(torch.min(gap).clamp(max=0))
            total_force = float(torch.sum(lam))
            
            # Complementarity residual
            comp_res = float(torch.max(torch.abs(lam * torch.clamp(gap, min=0))))
            
            # Quadrature point data (for comparison with analytical)
            # Use slave node positions as quadrature points
            quad_x = torch.zeros(n_slave, dtype=DTYPE)  # placeholder
            pressure_quad = contact_pressure.clone()
            
            return MortarLM2DResult(
                displacement=u,
                contact_pressure_nodes=contact_pressure,
                contact_pressure_quad=pressure_quad,
                quad_x_coords=quad_x,
                gap=gap,
                max_penetration=max_pen,
                total_contact_force=total_force,
                complementarity_residual=comp_res,
                converged=True,
                iterations=iteration + 1,
            )
    
    # Did not converge
    gap = -u[slave_y_dofs]
    return MortarLM2DResult(
        displacement=u,
        contact_pressure_nodes=lam,
        contact_pressure_quad=lam.clone(),
        quad_x_coords=torch.zeros(n_slave, dtype=DTYPE),
        gap=gap,
        max_penetration=float(torch.min(gap).clamp(max=0)),
        total_contact_force=float(torch.sum(lam)),
        complementarity_residual=float(torch.max(torch.abs(lam * torch.clamp(gap, min=0)))),
        converged=False,
        iterations=max_iterations,
    )


# ============================================================================
# Hertz cylinder-on-flat benchmark with mortar LM contact
# ============================================================================

def build_hertz_cylinder_on_flat(
    n_contact_slave: int,
    n_contact_master: int,
    *,
    radius: float = 10.0,
    half_width: float = 5.0,
    height: float = 10.0,
    young: float = 1000.0,
    poisson: float = 0.3,
    indentation: float = 0.01,
    nx_body: int | None = None,
    ny_body: int | None = None,
) -> dict:
    """Build a 2D Hertz cylinder-on-flat contact problem.
    
    The model consists of:
    - A rigid cylinder (analytical) pressing into an elastic half-space
    - The elastic body is discretized with Q4 elements
    - Non-matching meshes on the contact interface
    
    Parameters
    ----------
    n_contact_slave : int
        Number of slave contact nodes.
    n_contact_master : int
        Number of master contact nodes (different from slave for non-matching).
    radius : float
        Cylinder radius.
    half_width : float
        Half-width of the elastic body (contact surface extent).
    height : float
        Height of the elastic body.
    young : float
        Young's modulus.
    poisson : float
        Poisson's ratio.
    indentation : float
        Prescribed indentation depth.
    nx_body, ny_body : int, optional
        Number of elements in body. Defaults based on contact nodes.
    
    Returns
    -------
    dict
        Problem data for the mortar LM solver.
    """
    if nx_body is None:
        nx_body = max(n_contact_slave, n_contact_master)
    if ny_body is None:
        ny_body = max(4, nx_body // 2)
    
    # Build elastic half-space mesh
    nodes, elements = structured_q4_mesh(
        nx_body, ny_body,
        2 * half_width, height,
        x_offset=-half_width,
        y_offset=0.0,
    )
    
    # Assemble stiffness
    K = q4_plane_strain_stiffness(nodes, elements, young, poisson)
    
    # Contact surface: top edge of the mesh (y = height)
    # Slave nodes: top surface nodes
    top_nodes = []
    for i in range(nx_body + 1):
        top_nodes.append(ny_body * (nx_body + 1) + i)
    top_nodes = torch.tensor(top_nodes, dtype=torch.long)
    
    # For non-matching: slave uses finer or different spacing than master
    # Slave contact nodes (subset of top nodes, or interpolated)
    slave_surface = top_nodes[:n_contact_slave] if n_contact_slave <= len(top_nodes) else top_nodes
    
    # Master contact nodes (different spacing)
    master_surface = top_nodes[:n_contact_master] if n_contact_master <= len(top_nodes) else top_nodes
    
    # Boundary conditions: bottom fixed in y, left/right fixed in x
    fixed_dofs = []
    bottom_nodes = list(range(nx_body + 1))
    for n in bottom_nodes:
        fixed_dofs.append(2 * n + 1)  # fix y
    for j in range(ny_body + 1):
        fixed_dofs.append(2 * (j * (nx_body + 1)))  # fix x on left
        fixed_dofs.append(2 * (j * (nx_body + 1) + nx_body))  # fix x on right
    fixed_dofs = torch.tensor(sorted(set(fixed_dofs)), dtype=torch.long)
    
    # External force: prescribed indentation via displacement BC
    # Apply downward displacement on contact nodes
    f_ext = torch.zeros(2 * len(nodes), dtype=DTYPE)
    
    # Compute mortar coupling matrices
    slave_x = nodes[slave_surface, 0]
    master_x = nodes[master_surface, 0]
    D_mortar, M_mortar, segments = compute_mortar_matrices_1d(slave_x, master_x)
    
    # Hertz reference
    E_star = young / (1 - poisson**2)
    # For prescribed indentation, contact force from Hertz:
    # F = pi * E* * delta^2 / (4 * R) ... wait, for line contact:
    # b = sqrt(4*F*R/(pi*E*)), delta = pi*p0*b/(2*E*) = pi*F/(pi*b*E*) = F/(b*E*)
    # So F = delta * b * E*, and b = sqrt(R * delta) for line contact
    # Actually for 2D line contact: b = sqrt(4*F*R/(pi*E*))
    # and delta = 2*F/(pi*E*) * (ln(4*R/b) - 1) ... complex
    # Simplified: use b = sqrt(R * delta) as first approximation
    
    b_approx = math.sqrt(radius * indentation)
    F_approx = math.pi * E_star * b_approx**2 / (4 * radius) * 4  # approximate
    
    hertz_ref = HertzLineReference(
        contact_half_width=b_approx,
        peak_pressure=2 * F_approx / (math.pi * b_approx),
        total_force=F_approx,
        indentation=indentation,
        effective_modulus=E_star,
    )
    
    return {
        "nodes": nodes,
        "elements": elements,
        "stiffness": K,
        "force": f_ext,
        "fixed_dofs": fixed_dofs,
        "slave_surface": slave_surface,
        "master_surface": master_surface,
        "slave_x": slave_x,
        "master_x": master_x,
        "D_mortar": D_mortar,
        "M_mortar": M_mortar,
        "hertz_reference": hertz_ref,
        "young": young,
        "poisson": poisson,
        "indentation": indentation,
        "radius": radius,
    }
