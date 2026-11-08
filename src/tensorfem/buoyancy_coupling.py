"""Hydrostatic buoyancy coupling for floating plates.

When a plate floats on a liquid surface the hydrostatic pressure provides a
restoring force proportional to the local deflection:

    p_buoyancy = -rho_w * g * w

This is mathematically identical to a Winkler foundation with coefficient

    k_buoyancy = rho_w * g

where rho_w is the water density (≈1025 kg/m³ for seawater) and g is
gravitational acceleration (9.81 m/s²).

For a floating plate under a concentrated point load *P*, the coupled
problem is:

    D * nabla^4 w + k_b * w = P * delta(x - x_0, y - y_0)

where D = Eh³/(12(1-ν²)) is the flexural rigidity and k_b = ρ_w·g.

This module provides:
- Buoyancy foundation stiffness assembly (reuses Winkler infrastructure)
- Analytical reference solutions for floating plates
- Characteristic length scale computation

References
----------
[1] Hetényi, M., "Beams on Elastic Foundation", 1946.
[2] Timoshenko & Woinowsky-Krieger, "Theory of Plates and Shells", §62.
[3] Shtekel, A.I., "Floating Structures on Elastic Foundation", 1963.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .plate import structured_square_mesh, q4_mindlin_stiffness, _shape
from .winkler_foundation import q4_winkler_foundation_stiffness


@dataclass(frozen=True)
class BuoyancyResult:
    """Result of floating plate analysis."""
    displacement: torch.Tensor  # (ndof,)
    nodes: torch.Tensor         # (n_nodes, 2)
    elements: torch.Tensor      # (n_elements, 4)
    buoyancy_k: float           # rho_w * g
    point_load: float           # P
    load_position: tuple        # (x_0, y_0)


def buoyancy_coefficient(
    water_density: float = 1025.0,
    gravity: float = 9.81,
) -> float:
    """Return the buoyancy foundation coefficient k_b = rho_w * g.
    
    Parameters
    ----------
    water_density : float
        Water density (kg/m³). Default 1025 for seawater.
    gravity : float
        Gravitational acceleration (m/s²). Default 9.81.
    
    Returns
    -------
    k_b : float
        Buoyancy coefficient (N/m³).
    """
    return water_density * gravity


def characteristic_length(D: float, k_b: float) -> float:
    """Compute the characteristic length for a plate on buoyancy foundation.
    
    L_c = (4*D / k_b)^(1/4)
    
    This is the length scale over which the plate "feels" a point load.
    
    Parameters
    ----------
    D : float
        Flexural rigidity (N·m).
    k_b : float
        Buoyancy coefficient (N/m³).
    
    Returns
    -------
    L_c : float
        Characteristic length (m).
    """
    return (4.0 * D / k_b) ** 0.25


def solve_floating_plate_point_load(
    nx: int,
    side: float,
    young: float,
    poisson: float,
    thickness: float,
    water_density: float = 1025.0,
    gravity: float = 9.81,
    point_load: float = 1000.0,
    load_position: tuple = (0.5, 0.5),
    boundary: str = "free",
    dtype=torch.float64,
) -> BuoyancyResult:
    """Solve a floating plate under a concentrated point load.
    
    The plate floats on water, so the buoyancy provides a Winkler-type
    foundation with k = rho_w * g.
    
    Parameters
    ----------
    nx : int
        Number of elements per side.
    side : float
        Plate side length (m).
    young : float
        Young's modulus (Pa).
    poisson : float
        Poisson's ratio.
    thickness : float
        Plate thickness (m).
    water_density : float
        Water density (kg/m³).
    gravity : float
        Gravitational acceleration (m/s²).
    point_load : float
        Concentrated load magnitude (N).
    load_position : tuple
        (x, y) position of the point load.
    boundary : str
        Boundary condition: "free", "simply_supported", or "clamped".
    
    Returns
    -------
    result : BuoyancyResult
    """
    nodes, elements = structured_square_mesh(nx, side, dtype=dtype)
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes
    
    # Assembly
    K = torch.zeros((ndof, ndof), dtype=dtype)
    f = torch.zeros(ndof, dtype=dtype)
    
    k_b = buoyancy_coefficient(water_density, gravity)
    
    for conn in elements:
        xy = nodes[conn, :2]
        
        # Plate bending stiffness
        ke = q4_mindlin_stiffness(xy, young, poisson, thickness)
        
        # Buoyancy foundation (Winkler with k = rho_w * g)
        ke += q4_winkler_foundation_stiffness(xy, k_b)
        
        # Assemble
        dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
    
    # Apply point load (distributed to nearest node)
    x0, y0 = load_position
    min_dist = float('inf')
    load_node = 0
    for i in range(n_nodes):
        d = (nodes[i, 0] - x0)**2 + (nodes[i, 1] - y0)**2
        if d < min_dist:
            min_dist = d
            load_node = i
    f[3 * load_node] = point_load  # w-DOF
    
    # Boundary conditions
    tol = side * 1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - side).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - side).abs() < tol)
    edge = xedge | yedge
    
    if boundary == "simply_supported":
        fixed = torch.cat((
            3 * torch.nonzero(edge, as_tuple=False).flatten(),
            3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1,
            3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2,
        )).unique()
    elif boundary == "clamped":
        fixed = torch.cat((
            3 * torch.nonzero(edge, as_tuple=False).flatten(),
            3 * torch.nonzero(edge, as_tuple=False).flatten() + 1,
            3 * torch.nonzero(edge, as_tuple=False).flatten() + 2,
        )).unique()
    else:  # free - no boundary constraints
        fixed = torch.tensor([], dtype=torch.long)
    
    # Solve
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    
    K_ff = K[free][:, free]
    f_f = f[free]
    
    u_f = torch.linalg.solve(K_ff, f_f)
    
    u = torch.zeros(ndof, dtype=dtype)
    u[free] = u_f
    
    return BuoyancyResult(
        displacement=u,
        nodes=nodes,
        elements=elements,
        buoyancy_k=k_b,
        point_load=point_load,
        load_position=load_position,
    )


def analytical_floating_plate_center_deflection(
    side: float,
    D: float,
    k_b: float,
    P: float,
    M: int = 100,
    N: int = 100,
) -> float:
    """Analytical solution for simply supported floating plate under center point load.
    
    Uses double Fourier sine series:
    w(x,y) = sum_m sum_n (P * sin(m*pi*x0/a)*sin(n*pi*y0/b)) / 
             (a*b/4 * (D*((m*pi/a)^2+(n*pi/b)^2)^2 + k_b)) * sin(m*pi*x/a)*sin(n*pi*y/b)
    
    For center load at (a/2, b/2) and evaluation at center:
    w_center = sum_m sum_n (16*P / (a*b * (D*((m*pi/a)^2+(n*pi/b)^2)^2 + k_b))) 
               * sin(m*pi/2)^2 * sin(n*pi/2)^2
    
    Parameters
    ----------
    side : float
        Plate side length (square plate, a = b = side).
    D : float
        Flexural rigidity.
    k_b : float
        Buoyancy coefficient.
    P : float
        Point load magnitude.
    M, N : int
        Number of terms in each direction.
    
    Returns
    -------
    w_center : float
        Center deflection.
    """
    a = b = side
    w = 0.0
    for m in range(1, 2*M+1, 2):
        for n in range(1, 2*N+1, 2):
            alpha_m = m * math.pi / a
            beta_n = n * math.pi / b
            denom = D * (alpha_m**2 + beta_n**2)**2 + k_b
            # sin(m*pi/2) = ±1 for odd m
            sign_m = 1 if (m % 4 == 1) else -1
            sign_n = 1 if (n % 4 == 1) else -1
            w += 16.0 * P / (a * b * denom) * sign_m * sign_n
    return w
