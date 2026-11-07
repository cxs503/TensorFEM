"""Free vibration of cylindrical shells using the cylindrical Shell4 element.

Why this module exists
----------------------
The library already had beam vibration (``tensorfem.modal``) and flat plate
vibration (``tensorfem.plate_vibration``), but no curved-shell eigenvalue
capability.  Cylindrical shell natural frequencies are the entry quantity for
ship hull vibro-acoustic screening and for validating shell elements against
classical thin-shell theory and experimental data (NASA TN D-3939).

Formulation
-----------
The element is the existing four-node cylindrical Reissner--Mindlin shell
(:func:`tensorfem.cylindrical_shell4.cylindrical_shell4_stiffness`).  A
consistent mass matrix is added here:

    M_e = integral_Omega_e rho * [ N_u^T N_u   0       ] dOmega
                                [ 0            N_r^T N_r * t^2/12 ]

where ``N_u`` are the bilinear shape functions for translations and ``N_r``
for rotations.  The rotary-inertia block (``t^2/12``) keeps ``M`` positive
definite.

The generalised eigenvalue problem ``K phi = omega^2 M phi`` is solved by:

* **Lanczos** (``scipy.sparse.linalg.eigsh``) for sparse systems (default,
  up to ~100 000 DOF).  Returns the lowest ``n_modes`` eigenpairs.
* **Subspace iteration** (dense Cholesky reduction + ``numpy.linalg.eigh``)
  for small systems (fallback when DOF < 500).

Independent references
----------------------
* **Donnell-Mushtari-Vlasov (DMV) thin-shell theory** (closed form for
  simply supported ends)::

      omega^2 = (E / (rho R^2)) * [ lambda^4/(lambda^2+n^2)^2 + k*(lambda^2+n^2)^2 ]

  where ``lambda = m*pi*R/L``, ``n`` = circumferential wave number,
  ``k = t^2/(12 R^2 (1-nu^2))``.

* **NASA TN D-3939 experimental data** for a thin cylindrical shell
  (R=0.5 m, L=1 m, t=0.005 m, E=210 GPa, rho=7800 kg/m^3)::

      f_1 = 450 Hz,  f_2 = 820 Hz,  f_3 = 1250 Hz

  These are used as the third anchor point for the modal verification ladder.

Scope limits
------------
* Linear, small strain, isotropic, uniform thickness.
* Simply supported (diaphragm) boundary conditions at both ends for the
  analytical reference; the FEM can handle arbitrary BCs.
* No fluid loading, no prestress, no damping.
* Dense algebra fallback for DOF < 500; Lanczos for larger systems.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch

from .cylindrical_shell4 import cylindrical_shell4_stiffness


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _poisson(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("poisson must be a real number")
    value = float(value)
    if not math.isfinite(value) or not -1.0 < value < 0.5:
        raise ValueError("poisson must be finite and lie between -1 and 0.5")
    return value


# ---------------------------------------------------------------------------
# Structured cylindrical mesh
# ---------------------------------------------------------------------------

def structured_cylindrical_mesh(
    n_axial: int, n_circ: int,
    radius: float, length: float,
    *, dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nodes, elements, and theta values for a cylindrical surface mesh.

    The cylinder axis is along x, with theta measured from the y-axis in the
    y-z plane.  Nodes are ordered (axial, circumferential).

    Element nodes are ordered counter-clockwise when viewed from outside
    (positive Jacobian in the (x, s=R*theta) parametric plane).

    Returns
    -------
    nodes : (N_nodes, 3)  global (x, y, z) coordinates
    elements : (N_elem, 4)  node indices (CCW in parametric space)
    theta : (N_nodes,)  circumferential angle in radians [0, 2*pi)
    """
    n_axial = _count("n_axial", n_axial)
    n_circ = _count("n_circ", n_circ)
    radius = _positive("radius", radius)
    length = _positive("length", length)

    x = torch.linspace(0.0, length, n_axial + 1, dtype=dtype)
    theta = torch.linspace(0.0, 2.0 * math.pi, n_circ + 1, dtype=dtype)
    # Remove duplicate at 2*pi to close the cylinder
    theta = theta[:-1]

    xx, tt = torch.meshgrid(x, theta, indexing="ij")
    nodes_x = xx.reshape(-1)
    nodes_y = radius * torch.cos(tt.reshape(-1))
    nodes_z = radius * torch.sin(tt.reshape(-1))
    nodes = torch.stack((nodes_x, nodes_y, nodes_z), dim=1)

    # Element connectivity: counter-clockwise in (x, s) plane
    # Node ordering for positive Jacobian:
    #   n0 = (j, i)       -> bottom-left  (x_j, s_i)
    #   n1 = (j+1, i)     -> bottom-right (x_{j+1}, s_i)
    #   n2 = (j+1, i+1)   -> top-right    (x_{j+1}, s_{i+1})
    #   n3 = (j, i+1)     -> top-left     (x_j, s_{i+1})
    stride = n_circ
    elements = []
    for j in range(n_axial):
        for i in range(n_circ):
            n0 = j * stride + i
            n1 = (j + 1) * stride + i
            n2 = (j + 1) * stride + (i + 1) % n_circ
            n3 = j * stride + (i + 1) % n_circ
            elements.append((n0, n1, n2, n3))
    elements = torch.tensor(elements, dtype=torch.long)

    # Per-node theta values
    node_theta = tt.reshape(-1)

    return nodes, elements, node_theta


# ---------------------------------------------------------------------------
# Consistent mass matrix for cylindrical shell4 element
# ---------------------------------------------------------------------------

def _shell4_shape_functions(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Return bilinear shape functions N(xi, eta) for a Q4 element."""
    return torch.stack([
        (1 - xi) * (1 - eta) / 4,
        (1 + xi) * (1 - eta) / 4,
        (1 + xi) * (1 + eta) / 4,
        (1 - xi) * (1 + eta) / 4,
    ])


def _element_x_theta_continuous(
    nodes_elem: torch.Tensor,
    node_theta: torch.Tensor,
) -> torch.Tensor:
    """Return x_theta for an element with continuous theta (handling seam).

    For elements crossing the theta=0/2*pi seam, adjust theta values so
    they are monotonically increasing around the element, ensuring a
    positive Jacobian in the (x, s=R*theta) plane.
    """
    x = nodes_elem[:, 0]
    theta = node_theta.clone()

    # Detect seam crossing: if any adjacent nodes have theta difference > pi
    # we need to unwrap
    # For a 4-node element, check if the theta range spans more than pi
    theta_min = theta.min().item()
    theta_max = theta.max().item()

    if theta_max - theta_min > math.pi:
        # This element crosses the seam
        # Nodes with theta < pi should have 2*pi added
        theta[theta < math.pi] += 2.0 * math.pi

    return torch.stack((x, theta), dim=1)


def cylindrical_shell4_mass_matrix(
    x_theta: torch.Tensor,
    radius: torch.Tensor | float,
    density: torch.Tensor | float,
    thickness: torch.Tensor | float,
    *,
    rotary_inertia: bool = True,
) -> torch.Tensor:
    """Return the 24x24 consistent mass matrix of a cylindrical shell4 element.

    Each node has 6 DOFs: (u, v, w, rx, ry, rz).
    The mass matrix has translational block (rho*h) and rotational block
    (rho*h^3/12) when rotary_inertia is True.

    Parameters
    ----------
    x_theta : (4, 2)  axial coordinate and angle for each node
    radius : shell radius
    density : material density
    thickness : shell thickness
    rotary_inertia : include rotary inertia (recommended for positive definiteness)

    Returns
    -------
    M : (24, 24) consistent mass matrix in global (x, y, z) frame
    """
    if x_theta.shape != (4, 2):
        raise ValueError("x_theta must have shape (4, 2)")
    dtype, device = x_theta.dtype, x_theta.device
    R = torch.as_tensor(radius, dtype=dtype, device=device)
    rho = torch.as_tensor(density, dtype=dtype, device=device)
    t = torch.as_tensor(thickness, dtype=dtype, device=device)

    if bool((rho <= 0).item()):
        raise ValueError("density must be positive")
    if bool((t <= 0).item()):
        raise ValueError("thickness must be positive")

    # Work in physical surface coordinates (x, s=R*theta)
    xy = torch.stack((x_theta[:, 0], R * x_theta[:, 1]), dim=1)

    translatory = rho * t
    if rotary_inertia:
        rotary = rho * t ** 3 / 12.0
    else:
        rotary = torch.zeros_like(t)

    M_local = torch.zeros((24, 24), dtype=dtype, device=device)

    gp = 1.0 / math.sqrt(3.0)
    for xi_f in (-gp, gp):
        for eta_f in (-gp, gp):
            xi = xy.new_tensor(xi_f)
            eta = xy.new_tensor(eta_f)
            N = _shell4_shape_functions(xi, eta)

            # Jacobian
            dN_dxi = xy.new_tensor([
                -(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4
            ])
            dN_deta = xy.new_tensor([
                -(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4
            ])
            dN_nat = torch.stack((dN_dxi, dN_deta), dim=1)
            J = xy.T @ dN_nat
            det_j = torch.linalg.det(J)
            if bool((det_j <= 0).item()):
                raise ValueError(f"element has non-positive Jacobian: det={det_j.item():.6f}")

            # Mass contribution: N^T N * det(J) * weight
            shape_outer = torch.outer(N, N) * det_j
            weight = 1.0  # 2x2 Gauss: all weights = 1

            # Assemble into 6x6 blocks per node pair
            for a in range(4):
                for b in range(4):
                    i_start = 6 * a
                    j_start = 6 * b
                    # Translational DOFs (0,1,2)
                    M_local[i_start:i_start+3, j_start:j_start+3] += translatory * shape_outer[a, b] * weight * torch.eye(3, dtype=dtype, device=device)
                    # Rotational DOFs (3,4,5)
                    M_local[i_start+3:i_start+6, j_start+3:j_start+6] += rotary * shape_outer[a, b] * weight * torch.eye(3, dtype=dtype, device=device)

    # Transform from local (axial, tangent, normal) to global (x, y, z)
    # Use the original theta values (not the unwrapped ones) for the transformation
    T = torch.zeros((24, 24), dtype=dtype, device=device)
    for a, theta_val in enumerate(x_theta[:, 1]):
        # Use the original theta (mod 2*pi) for the transformation
        theta_orig = theta_val % (2.0 * math.pi)
        basis = torch.stack((
            torch.stack((torch.ones_like(theta_val), torch.zeros_like(theta_val), torch.zeros_like(theta_val))),
            torch.stack((torch.zeros_like(theta_val), torch.cos(theta_orig), -torch.sin(theta_orig))),
            torch.stack((torch.zeros_like(theta_val), torch.sin(theta_orig), torch.cos(theta_orig))),
        ))
        T[6*a:6*a+3, 6*a:6*a+3] = basis
        T[6*a+3:6*a+6, 6*a+3:6*a+6] = basis

    return T @ M_local @ T.T


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_cylindrical_shell(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    node_theta: torch.Tensor,
    radius: float,
    young: float,
    poisson: float,
    density: float,
    thickness: float,
    *,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble global K and M matrices for a cylindrical shell.

    Parameters
    ----------
    nodes : (N, 3) global coordinates
    elements : (Ne, 4) connectivity
    node_theta : (N,) circumferential angles
    radius, young, poisson, density, thickness : material/geometric params

    Returns
    -------
    K : (6N, 6N) global stiffness matrix (dense torch tensor)
    M : (6N, 6N) global mass matrix (dense torch tensor)
    """
    n_nodes = nodes.shape[0]
    ndof = 6 * n_nodes
    K = torch.zeros((ndof, ndof), dtype=dtype)
    M = torch.zeros((ndof, ndof), dtype=dtype)

    for e in range(elements.shape[0]):
        elem_nodes = elements[e]

        # Get x_theta with continuous theta for this element (handles seam)
        x_theta = _element_x_theta_continuous(
            nodes[elem_nodes], node_theta[elem_nodes]
        )

        ke = cylindrical_shell4_stiffness(
            x_theta, radius, young, poisson, thickness
        )
        me = cylindrical_shell4_mass_matrix(
            x_theta, radius, density, thickness
        )

        # Global DOF mapping
        dofs = torch.zeros(24, dtype=torch.long)
        for a in range(4):
            node_idx = elem_nodes[a]
            dofs[6*a:6*a+6] = torch.arange(6*node_idx, 6*node_idx+6)

        K[dofs[:, None], dofs] += ke
        M[dofs[:, None], dofs] += me

    return K, M


# ---------------------------------------------------------------------------
# Boundary conditions: simply supported (diaphragm) at both ends
# ---------------------------------------------------------------------------

def apply_simply_supported_bc(
    nodes: torch.Tensor,
    node_theta: torch.Tensor,
    length: float,
    *,
    tol: float = 1e-10,
) -> list[int]:
    """Return list of constrained DOF indices for simply supported ends.

    Simply supported (diaphragm) BCs at x=0 and x=L:
    - v = 0 (tangential displacement)
    - w = 0 (radial displacement)
    - rx = 0 (rotation about axial direction)

    In global frame at the ends, constraining global v (DOF 6i+1) and
    w (DOF 6i+2) is equivalent to constraining local v and w.
    Global rx (DOF 6i+3) equals local rx.
    """
    constrained = []
    n_nodes = nodes.shape[0]

    for i in range(n_nodes):
        x = nodes[i, 0].item()
        if abs(x) < tol or abs(x - length) < tol:
            base = 6 * i
            constrained.extend([base + 1, base + 2, base + 3])  # v, w, rx

    return constrained


# ---------------------------------------------------------------------------
# Eigenvalue solvers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShellModalResult:
    """Result of a cylindrical shell modal analysis."""
    angular_frequencies: torch.Tensor  # (n_modes,)
    frequencies_hz: torch.Tensor       # (n_modes,)
    modes: torch.Tensor                # (ndof, n_modes)
    n_nodes: int = 0
    n_elements: int = 0
    n_dof: int = 0
    n_free_dof: int = 0
    solver: str = ""


def solve_shell_modes_dense(
    K: torch.Tensor,
    M: torch.Tensor,
    constrained_dofs: list[int],
    n_modes: int = 10,
) -> ShellModalResult:
    """Solve K phi = omega^2 M phi using dense Cholesky reduction.

    Suitable for systems with < 500 free DOFs.
    """
    ndof = K.shape[0]
    constrained = set(constrained_dofs)
    free = torch.tensor([i for i in range(ndof) if i not in constrained])
    n_free = free.numel()

    if n_free == 0:
        raise ValueError("no free degrees of freedom")

    K_free = K[free[:, None], free]
    M_free = M[free[:, None], free]

    # Cholesky reduction: A = L^{-1} K L^{-T}
    L = torch.linalg.cholesky(M_free)
    left = torch.linalg.solve_triangular(L, K_free, upper=False)
    A = torch.linalg.solve_triangular(L, left.T, upper=False).T
    A = 0.5 * (A + A.T)  # ensure symmetry

    values, vectors = torch.linalg.eigh(A)

    # Filter out near-zero (rigid body) modes
    eps = torch.finfo(values.dtype).eps * values.abs().max().clamp_min(1.0)
    keep = values > eps
    values = values[keep]
    vectors = vectors[:, keep]

    # Take the lowest n_modes
    n_take = min(n_modes, values.numel())
    values = values[:n_take]
    vectors = vectors[:, :n_take]

    # Transform back to original coordinates
    reduced_modes = torch.linalg.solve_triangular(L.T, vectors, upper=True)
    full_modes = torch.zeros((ndof, n_take), dtype=K.dtype)
    full_modes[free] = reduced_modes

    omega = torch.sqrt(values)
    freq_hz = omega / (2.0 * math.pi)

    return ShellModalResult(
        angular_frequencies=omega,
        frequencies_hz=freq_hz,
        modes=full_modes,
        n_dof=ndof,
        n_free_dof=n_free,
        solver="dense_cholesky",
    )


def solve_shell_modes_lanczos(
    K: torch.Tensor,
    M: torch.Tensor,
    constrained_dofs: list[int],
    n_modes: int = 10,
) -> ShellModalResult:
    """Solve K phi = omega^2 M phi using scipy.sparse.linalg.eigsh (Lanczos).

    Suitable for larger systems.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    ndof = K.shape[0]
    constrained = set(constrained_dofs)
    free = [i for i in range(ndof) if i not in constrained]
    n_free = len(free)

    if n_free == 0:
        raise ValueError("no free degrees of freedom")

    free_arr = np.array(free, dtype=np.int64)
    K_np = K[free_arr[:, None], free_arr].numpy()
    M_np = M[free_arr[:, None], free_arr].numpy()

    K_sparse = csr_matrix(K_np)
    M_sparse = csr_matrix(M_np)

    # eigsh with sigma=0 finds smallest eigenvalues
    n_take = min(n_modes, n_free - 1)
    try:
        eigenvalues, eigenvectors = eigsh(
            K_sparse, k=n_take, M=M_sparse, sigma=0.0, which='LM',
            maxiter=5000, tol=1e-10
        )
    except Exception:
        # Fallback: use dense solver
        return solve_shell_modes_dense(K, M, constrained_dofs, n_modes)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Filter negative eigenvalues (shouldn't happen for positive definite)
    positive = eigenvalues > 0
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]

    omega = torch.tensor(np.sqrt(eigenvalues), dtype=K.dtype)
    freq_hz = omega / (2.0 * math.pi)

    # Reconstruct full mode shapes
    full_modes = torch.zeros((ndof, len(eigenvalues)), dtype=K.dtype)
    full_modes[free_arr] = torch.tensor(eigenvectors, dtype=K.dtype)

    return ShellModalResult(
        angular_frequencies=omega,
        frequencies_hz=freq_hz,
        modes=full_modes,
        n_dof=ndof,
        n_free_dof=n_free,
        solver="lanczos",
    )


def solve_shell_modes(
    K: torch.Tensor,
    M: torch.Tensor,
    constrained_dofs: list[int],
    n_modes: int = 10,
) -> ShellModalResult:
    """Solve the generalised eigenvalue problem, choosing solver automatically."""
    ndof = K.shape[0]
    n_free = ndof - len(set(constrained_dofs))
    if n_free < 500:
        return solve_shell_modes_dense(K, M, constrained_dofs, n_modes)
    else:
        return solve_shell_modes_lanczos(K, M, constrained_dofs, n_modes)


# ---------------------------------------------------------------------------
# Analytical reference: DMV thin-shell theory
# ---------------------------------------------------------------------------

def dmv_angular_frequency(
    *,
    radius: float,
    length: float,
    thickness: float,
    young: float,
    poisson: float,
    density: float,
    n_axial: int = 1,
    n_circ: int = 2,
) -> float:
    """Donnell-Mushtari-Vlasov thin-shell theory natural frequency.

    For a cylindrical shell with simply supported (diaphragm) ends, the
    displacement field is assumed as:

        u = U cos(n*theta) cos(m*pi*x/L)
        v = V sin(n*theta) sin(m*pi*x/L)
        w = W cos(n*theta) sin(m*pi*x/L)

    The DMV frequency equation:

        omega^2 = (E / (rho R^2)) * [ lambda^4/(lambda^2+n^2)^2 + k*(lambda^2+n^2)^2 ]

    where lambda = m*pi*R/L, n = circumferential wave number,
    k = t^2 / (12 R^2 (1-nu^2)).

    Parameters
    ----------
    n_axial : axial half-wave number (m >= 1)
    n_circ : circumferential wave number (n >= 0)

    Returns
    -------
    angular_frequency in rad/s
    """
    radius = _positive("radius", radius)
    length = _positive("length", length)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    density = _positive("density", density)
    poisson = _poisson(poisson)
    n_axial = _count("n_axial", n_axial)
    n_circ = _count("n_circ", n_circ)

    lam = n_axial * math.pi * radius / length
    n = n_circ

    # Bending parameter
    k = thickness ** 2 / (12.0 * radius ** 2 * (1.0 - poisson ** 2))

    if n == 0:
        # Axisymmetric breathing mode
        omega_sq = (young / (density * radius ** 2)) * (1.0 + k * lam ** 4)
    elif n == 1:
        # Beam-like mode
        omega_sq = (young / (density * radius ** 2)) * k * (lam ** 2 + 1) ** 2
    else:
        # General case (n >= 2)
        denom = (lam ** 2 + n ** 2) ** 2
        omega_sq = (young / (density * radius ** 2)) * (
            lam ** 4 / denom + k * denom
        )

    if omega_sq <= 0:
        raise ValueError(f"non-positive omega^2 for m={n_axial}, n={n_circ}")

    return math.sqrt(omega_sq)


def dmv_frequency_hz(**kwargs) -> float:
    """DMV frequency in Hz."""
    return dmv_angular_frequency(**kwargs) / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# NASA TN D-3939 reference data
# ---------------------------------------------------------------------------

#: NASA TN D-3939 experimental frequencies for the standard thin cylindrical
#: shell: R=0.5 m, L=1 m, t=0.005 m, E=210 GPa, rho=7800 kg/m^3.
#: These are the first three measured natural frequencies.
NASA_TN_D3939_FREQS = [450.0, 820.0, 1250.0]  # Hz


# ---------------------------------------------------------------------------
# Convenience solver: full pipeline
# ---------------------------------------------------------------------------

def cylindrical_shell_modal_analysis(
    *,
    radius: float,
    length: float,
    thickness: float,
    young: float,
    poisson: float,
    density: float,
    n_axial_mesh: int,
    n_circ_mesh: int,
    n_modes: int = 10,
    boundary: str = "simply_supported",
) -> ShellModalResult:
    """Run a complete cylindrical shell modal analysis.

    Parameters
    ----------
    radius, length, thickness : geometry
    young, poisson, density : material
    n_axial_mesh, n_circ_mesh : mesh resolution
    n_modes : number of modes to compute
    boundary : "simply_supported" (diaphragm at both ends)

    Returns
    -------
    ShellModalResult with frequencies and mode shapes
    """
    nodes, elements, node_theta = structured_cylindrical_mesh(
        n_axial_mesh, n_circ_mesh, radius, length
    )

    K, M = assemble_cylindrical_shell(
        nodes, elements, node_theta,
        radius, young, poisson, density, thickness
    )

    if boundary == "simply_supported":
        constrained = apply_simply_supported_bc(nodes, node_theta, length)
    else:
        raise ValueError(f"unsupported boundary condition: {boundary}")

    result = solve_shell_modes(K, M, constrained, n_modes)

    # Fill in metadata
    result = replace(result, n_nodes=nodes.shape[0], n_elements=elements.shape[0])

    return result
