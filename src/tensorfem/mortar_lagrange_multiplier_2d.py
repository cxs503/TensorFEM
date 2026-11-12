"""2D mortar contact with Lagrange multipliers for non-matching meshes.

This module implements frictionless unilateral contact in 2-D (plane-strain
Q4 solids) where the interface constraint is enforced by **mortar segment
integration** and **Lagrange multipliers** -- not by nodal (lumped) coupling
and not by penalty regularisation.

Theory
------
The mortar method puts the Lagrange multiplier (contact pressure) field on the
*slave* interface discretisation,

    p(x) = sum_i lambda_i phi_i(x),

and enforces the constraint in its integral (weak) form

    g_i = int_Gamma phi_i(x) g(x) dGamma,
    g(x) = g0(x) - u_n^slave(x) + u_n^master(x).

For piecewise-linear interface bases the discrete operators are the interface
mass matrix and the transfer matrix

    D_ij = int phi_i phi_j dGamma      (slave mass:  g = a - D u_s + M u_m)
    M_ij = int phi_i psi_j dGamma      (slave -> master transfer)
    a_i  = int phi_i g0 dGamma         (integrated initial gap)
    L_i  = int phi_i = sum_j D_ij      (tributary length)

The discrete problem is the saddle-point system

    [ K   B^T ] [u  ]   [f]
    [ B   0   ] [lam] = [a]

with ``B = D`` on the slave normal DOFs and ``B = -M`` on the master normal
DOFs, together with the complementarity conditions ``lambda >= 0``,
``g >= 0``, ``lambda * g = 0``.  It is solved here by a primal-dual
active-set iteration (finite termination for this LCP).

The integrated contact force per unit length is ``sum_i lambda_i L_i`` -- an
*integral*, not a nodal sum -- and it balances the support reaction.  Because
``sum_j M_ij = sum_j D_ij = L_i`` (partition of unity of both interface
bases), the total load transferred across the interface is conserved for any
difference between the slave and master discretisations, and a constant
pressure field is reproduced exactly on non-matching meshes.

Reference solution (corrected)
------------------------------
For a cylinder of radius ``R`` pressed with a line load ``F`` onto an elastic
half-plane (plane strain, ``E* = E / (1 - nu^2)``) the well-posed Hertz
relations are

    b  = sqrt(4 F R / (pi E*)),   p0 = 2 F / (pi b) = sqrt(F E* / (pi R)),
    p(x) = p0 sqrt(1 - (x/b)^2).

In 2-D the contact *approach* ``delta`` is **not** a well-posed function of
``F``: it diverges logarithmically with the arbitrary far-field reference
datum (Johnson 1985, Sec. 5.2).  The 3-D sphere identity ``a = sqrt(R delta)``
must therefore **not** be transplanted to a 2-D line contact: it is
inconsistent with ``b(F)`` above (12.8 % low) and mixes two incompatible
formula sets.  Verification in this module is accordingly done in *load
space*: the approach is prescribed, the reaction ``F`` is measured from the FE
solution, and ``b(F)`` / ``p0(F)`` are compared with the FE contact half-width
/ peak pressure.  ``HertzLineReference.indentation`` is kept for bookkeeping
only and is flagged datum dependent.

References
----------
[1] Wriggers, P., *Computational Contact Mechanics*, 2nd ed., Springer, 2006.
[2] Puso, M.A., "A 3D mortar method for solid mechanics", IJNME 59 (2004)
    295-315 (segment clipping and mortar integrals).
[3] Hertz, H., J. reine angew. Math. 92 (1882) 156-171.
[4] Johnson, K.L., *Contact Mechanics*, CUP, 1985, Sec. 5.2.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

DTYPE = torch.float64

_GP3 = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
_GW3 = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)


# ============================================================================
# Hertz line-contact reference solution
# ============================================================================

@dataclass(frozen=True)
class HertzLineReference:
    """Analytical Hertz line-contact solution (cylinder on half-plane).

    ``indentation`` is **datum dependent** in 2-D (logarithmic divergence) and
    must not be used as an acceptance quantity; use ``contact_half_width`` and
    ``peak_pressure``, which depend on the line load only.
    """
    contact_half_width: float   # b
    peak_pressure: float        # p0
    total_force: float          # F per unit length
    indentation: float          # delta (datum dependent, 2-D log divergence)
    effective_modulus: float    # E* = E/(1-nu^2)

    def consistency_residual(self) -> float:
        """Relative residual of the analytic identity ``F = pi p0 b / 2``.

        ``b = sqrt(4FR/(pi E*))``, ``p0 = 2F/(pi b)`` and ``F = pi p0 b/2`` are
        algebraically equivalent, so a non-zero residual means the reference
        bundle was assembled from incompatible formulas.
        """
        f_from_bundle = math.pi * self.peak_pressure * self.contact_half_width / 2.0
        return abs(self.total_force - f_from_bundle) / max(abs(self.total_force), 1e-300)


def hertz_line_reference(
    force_per_length: float,
    radius: float,
    young: float,
    poisson: float,
) -> HertzLineReference:
    """Hertz line-contact solution for a prescribed line load (mutually consistent)."""
    if force_per_length <= 0.0:
        raise ValueError("force_per_length must be positive")
    E_star = young / (1.0 - poisson**2)          # plane-strain effective modulus
    b = math.sqrt(4.0 * force_per_length * radius / (math.pi * E_star))
    p0 = 2.0 * force_per_length / (math.pi * b)
    delta = 2.0 * force_per_length / (math.pi * E_star) * math.log(4.0 * radius / b)
    return HertzLineReference(b, p0, force_per_length, delta, E_star)


def hertz_pressure_profile(x: torch.Tensor, b: float, p0: float) -> torch.Tensor:
    """Hertz pressure distribution ``p(x) = p0 * sqrt(1 - (x/b)^2)``."""
    xi = x / b
    return p0 * torch.sqrt(torch.clamp(1.0 - xi**2, min=0.0))


# ============================================================================
# Q4 plane-strain element matrices and assembly
# ============================================================================

def _q4_element_matrices(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young: float,
    poisson: float,
) -> torch.Tensor:
    """Vectorised Q4 plane-strain element stiffness matrices, shape (n_elem,8,8)."""
    coords = nodes[elements]                        # (n,4,2)
    c = young / ((1 + poisson) * (1 - 2 * poisson))
    D_mat = c * torch.tensor([
        [1 - poisson, poisson, 0.0],
        [poisson, 1 - poisson, 0.0],
        [0.0, 0.0, (1 - 2 * poisson) / 2],
    ], dtype=DTYPE)

    gp = 1.0 / math.sqrt(3.0)
    ke = torch.zeros((len(elements), 8, 8), dtype=DTYPE)
    for (xi, eta) in ((-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)):
        dNdxi = torch.tensor([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
        ], dtype=DTYPE) / 4

        J = torch.einsum("ki,nil->nkl", dNdxi, coords)
        detJ = torch.linalg.det(J)
        invJ = torch.linalg.inv(J)
        dNdx = torch.einsum("nlk,ki->nli", invJ, dNdxi)

        B = torch.zeros((len(elements), 3, 8), dtype=DTYPE)
        B[:, 0, 0::2] = dNdx[:, 0, :]
        B[:, 1, 1::2] = dNdx[:, 1, :]
        B[:, 2, 0::2] = dNdx[:, 1, :]
        B[:, 2, 1::2] = dNdx[:, 0, :]

        ke += detJ.view(-1, 1, 1) * (B.transpose(1, 2) @ D_mat @ B)
    return ke


def _element_dof_map(elements: torch.Tensor) -> torch.Tensor:
    return torch.stack([2 * elements, 2 * elements + 1], dim=2).reshape(len(elements), 8)


def q4_plane_strain_stiffness(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young: float,
    poisson: float,
) -> torch.Tensor:
    """Assemble the dense global stiffness matrix for Q4 plane strain."""
    n_dof = 2 * len(nodes)
    ke = _q4_element_matrices(nodes, elements, young, poisson)
    dof_map = _element_dof_map(elements)
    rows = dof_map.unsqueeze(2).expand(-1, 8, 8).reshape(-1)
    cols = dof_map.unsqueeze(1).expand(-1, 8, 8).reshape(-1)
    K = torch.zeros((n_dof, n_dof), dtype=DTYPE)
    K.index_put_((rows, cols), ke.reshape(-1), accumulate=True)
    return K


def q4_plane_strain_stiffness_csr(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young: float,
    poisson: float,
):
    """Assemble the global stiffness as a SciPy CSR matrix (large meshes)."""
    from scipy.sparse import coo_matrix

    ke = _q4_element_matrices(nodes, elements, young, poisson)
    dof_map = _element_dof_map(elements)
    rows = dof_map.unsqueeze(2).expand(-1, 8, 8).reshape(-1).numpy()
    cols = dof_map.unsqueeze(1).expand(-1, 8, 8).reshape(-1).numpy()
    vals = ke.reshape(-1).numpy()
    n_dof = 2 * len(nodes)
    return coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()


def element_strain_energy(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    displacement: torch.Tensor,
    young: float,
    poisson: float,
) -> torch.Tensor:
    """Element-wise strain energy ``1/2 u_e^T ke u_e`` (patch-test evidence)."""
    ke = _q4_element_matrices(nodes, elements, young, poisson)
    ue = displacement[_element_dof_map(elements)]
    return 0.5 * torch.einsum("ni,nij,nj->n", ue, ke, ue)


# ============================================================================
# Structured Q4 meshing
# ============================================================================

def _graded_axis(lo: float, length: float, n: int, power: float, from_end: bool) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, n + 1, dtype=DTYPE)
    if power == 1.0:
        return lo + length * t
    s = (1.0 - t) if from_end else t
    s = s**power
    if from_end:
        s = 1.0 - s
    return lo + length * s


def structured_q4_mesh(
    nx: int, ny: int,
    Lx: float, Ly: float,
    *,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    grading: float = 1.0,
    grading_x: float | None = None,
    grading_y: float | None = None,
    grade_from: str = "start",
    grade_from_x: str | None = None,
    grade_from_y: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured Q4 mesh over ``[x_offset, x_offset+Lx] x [y_offset, y_offset+Ly]``.

    ``grading`` (backward-compatible alias) applies to both axes;
    ``grading_x`` / ``grading_y`` override per axis.  ``grade_from='start'``
    concentrates nodes toward the low-coordinate end, ``'end'`` toward the high
    end (both end points stay exact).
    """
    gx = grading if grading_x is None else grading_x
    gy = grading if grading_y is None else grading_y
    fx = (grade_from_x or grade_from) == "end"
    fy = (grade_from_y or grade_from) == "end"

    x = _graded_axis(x_offset, Lx, nx, gx, fx)
    y = _graded_axis(y_offset, Ly, ny, gy, fy)
    grid = torch.zeros((ny + 1, nx + 1, 2), dtype=DTYPE)
    grid[:, :, 0] = x.unsqueeze(0)
    grid[:, :, 1] = y.unsqueeze(1)
    nodes = grid.reshape(-1, 2)

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elements.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    elements = torch.tensor(elements, dtype=torch.long)
    return nodes, elements


def top_edge_nodes(nx: int, ny: int) -> torch.Tensor:
    """Global node ids of the top edge (``y = const``) of a structured mesh."""
    base = ny * (nx + 1)
    return torch.arange(base, base + nx + 1, dtype=torch.long)


def bottom_edge_nodes(nx: int, ny: int) -> torch.Tensor:
    """Global node ids of the bottom edge of a structured mesh."""
    return torch.arange(0, nx + 1, dtype=torch.long)


# ============================================================================
# Mortar interface operators (1-D segment integration, non-matching meshes)
# ============================================================================

def compute_mortar_matrices_1d(
    slave_x: torch.Tensor,
    master_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mortar mass ``D`` and transfer ``M`` for a 1-D non-matching interface.

    The interface is clipped into segments whose ends are the union of the
    slave and master element ends (Puso 2004); on each segment the products
    ``phi_i phi_j`` and ``phi_i psi_j`` are integrated with 3-point Gauss
    quadrature, which is exact for piecewise-linear interface bases.

    Returns
    -------
    D : (n_slave, n_slave)   ``D_ij = int phi_i phi_j``
    M : (n_slave, n_master)  ``M_ij = int phi_i psi_j``
    breaks : (n_breaks,)     mortar segment end points
    """
    slave_x = torch.as_tensor(slave_x, dtype=DTYPE).reshape(-1)
    master_x = torch.as_tensor(master_x, dtype=DTYPE).reshape(-1)
    if slave_x.numel() < 2 or master_x.numel() < 2:
        raise ValueError("slave_x and master_x need at least two nodes")
    lo = max(float(slave_x[0]), float(master_x[0]))
    hi = min(float(slave_x[-1]), float(master_x[-1]))
    if hi <= lo:
        raise ValueError("slave and master interfaces do not overlap")

    inner = torch.cat([
        slave_x[(slave_x > lo) & (slave_x < hi)],
        master_x[(master_x > lo) & (master_x < hi)],
    ])
    breaks = torch.cat([torch.tensor([lo, hi], dtype=DTYPE), inner]).unique(sorted=True)

    xa, xb = breaks[:-1], breaks[1:]
    mid = 0.5 * (xa + xb)
    half = 0.5 * (xb - xa)

    ns, nm = slave_x.numel(), master_x.numel()
    D = torch.zeros((ns, ns), dtype=DTYPE)
    M = torch.zeros((ns, nm), dtype=DTYPE)

    gp = torch.tensor(_GP3, dtype=DTYPE)
    gw = torch.tensor(_GW3, dtype=DTYPE)
    x_q = (mid.unsqueeze(1) + half.unsqueeze(1) * gp.unsqueeze(0)).reshape(-1)
    w_q = (half.unsqueeze(1) * gw.unsqueeze(0)).reshape(-1)

    def locate(nodes: torch.Tensor):
        e = torch.searchsorted(nodes, x_q, right=True) - 1
        e = torch.clamp(e, 0, nodes.numel() - 2)
        xi = 2.0 * (x_q - nodes[e]) / (nodes[e + 1] - nodes[e]) - 1.0
        return e, xi

    e_s, xi_s = locate(slave_x)
    e_m, xi_m = locate(master_x)
    N_s = torch.stack([(1.0 - xi_s) / 2.0, (1.0 + xi_s) / 2.0], dim=1)
    N_m = torch.stack([(1.0 - xi_m) / 2.0, (1.0 + xi_m) / 2.0], dim=1)

    nq = x_q.numel()
    i2 = torch.arange(2)
    i_idx = e_s.unsqueeze(1) + i2.unsqueeze(0)
    m_idx = e_m.unsqueeze(1) + i2.unsqueeze(0)
    w = w_q.view(-1, 1, 1)

    D.index_put_(
        (i_idx.unsqueeze(2).expand(nq, 2, 2), i_idx.unsqueeze(1).expand(nq, 2, 2)),
        N_s.unsqueeze(2) * N_s.unsqueeze(1) * w, accumulate=True,
    )
    M.index_put_(
        (i_idx.unsqueeze(2).expand(nq, 2, 2), m_idx.unsqueeze(1).expand(nq, 2, 2)),
        N_s.unsqueeze(2) * N_m.unsqueeze(1) * w, accumulate=True,
    )
    return D, M, breaks


@dataclass(frozen=True)
class MortarInterface2D:
    """Discrete mortar interface operators for frictionless normal contact."""
    slave_x: torch.Tensor
    master_x: torch.Tensor
    D: torch.Tensor
    M: torch.Tensor
    L_slave: torch.Tensor
    L_master: torch.Tensor
    a: torch.Tensor
    slave_gap0: torch.Tensor | None = None
    master_gap0: torch.Tensor | None = None

    def partition_of_unity_error(self) -> float:
        """``max|rowsum(M) - rowsum(D)| / max rowsum(D)`` (constant-field transfer)."""
        diff = (self.M.sum(dim=1) - self.D.sum(dim=1)).abs().max()
        return float(diff / max(float(self.D.sum(dim=1).max()), 1e-300))

    def force_conservation_error(self) -> float:
        """``|sum(L_slave) - sum(L_master)| / sum(L_slave)`` (total load transfer)."""
        return abs(float(self.L_slave.sum() - self.L_master.sum())) / float(self.L_slave.sum())


def build_mortar_interface(
    slave_x: torch.Tensor,
    master_x: torch.Tensor,
    *,
    slave_gap0: torch.Tensor | None = None,
    master_gap0: torch.Tensor | None = None,
) -> MortarInterface2D:
    """Build mortar operators including the integrated initial gap.

    ``g0(x)`` is interpolated linearly from ``master_gap0`` (``psi``) and
    ``slave_gap0`` (``phi``); ``a = M @ master_gap0 - D @ slave_gap0``.
    """
    D, M, _ = compute_mortar_matrices_1d(slave_x, master_x)
    ns, nm = D.shape[0], M.shape[1]
    g_s = torch.zeros(ns, dtype=DTYPE) if slave_gap0 is None else \
        torch.as_tensor(slave_gap0, dtype=DTYPE).reshape(-1)
    g_m = torch.zeros(nm, dtype=DTYPE) if master_gap0 is None else \
        torch.as_tensor(master_gap0, dtype=DTYPE).reshape(-1)
    if g_s.numel() != ns or g_m.numel() != nm:
        raise ValueError("gap arrays must match the interface node counts")
    return MortarInterface2D(
        slave_x=torch.as_tensor(slave_x, dtype=DTYPE).reshape(-1),
        master_x=torch.as_tensor(master_x, dtype=DTYPE).reshape(-1),
        D=D, M=M,
        L_slave=D.sum(dim=1),
        L_master=M.sum(dim=0),
        a=M @ g_m - D @ g_s,
        slave_gap0=g_s,
        master_gap0=g_m,
    )


def _interp(nodes: torch.Tensor, values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation (clamped extrapolation) on ``nodes``."""
    e = torch.searchsorted(nodes, x, right=True) - 1
    e = torch.clamp(e, 0, nodes.numel() - 2)
    xi = (x - nodes[e]) / (nodes[e + 1] - nodes[e])
    return (1.0 - xi) * values[e] + xi * values[e + 1]


# ============================================================================
# Mortar contact solver (primal-dual active set)
# ============================================================================

@dataclass(frozen=True)
class MortarContact2DResult:
    """Result of a 2-D mortar Lagrange-multiplier contact solve."""
    displacement: torch.Tensor            # (ndof,)
    lam: torch.Tensor                     # (n_slave,) nodal contact pressure
    contact_pressure_nodes: torch.Tensor  # alias of ``lam``
    gap_nodes: torch.Tensor               # (n_slave,) mortar-integrated gap g_i
    quad_x: torch.Tensor                  # sample abscissae (nodes + midpoints)
    quad_pressure: torch.Tensor           # p(x_q) = sum lambda_i phi_i(x_q)
    quad_gap: torch.Tensor                # pointwise gap g(x_q)
    active: torch.Tensor                  # bool mask, active contact set
    contact_force: float                  # int p dGamma (per unit length)
    max_penetration: float                # max(0, -min pointwise gap)
    complementarity_residual: float       # max |lambda_i g_i|
    reaction_balance_error: float         # |support reaction - int p dGamma| / int p dGamma
    converged: bool
    iterations: int
    slave_uy: torch.Tensor
    master_uy: torch.Tensor | None = None


def pressure_half_width(slave_x: torch.Tensor, lam: torch.Tensor) -> float:
    """FE contact half-width: right end of the active pressure support.

    The nodal pressure is piecewise linear, so its support ends exactly at the
    first inactive node after the last active one; the value carries an O(h)
    discretisation bias and is reported as such.
    """
    act = torch.nonzero(lam > 0.0).flatten().flatten()
    if act.numel() == 0:
        return 0.0
    i = int(act.max())
    return float(slave_x[min(i + 1, slave_x.numel() - 1)])


def solve_mortar_contact_2d(
    K,
    f_ext: torch.Tensor,
    fixed_dofs: torch.Tensor,
    iface: MortarInterface2D,
    slave_y_dofs: torch.Tensor,
    *,
    master_y_dofs: torch.Tensor | None = None,
    young: float = 1.0,
    max_iterations: int = 60,
    gap_tolerance: float | None = None,
    solve_mode: str = "auto",
) -> MortarContact2DResult:
    """Solve frictionless 2-D mortar contact with Lagrange multipliers.

    Parameters
    ----------
    K : (ndof,ndof) dense ``torch`` tensor or SciPy CSR matrix
        Global stiffness of all bodies.
    f_ext : (ndof,) external load vector.
    fixed_dofs : DOF indices with prescribed zero displacement.
    iface : mortar interface operators (:func:`build_mortar_interface`).
    slave_y_dofs : global ``y`` DOFs of the slave interface nodes.
    master_y_dofs : global ``y`` DOFs of the master interface nodes, or
        ``None`` for a rigid master surface (no equilibrium DOFs; the
        integrated reaction is still captured by the force balance).
    solve_mode : ``"auto"``, ``"dense"`` or ``"sparse"``.

    Notes
    -----
    Primal-dual active set: solve the equality-constrained problem on the
    current active set, then update ``A <- {lam > 0} U {g < 0}``.  The
    iteration terminates finitely for this LCP; a non-converged solve is
    reported honestly (``converged=False``) instead of silently returning a
    partially converged state.
    """
    f_ext = torch.as_tensor(f_ext, dtype=DTYPE).reshape(-1)
    ndof = f_ext.numel()
    fixed = torch.as_tensor(fixed_dofs, dtype=torch.long).reshape(-1)
    free_mask = torch.ones(ndof, dtype=torch.bool)
    free_mask[fixed] = False
    free_idx = torch.nonzero(free_mask).flatten()
    n_free = int(free_idx.numel())

    D, M, a = iface.D, iface.M, iface.a
    n_s = D.shape[0]
    L = iface.L_slave

    slave_y_dofs = torch.as_tensor(slave_y_dofs, dtype=torch.long).reshape(-1)
    master_y_dofs = None if master_y_dofs is None else \
        torch.as_tensor(master_y_dofs, dtype=torch.long).reshape(-1)
    iface_dofs = slave_y_dofs if master_y_dofs is None else \
        torch.cat([slave_y_dofs, master_y_dofs])
    if bool((~free_mask[iface_dofs]).any()):
        raise ValueError("interface DOFs must be free")

    pos = torch.full((ndof,), -1, dtype=torch.long)
    pos[free_idx] = torch.arange(n_free, dtype=torch.long)
    # ``slave_pos`` / ``master_pos`` are positions inside the free-DOF vector
    # (used to build the constraint matrix); ``slave_y_dofs`` / ``master_y_dofs``
    # are global DOF ids (used to index the full displacement vector).
    slave_pos = pos[slave_y_dofs]
    master_pos = None if master_y_dofs is None else pos[master_y_dofs]

    if gap_tolerance is None:
        gap_tolerance = 1e-9 * float(L.sum())
    # Fixed physical scaling for the active-set function ``lambda - c g``:
    # c has the units of stiffness per unit length, and ``young / length`` is
    # a representative interface stiffness.  A *fixed* c keeps the iteration
    # stable; adapting c from ``max|lambda| / max|g|`` amplifies round-off as
    # soon as the converged gaps approach zero.
    active_scale = young / max(float(L.sum()), 1e-300)

    is_sparse_K = hasattr(K, "tocsr")
    use_dense = solve_mode == "dense" or (solve_mode == "auto" and n_free <= 3000)
    if use_dense:
        Kd = torch.as_tensor(K.toarray() if is_sparse_K else K, dtype=DTYPE)
        Kff = Kd[free_idx][:, free_idx]
    else:
        from scipy.sparse import csr_matrix
        Kcsr = K.tocsr() if is_sparse_K else csr_matrix(
            torch.as_tensor(K, dtype=DTYPE).numpy())
        idx = free_idx.numpy()
        Kff_sp = Kcsr[idx][:, idx]

    def assemble_B(act_idx: torch.Tensor) -> torch.Tensor:
        n_a = act_idx.numel()
        B = torch.zeros((n_a, n_free), dtype=DTYPE)
        rows = torch.arange(n_a)
        B.index_put_(
            (rows.unsqueeze(1).expand(-1, n_s),
             slave_pos.unsqueeze(0).expand(n_a, -1)),
            D[act_idx], accumulate=True)
        if master_pos is not None:
            B.index_put_(
                (rows.unsqueeze(1).expand(-1, M.shape[1]),
                 master_pos.unsqueeze(0).expand(n_a, -1)),
                -M[act_idx], accumulate=True)
        return B

    def solve_system(act_idx: torch.Tensor):
        B = assemble_B(act_idx)
        rhs = torch.cat([f_ext[free_idx], a[act_idx]])
        n_a = act_idx.numel()
        if use_dense:
            sys = torch.zeros((n_free + n_a, n_free + n_a), dtype=DTYPE)
            sys[:n_free, :n_free] = Kff
            sys[:n_free, n_free:] = B.T
            sys[n_free:, :n_free] = B
            sol = torch.linalg.solve(sys, rhs)
            return sol[:n_free], sol[n_free:]
        from scipy.sparse import bmat, csc_matrix
        from scipy.sparse.linalg import splu
        Bsp = csc_matrix(B.numpy())
        lu = splu(bmat([[Kff_sp, Bsp.T], [Bsp, None]], format="csc"))
        sol = torch.tensor(lu.solve(rhs.numpy()))
        return sol[:n_free], sol[n_free:]

    def integrated_gap(u: torch.Tensor) -> torch.Tensor:
        g = a - D @ u[slave_y_dofs]
        if master_y_dofs is not None:
            g = g + M @ u[master_y_dofs]
        return g

    active = a < 0.0 if bool((a < 0.0).any()) else (a <= 0.0)
    u = torch.zeros(ndof, dtype=DTYPE)
    lam = torch.zeros(n_s, dtype=DTYPE)
    g = a.clone()
    iterations = 0
    converged = False
    lam_tolerance = 0.0

    for it in range(max_iterations):
        iterations = it + 1
        act_idx = torch.nonzero(active).flatten()
        u_free, lam_act = solve_system(act_idx)
        u = torch.zeros(ndof, dtype=DTYPE)
        u[free_idx] = u_free
        g = integrated_gap(u)
        lam = torch.zeros(n_s, dtype=DTYPE)
        lam[act_idx] = lam_act

        # Scaled primal-dual active set (Hintermueller-Ito-Kunisch):
        #     A = { lambda - c g > 0 }.
        # Its fixed point provably satisfies lambda >= 0, g >= 0, lambda g = 0:
        # on A the enforced constraint g = 0 leaves lambda > 0, off A lambda = 0
        # leaves g >= 0.  The naive map {lambda > 0} U {g < 0} does not have
        # that property and can stall with lambda < 0 and g < 0 together.
        lam_max = float(lam.abs().max()) if n_s else 0.0
        lam_tolerance = 1e-9 * max(lam_max, 1e-30)

        new_active = (lam - active_scale * g) > 0.0
        open_gap = float(g[~new_active].min()) if bool((~new_active).any()) else 0.0
        lam_min = float(lam.min()) if n_s else 0.0
        if torch.equal(new_active, active):
            converged = (open_gap >= -gap_tolerance) and (lam_min >= -lam_tolerance)
            break
        active = new_active

    u_s = u[slave_y_dofs]
    u_m = None if master_y_dofs is None else u[master_y_dofs]
    g = integrated_gap(u)

    mid_x = 0.5 * (iface.slave_x[:-1] + iface.slave_x[1:])
    quad_x = torch.cat([iface.slave_x, mid_x]).sort().values
    quad_p = _interp(iface.slave_x, lam, quad_x).clamp(min=0.0)

    g0 = torch.zeros_like(quad_x)
    if iface.master_gap0 is not None:
        g0 = g0 + _interp(iface.master_x, iface.master_gap0, quad_x)
    if iface.slave_gap0 is not None:
        g0 = g0 - _interp(iface.slave_x, iface.slave_gap0, quad_x)
    quad_g = g0 - _interp(iface.slave_x, u_s, quad_x)
    if u_m is not None:
        quad_g = quad_g + _interp(iface.master_x, u_m, quad_x)

    force = float(torch.dot(L, lam))
    max_pen = float(max(0.0, -float(quad_g.min())))

    Ku = torch.tensor(K.tocsr() @ u.numpy()) if is_sparse_K else \
        torch.as_tensor(K, dtype=DTYPE) @ u
    contact = torch.zeros(ndof, dtype=DTYPE)
    contact.index_put_((slave_y_dofs,), D.T @ lam, accumulate=True)
    if master_y_dofs is not None:
        contact.index_put_((master_y_dofs,), -(M.T @ lam), accumulate=True)
    reaction = (Ku - f_ext) + contact
    support = float(reaction[fixed].sum())
    balance_err = abs(support - force) / max(force, 1e-300)

    comp = float(torch.max(torch.abs(lam * g))) if n_s else 0.0
    return MortarContact2DResult(
        displacement=u, lam=lam, contact_pressure_nodes=lam, gap_nodes=g,
        quad_x=quad_x, quad_pressure=quad_p, quad_gap=quad_g, active=active,
        contact_force=force, max_penetration=max_pen, complementarity_residual=comp,
        reaction_balance_error=balance_err, converged=converged, iterations=iterations,
        slave_uy=u_s, master_uy=u_m,
    )

# ============================================================================
# Benchmark A: rigid cylinder indenting an elastic block (Hertz line contact)
# ============================================================================

def build_rigid_cylinder_block(
    *,
    nx: int = 60,
    ny: int = 20,
    half_width: float = 4.0,
    height: float = 6.0,
    radius: float = 10.0,
    young: float = 1000.0,
    poisson: float = 0.3,
    indentation: float = 0.03,
    grading_x: float = 1.6,
    grading_y: float = 2.0,
    master_ratio: float = 1.5,
) -> dict:
    """Half-model of an elastic block indented by a rigid circular cylinder.

    Symmetry about ``x = 0`` is exploited, so only ``x >= 0`` is discretised:
    left edge ``u_x = 0`` (symmetry), bottom edge ``u_y = 0``, right edge free
    (traction free).  The cylinder surface
    ``y = R - sqrt(R^2 - x^2) - indentation`` is supplied as a **separate,
    non-matching** master polyline with ``master_ratio * nx`` segments; its
    geometry enters the mortar constraint only through the transfer matrix
    ``M``, exactly as for a deformable master.
    """
    nodes, elements = structured_q4_mesh(
        nx, ny, half_width, height,
        grading_x=grading_x, grading_y=grading_y,
        grade_from_x="start", grade_from_y="end",
    )
    slave = top_edge_nodes(nx, ny)
    slave_x = nodes[slave, 0]

    n_master = int(round(master_ratio * nx)) + 1
    master_x = torch.linspace(0.0, float(half_width), n_master, dtype=DTYPE)
    gap_master = radius - torch.sqrt(torch.clamp(radius**2 - master_x**2, min=0.0)) - indentation

    iface = build_mortar_interface(
        slave_x, master_x,
        slave_gap0=torch.zeros_like(slave_x),
        master_gap0=gap_master,
    )

    K = q4_plane_strain_stiffness_csr(nodes, elements, young, poisson)

    fixed = []
    for j in range(ny + 1):                      # symmetry: u_x = 0 on x = 0
        fixed.append(2 * (j * (nx + 1)))
    fixed.extend([2 * n + 1 for n in bottom_edge_nodes(nx, ny).tolist()])  # u_y = 0

    return {
        "nodes": nodes,
        "elements": elements,
        "stiffness": K,
        "force": torch.zeros(2 * len(nodes), dtype=DTYPE),
        "fixed_dofs": torch.tensor(sorted(set(fixed)), dtype=torch.long),
        "slave_nodes": slave,
        "slave_x": slave_x,
        "master_x": master_x,
        "interface": iface,
        "young": young,
        "poisson": poisson,
        "radius": radius,
        "indentation": indentation,
        "half_width": half_width,
        "height": height,
        "nx": nx,
        "ny": ny,
    }


def solve_hertz_indentation(problem: dict, **solver_kwargs) -> MortarContact2DResult:
    """Solve the rigid-cylinder indentation problem built above."""
    p = problem
    return solve_mortar_contact_2d(
        p["stiffness"], p["force"], p["fixed_dofs"], p["interface"],
        2 * p["slave_nodes"] + 1, young=p["young"], **solver_kwargs,
    )


def hertz_metrics(problem: dict, result: MortarContact2DResult,
                  *, half_model_factor: float = 2.0) -> dict:
    """Compare an FE indentation result with the Hertz line-contact solution.

    The line load ``F`` is measured from the FE reaction (``half_model_factor``
    converts the half-model integral to the full 2-D line load), and the
    analytic ``b(F)`` / ``p0(F)`` are then compared with the FE contact
    half-width and peak pressure.
    """
    slave_x = problem["slave_x"]
    F = half_model_factor * result.contact_force
    ref = hertz_line_reference(F, problem["radius"], problem["young"], problem["poisson"])

    b_fe = pressure_half_width(slave_x, result.lam)
    p0_fe = float(result.lam.max()) if result.lam.numel() else 0.0

    # pressure profile error inside the analytic contact zone
    x = result.quad_x
    mask = x <= ref.contact_half_width
    p_fe = result.quad_pressure[mask]
    p_ref = hertz_pressure_profile(x[mask], ref.contact_half_width, ref.peak_pressure)
    l2 = float(torch.linalg.vector_norm(p_fe - p_ref) / torch.linalg.vector_norm(p_ref))

    # characteristic edge spacing, for the observed order of convergence
    act = torch.nonzero(result.lam > 0.0).flatten()
    if act.numel() >= 2:
        h_edge = float((slave_x[act[-1]] - slave_x[act[-2]]))
        h_zone = float(torch.median(slave_x[act].diff())) if act.numel() >= 3 else h_edge
    else:
        h_edge = h_zone = float("nan")

    return {
        "line_load": F,
        "b_fe": b_fe,
        "b_ref": ref.contact_half_width,
        "b_error_rel": (b_fe - ref.contact_half_width) / ref.contact_half_width,
        "p0_fe": p0_fe,
        "p0_ref": ref.peak_pressure,
        "p0_error_rel": (p0_fe - ref.peak_pressure) / ref.peak_pressure,
        "pressure_l2_rel": l2,
        "n_active": int(act.numel()),
        "h_edge": h_edge,
        "h_zone": h_zone,
        "max_penetration": result.max_penetration,
        "complementarity_residual": result.complementarity_residual,
        "reaction_balance_error": result.reaction_balance_error,
        "reference_consistency": ref.consistency_residual(),
        "converged": result.converged,
        "iterations": result.iterations,
    }


def observed_order(h: list[float], err: list[float]) -> float:
    """Least-squares slope of ``log(err)`` versus ``log(h)`` (observed order)."""
    hs = [x for x, e in zip(h, err) if x > 0.0 and e > 0.0]
    es = [e for x, e in zip(h, err) if x > 0.0 and e > 0.0]
    if len(hs) < 2:
        return float("nan")
    lx = [math.log(x) for x in hs]
    ly = [math.log(e) for e in es]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den > 0.0 else float("nan")


def hertz_line_contact_study(
    levels: tuple[tuple[int, int], ...] = ((40, 12), (80, 16), (160, 24), (320, 32)),
    **build_kwargs,
) -> dict:
    """Mesh-refinement study of the rigid-cylinder Hertz line contact."""
    rows = []
    for (nx, ny) in levels:
        problem = build_rigid_cylinder_block(nx=nx, ny=ny, **build_kwargs)
        result = solve_hertz_indentation(problem)
        metrics = hertz_metrics(problem, result)
        metrics.update({"nx": nx, "ny": ny,
                        "n_dof": int(2 * len(problem["nodes"])),
                        "n_slave": int(len(problem["slave_x"])),
                        "n_master": int(len(problem["master_x"]))})
        rows.append(metrics)
    return {
        "levels": rows,
        "order_b": observed_order([r["h_zone"] for r in rows],
                                 [abs(r["b_error_rel"]) for r in rows]),
        "order_p0": observed_order([r["h_zone"] for r in rows],
                                  [abs(r["p0_error_rel"]) for r in rows]),
        "order_l2": observed_order([r["h_zone"] for r in rows],
                                  [r["pressure_l2_rel"] for r in rows]),
        "max_abs_b_error": max(abs(r["b_error_rel"]) for r in rows),
        "max_abs_p0_error": max(abs(r["p0_error_rel"]) for r in rows),
    }


# ============================================================================
# Benchmark B: two deformable blocks over a non-matching interface
# ============================================================================

def build_two_block_patch(
    *,
    nx_lower: int = 16,
    nx_upper: int = 11,
    ny_lower: int = 6,
    ny_upper: int = 5,
    width: float = 1.0,
    height_lower: float = 1.0,
    height_upper: float = 1.0,
    young: float = 1000.0,
    poisson: float = 0.3,
    traction: float = 1.0,
) -> dict:
    """Two stacked blocks with a **non-matching** contact interface.

    The lower block ``[0,W] x [0,H1]`` supplies the slave interface (its top
    edge, ``nx_lower + 1`` nodes); the upper block ``[0,W] x [H1,H1+H2]``
    supplies the master interface (its bottom edge, ``nx_upper + 1`` nodes).
    Both are deformable, so the mortar coupling ``M`` acts on real master DOFs.

    Applied load: uniform downward traction ``traction`` on the top surface.
    Supports: ``u_y = 0`` on the lower block bottom edge and ``u_x = 0`` on the
    ``x = 0`` edge of both blocks -- both compatible with the exact solution.

    Exact solution (plane strain, traction-free sides, uniform interface
    pressure): ``sigma_yy = -traction``, ``sigma_xx = 0``, hence
    ``eps_yy = -traction (1 - nu^2) / E`` and a uniform interface pressure
    equal to ``traction`` on every slave node.  Because the mortar operators
    reproduce constant fields exactly on non-matching meshes, the FE solution
    must hit this state to machine precision.
    """
    lo_nodes, lo_elems = structured_q4_mesh(nx_lower, ny_lower, width, height_lower)
    up_nodes, up_elems = structured_q4_mesh(
        nx_upper, ny_upper, width, height_upper, y_offset=height_lower)
    n_lo = len(lo_nodes)
    nodes = torch.cat([lo_nodes, up_nodes], dim=0)
    elements = torch.cat([lo_elems, up_elems + n_lo], dim=0)

    slave = top_edge_nodes(nx_lower, ny_lower)           # lower block top edge
    master = bottom_edge_nodes(nx_upper, ny_upper) + n_lo  # upper block bottom edge

    iface = build_mortar_interface(
        nodes[slave, 0], nodes[master, 0],
        slave_gap0=torch.zeros(len(slave), dtype=DTYPE),
        master_gap0=torch.zeros(len(master), dtype=DTYPE),
    )

    K = q4_plane_strain_stiffness_csr(nodes, elements, young, poisson)

    f_ext = torch.zeros(2 * len(nodes), dtype=DTYPE)
    top = top_edge_nodes(nx_upper, ny_upper) + n_lo
    x_top = nodes[top, 0]
    trib = torch.zeros(len(top), dtype=DTYPE)
    trib[:-1] += 0.5 * (x_top[1:] - x_top[:-1])
    trib[1:] += 0.5 * (x_top[1:] - x_top[:-1])
    f_ext[2 * top + 1] = -traction * trib

    fixed = [int(2 * n + 1) for n in bottom_edge_nodes(nx_lower, ny_lower)]
    for n in list(range(0, n_lo, nx_lower + 1)) + list(range(n_lo, len(nodes), nx_upper + 1)):
        fixed.append(2 * int(n))
    fixed = torch.tensor(sorted(set(fixed)), dtype=torch.long)

    return {
        "nodes": nodes,
        "elements": elements,
        "stiffness": K,
        "force": f_ext,
        "fixed_dofs": fixed,
        "slave_nodes": slave,
        "slave_x": nodes[slave, 0],
        "master_nodes": master,
        "master_x": nodes[master, 0],
        "interface": iface,
        "young": young,
        "poisson": poisson,
        "traction": traction,
        "width": width,
        "height_lower": height_lower,
        "height_upper": height_upper,
        "nx_lower": nx_lower,
        "nx_upper": nx_upper,
        "ny_lower": ny_lower,
        "ny_upper": ny_upper,
        "top_nodes": top,
    }


def solve_two_block_patch(problem: dict, **solver_kwargs) -> MortarContact2DResult:
    """Solve the non-matching two-block patch problem."""
    p = problem
    return solve_mortar_contact_2d(
        p["stiffness"], p["force"], p["fixed_dofs"], p["interface"],
        2 * p["slave_nodes"] + 1,
        master_y_dofs=2 * p["master_nodes"] + 1, young=p["young"], **solver_kwargs,
    )


def patch_metrics(problem: dict, result: MortarContact2DResult) -> dict:
    """Exact-solution comparison for the non-matching two-block patch test."""
    t = problem["traction"]
    E, nu = problem["young"], problem["poisson"]
    lam = result.lam
    p_max_dev = float((lam - t).abs().max()) / t

    uy_top = result.displacement[2 * problem["top_nodes"] + 1]
    exact_top = -t * (1.0 - nu**2) * (problem["height_lower"] + problem["height_upper"]) / E
    top_spread = float((uy_top - uy_top.mean()).abs().max())
    total_force = result.contact_force
    exact_force = t * problem["width"]

    return {
        "pressure_max_rel_dev": p_max_dev,
        "pressure_mean": float(lam.mean()),
        "exact_pressure": t,
        "contact_force": total_force,
        "exact_contact_force": exact_force,
        "force_rel_error": (total_force - exact_force) / exact_force,
        "top_uy_mean": float(uy_top.mean()),
        "top_uy_exact": exact_top,
        "top_uy_rel_error": (float(uy_top.mean()) - exact_top) / abs(exact_top),
        "top_uy_spread": top_spread,
        "partition_of_unity_error": problem["interface"].partition_of_unity_error(),
        "interface_force_conservation_error": problem["interface"].force_conservation_error(),
        "max_penetration": result.max_penetration,
        "complementarity_residual": result.complementarity_residual,
        "reaction_balance_error": result.reaction_balance_error,
        "n_slave": int(len(problem["slave_x"])),
        "n_master": int(len(problem["master_x"])),
        "converged": result.converged,
        "iterations": result.iterations,
    }


def nonmatching_patch_study(
    ratios: tuple[tuple[int, int], ...] = ((8, 5), (16, 11), (24, 7), (32, 21)),
    **build_kwargs,
) -> dict:
    """Exactness of constant-pressure transfer across non-matching interfaces."""
    rows = []
    for (nx_lo, nx_up) in ratios:
        problem = build_two_block_patch(nx_lower=nx_lo, nx_upper=nx_up, **build_kwargs)
        result = solve_two_block_patch(problem)
        m = patch_metrics(problem, result)
        m.update({"nx_lower": nx_lo, "nx_upper": nx_up})
        rows.append(m)
    return {
        "levels": rows,
        "max_pressure_dev": max(r["pressure_max_rel_dev"] for r in rows),
        "max_force_error": max(abs(r["force_rel_error"]) for r in rows),
        "max_top_uy_error": max(abs(r["top_uy_rel_error"]) for r in rows),
        "all_converged": all(r["converged"] for r in rows),
    }


# ============================================================================
# Backward-compatible aliases for the previous (broken) public surface
# ============================================================================

MortarLM2DResult = MortarContact2DResult


def build_hertz_cylinder_on_flat(n_contact_slave: int, n_contact_master: int, **kwargs) -> dict:
    """Legacy alias of :func:`build_rigid_cylinder_block`.

    The old entry point accepted node counts ``n_contact_slave`` /
    ``n_contact_master``; those map onto ``nx`` and ``master_ratio`` here.  The
    old build produced an all-zero load vector and an internally inconsistent
    Hertz reference (``b = sqrt(R delta)``), so this alias keeps the name but
    returns the corrected problem description.
    """
    nx = max(4, int(n_contact_slave))
    ratio = max(1.0, float(n_contact_master) / float(nx))
    kw = {k: v for k, v in kwargs.items() if k in {
        "ny", "half_width", "height", "radius", "young", "poisson", "indentation",
        "grading_x", "grading_y"}}
    return build_rigid_cylinder_block(nx=nx, master_ratio=ratio, **kw)
