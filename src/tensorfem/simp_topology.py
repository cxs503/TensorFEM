"""SIMP (Solid Isotropic Material with Penalization) topology optimisation.

2-D linear-elastic minimum-compliance formulation with density filtering,
OC (Optimality Criteria) update, and mesh-independence verification tools.

Key equations
-------------
SIMP interpolation:
    E_e = E_min + rho_e^p * (E_0 - E_min)          (p = 3)

Element compliance:
    c_e = u_e^T k_e(rho_e) u_e

Total compliance:
    C = sum_e c_e

Adjoint sensitivity (self-adjoint for compliance):
    dc/drho_e = -p * rho_e^(p-1) * (E_0 - E_min) * u_e^T k_e0 u_e

Cone (linear) density filter and its exact adjoint:
    rho_tilde_i = sum_j H_ij rho_j / Hsum_i ,  H_ij = max(0, r_min - dist(i,j))
    dC/drho_j   = sum_i H_ij (dC/drho_tilde_i) / Hsum_i        (transpose, NOT the
                  Sigmund sensitivity-filter heuristic -- see filter_sensitivity)

Helmholtz (PDE) density filter (:class:`PDEHelmholtzFilter`), equivalent cone
radius r = r_min / (2 sqrt 3) (Lazarov & Sigmund 2011):
    (r^2 K + M_lumped) phi = b(rho),  b_i = sum_e v_ei rho_e
    rho_tilde_e = sum_i v_ei phi_i / A_e
    dC/drho_j   = sum_i v_ji y_i ,  y = (r^2 K + M_lumped)^{-1} g
    g_i = sum_e (dC/drho_tilde_e / A_e) v_ei
    Configuration: SIMPConfig(filter_type="linear" | "helmholtz" | "none" |
    "sensitivity", solver="dense" | "sparse").

Verification helpers:
    validate_sensitivity_fd   analytic vs central-difference gradients at
                              delta = 1e-4/1e-5/1e-6 (filter adjoint included)
    checkerboard_index        normalised 4-neighbour oscillation indicator
    topology_complexity       perimeter / components / gray fraction
    run_simp_design           cantilever set-up + optimisation + metrics

OC update (volume-constrained):
    rho_e_new = max(0, min(1, rho_e * sqrt(-dc/drho_e / (lambda * V_e))))
    lambda found by bisection to satisfy volume constraint.

1-D analytical benchmark (variable cross-section bar):
    For a bar of length L, fixed at x=0, tip load F,
    with volume constraint integral A(x) dx = V*,
    the optimal compliance is C* = F^2 * L^2 / (E * V*).
    For p=1 (linear), this equals F^2 * L / (E0 * volume_frac).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------

def make_rectangular_mesh(nx: int, ny: int, *,
                          length: float = 1.0, height: float = 0.5,
                          dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured Q4 mesh on [0,length] x [-height/2, height/2]."""
    x = torch.linspace(0, length, nx + 1, dtype=dtype)
    y = torch.linspace(-height / 2, height / 2, ny + 1, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            cells.append((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1))
    return nodes, torch.tensor(cells, dtype=torch.long)


def make_1d_bar_mesh(n_elem: int, *,
                     length: float = 1.0,
                     dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0, length, n_elem + 1, dtype=dtype).reshape(-1, 1)
    elems = torch.stack((torch.arange(n_elem), torch.arange(1, n_elem + 1)), dim=1)
    return x, elems


# ---------------------------------------------------------------------------
# 2-D Q4 stiffness
# ---------------------------------------------------------------------------

def _elasticity_matrix_2d(young: torch.Tensor, poisson: torch.Tensor,
                          plane: str = "stress") -> torch.Tensor:
    z = torch.zeros_like(young)
    o = torch.ones_like(young)
    if plane == "stress":
        base = torch.stack((o, poisson, z, poisson, o, z, z, z, (o - poisson) / 2),
                           dim=-1).reshape(-1, 3, 3)
        return (young / (1 - poisson ** 2))[:, None, None] * base
    c = young / ((1 + poisson) * (1 - 2 * poisson))
    base = torch.stack((1 - poisson, poisson, z, poisson, 1 - poisson, z,
                        z, z, (1 - 2 * poisson) / 2), dim=-1).reshape(-1, 3, 3)
    return c[:, None, None] * base


def _b_matrix_q4(derivatives: torch.Tensor) -> torch.Tensor:
    ne, nn, _ = derivatives.shape
    b = torch.zeros((ne, 3, 2 * nn), dtype=derivatives.dtype, device=derivatives.device)
    b[:, 0, 0::2] = derivatives[:, :, 0]
    b[:, 1, 1::2] = derivatives[:, :, 1]
    b[:, 2, 0::2] = derivatives[:, :, 1]
    b[:, 2, 1::2] = derivatives[:, :, 0]
    return b


def _q4_shape_derivatives(xi: float, eta: float, dtype: torch.dtype,
                          device: torch.device) -> torch.Tensor:
    """Shape function derivatives w.r.t. natural coords for Q4 [4,2]."""
    return torch.tensor([
        [-(1 - eta), -(1 - xi)],
        [(1 - eta), -(1 + xi)],
        [(1 + eta), (1 + xi)],
        [-(1 + eta), (1 - xi)],
    ], dtype=dtype, device=device) / 4.0


def element_stiffness_q4(coords: torch.Tensor, dmat: torch.Tensor,
                         thickness: torch.Tensor) -> torch.Tensor:
    """2x2 Gauss integration for Q4 element stiffness [ne, 8, 8]."""
    g = 1.0 / math.sqrt(3.0)
    points = [(-g, -g), (g, -g), (g, g), (-g, g)]
    ne = coords.shape[0]
    dtype, dev = coords.dtype, coords.device
    ke = torch.zeros((ne, 8, 8), dtype=dtype, device=dev)
    for xi, eta in points:
        nat = _q4_shape_derivatives(xi, eta, dtype, dev).unsqueeze(0).expand(ne, -1, -1)
        jac = torch.einsum("eia,eib->eab", coords, nat)
        det_j = torch.linalg.det(jac)
        inv_j = torch.linalg.inv(jac)
        deriv = torch.einsum("ab,ebc->eac", nat[0], inv_j)
        b = _b_matrix_q4(deriv)
        ke = ke + (b.transpose(1, 2) @ dmat @ b) * (det_j * thickness)[:, None, None]
    return ke


def assemble_global_stiffness(elements: torch.Tensor, ke: torch.Tensor,
                              n_dofs: int) -> torch.Tensor:
    ne, nn = elements.shape
    edofs = torch.stack(tuple(2 * elements + i for i in (0, 1)), dim=2).reshape(ne, 2 * nn).long()
    rows = edofs[:, :, None].expand(-1, -1, 2 * nn).reshape(-1)
    cols = edofs[:, None, :].expand(-1, 2 * nn, -1).reshape(-1)
    K = torch.zeros((n_dofs, n_dofs), dtype=ke.dtype, device=ke.device)
    K = K.index_put((rows, cols), ke.reshape(-1), accumulate=True)
    return K


# ---------------------------------------------------------------------------
# SIMP core
# ---------------------------------------------------------------------------

@dataclass
class SIMPConfig:
    E0: float = 1.0
    E_min: float = 1e-9
    nu: float = 0.3
    p: float = 3.0
    thickness: float = 1.0
    volume_frac: float = 0.5
    r_min: float = 0.03
    max_iter: int = 200
    tol: float = 1e-4
    plane: str = "stress"
    dtype: torch.dtype = torch.float64
    #: density-filter family: ``"linear"`` (classic cone filter),
    #: ``"helmholtz"`` (PDE/Helmholtz filter) or ``"none"`` (negative control).
    filter_type: str = "linear"
    #: linear-system backend: ``"dense"`` (torch.linalg.solve) or ``"sparse"``
    #: (scipy SuperLU; required for meshes beyond a few thousand dofs).
    solver: str = "dense"


def compute_element_stiffness_base(nodes: torch.Tensor, elements: torch.Tensor,
                                   E0: float, nu: float, thickness: float,
                                   plane: str, dtype: torch.dtype) -> torch.Tensor:
    ne = elements.shape[0]
    young = torch.full((ne,), E0, dtype=dtype)
    poisson = torch.full((ne,), nu, dtype=dtype)
    thick = torch.full((ne,), thickness, dtype=dtype)
    dmat = _elasticity_matrix_2d(young, poisson, plane)
    coords = nodes[elements.long()]
    return element_stiffness_q4(coords, dmat, thick)


def simp_interpolate(rho: torch.Tensor, ke0: torch.Tensor,
                     E_min: float, E0: float, p: float) -> torch.Tensor:
    factor = (E_min + rho ** p * (E0 - E_min)) / E0
    return ke0 * factor[:, None, None]


def solve_system(K: torch.Tensor, forces: torch.Tensor,
                 fixed_dofs: torch.Tensor, refine: int = 0) -> torch.Tensor:
    """Dense constrained solve; ``refine`` steps of iterative refinement.

    Iterative refinement is off by default (the optimisation loop does not
    need it) and is switched on by the finite-difference gradient audit, where
    the *difference* of two nearly equal compliances must be resolved well
    below the solver round-off level.
    """
    mask = torch.ones(K.shape[0], dtype=torch.bool, device=K.device)
    mask[fixed_dofs.long()] = False
    free = torch.arange(K.shape[0], device=K.device)[mask]
    f = forces.to(K.dtype)
    rhs = f[free]
    Kff = K[free][:, free]
    u_free = torch.linalg.solve(Kff, rhs)
    for _ in range(int(refine)):
        u_free = u_free + torch.linalg.solve(Kff, rhs - Kff @ u_free)
    u = torch.zeros(K.shape[0], dtype=K.dtype, device=K.device)
    u = u.index_put((free,), u_free)
    return u


def element_dof_indices(elements: torch.Tensor) -> torch.Tensor:
    """[ne, 8] global dof indices (x, y interleaved) of the Q4 elements."""
    ne = elements.shape[0]
    return torch.stack(tuple(2 * elements.long() + i for i in (0, 1)), dim=2).reshape(ne, 8)


def assemble_global_stiffness_sparse(elements: torch.Tensor, ke: torch.Tensor,
                                     n_dofs: int):
    """Sparse (scipy CSR) assembly of the global stiffness matrix.

    Needed for meshes whose dof count makes the dense factorisation in
    :func:`solve_system` prohibitive (topology-optimisation meshes beyond a
    few thousand dofs).  scipy stays an optional dependency: the feature
    raises a clear ``RuntimeError`` when it is missing.
    """
    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - exercised only w/o scipy
        raise RuntimeError(
            "assemble_global_stiffness_sparse requires scipy; install it or "
            "use solver='dense'"
        ) from exc
    edofs = element_dof_indices(elements).cpu().numpy()
    ne = edofs.shape[0]
    rows = np.repeat(edofs, 8, axis=1).ravel()
    cols = np.tile(edofs, (1, 8)).ravel()
    vals = ke.detach().cpu().numpy().reshape(ne, 64).ravel()
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()


def solve_system_sparse(K, forces: torch.Tensor, fixed_dofs: torch.Tensor,
                        refine: int = 0) -> torch.Tensor:
    """Solve the constrained system with a sparse direct factorisation.

    ``refine`` > 0 adds iterative refinement against the same LU factors (used
    by the finite-difference gradient audit, see :func:`solve_system`).
    """
    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "solve_system_sparse requires scipy; install it or use solver='dense'"
        ) from exc
    n = K.shape[0]
    free = np.ones(n, dtype=bool)
    free[fixed_dofs.cpu().numpy().astype(np.int64)] = False
    Kff = sp.csc_matrix(K)[free][:, free]
    rhs = forces.detach().cpu().numpy()[free]
    u_free = spsolve(Kff, rhs)
    if refine:
        from scipy.sparse.linalg import splu
        lu = splu(Kff)
        u_free = lu.solve(rhs)
        for _ in range(int(refine)):
            u_free = u_free + lu.solve(rhs - Kff @ u_free)
    u = np.zeros(n, dtype=float)
    u[free] = u_free
    return torch.as_tensor(u, dtype=forces.dtype).reshape(-1)


def compute_compliance_and_sensitivity(
    rho: torch.Tensor, ke0: torch.Tensor, u: torch.Tensor,
    elements: torch.Tensor, E_min: float, E0: float, p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    factor = (E_min + rho ** p * (E0 - E_min)) / E0
    edofs = element_dof_indices(elements)
    u_e = u[edofs]
    ke_e = ke0 * factor[:, None, None]
    c_e = torch.einsum("ei,eij,ej->e", u_e, ke_e, u_e)
    compliance = c_e.sum()
    u_ke0_u = torch.einsum("ei,eij,ej->e", u_e, ke0, u_e)
    dc_drho = -p * rho ** (p - 1) * (E0 - E_min) / E0 * u_ke0_u
    return compliance, dc_drho


# ---------------------------------------------------------------------------
# Density filters (classic cone filter and Helmholtz/PDE filter)
# ---------------------------------------------------------------------------

def build_filter_matrix(nodes: torch.Tensor, elements: torch.Tensor,
                        r_min: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Classic cone (linear) density filter matrix.

    ``r_min <= 0`` disables the filter and returns the identity operator
    (``H = I``, ``Hsum = 1``); a zero radius must not produce ``0/0``.
    """
    ne = elements.shape[0]
    if r_min <= 0.0:
        return torch.eye(ne, dtype=dtype), torch.ones(ne, dtype=dtype)
    centroids = nodes[elements.long()].mean(dim=1)
    dist = torch.cdist(centroids, centroids)
    H = torch.clamp(r_min - dist, min=0.0).to(dtype)
    Hsum = H.sum(dim=1)
    return H, Hsum


def apply_density_filter(rho: torch.Tensor, H: torch.Tensor,
                         Hsum: torch.Tensor) -> torch.Tensor:
    return (H @ rho) / Hsum


def filter_sensitivity(dc_drho: torch.Tensor, rho: torch.Tensor | None,
                       H: torch.Tensor, Hsum: torch.Tensor) -> torch.Tensor:
    """Exact adjoint of the cone density filter.

    With ``rho_tilde_i = sum_j H_ij rho_j / Hsum_i`` the chain rule is

    .. math:: dC/drho_j = \sum_i H_{ij}\, (dC/drho_tilde_i) / Hsum_i,

    i.e. the *transposed* filter matrix applied to the sensitivity divided by
    the row sums.  ``rho`` is accepted for signature compatibility and is not
    used: the density-filter adjoint carries no explicit ``rho`` factor.

    .. warning::
       Do **not** replace this by the Sigmund (1997) sensitivity filter
       ``sum_i H_ij rho_i (dC/drho_tilde_i) / (rho_j Hsum_j)``.  That
       expression is a heuristic for filtering the *sensitivity* field; used
       as the gradient of a density-filtered objective it is wrong by O(50%)
       (see :func:`sensitivity_filter_heuristic` and
       ``docs/P218-N04-simp-filter-consistency.md``).
    """
    return H.transpose(0, 1) @ (dc_drho / Hsum)


def sensitivity_filter_heuristic(dc_drho: torch.Tensor, rho: torch.Tensor,
                                 H: torch.Tensor, Hsum: torch.Tensor) -> torch.Tensor:
    """Sigmund (1997) sensitivity filter — heuristic, *not* a filter adjoint.

    Kept for reference/regression only.  It is the historic remedy for
    checkerboard patterns when no density filter is used; it is not the
    gradient of any filtered density field.
    """
    denom = rho * Hsum
    num = H @ (rho * dc_drho)
    safe = torch.where(denom.abs() > 0.0, denom, torch.ones_like(denom))
    return torch.where(denom.abs() > 0.0, num / safe, torch.zeros_like(num))


class IdentityFilter:
    """No filtering (negative control)."""

    kind = "none"
    H = None
    Hsum = None

    def apply(self, rho: torch.Tensor) -> torch.Tensor:
        return rho

    def sensitivity(self, dc_drho_tilde: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        return dc_drho_tilde


class LinearFilter:
    """Classic cone/linear density filter (Bruns & Tortorelli 2001)."""

    kind = "linear"

    def __init__(self, H: torch.Tensor, Hsum: torch.Tensor) -> None:
        self.H = H
        self.Hsum = Hsum

    def apply(self, rho: torch.Tensor) -> torch.Tensor:
        return apply_density_filter(rho, self.H, self.Hsum)

    def sensitivity(self, dc_drho_tilde: torch.Tensor,
                    rho: torch.Tensor | None = None) -> torch.Tensor:
        return filter_sensitivity(dc_drho_tilde, rho, self.H, self.Hsum)


class SensitivityFilter(LinearFilter):
    """Legacy Sigmund sensitivity filter (heuristic, not an adjoint)."""

    kind = "sensitivity"

    def sensitivity(self, dc_drho_tilde: torch.Tensor,
                    rho: torch.Tensor | None = None) -> torch.Tensor:
        return sensitivity_filter_heuristic(dc_drho_tilde, rho, self.H, self.Hsum)


def _q4_scalar_matrices(coords: torch.Tensor):
    """Consistent mass and Laplacian matrices of a Q4 element plus areas.

    Parameters
    ----------
    coords : [ne, 4, 2] element corner coordinates.

    Returns
    -------
    mass : [ne, 4, 4]  ``int N_i N_j dA``
    lap  : [ne, 4, 4]  ``int grad N_i . grad N_j dA``
    area : [ne]
    nodal : [ne, 4]    ``int N_i dA`` (used to project element -> nodal fields)
    """
    g = 1.0 / math.sqrt(3.0)
    points = [(-g, -g), (g, -g), (g, g), (-g, g)]
    ne = coords.shape[0]
    dtype, dev = coords.dtype, coords.device
    mass = torch.zeros((ne, 4, 4), dtype=dtype, device=dev)
    lap = torch.zeros((ne, 4, 4), dtype=dtype, device=dev)
    area = torch.zeros(ne, dtype=dtype, device=dev)
    nodal = torch.zeros((ne, 4), dtype=dtype, device=dev)
    for xi, eta in points:
        nat = _q4_shape_derivatives(xi, eta, dtype, dev).unsqueeze(0).expand(ne, -1, -1)
        jac = torch.einsum("eia,eib->eab", coords, nat)
        det_j = torch.linalg.det(jac)
        inv = torch.linalg.inv(jac)
        # dN/dx = inv^T dN/dxi  -> [ne, 4, 2]
        grad = torch.einsum("eab,eib->eia", inv.transpose(1, 2), nat)
        n = torch.tensor([0.25 * (1 + si * xi) * (1 + sj * eta)
                          for si, sj in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
                         dtype=dtype, device=dev)
        mass += det_j[:, None, None] * torch.einsum("i,j->ij", n, n)[None]
        lap += det_j[:, None, None] * torch.einsum("eia,eja->eij", grad, grad)
        area = area + det_j
        nodal = nodal + det_j[:, None] * n[None, :]
    return mass, lap, area, nodal


def helmholtz_equiv_radius(r_min: float) -> float:
    """PDE-filter length ``r`` matching a cone-filter radius ``R``.

    Both kernels have the same second moment when ``r = R / (2 sqrt(3))``
    (Lazarov & Sigmund, 2011, *Int. J. Numer. Methods Eng.* 86:765-781).
    """
    return float(r_min) / (2.0 * math.sqrt(3.0))


def _element_scalar_matrix_sparse(elements: torch.Tensor, me: torch.Tensor, n: int):
    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the Helmholtz (PDE) density filter requires scipy; install scipy "
            "or use filter_type='linear'"
        ) from exc
    conn = elements.cpu().numpy()
    ne = conn.shape[0]
    rows = np.repeat(conn, 4, axis=1).ravel()
    cols = np.tile(conn, (1, 4)).ravel()
    vals = me.detach().cpu().numpy().reshape(ne, 16).ravel()
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


class PDEHelmholtzFilter:
    """Helmholtz PDE density filter with an exact adjoint.

    The implicit filter is ``rho_tilde - r^2 lap(rho_tilde) = rho`` (natural
    Neumann conditions).  It is discretised on the same Q4 mesh:

    .. math::

        (M + r^2 K)\, \varphi = b(\rho), \qquad
        \tilde\rho_e = \frac{1}{A_e}\sum_i v_{ei}\varphi_i,
        \qquad b_i = \sum_e v_{ei}\rho_e,

    with ``M_lumped`` the lumped (row-sum) mass matrix, ``K`` the Laplacian stiffness
    matrix, ``A_e`` the element area and ``v_{ei} = int_e N_i dA``.  ``A = r^2
    K + M_lumped`` is SPD, so a single sparse factorisation is reused both for
    the forward filter and for its adjoint.

    Notes
    -----
    The adjoint is *not* the transposed filter matrix: the element/nodal
    projection pair ``(v/A, v)`` is applied in reverse order, which is what the
    chain-rule sensitivity below implements.  Finite-difference checking of the
    full compliance sensitivity is the verification that it is correct.
    """

    kind = "helmholtz"

    def __init__(self, nodes: torch.Tensor, elements: torch.Tensor,
                 r_min: float, dtype: torch.dtype = torch.float64) -> None:
        try:
            import scipy.sparse as _sp
            from scipy.sparse.linalg import splu
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the Helmholtz (PDE) density filter requires scipy; install "
                "scipy or use filter_type='linear'"
            ) from exc
        self.r_min = float(r_min)
        self.r = helmholtz_equiv_radius(r_min)
        self.dtype = dtype
        coords = nodes[elements.long()].to(dtype)
        _mass, lap, area, nodal = _q4_scalar_matrices(coords)
        n_nodes = int(elements.max().item()) + 1
        K = _element_scalar_matrix_sparse(elements, lap, n_nodes)
        # lumped (row-sum) mass: the standard choice for the PDE filter, it
        # keeps ``A = r^2 K + diag(m)`` SPD, cheap and oscillation free.
        m_lump = np.zeros(n_nodes, dtype=float)
        np.add.at(m_lump, elements.cpu().numpy().astype(np.int64).ravel(),
                  nodal.cpu().numpy().ravel())
        A = (self.r ** 2 * K + _sp.diags(m_lump)).tocsc()
        self._lu = splu(A)
        self._conn = elements.cpu().numpy().astype(np.int64)
        self._v = nodal.cpu().numpy()
        self._area = area.cpu().numpy()
        self._n_nodes = n_nodes
        #: cone-filter matrix of the same physical radius (for comparison only)
        self.H = None
        self.Hsum = None

    # -- forward ---------------------------------------------------------
    def _project_to_nodes(self, rho: np.ndarray) -> np.ndarray:
        b = np.zeros(self._n_nodes, dtype=float)
        np.add.at(b, self._conn.ravel(), (self._v * rho[:, None]).ravel())
        return b

    def apply(self, rho: torch.Tensor) -> torch.Tensor:
        rho_np = rho.detach().cpu().numpy().astype(float)
        phi = self._lu.solve(self._project_to_nodes(rho_np))
        rho_t = (self._v * phi[self._conn]).sum(axis=1) / self._area
        return torch.as_tensor(rho_t, dtype=self.dtype).reshape(-1)

    # -- adjoint ---------------------------------------------------------
    def sensitivity(self, dc_drho_tilde: torch.Tensor, rho: torch.Tensor | None = None) -> torch.Tensor:
        dc_np = dc_drho_tilde.detach().cpu().numpy().astype(float)
        g = np.zeros(self._n_nodes, dtype=float)
        weighted = (dc_np / self._area)[:, None] * self._v
        np.add.at(g, self._conn.ravel(), weighted.ravel())
        y = self._lu.solve(g)
        out = (self._v * y[self._conn]).sum(axis=1)
        return torch.as_tensor(out, dtype=self.dtype).reshape(-1)


def build_density_filter(nodes: torch.Tensor, elements: torch.Tensor,
                         r_min: float, filter_type: str = "linear",
                         dtype: torch.dtype = torch.float64):
    """Factory returning the filter object used by :func:`run_simp`."""
    if filter_type in ("none", None):
        return IdentityFilter()
    if filter_type in ("linear", "sensitivity"):
        if r_min <= 0.0:
            return IdentityFilter()
        H, Hsum = build_filter_matrix(nodes, elements, r_min, dtype)
        cls = SensitivityFilter if filter_type == "sensitivity" else LinearFilter
        return cls(H, Hsum)
    if filter_type == "helmholtz":
        if r_min <= 0.0:
            raise ValueError("helmholtz filter needs r_min > 0")
        return PDEHelmholtzFilter(nodes, elements, r_min, dtype)
    raise ValueError(f"unknown filter_type {filter_type!r}")


# ---------------------------------------------------------------------------
# OC optimizer
# ---------------------------------------------------------------------------

def oc_update(rho: torch.Tensor, dc_drho: torch.Tensor,
              volumes: torch.Tensor, volume_target: float,
              move_limit: float = 0.2) -> torch.Tensor:
    lo, hi = 0.0, 1e9
    mid = 0.0
    update_factor = torch.sqrt(torch.clamp(-dc_drho / (volumes + 1e-15), min=0.0))

    for _ in range(50):
        mid = (lo + hi) / 2.0
        sqrt_mid = math.sqrt(mid + 1e-15)
        rho_new = torch.clamp(rho * update_factor / sqrt_mid, min=0.0, max=1.0)
        rho_new = torch.clamp(rho_new, min=rho - move_limit, max=rho + move_limit)
        vol = (rho_new * volumes).sum()
        if vol > volume_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break

    sqrt_mid = math.sqrt(mid + 1e-15)
    rho_new = torch.clamp(rho * update_factor / sqrt_mid, min=0.0, max=1.0)
    rho_new = torch.clamp(rho_new, min=rho - move_limit, max=rho + move_limit)
    return rho_new


# ---------------------------------------------------------------------------
# Full optimisation loop
# ---------------------------------------------------------------------------

@dataclass
class SIMPResult:
    rho: torch.Tensor
    compliance_history: list[float]
    volume_history: list[float]
    final_compliance: float
    final_volume: float
    n_iterations: int
    converged: bool
    ke0: torch.Tensor | None = None
    H: torch.Tensor | None = None
    Hsum: torch.Tensor | None = None
    elements: torch.Tensor | None = None
    nodes: torch.Tensor | None = None
    forces: torch.Tensor | None = None
    fixed_dofs: torch.Tensor | None = None
    final_displacement: torch.Tensor | None = None
    filter_kind: str = "linear"
    rho_tilde: torch.Tensor | None = None


def run_simp(nodes: torch.Tensor, elements: torch.Tensor,
             forces: torch.Tensor, fixed_dofs: torch.Tensor,
             config: SIMPConfig | None = None) -> SIMPResult:
    if config is None:
        config = SIMPConfig()

    dtype = config.dtype
    ne = elements.shape[0]
    n_dofs = 2 * nodes.shape[0]

    ke0 = compute_element_stiffness_base(
        nodes, elements, config.E0, config.nu, config.thickness,
        config.plane, dtype
    )

    # Element volumes
    coords = nodes[elements.long()]
    elem_areas = torch.zeros(ne, dtype=dtype)
    for xi, eta in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        nat = _q4_shape_derivatives(xi, eta, dtype, coords.device).unsqueeze(0).expand(ne, -1, -1)
        jac = torch.einsum("eia,eib->eab", coords, nat)
        det_j = torch.linalg.det(jac)
        elem_areas = elem_areas + det_j / 4.0
    volumes = elem_areas * config.thickness
    volume_target = config.volume_frac * volumes.sum()

    filt = build_density_filter(nodes, elements, config.r_min,
                                getattr(config, "filter_type", "linear"), dtype)
    H, Hsum = filt.H, filt.Hsum

    rho = torch.full((ne,), config.volume_frac, dtype=dtype)

    def _solve(ke_local: torch.Tensor) -> torch.Tensor:
        if getattr(config, "solver", "dense") == "sparse":
            K_local = assemble_global_stiffness_sparse(elements, ke_local, n_dofs)
            return solve_system_sparse(K_local, forces, fixed_dofs)
        K_local = assemble_global_stiffness(elements, ke_local, n_dofs)
        return solve_system(K_local, forces, fixed_dofs)

    compliance_history = []
    volume_history = []
    converged = False
    n_iter = 0
    u_final = None

    for iteration in range(config.max_iter):
        rho_tilde = filt.apply(rho)
        ke = simp_interpolate(rho_tilde, ke0, config.E_min, config.E0, config.p)
        u = _solve(ke)

        compliance, dc_drho = compute_compliance_and_sensitivity(
            rho_tilde, ke0, u, elements, config.E_min, config.E0, config.p
        )
        dc_drho_filtered = filt.sensitivity(dc_drho, rho)

        rho_new = oc_update(rho, dc_drho_filtered, volumes, volume_target)

        compliance_history.append(float(compliance))
        current_vol = float((rho_new * volumes).sum() / volumes.sum())
        volume_history.append(current_vol)

        change = torch.max(torch.abs(rho_new - rho))
        n_iter = iteration + 1
        if change < config.tol:
            converged = True
            rho = rho_new
            u_final = u
            break

        rho = rho_new

    if u_final is None:
        rho_tilde = filt.apply(rho)
        ke = simp_interpolate(rho_tilde, ke0, config.E_min, config.E0, config.p)
        final_compliance_checked, _ = compute_compliance_and_sensitivity(
            rho_tilde, ke0, _solve(ke), elements, config.E_min, config.E0, config.p
        )
        u_final = None
        compliance_history.append(float(final_compliance_checked))

    final_compliance = float(compliance_history[-1]) if compliance_history else 0.0
    final_volume = float(volume_history[-1]) if volume_history else 0.0

    return SIMPResult(
        rho=rho,
        compliance_history=compliance_history,
        volume_history=volume_history,
        final_compliance=final_compliance,
        final_volume=final_volume,
        n_iterations=n_iter,
        converged=converged,
        ke0=ke0, H=H, Hsum=Hsum,
        elements=elements, nodes=nodes,
        forces=forces, fixed_dofs=fixed_dofs,
        final_displacement=u_final,
        filter_kind=getattr(filt, "kind", "linear"),
        rho_tilde=filt.apply(rho),
    )


# ---------------------------------------------------------------------------
# 1-D analytical benchmark
# ---------------------------------------------------------------------------

def analytical_1d_compliance(F: float, L: float, E: float, V: float) -> float:
    """C* = F^2 * L^2 / (E * V) for optimal 1-D bar (p=1 linear case)."""
    return F ** 2 * L ** 2 / (E * V)


def run_1d_simp(n_elem: int, F: float = 1.0, L: float = 1.0,
                E0: float = 1.0, volume_frac: float = 0.5,
                max_iter: int = 500, tol: float = 1e-8,
                p: float = 1.0, dtype: torch.dtype = torch.float64) -> dict:
    """1-D bar optimisation with series compliance.

    For elements in series: C = F^2 * sum(Le / E_e)
    With E_e = rho_e^p * E0 and Le = L/n_elem.
    """
    ne = n_elem
    Le = L / n_elem

    rho = torch.full((ne,), volume_frac, dtype=dtype)
    volumes = torch.ones(ne, dtype=dtype) * Le
    volume_target = volume_frac * L

    for iteration in range(max_iter):
        E_e = rho ** p * E0
        # Sensitivity: dc/drho_e = -F^2 * Le * p * rho_e^(p-1) * E0 / E_e^2
        dc_drho = -F ** 2 * Le * p * rho ** (p - 1) * E0 / (E_e ** 2)

        # OC update
        lo, hi = 0.0, 1e9
        mid = 0.0
        update_factor = torch.sqrt(torch.clamp(-dc_drho / (volumes + 1e-15), min=0.0))

        for _ in range(50):
            mid = (lo + hi) / 2.0
            sqrt_mid = math.sqrt(mid + 1e-15)
            rho_new = torch.clamp(rho * update_factor / sqrt_mid, min=0.0, max=1.0)
            rho_new = torch.clamp(rho_new, min=rho - 0.2, max=rho + 0.2)
            vol = (rho_new * volumes).sum()
            if vol > volume_target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break

        sqrt_mid = math.sqrt(mid + 1e-15)
        rho_new = torch.clamp(rho * update_factor / sqrt_mid, min=0.0, max=1.0)
        rho_new = torch.clamp(rho_new, min=rho - 0.2, max=rho + 0.2)

        if torch.max(torch.abs(rho_new - rho)) < tol:
            rho = rho_new
            break
        rho = rho_new

    # Final compliance
    E_e = rho ** p * E0
    final_compliance = float(F ** 2 * (Le / E_e).sum())
    final_volume = float(rho.sum() * Le)

    # Analytical: for p=1, C* = F^2 * L / (E0 * volume_frac)
    analytical = F ** 2 * L / (E0 * volume_frac)

    return {
        "numerical_compliance": final_compliance,
        "analytical_compliance": analytical,
        "relative_error": abs(final_compliance - analytical) / analytical,
        "final_rho": rho,
        "final_volume": final_volume,
        "volume_error": abs(final_volume - volume_target),
    }


# ---------------------------------------------------------------------------
# Checkerboard metric
# ---------------------------------------------------------------------------

def _rho_grid(rho: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    if rho.numel() != nx * ny:
        raise ValueError(
            f"density field has {rho.numel()} entries but nx*ny = {nx * ny}"
        )
    return rho.reshape(ny, nx)


def checkerboard_metric(rho: torch.Tensor, elements: torch.Tensor | None,
                        nx: int, ny: int) -> float:
    """Mean absolute difference between 4-neighbour elements.

    A perfect 0/1 checkerboard gives 1.0, a uniform field 0.0.  ``elements`` is
    accepted for backward compatibility (only ``nx``/``ny`` are used, which
    describe the structured element order produced by
    :func:`make_rectangular_mesh`).
    """
    g = _rho_grid(rho, nx, ny)
    dx = (g[:, 1:] - g[:, :-1]).abs().mean() if nx > 1 else torch.zeros((), dtype=g.dtype)
    dy = (g[1:, :] - g[:-1, :]).abs().mean() if ny > 1 else torch.zeros((), dtype=g.dtype)
    return float(0.5 * (dx + dy))


def checkerboard_index(rho: torch.Tensor, nx: int, ny: int) -> float:
    """Checkerboard indicator: ``mean_e |rho_e - mean(neighbours)|``.

    Exactly 1.0 for an ideal 0/1 checkerboard, 0.0 for any constant field, and
    therefore a normalised measure of the mesh-scale oscillation that the
    density filter is supposed to remove (Sigmund & Petersson 1998;
    Diaz & Sigmund 1995).  The 4-neighbour average is taken over the elements
    that actually exist at the boundary of the grid.
    """
    g = _rho_grid(rho, nx, ny).to(torch.float64)
    nb = torch.zeros_like(g)
    cnt = torch.zeros_like(g)
    if nx > 1:
        nb[:, :-1] += g[:, 1:]
        cnt[:, :-1] += 1
        nb[:, 1:] += g[:, :-1]
        cnt[:, 1:] += 1
    if ny > 1:
        nb[:-1, :] += g[1:, :]
        cnt[:-1, :] += 1
        nb[1:, :] += g[:-1, :]
        cnt[1:, :] += 1
    if float(cnt.min()) == 0.0:  # 1x1 mesh
        return 0.0
    return float((g - nb / cnt).abs().mean())


def topology_complexity(rho: torch.Tensor, nx: int, ny: int,
                        threshold: float = 0.5) -> dict:
    """Perimeter-based topology complexity of a (near-)0/1 density field.

    ``perimeter`` is the total length of the 0/1 interfaces measured in element
    lengths (vertical + horizontal cut counts, domain boundary excluded), so a
    coarser topology has a strictly smaller perimeter.  ``n_components`` counts
    4-connected solid clusters and ``gray_fraction`` the share of elements in
    the transition band ``threshold +/- 0.1``.  All three decrease when the
    filter radius grows, which is what acceptance criterion C4 checks.
    """
    g = (_rho_grid(rho, nx, ny) >= threshold).to(torch.int64).cpu().numpy()
    dx = int(np.abs(np.diff(g, axis=1)).sum())
    dy = int(np.abs(np.diff(g, axis=0)).sum())
    perimeter = dx + dy
    visited = np.zeros_like(g, dtype=bool)
    n_comp = 0
    for j in range(ny):
        for i in range(nx):
            if g[j, i] and not visited[j, i]:
                n_comp += 1
                stack = [(j, i)]
                visited[j, i] = True
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nbb = a + da, b + db
                        if 0 <= na < ny and 0 <= nbb < nx and g[na, nbb] and not visited[na, nbb]:
                            visited[na, nbb] = True
                            stack.append((na, nbb))
    rho_np = rho.detach().cpu().numpy()
    gray = float(np.mean((rho_np > threshold - 0.1) & (rho_np < threshold + 0.1)))
    return {
        "perimeter": float(perimeter),
        "perimeter_per_area": float(perimeter / max(nx * ny, 1)),
        "n_components": int(n_comp),
        "gray_fraction": gray,
        "checkerboard_index": checkerboard_index(rho, nx, ny),
    }


# ---------------------------------------------------------------------------
# Sensitivity finite-difference validation
# ---------------------------------------------------------------------------

#: Iterative-refinement steps used inside the finite-difference audit.  Two
#: steps is the stagnation point of the compliance round-off floor
#: (~2e-10 -> 5e-11; a third step does not improve it), which is what makes the
#: delta = 1e-6 comparison meaningful in double precision.
FD_REFINEMENT = 2


def validate_sensitivity_fd(
    nodes: torch.Tensor, elements: torch.Tensor,
    forces: torch.Tensor, fixed_dofs: torch.Tensor,
    rho: torch.Tensor, config: SIMPConfig,
    n_samples: int = 3,
    delta: float | list[float] | tuple[float, ...] = 1e-6,
    seed: int | None = 0,
) -> list[dict]:
    """Compare the analytic compliance sensitivity against central differences.

    The analytic gradient contains the full chain rule including the density
    filter (:class:`LinearFilter` or :class:`PDEHelmholtzFilter`), so this is a
    test of both the SIMP adjoint and the filter adjoint.

    Parameters
    ----------
    rho : element densities at which the gradient is checked.
    delta : finite-difference step (or a sequence of steps, each reported).
    seed : seed of the element sample; ``None`` uses the global RNG (the
        legacy behaviour).

    Returns
    -------
    list of dicts with ``element``, ``delta``, ``fd_sensitivity``,
    ``analytical_sensitivity``, ``relative_error``, ``abs_error`` and
    ``clamped`` (True when the +/- perturbation left ``[0, 1]``, in which case
    the one-sided field is truncated and the comparison is only indicative).
    """
    ne = rho.shape[0]
    n_dofs = 2 * nodes.shape[0]
    dtype = config.dtype
    ke0 = compute_element_stiffness_base(
        nodes, elements, config.E0, config.nu, config.thickness,
        config.plane, dtype
    )
    filt = build_density_filter(nodes, elements, config.r_min,
                                getattr(config, "filter_type", "linear"), dtype)

    def compliance_of(rho_field: torch.Tensor) -> float:
        """Compliance of a perturbed design, evaluated as ``f . u``.

        ``f . u`` and the element sum ``sum_e u_e^T k_e u_e`` are the same
        number analytically; with ``E_min/E0 = 1e-9`` the element sum spans
        nine decades and loses about one digit, which matters when the central
        difference divides by ``2 delta = 2e-6``.  The load functional plus one
        step of iterative refinement lowers the round-off floor of the audit
        from ~2e-10 to ~5e-11 (see the report).
        """
        rho_t = filt.apply(rho_field)
        ke = simp_interpolate(rho_t, ke0, config.E_min, config.E0, config.p)
        if getattr(config, "solver", "dense") == "sparse":
            K = assemble_global_stiffness_sparse(elements, ke, n_dofs)
            u = solve_system_sparse(K, forces, fixed_dofs, refine=FD_REFINEMENT)
        else:
            K = assemble_global_stiffness(elements, ke, n_dofs)
            u = solve_system(K, forces, fixed_dofs, refine=FD_REFINEMENT)
        return float(forces @ u)

    rho_tilde = filt.apply(rho)
    ke = simp_interpolate(rho_tilde, ke0, config.E_min, config.E0, config.p)
    if getattr(config, "solver", "dense") == "sparse":
        K = assemble_global_stiffness_sparse(elements, ke, n_dofs)
        u = solve_system_sparse(K, forces, fixed_dofs)
    else:
        K = assemble_global_stiffness(elements, ke, n_dofs)
        u = solve_system(K, forces, fixed_dofs)
    _, dc_drho = compute_compliance_and_sensitivity(
        rho_tilde, ke0, u, elements, config.E_min, config.E0, config.p
    )
    analytic = filt.sensitivity(dc_drho, rho)

    if isinstance(delta, (float, int)):
        deltas = [float(delta)]
    else:
        deltas = [float(d) for d in delta]
    rng = random.Random(seed) if seed is not None else random
    indices = rng.sample(range(ne), min(n_samples, ne))

    results = []
    for idx in indices:
        for d in deltas:
            hi = min(1.0, float(rho[idx]) + d)
            lo = max(0.0, float(rho[idx]) - d)
            clamped = (hi - lo) < 2 * d - 1e-15
            rho_plus = rho.clone()
            rho_plus[idx] = hi
            rho_minus = rho.clone()
            rho_minus[idx] = lo
            c_plus = compliance_of(rho_plus)
            c_minus = compliance_of(rho_minus)
            fd = (c_plus - c_minus) / (hi - lo)
            a = float(analytic[idx])
            results.append({
                "element": int(idx),
                "delta": d,
                "fd_sensitivity": fd,
                "analytical_sensitivity": a,
                "relative_error": abs(fd - a) / abs(a) if abs(a) > 1e-15 else abs(fd - a),
                "abs_error": abs(fd - a),
                "clamped": bool(clamped),
            })
    return results


def run_simp_design(
    nx: int, ny: int, *,
    length: float = 2.0, height: float = 1.0,
    load: str = "tip",
    config: SIMPConfig | None = None,
) -> dict:
    """Convenience wrapper: MBB-style cantilever set-up + optimisation.

    Geometry: rectangular domain ``[0, length] x [-height/2, height/2]`` with
    a Q4 mesh, fully clamped left edge and a unit total vertical load applied
    either at the tip mid-height node (``load="tip"``) or uniformly over the
    right edge (``load="edge"``).  Returns the :class:`SIMPResult` together
    with the mesh/BC data and the topology metrics of the design and of the
    filtered field.
    """
    if config is None:
        config = SIMPConfig()
    nodes, elements = make_rectangular_mesh(nx, ny, length=length, height=height,
                                            dtype=config.dtype)
    n_dofs = 2 * nodes.shape[0]
    left = torch.where(nodes[:, 0] < 1e-10)[0]
    fixed_dofs = torch.cat([2 * left, 2 * left + 1]).long()
    forces = torch.zeros(n_dofs, dtype=config.dtype)
    if load == "tip":
        tip = torch.where((nodes[:, 0] > length - 1e-10) &
                          (nodes[:, 1].abs() < 1e-10))[0]
        forces[2 * tip + 1] = -1.0 / max(len(tip), 1)
    elif load == "edge":
        right = torch.where(nodes[:, 0] > length - 1e-10)[0]
        forces[2 * right + 1] = -1.0 / max(len(right), 1)
    else:
        raise ValueError(f"unknown load {load!r}")
    result = run_simp(nodes, elements, forces, fixed_dofs, config)
    rho_tilde = result.rho_tilde if result.rho_tilde is not None else result.rho
    return {
        "result": result,
        "nodes": nodes,
        "elements": elements,
        "forces": forces,
        "fixed_dofs": fixed_dofs,
        "nx": nx,
        "ny": ny,
        "n_dofs": n_dofs,
        "compliance": result.final_compliance,
        "volume_fraction": result.final_volume,
        "design_metrics": topology_complexity(result.rho, nx, ny),
        "filtered_metrics": topology_complexity(rho_tilde, nx, ny),
    }
