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

Density filter (Helmholtz-type, linear weight):
    rho_tilde_e = sum_j H_ej rho_j / sum_j H_ej
    H_ej = max(0, r_min - dist(e,j))

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
                 fixed_dofs: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(K.shape[0], dtype=torch.bool, device=K.device)
    mask[fixed_dofs.long()] = False
    free = torch.arange(K.shape[0], device=K.device)[mask]
    f = forces.to(K.dtype)
    rhs = f[free]
    u_free = torch.linalg.solve(K[free][:, free], rhs)
    u = torch.zeros(K.shape[0], dtype=K.dtype, device=K.device)
    u = u.index_put((free,), u_free)
    return u


def compute_compliance_and_sensitivity(
    rho: torch.Tensor, ke0: torch.Tensor, u: torch.Tensor,
    elements: torch.Tensor, E_min: float, E0: float, p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ne = rho.shape[0]
    factor = (E_min + rho ** p * (E0 - E_min)) / E0
    edofs = torch.stack(tuple(2 * elements + i for i in (0, 1)), dim=2).reshape(ne, 8).long()
    u_e = u[edofs]
    ke_e = ke0 * factor[:, None, None]
    c_e = torch.einsum("ei,eij,ej->e", u_e, ke_e, u_e)
    compliance = c_e.sum()
    u_ke0_u = torch.einsum("ei,eij,ej->e", u_e, ke0, u_e)
    dc_drho = -p * rho ** (p - 1) * (E0 - E_min) / E0 * u_ke0_u
    return compliance, dc_drho


# ---------------------------------------------------------------------------
# Density filter
# ---------------------------------------------------------------------------

def build_filter_matrix(nodes: torch.Tensor, elements: torch.Tensor,
                        r_min: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    ne = elements.shape[0]
    centroids = nodes[elements.long()].mean(dim=1)
    dist = torch.cdist(centroids, centroids)
    H = torch.clamp(r_min - dist, min=0.0).to(dtype)
    Hsum = H.sum(dim=1)
    return H, Hsum


def apply_density_filter(rho: torch.Tensor, H: torch.Tensor,
                         Hsum: torch.Tensor) -> torch.Tensor:
    return (H @ rho) / Hsum


def filter_sensitivity(dc_drho: torch.Tensor, rho: torch.Tensor,
                       H: torch.Tensor, Hsum: torch.Tensor) -> torch.Tensor:
    return (H @ (rho * dc_drho)) / (rho * Hsum + 1e-15)


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

    H, Hsum = build_filter_matrix(nodes, elements, config.r_min, dtype)

    rho = torch.full((ne,), config.volume_frac, dtype=dtype)

    compliance_history = []
    volume_history = []
    converged = False
    n_iter = 0
    u_final = None

    for iteration in range(config.max_iter):
        rho_tilde = apply_density_filter(rho, H, Hsum)
        ke = simp_interpolate(rho_tilde, ke0, config.E_min, config.E0, config.p)
        K = assemble_global_stiffness(elements, ke, n_dofs)
        u = solve_system(K, forces, fixed_dofs)

        compliance, dc_drho = compute_compliance_and_sensitivity(
            rho_tilde, ke0, u, elements, config.E_min, config.E0, config.p
        )
        dc_drho_filtered = filter_sensitivity(dc_drho, rho, H, Hsum)

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
        rho_tilde = apply_density_filter(rho, H, Hsum)
        ke = simp_interpolate(rho_tilde, ke0, config.E_min, config.E0, config.p)
        K = assemble_global_stiffness(elements, ke, n_dofs)
        u_final = solve_system(K, forces, fixed_dofs)

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
    V_total = volume_frac * L  # total available volume (unit area reference)

    rho = torch.full((ne,), volume_frac, dtype=dtype)
    volumes = torch.ones(ne, dtype=dtype) * Le
    volume_target = volume_frac * L

    for iteration in range(max_iter):
        E_e = rho ** p * E0
        # Series compliance: C = F^2 * sum(Le / E_e)
        compliance = F ** 2 * (Le / E_e).sum()

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

def checkerboard_metric(rho: torch.Tensor, elements: torch.Tensor,
                        nx: int, ny: int) -> float:
    """Quantify checkerboard pattern via adjacent element differences.
    
    For a perfect checkerboard (alternating 0/1), adjacent elements differ by 1,
    giving metric ≈ 1.0. For uniform fields, metric ≈ 0.0.
    """
    rho_grid = rho.reshape(ny, nx)
    metric = 0.0
    count = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            h_diff = abs(rho_grid[j, i] - rho_grid[j, i + 1])
            v_diff = abs(rho_grid[j, i] - rho_grid[j + 1, i])
            metric += float(h_diff + v_diff)
            count += 2
    return metric / max(count, 1)


# ---------------------------------------------------------------------------
# Sensitivity finite-difference validation
# ---------------------------------------------------------------------------

def validate_sensitivity_fd(
    nodes: torch.Tensor, elements: torch.Tensor,
    forces: torch.Tensor, fixed_dofs: torch.Tensor,
    rho: torch.Tensor, config: SIMPConfig,
    n_samples: int = 3, delta: float = 1e-7,
) -> list[dict]:
    ne = rho.shape[0]
    n_dofs = 2 * nodes.shape[0]
    ke0 = compute_element_stiffness_base(
        nodes, elements, config.E0, config.nu, config.thickness,
        config.plane, config.dtype
    )
    H, Hsum = build_filter_matrix(nodes, elements, config.r_min, config.dtype)

    rho_tilde = apply_density_filter(rho, H, Hsum)
    ke = simp_interpolate(rho_tilde, ke0, config.E_min, config.E0, config.p)
    K = assemble_global_stiffness(elements, ke, n_dofs)
    u = solve_system(K, forces, fixed_dofs)
    compliance, dc_drho = compute_compliance_and_sensitivity(
        rho_tilde, ke0, u, elements, config.E_min, config.E0, config.p
    )
    dc_drho_filtered = filter_sensitivity(dc_drho, rho, H, Hsum)

    indices = random.sample(range(ne), min(n_samples, ne))
    results = []

    for idx in indices:
        rho_plus = rho.clone()
        rho_plus[idx] = min(1.0, rho[idx] + delta)
        rho_tilde_plus = apply_density_filter(rho_plus, H, Hsum)
        ke_plus = simp_interpolate(rho_tilde_plus, ke0, config.E_min, config.E0, config.p)
        K_plus = assemble_global_stiffness(elements, ke_plus, n_dofs)
        u_plus = solve_system(K_plus, forces, fixed_dofs)
        c_plus, _ = compute_compliance_and_sensitivity(
            rho_tilde_plus, ke0, u_plus, elements, config.E_min, config.E0, config.p
        )

        rho_minus = rho.clone()
        rho_minus[idx] = max(0.0, rho[idx] - delta)
        rho_tilde_minus = apply_density_filter(rho_minus, H, Hsum)
        ke_minus = simp_interpolate(rho_tilde_minus, ke0, config.E_min, config.E0, config.p)
        K_minus = assemble_global_stiffness(elements, ke_minus, n_dofs)
        u_minus = solve_system(K_minus, forces, fixed_dofs)
        c_minus, _ = compute_compliance_and_sensitivity(
            rho_tilde_minus, ke0, u_minus, elements, config.E_min, config.E0, config.p
        )

        fd_sensitivity = float((c_plus - c_minus) / (2 * delta))
        analytical = float(dc_drho_filtered[idx])

        if abs(analytical) > 1e-15:
            rel_error = abs(fd_sensitivity - analytical) / abs(analytical)
        else:
            rel_error = abs(fd_sensitivity - analytical)

        results.append({
            "element": idx,
            "fd_sensitivity": fd_sensitivity,
            "analytical_sensitivity": analytical,
            "relative_error": rel_error,
        })

    return results
