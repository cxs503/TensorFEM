"""Small-deformation frictionless contact with rigid planes.

The sign convention is ``gap = offset + C @ u >= 0``.  A positive
multiplier acts in the positive plane-normal direction.  Contact satisfies
``lambda >= 0`` and ``lambda * gap = 0``.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ContactResult:
    displacement: torch.Tensor
    gap: torch.Tensor
    contact_force: torch.Tensor
    active: torch.Tensor
    structural_energy: torch.Tensor
    contact_energy: torch.Tensor


def _check(K: torch.Tensor, f: torch.Tensor, C: torch.Tensor, offset: torch.Tensor) -> None:
    n = K.shape[0]
    if K.ndim != 2 or K.shape != (n, n) or f.shape != (n,):
        raise ValueError("K must be square and f must have one entry per DOF")
    if C.ndim != 2 or C.shape[1] != n or offset.shape != (C.shape[0],):
        raise ValueError("C and offset have incompatible shapes")
    if not (K.dtype == f.dtype == C.dtype == offset.dtype):
        raise ValueError("all inputs must have the same dtype")


def solve_rigid_plane_contact(
    K: torch.Tensor,
    f: torch.Tensor,
    C: torch.Tensor,
    offset: torch.Tensor,
    *,
    method: str = "active_set",
    penalty: float | torch.Tensor = 1.0e6,
    tolerance: float = 1.0e-10,
    max_iterations: int = 50,
) -> ContactResult:
    """Solve ``K u - C.T lambda = f`` against frictionless rigid planes.

    ``active_set`` enforces the unilateral constraint exactly through a KKT
    solve. ``penalty`` uses ``lambda = penalty * max(-gap, 0)`` and therefore
    permits a small, measurable penetration.
    """
    _check(K, f, C, offset)
    if method not in {"active_set", "penalty"}:
        raise ValueError("method must be 'active_set' or 'penalty'")
    u_free = torch.linalg.solve(K, f)
    active = (offset + C @ u_free) < -tolerance
    p = torch.as_tensor(penalty, dtype=K.dtype, device=K.device)
    if method == "penalty" and p <= 0:
        raise ValueError("penalty must be positive")

    lam = torch.zeros(C.shape[0], dtype=K.dtype, device=K.device)
    u = u_free
    for _ in range(max_iterations):
        ids = torch.nonzero(active, as_tuple=False).flatten()
        if ids.numel() == 0:
            u = u_free
            lam = torch.zeros_like(lam)
        else:
            Ca, ga = C[ids], offset[ids]
            if method == "active_set":
                z = torch.zeros((ids.numel(), ids.numel()), dtype=K.dtype, device=K.device)
                A = torch.cat((torch.cat((K, -Ca.T), dim=1),
                               torch.cat((Ca, z), dim=1)), dim=0)
                rhs = torch.cat((f, -ga))
                sol = torch.linalg.solve(A, rhs)
                u = sol[: K.shape[0]]
                lam = torch.zeros_like(lam).index_put((ids,), sol[K.shape[0] :])
            else:
                u = torch.linalg.solve(K + p * Ca.T @ Ca, f - p * Ca.T @ ga)
                gaps = offset + C @ u
                lam = p * torch.clamp(-gaps, min=0)
        gaps = offset + C @ u
        new_active = (gaps < -tolerance) | (lam > tolerance)
        if torch.equal(new_active, active):
            break
        active = new_active
    else:
        raise RuntimeError("contact active set did not converge")

    gaps = offset + C @ u
    contact_energy = (0.5 * torch.sum(lam * torch.clamp(-gaps, min=0))
                      if method == "penalty" else K.new_zeros(()))
    return ContactResult(u, gaps, lam, active, 0.5 * u @ K @ u, contact_energy)


@dataclass(frozen=True)
class HertzSphereReference:
    contact_radius: torch.Tensor
    indentation: torch.Tensor
    maximum_pressure: torch.Tensor
    effective_modulus: torch.Tensor


def hertz_sphere_on_halfspace(
    force: torch.Tensor, radius: torch.Tensor, young_sphere: torch.Tensor,
    poisson_sphere: torch.Tensor, young_halfspace: torch.Tensor,
    poisson_halfspace: torch.Tensor,
) -> HertzSphereReference:
    """Classical elastic Hertz sphere/half-space analytical reference only."""
    inv_e = ((1 - poisson_sphere**2) / young_sphere
             + (1 - poisson_halfspace**2) / young_halfspace)
    effective = 1 / inv_e
    a = (3 * force * radius / (4 * effective)) ** (1 / 3)
    indentation = a**2 / radius
    maximum_pressure = 3 * force / (2 * torch.pi * a**2)
    return HertzSphereReference(a, indentation, maximum_pressure, effective)
