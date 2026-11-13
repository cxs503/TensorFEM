"""Implicit linear time integration: average-acceleration Newmark and HHT-alpha.

``newmark_linear`` is the classical average-acceleration scheme (beta = 1/4,
gamma = 1/2 by default).  ``hht_alpha`` is the Hilbert-Hughes-Taylor alpha
variant used as the default transient scheme by Abaqus (``*DYNAMIC``), ANSYS
(``TINTP``) and Code_Aster (``SCHEMA_THETA``): it keeps second-order accuracy
while adding *controllable* high-frequency numerical dissipation, so spurious
high-frequency content produced by spatial discretisation is damped instead of
polluting the resolved response.  ``hht_alpha(..., alpha=0.0)`` reduces exactly
to ``newmark_linear``.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TransientResult:
    time: torch.Tensor
    displacement: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor


def _check_transient_inputs(M, K, force, time, u0, v0, C):
    """Shared argument validation for the implicit linear integrators."""
    if M.ndim != 2 or M.shape != K.shape or M.shape[0] != M.shape[1]:
        raise ValueError("M and K must be equal square matrices")
    if force.shape != (time.numel(), M.shape[0]) or time.numel() < 2:
        raise ValueError("incompatible time or force history")
    dtv = time[1:] - time[:-1]
    if torch.any(dtv <= 0) or not torch.allclose(dtv, dtv[0]):
        raise ValueError("time must be uniformly increasing")
    if C is None:
        C = torch.zeros_like(M)
    if C.shape != M.shape or u0.shape != (M.shape[0],) or v0.shape != u0.shape:
        raise ValueError("incompatible damping or initial condition")
    u0 = u0.to(dtype=M.dtype, device=M.device)
    v0 = v0.to(dtype=M.dtype, device=M.device)
    return u0, v0, C, dtv[0]


def _newmark_constants(beta: float, gamma: float, dt: float):
    """Effective-stiffness and back-substitution constants of the Newmark family."""
    c0 = 1 / (beta * dt**2)
    c1 = gamma / (beta * dt)
    c2 = 1 / (beta * dt)
    c3 = 1 / (2 * beta) - 1
    c4 = gamma / beta - 1
    c5 = dt * (gamma / (2 * beta) - 1)
    return c0, c1, c2, c3, c4, c5


def newmark_linear(M: torch.Tensor, K: torch.Tensor, force: torch.Tensor,
                   time: torch.Tensor, u0: torch.Tensor, v0: torch.Tensor,
                   C: torch.Tensor | None = None, beta: float = .25,
                   gamma: float = .5) -> TransientResult:
    u0, v0, C, dt = _check_transient_inputs(M, K, force, time, u0, v0, C)
    n = time.numel()
    ndof = M.shape[0]
    u = torch.zeros((n, ndof), dtype=M.dtype)
    v = torch.zeros_like(u)
    a = torch.zeros_like(u)
    u[0] = u0
    v[0] = v0
    a[0] = torch.linalg.solve(M, force[0] - C @ v0 - K @ u0)
    c0, c1, c2, c3, c4, c5 = _newmark_constants(beta, gamma, dt)
    Ke = K + c0 * M + c1 * C
    for i in range(n - 1):
        rhs = (force[i + 1] + M @ (c0 * u[i] + c2 * v[i] + c3 * a[i])
               + C @ (c1 * u[i] + c4 * v[i] + c5 * a[i]))
        u[i + 1] = torch.linalg.solve(Ke, rhs)
        a[i + 1] = c0 * (u[i + 1] - u[i]) - c2 * v[i] - c3 * a[i]
        v[i + 1] = v[i] + dt * ((1 - gamma) * a[i] + gamma * a[i + 1])
    return TransientResult(time, u, v, a)


def hht_alpha(M: torch.Tensor, K: torch.Tensor, force: torch.Tensor,
              time: torch.Tensor, u0: torch.Tensor, v0: torch.Tensor,
              C: torch.Tensor | None = None, alpha: float = -1 / 3,
              beta: float | None = None, gamma: float | None = None) -> TransientResult:
    """Hilbert-Hughes-Taylor alpha integration of ``M a + C v + K u = F``.

    The scheme weights damping and stiffness at ``t_{n+1+alpha}``::

        M a_{n+1} + (1+alpha)(C v_{n+1} + K u_{n+1}) - alpha (C v_n + K u_n)
            = (1+alpha) F_{n+1} - alpha F_n

    with the Newmark displacement/velocity updates parameterised by ``beta`` and
    ``gamma``.  ``alpha in [-1/3, 0]`` keeps the scheme unconditionally stable
    and second-order accurate.  ``rho_inf = (1+alpha)/(1-alpha)`` is the
    spectral radius of the amplification matrix at ``omega*dt -> infinity``,
    i.e. the fraction of a highest-frequency (spurious) oscillation surviving
    one step: ``alpha = 0`` is the non-dissipative average-acceleration Newmark
    scheme (``rho_inf = 1``), ``alpha = -1/3`` the most dissipative stable
    setting (``rho_inf = 0.5``).  The defaults ``beta = (1-alpha)**2/4`` and
    ``gamma = 1/2 - alpha`` pair the scheme with maximal high-frequency
    dissipation for the chosen ``rho_inf`` while leaving the resolved
    low-frequency range second-order accurate.

    ``alpha = 0`` reproduces :func:`newmark_linear` bit-for-bit.
    """
    alpha = float(alpha)
    if not -1 / 3 - 1e-12 <= alpha <= 1e-12:
        raise ValueError("alpha must lie in [-1/3, 0] for unconditional stability")
    beta = (1 - alpha) ** 2 / 4 if beta is None else beta
    gamma = .5 - alpha if gamma is None else gamma
    if beta <= 0:
        raise ValueError("beta must be positive")
    u0, v0, C, dt = _check_transient_inputs(M, K, force, time, u0, v0, C)
    n = time.numel()
    ndof = M.shape[0]
    u = torch.zeros((n, ndof), dtype=M.dtype)
    v = torch.zeros_like(u)
    a = torch.zeros_like(u)
    u[0] = u0
    v[0] = v0
    a[0] = torch.linalg.solve(M, force[0] - C @ v0 - K @ u0)
    c0, c1, c2, c3, c4, c5 = _newmark_constants(beta, gamma, dt)
    sa = 1 + alpha
    Ke = sa * K + c0 * M + sa * c1 * C
    for i in range(n - 1):
        rhs = (sa * force[i + 1] - alpha * force[i]
               + M @ (c0 * u[i] + c2 * v[i] + c3 * a[i])
               + sa * (C @ (c1 * u[i] + c4 * v[i] + c5 * a[i]))
               + alpha * (C @ v[i] + K @ u[i]))
        u[i + 1] = torch.linalg.solve(Ke, rhs)
        a[i + 1] = c0 * (u[i + 1] - u[i]) - c2 * v[i] - c3 * a[i]
        v[i + 1] = v[i] + dt * ((1 - gamma) * a[i] + gamma * a[i + 1])
    return TransientResult(time, u, v, a)


def mechanical_energy(M: torch.Tensor, K: torch.Tensor,
                      u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return .5 * torch.einsum("ti,ij,tj->t", v, M, v) + .5 * torch.einsum("ti,ij,tj->t", u, K, u)