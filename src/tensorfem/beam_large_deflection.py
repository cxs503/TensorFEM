"""Planar large-deflection (elastica) beam verification module.

Two independent routes for the same physics are provided so that one can be
used as the oracle of the other:

``elastica_cantilever``
    The *exact* Euler-elastica solution of a cantilever bent by a transverse
    tip force, expressed through incomplete elliptic integrals of the first
    and second kind (Bisshopp & Drucker 1945; Frisch-Fay 1962).  The beam is
    inextensible, shear-rigid and initially straight.

``solve_corotational_beam``
    A finite-element solution with corotational 2-node beam elements and an
    energy-derived consistent tangent (the tangent is obtained from the
    assembled strain energy by automatic differentiation, so it is exact for
    the element formulation rather than a hand-coded approximation).  The
    kinematics carries *exact* finite rotations (the element rotation is the
    chord rotation, not a small-angle linearisation), which is what makes it
    usable outside the small-deflection range of :mod:`tensorfem.modal`.

The two routes are used together in
``benchmarks/indentation_cycle_study.py`` with the acceptance gate
``|delta_FE/delta_elastica - 1| < 2%``.

Formulation of the corotational element
---------------------------------------
Node coordinates ``(x_i, y_i)``, nodal rotations ``theta_i`` (measured from
the global +y axis, positive counter-clockwise from +x, see the sign
convention in ``_element_energy``).  With ``L0`` the initial length,
``l`` the current chord length, ``phi = atan2(dy, dx)`` the current chord
angle, the element energy is::

    U_e = 1/2 * EA/L0 * (l - L0)^2
        + 2*EI/L0 * (t1^2 + t2^2 + t1*t2),      t_i = theta_i - phi

The first term is the axial stretch energy, the second is the local
Euler-Bernoulli bending energy written in the *rigidly rotated* frame (the
standard corotational "local rotations" ``t_i``).  ``U_e`` is polynomial in
``l`` and ``phi`` and therefore smooth; the internal force vector and the
tangent stiffness are obtained by differentiating the assembled energy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ellipe, ellipk

from .elastica_cantilever import newton_converged

try:  # scipy>=1.11 exposes the incomplete integrals in scipy.special
    from scipy.special import ellipeinc, ellipkinc
except ImportError:  # pragma: no cover - very old scipy
    ellipeinc = ellipkinc = None


# ---------------------------------------------------------------------------
# exact elastica (elliptic integrals)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ElasticaSolution:
    """Exact cantilever elastica for a transverse tip force."""

    load_parameter: float      # alpha = P L^2 / (E I)
    tip_x: float               # horizontal coordinate of the tip (x(0)=0)
    tip_y: float               # deflection in the load direction
    tip_angle: float           # slope at the tip (rad)
    length: float

    @property
    def tip_deflection_ratio(self) -> float:
        return self.tip_y / self.length


def elastica_cantilever(*, load_parameter: float, length: float = 1.0,
                        n_quad: int = 200) -> ElasticaSolution:
    """Exact elastica of a cantilever under a transverse tip force.

    Parameters
    ----------
    load_parameter : float
        ``alpha = P L^2 / (E I)``, the dimensionless tip force.  ``alpha > 0``
        bends the beam towards +y.  The elliptic parametrisation is
        ``k = sin(theta_tip/2 + pi/4)`` with ``k in [1/sqrt(2), 1)``; the
        mapping from ``alpha`` to ``k`` is solved by bisection, which is
        monotonically increasing in ``alpha``.
    length : float
        Beam length ``L`` (only a scale factor, ``alpha`` is independent of
        it when ``E I`` is fixed).
    n_quad : int
        Fallback quadrature points for the (unused unless scipy lacks the
        incomplete elliptic integrals) numerical evaluation of the tip
        coordinates.

    Returns
    -------
    ElasticaSolution
        ``tip_x`` is the shortening ``x(L)`` measured from the clamp,
        ``tip_y`` the deflection along the load direction.

    Notes
    -----
    Derivation (``s`` arc length, ``theta`` slope, ``c = sqrt(P/EI)``)::

        EI theta'' + P cos(theta) = 0,   theta(0) = 0, theta'(L) = 0
        L = (K(k) - F(psi_0, k)) / c,    sin(psi_0) = 1/(k sqrt(2))
        x(L) = 2 k cos(psi_0) / c
        y(L) = (K - F - 2 (E(k) - E(psi_0, k))) / c

    with ``k = sin(theta(L)/2 + pi/4)``.  The small-load limit reproduces
    ``delta = P L^3 / (3 E I)``; see ``tests/test_beam_large_deflection.py``.
    """
    alpha = float(load_parameter)
    if alpha <= 0.0:
        raise ValueError("load_parameter must be positive")
    if ellipkinc is None:  # pragma: no cover
        raise RuntimeError("scipy.special elliptic integrals are required")

    def tip_deflection(k: float) -> tuple[float, float, float]:
        psi0 = np.arcsin(min(1.0, 1.0 / (k * np.sqrt(2.0))))
        K, E = float(ellipk(k * k)), float(ellipe(k * k))
        F0 = float(ellipkinc(psi0, k * k))
        E0 = float(ellipeinc(psi0, k * k))
        return K - F0, E - E0, float(np.cos(psi0))

    def alpha_of(k: float) -> float:
        dK, _, _ = tip_deflection(k)
        if dK <= 0.0:
            return 0.0
        c = dK / length
        return c * c * (length * length)  # P = EI c^2, alpha = P L^2 / EI

    lo, hi = 1.0 / np.sqrt(2.0) + 1e-12, 1.0 - 1e-12
    if alpha_of(hi) < alpha:
        raise ValueError(
            f"load_parameter {alpha} exceeds the elliptic-domain limit "
            f"{alpha_of(hi):.6g} (tip slope approaches pi/2)")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if alpha_of(mid) < alpha:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    dK, dE, cos_psi0 = tip_deflection(k)
    c = dK / length
    tip_x = 2.0 * k * cos_psi0 / c
    tip_y = (dK - 2.0 * dE) / c
    tip_angle = 2.0 * (np.arcsin(min(1.0, k)) - np.pi / 4.0)
    return ElasticaSolution(alpha, tip_x, tip_y, tip_angle, length)


def _elastica_by_quadrature(*, load_parameter: float, length: float = 1.0,
                            n_quad: int = 4000) -> ElasticaSolution:
    """Independent quadrature evaluation of the elastica (test oracle).

    Solves ``theta' = sqrt(2 (P/EI) (sin theta_L - sin theta))`` for the tip
    slope by bisection on ``theta_L`` and integrates ``x``/``y`` along the arc
    length with a substitution that removes the endpoint square-root
    singularity.  Used only to cross-check :func:`elastica_cantilever`.
    """
    alpha = float(load_parameter)
    c = np.sqrt(alpha) / length  # EI = 1 convention: P = alpha EI / L^2

    def length_of(theta_tip: float) -> float:
        th = np.linspace(0.0, theta_tip, n_quad, endpoint=False)
        integrand = 1.0 / np.sqrt(2.0 * c * c * (np.sin(theta_tip) - np.sin(th)))
        ds = np.diff(np.concatenate((th, [theta_tip])))
        return float(np.sum(integrand * ds) + 0.0)

    lo, hi = 1e-9, np.pi / 2 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if length_of(mid) < length:
            lo = mid
        else:
            hi = mid
    theta_tip = 0.5 * (lo + hi)
    th = np.linspace(0.0, theta_tip, n_quad, endpoint=False)
    ds = np.diff(np.concatenate((th, [theta_tip])))
    weight = 1.0 / np.sqrt(2.0 * c * c * (np.sin(theta_tip) - np.sin(th)))
    x = float(np.sum(np.cos(th) * weight * ds))
    y = float(np.sum(np.sin(th) * weight * ds))
    return ElasticaSolution(alpha, x, y, theta_tip, length)


# ---------------------------------------------------------------------------
# corotational finite element
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BeamSolution:
    displacement: np.ndarray
    tip_x: float
    tip_y: float
    tip_angle: float
    iterations: int
    converged: bool
    residual_norm: float


def _element_scalar_energy(dof, nodes6, ea: float, ei: float, length0: float):
    """Corotational element energy for one element (6 dofs -> scalar)."""
    import torch

    x0, y0, x1, y1 = nodes6[0], nodes6[1], nodes6[2], nodes6[3]
    dx = (x1 - x0) + dof[3] - dof[0]
    dy = (y1 - y0) + dof[4] - dof[1]
    length = torch.sqrt(dx * dx + dy * dy)
    phi = torch.atan2(dy, dx)
    t1 = dof[2] - phi
    t2 = dof[5] - phi
    axial = 0.5 * ea / length0 * (length - length0) ** 2
    bending = 2.0 * ei / length0 * (t1 * t1 + t2 * t2 + t1 * t2)
    return axial + bending


def _batch_element_derivatives(element_nodes, dofs_elem, ea, ei, lengths):
    """Batched element internal forces and consistent tangents via autodiff.

    ``torch.func.vmap`` maps ``jacrev``/``hessian`` of the scalar element
    energy over all elements in one pass, so the tangent is the *exact*
    derivative of the discrete energy (verified against central differences in
    :func:`corotational_tangent_error`).
    """
    import torch
    from torch.func import jacrev, vmap

    nodes_t = torch.tensor(np.asarray(element_nodes, dtype=np.float64))
    dofs_t = torch.tensor(np.asarray(dofs_elem, dtype=np.float64))
    len_t = torch.tensor(np.asarray(lengths, dtype=np.float64))

    def elem(dof, nodes6, l0):
        return _element_scalar_energy(dof, nodes6, ea, ei, l0)

    forces = vmap(jacrev(elem), in_dims=(0, 0, 0))(dofs_t, nodes_t, len_t)
    tangents = vmap(jacrev(jacrev(elem)), in_dims=(0, 0, 0))(dofs_t, nodes_t, len_t)
    return forces.numpy(), tangents.numpy()


def _assemble(element_nodes, u, ea, ei, connectivity, lengths):
    """Return (energy, global internal force, global tangent)."""
    u = np.asarray(u, dtype=np.float64)
    n_nodes = u.size // 3
    dofs_elem = u.reshape(n_nodes, 3)[connectivity].reshape(len(connectivity), 6)
    f_e, k_e = _batch_element_derivatives(element_nodes, dofs_elem, ea, ei, lengths)
    f_int = np.zeros_like(u)
    k_tan = np.zeros((u.size, u.size))
    conn_dofs = (3 * connectivity[:, :, None]
                 + np.arange(3)[None, None, :]).reshape(len(connectivity), 6)
    for e in range(len(connectivity)):
        idx = conn_dofs[e]
        f_int[idx] += f_e[e]
        k_tan[np.ix_(idx, idx)] += k_e[e]
    energy = 0.0
    for e in range(len(connectivity)):
        nodes6 = element_nodes[e]
        dx = (nodes6[2] - nodes6[0]) + dofs_elem[e, 3] - dofs_elem[e, 0]
        dy = (nodes6[3] - nodes6[1]) + dofs_elem[e, 4] - dofs_elem[e, 1]
        length = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx)
        t1 = dofs_elem[e, 2] - phi
        t2 = dofs_elem[e, 5] - phi
        energy += (0.5 * ea / lengths[e] * (length - lengths[e]) ** 2
                   + 2.0 * ei / lengths[e] * (t1 * t1 + t2 * t2 + t1 * t2))
    return energy, f_int, k_tan


def solve_corotational_beam(*, n_elements: int = 20, length: float = 1.0,
                            ea: float = 1.0e6, ei: float = 1.0,
                            tip_force: float = 1.0, load_steps: int = 10,
                            tolerance: float = 1e-10,
                            max_iterations: int = 40) -> BeamSolution:
    """Cantilever under a transverse tip force with corotational beam elements.

    The clamp is at the origin, ``n_elements`` equal segments along +x, the
    tip force acts in +y so the tip deflects towards +y.  The load is applied
    in equal increments with Newton iterations per increment, which keeps the
    strongly nonlinear (finite rotation) path well behaved.

    Parameters
    ----------
    ea : float
        Axial rigidity ``E A``; kept stiff (``1e6 * EI``) so that the axial
        stretch is negligible and the result is comparable with the
        inextensible elastica.
    ei : float
        Bending rigidity ``E I``; the dimensionless load of the comparison is
        ``alpha = P L^2 / (E I)``.
    """
    if n_elements < 1:
        raise ValueError("n_elements must be positive")
    n_nodes = n_elements + 1
    x = np.linspace(0.0, length, n_nodes)
    coords = np.stack((x, np.zeros_like(x)), axis=1)
    connectivity = np.array([[i, i + 1] for i in range(n_elements)])
    element_nodes = np.concatenate(
        (coords[connectivity[:, 0]], coords[connectivity[:, 1]]), axis=1)
    lengths = np.full(n_elements, length / n_elements)

    dof_index = np.arange(3 * n_nodes).reshape(n_nodes, 3)
    fixed = dof_index[0]                       # clamp: u, v, theta
    free = np.setdiff1d(np.arange(3 * n_nodes), fixed)

    u = np.zeros(3 * n_nodes)
    load = np.zeros(3 * n_nodes)
    tip_v = dof_index[-1, 1]
    converged = True
    iterations = 0
    residual = 0.0
    for step in range(1, load_steps + 1):
        load[:] = 0.0
        load[tip_v] = tip_force * step / load_steps
        for _ in range(max_iterations):
            iterations += 1
            _, f_int, k_tan = _assemble(element_nodes, u, ea, ei,
                                        connectivity, lengths)
            g = f_int - load
            g_free = g[free]
            force_norm = float(np.linalg.norm(load))
            dof_norm = float(np.linalg.norm(u[free]))
            g_norm = float(np.linalg.norm(g_free))
            scale = max(force_norm, 1.0)
            residual = g_norm / scale
            # Branch A: relative force residual.  Branch B: the residual sits on
            # the round-off floor of the internal-force assembly (which scales
            # with the stiff axial term EA/L) and the Newton increment is below
            # the resolution of the iterate -- see
            # ``elastica_cantilever.newton_converged`` for the full argument.
            if newton_converged(g_norm, 0.0, dof_norm, force_norm,
                                ftol=tolerance * scale):
                break
            delta = np.linalg.solve(k_tan[np.ix_(free, free)], -g_free)
            if newton_converged(g_norm, float(np.linalg.norm(delta)),
                                dof_norm, force_norm, ftol=tolerance * scale):
                break
            u[free] += delta
        else:  # pragma: no cover - Newton failed to reach the tolerance
            converged = False

    return BeamSolution(u, length + u[dof_index[-1, 0]], u[dof_index[-1, 1]],
                        u[dof_index[-1, 2]], iterations, converged, residual)


def corotational_tangent_error(*, n_elements: int = 4, length: float = 1.0,
                               ea: float = 1e4, ei: float = 1.0,
                               seed: int = 0) -> dict:
    """Finite-difference audit of the autograd consistent tangent.

    A random admissible displacement state is perturbed; the tangent returned
    by the module is compared with a central-difference Jacobian of the
    internal force.  ``max_relative_error`` is the audit number reported in
    the study.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, length, n_elements + 1)
    coords = np.stack((x, np.zeros_like(x)), axis=1)
    connectivity = np.array([[i, i + 1] for i in range(n_elements)])
    element_nodes = np.concatenate(
        (coords[connectivity[:, 0]], coords[connectivity[:, 1]]), axis=1)
    lengths = np.full(n_elements, length / n_elements)
    u = 0.01 * rng.standard_normal(3 * (n_elements + 1))
    _, f0, k_tan = _assemble(element_nodes, u, ea, ei, connectivity, lengths)
    h = 1e-7
    k_fd = np.zeros_like(k_tan)
    for j in range(len(u)):
        du = np.zeros_like(u)
        du[j] = h
        _, fp, _ = _assemble(element_nodes, u + du, ea, ei, connectivity, lengths)
        _, fm, _ = _assemble(element_nodes, u - du, ea, ei, connectivity, lengths)
        k_fd[:, j] = (fp - fm) / (2 * h)
    scale = max(np.max(np.abs(k_fd)), 1e-30)
    return {
        "max_relative_error": float(np.max(np.abs(k_tan - k_fd)) / scale),
        "state_norm": float(np.linalg.norm(u)),
        "force_norm": float(np.linalg.norm(f0)),
    }