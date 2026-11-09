"""Winkler elastic-foundation beam elements and unilateral (no-tension) contact.

This module owns the *beam-level* elastic-foundation capability that the
plate-level counterparts (:mod:`tensorfem.winkler_foundation`,
:mod:`tensorfem.foundation_plate_bending`) do not provide, plus a **unilateral**
(compression-only) contact solver that both a beam and a Q4 Mindlin plate can be
driven with:

1. consistent Winkler foundation stiffness ``k_e = integral k(x) N^T N dx`` on
   Euler-Bernoulli cubic Hermite elements and on two-node Timoshenko elements,
   together with the consistent nodal loads of a uniform distributed load;
2. assembly of a beam on a piecewise-constant (hence arbitrary) ``k(x)``
   foundation, with lumped nodal springs ``k_i = sum_e k_e L_e / 2`` for the
   one-sided model (tributary lumping, so ``sum_i k_i = integral k dx`` and the
   total reaction is conserved);
3. a bilateral solver, a *tensionless* (lift-off) active-set solver and a
   *rigid bed* (Signorini) active-set solver, all exposing the pointwise
   complementarity data ``(R, y)`` required by the KKT gate;
4. KKT / complementarity residuals, an energy ledger, element end moments and
   lift-off extent extraction;
5. a semi-analytic reference for the lift-off length of an infinite beam under
   a uniform load plus a central uplift force, and the Hetenyi closed form
   ``y_max = P beta / (2 k)`` for the bilateral case;
6. a Q4 Mindlin plate unilateral extension (lumped surface springs) for
   "out-of-plane spring distribution + lift-off" plate problems.

Conventions
-----------
* ``w`` is the transverse displacement, **positive downward** (the direction of
  a gravity load).  ``theta = dw/dx`` is the cross-section rotation.
* The distributed load ``q`` and the nodal loads are positive downward.
* The foundation coefficient ``k`` of the beam model is *per unit length*
  (F/L^2); the plate coefficient is per unit area (F/L^3).
* A node is *in contact* when ``w_i > g_i`` (``g`` = initial gap, positive when
  the foundation lies below the beam reference line).  The compressive reaction
  is ``R_i = k_i (w_i - g_i)^+ >= 0``; the separation (positive when the node
  lifted off) is ``y_i = max(g_i - w_i, 0) >= 0``.  The complementarity
  statement is ``R_i >= 0``, ``y_i >= 0``, ``R_i y_i = 0`` together with the
  spring law ``R_i = k_i (w_i - g_i)^+``.

Reference
---------
M. Hetenyi, *Beams on Elastic Foundation*, University of Michigan Press, 1946
(characteristic wavenumber ``beta = (k / 4EI)^(1/4)``).
Standard Timoshenko-beam kinematics ``gamma = w' - theta``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .plate import q4_mindlin_stiffness
from .winkler_foundation import q4_winkler_foundation_stiffness

__all__ = [
    "FOUNDATION_ELEMENTS",
    "BeamFoundationModel",
    "uniform_beam",
    "hermite_shapes",
    "hermite_second_derivatives",
    "eb_beam_stiffness",
    "hermite_foundation_stiffness",
    "hermite_foundation_stiffness_quadrature",
    "hermite_consistent_load",
    "timoshenko_beam_stiffness",
    "timoshenko_foundation_stiffness",
    "timoshenko_consistent_load",
    "assemble_beam_foundation",
    "solve_bilateral",
    "solve_tensionless",
    "solve_rigid_bed",
    "unilateral_residuals",
    "energy_ledger",
    "element_end_moments",
    "lift_off_extent",
    "hetenyi_point_load_deflection",
    "hetenyi_penetration",
    "characteristic_wavenumber",
    "lift_off_length_analytic",
    "q4_winkler_nodal_springs",
    "assemble_plate_foundation",
    "solve_plate_foundation",
]

FLOAT = torch.float64

#: element families understood by :func:`assemble_beam_foundation`.
FOUNDATION_ELEMENTS = ("eb", "timoshenko")


# ---------------------------------------------------------------------------
# quadrature
# ---------------------------------------------------------------------------
_GL_UNIT = {
    1: ([0.5], [1.0]),
    2: ([0.5 - 0.5 / 3.0**0.5, 0.5 + 0.5 / 3.0**0.5], [0.5, 0.5]),
    4: (
        [
            0.5 - 0.5 * (3.0 / 7.0 - 2.0 / 7.0 * (6.0 / 5.0) ** 0.5) ** 0.5,
            0.5 - 0.5 * (3.0 / 7.0 + 2.0 / 7.0 * (6.0 / 5.0) ** 0.5) ** 0.5,
            0.5 + 0.5 * (3.0 / 7.0 + 2.0 / 7.0 * (6.0 / 5.0) ** 0.5) ** 0.5,
            0.5 + 0.5 * (3.0 / 7.0 - 2.0 / 7.0 * (6.0 / 5.0) ** 0.5) ** 0.5,
        ],
        [
            0.6521451548625461 * 0.5,
            0.3478548451374538 * 0.5,
            0.3478548451374538 * 0.5,
            0.6521451548625461 * 0.5,
        ],
    ),
}


def _gauss_unit(order: int, dtype=FLOAT):
    if order not in _GL_UNIT:
        raise ValueError("supported Gauss orders on the unit interval: 1, 2, 4")
    xi, wq = _GL_UNIT[order]
    return torch.tensor(xi, dtype=dtype), torch.tensor(wq, dtype=dtype)


# ---------------------------------------------------------------------------
# Euler-Bernoulli cubic Hermite element
# ---------------------------------------------------------------------------
def hermite_shapes(xi: torch.Tensor, length: float):
    """Cubic Hermite shape functions on ``xi in [0, 1]`` for ``(w1,t1,w2,t2)``.

    Returns ``(N, dNdx)``, each of shape ``(4,)``.
    """
    L = float(length)
    xi = torch.as_tensor(xi, dtype=FLOAT)
    one = torch.ones_like(xi)
    N = torch.stack(
        (
            one - 3.0 * xi**2 + 2.0 * xi**3,
            L * (xi - 2.0 * xi**2 + xi**3),
            3.0 * xi**2 - 2.0 * xi**3,
            L * (-xi**2 + xi**3),
        )
    )
    dN = torch.stack(
        (
            (-6.0 * xi + 6.0 * xi**2) / L,
            one - 4.0 * xi + 3.0 * xi**2,
            (6.0 * xi - 6.0 * xi**2) / L,
            -2.0 * xi + 3.0 * xi**2,
        )
    )
    return N, dN


def hermite_second_derivatives(xi: torch.Tensor, length: float) -> torch.Tensor:
    """``d^2 N / dx^2`` of the cubic Hermite basis, shape ``(4,)``."""
    L = float(length)
    xi = torch.as_tensor(xi, dtype=FLOAT)
    return torch.stack(
        (
            (-6.0 + 12.0 * xi) / L**2,
            (-4.0 + 6.0 * xi) / L,
            (6.0 - 12.0 * xi) / L**2,
            (-2.0 + 6.0 * xi) / L,
        )
    )


def eb_beam_stiffness(E: float, Iz: float, L: float, *, dtype=FLOAT) -> torch.Tensor:
    """Exact 4x4 Euler-Bernoulli bending stiffness for ``(w1, t1, w2, t2)``."""
    if min(E, Iz, L) <= 0.0:
        raise ValueError("E, Iz and L must be positive")
    return E * Iz / L**3 * torch.tensor(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L**2, -6.0 * L, 2.0 * L**2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L**2, -6.0 * L, 4.0 * L**2],
        ],
        dtype=dtype,
    )


def hermite_foundation_stiffness(L: float, k: float, *, dtype=FLOAT) -> torch.Tensor:
    """Closed-form consistent Winkler matrix ``k * int N^T N dx`` (Hermite cubic).

    ``k L / 420 * [[156, 22L, 54, -13L], ...]`` -- the same pattern as the
    Hermite consistent mass matrix, because both are ``int N^T N dx``.
    """
    if L <= 0.0:
        raise ValueError("L must be positive")
    pattern = torch.tensor(
        [
            [156.0, 22.0 * L, 54.0, -13.0 * L],
            [22.0 * L, 4.0 * L**2, 13.0 * L, -3.0 * L**2],
            [54.0, 13.0 * L, 156.0, -22.0 * L],
            [-13.0 * L, -3.0 * L**2, -22.0 * L, 4.0 * L**2],
        ],
        dtype=dtype,
    )
    return (k * L / 420.0) * pattern


def hermite_foundation_stiffness_quadrature(L: float, k, *, order: int = 4,
                                            dtype=FLOAT) -> torch.Tensor:
    """``int_0^L k(x) N^T N dx`` by Gauss quadrature (``k`` scalar or callable of x)."""
    if L <= 0.0:
        raise ValueError("L must be positive")
    xi, wq = _gauss_unit(order, dtype=dtype)
    K = torch.zeros((4, 4), dtype=dtype)
    for a, b in zip(xi, wq):
        N, _ = hermite_shapes(a, L)
        kx = k(float(a) * L) if callable(k) else k
        K = K + (torch.as_tensor(kx, dtype=dtype) * b * L) * torch.outer(N, N)
    return K


def hermite_consistent_load(L: float, q, *, dtype=FLOAT) -> torch.Tensor:
    """Consistent nodal load of a uniform transverse load ``q`` (downward +)."""
    if L <= 0.0:
        raise ValueError("L must be positive")
    qq = torch.as_tensor(q, dtype=dtype)
    return qq * torch.tensor([L / 2.0, L**2 / 12.0, L / 2.0, -L**2 / 12.0], dtype=dtype)


# ---------------------------------------------------------------------------
# two-node Timoshenko element
# ---------------------------------------------------------------------------
def timoshenko_beam_stiffness(E: float, Iz: float, G: float, A: float, kappa: float,
                              L: float, *, dtype=FLOAT) -> torch.Tensor:
    """Two-node Timoshenko stiffness for ``(w1, t1, w2, t2)`` (shear-flexible).

    Uses ``phi = 12 EI / (kappa G A L^2)`` so equal-order (linear) interpolation
    does not lock in the thin limit.
    """
    if min(E, Iz, G, A, kappa, L) <= 0.0:
        raise ValueError("beam properties must be positive")
    phi = 12.0 * E * Iz / (kappa * G * A * L**2)
    return E * Iz / (L**3 * (1.0 + phi)) * torch.tensor(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, (4.0 + phi) * L**2, -6.0 * L, (2.0 - phi) * L**2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, (2.0 - phi) * L**2, -6.0 * L, (4.0 + phi) * L**2],
        ],
        dtype=dtype,
    )


def timoshenko_foundation_stiffness(L: float, k: float, *, dtype=FLOAT) -> torch.Tensor:
    """``int k N^T N dx`` for linear ``w`` interpolation (``theta`` rows vanish)."""
    if L <= 0.0:
        raise ValueError("L must be positive")
    return k * L / 6.0 * torch.tensor(
        [
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )


def timoshenko_consistent_load(L: float, q, *, dtype=FLOAT) -> torch.Tensor:
    """Consistent nodal load of a uniform load ``q`` on linear ``w`` functions."""
    if L <= 0.0:
        raise ValueError("L must be positive")
    qq = torch.as_tensor(q, dtype=dtype)
    return qq * torch.tensor([L / 2.0, 0.0, L / 2.0, 0.0], dtype=dtype)


# ---------------------------------------------------------------------------
# model + assembly
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BeamFoundationModel:
    """A straight beam on a Winkler foundation, discretised by the nodes ``x``.

    Attributes
    ----------
    x : (n,) tensor
        Node coordinates, strictly increasing.  DOF layout ``(w_0, t_0, w_1,
        t_1, ...)``.
    E, Iz : float
        Bending stiffness ``E*Iz``.
    k : (n-1,) tensor or float
        Foundation coefficient per element (piecewise-constant ``k(x)``).
    element : {"eb", "timoshenko"}
        ``"eb"`` = cubic Hermite (C1, 2 DOF/node); ``"timoshenko"`` = two-node
        linear ``(w, theta)`` with shear flexibility (requires ``G``, ``A``).
    q : (n-1,) tensor or float
        Uniform transverse load per unit length on each element (downward +).
    """

    x: torch.Tensor
    E: float
    Iz: float
    k: object = 0.0
    element: str = "eb"
    q: object = 0.0
    G: float = 0.0
    A: float = 0.0
    kappa: float = 5.0 / 6.0

    def __post_init__(self):
        object.__setattr__(self, "x", torch.as_tensor(self.x, dtype=FLOAT))
        n_elem = int(self.x.numel()) - 1
        if n_elem < 1:
            raise ValueError("at least two nodes are required")
        if bool(torch.any(torch.diff(self.x) <= 0)):
            raise ValueError("x must be strictly increasing")
        object.__setattr__(self, "k", self._per_element(self.k, n_elem, "k"))
        object.__setattr__(self, "q", self._per_element(self.q, n_elem, "q"))
        if self.element not in FOUNDATION_ELEMENTS:
            raise ValueError(f"element must be one of {FOUNDATION_ELEMENTS}")
        if self.element == "timoshenko" and min(self.G, self.A) <= 0.0:
            raise ValueError("timoshenko elements require G > 0 and A > 0")

    @staticmethod
    def _per_element(value, n_elem: int, name: str) -> torch.Tensor:
        t = torch.as_tensor(value, dtype=FLOAT)
        if t.ndim == 0:
            return t.repeat(n_elem)
        if tuple(t.shape) != (n_elem,):
            raise ValueError(f"{name} must be a scalar or have shape ({n_elem},)")
        return t

    @property
    def n_nodes(self) -> int:
        return int(self.x.numel())

    @property
    def ndof(self) -> int:
        return 2 * self.n_nodes


def uniform_beam(length: float, n_elements: int, E: float, Iz: float, k=0.0, *,
                 element: str = "eb", q=0.0, G: float = 0.0, A: float = 0.0,
                 kappa: float = 5.0 / 6.0, x0: float = 0.0) -> BeamFoundationModel:
    """Convenience constructor: uniform mesh on ``[x0, x0 + length]``."""
    if n_elements < 1:
        raise ValueError("n_elements must be positive")
    x = torch.linspace(x0, x0 + length, n_elements + 1, dtype=FLOAT)
    return BeamFoundationModel(x=x, E=E, Iz=Iz, k=k, element=element, q=q, G=G, A=A,
                               kappa=kappa)


def assemble_beam_foundation(model: BeamFoundationModel) -> dict:
    """Assemble bending stiffness, consistent foundation stiffness and load vector.

    Returns
    -------
    dict with ``x``, ``K`` (bending), ``Kf`` (consistent foundation), ``f``
    (distributed-load vector), ``k_node`` (lumped nodal springs used by the
    one-sided model), ``lengths`` and ``dofs``.
    """
    n = model.n_nodes
    K = torch.zeros((2 * n, 2 * n), dtype=FLOAT)
    Kf = torch.zeros((2 * n, 2 * n), dtype=FLOAT)
    f = torch.zeros(2 * n, dtype=FLOAT)
    lengths = torch.diff(model.x)
    k_node = torch.zeros(n, dtype=FLOAT)
    idx = torch.arange(n - 1)
    dofs = torch.stack((2 * idx, 2 * idx + 1, 2 * idx + 2, 2 * idx + 3), dim=1)
    for e in range(n - 1):
        L = float(lengths[e])
        ke = float(model.k[e])
        if model.element == "eb":
            kb = eb_beam_stiffness(model.E, model.Iz, L)
            kf = hermite_foundation_stiffness(L, ke)
            fe = hermite_consistent_load(L, float(model.q[e]))
        else:
            kb = timoshenko_beam_stiffness(model.E, model.Iz, model.G, model.A,
                                           model.kappa, L)
            kf = timoshenko_foundation_stiffness(L, ke)
            fe = timoshenko_consistent_load(L, float(model.q[e]))
        d = dofs[e]
        K[d[:, None], d] += kb
        Kf[d[:, None], d] += kf
        f[d] += fe
        k_node[e] += ke * L / 2.0
        k_node[e + 1] += ke * L / 2.0
    return {
        "x": model.x,
        "K": K,
        "Kf": Kf,
        "f": f,
        "k_node": k_node,
        "lengths": lengths,
        "dofs": dofs,
    }

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _dof_list(fixed: Sequence[int] | torch.Tensor | None, ndof: int) -> torch.Tensor:
    if fixed is None:
        return torch.zeros(0, dtype=torch.long)
    t = torch.as_tensor(fixed, dtype=torch.long).reshape(-1)
    if t.numel() and (int(t.min()) < 0 or int(t.max()) >= ndof):
        raise ValueError("fixed DOF index out of range")
    return torch.unique(t)


def _solve_reduced(K: torch.Tensor, rhs: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    ndof = int(rhs.numel())
    mask = torch.ones(ndof, dtype=torch.bool)
    mask[fixed] = False
    free = torch.nonzero(mask).reshape(-1)
    u = torch.zeros(ndof, dtype=K.dtype)
    if free.numel() == 0:
        return u
    Kff = K[free[:, None], free]
    chol, info = torch.linalg.cholesky_ex(Kff)
    if int(info) != 0:
        raise RuntimeError(
            "reduced system is singular or indefinite; the model lacks foundation "
            "stiffness or supports"
        )
    sol = torch.cholesky_solve(rhs[free].unsqueeze(-1), chol).squeeze(-1)
    if not bool(torch.all(torch.isfinite(sol))):  # pragma: no cover - defensive
        raise RuntimeError("reduced system produced non-finite displacements")
    u[free] = sol
    return u


def _w_dofs(n_nodes: int) -> torch.Tensor:
    return 2 * torch.arange(n_nodes)


def _as_gap(gap, n_nodes: int) -> torch.Tensor:
    g = torch.as_tensor(gap, dtype=FLOAT)
    if g.ndim == 0:
        return g.repeat(n_nodes)
    if tuple(g.shape) != (n_nodes,):
        raise ValueError(f"gap must be a scalar or have shape ({n_nodes},)")
    return g


def _total_load(asm: dict, nodal_loads) -> torch.Tensor:
    f = asm["f"].clone()
    if nodal_loads is not None:
        nl = torch.as_tensor(nodal_loads, dtype=FLOAT)
        if tuple(nl.shape) != (int(f.numel()),):
            raise ValueError(f"nodal_loads must have shape ({int(f.numel())},)")
        f = f + nl
    return f


def _result(model, asm, w, f, fixed, kind, **extra) -> dict:
    n = model.n_nodes
    wd = _w_dofs(n)
    balance = asm["K"] @ w
    out = {
        "kind": kind,
        "model": model,
        "assembly": asm,
        "x": model.x,
        "w": w,
        "w_nodes": w[wd],
        "theta": w[wd + 1],
        "bending_residual": balance,
        "loads": f,
        "fixed_dofs": fixed,
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# bilateral solver
# ---------------------------------------------------------------------------
def solve_bilateral(model: BeamFoundationModel, nodal_loads=None,
                    fixed_dofs=None) -> dict:
    """Solve the beam on a *bilateral* (tension-capable) Winkler foundation."""
    asm = assemble_beam_foundation(model)
    f = _total_load(asm, nodal_loads)
    fixed = _dof_list(fixed_dofs, model.ndof)
    Kt = asm["K"] + asm["Kf"]
    w = _solve_reduced(Kt, f, fixed)
    R = asm["Kf"] @ w
    wd = _w_dofs(model.n_nodes)
    return _result(model, asm, w, f, fixed, "bilateral",
                   reaction_nodal=R[wd],
                   spring_forces=R[wd],
                   iterations=0, converged=True,
                   support_reaction=(Kt @ w - f)[fixed])


# ---------------------------------------------------------------------------
# unilateral: tensionless elastic foundation (lift-off)
# ---------------------------------------------------------------------------
def solve_tensionless(model: BeamFoundationModel, nodal_loads=None, fixed_dofs=None,
                      gap=0.0, max_iter: int = 20, tol: float = 1e-12,
                      init_active=None) -> dict:
    """Active-set solution of a beam on a compression-only Winkler foundation.

    The one-sided model uses the lumped nodal springs ``k_i`` of
    :func:`assemble_beam_foundation`; a node carries a spring only while it
    penetrates (``w_i > g_i``).  Each iteration solves the linear system with
    the current active set and re-derives the set from the sign of the
    penetration, which is the classical no-tension trial-and-error iteration
    for this convex quadratic program; it terminates when the set repeats.
    """
    asm = assemble_beam_foundation(model)
    f = _total_load(asm, nodal_loads)
    fixed = _dof_list(fixed_dofs, model.ndof)
    n = model.n_nodes
    wd = _w_dofs(n)
    kn = asm["k_node"]
    g = _as_gap(gap, n)
    K = asm["K"]

    if init_active is None:
        # prefer the "no spring" predictor (a free beam); fall back to full
        # contact when that system is singular (e.g. unsupported beams)
        active = torch.zeros(n, dtype=torch.bool)
        try:
            _solve_reduced(K, f, fixed)
        except RuntimeError:
            active = torch.ones(n, dtype=torch.bool)
    else:
        active = torch.as_tensor(init_active, dtype=torch.bool).reshape(-1).clone()
        if int(active.numel()) != n:
            raise ValueError(f"init_active must have {n} entries")
    history: list[int] = []
    seen: set[tuple] = set()
    u = torch.zeros(model.ndof, dtype=FLOAT)
    converged = False
    for it in range(1, max_iter + 1):
        Kt = K.clone()
        Kt[wd, wd] += torch.where(active, kn, torch.zeros_like(kn))
        rhs = f.clone()
        if bool(active.any()):
            rhs[wd] += torch.where(active, kn * g, torch.zeros_like(kn))
        u = _solve_reduced(Kt, rhs, fixed)
        w_scale = float(u[wd].abs().max()) if n else 0.0
        geo_tol = tol * max(w_scale, 1e-30)
        new_active = (u[wd] - g) > geo_tol
        if not bool(new_active.any()):
            new_active[int(torch.argmax(u[wd] - g))] = True
        history.append(int(new_active.sum()))
        res_it = unilateral_residuals(K, f, u, kn, g, fixed=fixed, kind="tensionless")
        kkt = max(res_it["stationarity"], res_it["complementarity"],
                  res_it.get("spring_law", 0.0), -res_it["reaction_min"])
        if bool(torch.equal(new_active, active)) or kkt <= tol:
            converged = True
            break
        key = tuple(int(i) for i in torch.nonzero(new_active).reshape(-1).tolist())
        if key in seen:
            raise RuntimeError(
                f"tensionless active set cycled after {it} iterations "
                f"(contact counts {history})"
            )
        seen.add(key)
        active = new_active
    if not converged:
        raise RuntimeError(
            f"tensionless active set did not converge in {max_iter} iterations "
            f"(contact counts {history})"
        )

    R = f[wd] - (K @ u)[wd]
    if fixed.numel():
        fixed_n = set(int(d) for d in fixed.tolist() if int(d) % 2 == 0)
        if fixed_n & set(int(i) for i in torch.nonzero(active).reshape(-1).tolist()):
            raise ValueError("a fixed transverse DOF cannot carry a foundation spring")
    pen = torch.clamp(u[wd] - g, min=0.0)
    sep = torch.clamp(g - u[wd], min=0.0)
    res = unilateral_residuals(K, f, u, kn, g, fixed=fixed, kind="tensionless")
    return _result(model, asm, u, f, fixed, "tensionless",
                   gap=g, active=active, contact=active,
                   iterations=it, converged=True, contact_history=history,
                   penetration=pen, separation=sep,
                   reaction=R, reaction_nodal=R,
                   spring_forces=kn * pen,
                   residuals=res)


# ---------------------------------------------------------------------------
# unilateral: rigid bed (Signorini)
# ---------------------------------------------------------------------------
def solve_rigid_bed(model: BeamFoundationModel, nodal_loads=None, fixed_dofs=None,
                    gap=0.0, max_iter: int = 1500, tol: float = 1e-6,
                    init_active=None) -> dict:
    """Solution of a beam resting on a *rigid* one-sided bed (Signorini problem).

    The constraint is ``w_i <= g_i`` (no penetration into the bed), the bed
    reaction ``R_i >= 0`` is the multiplier and ``y_i = g_i - w_i >= 0`` the gap,
    with the complementarity statement ``R_i y_i = 0``.  This is the
    ``k -> infinity`` limit of :func:`solve_tensionless` and is used as its
    degeneracy check.

    Method: the problem is solved *dually*.  With ``A = S K^-1 S^T`` (the
    flexibility of the bed transverse DOFs) and ``b = S K^-1 f - g``, the bed
    reaction solves the non-negative quadratic program
    ``min 1/2 lam^T A lam - b^T lam, lam >= 0``; it is minimised by an
    accelerated projected-gradient (FISTA) iteration, whose solution identifies
    the contact set.  One primal equality solve on that set then yields the
    exact solution, so the reported no-penetration and complementarity
    residuals are limited by round-off only.  ``init_active`` (a boolean contact
    mask, e.g. from a large-``k`` elastic solution) warm-starts a final
    active-set polish when given.
    """
    asm = assemble_beam_foundation(model)
    f = _total_load(asm, nodal_loads)
    fixed = _dof_list(fixed_dofs, model.ndof)
    n = model.n_nodes
    wd = _w_dofs(n)
    g = _as_gap(gap, n)
    K = asm["K"]

    free = torch.tensor([d for d in range(model.ndof) if d not in set(fixed.tolist())],
                        dtype=torch.long)
    if free.numel() == 0:
        raise ValueError("all DOFs are fixed")
    pos = {int(d): j for j, d in enumerate(free.tolist())}
    bed = [i for i in range(n) if int(wd[i]) in pos]
    if not bed:
        raise ValueError("no free transverse DOF is available for the bed")
    Kr = K[free[:, None], free]
    try:
        Kr_inv_f = torch.linalg.solve(Kr, f[free])
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "the rigid-bed dual needs an invertible beam stiffness: support the "
            "beam (fixed_dofs) or use solve_tensionless for an unsupported beam"
        ) from exc

    nb = len(bed)
    S = torch.zeros((nb, int(free.numel())), dtype=FLOAT)
    for r, i in enumerate(bed):
        S[r, pos[int(wd[i])]] = 1.0
    Y = torch.linalg.solve(Kr, S.T)
    A = S @ Y
    b = S @ Kr_inv_f - g[bed]

    # ---- accelerated projected gradient on the dual QP
    if nb <= 800:
        lip = float(torch.linalg.eigvalsh(A).max())
    else:  # pragma: no cover - large meshes
        v = torch.ones(nb, dtype=FLOAT)
        for _ in range(200):
            v = A @ v
            v = v / torch.linalg.norm(v)
        lip = float(v @ (A @ v))
    lip = max(lip, 1e-30)
    lam = torch.zeros(nb, dtype=FLOAT)
    yk = lam.clone()
    tk = 1.0
    it = 0
    for it in range(1, max_iter + 1):
        grad = A @ yk - b
        lam_new = torch.clamp(yk - grad / lip, min=0.0)
        step = float((lam_new - lam).abs().max())
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * tk * tk) ** 0.5)
        yk = lam_new + ((tk - 1.0) / t_new) * (lam_new - lam)
        tk = t_new
        lam = lam_new
        if step <= tol * max(float(lam.abs().max()), 1e-30):
            break

    # ---- contact set from the dual, then one exact primal solve
    lam_scale = max(float(lam.abs().max()), 1e-30)
    active = torch.zeros(n, dtype=torch.bool)
    for r, i in enumerate(bed):
        if float(lam[r]) > 1e-9 * lam_scale:
            active[i] = True
    if init_active is not None:
        active = torch.as_tensor(init_active, dtype=torch.bool).reshape(-1).clone()

    polish_hist: list[int] = []
    converged = False
    u = torch.zeros(model.ndof, dtype=FLOAT)
    force_scale = max(float(f.abs().max()), 1e-30)
    for pol in range(1, 21):
        wc = wd[active]
        rhs = f.clone()
        if wc.numel():
            rhs = rhs - K[:, wc] @ g[active]
        u = torch.zeros(model.ndof, dtype=FLOAT)
        u[wc] = g[active]
        f2 = torch.tensor([d for d in range(model.ndof)
                           if d not in set(torch.cat([fixed, wc]).tolist())], dtype=torch.long)
        if f2.numel():
            u[f2] = torch.linalg.solve(K[f2[:, None], f2], rhs[f2])
        polish_hist.append(int(active.sum()))
        R_all = f[wd] - (K @ u)[wd]
        bed_nodes = [i for i in bed]
        pen_max, pen_arg = -1e30, None
        for i in bed_nodes:
            if not bool(active[i]):
                pen = float(u[wd[i]] - g[i])
                if pen > tol * max(float(u[wd].abs().max()), 1e-30):
                    if pen > pen_max:
                        pen_max, pen_arg = pen, i
        ten_min, ten_arg = 1e30, None
        for i in bed_nodes:
            if bool(active[i]):
                r = float(R_all[i])
                if r < -tol * force_scale and r < ten_min:
                    ten_min, ten_arg = r, i
        if pen_arg is None and ten_arg is None:
            converged = True
            break
        if pen_arg is not None and ten_arg is None:
            active[pen_arg] = True
        elif ten_arg is not None and pen_arg is None:
            active[ten_arg] = False
        else:
            active[pen_arg] = True
            active[ten_arg] = False
    if not converged:
        raise RuntimeError(
            f"rigid-bed polish did not reach complementarity in 20 steps "
            f"(contact counts {polish_hist})"
        )

    R = f[wd] - (K @ u)[wd]
    if fixed.numel():  # a clamped node carries a support reaction, not a bed one
        R[torch.tensor([int(d) // 2 for d in fixed.tolist() if int(d) % 2 == 0],
                       dtype=torch.long)] = 0.0
    sep = torch.clamp(g - u[wd], min=0.0)
    res = unilateral_residuals(K, f, u, None, g, fixed=fixed, kind="rigid")
    return _result(model, asm, u, f, fixed, "rigid",
                   gap=g, active=active, contact=active,
                   iterations=it + pol, dual_iterations=it, polish_iterations=pol,
                   converged=True, contact_history=polish_hist,
                   penetration=torch.zeros(n, dtype=FLOAT), separation=sep,
                   reaction=R, reaction_nodal=R, bed_nodes=torch.tensor(bed),
                   residuals=res)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def unilateral_residuals(K: torch.Tensor, f: torch.Tensor, w: torch.Tensor,
                         k_node: torch.Tensor | None, gap: torch.Tensor,
                         fixed: torch.Tensor | None = None,
                         kind: str = "tensionless",
                         w_dofs: torch.Tensor | None = None,
                         sign: float = 1.0) -> dict:
    """Pointwise KKT residuals of the one-sided solution ``w``.

    Returns a dict with

    ``stationarity``     ``max_i |(K w - f)_i| / f_scale`` over free DOFs,
    ``complementarity``  ``max_i |R_i y_i| / (f_scale * w_scale)``,
    ``reaction_min``     ``min_i R_i / f_scale`` (must be >= -tol),
    ``separation_min``   ``min_i y_i / w_scale`` (must be >= -tol),
    ``spring_law``       ``max_i |R_i - k_i (w_i - g_i)^+| / f_scale``,
    ``no_penetration``   ``max_i (w_i - g_i) / w_scale`` (rigid bed only),
    ``f_scale``, ``w_scale``.
    """
    n = int(gap.numel())
    wd = _w_dofs(n) if w_dofs is None else torch.as_tensor(w_dofs, dtype=torch.long)
    fixed = torch.zeros(0, dtype=torch.long) if fixed is None else fixed
    free = torch.tensor([d for d in range(int(f.numel())) if d not in set(fixed.tolist())],
                        dtype=torch.long)
    f_scale = max(float(f.abs().max()), 1e-30)
    w_scale = max(float(w[wd].abs().max()), 1e-30)
    R = f[wd] - (K @ w)[wd]
    # sign = +1 for a downward-positive beam, -1 for an upward-positive plate
    y = torch.clamp(-sign * (w[wd] - gap), min=0.0)
    pen = torch.clamp(sign * (w[wd] - gap), min=0.0)
    # supported transverse DOFs carry the support reaction, not a bed reaction
    _fixed = set(fixed.tolist())
    clamped = torch.tensor([i for i, d in enumerate(wd.tolist()) if int(d) in _fixed],
                           dtype=torch.long)
    if clamped.numel():
        R = R.clone()
        y = y.clone()
        R[clamped] = 0.0
        y[clamped] = 0.0
        pen[clamped] = 0.0
    if kind == "tensionless":
        # equilibrium must hold *including* the spring forces
        S = torch.zeros_like(f)
        if k_node is not None:
            S[wd] = k_node * pen
        resid = K @ w + S - f
        check = free
    else:
        # for a rigid bed the w-rows *define* the reaction, so stationarity is
        # checked on the remaining (rotation) rows
        resid = K @ w - f
        check = torch.tensor([d for d in free.tolist() if int(d) not in set(wd.tolist())],
                             dtype=torch.long)
    stationarity = float(resid[check].abs().max()) / f_scale if check.numel() else 0.0
    out = {
        "stationarity": stationarity,
        "complementarity": float((R * y).abs().max()) / (f_scale * w_scale),
        "reaction_min": float(R.min()) / f_scale,
        "separation_min": float(y.min()) / w_scale,
        "no_penetration": float(pen.max()) / w_scale,
        "f_scale": f_scale,
        "w_scale": w_scale,
    }
    if k_node is not None:
        out["spring_law"] = float((R - k_node * pen).abs().max()) / f_scale
    return out


def bending_moments(model: BeamFoundationModel, w: torch.Tensor) -> torch.Tensor:
    """Element end moments ``M = -EI kappa`` of shape ``(n-1, 2)`` (sagging +)."""
    asm = assemble_beam_foundation(model)
    dofs = asm["dofs"]
    out = torch.zeros((model.n_nodes - 1, 2), dtype=FLOAT)
    for e in range(model.n_nodes - 1):
        ue = w[dofs[e]]
        if model.element == "eb":
            B = hermite_second_derivatives(torch.tensor([0.0, 1.0], dtype=FLOAT),
                                           float(asm["lengths"][e]))
            out[e] = -model.E * model.Iz * (ue @ B)
        else:
            L = float(asm["lengths"][e])
            out[e] = -model.E * model.Iz * (ue[3] - ue[1]) / L * torch.tensor([1.0, 1.0],
                                                                            dtype=FLOAT)
    return out


def element_end_moments(model: BeamFoundationModel, w: torch.Tensor) -> torch.Tensor:
    """Alias of :func:`bending_moments` (element end moments ``(n-1, 2)``)."""
    return bending_moments(model, w)


def energy_ledger(model: BeamFoundationModel, w: torch.Tensor, nodal_loads=None,
                  gap=0.0, rigid: bool = False) -> dict:
    """Energy ledger ``U_beam + U_foundation = W_ext / 2`` for a one-sided solve.

    The identity holds exactly for a homogeneous quadratic problem with
    ``g = 0``; the returned ``relative_residual`` is the normalised violation.
    """
    asm = assemble_beam_foundation(model)
    f = _total_load(asm, nodal_loads)
    K = asm["K"]
    n = model.n_nodes
    wd = _w_dofs(n)
    g = _as_gap(gap, n)
    pen = torch.clamp(w[wd] - g, min=0.0)
    strain_beam = 0.5 * float(w @ (K @ w))
    strain_found = 0.0 if rigid else 0.5 * float(torch.sum(asm["k_node"] * pen**2))
    external = float(f @ w)
    half = 0.5 * external
    denom = max(abs(half), 1e-30)
    return {
        "strain_beam": strain_beam,
        "strain_foundation": strain_found,
        "external_work": external,
        "half_external_work": half,
        "relative_residual": abs(strain_beam + strain_found - half) / denom,
        "total_strain": strain_beam + strain_found,
    }


def lift_off_extent(model: BeamFoundationModel, w: torch.Tensor, gap=0.0) -> float:
    """Length of the separated region adjacent to node 0 (0.0 if none).

    The ``w = g`` crossing between the last separated node and the first
    contacted node is linearly interpolated, so the estimate is continuous in
    the mesh.
    """
    n = model.n_nodes
    x = model.x
    g = _as_gap(gap, n)
    w = torch.as_tensor(w, dtype=FLOAT).reshape(-1)
    wn = w if w.numel() == n else w[_w_dofs(n)]
    d = g - wn
    if float(d[0]) <= 0.0:
        return 0.0
    for j in range(1, n):
        if float(d[j]) <= 0.0:
            x0, x1 = float(x[j - 1]), float(x[j])
            d0, d1 = float(d[j - 1]), float(d[j])
            xc = x0 + (x1 - x0) * d0 / (d0 - d1)
            return xc - float(x[0])
    return float(x[-1] - x[0])


# ---------------------------------------------------------------------------
# Hetenyi closed forms
# ---------------------------------------------------------------------------
def characteristic_wavenumber(k: float, EI: float) -> float:
    """``beta = (k / 4EI)^(1/4)`` (1/length); the decay length is ``1/beta``."""
    if k <= 0.0 or EI <= 0.0:
        raise ValueError("k and EI must be positive")
    return (k / (4.0 * EI)) ** 0.25


def hetenyi_point_load_deflection(x_eval, x_load: float, P: float, k: float,
                                  EI: float) -> torch.Tensor:
    """Hetenyi deflection ``w = P beta / (2k) A(beta |x - x_load|)`` (downward +)."""
    from .hetenyi_reference import infinite_beam_point_load

    beta = characteristic_wavenumber(k, EI)
    return infinite_beam_point_load(torch.as_tensor(x_eval, dtype=FLOAT), x_load, P, beta, k)


def hetenyi_penetration(P: float, k: float, EI: float) -> float:
    """Maximum penetration of an infinite beam under a point load, ``P beta/(2k)``."""
    beta = characteristic_wavenumber(k, EI)
    return P * beta / (2.0 * k)


def lift_off_length_analytic(k: float, EI: float, q: float, P: float) -> float:
    """Lift-off half-length of an infinite beam under ``q`` (down) + ``P`` (up).

    The beam separates from the foundation over ``|x| < a``; in that free span
    ``EI w'''' = q`` and for ``|x| > a`` the bilateral Hetenyi solution holds.
    Continuity of ``w, w', M, V`` at ``x = a`` (no concentrated reaction there)
    reduces to one scalar equation for ``a``, solved here by bisection:

        m(a) = -1 - 2 beta a + beta P / q
        A(a) = 2 m q beta^2 / k - (q a^2 / 2 - P a / 2) / EI
        f(a) = (-q a^3 / 3 + P a^2 / 4) / EI + 2 m q beta^2 a / k
               - (q beta / k) (1 - m) = 0

    ``q = 0`` is a degenerate case (any lift-off length is admissible: the beam
    may rest on the foundation with zero pressure), so a positive ``q`` is
    required.
    """
    if q <= 0.0 or P <= 0.0:
        raise ValueError("a positive uniform load q and uplift P are required")
    beta = characteristic_wavenumber(k, EI)

    def m_of(a):
        return -1.0 - 2.0 * beta * a + beta * P / q

    def resid(a):
        m = m_of(a)
        return ((-q * a**3 / 3.0 + P * a**2 / 4.0) / EI + 2.0 * m * q * beta**2 * a / k
                - (q * beta / k) * (1.0 - m))

    grid = torch.linspace(0.0, 20.0 / beta, 4001, dtype=FLOAT)
    vals = [resid(float(a)) for a in grid]
    root = None
    for i in range(len(vals) - 1):
        if vals[i] == 0.0:
            root = float(grid[i])
            break
        if vals[i] * vals[i + 1] < 0.0:
            lo, hi = float(grid[i]), float(grid[i + 1])
            flo = vals[i]
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fm = resid(mid)
                if flo * fm <= 0.0:
                    hi = mid
                else:
                    lo, flo = mid, fm
            root = 0.5 * (lo + hi)
            break
    if root is None:
        raise ValueError(
            "no lift-off solution for this (k, EI, q, P); the uplift is too small "
            "to separate the beam"
        )
    return root


# ---------------------------------------------------------------------------
# plate extension: Q4 Mindlin + lumped surface springs + unilateral contact
# ---------------------------------------------------------------------------
def q4_winkler_nodal_springs(xy: torch.Tensor, k: float) -> torch.Tensor:
    """Lumped nodal surface springs of a Q4 element, ``k A_e / 4`` each."""
    if tuple(xy.shape) != (4, 2):
        raise ValueError("xy must have shape (4, 2)")
    area = 0.5 * float(
        (xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1])
        - (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1])
        + (xy[2, 0] - xy[0, 0]) * (xy[3, 1] - xy[0, 1])
        - (xy[3, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1])
    )
    return torch.full((4,), k * abs(area) / 4.0, dtype=FLOAT)


def assemble_plate_foundation(nodes: torch.Tensor, elements: torch.Tensor,
                              young: float, poisson: float, thickness: float,
                              k: float = 0.0, shear_treatment: str = "sri") -> dict:
    """Assemble the Q4 Mindlin plate stiffness and its Winkler contributions."""
    nodes = torch.as_tensor(nodes, dtype=FLOAT)
    elements = torch.as_tensor(elements, dtype=torch.long)
    n = int(nodes.shape[0])
    K = torch.zeros((3 * n, 3 * n), dtype=FLOAT)
    Kw = torch.zeros_like(K)
    k_node = torch.zeros(n, dtype=FLOAT)
    for conn in elements:
        xy = nodes[conn, :2]
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += q4_mindlin_stiffness(
            xy, young, poisson, thickness, shear_treatment=shear_treatment
        )
        if k != 0.0:
            Kw[dofs[:, None], dofs] += q4_winkler_foundation_stiffness(xy, k)
            k_node[conn] += q4_winkler_nodal_springs(xy, k)
    return {"nodes": nodes, "elements": elements, "K": K, "Kw": Kw, "k_node": k_node}


def solve_plate_foundation(nodes: torch.Tensor, elements: torch.Tensor,
                           young: float, poisson: float, thickness: float,
                           k: float, nodal_loads=None, fixed_dofs=None,
                           gap=0.0, unilateral: bool = True,
                           max_iter: int = 30, tol: float = 1e-12,
                           shear_treatment: str = "sri") -> dict:
    """Q4 Mindlin plate on a Winkler foundation, bilateral or one-sided.

    ``unilateral=True`` switches the lumped surface springs off at every node
    that lifts off (``w_i < g_i``), reusing the same active-set logic as the
    beam solver; ``unilateral=False`` reproduces the bilateral consistent
    foundation matrix of :func:`tensorfem.winkler_foundation`.

    Note: the historical ``shear_treatment="sri"`` Q4 element leaves rank-3
    hourglass modes in ``w`` that a *partially* engaged one-sided foundation
    cannot suppress (the system becomes singular once a region lifts off); use
    ``shear_treatment="mitc"`` for unilateral plate runs.
    """
    asm = assemble_plate_foundation(nodes, elements, young, poisson, thickness, k,
                                    shear_treatment=shear_treatment)
    n = int(asm["nodes"].shape[0])
    K = asm["K"]
    f = torch.zeros(3 * n, dtype=FLOAT)
    if nodal_loads is not None:
        nl = torch.as_tensor(nodal_loads, dtype=FLOAT).reshape(-1)
        if nl.numel() != 3 * n:
            raise ValueError(f"nodal_loads must have length {3 * n}")
        f = nl.clone()
    fixed = _dof_list(fixed_dofs, 3 * n)
    wd = 3 * torch.arange(n)
    g = _as_gap(gap, n)
    kn = asm["k_node"]

    if not unilateral:
        Kt = K + asm["Kw"]
        w = _solve_reduced(Kt, f, fixed)
        R = (asm["Kw"] @ w)[wd]
        return {"nodes": asm["nodes"], "elements": asm["elements"], "w": w,
                "w_nodes": w[wd], "reaction": R, "iterations": 0,
                "converged": True, "active": torch.ones(n, dtype=torch.bool),
                "penetration": torch.clamp(w[wd] - g, min=0.0),
                "separation": torch.clamp(g - w[wd], min=0.0),
                "kind": "plate-bilateral"}

    sign = -1.0  # Q4 Mindlin w is positive *upward*: contact when w < g
    active = torch.zeros(n, dtype=torch.bool)
    try:  # "no spring" predictor; singular (unsupported plate) -> full contact
        _solve_reduced(K, f, fixed)
    except RuntimeError:
        active = torch.ones(n, dtype=torch.bool)
    history: list[int] = []
    converged = False
    u = torch.zeros(3 * n, dtype=FLOAT)
    for it in range(1, max_iter + 1):
        Kt = K.clone()
        Kt[wd, wd] += torch.where(active, kn, torch.zeros_like(kn))
        rhs = f.clone()
        if bool(active.any()):
            rhs[wd] += torch.where(active, kn * g, torch.zeros_like(kn))
        u = _solve_reduced(Kt, rhs, fixed)
        w_scale = max(float(u[wd].abs().max()), 1e-30)
        new_active = (g - u[wd]) > tol * w_scale
        if not bool(new_active.any()):
            new_active[int(torch.argmax(g - u[wd]))] = True
        history.append(int(new_active.sum()))
        if bool(torch.equal(new_active, active)):
            converged = True
            break
        active = new_active
    if not converged:
        raise RuntimeError(
            f"plate unilateral active set did not converge in {max_iter} iterations "
            f"(contact counts {history})"
        )
    R = f[wd] - (K @ u)[wd]
    res = unilateral_residuals(K, f, u, kn, g, fixed=fixed, kind="tensionless",
                               w_dofs=wd, sign=sign)
    return {"nodes": asm["nodes"], "elements": asm["elements"], "w": u,
            "w_nodes": u[wd], "reaction": R, "iterations": it, "converged": True,
            "active": active, "gap": g,
            "penetration": torch.clamp(g - u[wd], min=0.0),
            "separation": torch.clamp(u[wd] - g, min=0.0),
            "residuals": res, "kind": "plate-unilateral"}


def evaluate_field(model: BeamFoundationModel, w: torch.Tensor, x_eval) -> torch.Tensor:
    """Evaluate the finite-element field ``w(x)`` at arbitrary ``x_eval``.

    Uses the element's own interpolation (cubic Hermite or linear), so an L2
    error computed from it is the true FE error rather than a post-processing
    (e.g. linear-interpolation) artefact.
    """
    x = model.x
    xe = torch.as_tensor(x_eval, dtype=FLOAT).reshape(-1)
    xc = xe.clamp(float(x[0]), float(x[-1]))
    idx = (torch.searchsorted(x, xc, right=True) - 1).clamp(0, model.n_nodes - 2)
    L = x[idx + 1] - x[idx]
    xi = (xc - x[idx]) / L
    zero = torch.zeros_like(xi)
    if model.element == "eb":
        N = torch.stack(
            (1.0 - 3.0 * xi**2 + 2.0 * xi**3,
             L * (xi - 2.0 * xi**2 + xi**3),
             3.0 * xi**2 - 2.0 * xi**3,
             L * (-xi**2 + xi**3)),
            dim=1,
        )
    else:
        N = torch.stack((1.0 - xi, zero, xi, zero), dim=1)
    dofs = torch.stack((2 * idx, 2 * idx + 1, 2 * idx + 2, 2 * idx + 3), dim=1)
    return (N * w[dofs]).sum(dim=1)


def l2_error(model: BeamFoundationModel, w: torch.Tensor, reference, *,
             order: int = 4) -> dict:
    """Relative ``L2`` error of the FE field against a callable reference.

    ``reference(x)`` must return the exact deflection.  The integral is
    evaluated with ``order``-point Gauss quadrature per element.
    """
    xi, wq = _gauss_unit(order)
    num = 0.0
    den = 0.0
    for e in range(model.n_nodes - 1):
        L = float(model.x[e + 1] - model.x[e])
        xe = float(model.x[e]) + xi.to(FLOAT) * L
        ref = torch.as_tensor(reference(xe), dtype=FLOAT).reshape(-1)
        fe = evaluate_field(model, w, xe)
        num += float(torch.sum(wq * (fe - ref) ** 2)) * L
        den += float(torch.sum(wq * ref**2)) * L
    return {"l2_abs": num**0.5, "l2_ref": den**0.5,
            "l2_relative": (num / den) ** 0.5 if den > 0 else float("inf")}
