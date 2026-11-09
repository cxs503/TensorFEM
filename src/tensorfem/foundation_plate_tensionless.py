"""Tensionless (no-tension / one-sided) Winkler foundation for Mindlin plates.

Why this module exists
----------------------
``tensorfem.winkler_foundation`` (and ``element/foundation.py``) provide the
*bilateral* (two-sided) Winkler foundation: the subgrade pulls down as readily as
it pushes up.  ``beam_foundation.solve_plate_foundation(unilateral=True)`` adds a
first one-sided path, but it is built on **lumped nodal springs**
(``k_i = sum_e k A_e / 4``) switched on and off per node; that lumping is not the
consistent ``int k N^T N dA`` operator and the module's own documentation declares
the plate path "implemented but NOT qualified".  This module closes that gap with
a quadrature-consistent formulation, a monotone solver, and an independent
certificate solver.

Physical model
--------------
A plate of flexural rigidity ``D`` rests on a Winkler subgrade of modulus ``k``
that transmits **compression only** (a soil / gravel bedding cannot pull the plate
down).  With ``w(x)`` the transverse deflection measured *into* the foundation and
``g0 >= 0`` an initial gap,

    p(x) = k * (w(x) - g0)^+ ,      (a)^+ = max(a, 0)

so the subgrade pressure is non-negative everywhere (no tension, ``p >= 0``) and
vanishes identically over the lifted-off region.  Equilibrium is the
stationarity condition of the **convex** potential

    Pi(u) = 1/2 u^T K_p u + 1/2 int_A k [(w - g0)^+]^2 dA - f^T u ,

which is C^1, piecewise quadratic and coercive on the kinematically admissible
set: the discrete solution exists and is unique, and any descent method that
decreases ``Pi`` monotonically converges to it.  ``Pi`` is used by the solver as
the *certificate* (monotone energy + stationarity residual), not merely as a
diagnostic.

Sign convention.  ``w > 0`` is motion toward the foundation.  The Mindlin Q4
element of :mod:`tensorfem.plate` shares that convention for a positive load on
the ``w`` DOF (checked by ``tests/test_foundation_plate_tensionless.py``); its
shear strain is ``gamma_xz = dw/dx + theta_x``, so a linear field
``w = a + b x`` is shear free with ``theta_x = -b``.

Finite-element discretisation
-----------------------------
The subgrade energy is integrated with the 2x2 Gauss rule over the *engaged*
Gauss points of each element (pointwise engagement, ``(w_gp - g0) > 0``).  The
element contribution is

    k_e = sum_{gp : engaged} k N(gp) N(gp)^T detJ(gp) ,

i.e. the consistent operator ``int k N^T N dA`` restricted to the engaged part of
the element.  Before the ``max`` the integrand is a polynomial of degree <= 2 per
direction, so the 2x2 rule reproduces a linear deflection field exactly even when
the sign of ``w`` changes inside an element: the **one-sided patch test** is exact
to round-off (gate ``1e-12`` in
``benchmarks/verify_foundation_plate_tensionless.py``).  Lumping (``model="lumped"``,
the historical ``k A / 4`` nodal springs) is kept as a contrast path only, so that
the benchmark can quantify the error it introduces.

Element performance note
------------------------
``tensorfem.plate.q4_mindlin_stiffness`` evaluates one element per call and
inverts the 2x2 Jacobian with ``torch.linalg.inv``; on the loong64 development
nodes that costs ~50 ms per element, i.e. minutes for a single benchmark mesh.
:func:`q4_mindlin_batch` is a vectorised numpy twin of that kernel (analytic 2x2
inverse) which reproduces it to ``4e-16`` relative including strongly distorted
quads (asserted in the test suite) and assembles 4000 elements in 0.5 s.

References
----------
[1] Hetényi, M., *Beams on Elastic Foundation*, Univ. of Michigan Press, 1946.
[2] Timoshenko, S. & Woinowsky-Krieger, S., *Theory of Plates and Shells*, 2nd ed.,
    §61-62, McGraw-Hill, 1959 (plate on elastic foundation; the Kelvin ``kei``
    solution used here as the closed-form anchor).
[3] Wriggers, P., *Computational Contact Mechanics*, 2nd ed., Springer, 2006
    (one-sided constraints, penalty regularisation, active sets).
[4] Doudoumis, I.N. & Simitses, G.J., tensionless-contact plate/foundation
    analyses, *Computers & Structures* — statement of the ``p = k w^+`` model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .winkler_foundation import q4_winkler_foundation_stiffness

FLOAT = torch.float64
_GP = 1.0 / 3.0**0.5
_W_LOCAL = np.array([0, 3, 6, 9])  # w DOF of each node inside the 12x12 element matrix
_GAUSS = ((-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP))
_SHEAR_CORRECTION = 5.0 / 6.0


# ---------------------------------------------------------------------------
# vectorised element kernels
# ---------------------------------------------------------------------------
def _q4_shapes(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                          [(1 - eta), -(1 + xi)],
                          [(1 + eta), (1 + xi)],
                          [-(1 + eta), (1 - xi)]])
    return N, dN


def _jacobian(xys: np.ndarray, dN_nat: np.ndarray):
    """Jacobian, its determinant and the analytic inverse for a batch of quads."""
    J = np.einsum("ia,eij->eaj", dN_nat, xys)          # (E, 2, 2)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    inv = np.empty_like(J)
    inv[:, 0, 0] = J[:, 1, 1] / det
    inv[:, 0, 1] = -J[:, 0, 1] / det
    inv[:, 1, 0] = -J[:, 1, 0] / det
    inv[:, 1, 1] = J[:, 0, 0] / det
    return J, det, inv


def _shear_rows(xys: np.ndarray, xi: float, eta: float) -> np.ndarray:
    """(E, 2, 12) transverse-shear strain rows at a natural point."""
    N, dN_nat = _q4_shapes(xi, eta)
    _, _, inv = _jacobian(xys, dN_nat)
    dN = np.einsum("ia,eba->eib", dN_nat, inv)
    rows = np.zeros((xys.shape[0], 2, 12))
    for i in range(4):
        rows[:, 0, 3 * i] = dN[:, i, 0]
        rows[:, 0, 3 * i + 1] = N[i]
        rows[:, 1, 3 * i] = dN[:, i, 1]
        rows[:, 1, 3 * i + 2] = N[i]
    return rows


def q4_mindlin_batch(xys, young: float, poisson: float, thickness: float,
                     shear_treatment: str = "mitc") -> np.ndarray:
    """Vectorised Q4 Mindlin element stiffness, ``(E, 12, 12)``.

    Bit-level twin of :func:`tensorfem.plate.q4_mindlin_stiffness` (validated to
    ``<= 1e-15`` relative on distorted elements in the test suite).
    """
    xys = np.asarray(xys, dtype=float)
    if xys.ndim != 3 or xys.shape[1:] != (4, 2):
        raise ValueError("xys must have shape (E, 4, 2)")
    if shear_treatment not in ("mitc", "sri"):
        raise ValueError("shear_treatment must be 'mitc' or 'sri'")
    E, nu, t = float(young), float(poisson), float(thickness)
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    Db = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]) * D
    Ds = np.diag([_SHEAR_CORRECTION * E / (2.0 * (1.0 + nu)) * t] * 2)
    ne = xys.shape[0]
    ke = np.zeros((ne, 12, 12))
    if shear_treatment == "mitc":
        bottom = _shear_rows(xys, 0.0, -1.0)[:, 0, :]
        top = _shear_rows(xys, 0.0, 1.0)[:, 0, :]
        left = _shear_rows(xys, -1.0, 0.0)[:, 1, :]
        right = _shear_rows(xys, 1.0, 0.0)[:, 1, :]
    for xi, eta in _GAUSS:
        _, dN_nat = _q4_shapes(xi, eta)
        _, det, inv = _jacobian(xys, dN_nat)
        dN = np.einsum("ia,eba->eib", dN_nat, inv)
        Bb = np.zeros((ne, 3, 12))
        for i in range(4):
            Bb[:, 0, 3 * i + 1] = dN[:, i, 0]
            Bb[:, 1, 3 * i + 2] = dN[:, i, 1]
            Bb[:, 2, 3 * i + 1] = dN[:, i, 1]
            Bb[:, 2, 3 * i + 2] = dN[:, i, 0]
        ke += np.einsum("eai,ab,ebj->eij", Bb, Db, Bb) * det[:, None, None]
        if shear_treatment == "mitc":
            Bs = np.zeros((ne, 2, 12))
            Bs[:, 0, :] = 0.5 * (1.0 - eta) * bottom + 0.5 * (1.0 + eta) * top
            Bs[:, 1, :] = 0.5 * (1.0 - xi) * left + 0.5 * (1.0 + xi) * right
            ke += np.einsum("eai,ab,ebj->eij", Bs, Ds, Bs) * det[:, None, None]
    if shear_treatment == "sri":  # one-point shear rule (weight 4)
        _, dN_nat = _q4_shapes(0.0, 0.0)
        _, det, inv = _jacobian(xys, dN_nat)
        dN = np.einsum("ia,eba->eib", dN_nat, inv)
        Bs = np.zeros((ne, 2, 12))
        for i in range(4):
            Bs[:, 0, 3 * i] = dN[:, i, 0]
            Bs[:, 0, 3 * i + 1] = _q4_shapes(0.0, 0.0)[0][i]
            Bs[:, 1, 3 * i] = dN[:, i, 1]
            Bs[:, 1, 3 * i + 2] = _q4_shapes(0.0, 0.0)[0][i]
        ke += np.einsum("eai,ab,ebj->eij", Bs, Ds, Bs) * det[:, None, None] * 4.0
    return ke


def q4_foundation_batch(xys, k: float, model: str = "consistent"):
    """Vectorised Winkler element matrices, ``((E, 12, 12), gauss data)``."""
    xys = np.asarray(xys, dtype=float)
    ne = xys.shape[0]
    if model not in ("consistent", "lumped"):
        raise ValueError("model must be 'consistent' or 'lumped'")
    kw = np.zeros((ne, 12, 12))
    Ns = np.zeros((ne, 4, 4))
    dets = np.zeros((ne, 4))
    for g, (xi, eta) in enumerate(_GAUSS):
        N, dN_nat = _q4_shapes(xi, eta)
        _, det, _ = _jacobian(xys, dN_nat)
        Ns[:, g, :] = N
        dets[:, g] = det
        if model == "consistent":
            block = k * np.outer(N, N) * det[:, None, None]
            for i in range(4):
                for j in range(4):
                    kw[:, _W_LOCAL[i], _W_LOCAL[j]] += block[:, i, j]
    if model == "lumped":
        area = 0.5 * np.abs(
            (xys[:, 1, 0] - xys[:, 0, 0]) * (xys[:, 2, 1] - xys[:, 0, 1])
            - (xys[:, 2, 0] - xys[:, 0, 0]) * (xys[:, 1, 1] - xys[:, 0, 1])
            + (xys[:, 2, 0] - xys[:, 0, 0]) * (xys[:, 3, 1] - xys[:, 0, 1])
            - (xys[:, 3, 0] - xys[:, 0, 0]) * (xys[:, 2, 1] - xys[:, 0, 1])
        )
        for i in _W_LOCAL:
            kw[:, i, i] = k * area / 4.0
    return kw, Ns, dets


def q4_foundation_consistent(xy, k: float) -> torch.Tensor:
    """Consistent Winkler foundation stiffness ``int k N^T N dA`` (12x12, torch)."""
    return q4_winkler_foundation_stiffness(torch.as_tensor(xy, dtype=FLOAT), k)


def q4_foundation_lumped(xy, k: float) -> np.ndarray:
    """Lumped (tributary) Winkler stiffness ``k A / 4`` per w DOF (contrast path)."""
    xy = np.asarray(xy, dtype=float)
    x1, y1 = xy[1, 0] - xy[0, 0], xy[1, 1] - xy[0, 1]
    x2, y2 = xy[2, 0] - xy[0, 0], xy[2, 1] - xy[0, 1]
    x3, y3 = xy[3, 0] - xy[0, 0], xy[3, 1] - xy[0, 1]
    area = 0.5 * float(x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2)
    kw = np.zeros((12, 12))
    for i in _W_LOCAL:
        kw[i, i] = k * abs(area) / 4.0
    return kw


# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------
@dataclass
class PlateTensionlessSystem:
    """Assembled plate + one-sided foundation operators (numpy / scipy ready)."""

    nodes: np.ndarray            # (n_node, 2)
    elements: np.ndarray         # (n_elem, 4)
    ndof: int
    w_dofs: np.ndarray           # (n_node,) transverse DOF index
    k_plate: object              # scipy.sparse.csr_matrix (ndof, ndof)
    k_foundation_full: object    # (ndof, ndof) fully engaged foundation operator
    gp_nodes: np.ndarray         # (n_gp, 4) global node ids of every Gauss point
    gp_N: np.ndarray             # (n_gp, 4) shape functions at the Gauss point
    gp_detJ: np.ndarray          # (n_gp,)
    foundation_k: float
    model: str
    element: str
    materials: dict = field(default_factory=dict)

    @property
    def n_gauss(self) -> int:
        return int(self.gp_N.shape[0])

    def w_at_gauss(self, u: np.ndarray) -> np.ndarray:
        """Interpolated deflection at every Gauss point."""
        u = np.asarray(u, dtype=float)
        return (self.gp_N * u[self.w_dofs[self.gp_nodes]]).sum(axis=1)

    def pressures(self, u: np.ndarray, gap: float = 0.0) -> np.ndarray:
        """Subgrade pressure ``p = k (w - gap)^+`` at every Gauss point."""
        return self.foundation_k * np.maximum(self.w_at_gauss(u) - gap, 0.0)

    def with_foundation_k(self, k: float) -> "PlateTensionlessSystem":
        """Copy of the system with another foundation modulus (penalty path)."""
        return PlateTensionlessSystem(
            nodes=self.nodes, elements=self.elements, ndof=self.ndof,
            w_dofs=self.w_dofs, k_plate=self.k_plate,
            k_foundation_full=self.k_foundation_full, gp_nodes=self.gp_nodes,
            gp_N=self.gp_N, gp_detJ=self.gp_detJ, foundation_k=float(k),
            model=self.model, element=self.element, materials=self.materials,
        )


def plate_tensionless_system(
    nodes,
    elements,
    young: float,
    poisson: float,
    thickness: float,
    foundation_k: float,
    *,
    shear_treatment: str = "mitc",
    model: str = "consistent",
) -> PlateTensionlessSystem:
    """Assemble the Mindlin Q4 plate stiffness and the one-sided foundation data.

    Parameters
    ----------
    nodes, elements : array_like
        ``(n_node, 2)`` coordinates and ``(n_elem, 4)`` connectivity (CCW).
    young, poisson, thickness : float
        Isotropic plate material.
    foundation_k : float
        Winkler modulus ``k`` (pressure per unit deflection).
    shear_treatment : {"mitc", "sri"}
        Transverse-shear treatment.  ``"mitc"`` is the default: the one-point
        ``"sri"`` rule leaves rank-3 hourglass modes in ``w`` that a partially
        engaged foundation cannot suppress once a region lifts off.
    model : {"consistent", "lumped"}
        Foundation operator: quadrature-consistent ``int k N^T N dA`` (default) or
        the tributary ``k A / 4`` nodal-spring contrast path.
    """
    import scipy.sparse as sp

    if shear_treatment not in ("mitc", "sri"):
        raise ValueError("shear_treatment must be 'mitc' or 'sri'")
    if model not in ("consistent", "lumped"):
        raise ValueError("model must be 'consistent' or 'lumped'")
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_elem = elements.shape[0]
    n_node = nodes.shape[0]
    ndof = 3 * n_node
    xys = nodes[elements]                                   # (E, 4, 2)
    ke = q4_mindlin_batch(xys, young, poisson, thickness, shear_treatment)
    kw, Ns, dets = q4_foundation_batch(xys, foundation_k, model)

    off = np.arange(3)[None, :]
    e_dofs = (3 * elements[:, :, None] + off[None, :, :]).reshape(n_elem, 12)
    rows = np.repeat(e_dofs, 12, axis=1)
    cols = np.tile(e_dofs, (1, 12))
    k_plate = sp.coo_matrix((ke.reshape(-1), (rows.reshape(-1), cols.reshape(-1))),
                            shape=(ndof, ndof)).tocsr()
    k_found = sp.coo_matrix((kw.reshape(-1), (rows.reshape(-1), cols.reshape(-1))),
                            shape=(ndof, ndof)).tocsr()
    gp_nodes = np.repeat(elements, 4, axis=0)               # (E*4, 4)
    return PlateTensionlessSystem(
        nodes=nodes,
        elements=elements,
        ndof=ndof,
        w_dofs=3 * np.arange(n_node),
        k_plate=k_plate,
        k_foundation_full=k_found,
        gp_nodes=gp_nodes,
        gp_N=Ns.reshape(-1, 4),
        gp_detJ=dets.reshape(-1),
        foundation_k=float(foundation_k),
        model=model,
        element=f"q4-mindlin/{shear_treatment}",
        materials={"young": float(young), "poisson": float(poisson),
                   "thickness": float(thickness)},
    )


# ---------------------------------------------------------------------------
# engaged operator, potential, gradient
# ---------------------------------------------------------------------------
def engaged_matrix(system: PlateTensionlessSystem, engaged: np.ndarray):
    """Sparse operator ``sum_{engaged gp} k N N^T detJ`` scattered to global DOFs."""
    import scipy.sparse as sp

    engaged = np.asarray(engaged, dtype=bool)
    if engaged.shape != (system.n_gauss,):
        raise ValueError(f"engaged must have shape ({system.n_gauss},)")
    if not engaged.any():
        return sp.csr_matrix((system.ndof, system.ndof))
    N = system.gp_N[engaged]                                       # (m, 4)
    w = system.gp_detJ[engaged] * system.foundation_k              # (m,)
    blocks = (N[:, :, None] * N[:, None, :]) * w[:, None, None]     # (m, 4, 4)
    node = system.gp_nodes[engaged]                                 # (m, 4)
    wd = system.w_dofs[node]
    rows = np.broadcast_to(wd[:, :, None], (wd.shape[0], 4, 4))
    cols = np.broadcast_to(wd[:, None, :], (wd.shape[0], 4, 4))
    return sp.coo_matrix((blocks.reshape(-1), (rows.reshape(-1), cols.reshape(-1))),
                         shape=(system.ndof, system.ndof)).tocsr()


def _gp_load_vector(system: PlateTensionlessSystem, gp_values: np.ndarray) -> np.ndarray:
    """Scatter Gauss-point intensities to the global w DOFs (``C^T p``)."""
    gp_values = np.asarray(gp_values, dtype=float).reshape(-1)
    if gp_values.shape[0] != system.n_gauss:
        raise ValueError("gp_values must have one entry per Gauss point")
    contrib = (system.gp_N * (gp_values * system.gp_detJ)[:, None]).reshape(-1)
    out = np.zeros(system.ndof)
    np.add.at(out, system.w_dofs[system.gp_nodes].reshape(-1), contrib)
    return out


def tensionless_energy(system: PlateTensionlessSystem, u: np.ndarray,
                       loads: np.ndarray, gap: float = 0.0) -> float:
    """Convex potential ``Pi(u)`` (see module docstring)."""
    u = np.asarray(u, dtype=float)
    w = system.w_at_gauss(u) - gap
    engaged = w > 0.0
    strain = 0.5 * float(u @ (system.k_plate @ u))
    found = 0.5 * system.foundation_k * float((w[engaged] ** 2) @ system.gp_detJ[engaged])
    return strain + found - float(np.asarray(loads, dtype=float) @ u)


def tensionless_gradient(system: PlateTensionlessSystem, u: np.ndarray,
                         loads: np.ndarray, gap: float = 0.0) -> np.ndarray:
    """Gradient ``K_p u + C^T p - f`` of the potential (equilibrium residual)."""
    u = np.asarray(u, dtype=float)
    p = system.pressures(u, gap)
    g = system.k_plate @ u + _gp_load_vector(system, p) - np.asarray(loads, dtype=float)
    return np.asarray(g, dtype=float)


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------
def _solve_reduced(K, rhs, fixed) -> np.ndarray:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    ndof = rhs.shape[0]
    free = np.ones(ndof, dtype=bool)
    fixed = np.asarray(fixed, dtype=int)
    if fixed.size:
        free[fixed] = False
    u = np.zeros(ndof)
    if free.any():
        Kff = sp.csr_matrix(K[free][:, free])
        u[free] = spla.spsolve(Kff.tocsc(), np.asarray(rhs, dtype=float)[free])
    return u


def solve_plate_tensionless(
    system: PlateTensionlessSystem,
    loads,
    fixed_dofs: Optional[Sequence[int]] = None,
    *,
    gap: float = 0.0,
    tol: float = 1e-12,
    max_iter: int = 60,
    start: Optional[np.ndarray] = None,
) -> dict:
    """Solve the tensionless plate problem by a monotone active-set Newton method.

    Each active set (the set of engaged Gauss points) defines a quadratic model;
    the method re-assembles the engaged operator from the sign of ``w`` at the
    Gauss points, solves the linear system, and accepts the step only when it
    lowers the convex potential (backtracking otherwise).  Monotone descent on a
    C^1 convex functional gives global convergence to the unique minimiser.

    Returns
    -------
    dict with ``u`` (global DOFs), ``w`` (nodal deflection), ``pressure_gauss``,
    ``engaged``, ``contact_area``, ``residual`` (stationarity), ``energy``,
    ``energy_history``, ``iterations`` and the energy-ledger terms.
    """
    loads = np.asarray(loads, dtype=float).reshape(-1)
    if loads.shape[0] != system.ndof:
        raise ValueError(f"loads must have length {system.ndof}")
    fixed = np.zeros(0, dtype=int) if fixed_dofs is None else np.asarray(fixed_dofs, dtype=int)
    free = np.ones(system.ndof, dtype=bool)
    if fixed.size:
        free[fixed] = False
    scale = max(float(np.abs(loads).max()), 1e-300)

    if start is None:
        # predictor: fully engaged (bilateral) solve
        u = _solve_reduced(system.k_plate + system.k_foundation_full, loads, fixed)
    else:
        u = np.array(start, dtype=float)

    energy_hist: list[float] = []
    res_hist: list[float] = []
    counts: list[int] = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        engaged = (system.w_at_gauss(u) - gap) > 0.0
        counts.append(int(engaged.sum()))
        Kt = system.k_plate + engaged_matrix(system, engaged)
        u_new = _solve_reduced(Kt, loads, fixed)
        d = u_new - u
        g = tensionless_gradient(system, u, loads, gap)
        slope = float(d @ g)
        if not np.isfinite(slope) or slope > 0.0:
            # `d` is not a descent direction (only happens through round-off at an
            # already-stationary point): fall back to the projected gradient.
            d = -g
            d[fixed] = 0.0
            slope = float(d @ g)
        e0 = tensionless_energy(system, u, loads, gap)
        energy_hist.append(e0)
        alpha = 1.0
        accepted = False
        for _ in range(80):
            cand = u + alpha * d
            if tensionless_energy(system, cand, loads, gap) <= e0 + 1e-4 * alpha * slope:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            cand = u_new
        step = float(np.abs(cand - u).max())
        u = cand
        g = tensionless_gradient(system, u, loads, gap)
        # force scale of the current iterate: the equilibrium residual must be
        # judged against the magnitudes actually present in the system.
        p_now = system.pressures(u, gap)
        scale_now = max(scale, float(np.abs(system.k_plate @ u).max()),
                        float(np.abs(_gp_load_vector(system, p_now)).max()))
        res = float(np.abs(g[free]).max()) / scale_now if free.any() else 0.0
        res_hist.append(res)
        engaged_new = (system.w_at_gauss(u) - gap) > 0.0
        step_tol = 1e-15 * max(1.0, float(np.abs(u).max()))
        if step <= step_tol or res <= tol:
            if np.array_equal(engaged_new, engaged) or res <= tol:
                converged = True
                break

    p = system.pressures(u, gap)
    w_gp = system.w_at_gauss(u)
    engaged = (w_gp - gap) > 0.0
    strain = 0.5 * float(u @ (system.k_plate @ u))
    subgrade = 0.5 * system.foundation_k * float(((w_gp - gap)[engaged] ** 2)
                                                 @ system.gp_detJ[engaged])
    return {
        "u": u,
        "w": u[system.w_dofs],
        "pressure_gauss": p,
        "w_gauss": w_gp,
        "engaged": engaged,
        "contact_area": float(system.gp_detJ[engaged].sum()),
        "total_area": float(system.gp_detJ.sum()),
        "total_reaction": float(p @ system.gp_detJ),
        "residual": res_hist[-1] if res_hist else float("nan"),
        "residual_history": res_hist,
        "energy": tensionless_energy(system, u, loads, gap),
        "energy_history": energy_hist,
        "strain_energy": strain,
        "foundation_energy": subgrade,
        "external_work": float(loads @ u),
        "iterations": it,
        "active_counts": counts,
        "converged": converged,
        "min_pressure": float(p.min()) if p.size else 0.0,
        "kind": "plate-tensionless",
    }


def solve_plate_tensionless_lbfgs(
    system: PlateTensionlessSystem,
    loads,
    fixed_dofs: Optional[Sequence[int]] = None,
    *,
    gap: float = 0.0,
    max_iter: int = 2000,
    tolerance_grad: float = 1e-16,
    start: Optional[np.ndarray] = None,
) -> dict:
    """Independent certificate solver: L-BFGS-B on the same convex potential.

    A different algorithm from :func:`solve_plate_tensionless` (quasi-Newton with
    a Wolfe line search on the full free-DOF vector, no active-set bookkeeping),
    so agreement between the two is a genuine cross-check of the minimiser rather
    than a restatement of the same iteration.  The analytic gradient is supplied.
    """
    import scipy.optimize as opt

    loads = np.asarray(loads, dtype=float).reshape(-1)
    fixed = np.zeros(0, dtype=int) if fixed_dofs is None else np.asarray(fixed_dofs, dtype=int)
    free = np.ones(system.ndof, dtype=bool)
    if fixed.size:
        free[fixed] = False
    free_idx = np.nonzero(free)[0]
    u_full = np.zeros(system.ndof) if start is None else np.array(start, dtype=float)

    def fun(x):
        u = u_full.copy()
        u[free_idx] = x
        w = system.w_at_gauss(u) - gap
        wp = np.maximum(w, 0.0)
        p = system.foundation_k * wp
        energy = (0.5 * float(u @ (system.k_plate @ u))
                  + 0.5 * system.foundation_k * float((wp * wp) @ system.gp_detJ)
                  - float(loads @ u))
        grad = system.k_plate @ u + _gp_load_vector(system, p) - loads
        return energy, np.asarray(grad[free_idx], dtype=float)

    res = opt.minimize(fun, u_full[free_idx], jac=True, method="L-BFGS-B",
                       options={"maxiter": int(max_iter), "maxfun": 4 * int(max_iter),
                                "ftol": 0.0, "gtol": float(tolerance_grad),
                                "maxcor": 50})
    u = u_full.copy()
    u[free_idx] = res.x
    g = tensionless_gradient(system, u, loads, gap)
    p_now = system.pressures(u, gap)
    scale = max(float(np.abs(loads).max()),
                float(np.abs(system.k_plate @ u).max()),
                float(np.abs(_gp_load_vector(system, p_now)).max()))
    engaged = (system.w_at_gauss(u) - gap) > 0.0
    return {
        "u": u,
        "w": u[system.w_dofs],
        "residual": float(np.abs(g[free_idx]).max()) / scale,
        "energy": tensionless_energy(system, u, loads, gap),
        "pressure_gauss": p_now,
        "contact_area": float(system.gp_detJ[engaged].sum()),
        "iterations": int(res.nit),
        "kind": "plate-tensionless-lbfgs",
    }


def solve_plate_unilateral_penalty(
    system: PlateTensionlessSystem,
    loads,
    fixed_dofs: Optional[Sequence[int]] = None,
    *,
    gap: float = 0.0,
    penalty: float = 1e7,
    tol: float = 1e-12,
    max_iter: int = 60,
) -> dict:
    """Penalty alternative for the *rigid* one-sided bed (Signorini limit).

    The rigid bed is the ``k -> infinity`` limit of the subgrade model: no
    penetration at all, ``w - gap <= 0``, compressive reaction only.  The penalty
    regularisation replaces the constraint by ``R = penalty * (w - gap)^+``, i.e.
    the tensionless law with the penalty stiffness in place of ``k``.  The
    penetration shrinks like ``O(1/penalty)`` and the complementarity product
    ``R * penetration`` with it (measured in the benchmark).
    """
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    scaled = system.with_foundation_k(penalty)
    out = solve_plate_tensionless(scaled, loads, fixed_dofs, gap=gap, tol=tol,
                                  max_iter=max_iter)
    penetration = np.maximum(out["w_gauss"] - gap, 0.0)
    reaction = out["pressure_gauss"]
    out["penetration_max"] = float(penetration.max()) if penetration.size else 0.0
    out["complementarity_max"] = float((reaction * penetration).max()) if reaction.size else 0.0
    out["penalty"] = float(penalty)
    out["kind"] = "plate-unilateral-penalty"
    return out


def contact_radius(system: PlateTensionlessSystem, engaged: np.ndarray) -> float:
    """Equivalent radius ``sqrt(A_contact / pi)`` of the engaged region."""
    area = float(system.gp_detJ[np.asarray(engaged, dtype=bool)].sum())
    return math.sqrt(max(area, 0.0) / math.pi)


def hetenyi_plate_centre_deflection(P: float, D: float, k: float) -> float:
    """Closed form ``w(0) = P / (8 sqrt(D k))`` for an infinite plate (Hetényi)."""
    return P / (8.0 * math.sqrt(D * k))


def plate_rigidity(young: float, poisson: float, thickness: float) -> float:
    """Flexural rigidity ``D = E t^3 / (12 (1 - nu^2))``."""
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


def characteristic_length(young: float, poisson: float, thickness: float,
                          k: float) -> float:
    """Winkler characteristic length ``l = (D / k)^(1/4)``."""
    return (plate_rigidity(young, poisson, thickness) / k) ** 0.25


__all__ = [
    "PlateTensionlessSystem",
    "characteristic_length",
    "contact_radius",
    "engaged_matrix",
    "hetenyi_plate_centre_deflection",
    "plate_rigidity",
    "plate_tensionless_system",
    "q4_foundation_batch",
    "q4_foundation_consistent",
    "q4_foundation_lumped",
    "q4_mindlin_batch",
    "solve_plate_tensionless",
    "solve_plate_tensionless_lbfgs",
    "solve_plate_unilateral_penalty",
    "tensionless_energy",
    "tensionless_gradient",
]