"""Differentiable D2Q9 lid-driven cavity solver (graph-preserving lid injection).

Anchor: ``GAP-PP-2-lid-wall-diff`` — 可微运动壁面 BC（graph-preserving lid 注入）.

This is the fluid core required by the JOUR-DIFFLBM E2 square-cavity inverse
experiment: a square cavity whose top lid moves with velocity ``u_lid`` while
the other three walls are at rest.  The solver is a from-scratch D2Q9 BGK
lattice-Boltzmann code (NumPy, float64) in which the *lattice graph is fixed*
and the lid motion enters **only** through the boundary closure parameter:

* the D2Q9 connectivity (population -> source-node map, bounce-back masks,
  wall-link masks) is built once from the geometry and never changes;
* the lid velocity ``u_lid`` is injected into the Ladd moving-wall half-way
  bounce-back closure

      f_i(x_f, t+1) = f_opp(i)(x_f, t) + 2 w_i rho (c_i . u_wall) / cs^2

  at the top-wall links, exactly as in :mod:`tensorfem.lbm_channel2d`
  (cross-validated in ``XVAL-LBM-MOVING-WALL-HALFWAY-BB-FLOAT64``).

Differentiability
-----------------
Forward-mode automatic differentiation is implemented with a minimal
dual-number field class :class:`Dual` (value + tangent, both float64
arrays).  Every operation of the time evolution (macroscopic moments,
equilibrium, BGK collision, streaming, wall closure, observable) is a
polynomial in the populations and the parameters, so the tangent propagates
by the product rule with no approximation.  A run seeded on ``tau`` returns
both the observable ``J`` and its exact derivative ``dJ/dtau``; seeded on
``u_lid`` it returns ``dJ/du_lid``.

The parameters are the relaxation time ``tau`` (viscosity
``nu = (tau - 1/2) / 3`` in lattice units) and the lid speed ``u_lid``.
Both are plain Python floats at the API level; the tangent seed is chosen by
the caller.

Unit system
-----------
Lattice units ``dx = dt = rho0 = 1``.  For a physical cavity of height
``H_phys`` and lid speed ``U_phys``:

    u_lid = U_phys * dt_phys / dx_phys
    nu_lat = nu_phys * dt_phys / dx_phys**2
    Re = U_phys * H_phys / nu_phys = u_lid * (ny - 2) / nu_lat

References
----------
[1] Ladd, A.J.C. (1994). Numerical simulations of particulate suspensions
    via a discretized Boltzmann equation. Part 1. Theoretical foundation.
    J. Fluid Mech. 271, 285-309. DOI: 10.1017/S0022112094001771
    (moving-wall bounce-back).  [REF-VERIFY: journal corrected from a
    previously-cited J. Supercomput entry, which does not exist.]
[2] Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for
    incompressible flow using the Navier-Stokes equations and a multigrid
    method. J. Comput. Phys. 48, 387-411. DOI: 10.1016/0021-9991(82)90058-4
    (lid-driven cavity benchmark, Re = 100).  [REF-VERIFY: verified.]
[3] Guo, Z., Zheng, C., Shi, B. (2002). Discrete lattice effects on the
    forcing term in the lattice Boltzmann method. Phys. Rev. E 65, 046308.
    DOI: 10.1103/PhysRevE.65.046308 (discrete-lattice-effect-corrected
    forcing, implemented by :mod:`tensorfem.lbm_channel2d` /
    :mod:`tensorfem.lbm_forcing`; this module itself carries no force term).
    [REF-VERIFY: re-cited — the earlier "Phys. Fluids" attribution mixed the
    title of this paper with the venue/volume/pages/DOI of the extrapolation
    boundary-condition paper, and that DOI is not registered; see
    docs/JOUR-DIFFLBM-REF-VERIFY-REPAIR.md.]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lbm_references import JOUR_DIFFLBM_CORE_REFS, render_reference_block

#: the citation block above, rendered from the verified registry — the
#: docstring must agree with it (tests/test_lbm_cavity_diff_ref_verify.py)
__references__ = render_reference_block(JOUR_DIFFLBM_CORE_REFS)

DTYPE = np.float64

# D2Q9 lattice velocities, weights, opposite map
CX = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1), dtype=np.int64)
CY = np.array((0, 0, 1, 0, -1, 1, 1, -1, -1), dtype=np.int64)
W9 = np.array((4.0 / 9.0,) + (1.0 / 9.0,) * 4 + (1.0 / 36.0,) * 4, dtype=DTYPE)
OPP = np.array((0, 3, 4, 1, 2, 7, 8, 5, 6), dtype=np.int64)
CS2 = 1.0 / 3.0

_W = W9.reshape(9, 1, 1)
_CX = CX.reshape(9, 1, 1).astype(DTYPE)
_CY = CY.reshape(9, 1, 1).astype(DTYPE)


# ---------------------------------------------------------------------------
# forward-mode AD field
# ---------------------------------------------------------------------------
class Dual:
    """A field carrying a value ``v`` and a tangent ``t`` (same shape).

    Supports the polynomial operations used by the LBM evolution:
    ``+ - * /`` with other ``Dual`` fields or plain scalars/arrays,
    axis-sums, and index shifts.  ``t`` is the directional derivative of
    ``v`` along the seeded parameter direction.
    """

    __slots__ = ("v", "t")

    def __init__(self, v: np.ndarray, t: np.ndarray | None = None):
        self.v = v
        self.t = np.zeros_like(v) if t is None else t

    # -- arithmetic ---------------------------------------------------------
    def __add__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v + o.v, self.t + o.t)
        return Dual(self.v + o, self.t)

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v - o.v, self.t - o.t)
        return Dual(self.v - o, self.t)

    def __rsub__(self, o):
        return Dual(o - self.v, -self.t)

    def __neg__(self):
        return Dual(-self.v, -self.t)

    def __mul__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v * o.v, self.t * o.v + self.v * o.t)
        return Dual(self.v * o, self.t * o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Dual):
            inv = 1.0 / o.v
            return Dual(self.v * inv,
                      (self.t * o.v - self.v * o.t) * inv * inv)
        return Dual(self.v / o, self.t / o)

    def __rtruediv__(self, o):
        inv = 1.0 / self.v
        return Dual(o * inv, -o * self.t * inv * inv)

    # -- reductions / indexing ---------------------------------------------
    def sum(self, axis: int | None = None) -> "Dual":
        return Dual(self.v.sum(axis=axis), self.t.sum(axis=axis))

    def shift(self, idx) -> "Dual":
        """Gather ``self`` from the precomputed flat source indices ``idx``."""
        return Dual(self.v.reshape(-1)[idx].reshape(self.v.shape),
                   self.t.reshape(-1)[idx].reshape(self.t.shape))

    def scalar(self) -> tuple[float, float]:
        return float(self.v), float(self.t)


def pop(d: Dual, i: int) -> Dual:
    """The i-th population of a (9, ny, nx) Dual field."""
    return Dual(d.v[i], d.t[i])


def where(mask: np.ndarray, a: Dual, b: Dual) -> Dual:
    """Elementwise select between two Dual fields by a plain boolean mask."""
    return Dual(np.where(mask, a.v, b.v), np.where(mask, a.t, b.t))


def _expand(d: Dual) -> Dual:
    """Add a leading axis: (ny, nx) -> (1, ny, nx)."""
    return Dual(d.v[None], d.t[None])


def equilibrium(rho: Dual, ux: Dual, uy: Dual) -> Dual:
    """D2Q9 equilibrium distribution (polynomial in the moments).

    Note: every binary op keeps the ``Dual`` on the left so that NumPy never
    sees a bare array on the left of a ``Dual`` (which would make it build an
    object array and dispatch per element).
    """
    ux1, uy1 = _expand(ux), _expand(uy)
    cu = ux1 * _CX + uy1 * _CY
    cu2 = cu * cu
    u2 = ux * ux + uy * uy
    poly = 1.0 + cu * 3.0 + cu2 * 4.5 - _expand(u2) * 1.5
    return _expand(rho) * _W * poly


# ---------------------------------------------------------------------------
# cavity solver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CavityConfig:
    """Square-cavity configuration (lattice units dx = dt = rho0 = 1).

    The fluid occupies nodes ``1..nx-2`` x ``1..ny-2``; the four boundary
    rows/columns are solid.  The top wall moves with ``u_lid`` in +x.
    """

    nx: int
    ny: int
    nu: float
    u_lid: float = 0.05
    steps: int = 1000

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def reynolds(self) -> float:
        return self.u_lid * (self.ny - 2) / self.nu


class CavityLBM:
    """Differentiable lid-driven cavity (D2Q9 BGK, half-way bounce-back).

    The *graph* of the lattice — shift index arrays, ``from_solid`` masks and
    the top-wall link masks — depends only on ``(nx, ny)``.  ``u_lid`` and
    ``tau`` are parameters of the evolution, not of the graph: changing them
    leaves every mask and index array bit-identical (see
    :meth:`graph_signature`).
    """

    def __init__(self, cfg: CavityConfig):
        self.cfg = cfg
        self.nx, self.ny = cfg.nx, cfg.ny
        self.solid = np.zeros((self.ny, self.nx), dtype=bool)
        self.solid[0, :] = True
        self.solid[-1, :] = True
        self.solid[:, 0] = True
        self.solid[:, -1] = True

        # source-node index for each population (out-of-range -> solid)
        self._src: list[np.ndarray] = [None] * 9
        self.from_solid: list[np.ndarray] = [None] * 9
        self.top_link: list[np.ndarray] = [None] * 9
        self.wall_out: list[np.ndarray] = [None] * 9
        jj, ii = np.meshgrid(np.arange(self.ny), np.arange(self.nx),
                             indexing="ij")
        for i in range(1, 9):
            sj, si = jj - CY[i], ii - CX[i]
            inside = (sj >= 0) & (sj < self.ny) & (si >= 0) & (si < self.nx)
            src = np.where(inside, sj * self.nx + si, 0)
            self._src[i] = src
            fs = ~inside | self.solid[sj.clip(0, self.ny - 1),
                                  si.clip(0, self.nx - 1)]
            self.from_solid[i] = fs
            # link whose source node sits on the top wall (moving lid):
            # populations entering the fluid from the lid
            tl = fs & (sj == self.ny - 1)
            self.top_link[i] = tl
            # link whose destination node sits on the top wall: populations
            # striking the lid (used by the momentum-exchange drag)
            dj, di = jj + CY[i], ii + CX[i]
            self.wall_out[i] = ((dj == self.ny - 1) & (di >= 0)
                                & (di < self.nx))

        self._f_post: Dual | None = None
        self.time = 0

    # -- graph introspection ------------------------------------------------
    def graph_signature(self) -> tuple:
        """Hashable signature of the lattice graph (geometry only).

        Two configurations with the same grid share the graph iff these
        signatures match; ``u_lid`` and ``tau`` must not appear here.
        """
        return (
            (self.nx, self.ny),
            tuple(s.tobytes() for s in self._src[1:]),
            tuple(m.tobytes() for m in self.from_solid[1:]),
            tuple(m.tobytes() for m in self.top_link[1:]),
            tuple(m.tobytes() for m in self.wall_out[1:]),
        )

    # -- fields -------------------------------------------------------------
    def _init(self) -> Dual:
        f = np.broadcast_to(W9.reshape(9, 1, 1), (9, self.ny, self.nx)).copy()
        return Dual(f)

    def _collide(self, f: Dual, inv_tau: Dual) -> Dual:
        rho = f.sum(axis=0)
        # solid nodes carry no populations; unit density keeps the moment
        # divisions finite (their output is zeroed below)
        rho = Dual(np.where(self.solid, 1.0, rho.v),
                  np.where(self.solid, 0.0, rho.t))
        ux = (f * _CX).sum(axis=0) / rho
        uy = (f * _CY).sum(axis=0) / rho
        feq = equilibrium(rho, ux, uy)
        out = f + inv_tau * (feq - f)
        out.v[:, self.solid] = 0.0
        out.t[:, self.solid] = 0.0
        return out

    def _stream(self, f_post: Dual, u_lid: Dual) -> Dual:
        f_new = [None] * 9
        f_new[0] = pop(f_post, 0)
        for i in range(1, 9):
            streamed = pop(f_post, i).shift(self._src[i])
            f_new[i] = where(self.from_solid[i], pop(f_post, OPP[i]),
                            streamed)
        # graph-preserving lid injection: only the top-wall links carry the
        # moving-wall correction; the graph itself is untouched
        rho = f_post.sum(axis=0)
        for i in (4, 7, 8):
            cu = CX[i] * u_lid
            if cu.v == 0.0 and cu.t == 0.0:
                continue
            corr = (2.0 * W9[i] / CS2) * rho * cu * self.top_link[i]
            f_new[i] = f_new[i] + corr
        out = Dual(np.stack([d.v for d in f_new]),
                   np.stack([d.t for d in f_new]))
        out.v[:, self.solid] = 0.0
        out.t[:, self.solid] = 0.0
        return out

    def step(self, f: Dual, inv_tau: Dual, u_lid: Dual) -> Dual:
        f_post = self._collide(f, inv_tau)
        self._f_post = f_post
        self.time += 1
        return self._stream(f_post, u_lid)

    # -- observables ---------------------------------------------------------
    def velocity(self, f: Dual) -> tuple[Dual, Dual, Dual]:
        rho = f.sum(axis=0)
        rho = Dual(np.where(self.solid, 1.0, rho.v),
                  np.where(self.solid, 0.0, rho.t))
        ux = (f * _CX).sum(axis=0) / rho
        uy = (f * _CY).sum(axis=0) / rho
        return rho, ux, uy

    def lid_drag(self, f_post: Dual, u_lid: Dual) -> Dual:
        """x-momentum handed to the lid per unit depth (momentum exchange).

        ``F_x = sum_i 2 c_ix (f_i* - w_i rho (c_i . u_w) / cs^2)`` summed over
        the populations ``i in {2,5,6}`` that strike the top wall.  Negative
        for a lid dragged along +x (the fluid pulls back).
        """
        rho = f_post.sum(axis=0)
        F = Dual(np.zeros((), dtype=DTYPE))
        for i in (2, 5, 6):
            cu = CX[i] * u_lid
            term = 2.0 * cu * (pop(f_post, i) - W9[i] * rho * (cu / CS2))
            F = F + term * self.wall_out[i]
        return F.sum()

    def _params(self, seed: str, tau: float | None, u_lid: float | None
                ) -> tuple[Dual, Dual]:
        tau = self.cfg.tau if tau is None else tau
        u = self.cfg.u_lid if u_lid is None else u_lid
        inv_tau = Dual(np.array(1.0 / tau),
                      np.array(-1.0 / tau ** 2) if seed == "tau" else None)
        ul = Dual(np.array(u), np.array(1.0) if seed == "u_lid" else None)
        return inv_tau, ul

    def run(self, seed: str = "tau", steps: int | None = None,
            tau: float | None = None, u_lid: float | None = None
            ) -> tuple[float, float]:
        """Run ``steps`` steps; return ``(J, dJ/dseed)`` for J = lid drag."""
        inv_tau, ul = self._params(seed, tau, u_lid)
        f = self._init()
        n = self.cfg.steps if steps is None else steps
        for _ in range(n):
            f = self.step(f, inv_tau, ul)
        J, dJ = self.lid_drag(self._f_post, ul).scalar()
        return J, dJ

    def run_velocity(self, seed: str = "tau", steps: int | None = None,
                     tau: float | None = None, u_lid: float | None = None
                   ) -> tuple[Dual, Dual, Dual]:
        """Run and return ``(rho, ux, uy)`` at the final step (Dual fields)."""
        inv_tau, ul = self._params(seed, tau, u_lid)
        f = self._init()
        n = self.cfg.steps if steps is None else steps
        for _ in range(n):
            f = self.step(f, inv_tau, ul)
        return self.velocity(f)
