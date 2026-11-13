"""Two-component Shan-Chen (MCMP) D2Q9 lattice-Boltzmann solver with a
barycentric mixture velocity and Guo forcing.

Why this module
---------------
``lbm_channel2d`` provides the single-component fluid used by the FSI
verification face.  Gas-layer drag reduction, cavitation inception and
gas/liquid mixing all need a *two-component* (multi-component multiphase,
MCMP) Shan-Chen model, and the established open-source implementations
(``lssfau/lbmpy`` ``tests/shan_chen/test_shan_chen_two_component.py``,
``CFD-GO/TCLB`` ``models/multiphase/experimental/d2q9_pp_MCMP``) all use the
same three ingredients that the naive "each component advects with its own
velocity" formulation is missing:

1. a **barycentric (mixture) velocity** shared by both components in the
   equilibrium distribution,
2. a **Guo (2002) second-order forcing** term with the ``(1 - 1/(2 tau))``
   discrete-lattice correction instead of a plain ``tau``-shift of the
   equilibrium velocity,
3. an **exponential pseudo-potential** ``psi(rho) = rho0 (1 - exp(-rho/rho0))``
   (Shan-Chen 1993 exponential form) that keeps the interaction force bounded
   at high density instead of the linear ``psi = rho``.

Model (lattice units, ``dx = dt = 1``, ``cs^2 = 1/3``)
-----------------------------------------------------
For each component ``s in {a, b}`` the populations ``f_i^s`` obey

    f_i^s(x + c_i, t+1) = f_i^s(x, t) - (f_i^s - f_i^{eq,s}) / tau_s + S_i^s

    f_i^{eq,s} = w_i rho_s [1 + 3 c_i.u_s^* + 4.5 (c_i.u_s^*)^2 - 1.5 u_s^*^2]
    u_s^*      = u' + tau_s F_s / rho_s                       (forced velocity)
    S_i^s      = (1 - 1/(2 tau_s)) w_i [ 3 (c_i - u').F_s + 9 (c_i.u')(c_i.F_s) ]

with the interaction force (Shan-Chen 1993, exponential pseudo-potential)

    F_s(x) = - psi_s(x) sum_{s'} g_{ss'} sum_i w_i c_i psi_{s'}(x + c_i)

and the **barycentric** mixture velocity built from the raw momenta ``m_s =
sum_i c_i f_i^s`` with the half-force correction

    u' = sum_s (m_s + F_s/2)/tau_s  /  sum_s rho_s/tau_s        (mode "tau")

``mode="mass"`` replaces ``1/tau_s`` by ``1`` (mass-weighted average, what the
lbmpy test calls ``u_bary``; identical to ``"tau"`` whenever
``tau_a == tau_b``) and ``mode="self"`` lets every component use its own
``(m_s + F_s/2)/rho_s`` -- the naive scheme that the cross-validation below
falsifies.

References
----------
[1] Shan, X., Chen, H. (1993). Phys. Rev. E 47, 1815 (pseudo-potential).
[2] Shan, X., Doolen, G. (1996). J. Stat. Phys. 81, 379 (MCMP, barycentric u).
[3] Guo, Z., Zheng, C., Shi, B. (2002). Phys. Fluids 14, 2007 (forcing).
[4] Sukop, M.C., Thorne, D.T. (2006). Lattice Boltzmann Modeling, ch. 9.
[5] Krueger et al. (2017). The Lattice Boltzmann Method, sec. 9.3 (MCMP).
[6] lssfau/lbmpy tests/shan_chen/test_shan_chen_two_component.py -- the
    open-source reference that ``examples/lbm_shan_chen_mcmp_barycentric_guo.py``
    cross-validates against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DTYPE = np.float64

# D2Q9 lattice (identical ordering to lbm_channel2d)
CX = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1), dtype=np.int64)
CY = np.array((0, 0, 1, 0, -1, 1, 1, -1, -1), dtype=np.int64)
W9 = np.array((4.0 / 9.0,) + (1.0 / 9.0,) * 4 + (1.0 / 36.0,) * 4, dtype=DTYPE)
CS2 = 1.0 / 3.0

_W = W9.reshape(9, 1, 1)
_CX = CX.reshape(9, 1, 1).astype(DTYPE)
_CY = CY.reshape(9, 1, 1).astype(DTYPE)


def _shift_periodic(t: np.ndarray, i: int) -> np.ndarray:
    """Stream one population by one cell, periodic in both directions."""
    return np.roll(np.roll(t, int(CY[i]), axis=0), int(CX[i]), axis=1)


def equilibrium_d2q9(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium with shape (9, ny, nx)."""
    cu = _CX * ux[None] + _CY * uy[None]
    usq = ux * ux + uy * uy
    return _W * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def psi_exponential(rho: np.ndarray, rho0: float = 1.0) -> np.ndarray:
    """Shan-Chen exponential pseudo-potential ``rho0 (1 - exp(-rho/rho0))``."""
    return rho0 * (1.0 - np.exp(-rho / rho0))


def psi_linear(rho: np.ndarray, rho0: float = 1.0) -> np.ndarray:
    """Linear pseudo-potential ``psi = rho`` (legacy Shan-Chen form)."""
    return rho


@dataclass
class MCMPConfig:
    """Two-component Shan-Chen configuration (lattice units)."""

    nx: int = 64
    ny: int = 64
    tau_a: float = 1.0
    tau_b: float = 1.0
    g_aa: float = 0.0
    g_ab: float = 6.0
    g_ba: float = 6.0
    g_bb: float = 0.0
    rho0: float = 1.0
    psi_model: str = "exp"          # "exp" | "linear"
    velocity_mode: str = "tau"      # "tau" | "mass" | "self"
    force_scheme: str = "guo"       # "guo" | "shift" (tau-shifted equilibrium only)
    source_velocity: str = "forced"  # "forced" (Guo) | "bary" for the velocity in the source term
    eq_shift: str = "half"          # "half" (Guo F/2rho) | "none" | "tau" (classic SC)

    def psi(self, rho: np.ndarray) -> np.ndarray:
        if self.psi_model == "exp":
            return psi_exponential(rho, self.rho0)
        if self.psi_model == "linear":
            return psi_linear(rho, self.rho0)
        raise ValueError(f"unknown psi_model {self.psi_model!r}")

    def g_matrix(self) -> np.ndarray:
        return np.array([[self.g_aa, self.g_ab],
                         [self.g_ba, self.g_bb]], dtype=DTYPE)


class ShanChenMCMP:
    """Periodic two-component Shan-Chen lattice-Boltzmann solver (D2Q9).

    State: ``f`` with shape ``(2, 9, ny, nx)`` (component, direction, y, x).
    """

    def __init__(self, cfg: MCMPConfig):
        self.cfg = cfg
        if cfg.tau_a <= 0.5 or cfg.tau_b <= 0.5:
            raise ValueError("tau must exceed 0.5")
        if cfg.velocity_mode not in ("tau", "mass", "self"):
            raise ValueError(f"unknown velocity_mode {cfg.velocity_mode!r}")
        if cfg.force_scheme not in ("guo", "shift"):
            raise ValueError(f"unknown force_scheme {cfg.force_scheme!r}")
        self.nx, self.ny = cfg.nx, cfg.ny
        self.f = np.zeros((2, 9, cfg.ny, cfg.nx), dtype=DTYPE)
        self.t = 0
        self.tau = np.array([cfg.tau_a, cfg.tau_b], dtype=DTYPE)
        self.G = cfg.g_matrix()

    # ------------------------------------------------------------------ setup
    def init_uniform(self, rho_a, rho_b, ux=None, uy=None) -> None:
        """Initialise from local densities (equilibrium at rest by default)."""
        ny, nx = self.ny, self.nx
        ra = np.full((ny, nx), float(rho_a), dtype=DTYPE)
        rb = np.full((ny, nx), float(rho_b), dtype=DTYPE)
        ux = np.zeros((ny, nx), dtype=DTYPE) if ux is None else np.asarray(ux, DTYPE)
        uy = np.zeros((ny, nx), dtype=DTYPE) if uy is None else np.asarray(uy, DTYPE)
        self.f[0] = equilibrium_d2q9(ra, ux, uy)
        self.f[1] = equilibrium_d2q9(rb, ux, uy)
        self.t = 0

    def init_step_x(self, lo: float = 0.1, hi: float = 0.9) -> None:
        """lbmpy reference initial condition: a step in x at ``x = nx/2``.

        Component a is ``lo`` on the left half and ``hi`` on the right half,
        component b is the complement.  The step sits *exactly* on the
        periodic seam so that two interfaces form.
        """
        nx, ny = self.nx, self.ny
        half = nx // 2
        ra = np.empty((ny, nx), dtype=DTYPE)
        rb = np.empty((ny, nx), dtype=DTYPE)
        ra[:, :half] = lo
        ra[:, half:] = hi
        rb[:, :half] = hi
        rb[:, half:] = lo
        self.f[0] = equilibrium_d2q9(ra, np.zeros_like(ra), np.zeros_like(ra))
        self.f[1] = equilibrium_d2q9(rb, np.zeros_like(rb), np.zeros_like(rb))
        self.t = 0

    def init_bubble(self, radius: float, rho_in: float, rho_out: float,
                    interface_width: float = 2.0, inner: int = 0) -> None:
        """Circular drop/bubble of component ``inner`` in the other component.

        A ``tanh`` density profile of width ``interface_width`` is used so that
        the initial state is already close to diffusive equilibrium.
        """
        ny, nx = self.ny, self.nx
        yy, xx = np.mgrid[0:ny, 0:nx].astype(DTYPE)
        cy, cx = 0.5 * (ny - 1), 0.5 * (nx - 1)
        r = np.hypot(yy - cy, xx - cx)
        s = 0.5 * (1.0 - np.tanh(2.0 * (r - radius) / interface_width))  # 1 inside
        rho = np.empty((2, ny, nx), dtype=DTYPE)
        rho[inner] = rho_out + (rho_in - rho_out) * s
        rho[1 - inner] = rho_in + (rho_out - rho_in) * s
        for k in (0, 1):
            self.f[k] = equilibrium_d2q9(rho[k], np.zeros_like(r), np.zeros_like(r))
        self.t = 0

    # ------------------------------------------------------------- diagnostics
    def density(self) -> np.ndarray:
        """Component densities, shape (2, ny, nx)."""
        return self.f.sum(axis=1)

    def raw_momentum(self) -> np.ndarray:
        """Raw momenta ``m_s = sum_i c_i f_i^s``, shape (2, 2, ny, nx)."""
        mx = (self.f * _CX).sum(axis=1)
        my = (self.f * _CY).sum(axis=1)
        return np.stack((mx, my), axis=1)

    def interaction_force(self, rho: np.ndarray | None = None,
                          psi: np.ndarray | None = None) -> np.ndarray:
        """Shan-Chen interaction force, shape (2, 2, ny, nx) = (component, xy).

        ``F_s = - psi_s(x) sum_{s'} g_{ss'} sum_i w_i c_i psi_{s'}(x + c_i)``
        evaluated with the *periodic* neighbourhood of every cell.
        """
        if rho is None:
            rho = self.density()
        if psi is None:
            psi = self.cfg.psi(rho)
        # weighted neighbour sum  sum_i w_i c_i psi_{s'}(x + c_i)  (shape (9,...))
        F = np.empty((2, 2, self.ny, self.nx), dtype=DTYPE)
        for s in (0, 1):
            acc_x = np.zeros((self.ny, self.nx), dtype=DTYPE)
            acc_y = np.zeros((self.ny, self.nx), dtype=DTYPE)
            for d in range(9):
                if CX[d] == 0 and CY[d] == 0:
                    continue
                nb = np.roll(np.roll(psi, int(CY[d]), axis=1), int(CX[d]), axis=2)
                # cross interaction: sum_s' g_{s s'} psi_{s'}(x + c_d)
                gpsi = self.G[s, 0] * nb[0] + self.G[s, 1] * nb[1]
                acc_x += (W9[d] * float(CX[d])) * gpsi
                acc_y += (W9[d] * float(CY[d])) * gpsi
            F[s, 0] = -psi[s] * acc_x
            F[s, 1] = -psi[s] * acc_y
        return F

    def barycentric_velocity(self, rho: np.ndarray | None = None,
                             m: np.ndarray | None = None,
                             F: np.ndarray | None = None) -> np.ndarray:
        """Mixture velocity ``u'`` used in the equilibrium, shape (2, ny, nx).

        ``mode="tau"``  : ``u' = sum_s (m_s + F_s/2)/tau_s / sum_s rho_s/tau_s``
        ``mode="mass"`` : ``u' = sum_s (m_s + F_s/2) / sum_s rho_s``
        ``mode="self"`` : returns the per-component ``(m_s + F_s/2)/rho_s``
        """
        if rho is None:
            rho = self.density()
        if m is None:
            m = self.raw_momentum()
        if F is None:
            F = self.interaction_force(rho)
        num = m + 0.5 * F
        mode = self.cfg.velocity_mode
        if mode == "self":
            return num / rho[:, None]
        if mode == "tau":
            wgt = 1.0 / self.tau
        else:
            wgt = np.ones(2)
        num_w = (num * wgt[:, None, None]).sum(axis=0)      # (2, ny, nx)
        den = (rho * wgt[:, None, None]).sum(axis=0)              # (ny, nx)
        u = num_w / den[None]
        return np.repeat(u[None], 2, axis=0)                # same u' for both components

    # ------------------------------------------------------------- evolution
    def step(self) -> None:
        """One full LBM step (force -> collide -> stream, periodic)."""
        cfg = self.cfg
        rho = self.density()
        m = self.raw_momentum()
        F = self.interaction_force(rho)
        u_bary = self.barycentric_velocity(rho, m, F)
        # equilibrium velocity shift: "half" = Guo F/(2 rho) (lbmpy ForceModel.GUO),
        # "tau" = classic Shan-Chen tau F/rho, "none" = equilibrium on u' itself
        if cfg.eq_shift == "half":
            shift = 0.5
        elif cfg.eq_shift == "tau":
            shift = self.tau[:, None, None]
        elif cfg.eq_shift == "none":
            shift = 0.0
        else:
            raise ValueError(f"unknown eq_shift {cfg.eq_shift!r}")
        u_forced = u_bary + shift * F / rho[:, None]

        f_new = np.empty_like(self.f)
        for s in (0, 1):
            tau = self.tau[s]
            ux = u_forced[s, 0]
            uy = u_forced[s, 1]
            feq = equilibrium_d2q9(rho[s], ux, uy)
            out = self.f[s] + (feq - self.f[s]) / tau
            if cfg.force_scheme == "guo":
                if cfg.source_velocity == "bary":
                    sx, sy = u_bary[s, 0], u_bary[s, 1]
                else:
                    sx, sy = ux, uy
                cu = _CX * sx[None] + _CY * sy[None]
                cF = _CX * F[s, 0][None] + _CY * F[s, 1][None]
                uF = (sx * F[s, 0] + sy * F[s, 1])[None]
                src = _W * (3.0 * (cF - uF) + 9.0 * cu * cF)
                out += (1.0 - 0.5 / tau) * src
            for d in range(9):
                f_new[s, d] = np.roll(np.roll(out[d], int(CY[d]), axis=0),
                                      int(CX[d]), axis=1)
        self.f = f_new
        self.t += 1

    def run(self, steps: int, callback=None) -> None:
        for _ in range(steps):
            self.step()
            if callback is not None:
                callback(self)

    # ------------------------------------------------------------- observables
    def velocity(self) -> np.ndarray:
        """Barycentric velocity at the current state, shape (2, ny, nx)."""
        rho = self.density()
        m = self.raw_momentum()
        F = self.interaction_force(rho)
        return self.barycentric_velocity(rho, m, F).mean(axis=0)

    def spurious_current(self) -> float:
        """``max |u'|`` over the domain (spurious current for a static drop)."""
        u = self.velocity()
        return float(np.sqrt(u[0] ** 2 + u[1] ** 2).max())

    def total_mass(self) -> np.ndarray:
        return self.density().sum(axis=(1, 2))
