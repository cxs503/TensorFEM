"""Free-surface D2Q9 lattice-Boltzmann with Thürey's mass-conserving VOF exchange.

Anchor
------
``XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE`` — independent verification of the
volume-of-fluid (VOF) *mass-exchange* step of the Thürey free-surface LBM
lineage (``Maarten-vd-Sande/lbm`` and the original Thürey / Körner
formulations) against analytic references and against the naive clipping
baseline that the exchange replaces.

Model
-----
NumPy/float64 D2Q9 BGK solver with four cell types:

============  ==========================================================
flag          meaning
============  ==========================================================
``GAS``       empty cell (no populations, no liquid mass)
``INTERFACE`` partially filled; carries the liquid mass ``m`` (VOF)
``FLUID``     completely filled; carries populations only
``SOLID``     wall; half-way bounce-back
============  ==========================================================

Liquid mass lives on interface cells, exactly as in Thürey & Rüde; ``FLUID``
cells are full by definition, and the LBM density ``rho = sum_i f_i`` is the
pressure variable (``p = c_s^2 rho``).

Thürey mass exchange
--------------------
With ``f*`` the post-collision populations, the mass an interface cell ``x``
receives from direction ``i`` is

* neighbour ``FLUID``:     ``dm_i = f*_opp(x+e_i) - f*_i(x)``
* neighbour ``INTERFACE``: ``dm_i = 0.5 * [ (f*_opp(x+e_i) - f*_i(x))
                                         + (f*_i(x+e_i) - f*_opp(x)) ]``
* neighbour ``GAS/SOLID``: ``dm_i = 0``

The interface/interface term is antisymmetric under the mirrored pair, hence
``dm_i(x) + dm_opp(x+e_i) = 0``: interface-interface exchange is *exactly*
conservative (verified to machine precision in the test suite).

After streaming, the *exchange* step handles cells that overflow or run dry:

* ``m > rho`` -> the cell is promoted to ``FLUID`` and its surplus is handed to
  neighbouring interface cells in proportion to their free capacity
  ``rho - m`` (three Jacobi passes plus a greedy sweep, so nothing is lost);
* ``m < 0``   -> the cell is demoted to ``GAS`` and its deficit is taken from
  neighbouring interface cells in proportion to their content.

Cells that just filled up open their gas neighbours as empty interface cells,
which is how the free surface advances into the gas.

Plain clipping — zero the overflow, drop the deficit, convert — is the shortcut
this step replaces: it destroys liquid at every conversion.  Both variants are
implemented here (``FreeSurfaceConfig.mass_exchange``) so the defect is
measured rather than asserted.

Known limitation (documented, measured in the report): the gas-side
reconstruction at the surface is not mass-neutral, so the *volumetric* liquid
mass defined as ``sum_FLUID rho + sum_INTERFACE m`` drifts once a transient
run produces a large population of thin-film interface cells.  A deliberately
exactly-conservative alternative (pin every full cell at mass 1 and push the
deviation out) was implemented and rejected: it conserves volume to machine
precision but destroys the pressure support, so a resting column free-falls and
a dam break never collapses.

References
----------
[1] Körner, C., Thies, M., Hofmann, T., Thürey, N., Rüde, U. (2005).
    Lattice Boltzmann model for free surface flow for modeling foaming.
    J. Stat. Phys. 121, 179-196.
[2] Thürey, N., Rüde, U. (2009). Stable free surface flows with the lattice
    Boltzmann method on adaptively coarsened grids. Comput. Vis. Sci. 12,
    247-263.
[3] Thürey, N., Mehl, M., Rüde, U. (2006). Interactive free surface fluids
    with the lattice Boltzmann method. Univ. Erlangen, tech. report 05-15.
[4] Ritter, A. (1892). Die Fortpflanzung der Wasserwellen. Z. Ver. Dtsch.
    Ing. 36, 947 (dam-break, dry-bed exact solution).
[5] Guo, Z., Zheng, C., Shi, B. (2002). Phys. Fluids 14, 2007 (forcing).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DTYPE = np.float64

GAS = 0
INTERFACE = 1
FLUID = 2
SOLID = 3

CX = np.array((0, 1, 0, -1, 0, 1, -1, -1, 1), dtype=np.int64)
CY = np.array((0, 0, 1, 0, -1, 1, 1, -1, -1), dtype=np.int64)
W9 = np.array((4.0 / 9.0,) + (1.0 / 9.0,) * 4 + (1.0 / 36.0,) * 4, dtype=DTYPE)
OPP = np.array((0, 3, 4, 1, 2, 7, 8, 5, 6), dtype=np.int64)
CS2 = 1.0 / 3.0

_W = W9.reshape(9, 1, 1).astype(DTYPE)
_CX = CX.reshape(9, 1, 1).astype(DTYPE)
_CY = CY.reshape(9, 1, 1).astype(DTYPE)


def _sample(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``out[..., y, x] = a[..., y+dy, x+dx]`` with zero fill outside the box."""
    out = np.zeros_like(a)
    ny, nx = a.shape[-2], a.shape[-1]
    xs = slice(max(dx, 0), nx + min(dx, 0))
    xd = slice(max(-dx, 0), nx + min(-dx, 0))
    ys = slice(max(dy, 0), ny + min(dy, 0))
    yd = slice(max(-dy, 0), ny + min(-dy, 0))
    out[..., yd, xd] = a[..., ys, xs]
    return out


def equilibrium(rho, ux, uy) -> np.ndarray:
    """D2Q9 equilibrium, shape ``(9, ny, nx)``."""
    cu = _CX * ux[None] + _CY * uy[None]
    usq = ux * ux + uy * uy
    return _W * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


@dataclass
class FreeSurfaceConfig:
    """Lattice-unit configuration (dx = dt = rho_liquid = 1)."""

    nx: int = 120
    ny: int = 80
    nu: float = 2.0e-3            # kinematic viscosity (lattice)
    gravity: float = 1.0e-5       # body acceleration magnitude, acts in -y
    rho_atm: float = 1.0          # gas-side density used by the reconstruction
    mass_exchange: bool = True    # False -> naive clipping baseline
    redistribute_passes: int = 3

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def omega(self) -> float:
        return 1.0 / self.tau


@dataclass
class FreeSurfaceState:
    """Probe record appended to :attr:`FreeSurfaceLBM.history`."""

    step: int = 0
    total_mass: float = 0.0
    interface_mass: float = 0.0
    max_speed: float = 0.0
    n_fluid: int = 0
    n_interface: int = 0


class FreeSurfaceLBM:
    """D2Q9 free-surface solver with Thürey mass exchange."""

    def __init__(self, cfg: FreeSurfaceConfig | None = None):
        self.cfg = cfg or FreeSurfaceConfig()
        nx, ny = self.cfg.nx, self.cfg.ny
        self.flags = np.full((ny, nx), GAS, dtype=np.int8)
        self.f = np.zeros((9, ny, nx), dtype=DTYPE)
        self.mass = np.zeros((ny, nx), dtype=DTYPE)
        self.step_count = 0
        self.history: list[FreeSurfaceState] = []
        #: liquid mass the exchange could not place (must stay ~1e-15)
        self.unplaced_mass = 0.0

    # ------------------------------------------------------------------ setup
    def solid_bottom(self, thickness: int = 1) -> None:
        self.flags[:thickness, :] = SOLID

    def solid_box(self, y0: int, y1: int, x0: int, x1: int) -> None:
        self.flags[y0:y1, x0:x1] = SOLID

    def fill_block(self, x0: int, x1: int, y0: int, y1: int) -> None:
        """Fill a rectangle with liquid (interior FLUID, bounding ring INTERFACE)."""
        blk = np.zeros_like(self.flags, dtype=bool)
        blk[y0:y1, x0:x1] = True
        blk &= self.flags != SOLID
        self.flags[blk] = FLUID
        self.mass[blk] = self.cfg.rho_atm
        ring = blk & self._adjacent_mask(~blk)
        self.flags[ring] = INTERFACE
        self.mass[ring] = self.cfg.rho_atm
        self._reset_populations()

    def _adjacent_mask(self, mask: np.ndarray) -> np.ndarray:
        oi = mask.astype(np.int8)
        out2 = np.zeros_like(mask, dtype=bool)
        for i in range(1, 9):
            out2 |= _sample(oi, int(CX[i]), int(CY[i])) > 0
        return out2

    def _reset_populations(self) -> None:
        liq = (self.flags == FLUID) | (self.flags == INTERFACE)
        zeros = np.zeros_like(self.mass)
        f0 = equilibrium(np.full_like(self.mass, self.cfg.rho_atm), zeros, zeros)
        self.f = np.where(liq[None], f0, 0.0)

    # ------------------------------------------------------------ diagnostics
    def macroscopic(self):
        rho = self.f.sum(axis=0)
        px = (self.f * _CX).sum(axis=0)
        py = (self.f * _CY).sum(axis=0)
        safe = np.where(rho > 1e-30, rho, 1.0)
        return rho, px / safe, py / safe

    def total_mass(self) -> float:
        """Volumetric liquid mass: interface ``m`` plus full-cell density."""
        liq = (self.flags == FLUID) | (self.flags == INTERFACE)
        rho, _, _ = self.macroscopic()
        return float(np.where(self.flags == INTERFACE, self.mass,
                              np.where(liq, rho, 0.0)).sum())

    def interface_mass(self) -> float:
        """Liquid mass held on interface cells (the exchanged quantity)."""
        return float(np.where(self.flags == INTERFACE, self.mass, 0.0).sum())

    def fill_fraction(self) -> np.ndarray:
        """VOF fraction per cell (0 in gas/solid, 1 in fluid cells)."""
        liq = (self.flags == FLUID) | (self.flags == INTERFACE)
        rho, _, _ = self.macroscopic()
        safe = np.where(rho > 1e-30, rho, 1.0)
        eps = np.where(self.flags == INTERFACE, np.clip(self.mass / safe, 0.0, 1.0), 1.0)
        return np.where(liq, eps, 0.0)

    def liquid_volume(self) -> float:
        return float(self.fill_fraction().sum())

    def column_height(self) -> np.ndarray:
        """Per-column liquid height (sum of fill fractions over y)."""
        return self.fill_fraction().sum(axis=0)

    def front_position(self, threshold: float = 0.05) -> int:
        """Rightmost column whose liquid height exceeds ``threshold``."""
        h = self.column_height()
        idx = np.nonzero(h > threshold)[0]
        return int(idx[-1]) if idx.size else -1

    def probe(self) -> FreeSurfaceState:
        _, ux, uy = self.macroscopic()
        liq = (self.flags == FLUID) | (self.flags == INTERFACE)
        sp = np.sqrt(ux * ux + uy * uy)
        return FreeSurfaceState(
            step=self.step_count,
            total_mass=self.total_mass(),
            interface_mass=self.interface_mass(),
            max_speed=float(sp[liq].max()) if liq.any() else 0.0,
            n_fluid=int((self.flags == FLUID).sum()),
            n_interface=int((self.flags == INTERFACE).sum()),
        )

    # -------------------------------------------------------------- main loop
    def step(self) -> None:
        cfg = self.cfg
        liq = (self.flags == FLUID) | (self.flags == INTERFACE)
        rho, ux, uy = self.macroscopic()

        # 1) collision (BGK) + gravity forcing on liquid cells
        feq = equilibrium(rho, ux, uy)
        f_post = self.f - cfg.omega * (self.f - feq)
        f_post = f_post + 3.0 * _W * rho[None] * (_CY * (-cfg.gravity))
        f_post = np.where(liq[None], f_post, 0.0)

        # 2) streaming with gas-side reconstruction and half-way bounce-back
        f_new = self._stream(f_post, ux, uy, liq)
        rho_new = f_new.sum(axis=0)

        # 3) Thürey mass flux on interface cells
        dm = self._mass_flux(f_post)
        self.mass = self.mass + np.where(self.flags == INTERFACE, dm, 0.0)

        # 4) exchange / conversions / redistribution
        self.f = f_new
        self._exchange(np.where(rho_new > 1e-30, rho_new, cfg.rho_atm))

        # 5) flag bookkeeping (liquid touching gas is an interface cell)
        self._update_flags()

        self.step_count += 1
        self.history.append(self.probe())

    def _stream(self, f_post, ux, uy, liq):
        f_new = np.zeros_like(self.f)
        f_new[0] = np.where(liq, f_post[0], 0.0)
        usq = ux * ux + uy * uy
        for i in range(1, 9):
            cx, cy = int(CX[i]), int(CY[i])
            o = int(OPP[i])
            src_flag = _sample(self.flags.astype(np.int8), -cx, -cy)
            src_ok = (src_flag == FLUID) | (src_flag == INTERFACE)
            src_solid = src_flag == SOLID

            pop = _sample(f_post[i], -cx, -cy)      # population from upstream

            cu = float(CX[i]) * ux + float(CY[i]) * uy
            feq_i = W9[i] * self.cfg.rho_atm * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
            feq_o = W9[o] * self.cfg.rho_atm * (1.0 - 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
            recon = feq_i + (feq_o - f_post[o])

            val = np.where(src_ok, pop, np.where(src_solid, f_post[o], recon))
            f_new[i] = np.where(liq, val, 0.0)
        return f_new

    def _mass_flux(self, f_post) -> np.ndarray:
        """Thürey mass flux, evaluated on interface cells only."""
        dm = np.zeros_like(self.mass)
        iface = self.flags == INTERFACE
        fl = self.flags.astype(np.int8)
        for i in range(1, 9):
            cx, cy = int(CX[i]), int(CY[i])
            o = int(OPP[i])
            nb = _sample(fl, cx, cy)
            f_o_nb = _sample(f_post[o], cx, cy)     # f_opp at x + e_i
            f_i_nb = _sample(f_post[i], cx, cy)     # f_i   at x + e_i
            fluid_term = f_o_nb - f_post[i]
            pair_term = 0.5 * ((f_o_nb - f_post[i]) + (f_i_nb - f_post[o]))
            contrib = np.where(nb == FLUID, fluid_term,
                               np.where(nb == INTERFACE, pair_term, 0.0))
            dm += np.where(iface, contrib, 0.0)
        return dm

    # -------------------------------------------------- Thürey exchange step
    def _exchange(self, rho: np.ndarray) -> None:
        """Convert over/under-filled interface cells and hand the surplus on."""
        iface = self.flags == INTERFACE
        if not iface.any():
            return
        full = iface & (self.mass > rho)
        empty = iface & (self.mass < 0.0)

        excess = np.zeros_like(self.mass)
        excess += np.where(full, self.mass - rho, 0.0)
        self.mass = np.where(full, rho, self.mass)
        self.flags = np.where(full, FLUID, self.flags).astype(np.int8)

        excess += np.where(empty, self.mass, 0.0)     # negative
        self.mass = np.where(empty, 0.0, self.mass)
        self.flags = np.where(empty, GAS, self.flags).astype(np.int8)

        # a cell that just filled up opens its gas neighbours for the surplus
        if np.any(full):
            open_gas = self._adjacent_mask(full) & (self.flags == GAS)
            if np.any(open_gas):
                self.flags = np.where(open_gas, INTERFACE, self.flags).astype(np.int8)
                self.mass = np.where(open_gas, 0.0, self.mass)
                zeros = np.zeros_like(self.mass)
                f0 = equilibrium(np.full_like(self.mass, self.cfg.rho_atm), zeros, zeros)
                self.f = np.where(open_gas[None], f0, self.f)

        if not np.any(excess):
            return
        if not self.cfg.mass_exchange:
            # deliberate baseline: clipping only, surplus/deficit is dropped
            self.unplaced_mass += float(excess.sum())
            self.mass = np.clip(self.mass, 0.0, rho)
            return
        self._distribute(excess, rho)

    def _distribute(self, pool: np.ndarray, rho: np.ndarray) -> float:
        """Conservative placement of a signed surplus into interface capacity.

        Donor ``c`` hands receiver ``c + e_i`` the fraction ``pool_c * w_i /
        sum_i w_i`` of its surplus (``w_i`` = receiver's free capacity for a
        surplus, its current content for a deficit).  Weights are normalised per
        donor, so what leaves the donors equals ``sum(pool)`` exactly: the
        exchange moves liquid, it never creates or destroys it.  Whatever a
        corner of the domain cannot absorb is placed greedily, so the residual
        is floating-point noise.  Returns that residual.
        """
        pool = pool.copy()
        if not np.any(pool):
            return 0.0
        for _ in range(max(1, self.cfg.redistribute_passes)):
            if not np.any(pool):
                break
            iface_l = self.flags == INTERFACE
            w_pos = np.where(iface_l, np.clip(rho - self.mass, 0.0, None), 0.0)
            w_neg = np.where(iface_l, np.clip(self.mass, 0.0, None), 0.0)
            weight_sum = np.zeros_like(pool)
            for i in range(1, 9):
                weight_sum += _sample(w_pos, int(CX[i]), int(CY[i]))
            for i in range(1, 9):
                cx, cy = int(CX[i]), int(CY[i])
                w = np.where(pool > 0.0, _sample(w_pos, cx, cy), _sample(w_neg, cx, cy))
                tot = np.where(weight_sum > 1e-300, weight_sum, 1.0)
                flux = np.where(weight_sum > 1e-300, pool * w / tot, 0.0)
                self.mass = self.mass + np.where(iface_l, _sample(flux, -cx, -cy), 0.0)
                pool = pool - flux

        residual = float(pool.sum())
        if abs(residual) > 1e-14:
            iface_l = self.flags == INTERFACE
            flat_m = self.mass.ravel().copy()
            if residual > 0.0:
                room = np.where(iface_l, np.clip(rho - self.mass, 0.0, None), 0.0)
                flat_room = room.ravel()
                left = residual
                for k in np.argsort(flat_room)[::-1]:
                    if left <= 0.0 or flat_room[k] <= 0.0:
                        break
                    add = min(flat_room[k], left)
                    flat_m[k] += add
                    left -= add
                residual = left
            else:
                left = -residual
                for k in np.argsort(np.where(iface_l, self.mass, 0.0).ravel())[::-1]:
                    if left <= 0.0 or flat_m[k] <= 0.0:
                        break
                    cut = min(flat_m[k], left)
                    flat_m[k] -= cut
                    left -= cut
                residual = -left
            self.mass = flat_m.reshape(self.mass.shape)
        self.unplaced_mass += residual
        return residual

    def _update_flags(self) -> None:
        """A liquid cell touched by gas is a surface cell (keeps its mass)."""
        fl = self.flags
        liq = (fl == FLUID) | (fl == INTERFACE)
        newly_if = (self._adjacent_mask(fl == GAS) & liq) & (fl == FLUID)
        if newly_if.any():
            rho, _, _ = self.macroscopic()
            self.mass = np.where(newly_if, rho, self.mass)
            fl = np.where(newly_if, INTERFACE, fl).astype(np.int8)
        self.flags = fl.astype(np.int8)
        keep = ((fl == FLUID) | (fl == INTERFACE))[None]
        self.f = np.where(keep, self.f, 0.0)

    # ------------------------------------------------------------------ driver
    def run(self, n_steps: int, probe_every: int = 0) -> list[FreeSurfaceState]:
        out = []
        for k in range(n_steps):
            self.step()
            if probe_every and (k + 1) % probe_every == 0:
                out.append(self.history[-1])
        return out


# ---------------------------------------------------------------------------
# Analytic references
# ---------------------------------------------------------------------------
def hydrostatic_density(y, surface_y: float, gravity: float,
                        rho_atm: float = 1.0) -> np.ndarray:
    """``rho(y) = rho_atm + 3 g (y_surface - y)`` (p = c_s^2 rho)."""
    return rho_atm + 3.0 * gravity * (surface_y - np.asarray(y, dtype=float))


def ritter_surface(x, t: float, g: float, h0: float, dam_x: float = 0.0):
    """Ritter (1892) dry-bed dam-break surface height.

    ``h = h0`` for ``x - dam_x <= -sqrt(g h0) t`` and
    ``h = (2 sqrt(g h0) - (x-dam_x)/t)^2 / (9 g)`` up to the front at
    ``dam_x + 2 sqrt(g h0) t``, zero beyond.
    """
    x = np.asarray(x, dtype=float) - dam_x
    c = np.sqrt(g * h0)
    h = np.zeros_like(x)
    h[x <= -c * t] = h0
    mid = (x > -c * t) & (x < 2.0 * c * t)
    h[mid] = (2.0 * c - x[mid] / t) ** 2 / (9.0 * g)
    return h


def ritter_front(t, g: float, h0: float, dam_x: float = 0.0):
    """Ritter front position ``x_f = dam_x + 2 sqrt(g h0) t``."""
    return dam_x + 2.0 * np.sqrt(g * h0) * np.asarray(t, dtype=float)