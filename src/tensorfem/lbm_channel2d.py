"""Two-dimensional D2Q9 lattice-Boltzmann channel solver with immersed boundaries.

This module is the *fluid* half of the TensorFEM x TensorLBM FSI verification
face.  It is a from-scratch D2Q9 BGK solver (NumPy, float64) with

* half-way bounce-back walls and circular obstacles (no-slip),
* parabolic velocity inlet / zero-gradient outlet,
* Guo body forcing for immersed-boundary (IB) momentum transfer,
* a Peskin 4-point delta kernel for Lagrangian marker exchange.

Unit system
-----------
The solver works in *lattice* units (dx = dt = rho0 = 1).  Physical quantities
are recovered through

    u_phys = u_lat * dx_phys / dt_phys
    nu_phys = nu_lat * dx_phys**2 / dt_phys
    F_phys = F_lat * rho_phys * dx_phys**2 / dt_phys**2  (per unit depth)

The last relation follows because a lattice force unit is
``rho0 * dx**3 / dt**2`` in 3-D, i.e. ``rho0 * dx**2 / dt**2`` per unit depth
in 2-D.

Why a new solver?
-----------------
The pre-existing ``fsi_turek_hron`` module models the cylinder wake with a
*harmonic forcing at a tuned frequency* (``f_vortex = 3.0``, ``F0 = 17.5``,
both calibrated to the reference answer).  That is a MOCK for the fluid side:
it cannot be falsified by a flow quantity.  This module resolves the wake from
the Navier-Stokes equations so the shedding frequency and the hydrodynamic
loads are *outputs*, not inputs.

References
----------
[1] Guo, Z., Zheng, C., Shi, B. (2002). Phys. Fluids 14, 2007.
[2] Peskin, C.S. (2002). Acta Numerica 11, 479 (IB delta kernel).
[3] Turek, S., Hron, J. (2006). LNCS 53, 371 (2D-2 / FSI-2 benchmarks).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium distribution (shape (9, ny, nx))."""
    cu = _CX * ux[None] + _CY * uy[None]
    usq = ux * ux + uy * uy
    return _W * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rho, ux, uy) from populations."""
    rho = f.sum(axis=0)
    px = (f * _CX).sum(axis=0)
    py = (f * _CY).sum(axis=0)
    safe = np.where(rho > 1e-30, rho, 1.0)
    return rho, px / safe, py / safe


def _shift(t: np.ndarray, i: int) -> np.ndarray:
    """Stream one population by one lattice cell (zero-fill, no wrap)."""
    cx, cy = int(CX[i]), int(CY[i])
    out = np.zeros_like(t)
    ny, nx = t.shape
    xs_dst = slice(max(cx, 0), nx + min(cx, 0))
    xs_src = slice(max(-cx, 0), nx + min(-cx, 0))
    ys_dst = slice(max(cy, 0), ny + min(cy, 0))
    ys_src = slice(max(-cy, 0), ny + min(-cy, 0))
    out[ys_dst, xs_dst] = t[ys_src, xs_src]
    return out


def _shift_mask(mask: np.ndarray, i: int) -> np.ndarray:
    """Cells whose upstream neighbour (direction i) is solid."""
    return _shift(mask.astype(DTYPE), i) > 0.5


def peskin_delta(r: np.ndarray) -> np.ndarray:
    """Peskin 4-point cosine delta function."""
    a = np.abs(r)
    out = np.zeros_like(a)
    m1 = a <= 1.0
    m2 = (a > 1.0) & (a <= 2.0)
    out[m1] = (3.0 - 2.0 * a[m1] + np.sqrt(1.0 + 4.0 * a[m1] - 4.0 * a[m1] ** 2)) / 8.0
    out[m2] = (5.0 - 2.0 * a[m2] - np.sqrt(-7.0 + 12.0 * a[m2] - 4.0 * a[m2] ** 2)) / 8.0
    return out


@dataclass
class ChannelConfig:
    """Geometry / flow parameters (lattice units)."""

    nx: int
    ny: int
    u_mean: float          # lattice mean inlet velocity
    nu: float              # lattice kinematic viscosity
    cylinder_center: tuple[float, float] | None = None  # lattice index space
    cylinder_radius: float = 0.0
    perturbation: float = 0.0   # amplitude of the symmetry-breaking v-velocity

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def u_max(self) -> float:
        return 1.5 * self.u_mean


class ChannelLBM:
    """D2Q9 BGK channel with a circular (bounce-back) obstacle.

    Cell centres sit at ``(i + 0.5) * dx``; boundary rows ``j = 0`` and
    ``j = ny - 1`` are solid so that half-way bounce-back places the walls
    exactly on the channel faces.
    """

    def __init__(self, cfg: ChannelConfig):
        self.cfg = cfg
        self.nx, self.ny = cfg.nx, cfg.ny
        if cfg.tau <= 0.5:
            raise ValueError("tau must exceed 0.5 for positive viscosity")
        self.solid = np.zeros((self.ny, self.nx), dtype=bool)
        self.obstacle = np.zeros((self.ny, self.nx), dtype=bool)
        self.solid[0, :] = True
        self.solid[-1, :] = True
        if cfg.cylinder_center is not None:
            self._add_disk(cfg.cylinder_center, cfg.cylinder_radius)
        self.f = self._initial_field()
        self.time = 0
        self._from_solid = None      # cache of upstream-solid masks
        self._scratch = None
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        """Cache the static streaming masks and the equilibrium scratch."""
        self._from_solid = [_shift_mask(self.solid, i) for i in range(9)]
        # precomputed boundary links for the momentum-exchange force
        self._mem_links = []
        boundary = ~self.solid
        from_obstacle = [(_shift(self.obstacle.astype(DTYPE), i) > 0.5) & boundary
                         for i in range(9)]
        for i in range(1, 9):
            idx = np.flatnonzero(from_obstacle[i].ravel())
            if idx.size:
                self._mem_links.append((OPP[i], idx, CX[OPP[i]], CY[OPP[i]]))
        self._scratch = {
            "cu": np.empty((9, self.ny, self.nx), dtype=DTYPE),
            "tmp": np.empty((9, self.ny, self.nx), dtype=DTYPE),
            "feq": np.empty((9, self.ny, self.nx), dtype=DTYPE),
        }

    # ------------------------------------------------------------------ setup
    def _add_disk(self, center, radius) -> None:
        yy, xx = np.meshgrid(
            np.arange(self.ny, dtype=DTYPE), np.arange(self.nx, dtype=DTYPE),
            indexing="ij")
        d2 = (xx + 0.5 - center[0]) ** 2 + (yy + 0.5 - center[1]) ** 2
        self.obstacle |= d2 < radius * radius
        self.solid |= self.obstacle

    def inlet_profile(self) -> np.ndarray:
        """Parabolic inlet velocity (lattice): u_max at the channel centre."""
        y = np.arange(self.ny, dtype=DTYPE)
        span = self.ny - 3.0          # fluid rows are 1 .. ny-2
        eta = (y - 1.0) / span
        return np.clip(4.0 * self.cfg.u_max * eta * (1.0 - eta), 0.0, None)

    def _initial_field(self) -> np.ndarray:
        ux = np.repeat(self.inlet_profile()[:, None], self.nx, axis=1)
        uy = np.zeros_like(ux)
        if self.cfg.perturbation:
            # localised antisymmetric kick downstream of the obstacle
            x0 = 0.5 * self.nx
            yy, xx = np.meshgrid(np.arange(self.ny, dtype=DTYPE),
                                 np.arange(self.nx, dtype=DTYPE), indexing="ij")
            uy = self.cfg.perturbation * (
                np.exp(-((xx - x0) / (0.1 * self.nx)) ** 2)
                * np.exp(-((yy - 0.5 * self.ny) / (0.15 * self.ny)) ** 2))
        f = equilibrium(np.ones_like(ux), ux, uy)
        f[:, self.solid] = 0.0
        return f

    def _equilibrium_into(self, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        """Write the D2Q9 equilibrium into the reusable scratch buffer."""
        cu = self._scratch["cu"]
        tmp = self._scratch["tmp"]
        out = self._scratch["feq"]
        np.multiply(_CX, ux[None], out=cu)
        cu += _CY * uy[None]
        np.multiply(cu, cu, out=tmp)
        tmp *= 4.5
        tmp += 1.0
        tmp += 3.0 * cu
        tmp -= 1.5 * (ux * ux + uy * uy)[None]
        np.multiply(_W, rho[None], out=out)
        out *= tmp
        return out

    # -------------------------------------------------------------- evolution
    def _collide(self, body_force: np.ndarray | None = None,
                 extra_solid: np.ndarray | None = None) -> np.ndarray:
        """BGK collision (+ optional Guo forcing); returns post-collision f."""
        tau = self.cfg.tau
        rho, ux, uy = macroscopic(self.f)
        fluid = ~self.solid if extra_solid is None else ~(self.solid | extra_solid)
        if extra_solid is not None:
            rho = np.where(fluid, rho, 1.0)
            ux = np.where(fluid, ux, 0.0)
            uy = np.where(fluid, uy, 0.0)
        # a moving body can uncover cells whose populations were zeroed while
        # they were solid: refill them with the resting equilibrium so the
        # macroscopic moments never divide by a zero density
        empty = fluid & (rho < 1e-12)
        if empty.any():
            self.f[:, empty] = _W.ravel()[:, None]
            rho = np.where(empty, 1.0, rho)
        rho = np.where(fluid, rho, 1.0)
        ux = np.where(fluid, ux, 0.0)
        uy = np.where(fluid, uy, 0.0)

        if body_force is not None:
            fx = body_force[0]
            fy = body_force[1]
            ux_g = ux + 0.5 * fx / rho
            uy_g = uy + 0.5 * fy / rho
            cu = self._scratch["cu"]
            np.multiply(_CX, ux_g[None], out=cu)
            cu += _CY * uy_g[None]
            cdotF = _CX * fx[None] + _CY * fy[None]
            udotF = (ux_g * fx + uy_g * fy)[None]
            src = _W * (3.0 * (cdotF - udotF) + 9.0 * cu * cdotF)
            f_eq = self._equilibrium_into(rho, ux_g, uy_g)
            out = self.f + (f_eq - self.f) / tau + (1.0 - 0.5 / tau) * src
        else:
            f_eq = self._equilibrium_into(rho, ux, uy)
            out = self.f + (f_eq - self.f) / tau
        out[:, ~fluid] = 0.0
        return out

    def _stream(self, f_post: np.ndarray) -> None:
        """Half-way bounce-back streaming + inlet/outlet boundary conditions."""
        solid = self.solid
        from_solid = self._from_solid
        f_new = np.empty_like(self.f)
        for i in range(1, 9):
            streamed = _shift(f_post[i], i)
            f_new[i] = np.where(from_solid[i], f_post[OPP[i]], streamed)
        f_new[0] = f_post[0]
        f_new[:, solid] = 0.0
        u_in = self.inlet_profile()
        f_new[:, :, 0] = equilibrium(np.ones((self.ny, 1)),
                                     u_in[:, None], np.zeros((self.ny, 1)))[:, :, 0]
        f_new[:, :, -1] = f_new[:, :, -2]
        self.f = f_new
        self._f_post = f_post
        self.time += 1

    def step(self, body_force: np.ndarray | None = None) -> None:
        """Advance one lattice step with an optional Eulerian body force."""
        self._stream(self._collide(body_force))

    def step_cells_forced(self, cell_mask: np.ndarray,
                          target_velocity: np.ndarray) -> tuple[np.ndarray, float]:
        """Advance one step forcing a cell set to a prescribed velocity.

        Direct (immersed-boundary) forcing: the post-collision velocity ``u*``
        is pushed to ``target_velocity`` inside ``cell_mask`` by a body force
        ``f = rho (u_target - u*)``; the momentum handed to the fluid is
        therefore ``sum(f)`` and the reaction on the body is ``-sum(f)``.

        Returns
        -------
        force_density : (2, ny, nx) body-force density handed to the fluid
        u_star : (ux, uy) velocity *before* the forcing (needed for the
            implicit added-mass treatment of the structural reaction)
        """
        tau = self.cfg.tau
        f_post = self._collide()
        rho, ux, uy = macroscopic(f_post)
        mask = cell_mask
        # body force density required to impose the target velocity
        fx = np.where(mask, rho * (target_velocity[0] - ux), 0.0)
        fy = np.where(mask, rho * (target_velocity[1] - uy), 0.0)
        cu = self._scratch["cu"]
        np.multiply(_CX, ux[None], out=cu)
        cu += _CY * uy[None]
        cdotF = _CX * fx[None] + _CY * fy[None]
        udotF = (ux * fx + uy * fy)[None]
        src = _W * (3.0 * (cdotF - udotF) + 9.0 * cu * cdotF)
        f_post = f_post + (1.0 - 0.5 / tau) * src
        self._stream(f_post)
        body_force = np.stack([fx, fy])
        return body_force, (ux.copy(), uy.copy())

    def step_with_mask(self, mask: np.ndarray) -> np.ndarray:
        """Advance one step with an extra (moving) solid region ``mask``.

        The extra region is treated as a no-slip solid through half-way
        bounce-back, so no fluid can cross it (a volume-penalty/direct-forcing
        approach otherwise lets the flow leak through a thin body and inflates
        the reaction force).  The force on the region is returned from the
        momentum exchange of its boundary links (lattice units, per unit
        depth).
        """
        solid = self.solid | mask
        prev = getattr(self, "_prev_eff_solid", None)
        f_post = self._collide(extra_solid=mask)
        f_new = np.empty_like(self.f)
        for i in range(1, 9):
            streamed = _shift(f_post[i], i)
            from_solid = _shift_mask(solid, i)
            f_new[i] = np.where(from_solid, f_post[OPP[i]], streamed)
        f_new[0] = f_post[0]
        f_new[:, solid] = 0.0
        # cells freed by the moving body re-enter the fluid at rest (rho = 1)
        if prev is not None:
            freed = prev & ~solid
            if freed.any():
                n_free = int(freed.sum())
                f_new[:, freed] = equilibrium(np.ones((n_free, 1)),
                                              np.zeros((n_free, 1)),
                                              np.zeros((n_free, 1)))[:, :, 0]
        u_in = self.inlet_profile()
        f_new[:, :, 0] = equilibrium(np.ones((self.ny, 1)),
                                     u_in[:, None], np.zeros((self.ny, 1)))[:, :, 0]
        f_new[:, :, -1] = f_new[:, :, -2]
        self.f = f_new
        self._f_post = f_post
        self._prev_eff_solid = solid
        self._solid_eff = solid
        self.time += 1
        return self._mask_exchange_force(mask, f_post, solid)

    def _mask_exchange_force(self, mask: np.ndarray, f_post: np.ndarray,
                             solid: np.ndarray) -> np.ndarray:
        """Momentum-exchange force field on the extra solid region ``mask``.

        Returns a (2, ny, nx) array whose non-zero entries sit on the *fluid*
        nodes adjacent to the body and carry the force imparted to the wall
        (lattice units).  Summing the array gives the resultant.
        """
        fmap = np.zeros((2, self.ny, self.nx), dtype=DTYPE)
        fluid = ~solid
        for i in range(1, 9):
            from_mask = _shift_mask(mask, i) & fluid
            idx = np.flatnonzero(from_mask.ravel())
            if idx.size == 0:
                continue
            contrib = 2.0 * f_post[OPP[i]].ravel()[idx]
            np.add.at(fmap[0].ravel(), idx, contrib * CX[OPP[i]])
            np.add.at(fmap[1].ravel(), idx, contrib * CY[OPP[i]])
        return fmap

    # -------------------------------------------------------------- diagnosis
    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Velocity field with solid cells set to zero (never NaN)."""
        _, ux, uy = macroscopic(self.f)
        m = ~getattr(self, "_solid_eff", self.solid)
        return np.where(m, ux, 0.0), np.where(m, uy, 0.0)

    def density(self) -> np.ndarray:
        return self.f.sum(axis=0)

    def kinetic_energy(self) -> float:
        ux, uy = self.velocity()
        u2 = np.where(self.solid, 0.0, ux * ux + uy * uy)
        return float(0.5 * u2.sum())

    def viscous_dissipation_rate(self) -> float:
        """Instantaneous viscous dissipation ``2 rho nu S_ij S_ij`` (lattice)."""
        ux, uy = self.velocity()
        m = ~self.solid
        ux = np.where(m, ux, 0.0)
        uy = np.where(m, uy, 0.0)
        dux_dy, dux_dx = np.gradient(ux)
        duy_dy, duy_dx = np.gradient(uy)
        sxx, syy = dux_dx, duy_dy
        sxy = 0.5 * (dux_dy + duy_dx)
        s2 = sxx * sxx + syy * syy + 2.0 * sxy * sxy
        rho = self.density()
        return float((2.0 * self.cfg.nu * rho * s2)[m].sum())

    def inlet_energy_flux(self) -> float:
        """0.5 * rho * u^3 integrated over the inlet column (lattice)."""
        ux, _ = self.velocity()
        u = np.clip(ux[:, 0], 0.0, None)
        return float((0.5 * u ** 3).sum())

    def outlet_energy_flux(self) -> float:
        ux, _ = self.velocity()
        u = np.clip(ux[:, -1], 0.0, None)
        return float((0.5 * u ** 3).sum())

    def momentum_exchange_force(self) -> np.ndarray:
        """Force on the bounce-back bodies from boundary-link momentum exchange.

        For a half-way bounce-back link the momentum handed to the wall is
        ``c_j * (f_j - f_opp(j))`` where ``j`` points into the wall.  Returns
        the lattice force ``(Fx, Fy)`` (per unit depth).
        """
        f_post = self._f_post
        eff = getattr(self, "_solid_eff", self.solid)
        fluid_flat = (~eff).ravel()
        F = np.zeros(2, dtype=DTYPE)
        for j, idx, cxj, cyj in self._mem_links:
            live = idx[fluid_flat[idx]]
            if live.size == 0:
                continue
            int_mom = float(f_post[j].ravel()[live].sum())
            F[0] += 2.0 * int_mom * cxj
            F[1] += 2.0 * int_mom * cyj
        return F


# ---------------------------------------------------------------------------
# Immersed boundary (Lagrangian markers) coupling
# ---------------------------------------------------------------------------

class ImmersedBody:
    """Lagrangian marker set exchanged with the LBM field.

    Marker positions are in *lattice index space* (cell centres at
    ``(i + 0.5, j + 0.5)``).  ``spread``/``gather`` use the Peskin 4-point
    kernel; the per-cell weights of every marker are normalised to 1 so that a
    constant force density is transferred without loss (discrete
    conservation).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, nx: int, ny: int):
        x = np.asarray(x, dtype=DTYPE)
        y = np.asarray(y, dtype=DTYPE)
        if x.shape != y.shape:
            raise ValueError("marker x/y must have identical shape")
        self.nx, self.ny = nx, ny
        self.move(x, y)

    def _build_weights(self) -> list:
        j0 = np.floor(self.y).astype(np.int64) - 1
        i0 = np.floor(self.x).astype(np.int64) - 1
        entries = []
        for dj in range(4):
            for di in range(4):
                j = j0 + dj
                i = i0 + di
                w = peskin_delta(self.y - j) * peskin_delta(self.x - i)
                valid = (j >= 0) & (j < self.ny) & (i >= 0) & (i < self.nx)
                entries.append((np.clip(j, 0, self.ny - 1),
                                np.clip(i, 0, self.nx - 1),
                                np.where(valid, w, 0.0)))
        return entries

    def gather(self, field: np.ndarray) -> np.ndarray:
        """Interpolate an Eulerian field (ny, nx) to the markers."""
        out = np.zeros_like(self.x)
        for j, i, w in self._weights:
            out += w * field[j, i]
        return out

    def spread(self, values: np.ndarray) -> np.ndarray:
        """Spread marker values (n, 2) onto a (2, ny, nx) Eulerian field."""
        n = self.ny * self.nx
        g = np.zeros((2, n), dtype=DTYPE)
        for j, i, w in self._weights:
            idx = j * self.nx + i
            g[0] += np.bincount(idx, weights=w * values[:, 0], minlength=n)
            g[1] += np.bincount(idx, weights=w * values[:, 1], minlength=n)
        return g.reshape(2, self.ny, self.nx)

    def move(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.array(x, dtype=DTYPE)
        self.y = np.array(y, dtype=DTYPE)
        self._weights = self._build_weights()


def ibm_force(
    body: ImmersedBody,
    ux: np.ndarray,
    uy: np.ndarray,
    target_velocity: np.ndarray,
    rho: float = 1.0,
    iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct-forcing IBM with multi-direct iterations.

    Parameters
    ----------
    body : ImmersedBody
    ux, uy : Eulerian velocity estimate (lattice)
    target_velocity : (n_markers, 2) desired marker velocity (lattice)
    rho : reference density (lattice, = 1)
    iterations : number of multi-direct sweeps

    Returns
    -------
    force_grid : (2, ny, nx) Eulerian force density (lattice units)
    delta_total : (n_markers, 2) accumulated marker velocity correction
    """
    force_grid = np.zeros((2, body.ny, body.nx), dtype=DTYPE)
    ux_w = ux.copy()
    uy_w = uy.copy()
    delta_total = np.zeros_like(target_velocity)

    for _ in range(max(1, iterations)):
        u_int = np.stack([body.gather(ux_w), body.gather(uy_w)], axis=1)
        delta = target_velocity - u_int
        delta_total += delta
        g = body.spread(rho * delta)
        force_grid += g
        ux_w = ux_w + g[0]
        uy_w = uy_w + g[1]

    return force_grid, delta_total