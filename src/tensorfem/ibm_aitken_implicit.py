"""Implicit immersed-boundary (IBM) FSI with Aitken dynamic relaxation.

Implements, inside TensorFEM, the *implicit* immersed-boundary direct-forcing
scheme with per-time-step fixed-point sub-iterations under-relaxed by Aitken's
Delta^2 rule -- the coupling recipe of the open-source LIFE solver (Lattice
Boltzmann - Immersed boundary - Finite Element, J. O'Connor 2019, GPLv3).

Ported recipe (line references to LIFE, read-only cross-check, no code copied):
``src/Objects.cpp:26-60`` (sub-iteration loop), ``:150-213`` (predictor +
Aitken relaxation), ``:235-321`` (epsilon conservation system),
``src/IBMNode.cpp:26-136`` (interpolate / direct force / spread), ``:144-204``
(support search + marker spacing), ``inc/Utils.h:220-233`` (3-point delta),
``src/FEMElement.cpp:99-116`` (corotational internal force) and
``src/FEMBody.cpp:244-289`` (sub-iteration residual, Aitken numerator /
denominator).  Numeric expectations used by the cross-validation benchmark come
from the published inventory
``TensorPlatform/docs/oss_inventory/FEAT-OSS-INVENTORY-LBM_LIFE.md`` (zero
hand-copy of source values: they are re-measured here).

Deliberate deviations from LIFE (documented in the benchmark report):
* markers coincide with the beam centreline nodes (LIFE offsets them to the
  structure surface and integrates element shape functions exactly);
* the fluid is a periodic D2Q9 BGK box without regularised inflow/outflow or
  wall boundary conditions -- the coupling algorithm, not the boundary
  treatment, is the object of this port;
* the structural tangent is obtained by central differencing the corotational
  internal force instead of a hand-coded geometric stiffness matrix.

Public API: :class:`IBMInterface`, :class:`AitkenRelaxation`,
:class:`CorotationalBeam2D`, :class:`LBMFluid2D`, :class:`ImplicitIBMFSI`,
:func:`beam_natural_frequencies`, :func:`cantilever_free_vibration`,
:func:`solve`.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass

import numpy as np

__all__ = [
    "STENCIL_WIDTH",
    "dirac_delta_3pt",
    "find_support",
    "marker_spacing",
    "epsilon_conservation",
    "IBMInterface",
    "AitkenRelaxation",
    "BeamProperties",
    "CorotationalBeam2D",
    "LBMFluid2D",
    "ImplicitIBMFSI",
    "beam_natural_frequencies",
    "cantilever_free_vibration",
    "solve",
]

#: Half-width of the 3-point kernel support (LIFE ``inc/Utils.h:220-233``).
STENCIL_WIDTH = 1.5
#: Support window: nearest site +-2 lattice units (LIFE ``IBMNode.cpp:144``).
SUPPORT_WINDOW = 2


def dirac_delta_3pt(dist):
    """3-point discrete Dirac delta with support 1.5 (Roma/Peskin family).

    ``(1 + sqrt(1 - 3 r^2)) / 3`` for ``|r| <= 1/2``,
    ``(5 - 3|r| - sqrt(1 - 3(1-|r|)^2)) / 6`` for ``1/2 < |r| < 3/2``, 0 else.
    The partition of unity ``sum_i delta(x - i) = 1`` holds to round-off.
    """
    a = np.abs(np.asarray(dist, dtype=float))
    out = np.zeros_like(a)
    inner = a <= 0.5
    mid = (a > 0.5) & (a < STENCIL_WIDTH)
    out[inner] = (1.0 + np.sqrt(np.maximum(1.0 - 3.0 * a[inner] ** 2, 0.0))) / 3.0
    out[mid] = (5.0 - 3.0 * a[mid]
                - np.sqrt(np.maximum(-3.0 * (1.0 - a[mid]) ** 2 + 1.0, 0.0))) / 6.0
    return out


def find_support(position, dx, shape):
    """Lattice sites ``(i, j)`` and weights ``w`` of one marker stencil."""
    nx, ny = shape
    px, py = float(position[0]) / dx, float(position[1]) / dx
    inear, jnear = int(round(px)), int(round(py))
    ii, jj, ww = [], [], []
    for i in range(inear - SUPPORT_WINDOW, inear + SUPPORT_WINDOW + 1):
        dxd = abs(px - i)
        if dxd >= STENCIL_WIDTH or i < 0 or i > nx - 1:
            continue
        wx = float(dirac_delta_3pt(np.array([dxd]))[0])
        for j in range(jnear - SUPPORT_WINDOW, jnear + SUPPORT_WINDOW + 1):
            dyd = abs(py - j)
            if dyd >= STENCIL_WIDTH or j < 0 or j > ny - 1:
                continue
            ii.append(i)
            jj.append(j)
            ww.append(wx * float(dirac_delta_3pt(np.array([dyd]))[0]))
    return np.asarray(ii, dtype=int), np.asarray(jj, dtype=int), np.asarray(ww)


def marker_spacing(positions, dx):
    """``ds_i = min_{j != i} |x_i - x_j| / dx`` (LIFE ``IBMNode.cpp:182-204``)."""
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if n < 2:
        return np.ones(n)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2)) / dx
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def epsilon_conservation(positions, supports, ds, dx):
    """Dense epsilon system ``A eps = 1``, ``A_ij = sum_s d_i d_j ds_j``.

    Pinelli et al. (2010) redistribution solved as in LIFE
    ``src/Objects.cpp:235-321``; it makes the transmission of a *uniform*
    marker force to the lattice exact to round-off.
    """
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    a = np.zeros((n, n))
    for i in range(n):
        ii, jj, ww = supports[i]
        for j in range(n):
            dj = (dirac_delta_3pt(np.abs(pos[j, 0] / dx - ii))
                  * dirac_delta_3pt(np.abs(pos[j, 1] / dx - jj)))
            a[i, j] = float(np.dot(ww, dj)) * ds[j]
    return np.linalg.solve(a, np.ones(n))


class IBMInterface:
    """Immersed-boundary operators on a 2-D lattice (site ``(i, j)``).

    Parameters
    ----------
    positions : (n, 2) array of marker coordinates in physical units.
    shape : (nx, ny) lattice size, ``dx`` lattice spacing.
    correct_epsilon : apply the epsilon conservation correction (LIFE default)
        or use ``eps = 1`` (force loss).
    """

    def __init__(self, positions, shape, dx=1.0, correct_epsilon=True):
        self.dx = float(dx)
        self.shape = (int(shape[0]), int(shape[1]))
        self.correct_epsilon = bool(correct_epsilon)
        self.positions = np.asarray(positions, dtype=float).copy()
        self.move(self.positions)

    @property
    def eps_used(self):
        return self.eps if self.correct_epsilon else np.ones_like(self.eps)

    def move(self, positions):
        """Move markers and refresh support / spacing / epsilon."""
        self.positions = np.asarray(positions, dtype=float).copy()
        self.ds = marker_spacing(self.positions, self.dx)
        self.supports = [find_support(p, self.dx, self.shape) for p in self.positions]
        if not all(len(s[0]) for s in self.supports):
            raise ValueError("a marker stencil left the lattice domain")
        self.eps = epsilon_conservation(self.positions, self.supports, self.ds, self.dx)

    def interpolate(self, rho, u):
        """Euler -> Lagrange: marker density and momentum (LIFE ``:26-48``)."""
        rho = np.asarray(rho)
        u = np.asarray(u)
        n = len(self.positions)
        rho_i = np.zeros(n)
        mom_i = np.zeros((n, 2))
        for k, (ii, jj, ww) in enumerate(self.supports):
            vals = rho[ii, jj]
            rho_i[k] = float(np.dot(vals, ww))
            mom_i[k, 0] = float(np.dot(vals * u[ii, jj, 0], ww))
            mom_i[k, 1] = float(np.dot(vals * u[ii, jj, 1], ww))
        return rho_i, mom_i

    def force(self, rho_i, mom_i, u_body, dt=1.0):
        """Implicit direct forcing ``F = 2 (dt/dx rho_i u_body - m_i)``."""
        return 2.0 * ((dt / self.dx) * np.asarray(rho_i)[:, None] * np.asarray(u_body)
                      - np.asarray(mom_i))

    def spread(self, forces):
        """Lagrange -> Euler: ``f += F_i eps_i ds_i delta`` (LIFE ``:61-94``)."""
        f = np.zeros(self.shape + (2,))
        ep = self.eps_used
        for k, (ii, jj, ww) in enumerate(self.supports):
            f[ii, jj, 0] += forces[k, 0] * ep[k] * self.ds[k] * ww
            f[ii, jj, 1] += forces[k, 1] * ep[k] * self.ds[k] * ww
        return f

    def uniform_force_transmission_error(self):
        """``max_i |interp_i(spread(F = e_x)) - 1|`` -- what epsilon targets."""
        f = self.spread(np.tile([1.0, 0.0], (len(self.positions), 1)))
        err = 0.0
        for k, (ii, jj, ww) in enumerate(self.supports):
            err = max(err, abs(float(np.dot(f[ii, jj, 0], ww)) - 1.0))
        return err


class AitkenRelaxation:
    """Aitken Delta^2 (Irons-Tuck) dynamic under-relaxation.

    Update rule as used by LIFE (``src/Objects.cpp:184-190``)::

        R_k     = U_k - U_km1
        omega  <- -omega * (R_km1 . (R_k - R_km1)) / |R_k - R_km1|^2

    the first sub-iteration of each step is clipped to ``+-relax_max``.  For a
    scalar linear fixed point ``U -> g U + b`` the converged value is the
    analytically optimal ``omega* = 1 / (1 - g)``.
    """

    def __init__(self, relax_max=1.0, relax0=1.0):
        self.relax_max = float(relax_max)
        self.omega = float(relax0)
        self.num = 0.0
        self.den = 0.0
        self.history = []

    def reset(self, relax0=None):
        self.omega = float(relax0 if relax0 is not None else self.omega)
        self.num = self.den = 0.0
        self.history = []

    def omega_for(self, r_km1, r_k):
        """Return the relaxation factor to use for the *next* correction."""
        r_km1 = np.asarray(r_km1, dtype=float)
        r_k = np.asarray(r_k, dtype=float)
        dr = r_k - r_km1
        self.num = float(np.dot(r_km1, dr))
        self.den = float(np.dot(dr, dr))
        if self.den > 0.0:
            omega = -self.omega * self.num / self.den
        else:
            omega = self.omega
        self.omega = math.copysign(min(abs(omega), self.relax_max), omega)
        self.history.append(self.omega)
        return self.omega


@dataclass
class BeamProperties:
    """Beam data; ``rho`` is mass **per unit length**."""

    E: float
    A: float
    I: float
    rho: float
    length: float
    thickness: float = 1.0


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class CorotationalBeam2D:
    """2-node corotational beam (axial + Euler-Bernoulli bending, 3 DOF/node).

    Internal force law identical to LIFE ``src/FEMElement.cpp:99-116``
    (``F0 = EA/L0 (L^2-L0^2)/(L+L0)``, ``M1 = 2EI/L0 (2 th1 + th2)``,
    ``M2 = 2EI/L0 (th1 + 2 th2)``, rotated into the element frame).  Time
    integration is implicit Newmark (``beta = 1/4``, ``gamma = 1/2``) with a
    Newton loop whose tangent is the central difference of that force law.
    """

    def __init__(self, n_elements=16, props=None):
        self.props = props or BeamProperties(E=1.0e5, A=1.0, I=1.0 / 12.0,
                                             rho=1.0, length=1.0, thickness=1.0)
        self.n_elements = int(n_elements)
        self.n_nodes = self.n_elements + 1
        self.n_dof = 3 * self.n_nodes
        x = np.linspace(0.0, self.props.length, self.n_nodes)
        self.x0 = np.stack((x, np.zeros_like(x)), axis=1)
        self.u = np.zeros(self.n_dof)
        self.v = np.zeros(self.n_dof)
        self.a = np.zeros(self.n_dof)
        self.u_n = self.u.copy()
        self.u_nm1 = self.u.copy()
        self.u_nm2 = self.u.copy()
        self.v_n = self.v.copy()
        self.a_n = self.a.copy()
        self.free = np.arange(3, self.n_dof)      # node 0 clamped (u, w, theta)
        self._assemble_constant_matrices()

    # -- matrices -----------------------------------------------------------
    def _element_matrices(self):
        p = self.props
        L0 = p.length / self.n_elements
        idx_b = [1, 2, 4, 5]
        m = np.zeros((6, 6))
        ml = p.rho * L0
        m[np.ix_([0, 3], [0, 3])] += ml / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        m[np.ix_(idx_b, idx_b)] += ml / 420.0 * np.array(
            [[156.0, 22 * L0, 54.0, -13 * L0],
             [22 * L0, 4 * L0 ** 2, 13 * L0, -3 * L0 ** 2],
             [54.0, 13 * L0, 156.0, -22 * L0],
             [-13 * L0, -3 * L0 ** 2, -22 * L0, 4 * L0 ** 2]])
        k = np.zeros((6, 6))
        k[np.ix_([0, 3], [0, 3])] += (p.E * p.A / L0) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        ei = p.E * p.I / L0 ** 3
        k[np.ix_(idx_b, idx_b)] += ei * np.array(
            [[12.0, 6 * L0, -12.0, 6 * L0],
             [6 * L0, 4 * L0 ** 2, -6 * L0, 2 * L0 ** 2],
             [-12.0, -6 * L0, 12.0, -6 * L0],
             [6 * L0, 2 * L0 ** 2, -6 * L0, 4 * L0 ** 2]])
        return L0, m, k

    def _assemble_constant_matrices(self):
        _, m, k = self._element_matrices()
        self.M = np.zeros((self.n_dof, self.n_dof))
        self.K_lin = np.zeros((self.n_dof, self.n_dof))
        for e in range(self.n_elements):
            sl = slice(3 * e, 3 * e + 6)
            self.M[sl, sl] += m
            self.K_lin[sl, sl] += k

    # -- kinematics ---------------------------------------------------------
    def markers(self, u=None):
        """IBM markers: beam centreline nodes in the deformed state."""
        u = self.u if u is None else np.asarray(u)
        return self.x0 + u.reshape(self.n_nodes, 3)[:, :2]

    def node_velocities(self):
        return self.v.reshape(self.n_nodes, 3)[:, :2].copy()

    def tip(self, u=None):
        u = self.u if u is None else np.asarray(u)
        return np.array([self.x0[-1, 0] + u[-3], u[-2]])

    # -- element force ------------------------------------------------------
    def internal_force(self, u):
        p = self.props
        L0 = p.length / self.n_elements
        u = np.asarray(u, dtype=float)
        f = np.zeros(self.n_dof)
        for e in range(self.n_elements):
            n0, n1 = e, e + 1
            dx = (self.x0[n1, 0] - self.x0[n0, 0]) + u[3 * n1] - u[3 * n0]
            dy = (self.x0[n1, 1] - self.x0[n0, 1]) + u[3 * n1 + 1] - u[3 * n0 + 1]
            L = math.hypot(dx, dy)
            phi = math.atan2(dy, dx)
            th1 = _wrap(u[3 * n0 + 2] - phi)
            th2 = _wrap(u[3 * n1 + 2] - phi)
            f0 = p.E * p.A / L0 * (L ** 2 - L0 ** 2) / (L + L0)
            m1 = 2.0 * p.E * p.I / L0 * (2.0 * th1 + th2)
            m2 = 2.0 * p.E * p.I / L0 * (th1 + 2.0 * th2)
            floc = np.array([-f0, (m1 + m2) / L0, m1, f0, -(m1 + m2) / L0, m2])
            c, s = math.cos(phi), math.sin(phi)
            T = np.array([[c, s, 0, 0, 0, 0],
                          [-s, c, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, c, s, 0],
                          [0, 0, 0, -s, c, 0],
                          [0, 0, 0, 0, 0, 1]])
            f[3 * e:3 * e + 6] += T.T @ floc
        return f

    def tangent(self, u, rel_step=1e-7):
        """Central-difference tangent of :meth:`internal_force`."""
        u = np.asarray(u, dtype=float)
        h = rel_step * self.props.length
        f0 = self.internal_force(u)
        k = np.zeros((self.n_dof, self.n_dof))
        for j in range(self.n_dof):
            up = u.copy()
            up[j] += h
            k[:, j] = (self.internal_force(up) - f0) / h
        return 0.5 * (k + k.T)

    def strain_energy(self, u):
        p = self.props
        L0 = p.length / self.n_elements
        u = np.asarray(u, dtype=float)
        e = 0.0
        for el in range(self.n_elements):
            n0, n1 = el, el + 1
            dx = (self.x0[n1, 0] - self.x0[n0, 0]) + u[3 * n1] - u[3 * n0]
            dy = (self.x0[n1, 1] - self.x0[n0, 1]) + u[3 * n1 + 1] - u[3 * n0 + 1]
            L = math.hypot(dx, dy)
            phi = math.atan2(dy, dx)
            th1 = _wrap(u[3 * n0 + 2] - phi)
            th2 = _wrap(u[3 * n1 + 2] - phi)
            e += 0.5 * p.E * p.A / L0 * (L - L0) ** 2
            e += 2.0 * p.E * p.I / L0 * (th1 ** 2 + th2 ** 2 + th1 * th2)
        return e

    # -- loads --------------------------------------------------------------
    def apply_marker_loads(self, marker_forces):
        """Lumped nodal projection of Lagrange marker forces (markers = nodes)."""
        load = np.zeros(self.n_dof)
        f = np.asarray(marker_forces, dtype=float)
        for k in range(min(len(f), self.n_nodes)):
            load[3 * k] += f[k, 0]
            load[3 * k + 1] += f[k, 1]
        return load

    # -- time integration ---------------------------------------------------
    def predict(self, t_index):
        """Second-order displacement predictor (LIFE ``FEMBody.cpp:262-286``)."""
        if t_index > 2:
            u = 2.5 * self.u_n - 2.0 * self.u_nm1 + 0.5 * self.u_nm2
        elif t_index == 2:
            u = 2.0 * self.u_n - self.u_nm1
        else:
            u = self.u_n.copy()
        u[:3] = 0.0
        return u

    def newmark_solve(self, load, dt, beta=0.25, gamma=0.5, tol=1e-10, max_iter=6):
        """Implicit Newmark step from the current state; returns diagnostics."""
        free = self.free
        u_pred = self.u_n + dt * self.v_n + dt ** 2 * (0.5 - beta) * self.a_n
        u_pred[:3] = 0.0
        acc_factor = 1.0 / (beta * dt ** 2)
        u = np.asarray(self.u, dtype=float).copy()
        u[:3] = 0.0
        it, res = 0, float("inf")
        for it in range(1, max_iter + 1):
            a = (u - u_pred) * acc_factor
            r = self.M @ a + self.internal_force(u) - load
            res = float(np.linalg.norm(r[free]))
            scale = max(1.0, float(np.linalg.norm(load[free])))
            if res <= tol * scale:
                break
            keff = self.tangent(u) + self.M * acc_factor
            u[free] += np.linalg.solve(keff[np.ix_(free, free)], -r[free])
        self.u = u
        self.a = (u - u_pred) * acc_factor
        self.v = self.v_n + dt * ((1.0 - gamma) * self.a_n + gamma * self.a)
        return u, res <= tol * max(1.0, float(np.linalg.norm(load[free]))), it, res

    def advance_history(self):
        self.u_nm2 = self.u_nm1.copy()
        self.u_nm1 = self.u_n.copy()
        self.u_n = self.u.copy()
        self.v_n = self.v.copy()
        self.a_n = self.a.copy()

    def set_initial_deflection(self, amplitude):
        """Static clamped-beam deflection shape of the given tip amplitude."""
        x = self.x0[:, 0] / self.props.length
        u = np.zeros(self.n_dof)
        u[1::3] = amplitude * (3.0 * x ** 2 - x ** 3) / 2.0
        u[2::3] = amplitude * (6.0 * x - 3.0 * x ** 2) / (2.0 * self.props.length)
        u[:3] = 0.0
        self.u = u.copy()
        self.u_n = u.copy()
        self.u_nm1 = u.copy()
        self.u_nm2 = u.copy()
        self.v[:] = 0.0
        self.a[:] = 0.0
        self.v_n[:] = 0.0
        self.a_n[:] = 0.0


def beam_natural_frequencies(beam, n_modes=3):
    """Eigen-frequencies (Hz) of the linearised clamped beam (``K_lin, M``).

    Cholesky whitening of the generalised problem plus ``numpy.linalg.eigvalsh``:
    ``scipy.linalg.eigh`` segfaults on this loong64 image (see the X-val report
    "friction" section), so the LAPACK backend is numpy's.
    """
    free = beam.free
    k = beam.K_lin[np.ix_(free, free)]
    m = beam.M[np.ix_(free, free)]
    lc = np.linalg.cholesky(m)
    a = np.linalg.solve(lc, np.linalg.solve(lc, k).T).T
    lam = np.linalg.eigvalsh(0.5 * (a + a.T))
    f = np.sqrt(np.clip(np.sort(lam), 0.0, None)) / (2.0 * math.pi)
    return f[:n_modes]


class LBMFluid2D:
    """D2Q9 BGK fluid on a periodic box with Guo (half-split) forcing.

    ``rho = sum f``, ``u = (sum c f + F/2) / rho``,
    ``f <- f - omega (f - f_eq) + (1 - omega/2) F_i`` with the second-order
    Hermite equilibrium.  No inflow/outflow or wall boundary conditions: the
    coupling algorithm, not the boundary treatment, is under test.
    """

    D = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
    W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

    def __init__(self, shape, omega=1.8, rho0=1.0):
        self.shape = (int(shape[0]), int(shape[1]))
        self.omega = float(omega)
        self.rho = np.full(self.shape, float(rho0))
        self.u = np.zeros(self.shape + (2,))
        self.force_ibm = np.zeros(self.shape + (2,))
        self.f = self.equilibrium(self.rho, self.u)

    def equilibrium(self, rho, u):
        c = self.D
        cu = u[..., 0, None] * c[:, 0] + u[..., 1, None] * c[:, 1]
        usq = (u ** 2).sum(axis=-1)[..., None]
        return self.W * np.asarray(rho)[..., None] * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq)

    def macroscopic(self):
        f, c = self.f, self.D
        rho = f.sum(axis=-1)
        mom = (f[..., :, None] * c[None, None, :, :]).sum(axis=-2) + 0.5 * self.force_ibm
        return rho, mom / rho[..., None]

    def step(self):
        rho, u = self.macroscopic()
        self.rho, self.u = rho, u
        f_eq = self.equilibrium(rho, u)
        cu = u[..., 0, None] * self.D[:, 0] + u[..., 1, None] * self.D[:, 1]
        cf = self.D[:, 0] * self.force_ibm[..., 0, None] + self.D[:, 1] * self.force_ibm[..., 1, None]
        src = (1.0 - 0.5 * self.omega) * 3.0 * self.W * (cf + 3.0 * cu * cf)
        f = self.f - self.omega * (self.f - f_eq) + src
        self.f = np.empty_like(f)
        for a in range(9):
            self.f[..., a] = np.roll(np.roll(f[..., a], int(self.D[a, 0]), axis=0),
                                     int(self.D[a, 1]), axis=1)
        return rho, u


class ImplicitIBMFSI:
    """Block Gauss-Seidel sub-iteration FSI loop with Aitken relaxation.

    One :meth:`step` advances the fluid once (as LIFE does), drives the
    interface fixed point to ``sub_tol`` and spreads the IBM force only after
    convergence.
    """

    def __init__(self, beam, fluid, ibm=None, dt=1.0, dx=1.0, ref_length=None,
                 sub_tol=1e-8, max_sub_iter=20, relax_max=1.0,
                 relax_mode="aitken", fixed_omega=1.0, correct_epsilon=True):
        self.beam = beam
        self.fluid = fluid
        self.dt = float(dt)
        self.dx = float(dx)
        self.ref_length = float(ref_length if ref_length is not None else beam.props.length)
        self.sub_tol = float(sub_tol)
        self.max_sub_iter = int(max_sub_iter)
        self.relax_mode = relax_mode
        self.fixed_omega = float(fixed_omega)
        self.ibm = ibm or IBMInterface(beam.markers(), fluid.shape, dx=dx,
                                       correct_epsilon=correct_epsilon)
        self.aitken = AitkenRelaxation(relax_max=relax_max, relax0=relax_max)
        self.sub_iterations = []
        self.n_maxit_hits = 0

    def step(self, t_index):
        self.fluid.step()                      # fluid advances once per step
        rho, u = self.fluid.macroscopic()
        beam = self.beam
        beam.u = beam.predict(t_index)
        u_km1 = beam.u.copy()
        r_km1 = None
        omega = self.aitken.relax_max if self.relax_mode == "aitken" else self.fixed_omega
        converged, res, it = False, float("inf"), 0
        for it in range(1, self.max_sub_iter + 1):
            self.ibm.move(beam.markers())
            rho_i, mom_i = self.ibm.interpolate(rho, u)
            f_lag = self.ibm.force(rho_i, mom_i, beam.node_velocities(), dt=self.dt)
            load = beam.apply_marker_loads(-f_lag)          # reaction on the body
            u_new, _ok, _nit, _nres = beam.newmark_solve(load, self.dt)
            r_k = u_new - u_km1
            res = float(np.linalg.norm(r_k)) / (self.ref_length * math.sqrt(len(beam.free)))
            if res <= self.sub_tol:
                converged = True
                break
            if self.relax_mode == "aitken" and it >= 2:
                omega = self.aitken.omega_for(r_km1, r_k)
            else:
                omega = self.aitken.relax_max if self.relax_mode == "aitken" else self.fixed_omega
            r_km1 = r_k.copy()
            beam.u = u_km1 + omega * (u_new - u_km1)
            u_km1 = beam.u.copy()  # update for next sub-iteration
        if not converged:
            self.n_maxit_hits += 1
        self.ibm.move(beam.markers())
        rho_i, mom_i = self.ibm.interpolate(rho, u)
        f_lag = self.ibm.force(rho_i, mom_i, beam.node_velocities(), dt=self.dt)
        self.fluid.force_ibm = self.ibm.spread(f_lag)
        beam.advance_history()
        self.sub_iterations.append(it)
        return {"sub_iterations": it, "residual": res, "converged": converged,
                "omega": omega, "tip": beam.tip().tolist()}

    def run(self, n_steps, t_start=1):
        """Run ``n_steps`` steps; a diverging interface (marker leaving the
        lattice) is recorded instead of raised, so that a failed relaxation
        scheme is *evidence* rather than a crash."""
        hist = []
        self.diverged = False
        self.diverged_reason = ""
        for k in range(n_steps):
            try:
                hist.append(self.step(t_start + k))
            except (ValueError, FloatingPointError) as exc:
                self.diverged = True
                self.diverged_reason = str(exc)
                break
        return hist


def _estimate_period(series, dt):
    """Period from upward zero crossings of a (damped) oscillation."""
    y = np.asarray(series, dtype=float)
    y = y - y.mean()
    idx = np.where((y[:-1] <= 0.0) & (y[1:] > 0.0))[0]
    if len(idx) < 2:
        return float("nan")
    return float((idx[-1] - idx[0]) * dt / (len(idx) - 1))


def _section():
    return BeamProperties(E=1.0e5, A=1.0, I=1.0 / 12.0, rho=1.0,
                          length=1.0, thickness=1.0)


def cantilever_free_vibration(n_steps=400, n_elements=16, dt=2.0e-3,
                              amplitude=0.05, fluid_rho=0.0, lattice=(96, 96),
                              relax_mode="aitken", fixed_omega=1.0,
                              sub_tol=1e-8, max_sub_iter=20,
                              correct_epsilon=True):
    """Clamped beam released from a static deflection, optionally submerged.

    ``fluid_rho = 0`` runs the structure alone (vacuum reference); otherwise a
    D2Q9 box initially at rest surrounds the beam and the implicit IBM/Aitken
    coupling transfers the hydrodynamic load.
    """
    beam = CorotationalBeam2D(n_elements=n_elements, props=_section())
    beam.set_initial_deflection(amplitude)
    if fluid_rho <= 0.0:
        tips = []
        for k in range(1, n_steps + 1):
            beam.u = beam.predict(k)
            beam.newmark_solve(np.zeros(beam.n_dof), dt)
            beam.advance_history()
            tips.append(beam.tip()[1])
        period = _estimate_period(tips, dt)
        return {"mode": "vacuum", "dt": dt, "n_steps": n_steps,
                "tip_history": tips, "period": period,
                "frequency": 1.0 / period if period == period else float("nan"),
                "sub_iterations": [], "maxit_hits": 0,
                "eigen_frequencies": beam_natural_frequencies(beam).tolist()}
    # scale the unit beam into the lattice box
    scale = 0.45 * float(lattice[0])
    beam.x0 = beam.x0 * scale + np.array([0.15 * lattice[0], 0.5 * lattice[1]])
    beam.props = BeamProperties(E=beam.props.E / scale, A=beam.props.A * scale,
                                I=beam.props.I * scale ** 3, rho=beam.props.rho,
                                length=beam.props.length * scale,
                                thickness=beam.props.thickness * scale)
    beam._assemble_constant_matrices()
    beam.set_initial_deflection(amplitude * scale)
    fluid = LBMFluid2D(lattice, omega=1.8, rho0=fluid_rho)
    solver = ImplicitIBMFSI(beam, fluid, dt=dt, dx=1.0, ref_length=beam.props.length,
                            sub_tol=sub_tol, max_sub_iter=max_sub_iter,
                            relax_mode=relax_mode, fixed_omega=fixed_omega,
                            correct_epsilon=correct_epsilon)
    hist = solver.run(n_steps, t_start=1)
    tips = [h["tip"][1] for h in hist]
    period = _estimate_period(tips, dt)
    return {"mode": "submerged", "dt": dt, "n_steps": n_steps,
            "steps_completed": len(hist),
            "diverged": bool(getattr(solver, "diverged", False)),
            "diverged_reason": getattr(solver, "diverged_reason", ""),
            "tip_history": tips, "period": period,
            "frequency": 1.0 / period if period == period else float("nan"),
            "sub_iterations": [h["sub_iterations"] for h in hist],
            "maxit_hits": solver.n_maxit_hits,
            "converged_all": all(h["converged"] for h in hist),
            "relax": [h["omega"] for h in hist],
            "lattice": list(lattice), "beam_scale": scale}


def solve(kind="vacuum", **kwargs):
    """Generic entry: ``kind`` in ``{"vacuum", "submerged"}``."""
    if kind == "vacuum":
        kwargs.setdefault("fluid_rho", 0.0)
    return cantilever_free_vibration(**kwargs)


def main(argv=None):
    """CLI: run the vacuum or submerged cantilever and dump the record as JSON."""
    argv = list(sys.argv[1:] if argv is None else argv)
    kind = argv[0] if argv else "vacuum"
    kw = {}
    for token in argv[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            kw[k] = json.loads(v)
    rec = solve(kind, **kw)
    print(json.dumps(rec))
    return rec


if __name__ == "__main__":        # pragma: no cover
    main()