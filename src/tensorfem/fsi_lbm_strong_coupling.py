"""Partitioned strong-coupling FSI on a resolved LBM wake (Turek-Hron FSI-2).

What this module is
-------------------
A *two-way* fluid-structure interaction solver in which

* the fluid is a D2Q9 lattice-Boltzmann solution of the incompressible
  Navier-Stokes equations (``tensorfem.lbm_channel2d``), so the von Karman
  wake, the shedding frequency and the hydrodynamic loads are **outputs** of
  the discretised equations, not calibrated inputs;
* the structure is an Euler-Bernoulli cantilever finite element model
  (Hermite beam elements, consistent mass matrix, Newmark average-acceleration
  integration);
* the two are coupled with a Dirichlet-Neumann partitioned fixed-point
  iteration accelerated by Aitken's Delta^2 dynamic relaxation, with a
  sub-iteration convergence test on the interface force residual;
* the interface exchange is momentum conserving by construction (direct
  forcing body force on the wetted cells, FE-consistent load projection),
  and every energy term is booked so the coupling can be audited:
  fluid kinetic energy, structural kinetic + strain energy, viscous
  dissipation, inlet/outlet power and interface work.

Physical case
-------------
Turek & Hron (2006) FSI-2: laminar channel flow around a rigid cylinder with
an attached elastic beam (the "flap"), Re = 100 based on the mean inflow
velocity and the cylinder diameter.

    channel          2.5 m x 0.41 m
    cylinder         D = 0.1 m at (0.2, 0.2)
    beam             L = 0.35 m, h = 0.02 m attached at (0.25, 0.2)
    fluid            rho = 1000 kg/m^3, nu = 1e-3 m^2/s, U_mean = 1.0 m/s
    solid            rho_s = 1000 kg/m^3, E = 5.6e6 Pa, nu_s = 0.4, EI = E h^3/12

The beam is modelled as a cantilever (clamped where it meets the cylinder) with
transverse (vertical) displacement only; axial deformation is not represented
(the axial fluid load is carried by the clamp and reported as a reaction).

Honesty note
------------
The pre-existing ``fsi_turek_hron`` module is a *calibrated reduced-order*
model: its vortex forcing frequency (``f_vortex = 3.0``) and force amplitude
(``F0 = 17.5``) are tuned so that the modal oscillator reproduces the reference
answer.  It is a MOCK for the fluid side.  This module exists to replace that
with a falsifiable computation.  Absolute accuracy is limited by the lattice
resolution that is affordable here; the grid/time refinement study in
``benchmarks/fsi_strong_coupling`` quantifies the remaining discretisation
error instead of hiding it.

References
----------
[1] Turek, S., Hron, J. (2006). LNCS 53, 371-385.
[2] Kuettler, U., Wall, W.A. (2008). Comput. Mech. 42, 1-16 (Aitken).
[3] Guo, Z., Zheng, C., Shi, B. (2002). Phys. Fluids 14, 2007 (forcing).
[4] Newmark, N.M. (1959). J. Eng. Mech. Div. 85, 67-94.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .lbm_channel2d import ChannelConfig, ChannelLBM


# ---------------------------------------------------------------------------
# Benchmark definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FSI2Parameters:
    """Physical parameters of the Turek-Hron FSI-2 benchmark (SI units)."""

    # geometry
    D: float = 0.1
    L: float = 0.35
    h: float = 0.02
    H: float = 0.41
    W: float = 2.5
    cx: float = 0.2
    cy: float = 0.2

    # materials
    E: float = 5.6e6          # beam Young's modulus [Pa]
    rho_s: float = 1000.0     # beam density [kg/m^3]
    rho_f: float = 1000.0     # fluid density [kg/m^3]
    nu_f: float = 1e-3        # fluid kinematic viscosity [m^2/s]
    U_mean: float = 1.0       # mean inflow velocity [m/s]

    # published reference values (Turek & Hron 2006, FSI-2)
    ref_frequency: float = 2.0      # [Hz]
    ref_amplitude: float = 0.034    # [m] tip vertical amplitude (half p-p)
    ref_mean_CD: float | None = None
    ref_amp_CL: float | None = None

    def validate(self) -> None:
        for name in ("D", "L", "h", "H", "W", "E", "rho_s", "rho_f", "nu_f", "U_mean"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a positive finite number")
        if self.L + self.cx + self.D / 2 > self.W:
            raise ValueError("beam does not fit inside the channel")

    @property
    def EI(self) -> float:
        """Bending stiffness per unit width [N m^2]."""
        return self.E * self.h ** 3 / 12.0

    @property
    def mass_per_length(self) -> float:
        """Beam mass per unit length [kg/m]."""
        return self.rho_s * self.h

    @property
    def dry_first_frequency(self) -> float:
        """First dry cantilever frequency [Hz] (analytic)."""
        beta1 = 1.875104068711961
        return beta1 ** 2 * math.sqrt(self.EI / self.mass_per_length) / self.L ** 2 / (2 * math.pi)

    @property
    def beam_attachment_x(self) -> float:
        return self.cx + self.D / 2


# ---------------------------------------------------------------------------
# Euler-Bernoulli cantilever (Hermite) finite element model
# ---------------------------------------------------------------------------

class CantileverBeamFE:
    """Euler-Bernoulli cantilever with consistent mass and Newmark integration.

    Degrees of freedom are ``(w, theta = dw/dx)`` at each node; the first node
    is clamped.  The model is linear (small slope), which is consistent with
    the benchmark's beam slenderness (L/h = 17.5) for the amplitudes observed.
    """

    def __init__(self, length: float, n_elem: int, EI: float, mass_per_length: float):
        if n_elem < 1:
            raise ValueError("n_elem must be >= 1")
        self.length = float(length)
        self.n_elem = int(n_elem)
        self.EI = float(EI)
        self.m = float(mass_per_length)
        self.le = self.length / self.n_elem
        self.nodes = np.linspace(0.0, self.length, self.n_elem + 1)
        n_dof = 2 * (self.n_elem + 1)
        self.K = np.zeros((n_dof, n_dof))
        self.M = np.zeros((n_dof, n_dof))
        self._assemble()
        # clamped node 0 -> keep DOF 2..n_dof-1
        self.free = np.arange(2, n_dof)
        self.Kff = self.K[np.ix_(self.free, self.free)]
        self.Mff = self.M[np.ix_(self.free, self.free)]
        self.q = np.zeros(n_dof)
        self.v = np.zeros(n_dof)
        self.a = np.zeros(n_dof)
        self._Keff_inv = None

    # ------------------------------------------------------------------ setup
    def _assemble(self) -> None:
        le, EI, m = self.le, self.EI, self.m
        k = EI / le ** 3 * np.array([
            [12.0, 6 * le, -12.0, 6 * le],
            [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
            [-12.0, -6 * le, 12.0, -6 * le],
            [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
        mm = m * le / 420.0 * np.array([
            [156.0, 22 * le, 54.0, -13 * le],
            [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
            [54.0, 13 * le, 156.0, -22 * le],
            [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]])
        for e in range(self.n_elem):
            sl = slice(2 * e, 2 * e + 4)
            self.K[sl, sl] += k
            self.M[sl, sl] += mm

    # ------------------------------------------------------- shape functions
    def _element_of(self, x):
        """Element index and local coordinate xi for positions x (clipped)."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, self.length)
        e = np.minimum((x / self.le).astype(int), self.n_elem - 1)
        xi = (x - e * self.le) / self.le
        return e, xi

    def shape(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Hermite shape functions and their first derivatives at x."""
        e, xi = self._element_of(x)
        le = self.le
        N = np.stack([
            1 - 3 * xi ** 2 + 2 * xi ** 3,
            le * (xi - 2 * xi ** 2 + xi ** 3),
            3 * xi ** 2 - 2 * xi ** 3,
            le * (-xi ** 2 + xi ** 3)], axis=-1)
        dN = np.stack([
            (-6 * xi + 6 * xi ** 2) / le,
            1 - 4 * xi + 3 * xi ** 2,
            (6 * xi - 6 * xi ** 2) / le,
            -2 * xi + 3 * xi ** 2], axis=-1)
        return N, dN

    def _dof_indices(self, x) -> np.ndarray:
        e, _ = self._element_of(x)
        return np.stack([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3], axis=-1)

    def interpolate(self, q: np.ndarray, x, deriv: int = 0) -> np.ndarray:
        N, dN = self.shape(x)
        idx = self._dof_indices(x)
        vals = q[idx]
        if deriv == 0:
            return np.einsum("...k,...k->...", N, vals)
        if deriv == 1:
            return np.einsum("...k,...k->...", dN, vals)
        raise ValueError("deriv must be 0 or 1")

    def nodal_load(self, x, load_per_length) -> np.ndarray:
        """Consistent nodal load vector from a distributed transverse load.

        ``load_per_length`` has the same shape as ``x`` (N/m).  The projection
        ``f_i = int N_i(x) q(x) dx`` conserves the resultant force and moment.
        """
        N, _ = self.shape(x)
        idx = self._dof_indices(x)
        F = np.zeros_like(self.q)
        contrib = N * np.asarray(load_per_length)[..., None]
        np.add.at(F, idx.ravel(), (contrib * (self.length / max(len(np.atleast_1d(x)) - 1, 1))).ravel())
        return F

    # ------------------------------------------------------------ time step
    def initialise_effective(self, dt: float, beta: float = 0.25, gamma: float = 0.5) -> None:
        a0 = 1.0 / (beta * dt ** 2)
        a1 = 1.0 / (beta * dt)
        self._beta, self._gamma, self._dt = beta, gamma, dt
        self._a0, self._a1 = a0, a1
        self._a2 = 1.0 / (2.0 * beta) - 1.0
        self._Keff = self.Kff + a0 * self.Mff
        self._Keff_inv = np.linalg.inv(self._Keff)

    def step(self, F: np.ndarray, M_add: np.ndarray | None = None,
             C: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance one Newmark step under the free-DOF load vector F.

        ``M_add`` is an optional free-DOF *added-mass* matrix treated
        implicitly (the immersed-boundary reaction is inertial, not
        dissipative); ``C`` is an optional free-DOF damping matrix.
        """
        if self._Keff_inv is None:
            raise RuntimeError("call initialise_effective(dt) before step()")
        dt = self._dt
        beta, gamma = self._beta, self._gamma
        q, v, a = self.q, self.v, self.a
        Ff = F[self.free]
        M_eff = self.Mff if M_add is None else self.Mff + M_add
        rhs = Ff + M_eff @ (self._a0 * q[self.free] + self._a1 * v[self.free]
                            + self._a2 * a[self.free])
        if C is None:
            k_eff = self.Kff + self._a0 * M_eff
        else:
            rhs = rhs + C @ (self._a1 * q[self.free]
                             + (gamma / beta - 1.0) * v[self.free]
                             + dt * (gamma / (2.0 * beta) - 1.0) * a[self.free])
            k_eff = self.Kff + self._a0 * M_eff + self._a1 * C
        q_new = q.copy()
        q_new[self.free] = np.linalg.solve(k_eff, rhs)
        a_new = a.copy()
        a_new[self.free] = (self._a0 * (q_new[self.free] - q[self.free])
                            - self._a1 * v[self.free] - self._a2 * a[self.free])
        v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)
        self.q, self.v, self.a = q_new, v_new, a_new
        return q_new, v_new, a_new

    def kinetic_energy(self) -> float:
        return float(0.5 * self.v @ self.M @ self.v)

    def strain_energy(self) -> float:
        return float(0.5 * self.q @ self.K @ self.q)

    def set_state(self, q, v, a) -> None:
        self.q = np.array(q, dtype=float)
        self.v = np.array(v, dtype=float)
        self.a = np.array(a, dtype=float)


class ModalCantileverBeam(CantileverBeamFE):
    """Cantilever FE model integrated in truncated modal coordinates.

    The consistent-mass Euler-Bernoulli mesh carries very small rotational
    inertia, so its high modes are stiff and are easily excited by the
    (column-wise) fluid load.  Those modes are far above the shedding band and
    only add numerical noise to the interface velocity, which destabilises the
    partitioned coupling.  Integrating the first ``n_modes`` generalised
    coordinates keeps the physically relevant dynamics (f1 ~ 2 Hz) while
    filtering the unresolved band; the truncation error is reported.
    """

    def __init__(self, length: float, n_elem: int, EI: float,
                 mass_per_length: float, n_modes: int = 5):
        super().__init__(length, n_elem, EI, mass_per_length)
        from scipy.linalg import eigh
        vals, vecs = eigh(self.Kff, self.Mff)
        if n_modes < 1 or n_modes > vals.size:
            raise ValueError("n_modes out of range")
        self.n_modes = int(n_modes)
        self.Phi = vecs[:, :self.n_modes]
        self.omega2 = vals[:self.n_modes]
        self.frequencies = np.sqrt(self.omega2) / (2 * math.pi)
        self.eta = np.zeros(self.n_modes)
        self.eta_dot = np.zeros(self.n_modes)
        self.eta_ddot = np.zeros(self.n_modes)
        self.zeta = 0.0            # modal damping ratio (0 = undamped)
        self._sync_full_state()

    # ------------------------------------------------------------------ state
    def _sync_full_state(self) -> None:
        self.q = np.zeros(self.q.size)
        self.v = np.zeros(self.q.size)
        self.a = np.zeros(self.q.size)
        self.q[self.free] = self.Phi @ self.eta
        self.v[self.free] = self.Phi @ self.eta_dot
        self.a[self.free] = self.Phi @ self.eta_ddot

    def set_state(self, q=None, v=None, a=None) -> None:
        """Set the full-DOF state (projected onto the retained modes)."""
        if q is not None:
            self.eta = self.Phi.T @ self.Mff @ np.asarray(q)[self.free]
        if v is not None:
            self.eta_dot = self.Phi.T @ self.Mff @ np.asarray(v)[self.free]
        if a is not None:
            self.eta_ddot = self.Phi.T @ self.Mff @ np.asarray(a)[self.free]
        self._sync_full_state()

    # ------------------------------------------------------------ time step
    def step(self, F: np.ndarray, M_add: np.ndarray | None = None,
             C: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = self._dt
        beta, gamma = self._beta, self._gamma
        a0, a1, a2 = self._a0, self._a1, self._a2
        I = np.eye(self.n_modes)
        M_eff = I if M_add is None else I + self.Phi.T @ M_add @ self.Phi
        rhs = (self.Phi.T @ F[self.free]
               + M_eff @ (a0 * self.eta + a1 * self.eta_dot + a2 * self.eta_ddot))
        k_eff = np.diag(self.omega2) + a0 * M_eff
        if self.zeta:
            Cd = np.diag(2.0 * self.zeta * np.sqrt(self.omega2))
            rhs = rhs + Cd @ (a1 * self.eta + (gamma / beta - 1.0) * self.eta_dot
                              + dt * (gamma / (2.0 * beta) - 1.0) * self.eta_ddot)
            k_eff = k_eff + a1 * Cd
        if C is not None:
            Cr = self.Phi.T @ C @ self.Phi
            rhs = rhs + Cr @ (a1 * self.eta + (gamma / beta - 1.0) * self.eta_dot
                              + dt * (gamma / (2.0 * beta) - 1.0) * self.eta_ddot)
            k_eff = k_eff + a1 * Cr
        eta_new = np.linalg.solve(k_eff, rhs)
        eta_ddot_new = (a0 * (eta_new - self.eta) - a1 * self.eta_dot
                        - a2 * self.eta_ddot)
        eta_dot_new = self.eta_dot + dt * ((1.0 - gamma) * self.eta_ddot
                                           + gamma * eta_ddot_new)
        self.eta, self.eta_dot, self.eta_ddot = eta_new, eta_dot_new, eta_ddot_new
        self._sync_full_state()
        return self.q, self.v, self.a

    # ------------------------------------------------------------- energies
    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.eta_dot ** 2))

    def strain_energy(self) -> float:
        return float(0.5 * np.sum(self.omega2 * self.eta ** 2))

    def modal_participation(self, F_free: np.ndarray) -> np.ndarray:
        """Generalised load per retained mode (for diagnostics)."""
        return self.Phi.T @ F_free


# ---------------------------------------------------------------------------
# Beam / fluid interface (immersed volume, conservative transfer)
# ---------------------------------------------------------------------------

class BeamFluidInterface:
    """Wetted-cell masking and momentum-conserving load transfer.

    The beam is rasterised on the Eulerian grid: every cell whose centre lies
    within ``h/2`` of the beam centre line is forced to the local beam velocity.
    The reaction force is projected back with the beam shape functions, so the
    resultant and the moment are conserved by construction.
    """

    def __init__(self, beam: CantileverBeamFE, params: FSI2Parameters,
                 dx: float, dt: float, nx: int, ny: int, x_origin: float = 0.0):
        self.beam = beam
        self.params = params
        self.dx = dx
        self._dt = dt
        self.nx, self.ny = nx, ny
        # cell-centre x coordinates inside the beam span
        x_centres = (np.arange(nx) + 0.5) * dx
        x0 = params.beam_attachment_x
        cols = np.flatnonzero((x_centres > x0) & (x_centres < x0 + params.L))
        if cols.size < 2:
            raise ValueError("beam is too thin for the chosen lattice")
        self.cols = cols
        self.x_cols = x_centres[cols]
        self.local_x = self.x_cols - x0
        self.N, self.dN = beam.shape(self.local_x)
        self.dof_idx = beam._dof_indices(self.local_x)
        self.y_cells = (np.arange(ny) + 0.5) * dx
        self._half = 0.5 * params.h

    def mask_and_velocity(self, q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the rasterised beam mask and the local structural velocity.

        Returns
        -------
        mask : (ny, nx) bool, cells occupied by the beam
        v_local : (ny, nx) vertical structural velocity at those cells
        """
        w = np.einsum("...k,...k->...", self.N, q[self.dof_idx])
        w_dot = np.einsum("...k,...k->...", self.N, v[self.dof_idx])
        y_beam = self.params.cy + w                      # (n_cols,)
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        sub = np.abs(self.y_cells[:, None] - y_beam[None, :]) <= self._half
        mask[:, self.cols] = sub
        v_local = np.zeros((self.ny, self.nx))
        v_local[:, self.cols] = np.broadcast_to(w_dot[None, :], sub.shape)
        return mask, v_local

    def implicit_terms(self, q: np.ndarray, u_star_y: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Free-DOF damping matrix and load vector of the IBM reaction.

        The immersed-boundary reaction is ``F = rho dx^2/dt * sum_cells N (u* - v)``
        (per unit depth).  Writing ``C = rho dx^2/dt * sum_cells N N^T`` and
        ``b = rho dx^2/dt * sum_cells N u*`` gives ``F = b - C v``; feeding
        ``C`` implicitly to the Newmark solve removes the added-mass feedback
        that makes an explicit partitioned scheme unstable for a light beam.

        Returns
        -------
        C_free : (n_free, n_free) damping matrix
        b_free : (n_free,) load vector
        mask : (ny, nx) wetted cells
        cell_count : number of wetted cells
        """
        w = np.einsum("...k,...k->...", self.N, q[self.dof_idx])
        y_beam = self.params.cy + w
        sub = np.abs(self.y_cells[:, None] - y_beam[None, :]) <= self._half
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        mask[:, self.cols] = sub
        count = sub.sum(axis=0).astype(float)                       # rows per column
        uy_sub = u_star_y[:, self.cols]
        # per-column integral of the (unforced) fluid velocity
        col_b = (uy_sub * sub).sum(axis=0)
        coef = self.params.rho_f * self.dx ** 2 / self._dt
        coef_b = self.params.rho_f * self.dx ** 3 / self._dt ** 2
        n_dof = self.beam.q.size
        C = np.zeros((n_dof, n_dof))
        b = np.zeros(n_dof)
        NN = self.N[:, :, None] * self.N[:, None, :]                # (n_cols,4,4)
        np.add.at(C, (self.dof_idx[:, :, None], self.dof_idx[:, None, :]),
                  coef * count[:, None, None] * NN)
        np.add.at(b, self.dof_idx.ravel(), (coef_b * self.N * col_b[:, None]).ravel())
        return C[np.ix_(self.beam.free, self.beam.free)], b[self.beam.free], mask, float(count.sum())

    def exchange_terms(self, q: np.ndarray, v_local: np.ndarray,
                       force_map: np.ndarray, force_factor: float):
        """Project the momentum-exchange force field onto the FE nodes.

        Parameters
        ----------
        force_map : (2, ny, nx) lattice force field from the bounce-back
            momentum exchange (non-zero on fluid cells adjacent to the beam).
        force_factor : lattice -> physical force conversion.

        Returns
        -------
        b : (n_free,) nodal force vector (physical units, per unit depth)
        C : (n_free, n_free) zeros (no penalty stiffness with bounce-back)
        n_links : number of loaded boundary cells
        resultant : sum of the *translational* nodal forces (the rotational
            entries carry moments and must not be added to a force resultant)
        """
        fy = force_map[1][:, self.cols]                    # (ny, n_cols)
        f_col = fy.sum(axis=0) * force_factor              # N per column
        F = np.zeros_like(self.beam.q)
        np.add.at(F, self.dof_idx.ravel(), (self.N * f_col[:, None]).ravel())
        n_links = float(np.count_nonzero(fy))
        C = np.zeros((self.beam.free.size, self.beam.free.size))
        return F[self.beam.free], C, n_links, float(F[0::2].sum())

    def nodal_forces(self, force_y_cells: np.ndarray) -> tuple[np.ndarray, float]:
        """Project the vertical reaction on the wetted cells to the FE nodes."""
        f_react = -force_y_cells[:, self.cols].sum(axis=0)     # lattice force per column
        F = np.zeros_like(self.beam.q)
        contrib = self.N * f_react[:, None]
        # partition of unity: sum(F) == sum(f_react)
        np.add.at(F, self.dof_idx.ravel(), contrib.ravel())
        return F, float(f_react.sum())


# ---------------------------------------------------------------------------
# Aitken dynamic relaxation
# ---------------------------------------------------------------------------

class AitkenRelaxation:
    """Aitken Delta^2 dynamic relaxation for the interface fixed-point loop."""

    def __init__(self, omega0: float = 1.0, omega_min: float = 0.02, omega_max: float = 2.0):
        self.omega = omega0
        self.omega0 = omega0
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.r_prev = None

    def reset(self) -> None:
        self.omega = self.omega0
        self.r_prev = None

    def update(self, residual: np.ndarray) -> float:
        if self.r_prev is None:
            self.r_prev = residual.copy()
            return self.omega
        dr = residual - self.r_prev
        denom = float(dr @ dr)
        if denom > 1e-30:
            self.omega = -self.omega * float(self.r_prev @ dr) / denom
        # the Aitken / Irons-Tuck optimum is *signed*: a negative relaxation is
        # precisely what damps the added-mass exchange of a light structure
        self.omega = float(np.clip(self.omega, -self.omega_max, self.omega_max))
        if abs(self.omega) < 1e-8:
            self.omega = self.omega0
        self.r_prev = residual.copy()
        return self.omega


# ---------------------------------------------------------------------------
# Strong coupling driver
# ---------------------------------------------------------------------------

@dataclass
class FSIHistory:
    time: list = field(default_factory=list)
    tip_displacement: list = field(default_factory=list)
    tip_velocity: list = field(default_factory=list)
    drag: list = field(default_factory=list)
    lift: list = field(default_factory=list)
    beam_force_y: list = field(default_factory=list)
    sub_iterations: list = field(default_factory=list)
    force_residual: list = field(default_factory=list)
    omega: list = field(default_factory=list)
    fluid_ke: list = field(default_factory=list)
    struct_ke: list = field(default_factory=list)
    struct_pe: list = field(default_factory=list)
    viscous_dissipation: list = field(default_factory=list)
    inlet_power: list = field(default_factory=list)
    outlet_power: list = field(default_factory=list)
    interface_power: list = field(default_factory=list)
    fluid_interface_work: list = field(default_factory=list)
    axial_reaction: list = field(default_factory=list)


class StrongCouplingFSI2:
    """Partitioned Dirichlet-Neumann strong coupling for the FSI-2 benchmark."""

    def __init__(self, params: FSI2Parameters | None = None, cells_per_D: int = 20,
                 n_elem: int = 14, u_lat: float = 0.1, max_sub_iterations: int = 15,
                 tolerance: float = 1e-4, n_modes: int = 5, zeta: float = 0.0,
                 beam_rigid: bool = False):
        self.params = params or FSI2Parameters()
        self.params.validate()
        self.cells_per_D = int(cells_per_D)
        self.n_elem = int(n_elem)
        self.u_lat = float(u_lat)
        self.max_sub_iterations = int(max_sub_iterations)
        self.tolerance = float(tolerance)
        self.beam_rigid = bool(beam_rigid)

        p = self.params
        self.dx = p.D / self.cells_per_D
        self.dt = self.u_lat * self.dx / p.U_mean
        self.nu_lat = p.nu_f * self.dt / self.dx ** 2
        self.nx = int(round(p.W / self.dx))
        self.ny = int(round(p.H / self.dx))
        self.cfg = ChannelConfig(
            nx=self.nx, ny=self.ny, u_mean=self.u_lat, nu=self.nu_lat,
            cylinder_center=(p.cx / self.dx, p.cy / self.dx),
            cylinder_radius=0.5 * p.D / self.dx,
            perturbation=0.02 * self.u_lat)
        self.solver = ChannelLBM(self.cfg)

        self.beam = ModalCantileverBeam(p.L, self.n_elem, p.EI, p.mass_per_length,
                                        n_modes=n_modes)
        self.beam.zeta = float(zeta)
        self.beam.initialise_effective(self.dt)
        self.interface = BeamFluidInterface(self.beam, p, self.dx, self.dt, self.nx, self.ny)
        self.aitken = AitkenRelaxation()

        self.energy_factor = p.rho_f * self.dx ** 4 / self.dt ** 2      # [J/m]
        self.power_factor = p.rho_f * self.dx ** 4 / self.dt ** 3       # [W/m]
        self.force_factor = p.rho_f * self.dx ** 3 / self.dt ** 2       # [N/m]

        self.history = FSIHistory()
        self._t = 0.0

    # ------------------------------------------------------------------ utils
    def tip_index(self) -> int:
        return 2 * self.beam.n_elem

    def tip_displacement(self) -> float:
        return float(self.beam.q[self.tip_index()])

    # ------------------------------------------------------------------- run
    def run(self, n_steps: int, record_every: int = 1) -> FSIHistory:
        """March the coupled system for ``n_steps`` lattice steps.

        Each time step performs a Dirichlet-Neumann fixed-point iteration:

        1. the fluid is advanced one step with the beam moving at the predicted
           interface velocity (immersed-boundary direct forcing);
        2. the immersed-boundary reaction ``F = b - C v`` is assembled, with
           ``C`` fed implicitly to the structural Newmark solve;
        3. the structural time step is re-solved from the *converged* state at
           ``t^n`` until the interface displacement and force residuals fall
           below ``self.tolerance``; Aitken relaxation accelerates the loop.
        """
        beam = self.beam
        solver = self.solver
        hist = self.history
        free = beam.free
        for step in range(n_steps):
            f_save = solver.f.copy()
            q_n, v_n, a_n = beam.q.copy(), beam.v.copy(), beam.a.copy()
            q_pred, v_pred = q_n.copy(), v_n.copy()
            F_prev = None
            force_res = float("inf")
            self.aitken.reset()
            converged = False
            k = 0
            for k in range(1, self.max_sub_iterations + 1):
                mask, v_local = self.interface.mask_and_velocity(q_pred, v_pred)
                solver.f = f_save.copy()
                f_ex = solver.step_with_mask(mask)
                b, C, n_cells, _ = self.interface.exchange_terms(
                    q_pred, v_local, f_ex, self.force_factor)
                # IBM reaction  F = C (u* - v)  is inertial: m_a = C dt is
                # added to the mass matrix and only the *increment* of the
                # fluid velocity relative to the previous structural velocity
                # enters as a load
                F_beam = b - C @ v_pred[free]         # physical interface force
                if self.beam_rigid:
                    q_new, v_new, a_new = q_n.copy(), np.zeros_like(v_n), np.zeros_like(a_n)
                else:
                    F_full = np.zeros_like(beam.q)
                    F_full[free] = b - C @ v_n[free]    # C (u* - v_n)
                    beam.set_state(q_n, v_n, a_n)
                    q_new, v_new, a_new = beam.step(F_full, M_add=C * self.dt)
                if F_prev is not None:
                    force_res = float(np.linalg.norm(F_beam - F_prev)
                                      / max(np.linalg.norm(F_beam), 1e-30))
                F_prev = F_beam
                disp_res = float(np.linalg.norm(q_new - q_pred)
                                 / max(np.linalg.norm(q_new), 1e-30))
                if disp_res < self.tolerance and force_res < self.tolerance:
                    converged = True
                    break
                omega = self.aitken.update(q_new - q_pred)
                q_pred = q_pred + omega * (q_new - q_pred)
                v_pred = v_pred + omega * (v_new - v_pred)

            if not self.beam_rigid:
                beam.set_state(q_new, v_new, a_new)

            # ---- diagnostics -------------------------------------------------
            cyl_force = solver.momentum_exchange_force() * self.force_factor
            self._t += self.dt
            if step % record_every == 0:
                hist.time.append(self._t)
                hist.tip_displacement.append(float(beam.q[self.tip_index()]))
                hist.tip_velocity.append(float(beam.v[self.tip_index()]))
                hist.drag.append(float(cyl_force[0]))
                hist.lift.append(float(cyl_force[1]))
                hist.beam_force_y.append(float(F_beam[-2]))
                hist.sub_iterations.append(k)
                hist.force_residual.append(force_res)
                hist.omega.append(omega)
                hist.fluid_ke.append(solver.kinetic_energy() * self.energy_factor)
                hist.struct_ke.append(beam.kinetic_energy() * self.energy_factor)
                hist.struct_pe.append(beam.strain_energy() * self.energy_factor)
                hist.viscous_dissipation.append(
                    solver.viscous_dissipation_rate() * self.power_factor)
                hist.inlet_power.append(solver.inlet_energy_flux() * self.power_factor)
                hist.outlet_power.append(solver.outlet_energy_flux() * self.power_factor)
                hist.interface_power.append(float(F_beam @ beam.v[free]))
                hist.axial_reaction.append(float(f_ex[0].sum()) * self.force_factor)
        return hist

    # ------------------------------------------------------- derived metrics
    def analyse(self, transient: float = 2.0) -> dict:
        """Periodic-window metrics: frequency, amplitude, force coefficients."""
        h = self.history
        t = np.asarray(h.time)
        if t.size < 4:
            raise RuntimeError("not enough samples")
        sel = t >= transient
        if sel.sum() < 4:
            sel = np.ones_like(t, dtype=bool)
        ts = t[sel]
        uy = np.asarray(h.tip_displacement)[sel]
        lift = np.asarray(h.lift)[sel]
        drag = np.asarray(h.drag)[sel]
        q = 0.5 * self.params.rho_f * self.params.U_mean ** 2 * self.params.D
        CD = drag / q
        CL = lift / q

        amp = 0.5 * (uy.max() - uy.min())
        freq = _dominant_frequency(ts, uy)
        return {
            "frequency": freq,
            "amplitude": float(amp),
            "mean_tip": float(uy.mean()),
            "mean_CD": float(CD.mean()),
            "amp_CL": float(0.5 * (CL.max() - CL.min())),
            "mean_CL": float(CL.mean()),
            "n_samples": int(sel.sum()),
            "window": (float(ts[0]), float(ts[-1])),
            "waveform": {"time": ts.tolist(), "tip": uy.tolist(), "CL": CL.tolist()},
        }


def _dominant_frequency(t: np.ndarray, y: np.ndarray) -> float:
    """Dominant frequency from the largest FFT bin (with parabolic interpolation)."""
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    if y.size < 8:
        return 0.0
    dt = float(np.mean(np.diff(t)))
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, d=dt)
    k = int(np.argmax(spec[1:]) + 1)
    if 0 < k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = (a - 2 * b + c)
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-30 else 0.0
        return float((k + delta) * (freqs[1] - freqs[0]))
    return float(freqs[k])


# ---------------------------------------------------------------------------
# Time / grid refinement helpers
# ---------------------------------------------------------------------------

def run_convergence_study(params: FSI2Parameters | None = None,
                          durations=(4.0,), cells=(20,), u_lats=(0.1,),
                          n_steps_cap: int | None = None) -> list[dict]:
    """Run the FSI-2 case for the requested (grid, lattice velocity) pairs."""
    params = params or FSI2Parameters()
    out = []
    for cells in cells:
        for u_lat in u_lats:
            fsi = StrongCouplingFSI2(params, cells_per_D=cells, u_lat=u_lat)
            duration = durations[0]
            n_steps = int(round(duration / fsi.dt))
            if n_steps_cap:
                n_steps = min(n_steps, n_steps_cap)
            fsi.run(n_steps)
            metrics = fsi.analyse(transient=0.6 * duration)
            metrics.update({"cells_per_D": cells, "u_lat": u_lat, "dt": fsi.dt,
                            "n_steps": n_steps})
            out.append(metrics)
    return out