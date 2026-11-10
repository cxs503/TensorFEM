"""Two-dimensional unsteady vortex-lattice solver for a flat plate.

This is the *fluid* half of the partitioned (weak-coupling) FSI framework used by
``tensorfem.fsi_flutter_vlm``.  It is an independent numerical solution of the
same incompressible potential-flow problem for which Theodorsen (1935) gives the
closed-form answer, so a comparison between the two is a genuine verification
instead of a tautology.

Discretisation
--------------
* The plate occupies ``-b <= x <= b`` with the elastic axis at ``x = a*b``.
* ``n_panels`` cosine-spaced panels; a lumped bound vortex (strength ``Gamma_k``)
  sits at the quarter-chord of each panel and the flow-tangency condition is
  imposed at its three-quarter point (classical 1/4-3/4 vortex lattice).
* Vorticity leaves the trailing edge as a *sheet*: every time step a segment
  ``[b, b + U*dt]`` of uniform vorticity is shed, so the wake kernel is the exact
  logarithmic kernel of a uniform vortex sheet rather than a point-vortex 1/r
  kernel (which is what makes naive 2-D vortex shedding blow up: a point vortex
  sitting one panel away from a collocation point radiates a spurious
  ``Gamma/(2*pi*r)``).
* The wake convects downstream at the free-stream speed (linearised flat wake,
  the same kinematic assumption Theodorsen's theory makes: no roll-up).
* The strength of the newly shed segment is *solved together with* the bound
  circulation through Kelvin's theorem (implicit Kelvin closure), which keeps the
  circulation of the fluid constant to machine precision and avoids the explicit
  ``shed -> wake -> bound`` feedback that destabilises explicit shedding.

Surface pressure
----------------
Unsteady Bernoulli on the two sides of the plate gives, with
``dPhi_j = -sum_{k<=j} Gamma_k`` the potential jump at control point ``j``,

    dp_j = rho * ( U*gamma_j - d(dPhi_j)/dt ),
    L    = -sum_j dp_j*dx_j                (positive up),
    M    =  sum_j dp_j*dx_j*(xv_j - a*b)   (positive nose-up about the EA),

which reduces to the Kutta-Joukowski lift ``L = 2*pi*rho*U**2*b*alpha`` in the
steady limit and reproduces Theodorsen's unsteady lift and moment (including the
non-circulatory apparent-mass part) for harmonic motion, see
``UnsteadyVLM2D.harmonic_transfer``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AeroLoads", "UnsteadyVLM2D"]


@dataclass(frozen=True)
class AeroLoads:
    """Aerodynamic loads on the plate strip (per unit span)."""

    lift: float
    """Vertical force, positive up."""
    moment: float
    """Pitching moment about the elastic axis, positive nose-up."""
    circulation: float
    """Total bound circulation (sum of the panel vortices)."""


class UnsteadyVLM2D:
    """Time-marching 2-D vortex lattice for a flat plate in uniform flow.

    Parameters
    ----------
    semi_chord:
        ``b``; the plate spans ``[-b, b]``.
    n_panels:
        Number of chordwise panels (cosine spaced; 32-64 is converged for
        reduced frequencies up to ``k ~ 0.5``).
    dt:
        Time step.  ``U*dt/b = 0.05-0.2`` is the usual working range.
    U:
        Free-stream speed (in whatever unit system the caller uses; internally
        the aerodynamics only sees ``U*dt`` and ``U`` in the pressure).
    ea:
        Elastic-axis position ``a`` in semi-chords from mid-chord.
    core:
        Wake-vortex core radius as a fraction of ``U*dt`` (regularises the
        logarithmic kernel right at the trailing edge).
    wake_length:
        Wake truncated ``wake_length`` semi-chords downstream of the trailing
        edge; the truncated tail is negligible for ``>= 20`` semi-chords.
    rho:
        Fluid density (loads are linear in ``rho``).
    """

    def __init__(
        self,
        semi_chord: float = 1.0,
        n_panels: int = 48,
        dt: float = 0.1,
        U: float = 1.0,
        ea: float = 0.0,
        core: float = 0.05,
        wake_length: float = 40.0,
        rho: float = 1.0,
    ) -> None:
        if n_panels < 4:
            raise ValueError("n_panels must be at least 4")
        if dt <= 0 or U <= 0:
            raise ValueError("dt and U must be positive")
        self.b = float(semi_chord)
        self.n = int(n_panels)
        self.dt = float(dt)
        self.U = float(U)
        self.a = float(ea)
        self.rho = float(rho)
        self.core = float(core)
        self.wake_length = float(wake_length)

        theta = np.linspace(0.0, np.pi, self.n + 1)
        edges = -self.b * np.cos(theta)
        self.edges = edges
        self.dx = np.diff(edges)
        self.x_vortex = edges[:-1] + 0.25 * self.dx
        self.x_ctrl = edges[:-1] + 0.75 * self.dx
        gap = self.x_ctrl[:, None] - self.x_vortex[None, :]
        # vertical velocity induced at control point i by unit bound vortex k
        self._bound_influence = gap / (2.0 * np.pi * np.maximum(gap ** 2, 1e-18))

        self._eps = self.core * self.U * self.dt
        # influence of the segment that is about to be shed (body frame, fixed)
        self._new_wake_influence = self._sheet_wake_influence(
            self.x_ctrl, self.b, self.b + self.U * self.dt
        )
        size = self.n + 1
        matrix = np.zeros((size, size))
        matrix[: self.n, : self.n] = self._bound_influence
        matrix[: self.n, self.n] = self._new_wake_influence
        matrix[self.n, : self.n] = 1.0  # implicit Kelvin: sum(Gamma) + Gamma_shed
        matrix[self.n, self.n] = 1.0
        self._inverse = np.linalg.inv(matrix)

        self._wake_a = np.zeros(0)
        self._wake_b = np.zeros(0)
        self._wake_gamma = np.zeros(0)
        self._gamma = np.zeros(self.n)
        self._gamma_prev = np.zeros(self.n)
        self._phi_prev = np.zeros(self.n)
        self.shed_history: list[float] = []

    # ---------------------------------------------------------------- kernel
    def _sheet_wake_influence(self, x, xa, xb):
        """Vertical velocity at ``x`` from a uniform vortex sheet on ``[xa, xb]``."""
        eps2 = self._eps ** 2
        log = np.log(((x - xa) ** 2 + eps2) / ((x - xb) ** 2 + eps2))
        return log / (4.0 * np.pi * (xb - xa))

    def _wake_downwash(self) -> np.ndarray:
        if self._wake_gamma.size == 0:
            return np.zeros(self.n)
        influence = self._sheet_wake_influence(
            self.x_ctrl[:, None], self._wake_a[None, :], self._wake_b[None, :]
        )
        return influence @ self._wake_gamma

    # ------------------------------------------------------------- interface
    def reset(self, alpha: float = 0.0, alpha_dot: float = 0.0) -> None:
        """Restart from the steady attached-flow state at incidence ``alpha``.

        Starting from the steady state (instead of an impulsive start) is the
        standard initialisation for aeroelastic marching: the starting vortex is
        left far downstream where its influence on the plate is negligible.
        """
        downwash = -(self.U * alpha) + np.zeros(self.n)
        gamma = np.linalg.solve(self._bound_influence, downwash)
        self._gamma = gamma
        self._gamma_prev = gamma.copy()
        self._phi_prev = -np.cumsum(gamma)
        self._wake_a = np.zeros(0)
        self._wake_b = np.zeros(0)
        self._wake_gamma = np.zeros(0)
        self.shed_history = []

    def advance(
        self, h: float, alpha: float, h_dot: float, alpha_dot: float
    ) -> AeroLoads:
        """Advance one time step for the plate motion ``(h, alpha, h_dot, alpha_dot)``.

        ``h`` is the plunge displacement (positive down), ``alpha`` the pitch
        (positive nose-up), both about the elastic axis.
        """
        U, dt, b, a = self.U, self.dt, self.b, self.a
        # 1. convect the wake downstream at the free-stream speed
        self._wake_a = self._wake_a + U * dt
        self._wake_b = self._wake_b + U * dt

        # 2. flow tangency at the control points, implicit Kelvin closure
        downwash = -(h_dot + U * alpha + (self.x_ctrl - a * b) * alpha_dot)
        rhs = np.empty(self.n + 1)
        rhs[: self.n] = downwash - self._wake_downwash()
        rhs[self.n] = self._gamma_prev.sum()
        solution = self._inverse @ rhs
        gamma = solution[: self.n]
        gamma_shed = solution[self.n]

        # 3. attach the freshly shed segment and truncate the far wake
        self._wake_a = np.concatenate(([b], self._wake_a))
        self._wake_b = np.concatenate(([b + U * dt], self._wake_b))
        self._wake_gamma = np.concatenate(([gamma_shed], self._wake_gamma))
        keep = self._wake_a < b + self.wake_length
        if not keep.all():
            self._wake_a = self._wake_a[keep]
            self._wake_b = self._wake_b[keep]
            self._wake_gamma = self._wake_gamma[keep]

        # 4. unsteady Bernoulli pressure, integrate loads
        phi = -np.cumsum(gamma)
        dp = self.rho * (U * gamma / self.dx - (phi - self._phi_prev) / dt)
        lift = float(-(dp @ self.dx))
        moment = float((dp * self.dx) @ (self.x_vortex - a * b))

        self._phi_prev = phi
        self._gamma_prev = gamma
        self._gamma = gamma
        self.shed_history.append(float(gamma_shed))
        return AeroLoads(lift=lift, moment=moment, circulation=float(gamma.sum()))

    # ------------------------------------------------------------ validation
    def harmonic_transfer(
        self,
        k: float,
        amp_h: float = 0.0,
        amp_alpha: float = 1e-3,
        *,
        periods: int = 8,
        steps_per_period: int = 200,
        fit_periods: float = 3.0,
    ) -> tuple[complex, complex]:
        """Complex lift and moment for ``h = amp_h sin(wt)``, ``alpha = amp_alpha sin(wt)``.

        Returns ``(L_hat, M_hat)`` such that ``L(t) = Re[L_hat exp(i w t)]`` after a
        steady periodic state has been reached.  Comparing these with
        ``tensorfem.theodorsen_pk.aero_force_matrices`` for the same ``k`` is the
        aerodynamic verification of the solver.
        """
        omega = k * self.U / self.b
        period = 2.0 * np.pi / omega
        dt = period / steps_per_period
        solver = UnsteadyVLM2D(
            semi_chord=self.b,
            n_panels=self.n,
            dt=dt,
            U=self.U,
            ea=self.a,
            core=self.core,
            wake_length=self.wake_length,
            rho=self.rho,
        )
        solver.reset()
        n_steps = int(periods * steps_per_period)
        times = np.empty(n_steps)
        lifts = np.empty(n_steps)
        moments = np.empty(n_steps)
        for step in range(n_steps):
            t = (step + 1) * dt
            s, c = np.sin(omega * t), np.cos(omega * t)
            loads = solver.advance(
                amp_h * s, amp_alpha * s, amp_h * omega * c, amp_alpha * omega * c
            )
            times[step] = t
            lifts[step] = loads.lift
            moments[step] = loads.moment
        window = times > times[-1] - fit_periods * period
        t = times[window]
        basis = np.stack([np.sin(omega * t), np.cos(omega * t)], axis=1)

        def coefficient(series: np.ndarray) -> complex:
            (sin_coeff, cos_coeff), *_ = np.linalg.lstsq(basis, series[window], rcond=None)
            return complex(cos_coeff, -sin_coeff)

        return coefficient(lifts), coefficient(moments)