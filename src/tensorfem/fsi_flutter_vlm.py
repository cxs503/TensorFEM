"""Partitioned (weak-coupling) FSI flutter analysis: elastic plate in uniform flow.

The framework
-------------
The fluid and the structure are solved by two independent codes and coupled at
the interface every time step (explicit staggered / weak coupling):

* **fluid**: :class:`tensorfem.unsteady_vlm.UnsteadyVLM2D`, a time-domain
  vortex-lattice solution of the incompressible potential flow past the plate.
  It returns the unsteady lift and pitching moment for the instantaneous plate
  motion.
* **structure**: a two-degree-of-freedom modal model (plunge ``h`` = first
  cantilever bending mode, pitch ``alpha`` = first torsion mode) whose
  frequencies come from a real TensorFEM finite-element modal solve of the
  plate (``tensorfem.modal.cantilever_beam_modes``) and a Saint-Venant torsion
  model for the torsional branch.  The modal equations of motion are integrated
  with classical RK4.
* **reference**: :mod:`tensorfem.theodorsen_pk`, the frequency-domain
  Theodorsen p-k solution for the same section.  The two solutions share only
  the physical model (potential flow, flat plate, linear structure) and no
  numerical machinery, so agreement between them is a verification of both.

Geometry of the reference case (campaign node P88W1-N07)
-------------------------------------------------------
A cantilever rectangular plate ``L x c x t = 1.0 x 0.2 x 0.01 m`` of aluminium
(``E = 70 GPa``, ``nu = 0.3``, ``rho_s = 2700 kg/m^3``) in air
(``rho_f = 1.225 kg/m^3``).  The span-to-chord ratio is 5, so two-dimensional
strip aerodynamics at the reference section is a defensible (documented)
approximation; the aerodynamic loads are applied at the section through the
modal generalised force of the first bending/torsion modes.

Growth rate, frequency and flutter boundary
-------------------------------------------
``march`` returns the time history; :func:`growth_and_frequency` extracts the
exponential growth rate from the peaks of ``alpha(t)`` and the period from its
zero crossings.  :func:`flutter_boundary_marching` searches ``U_cr`` as the zero
crossing of the growth rate with a secant iteration, using *only* the
time-domain solver (the frequency-domain reference is not used to bracket it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .modal import cantilever_beam_modes, cantilever_exact_angular_frequency
from .theodorsen_pk import (
    TypicalSection,
    flutter_boundary,
    pk_continuation,
    pk_eigenvalues,
)
from .unsteady_vlm import UnsteadyVLM2D

__all__ = [
    "FlutterPlate",
    "ImpulseResponse",
    "march",
    "growth_and_frequency",
    "growth_rate",
    "flutter_boundary_marching",
    "mass_ratio_scaling",
    "aerodynamic_verification",
    "verification_bundle",
]


# ---------------------------------------------------------------------------
# Reference structure: cantilever plate, FE modal data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlutterPlate:
    """Cantilever rectangular plate in uniform flow (campaign node P88W1-N07).

    Defaults are the node specification: ``1.0 x 0.2 x 0.01 m`` aluminium plate in
    air.  ``elements`` is the number of beam elements of the TensorFEM modal
    solve that supplies the bending frequency.
    """

    span: float = 1.0
    chord: float = 0.2
    thickness: float = 0.01
    young: float = 70e9
    poisson: float = 0.3
    rho_solid: float = 2700.0
    rho_fluid: float = 1.225
    elements: int = 24
    static_unbalance: float = 0.0

    # -- mass / stiffness / frequencies -----------------------------------
    @property
    def semi_chord(self) -> float:
        return 0.5 * self.chord

    @property
    def mass_per_span(self) -> float:
        return self.rho_solid * self.chord * self.thickness

    @property
    def bending_stiffness(self) -> float:
        """``EI`` for out-of-plane bending of the plate strip."""
        return self.young * self.chord * self.thickness ** 3 / 12.0

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def torsion_constant(self) -> float:
        """Saint-Venant ``J`` of a thin rectangle, ``beta*c*t**3``."""
        ratio = self.thickness / self.chord
        beta = (1.0 / 3.0) * (1.0 - 0.63 * ratio + 0.052 * ratio ** 5)
        return beta * self.chord * self.thickness ** 3

    @property
    def torsional_stiffness(self) -> float:
        return self.shear_modulus * self.torsion_constant

    @property
    def polar_inertia(self) -> float:
        """Polar mass moment of inertia per unit span about the mid-chord."""
        return (
            self.rho_solid
            * self.chord
            * self.thickness
            * (self.chord ** 2 + self.thickness ** 2)
            / 12.0
        )

    @property
    def omega_h(self) -> float:
        """First cantilever bending frequency from the TensorFEM FE modal solve."""
        result = cantilever_beam_modes(
            length=self.span,
            elements=self.elements,
            EI=self.bending_stiffness,
            rho_a=self.mass_per_span,
            modes=1,
        )
        return float(result.angular_frequencies[0])

    @property
    def omega_h_exact(self) -> float:
        """Closed-form Euler-Bernoulli reference for :attr:`omega_h`."""
        return cantilever_exact_angular_frequency(
            self.span, self.bending_stiffness, self.mass_per_span, mode=1
        )

    @property
    def omega_alpha(self) -> float:
        """First cantilever torsion frequency (Saint-Venant, uniform section)."""
        return (math.pi / (2.0 * self.span)) * math.sqrt(
            self.torsional_stiffness / self.polar_inertia
        )

    # -- non-dimensional aeroelastic section ------------------------------
    @property
    def section(self) -> TypicalSection:
        m = self.mass_per_span
        mu = m / (math.pi * self.rho_fluid * self.semi_chord ** 2)
        r_alpha = math.sqrt(self.polar_inertia / (m * self.semi_chord ** 2))
        return TypicalSection(
            mu=mu,
            sigma=self.omega_h / self.omega_alpha,
            r_alpha=r_alpha,
            x_alpha=self.static_unbalance,
            a=0.0,
        )

    def velocity_scale(self) -> float:
        """``b*omega_alpha`` in m/s: dimensional velocity = scale * reduced velocity."""
        return self.semi_chord * self.omega_alpha

    def reduced_velocity(self, U: float) -> float:
        return U / self.velocity_scale()

    def dimensional_velocity(self, U_reduced: float) -> float:
        return U_reduced * self.velocity_scale()


# ---------------------------------------------------------------------------
# Time marching (weak coupling)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImpulseResponse:
    """Free-decay/growth record of the coupled system at one reduced velocity."""

    U: float
    time: np.ndarray
    alpha: np.ndarray
    h: np.ndarray
    growth: float
    frequency: float
    steps: int
    dt: float

    @property
    def unstable(self) -> bool:
        return self.growth > 0.0


def _inv2(matrix: np.ndarray) -> np.ndarray:
    (a, b), (c, d) = matrix
    det = a * d - b * c
    return np.array([[d, -b], [-c, a]]) / det


def march(
    section: TypicalSection,
    U: float,
    *,
    periods: float = 5.0,
    dt_factor: float = 0.1,
    alpha0: float = 1e-3,
    alpha_dot0: float | None = None,
    max_amplitude: float = 1e6,
    n_panels: int = 64,
    wake_length: float = 60.0,
    core: float = 0.05,
) -> ImpulseResponse:
    """Time-march the coupled system at reduced velocity ``U``.

    Units: ``b = 1``, ``rho = 1``, ``omega_alpha = 1`` (so ``U`` is the reduced
    velocity and time is measured in ``1/omega_alpha``).  The structural model is
    ``Ms q̈ + Ks q = [-L, M]`` with ``q = [h, alpha]``; the aerodynamic loads are
    frozen over each structural step (explicit staggered coupling).  The time step
    is ``dt = dt_factor/U``, i.e. the wake advances ``dt_factor`` semi-chords per
    step; ``periods`` is measured against the coupled flutter frequency implied by
    the structural pitch frequency, ``omega = max(sigma, 0.35)``.
    """
    if U <= 0:
        raise ValueError("U must be positive")
    m = section.mass
    ms = np.array([[m, m * section.x_alpha], [m * section.x_alpha, section.inertia]])
    ks = np.array([[m * section.sigma ** 2, 0.0], [0.0, section.inertia]])
    ms_inv = _inv2(ms)

    dt = dt_factor / U
    omega_ref = max(section.sigma, 0.35)
    t_max = periods * 2.0 * math.pi / omega_ref
    n_steps = max(int(round(t_max / dt)), 100)
    dt = t_max / n_steps

    aero = UnsteadyVLM2D(
        semi_chord=1.0,
        n_panels=n_panels,
        dt=dt,
        U=U,
        ea=section.a,
        core=core,
        wake_length=wake_length,
        rho=1.0,
    )
    aero.reset()

    def acceleration(q: np.ndarray, loads: np.ndarray) -> np.ndarray:
        return ms_inv @ (loads - ks @ q)

    if alpha_dot0 is None:
        # impulsive start: a pure displacement start can sit almost exactly on the
        # static aeroelastic equilibrium (alpha_eq/alpha0 ~ U^2/(mu*r_alpha^2)),
        # which leaves a nearly constant signal that no damping estimator can use.
        alpha_dot0 = 0.4 * alpha0 * max(section.sigma, 0.35)
    q = np.array([0.0, alpha0])
    qd = np.array([0.0, alpha_dot0])
    time = np.empty(n_steps + 1)
    alpha = np.empty(n_steps + 1)
    plunge = np.empty(n_steps + 1)
    time[0] = 0.0
    alpha[0] = q[1]
    plunge[0] = q[0]
    for step in range(n_steps):
        loads = aero.advance(q[0], q[1], qd[0], qd[1])
        force = np.array([-loads.lift, loads.moment])
        k1q, k1v = qd, acceleration(q, force)
        k2q = qd + 0.5 * dt * k1v
        k2v = acceleration(q + 0.5 * dt * k1q, force)
        k3q = qd + 0.5 * dt * k2v
        k3v = acceleration(q + 0.5 * dt * k2q, force)
        k4q = qd + dt * k3v
        k4v = acceleration(q + dt * k3q, force)
        q = q + dt * (k1q + 2 * k2q + 2 * k3q + k4q) / 6.0
        qd = qd + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        time[step + 1] = time[step] + dt
        alpha[step + 1] = q[1]
        plunge[step + 1] = q[0]
        if max(abs(q[0]), abs(q[1])) > max_amplitude:
            # stop an exponentially growing run before the amplitudes (and the
            # log-envelope fit) run away; the linear model is only meaningful for
            # small motions anyway
            time = time[: step + 2]
            alpha = alpha[: step + 2]
            plunge = plunge[: step + 2]
            break
    if not np.isfinite(alpha).all():
        raise RuntimeError("coupled march diverged (non-finite response)")
    growth, frequency = growth_and_frequency(
        time, alpha, windows=(0.45, 0.55, 0.65)
    )
    return ImpulseResponse(
        U=float(U),
        time=time,
        alpha=alpha,
        h=plunge,
        growth=growth,
        frequency=frequency,
        steps=n_steps,
        dt=dt,
    )


def _dominant_frequency(
    time: np.ndarray, signal: np.ndarray, *, min_cycles: float = 1.5
) -> float:
    """Dominant angular frequency of ``signal`` by zero-padded FFT + parabolic peak fit.

    Bins below ``min_cycles`` cycles across the record are ignored, otherwise the
    DC/sub-harmonic leakage of a strongly growing record wins and the
    demodulation window becomes shorter than one period.
    """
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("time vector must be increasing")
    n = signal.size
    taper = np.hanning(n)
    pad = 8 * n
    spectrum = np.abs(np.fft.rfft(signal * taper, n=pad))
    freqs = np.fft.rfftfreq(pad, d=dt) * 2.0 * math.pi
    if spectrum.size < 3:
        raise ValueError("signal too short for a frequency estimate")
    span = float(time[-1] - time[0])
    if span > 0:
        floor = 2.0 * math.pi * min_cycles / span
        spectrum = spectrum.copy()
        spectrum[freqs < floor] = 0.0
    peak = int(np.argmax(spectrum[1:]) + 1)
    if 1 <= peak < spectrum.size - 1:  # parabolic refinement in log magnitude
        y0, y1, y2 = (math.log(max(spectrum[k], 1e-300)) for k in (peak - 1, peak, peak + 1))
        denom = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        return float(freqs[peak] + shift * (freqs[1] - freqs[0]))
    return float(freqs[peak])


def growth_and_frequency(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    window: float = 0.6,
    refine: int = 2,
    windows: tuple[float, ...] | None = None,
) -> tuple[float, float]:
    """Exponential growth rate and angular frequency of a decaying/growing signal.

    A free response of the coupled system contains both aeroelastic modes, so a
    peak-envelope fit is biased by whichever mode dominates the peaks.  The signal
    is therefore demodulated with a sliding least-squares quadrature window of 1.5
    periods, ``signal(t) ~ A(t)*sin(omega*t + psi(t))``:

    * ``log A(t)`` slope -> growth rate ``sigma``;
    * ``psi(t)`` slope -> frequency correction, iterated ``refine`` times starting
      from the zero-padded FFT peak of the record.

    Only the last ``window`` fraction of the record is used, so the transient of
    the strongly damped branch has decayed before the fit.
    """
    if windows is not None:
        estimates = [
            growth_and_frequency(time, signal, window=w, refine=refine)
            for w in windows
            if w > 0.05
        ]
        growths = np.array([g for g, _ in estimates])
        index = int(np.argsort(growths)[len(growths) // 2])
        return float(growths[index]), float(estimates[index][1])

    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if signal.size < 64:
        raise ValueError("signal too short for growth-rate estimation")

    t_start = time[0] + (1.0 - window) * (time[-1] - time[0])
    keep = time >= t_start
    t_win, s_win = time[keep], signal[keep]
    scale = np.max(np.abs(s_win))
    if scale > 0:
        s_win = s_win / scale  # scale-invariant: growth rate is unaffected
    # The explicit staggered coupling leaves a small step-scale numerical
    # oscillation (amplitude ~1e-6 of the initial condition).  Once the physical
    # transient has decayed it dominates the spectrum, so the record is smoothed
    # over a small fraction of the structural period and its static aeroelastic
    # offset (equilibrium pitch under a constant load) is removed.
    coarse_omega = _dominant_frequency(t_win, s_win - s_win.mean())
    dt_win = float(np.median(np.diff(t_win)))
    if np.isfinite(coarse_omega) and coarse_omega > 0:
        smooth_steps = int(round(0.03 * 2.0 * math.pi / coarse_omega / dt_win))
    else:  # pragma: no cover
        smooth_steps = 3
    smooth_steps = int(np.clip(smooth_steps, 1, max(1, len(s_win) // 50)))
    if smooth_steps > 3:
        kernel = np.ones(smooth_steps) / smooth_steps
        s_win = np.convolve(s_win, kernel, mode="same")
    omega = _dominant_frequency(t_win, s_win - s_win.mean())

    growth = float("nan")
    for _ in range(max(refine, 1)):
        period = 2.0 * math.pi / omega
        half = 0.75 * period
        centers = np.arange(t_win[0] + half, t_win[-1] - half, 0.2 * period)
        if centers.size < 5:
            centers = np.linspace(t_win[0] + half, t_win[-1] - half, 8)
        amplitudes = np.empty(centers.size)
        phases = np.empty(centers.size)
        for index, center in enumerate(centers):
            mask = np.abs(t_win - center) <= half
            basis = np.stack(
                [
                    np.sin(omega * t_win[mask]),
                    np.cos(omega * t_win[mask]),
                    np.ones(mask.sum()),
                ],
                axis=1,
            )
            (a, b, _offset), *_ = np.linalg.lstsq(basis, s_win[mask], rcond=None)
            amplitudes[index] = math.hypot(a, b)
            phases[index] = math.atan2(b, a)
        usable = amplitudes > 1e-10 * amplitudes.max()
        if usable.sum() < 4:
            raise ValueError("signal amplitude vanished before the analysis window")
        growth = float(np.polyfit(centers[usable], np.log(amplitudes[usable]), 1)[0])
        drift = float(np.polyfit(centers[usable], np.unwrap(phases[usable]), 1)[0])
        if not np.isfinite(drift) or abs(drift) > 0.5 * omega:
            break  # phase unwrapping lost the mode: keep the FFT frequency
        omega = omega + drift
    if not np.isfinite(omega) or omega <= 0 or not np.isfinite(growth):
        raise ValueError("growth-rate fit failed: non-physical frequency/damping")
    return growth, float(omega)


def growth_rate(
    section: TypicalSection, U: float, *, periods: float = 5.0, **kwargs
) -> float:
    """Growth rate ``Re(p)`` at reduced velocity ``U`` from the time-domain solver."""
    return march(section, U, periods=periods, **kwargs).growth


def flutter_boundary_marching(
    section: TypicalSection,
    *,
    U_lo: float = 0.3,
    U_hi: float | None = None,
    periods: float = 5.0,
    max_iter: int = 8,
    tol: float = 3e-3,
    **kwargs,
) -> dict:
    """Locate ``U_cr`` from the time-domain growth rate alone (bracketing + bisection).

    ``U_hi`` is only an upper bound of the search window; when ``None`` the window
    is expanded geometrically until the growth rate changes sign.  No
    frequency-domain information is used, so the result is an independent check of
    :func:`tensorfem.theodorsen_pk.flutter_boundary`.
    """
    hi = U_hi if U_hi is not None else max(4.0 * section.sigma, 1.0)
    history: list[dict] = []
    g_lo = growth_rate(section, U_lo, periods=periods, **kwargs)
    history.append({"U": U_lo, "growth": g_lo})
    if g_lo >= 0:
        raise RuntimeError(f"section unstable already at U={U_lo}")
    for _ in range(12):
        g_hi = growth_rate(section, hi, periods=periods, **kwargs)
        history.append({"U": hi, "growth": g_hi})
        if g_hi > 0:
            break
        U_lo, g_lo = hi, g_hi
        hi *= 1.6
    else:  # pragma: no cover
        raise RuntimeError("no flutter crossing found in the expanded window")

    # bisection on the bracketed sign change: the growth-rate estimator carries a
    # few percent of noise, so a secant step can extrapolate to a wrong branch,
    # while bisection keeps the bracket (and therefore the answer) honest.
    for _ in range(max_iter):
        midpoint = 0.5 * (U_lo + hi)
        g_mid = growth_rate(section, midpoint, periods=periods, **kwargs)
        history.append({"U": midpoint, "growth": g_mid})
        if g_mid < 0:
            U_lo, g_lo = midpoint, g_mid
        else:
            hi, g_hi = midpoint, g_mid
        if hi - U_lo < tol * max(1.0, midpoint):
            break
    U_cr = 0.5 * (U_lo + hi)
    tracked = pk_continuation(section, 0.1 * U_cr, 1.05 * U_cr, n_steps=6)[-1]
    return {
        "U_cr": U_cr,
        "U_bracket": [U_lo, hi],
        "omega_fl_reference": tracked.omega,
        "growth_reference": tracked.growth_rate,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def aerodynamic_verification(
    k_values: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5), n_panels: int = 48
) -> list[dict]:
    """Lift/moment of the UVLM versus Theodorsen for harmonic pitch and plunge."""
    from .theodorsen_pk import aero_force_matrices

    rows = []
    for k in k_values:
        solver = UnsteadyVLM2D(n_panels=n_panels, dt=1.0, U=1.0)
        for mode, amp_h, amp_alpha in (("pitch", 0.0, 1e-3), ("plunge", 1e-3, 0.0)):
            lift_hat, moment_hat = solver.harmonic_transfer(
                k, amp_h=amp_h, amp_alpha=amp_alpha
            )
            omega = k
            A2, A1, A0 = aero_force_matrices(1.0, k, 0.0, rho=1.0, b=1.0)
            q = np.array([-1j * amp_h, -1j * amp_alpha], dtype=complex)
            force = A2 @ (-omega ** 2 * q) + A1 @ (1j * omega * q) + A0 @ q
            lift_ref, moment_ref = -force[0], force[1]
            rows.append(
                {
                    "k": float(k),
                    "mode": mode,
                    "lift_ratio": float(abs(lift_hat) / abs(lift_ref)),
                    "lift_phase_deg": float(np.degrees(np.angle(lift_hat / lift_ref))),
                    "moment_ratio": float(abs(moment_hat) / abs(moment_ref)),
                    "moment_phase_deg": float(np.degrees(np.angle(moment_hat / moment_ref))),
                }
            )
    return rows


def mass_ratio_scaling(
    mu_values: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0, 140.0),
    *,
    sigma: float = 0.2,
    r_alpha: float = 0.5,
    use_marching: bool = False,
    **kwargs,
) -> dict:
    """Fit ``U_cr ∝ mu**n`` and report the exponent ``n`` (asymptote: ``n = 0.5``).

    With ``use_marching`` the critical velocities come from the time-domain
    coupled solver, otherwise from the p-k reference (much cheaper).
    """
    rows = []
    for mu in mu_values:
        section = TypicalSection(mu=mu, sigma=sigma, r_alpha=r_alpha)
        if use_marching:
            u_cr = flutter_boundary_marching(section, **kwargs)["U_cr"]
        else:
            u_cr = flutter_boundary(section).U_cr
        rows.append({"mu": float(mu), "U_cr": float(u_cr)})
    log_mu = np.log([row["mu"] for row in rows])
    log_u = np.log([row["U_cr"] for row in rows])
    exponent = float(np.polyfit(log_mu, log_u, 1)[0])
    return {"points": rows, "exponent": exponent, "error_vs_half": abs(exponent - 0.5) / 0.5}


def verification_bundle(
    *,
    mu: float = 20.0,
    sigma: float = 0.2,
    r_alpha: float = 0.5,
    n_panels: int = 64,
    periods: float = 5.0,
) -> dict:
    """End-to-end verification of one typical section (used by the N07 benchmark).

    Returns the frequency-domain boundary, the time-domain boundary, the growth
    rates at a few velocities, the mass-ratio exponent and the negative control.
    """
    section = TypicalSection(mu=mu, sigma=sigma, r_alpha=r_alpha)
    reference = flutter_boundary(section)
    marching = flutter_boundary_marching(
        section, periods=periods, n_panels=n_panels
    )
    u_cr_err = abs(marching["U_cr"] - reference.U_cr) / reference.U_cr

    growth_rows = []
    for factor in (0.9, 1.0, 1.1):
        U = factor * reference.U_cr
        sim = march(section, U, periods=periods, n_panels=n_panels)
        eig = pk_eigenvalues(section, U)
        growth_rows.append(
            {
                "U": U,
                "growth_marching": sim.growth,
                "growth_pk": eig.growth_rate,
                "omega_marching": sim.frequency,
                "omega_pk": eig.frequency,
                "growth_rel_error": abs(sim.growth - eig.growth_rate)
                / max(abs(eig.growth_rate), 1e-12),
                "omega_rel_error": abs(sim.frequency - eig.frequency) / eig.frequency,
            }
        )

    scaling = mass_ratio_scaling()
    low_density = TypicalSection(mu=10.0 * mu, sigma=sigma, r_alpha=r_alpha)
    negative = flutter_boundary(low_density)
    return {
        "section": {
            "mu": mu,
            "sigma": sigma,
            "r_alpha": r_alpha,
            "x_alpha": section.x_alpha,
            "a": section.a,
        },
        "reference": {
            "U_cr": reference.U_cr,
            "omega_fl": reference.omega_fl,
            "k_fl": reference.k_fl,
        },
        "marching": marching,
        "U_cr_rel_error": u_cr_err,
        "growth_rows": growth_rows,
        "mass_ratio_scaling": scaling,
        "negative_control": {
            "mu_low_density": low_density.mu,
            "U_cr_low_density": negative.U_cr,
            "ratio": negative.U_cr / reference.U_cr,
            "expected_ratio": math.sqrt(low_density.mu / mu),
        },
    }