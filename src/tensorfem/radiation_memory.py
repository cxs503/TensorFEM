"""Cummins (time-domain) radiation memory kernel with a Kramers-Kronig causality gate.

Why this module exists
----------------------
Every hydrodynamic coupling already in this repository stops at a *constant*
added mass:

* :mod:`tensorfem.added_mass` adds ``m_a = c_m * rho_w * L`` per unit area,
* :mod:`tensorfem.hydro_added_mass` derives the ``k -> 0`` (Rayleigh) added mass
  of a baffled panel,
* :mod:`tensorfem.fsi_wet_modal` shifts a dry mode to the wet frequency with an
  added-mass ratio ``mu``, and
* :mod:`tensorfem.buoyancy_coupling` restores the hydrostatic stiffness.

None of them carries **radiation damping**, and none of them can represent the
frequency dependence of the fluid reaction.  For a floating structure that is a
structural-dynamics error, not a cosmetic one: the radiation damping is what sets
the decay rate of every rigid-body and hydroelastic mode, and the added mass that
belongs with it is only defined *together* with it, through causality.

The Cummins (1962) formulation supplies exactly that.  With ``A_inf`` the
infinite-frequency added mass and ``K(t)`` the radiation *memory kernel*, the
fluid reaction on a body moving with velocity ``v`` is

    F_rad(t) = -A_inf * a(t) - integral_0^t K(t - tau) * v(tau) dtau,

and the frequency-domain hydrodynamic coefficients ``A(omega)``, ``B(omega)`` are
the cosine/sine transforms of the same kernel (Ogilvie 1964):

    B(omega)      = integral_0^inf K(t) cos(omega t) dt              (1)
    A(omega)-A_inf = -(1/omega) integral_0^inf K(t) sin(omega t) dt   (2)
      <=>  K(t)   = (2/pi) integral_0^inf B(omega) cos(omega t) domega  (3)

Because ``K(t) = 0`` for ``t < 0``, ``B(omega)`` and ``A(omega)`` are not free
functions: they are a Hilbert (Kramers-Kronig) pair,

    A(omega) - A_inf = -(2/pi) CPV integral_0^inf B(s) / (omega^2 - s^2) ds.  (4)

The finite-band residual in (4) tests consistency at the chosen frequency
resolution and bandwidth. Both missing tails and inconsistent tables can
increase it. Passivity additionally requires nonnegative radiation damping
(or a positive semidefinite damping matrix).

Numerics
--------
* (3) is a trapezoidal cosine transform; the caller owns the truncation of the
  frequency band (see :func:`kernel_truncation_estimate`, which bounds the
  neglected tail for a decaying damping curve).
* (4) uses singularity subtraction on the supplied band [L, S]. Integrate
  (B(s)-B(w))/(w^2-s^2) by quadrature and add B(w)/(2w) times
  log((S+w)/(S-w)) - log((w+L)/(w-L)) before multiplying by -2/pi.
  Requested frequencies lie strictly inside the band. A supplied B(0)
  extends the trapezoidal rule to L=0.
* The time integrator uses the trapezoidal memory quadrature with the
  ``K(0) v(t) / 2`` term taken implicitly (it enters the effective damping), so
  the scheme stays second order and stable for stiff kernels.

Conventions
-----------
SI units throughout (m, kg, s, rad/s).  ``omega[0]`` is the lowest tabulated
frequency. Supply an independently known zero-frequency endpoint when the
table omits it. Without that input the transform covers the measured band.

The module is deliberately free of any hull geometry: ``A``/``B`` are *inputs*,
so the coefficients of any radiation solver (this repository's own or an
external BEM) can be fed in through the public functions without touching the
maths.  :func:`single_pole_pair` is the exact reference pair used by the test
suite, and :func:`decay_damping_ratio` is the bridge back to the wet-modal
world (:mod:`tensorfem.fsi_wet_modal`), where a frequency-dependent radiation
damping ratio
``zeta(omega_n) = B(omega_n) / (2 omega_n (M + A(omega_n)))``
is the quantity a mode needs.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numpy import trapezoid
except ImportError:  # NumPy 1.26
    from numpy import trapz as trapezoid

__all__ = [
    "single_pole_pair",
    "cummins_kernel",
    "damping_from_kernel",
    "added_mass_from_kernel",
    "kernel_truncation_estimate",
    "kramers_kronig_added_mass",
    "kramers_kronig_residual",
    "memory_force",
    "fit_exponential_kernel",
    "StateSpaceMemory",
    "simulate_cummins",
    "decay_damping_ratio",
]


# --------------------------------------------------------------------------- #
# reference pair
# --------------------------------------------------------------------------- #
def single_pole_pair(times, omega, coefficient=1.0, rate=1.0, added_mass_inf=0.0):
    """Exact causal reference: ``K(t) = c exp(-rate t)`` for ``t >= 0``.

    The transforms (1)-(2) are elementary,

        B(omega)       = c rate / (rate^2 + omega^2),
        A(omega)-A_inf = -c / (rate^2 + omega^2),

    so this pair satisfies the Kramers-Kronig relation (4) exactly (the negative
    added-mass increment is the signature of a kernel with a net positive area;
    real hull kernels cross zero, which flips the sign of the increment at high
    frequency).  Every transform in this module is validated against it.
    """
    times = np.asarray(times, dtype=float)
    omega = np.asarray(omega, dtype=float)
    kernel = coefficient * np.exp(-rate * times)
    denom = rate * rate + omega * omega
    damping = coefficient * rate / denom
    added_mass = np.asarray(added_mass_inf, dtype=float) - coefficient / denom
    return {"kernel": kernel, "damping": damping, "added_mass": added_mass}


def _include_origin(omega, damping, zero_frequency_damping):
    if zero_frequency_damping is not None:
        endpoint = float(zero_frequency_damping)
        if not math.isfinite(endpoint):
            raise ValueError("zero_frequency_damping must be finite")
        if omega[0] > 0:
            omega = np.r_[0.0, omega]
            damping = np.r_[endpoint, damping]
        elif damping[0] != endpoint:
            raise ValueError("zero_frequency_damping conflicts with the tabulated origin")
    return omega, damping


def cummins_kernel(omega, damping, times, *, zero_frequency_damping=None):
    """Memory kernel ``K(t) = (2/pi) integral_0^inf B(omega) cos(omega t) domega``.

    Parameters
    ----------
    omega, damping : array_like, shape (n,)
        Tabulated frequency band, strictly increasing, and the damping
        coefficient at those frequencies.  The band must resolve ``B`` down to a
        small fraction of its peak (:func:`kernel_truncation_estimate` reports
        what the neglected tail contributes).
    zero_frequency_damping : float or None
        Independently known B(0). Include the interval from zero to the first
        tabulated frequency with the same trapezoid rule. If omitted, integrate
        only the measured band. The caller must resolve low-frequency structure.
    times : array_like, shape (m,)
        Lags at which ``K`` is wanted.  Lags start at ``0`` - the kernel is
        symmetric and defined for ``t >= 0`` only.

    Returns
    -------
    ndarray, shape (m,)
    """
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    _check_band(omega)
    if damping.shape != omega.shape:
        raise ValueError("damping must have the same shape as omega")
    if not (np.isfinite(omega).all() and np.isfinite(damping).all() and np.isfinite(times).all()):
        raise ValueError("frequencies, damping and lags must be finite")
    omega, damping = _include_origin(omega, damping, zero_frequency_damping)
    if np.any(times < 0.0):
        raise ValueError("times must be non-negative; K is causal on t >= 0")
    return _chunked_cosine_transform(omega, damping, times, 2.0 / math.pi)


def damping_from_kernel(times, kernel, omega):
    """``B(omega) = integral_0^inf K(t) cos(omega t) dt`` - the inverse of (3)."""
    times = np.asarray(times, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if times.shape != kernel.shape:
        raise ValueError("kernel must have the same shape as times")
    return _chunked_cosine_transform(times, kernel, omega, 1.0)


def added_mass_from_kernel(times, kernel, omega, added_mass_inf):
    """Added mass from the kernel: ``A(w) = A_inf - (1/w) integral K(t) sin(w t) dt``.

    The minus sign is the one that makes the Cummins reaction
    ``-A_inf a - integral K v`` equal the frequency-domain ``-A a - B v`` (Ogilvie
    1964); with a plus sign the reconstruction instead returns ``2 A_inf - A``.
    ``omega -> 0`` is excluded: there the relation degenerates to the static added
    mass, which is a separate zero-frequency solve.
    """
    times = np.asarray(times, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if times.shape != kernel.shape:
        raise ValueError("kernel must have the same shape as times")
    if np.any(omega <= 0.0):
        raise ValueError("added_mass_from_kernel needs omega > 0 (see docstring)")
    transformed = _chunked_sine_transform(times, kernel, omega)
    return np.asarray(added_mass_inf, dtype=float) - transformed / omega


def kernel_truncation_estimate(omega, damping, times, tail_exponent=2.0, *, zero_frequency_damping=None):
    """Bound on the part of ``K(t)`` coming from ``omega > omega_max``.

    The integrand of (3) is bounded by ``B(omega)``, so a damping curve that
    decays as ``B ~ B_max (omega_max/omega)^p`` for ``omega > omega_max``
    contributes at most

        (2/pi) * B_max * omega_max^p * integral_{omega_max}^inf omega^-p domega
            = (2/pi) * B_max * omega_max / (p - 1)

    to ``||K||_inf`` (and less to a time-average, since the cosine oscillates).
    ``p = 2`` is the conservative default for a 3-D body, whose damping falls off
    faster than that above the first radiation hump.

    Returns ``dict`` with ``bound`` (absolute, same units as ``K``),
    ``peak_kernel`` (``max |K(t)|`` on the supplied lags) and ``relative``.
    """
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)
    times = np.asarray(times, dtype=float)
    _check_band(omega)
    if not math.isfinite(tail_exponent) or tail_exponent <= 1.0:
        raise ValueError("tail_exponent must exceed 1 for a converged tail bound")
    tail_amplitude = abs(float(damping[-1]))
    bound = (2.0 / math.pi) * tail_amplitude * omega[-1] / (tail_exponent - 1.0)
    kernel = cummins_kernel(omega, damping, times, zero_frequency_damping=zero_frequency_damping)
    peak_kernel = float(np.max(np.abs(kernel))) if kernel.size else 0.0
    return {
        "bound": bound,
        "peak_kernel": peak_kernel,
        "relative": bound / peak_kernel if peak_kernel > 0.0 else math.inf,
    }


# --------------------------------------------------------------------------- #
# (4) Kramers-Kronig causality gate
# --------------------------------------------------------------------------- #
def kramers_kronig_added_mass(omega, damping, added_mass_inf, at=None, *, zero_frequency_damping=None):
    """Added mass reconstructed from ``B(omega)`` through the KK relation (4).

    Parameters
    ----------
    omega, damping : array_like, shape (n,)
        Tabulated band (``omega`` strictly increasing, ``n >= 5``).
    added_mass_inf : float
        ``A(omega -> inf)``; the KK transform fixes only the shape of ``A``, this
        constant fixes its level.
    at : array_like, optional
        Frequencies where the reconstruction is wanted, strictly inside
        ``(omega[0], omega[-1])``.  At a band edge the CPV integral of
        ``1/(omega^2-s^2)`` is logarithmically unbounded because the band beyond
        the table is missing, and that divergence belongs to the truncation.
        Defaults to ``omega[1:-1]``.

    Returns
    -------
    ndarray, shape (n_at,)
        Reconstructed added mass, same convention as the input ``B`` (the
        coefficient multiplying ``-acceleration``).

    Notes
    -----
    Subtract B(w) before quadrature; the pole then has the finite limit
    -B'(w)/(2w). Integrate the subtracted constant analytically between the
    actual lower and upper endpoints of the band. Optional B(0) extends the
    interval in the same way as in cummins_kernel.
    """
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)
    _check_band(omega)
    if damping.shape != omega.shape:
        raise ValueError("damping must have the same shape as omega")
    if not np.isfinite(damping).all() or not np.isfinite(added_mass_inf):
        raise ValueError("damping and added mass must be finite")
    band_top = float(omega[-1])
    if at is None:
        at = omega[1:-1]
    omega, damping = _include_origin(omega, damping, zero_frequency_damping)
    at = np.atleast_1d(np.asarray(at, dtype=float))
    if at.ndim != 1 or not np.isfinite(at).all():
        raise ValueError("evaluation frequencies must form a finite vector")
    if np.any(at <= omega[0]) or np.any(at >= band_top):
        raise ValueError(
            "KK reconstruction needs frequencies strictly inside the open "
            "tabulated band (the CPV integral is unbounded at a band edge)"
        )
    out = np.empty(at.size)
    for i, w in enumerate(at):
        w = float(w)
        b_w = float(np.interp(w, omega, damping))
        hit = omega == w
        regular = np.empty_like(omega)
        regular[~hit] = (damping[~hit] - b_w) / ((w - omega[~hit]) * (w + omega[~hit]))
        if hit.any():
            idx = int(np.flatnonzero(hit)[0])
            slope = (damping[idx+1] - damping[idx-1]) / (omega[idx+1] - omega[idx-1])
            regular[idx] = -slope / (2*w)
        upper = math.log((band_top + w) / (band_top - w))
        lower = math.log((w + omega[0]) / (w - omega[0]))
        principal = trapezoid(regular, omega) + b_w * (upper-lower) / (2*w)
        out[i] = added_mass_inf - (2.0/math.pi) * principal
    return out


def kramers_kronig_residual(
    omega,
    damping,
    added_mass,
    added_mass_inf,
    tol=0.03,
    band_fraction=0.9,
    pointwise_band=0.05,
    edge_intervals=2,
    *, zero_frequency_damping=None,
):
    """Causality gate: does the tabulated ``(A, B)`` pair satisfy (4)?

    Two residuals are reported because they answer different questions:

    ``residual`` (the gate metric)
        ``max |A_KK - A|`` over the interior band, normalised by the *dynamic
        range* ``max |A - A_inf|``.  This is well defined even where ``A`` has
        relaxed back onto ``A_inf`` (a pointwise ratio there is meaningless), and
        it is deliberately strict: for a floating body ``A - A_inf`` is only a
        fraction of ``A``, so a 3% residual on the dynamic range is a much
        tighter statement than 3% on ``A`` itself.

    ``residual_pointwise``
        ``max |A_KK - A| / |A - A_inf|`` restricted to the frequencies where
        ``|A - A_inf|`` exceeds ``pointwise_band`` times its range, i.e. where a
        relative statement is meaningful.  Diagnostic only.

    Parameters
    ----------
    omega, damping, added_mass : array_like, shape (n,)
        Hydrodynamic coefficients from the radiation solver.
    added_mass_inf : float
        Infinite-frequency added mass used by that solver.
    tol : float
        Pass threshold on ``residual``.
    band_fraction : float
        Only frequencies ``omega <= band_fraction * omega_max`` enter the
        residual; near the top of the band the truncation term
        ``ln|(S+omega)/(S-omega)|`` of the KK quadrature blows up without the
        fluid being at fault, so including it would measure the caller's band
        choice rather than causality.
    pointwise_band : float
        See above.
    edge_intervals : int
        Number of tabulated intervals dropped at the bottom of the band.  The
        KK quadrature cannot resolve ``omega`` within a few grid spacings of
        ``omega_min``: the missing ``s < omega_min`` band and the pole sit at the
        same scale there.  ``docs/marine-controls.md`` quantifies the
        resulting residual as a function of table resolution.

    Returns
    -------
    dict with ``residual``, ``residual_pointwise``, ``passed``, ``tol``,
    ``reconstructed`` (at the interior band), ``worst_omega``, ``band``.
    """
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)
    added_mass = np.asarray(added_mass, dtype=float)
    _check_band(omega)
    if not (damping.shape == omega.shape == added_mass.shape):
        raise ValueError("omega, damping and added_mass must share one shape")
    if not (0.0 < band_fraction < 1.0):
        raise ValueError("band_fraction must lie in (0, 1)")
    if edge_intervals < 1:
        raise ValueError("edge_intervals must be at least 1")
    floor_index = min(int(edge_intervals), omega.size - 3)
    mask = (omega <= band_fraction * omega[-1]) & (np.arange(omega.size) > floor_index)
    if np.count_nonzero(mask) < 3:
        raise ValueError("interior band too short for a residual; widen the band")
    at = omega[mask]
    reconstructed = kramers_kronig_added_mass(omega, damping, added_mass_inf, at=at,
                                               zero_frequency_damping=zero_frequency_damping)
    deviation = np.abs(reconstructed - added_mass[mask])
    dynamic = np.abs(added_mass[mask] - added_mass_inf)
    scale = float(np.max(dynamic))
    if scale <= 0.0:
        raise ValueError("added mass equals A_inf on the band; residual undefined")
    relative = deviation / scale
    worst = int(np.argmax(relative))
    wide = dynamic > pointwise_band * scale
    residual_pointwise = (
        float(np.max(deviation[wide] / dynamic[wide])) if np.any(wide) else 0.0
    )
    residual = float(relative[worst])
    return {
        "residual": residual,
        "residual_pointwise": residual_pointwise,
        "passed": bool(residual <= tol),
        "tol": float(tol),
        "reconstructed": reconstructed,
        "worst_omega": float(at[worst]),
        "band": (float(at[0]), float(at[-1])),
    }


# --------------------------------------------------------------------------- #
# memory force / convolution
# --------------------------------------------------------------------------- #
def memory_force(times, kernel, velocity):
    """``integral_0^t K(t - tau) v(tau) dtau`` on the sample grid (trapezoidal).

    ``kernel`` is ``K`` sampled at the same lags as ``times``; ``velocity`` is the
    velocity history on the same grid.  The convolution is evaluated at every
    sample (``O(n^2)``), which is the honest reference for the recursive
    state-space route (:class:`StateSpaceMemory`).
    """
    times = np.asarray(times, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if not (times.shape == kernel.shape == velocity.shape):
        raise ValueError("times, kernel and velocity must share one shape")
    dt = np.diff(times)
    if np.any(dt <= 0.0) or not np.allclose(dt, dt[0], rtol=1e-9, atol=0.0):
        raise ValueError("times must be uniformly spaced and increasing")
    n = times.size
    out = np.empty(n)
    for i in range(n):
        lag = kernel[: i + 1][::-1]
        hist = velocity[: i + 1]
        weights = np.ones(i + 1)
        weights[0] = weights[-1] = 0.5
        out[i] = dt[0] * np.dot(weights * lag, hist)
    return out


# --------------------------------------------------------------------------- #
# (becomes O(1) per step) exponential / state-space kernel
# --------------------------------------------------------------------------- #
def fit_exponential_kernel(times, kernel, n_modes=8, decay_rates=None):
    """Least-squares fit ``K(t) ~ sum_j c_j exp(-lambda_j t)`` with positive rates.

    Fixed (log-spaced) rates keep this a linear least-squares problem, which is
    all the state-space route needs: the modes are quadrature nodes of the
    kernel's decay spectrum, in the same spirit as a Laguerre expansion.  Column
    scaling is applied before the solve so the basis does not become
    ill-conditioned when the rates span decades.

    Parameters
    ----------
    times, kernel : array_like, shape (n,)
        Sampled kernel on a uniform grid starting at ``0``.
    n_modes : int
        Number of exponential modes (i.e. the state dimension of the
        approximation).
    decay_rates : array_like, optional
        Explicit rates ``lambda_j > 0`` (mode ``exp(-lambda_j t)``).  Defaults to
        ``n_modes`` rates log-spaced between ``0.3/T`` and ``30/T`` (``T`` the kernel
        window: a truncated kernel that has not decayed inside the window cannot
        be represented whatever the rates, so rates far above ``1/dt`` buy
        nothing and destroy the conditioning of the basis).

    Returns
    -------
    dict with ``coefficients``, ``rates``, ``fit`` (the approximation sampled on
    ``times``), ``relative_error`` (``max |fit-K| / max|K|``) and ``residual_l2``.
    """
    times = np.asarray(times, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if times.shape != kernel.shape:
        raise ValueError("kernel must have the same shape as times")
    if times[0] != 0.0:
        raise ValueError("times must start at 0 (kernel lag grid)")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    if decay_rates is None:
        span = float(times[-1] - times[0])
        dt = float(times[1] - times[0])
        if span <= 0.0 or dt <= 0.0:
            raise ValueError("times must span a positive interval")
        lo = 0.3 / span
        hi = 30.0 / span
        rates = np.exp(np.linspace(math.log(lo), math.log(hi), n_modes))
    else:
        rates = np.asarray(decay_rates, dtype=float)
        if np.any(rates <= 0.0):
            raise ValueError("decay rates must be positive (Cummins: causal K)")
        n_modes = rates.size
    basis = np.exp(-np.outer(times, rates))
    column_norm = np.linalg.norm(basis, axis=0)
    column_norm[column_norm == 0.0] = 1.0
    scaled = basis / column_norm
    coefficients, *_ = np.linalg.lstsq(scaled, kernel, rcond=None)
    coefficients = coefficients / column_norm
    fit = basis @ coefficients
    peak = float(np.max(np.abs(kernel)))
    residual = fit - kernel
    return {
        "coefficients": coefficients,
        "rates": rates,
        "fit": fit,
        "relative_error": float(np.max(np.abs(residual)) / peak) if peak > 0 else 0.0,
        "residual_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(kernel), 1e-300)),
    }


class StateSpaceMemory:
    """Recursive-convolution radiation memory ``K(t) = sum_j c_j exp(p_j t)``.

    The memory force ``integral_0^t K(t-tau) v(tau) dtau`` is carried by the
    states ``z_j`` with

        dz_j/dt = p_j z_j + v(t),      F_mem = sum_j c_j z_j,

    discretised with a zero-order hold on the velocity (exact for a piecewise
    constant input), so one step costs ``O(n_modes)`` instead of ``O(n_steps)``.
    Poles are required to lie in the closed left half plane: a pole with a
    positive real part would mean a kernel that grows with lag, i.e. energy
    injection by a passive fluid.
    """

    def __init__(self, coefficients, poles):
        coefficients = np.asarray(coefficients, dtype=complex)
        poles = np.asarray(poles, dtype=complex)
        if coefficients.shape != poles.shape:
            raise ValueError("coefficients and poles must share one shape")
        if np.any(poles.real > 1e-12 * np.maximum(1.0, np.abs(poles))):
            raise ValueError("radiation-memory poles must have Re(p) <= 0")
        self.coefficients = coefficients
        self.poles = poles
        self._state = np.zeros_like(poles)

    def reset(self):
        self._state = np.zeros_like(self.poles)

    def step(self, velocity, dt):
        """Advance one step with velocity held constant over ``[t, t+dt]``."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        decay = np.exp(self.poles * dt)
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.where(
                np.abs(self.poles) > 1e-12,
                (decay - 1.0) / self.poles,
                dt,  # p -> 0 limit
            )
        self._state = decay * self._state + gain * velocity
        return float(np.real(np.dot(self.coefficients, self._state)))

    def force(self, velocity_history, dt):
        """Memory force for a prescribed velocity history (list of samples)."""
        self.reset()
        out = np.empty(len(velocity_history))
        for i, v in enumerate(velocity_history):
            out[i] = self.step(v, dt)
        return out


# --------------------------------------------------------------------------- #
# time-domain integration of the Cummins equation
# --------------------------------------------------------------------------- #
def simulate_cummins(
    mass,
    added_mass_inf,
    restoring,
    kernel_times,
    kernel,
    dt,
    n_steps,
    force=None,
    viscous_damping=0.0,
    x0=0.0,
    v0=0.0,
    state_space=None,
):
    """Integrate ``(M + A_inf) a + int_0^t K(t-tau) v dtau + C x + B_v v = F(t)``.

    Multi-degree-of-freedom input is supported by passing matrices/vectors:
    ``mass``, ``added_mass_inf``, ``restoring``, ``viscous_damping`` and
    ``kernel`` may all be ``(ndof, ndof)`` (``kernel`` then has shape
    ``(n_lag, ndof, ndof)``), and ``force`` is a callable
    ``force(t) -> ndarray(ndof)``.  A scalar kernel is applied as a diagonal
    radiation operator.

    Scheme: trapezoidal (second order) in time for both the equation of motion
    and the memory quadrature.  With ``M_{n+1}`` written out,

        M_{n+1} = dt [ 1/2 K_0 v_{n+1} + sum_{j=1..n} K_{n+1-j} v_j
                       + 1/2 K_{n+1} v_0 ],

    the only term carrying the unknown ``v_{n+1}`` is ``1/2 dt K_0 v_{n+1}``; it
    is folded into the effective damping (together with the ``1/2 dt C`` term
    from ``x_{n+1} = x_n + 1/2 dt (v_n + v_{n+1})``), which keeps the scheme
    stable for a stiff kernel while leaving it a linear solve per step.

    Parameters
    ----------
    kernel_times : array_like, shape (n_lag,)
        Lags ``0 .. (n_lag-1) * dt`` at which ``kernel`` is sampled (uniform grid
        starting at zero; the lag spacing must equal ``dt``).
    dt : float
        Time step.
    n_steps : int
        Number of steps to integrate (``n_steps + 1`` samples returned).
    state_space : StateSpaceMemory, optional
        If given, the recursive model supplies the entire memory force using
        piecewise-constant velocity instead of the direct trapezoidal convolution.  The recursive state
        is advanced with the velocity of the previous step (zero-order hold), so
        that route is first order in the memory coupling; the direct route is the
        reference and the two must agree to ``O(dt)`` (asserted in the tests).

    Returns
    -------
    dict with ``t``, ``x``, ``v``, ``a``, ``memory_force``.  With scalar input the
    state arrays are 1-D; with matrix input they are ``(n_steps+1, ndof)``.
    The kernel window ``(n_lag - 1) * dt`` truncates the memory: a lag beyond the
    window contributes nothing.  Set ``n_lag`` so that ``K`` has decayed over the
    window (the returned ``memory_window`` says what was used).

    ``memory_force`` is the *positive* convolution
    ``int_0^t K(t-tau) v dtau`` (the fluid reaction also carries ``-A_inf a``).
    """
    kernel_times = np.asarray(kernel_times, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if kernel_times.ndim != 1 or kernel_times.size < 2 or kernel_times[0] != 0.0:
        raise ValueError("kernel_times must be a 1-D grid of >= 2 lags starting at 0")
    if not np.allclose(np.diff(kernel_times), dt, rtol=1e-9, atol=0):
        raise ValueError("kernel lag spacing must equal dt")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    n_lag = kernel_times.size

    scalar = np.ndim(mass) == 0 and np.ndim(added_mass_inf) == 0
    matrix_mass = mass if np.ndim(mass) else added_mass_inf
    ndof = 1 if scalar else np.atleast_2d(matrix_mass).shape[0]
    if kernel.shape == (n_lag,):
        kernel_mats = np.zeros((n_lag, ndof, ndof))
        for i in range(ndof):
            kernel_mats[:, i, i] = kernel
    elif kernel.ndim == 3 and kernel.shape[0] == n_lag and kernel.shape[1:] == (ndof, ndof):
        kernel_mats = kernel
    else:
        raise ValueError("kernel must be (n_lag,) or (n_lag, ndof, ndof)")

    def _as_matrix(value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.eye(ndof) * float(arr)
        if arr.shape == (ndof, ndof):
            return arr
        raise ValueError("matrix input must be (ndof, ndof) or a scalar")

    m_eff = _as_matrix(mass) + _as_matrix(added_mass_inf)
    rest = _as_matrix(restoring)
    visc = _as_matrix(viscous_damping)
    k0 = kernel_mats[0]
    forcing = (lambda _t: np.zeros(ndof)) if force is None else force

    if state_space is not None:
        if ndof != 1:
            raise ValueError("recursive memory requires a scalar motion")
        state_space.reset()
    implicit_memory = 0.5 * dt * k0 if state_space is None else np.zeros_like(k0)
    lhs = m_eff / dt + 0.5 * (visc + implicit_memory) + 0.25 * dt * rest
    x = np.zeros((n_steps + 1, ndof))
    v = np.zeros((n_steps + 1, ndof))
    a = np.zeros((n_steps + 1, ndof))
    mem = np.zeros((n_steps + 1, ndof))
    x[0] = x0
    v[0] = v0
    t = np.arange(n_steps + 1) * dt

    last_force = np.asarray(forcing(t[0]), dtype=float)
    a[0] = np.linalg.solve(m_eff, last_force - rest @ x[0] - visc @ v[0])
    for n in range(n_steps):
        # trapezoid memory over the available lags; the j == n term is implicit
        m = min(n + 1, n_lag - 1)
        if state_space is not None:
            known = np.array([state_space.step(float(v[n][0]), dt)])
        else:
            hist = v[n - m + 1 : n + 1][::-1]      # v_n, v_{n-1}, ..., v_{n-m+1}
            lags = kernel_mats[1 : m + 1]          # K_1 .. K_m
            weights = np.ones(m)
            weights[-1] = 0.5  # both endpoints of the available integration window
            known = dt * np.einsum("kij,kj->i", lags, hist * weights[:, None])
        next_force = np.asarray(forcing(t[n + 1]), dtype=float)
        # Average both endpoint forces in the momentum equation, as well as
        # velocities in the displacement equation (trapezoidal Newmark).
        rhs = (0.5*(last_force + next_force - mem[n] - known - visc @ v[n])
               - rest @ (x[n] + 0.25*dt*v[n]) + m_eff @ v[n] / dt)
        v[n + 1] = np.linalg.solve(lhs, rhs)
        x[n + 1] = x[n] + 0.5*dt*(v[n] + v[n+1])
        mem[n + 1] = known + implicit_memory @ v[n+1]
        a[n + 1] = np.linalg.solve(m_eff, next_force - rest @ x[n+1] - visc @ v[n+1] - mem[n+1])
        last_force = next_force
    memory_window = float((n_lag - 1) * dt)
    if scalar:
        return {
            "t": t,
            "memory_window": memory_window,
            "x": x[:, 0],
            "v": v[:, 0],
            "a": a[:, 0],
            "memory_force": mem[:, 0],
        }
    return {"t": t, "x": x, "v": v, "a": a, "memory_force": mem,
            "memory_window": memory_window}


def decay_damping_ratio(omega, added_mass, damping, mass, omega_n):
    """``zeta = B(omega_n) / (2 omega_n (M + A(omega_n)))`` (linearised).

    This is the bridge to :mod:`tensorfem.fsi_wet_modal`: the wet frequency gives
    the frequency at which the *frequency-dependent* radiation damping must be
    read, and the added mass at that same frequency is what belongs with it.
    """
    omega = np.asarray(omega, dtype=float)
    added_mass = np.asarray(added_mass, dtype=float)
    damping = np.asarray(damping, dtype=float)
    if not (omega.shape == added_mass.shape == damping.shape):
        raise ValueError("omega, added_mass and damping must share one shape")
    if not (omega[0] <= omega_n <= omega[-1]):
        raise ValueError("omega_n lies outside the tabulated band")
    b_n = float(np.interp(omega_n, omega, damping))
    a_n = float(np.interp(omega_n, omega, added_mass))
    inertia = float(mass) + a_n
    if inertia <= 0.0 or omega_n <= 0.0:
        raise ValueError("non-physical inertia or frequency")
    return b_n / (2.0 * omega_n * inertia)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _chunked_cosine_transform(grid, values, targets, prefactor, chunk_elements=4_000_000):
    """Trapezoidal ``prefactor * integral values(w) cos(w t) dw`` over a grid.

    Chunked over ``targets`` so the dense ``(n_targets, n_grid)`` cosine matrix
    never has to be materialised: a 2000-lag kernel on a 20 000-point band is
    otherwise a 320 MB temporary.
    """
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    out = np.empty(targets.size)
    step = max(1, int(chunk_elements // max(grid.size, 1)))
    for start in range(0, targets.size, step):
        block = targets[start : start + step]
        out[start : start + block.size] = trapezoid(
            values[None, :] * np.cos(np.outer(block, grid)), grid, axis=1
        )
    return prefactor * out


def _chunked_sine_transform(grid, values, targets, chunk_elements=4_000_000):
    """Trapezoidal ``integral values(w) sin(w t) dw`` (no prefactor); see above."""
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    out = np.empty(targets.size)
    step = max(1, int(chunk_elements // max(grid.size, 1)))
    for start in range(0, targets.size, step):
        block = targets[start : start + step]
        out[start : start + block.size] = trapezoid(
            values[None, :] * np.sin(np.outer(block, grid)), grid, axis=1
        )
    return out


def _check_band(omega):
    if omega.ndim != 1 or omega.size < 5:
        raise ValueError("a frequency band needs at least 5 samples")
    if not np.isfinite(omega).all():
        raise ValueError("frequency band must be finite")
    if np.any(np.diff(omega) <= 0.0):
        raise ValueError("omega must be strictly increasing")
    if np.any(omega < 0.0):
        raise ValueError("omega must be non-negative")