"""Wave spectra and short-term statistics for seakeeping load cases.

``tensorfem.marine_hydrodynamics`` generates loads from a *regular* Airy wave:
one height, one period.  Real seaways are irregular, so a design case needs a
spectrum, the spectral moments that follow from it, and the short-term extreme
statistics that a structural screen actually asks for.  This module adds that
missing link and nothing else -- it is not a seakeeping solver: no RAO, no
response spectrum, no long-term scatter diagram.

Two spectra are provided:

``PiersonMoskowitz``
    Fully developed sea,

        S(w) = alpha * g^2 * w^-5 * exp(-1.25 * (wp / w)^4),

    normalised here by the significant wave height ``Hs = 4 sqrt(m0)``.  All of
    its spectral moments of order ``n < 4`` have a **closed form**

        m_n = alpha g^2 / 4 * beta^((n-4)/4) * wp^(n-4) * Gamma((4-n)/4),

    which makes it an exact oracle for the quadrature used below.

``Jonswap``
    Fetch-limited sea, ``S_J = S_PM * gamma^r`` with the Hasselmann peak
    enhancement ``r = exp(-(w - wp)^2 / (2 sigma^2 wp^2))``, ``sigma = 0.07``
    for ``w <= wp`` and ``0.09`` above.  Its moments have no closed form and are
    integrated numerically; the amplitude ``alpha`` is solved *exactly* from the
    Hs normalisation because ``m0`` is linear in ``alpha``.

Derived quantities follow the standard narrow-band / Rayleigh package:

* ``Hs = 4 sqrt(m0)``, ``Tz = 2 pi sqrt(m0/m2)``, ``Tm = 2 pi m0/m1``,
  ``Tc = 2 pi sqrt(m2/m4)`` (all periods in seconds)
* Rayleigh exceedance ``P(H > h) = exp(-h^2 / (8 m0))``
* most probable largest wave in ``N`` cycles, from the exact median of the
  sample maximum ``(1 - exp(-h^2/(8 m0)))^N = 1/2``, together with the classic
  asymptotic ``sqrt(8 m0 ln N)``

The narrow-band assumption is *not* free: it over-predicts extremes by up to
about 10 % for realistic bandwidths, so the numbers here are a screening
envelope, not a certified extreme-value model.

Everything is validated in ``docs/marine-wave-spectrum-verification.md``; the
driver is ``scripts/audit_wave_spectrum.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

GRAVITY = 9.80665
"""Standard gravity [m/s^2] used by every spectrum unless overridden."""

__all__ = [
    "GRAVITY",
    "PiersonMoskowitz",
    "Jonswap",
    "ShortTermStatistics",
    "short_term_statistics",
    "composite_gauss_legendre",
    "rayleigh_most_probable_extreme",
    "rayleigh_exceedance",
    "irregular_sea_realisation",
    "equivalent_design_wave",
]


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# quadrature
# ---------------------------------------------------------------------------
_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order not in _GL_CACHE:
        _GL_CACHE[order] = np.polynomial.legendre.leggauss(order)
    return _GL_CACHE[order]


def composite_gauss_legendre(
    function,
    lower: float,
    upper: float,
    intervals: int,
    *,
    order: int = 16,
) -> float:
    """Composite Gauss-Legendre quadrature of ``function`` on ``[lower, upper]``.

    The spectrum moments are integrated on a geometric grid, so this is a small
    self-contained helper rather than a dependency on SciPy.
    """
    if intervals < 1:
        raise ValueError("intervals must be >= 1")
    nodes, weights = _gauss_legendre(order)
    edges = np.linspace(lower, upper, intervals + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    centre = 0.5 * (edges[1:] + edges[:-1])
    points = centre[:, None] + half[:, None] * nodes[None, :]
    values = np.asarray([function(float(p)) for p in points.reshape(-1)]).reshape(points.shape)
    return float(np.sum(half[:, None] * weights[None, :] * values))


def _moment_in_log_frequency(
    density_scalar,
    omega_p: float,
    order: int,
    *,
    tolerance: float = 1e-13,
    t_limit: float = 45.0,
    max_intervals: int = 8192,
) -> dict[str, object]:
    """Return ``integral w^order S(w) dw`` by refinement in ``t = ln(w/omega_p)``.

    Substituting ``w = omega_p exp(t)`` turns the semi-infinite frequency axis
    into a bounded interval on which the integrand decays doubly exponentially.
    Refinement is nested (every level doubles the number of panels), so the
    returned ``converged`` flag also gives a usable quadrature error estimate.
    """

    def integrand(t: float) -> float:
        omega = omega_p * math.exp(t)
        return omega**order * density_scalar(omega) * omega

    intervals = 32
    history: list[tuple[int, float]] = []
    previous = None
    while intervals <= max_intervals:
        value = composite_gauss_legendre(integrand, -t_limit, t_limit, intervals)
        history.append((intervals, value))
        if previous is not None and abs(value - previous) <= tolerance * max(abs(value), 1e-300):
            return {
                "moment": value,
                "intervals": intervals,
                "converged": True,
                "history": history,
            }
        previous = value
        intervals *= 2
    return {
        "moment": float("nan"),
        "intervals": intervals // 2,
        "converged": False,
        "history": history,
    }


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PiersonMoskowitz:
    """Pierson-Moskowitz spectrum normalised by the significant wave height."""

    significant_wave_height: float
    peak_period: float
    gravity: float = GRAVITY
    beta: float = 1.25
    """Phillips-type peak shape constant of the classical PM form."""

    def __post_init__(self) -> None:
        _positive(self.significant_wave_height, "significant_wave_height")
        _positive(self.peak_period, "peak_period")
        _positive(self.gravity, "gravity")
        _positive(self.beta, "beta")

    # -- basic scalars ----------------------------------------------------
    @property
    def peak_angular_frequency(self) -> float:
        return 2.0 * math.pi / self.peak_period

    @property
    def variance(self) -> float:
        """``m0 = (Hs / 4)^2`` -- the normalisation that defines alpha."""
        return (self.significant_wave_height / 4.0) ** 2

    @property
    def alpha(self) -> float:
        """Phillips constant implied by the Hs normalisation (closed form)."""
        wp = self.peak_angular_frequency
        return 4.0 * self.beta * self.variance * wp**4 / self.gravity**2

    # -- spectrum ---------------------------------------------------------
    def density(self, omega):
        if isinstance(omega, torch.Tensor):
            w = omega.clamp_min(torch.finfo(omega.dtype).tiny)
            s = self.peak_angular_frequency / w
            value = self.alpha * self.gravity**2 * w ** (-5) * torch.exp(-self.beta * s**4)
            return torch.where(omega > 0, value, torch.zeros_like(value))
        omega = float(omega)
        if omega <= 0.0:
            return 0.0
        return self._density_scalar(omega)

    def _density_scalar(self, omega: float) -> float:
        s = self.peak_angular_frequency / omega
        return self.alpha * self.gravity**2 * omega ** (-5) * math.exp(-self.beta * s**4)

    # -- exact oracle -----------------------------------------------------
    def closed_form_moment(self, order: int) -> float:
        """Exact ``m_order`` for ``order < 4`` (``m4`` diverges for PM)."""
        if order >= 4:
            raise ValueError("Pierson-Moskowitz moments of order >= 4 diverge")
        wp = self.peak_angular_frequency
        gamma_argument = 0.25 * (4 - order)
        return (
            self.alpha
            * self.gravity**2
            / 4.0
            * self.beta ** ((order - 4) / 4.0)
            * wp ** (order - 4)
            * math.gamma(gamma_argument)
        )

    def moment(self, order: int, *, tolerance: float = 1e-13) -> dict[str, object]:
        if order >= 4:
            raise ValueError("Pierson-Moskowitz moments of order >= 4 diverge")
        return _moment_in_log_frequency(
            self._density_scalar, self.peak_angular_frequency, order, tolerance=tolerance
        )


@dataclass(frozen=True)
class Jonswap:
    """JONSWAP spectrum normalised by the significant wave height."""

    significant_wave_height: float
    peak_period: float
    peak_enhancement: float = 3.3
    gravity: float = GRAVITY
    beta: float = 1.25
    sigma_low: float = 0.07
    sigma_high: float = 0.09
    _alpha: float = field(init=False, repr=False, compare=False, default=float("nan"))

    def __post_init__(self) -> None:
        _positive(self.significant_wave_height, "significant_wave_height")
        _positive(self.peak_period, "peak_period")
        _positive(self.peak_enhancement, "peak_enhancement")
        _positive(self.gravity, "gravity")
        _positive(self.beta, "beta")
        _positive(self.sigma_low, "sigma_low")
        _positive(self.sigma_high, "sigma_high")
        if self.peak_enhancement < 1.0:
            raise ValueError("peak_enhancement must be >= 1 (gamma = 1 is the PM limit)")

    @property
    def peak_angular_frequency(self) -> float:
        return 2.0 * math.pi / self.peak_period

    @property
    def variance(self) -> float:
        return (self.significant_wave_height / 4.0) ** 2

    def _sigma_factor(self, omega: float) -> float:
        if omega <= self.peak_angular_frequency:
            return math.exp(
                -((omega - self.peak_angular_frequency) ** 2)
                / (2.0 * self.sigma_low**2 * self.peak_angular_frequency**2)
            )
        return math.exp(
            -((omega - self.peak_angular_frequency) ** 2)
            / (2.0 * self.sigma_high**2 * self.peak_angular_frequency**2)
        )

    def _unnormalised_density(self, omega: float) -> float:
        """``S / alpha``: the Hs normalisation only scales this functional."""
        if omega <= 0.0:
            return 0.0
        s = self.peak_angular_frequency / omega
        peak = self.peak_enhancement**self._sigma_factor(omega)
        return self.gravity**2 * omega ** (-5) * math.exp(-self.beta * s**4) * peak

    @property
    def alpha(self) -> float:
        """Phillips constant solved from ``Hs = 4 sqrt(m0)`` (exact for m0)."""
        if not math.isfinite(self._alpha):
            integral = _moment_in_log_frequency(
                self._unnormalised_density, self.peak_angular_frequency, 0
            )
            if not integral["converged"]:
                raise RuntimeError("JONSWAP normalisation integral did not converge")
            object.__setattr__(self, "_alpha", self.variance / float(integral["moment"]))
        return self._alpha

    @property
    def peak_enhancement_ratio(self) -> float:
        """``alpha / alpha_PM``: how much energy the peak factor adds."""
        return self.alpha / PiersonMoskowitz(
            self.significant_wave_height, self.peak_period, self.gravity, self.beta
        ).alpha

    def density(self, omega):
        if isinstance(omega, torch.Tensor):
            wp = self.peak_angular_frequency
            w = omega.clamp_min(torch.finfo(omega.dtype).tiny)
            s = wp / w
            sigma = torch.where(omega <= wp, self.sigma_low, self.sigma_high)
            exponent = -((omega - wp) ** 2) / (2.0 * sigma**2 * wp**2)
            # peak enhancement gamma^r with r = exp(-(w-wp)^2/(2 sigma^2 wp^2))
            peak = self.peak_enhancement ** torch.exp(exponent)
            value = (
                self.alpha
                * self.gravity**2
                * w ** (-5)
                * torch.exp(-self.beta * s**4)
                * peak
            )
            return torch.where(omega > 0, value, torch.zeros_like(value))
        omega = float(omega)
        return self.alpha * self._unnormalised_density(omega)

    def moment(self, order: int, *, tolerance: float = 1e-13) -> dict[str, object]:
        return _moment_in_log_frequency(
            lambda w: self.alpha * self._unnormalised_density(w),
            self.peak_angular_frequency,
            order,
            tolerance=tolerance,
        )


# ---------------------------------------------------------------------------
# short-term statistics
# ---------------------------------------------------------------------------
def rayleigh_exceedance(height, m0: float):
    """``P(H > height) = exp(-height^2 / (8 m0))`` for a narrow-band sea."""
    _positive(m0, "m0")
    if isinstance(height, torch.Tensor):
        return torch.exp(-(height**2) / (8.0 * m0))
    return math.exp(-(float(height) ** 2) / (8.0 * m0))


def rayleigh_most_probable_extreme(n_cycles: float, m0: float, *, exact: bool = True) -> float:
    """Most probable largest wave height in ``n_cycles`` cycles.

    ``exact=True`` inverts the exact sample-maximum median
    ``(1 - exp(-h^2/(8 m0)))^N = 1/2``; ``exact=False`` returns the classic
    narrow-band asymptotic ``sqrt(8 m0 ln N)`` used in rule scantlings.
    """
    _positive(n_cycles, "n_cycles")
    _positive(m0, "m0")
    if n_cycles <= 1.0:
        raise ValueError("n_cycles must be > 1 for an extreme statistic")
    if not exact:
        return math.sqrt(8.0 * m0 * math.log(n_cycles))
    # 1 - exp(-h^2/(8 m0)) = 2^(-1/N)  ->  h = sqrt(8 m0 ln(1/(1 - 2^(-1/N))))
    p = 2.0 ** (-1.0 / n_cycles)
    return math.sqrt(8.0 * m0 * math.log(1.0 / (1.0 - p)))


@dataclass(frozen=True)
class ShortTermStatistics:
    """Spectral moments and the derived narrow-band statistics."""

    significant_wave_height: float
    m0: float
    m1: float
    m2: float
    m4: float | None = None
    moment_sources: dict[str, object] = field(default_factory=dict, compare=False)

    @property
    def hs(self) -> float:
        return 4.0 * math.sqrt(self.m0)

    @property
    def mean_period(self) -> float:
        """``Tm = 2 pi m0 / m1`` (mean period, seconds)."""
        return 2.0 * math.pi * self.m0 / self.m1

    @property
    def zero_crossing_period(self) -> float:
        """``Tz = 2 pi sqrt(m0/m2)`` (mean zero-up-crossing period, seconds)."""
        return 2.0 * math.pi * math.sqrt(self.m0 / self.m2)

    @property
    def crest_period(self) -> float | None:
        """``Tc = 2 pi sqrt(m2/m4)``; undefined when ``m4`` diverges (PM)."""
        if self.m4 is None or not math.isfinite(self.m4):
            return None
        return 2.0 * math.pi * math.sqrt(self.m2 / self.m4)

    def most_probable_extreme(self, n_cycles: float, *, exact: bool = True) -> float:
        return rayleigh_most_probable_extreme(n_cycles, self.m0, exact=exact)

    def as_dict(self) -> dict[str, object]:
        crest = self.crest_period
        return {
            "hs_m": self.hs,
            "target_hs_m": self.significant_wave_height,
            "m0_m2": self.m0,
            "m1_m2_per_s": self.m1,
            "m2_m2_per_s2": self.m2,
            "m4_m2_per_s4": self.m4,
            "mean_period_s": self.mean_period,
            "zero_crossing_period_s": self.zero_crossing_period,
            "crest_period_s": crest,
            "hs_1000_cycles_m": self.most_probable_extreme(1000),
            "hs_1000_cycles_asymptote_m": self.most_probable_extreme(1000, exact=False),
        }


def _statistics_from_moments(spectrum, orders=(0, 1, 2), *, tolerance: float = 1e-13):
    values: dict[int, float] = {}
    sources: dict[str, object] = {}
    for order in orders:
        report = spectrum.moment(order, tolerance=tolerance)
        if not report["converged"]:
            raise RuntimeError(f"moment m{order} did not converge for {type(spectrum).__name__}")
        values[order] = float(report["moment"])
        sources[f"m{order}"] = report
    m4 = values.get(4)
    return ShortTermStatistics(
        significant_wave_height=spectrum.significant_wave_height,
        m0=values[0],
        m1=values[1],
        m2=values[2],
        m4=m4,
        moment_sources=sources,
    )


def short_term_statistics(spectrum, *, tolerance: float = 1e-13) -> ShortTermStatistics:
    """Moments m0/m1/m2 plus the Rayleigh package for ``spectrum``."""
    return _statistics_from_moments(spectrum, (0, 1, 2), tolerance=tolerance)


# ---------------------------------------------------------------------------
# time-domain realisation
# ---------------------------------------------------------------------------
def irregular_sea_realisation(
    spectrum,
    *,
    n_samples: int = 2**14,
    omega_max_factor: float = 6.0,
    seed: int = 0,
    sample_interval: float | None = None,
) -> dict[str, object]:
    """Realise one record of surface elevation from ``spectrum``.

    The record is built in the frequency domain (random phases, amplitude
    ``sqrt(2 S dw)``) and transformed with an inverse FFT, so the cost is
    ``O(n log n)`` rather than ``O(n_components * n_samples)``.  Returns the
    record plus the statistics recovered *from the record itself* (standard
    deviation, zero-up-crossing period), which is what the verification
    compares against the spectral prediction.
    """
    if isinstance(spectrum, torch.Tensor):
        raise TypeError("spectrum must be a PiersonMoskowitz or Jonswap instance")
    n_samples = int(n_samples)
    if n_samples < 16 or (n_samples & (n_samples - 1)) != 0:
        raise ValueError("n_samples must be a power of two and >= 16")
    _positive(omega_max_factor, "omega_max_factor")

    omega_p = spectrum.peak_angular_frequency
    omega_max = omega_max_factor * omega_p
    n_half = n_samples // 2
    delta_omega = omega_max / n_half
    record_length = 2.0 * math.pi / delta_omega
    dt = record_length / n_samples if sample_interval is None else _positive(
        sample_interval, "sample_interval"
    )

    frequencies = delta_omega * torch.arange(1, n_half + 1, dtype=torch.float64)
    density = spectrum.density(frequencies)
    amplitudes = torch.sqrt(2.0 * density * delta_omega)

    generator = torch.Generator().manual_seed(int(seed))
    phases = 2.0 * math.pi * torch.rand(n_half, generator=generator, dtype=torch.float64)
    # positive-frequency coefficients of eta = sum A_k cos(w_k t + phi_k);
    # the Nyquist bin is dropped so the irfft stays unambiguous.
    coefficients = torch.zeros(n_half + 1, dtype=torch.complex128)
    coefficients[1:n_half] = 0.5 * amplitudes[: n_half - 1] * torch.exp(1j * phases[: n_half - 1])
    elevation = torch.fft.irfft(coefficients, n=n_samples) * n_samples

    time = dt * torch.arange(n_samples, dtype=torch.float64)
    band = slice(0, n_half - 1)  # Nyquist bin dropped above
    band_m0 = float(torch.sum(density[band] * delta_omega))
    band_m2 = float(torch.sum(frequencies[band] ** 2 * density[band] * delta_omega))
    predicted_m0 = float(torch.sum(0.5 * amplitudes[band] ** 2))
    measured_std = float(torch.std(elevation, unbiased=False))
    # zero up-crossings of the realised record
    sign_previous = elevation[:-1]
    sign_next = elevation[1:]
    crossings = torch.nonzero((sign_previous <= 0) & (sign_next > 0)).flatten()
    if len(crossings) > 1:
        # linear interpolation of the crossing instants
        t0 = time[crossings]
        e0 = elevation[crossings]
        e1 = elevation[crossings + 1]
        instants = t0 - e0 * dt / (e1 - e0)
        intervals = torch.diff(instants)
        measured_tz = float(torch.mean(intervals))
    else:
        measured_tz = float("nan")
    heights = _zero_crossing_wave_heights(elevation, crossings)
    return {
        "time": time,
        "elevation": elevation,
        "wave_heights": heights,
        "sample_interval": dt,
        "record_length": dt * n_samples,
        "delta_omega": delta_omega,
        "omega_max": omega_max,
        "truncated_m0": predicted_m0,
        "band_m0": band_m0,
        "band_m2": band_m2,
        "band_zero_crossing_period": 2.0 * math.pi * math.sqrt(band_m0 / band_m2),
        "measured_std": measured_std,
        "measured_hs": 4.0 * measured_std,
        "measured_zero_crossing_period": measured_tz,
        "zero_upcrossings": int(len(crossings)),
        "max_wave_height": float(torch.max(heights)) if len(heights) else float("nan"),
        "mean_wave_height": float(torch.mean(heights)) if len(heights) else float("nan"),
        "seed": int(seed),
    }


def _zero_crossing_wave_heights(elevation: torch.Tensor, crossings: torch.Tensor) -> torch.Tensor:
    """Down-crossing-free wave heights: max minus min between up-crossings."""
    if len(crossings) < 2:
        return torch.zeros(0, dtype=torch.float64)
    segments = [
        elevation[crossings[i] : crossings[i + 1] + 1] for i in range(len(crossings) - 1)
    ]
    return torch.tensor(
        [float(torch.max(seg) - torch.min(seg)) for seg in segments], dtype=torch.float64
    )


def equivalent_design_wave(spectrum, *, height_ratio: float = 1.0, period: str = "peak"):
    """Regular-airywave equivalent of ``spectrum`` for Morison-type load cases.

    ``height_ratio`` selects the wave height as ``height_ratio * Hs``; ``period``
    is ``"peak"`` (Tp) or ``"zero_crossing"`` (Tz).  The result is an
    ``AiryWave`` so the existing ``morison_base_actions`` path can be driven
    from a spectral description.
    """
    from .marine_hydrodynamics import AiryWave

    if period not in {"peak", "zero_crossing"}:
        raise ValueError("period must be 'peak' or 'zero_crossing'")
    _positive(height_ratio, "height_ratio")
    stats = short_term_statistics(spectrum)
    chosen = spectrum.peak_period if period == "peak" else stats.zero_crossing_period
    return AiryWave(
        height=height_ratio * stats.hs,
        period=chosen,
        depth=_equivalent_depth(spectrum),
    )


def _equivalent_depth(spectrum) -> float:
    """Deep-water equivalent: 20 peak wavelengths, so ``tanh(kh)`` is 1 to 1e-17."""
    wp = spectrum.peak_angular_frequency
    k = wp**2 / spectrum.gravity
    return 20.0 * 2.0 * math.pi / k