"""Structure-acoustic radiation chain: FEM modes -> radiated power -> sigma.

This module wires the structural and acoustic halves together:

1. :func:`fem_modal_analysis` runs the Q4 Mindlin eigenproblem of
   :func:`tensorfem.plate_vibration.solve_plate_vibration` for a
   simply-supported or clamped rectangular plate and keeps the eigenpairs next
   to the analytical reference (Kirchhoff eigenvalues for simple support,
   Leissa's tabulated coefficients for the clamped square plate).

2. :meth:`FemModalAnalysis.surface_velocity_field` turns one eigenvector into a
   :class:`tensorfem.plate_acoustics.SurfaceVelocityField` by bilinear Q4
   interpolation on a Gauss grid (scaled to unit mean square velocity, so the
   radiation efficiency is independent of the eigenvector normalisation).

3. :func:`fem_radiation_check` / :func:`sine_mode_radiation_check` evaluate the
   radiation efficiency ``sigma = P_rad/(rho c S <v^2>)`` through the
   independent Rayleigh, far-field and (sine modes only) closed-form
   angular-spectrum routes of :mod:`tensorfem.plate_acoustics`, and report the
   gaps between them.

4. :func:`band_radiation_efficiency` builds the classical panel curve: for each
   band every resonating mode shares the same frequency (equal modal energy),
   which reproduces Maidanik's ``sigma ~ (f/f_c)^2`` rise below coincidence and
   the ``sigma -> 1`` plateau above it.  :func:`asymptotic_report` fits those two
   asymptotes and locates the knee, which is then compared with the analytical
   coincidence frequency ``f_c = c^2/(1.8 c_L h)``.

References
----------
* Maidanik, G. (1962). J. Acoust. Soc. Am. 34(6), 809-826.
* Leissa, A. W. (1969). *Vibration of Plates*, NASA SP-160 (CCCC table).
* Fahy, F. & Gardonio, P. (2007). *Sound and Structural Vibration*, 2nd ed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from .plate_acoustics import (
    PlateAcousticConfig,
    SurfaceVelocityField,
    _count,
    _gauss_grid,
    _positive,
    far_field_power,
    rayleigh_power,
    sine_mode_field,
    spectral_power_sine_mode,
)
from .plate_vibration import (
    BOUNDARY_CLAMPED,
    BOUNDARY_SIMPLE,
    PlateVibrationResult,
    solve_plate_vibration,
)

__all__ = [
    "BAND_RATIO_THIRD_OCTAVE",
    "HARMONIC_TIME_FACTOR",
    "CLAMPED_SQUARE_COEFFICIENTS",
    "BandRadiationResult",
    "FemModalAnalysis",
    "PlateRadiationResult",
    "asymptotic_report",
    "band_radiation_efficiency",
    "coincidence_knee_ratio",
    "coincidence_peak_ratio",
    "fem_modal_analysis",
    "fem_radiation_check",
    "fit_power_law_exponent",
    "frequency_sweep",
    "lattice_label",
    "mode_pairs_in_band",
    "plateau_level",
    "sine_mode_radiation_check",
    "simply_supported_mode_frequencies",
]

BAND_RATIO_THIRD_OCTAVE = 2.0 ** (1.0 / 3.0)

#: Leissa (1969, NASA SP-160) eigenvalue coefficients
#: ``lambda = omega * a^2 * sqrt(rho_s*h/D)`` of the clamped square plate, first
#: five modes (modes 2 and 3 are the degenerate (1,2)/(2,1) pair).
CLAMPED_SQUARE_COEFFICIENTS = (35.985, 73.394, 73.394, 108.216, 131.581)


def _relative_gap(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-30)


#: Harmonic time average: ``<v^2>(t) = |V|^2/2`` for a complex velocity amplitude
#: ``V``.  ``SurfaceVelocityField.mean_square_velocity`` is the spatial mean of
#: ``|V|^2`` (peak convention); the radiation efficiency is defined with the
#: time-and-space averaged mean square velocity, hence the factor 1/2.  This is
#: the convention in which a plane piston and a smooth plate mode both give
#: ``sigma -> 1`` far above coincidence (Leppington's asymptote).
HARMONIC_TIME_FACTOR = 0.5


def _efficiency(
    power: float, *, rho: float, c: float, area: float, msq: float
) -> float:
    """``sigma = P_rad / (rho c S <v^2>)`` with ``<v^2> = msq/2`` (harmonic)."""
    return power / (rho * c * area * msq * HARMONIC_TIME_FACTOR)


# ---------------------------------------------------------------------------
# Analytical references
# ---------------------------------------------------------------------------

def clamped_square_frequency(config: PlateAcousticConfig, index: int) -> float:
    """Clamped *square* plate eigenvalue from the Leissa table, in Hz."""
    index = _count("index", index)
    if index > len(CLAMPED_SQUARE_COEFFICIENTS):
        raise ValueError(
            f"clamped reference table holds {len(CLAMPED_SQUARE_COEFFICIENTS)} modes"
        )
    if abs(config.length - config.width) > 1e-12 * config.length:
        raise ValueError("the Leissa reference table applies to a square plate")
    return CLAMPED_SQUARE_COEFFICIENTS[index - 1] / (
        2.0 * math.pi * config.length ** 2
    ) * math.sqrt(config.bending_rigidity / config.mass_per_area)


def simply_supported_mode_frequencies(
    config: PlateAcousticConfig, count: int
) -> tuple[tuple[tuple[int, int], ...], torch.Tensor]:
    """First ``count`` simply-supported Kirchhoff eigenvalues and their ``(m, n)``.

    Modes are enumerated on the ``(m, n)`` lattice and sorted by frequency.
    Degenerate pairs (``f_12 == f_21`` on a square plate) keep lattice order,
    which is harmless because their frequencies agree to machine precision.
    """
    count = _count("count", count)
    limit = int(math.ceil(4.0 * math.sqrt(count) + 4.0))
    candidates = []
    for m in range(1, limit + 1):
        for n in range(1, limit + 1):
            candidates.append((config.mode_frequency(m, n), m, n))
    candidates.sort()
    selected = candidates[:count]
    frequencies = torch.tensor([item[0] for item in selected], dtype=torch.float64)
    return tuple((item[1], item[2]) for item in selected), frequencies


# ---------------------------------------------------------------------------
# FEM modal analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FemModalAnalysis:
    """FEM eigenpairs of a plate plus the analytical reference frequencies."""

    config: PlateAcousticConfig
    boundary: str
    element_counts: tuple[int, int]
    frequencies_hz: torch.Tensor
    mode_shapes: torch.Tensor
    nodes: torch.Tensor
    elements: torch.Tensor
    reference_frequencies_hz: Optional[torch.Tensor] = None

    @property
    def mode_count(self) -> int:
        return int(self.frequencies_hz.numel())

    def relative_frequency_error(self) -> Optional[torch.Tensor]:
        """Per-mode ``|f_fem - f_ref| / f_ref`` (None without a reference)."""
        if self.reference_frequencies_hz is None:
            return None
        return (self.frequencies_hz - self.reference_frequencies_hz).abs() / (
            self.reference_frequencies_hz.abs()
        )

    def transverse_mode(self, mode_index: int) -> torch.Tensor:
        """Transverse (w) degrees of freedom of one mode on the node grid."""
        mode_index = _count("mode_index", mode_index)
        if mode_index > self.mode_count:
            raise ValueError(
                f"mode_index exceeds the {self.mode_count} computed modes"
            )
        nx, ny = self.element_counts
        w = self.mode_shapes[0::3, mode_index - 1]
        sign = 1.0 if float(w.sum()) >= 0.0 else -1.0
        return (sign * w).reshape(ny + 1, nx + 1)

    def surface_velocity_field(
        self, mode_index: int, *, points: int = 48, rms: Optional[float] = 1.0
    ) -> SurfaceVelocityField:
        """Bilinear Q4 interpolation of one FEM mode on a Gauss grid."""
        points = _count("points", points, minimum=2)
        nx, ny = self.element_counts
        x, y, wx, wy = _gauss_grid(self.config.length, self.config.width, points)
        w = self.transverse_mode(mode_index)          # (ny+1, nx+1)
        hx = self.config.length / nx
        hy = self.config.width / ny
        ix = torch.clamp((x / hx).floor().long(), 0, nx - 1)
        iy = torch.clamp((y / hy).floor().long(), 0, ny - 1)
        xi = (x / hx - ix.to(x.dtype))[None, :]
        eta = (y / hy - iy.to(y.dtype))[:, None]
        velocity = (
            (1.0 - xi) * (1.0 - eta) * w[iy][:, ix]
            + xi * (1.0 - eta) * w[iy][:, ix + 1]
            + xi * eta * w[iy + 1][:, ix + 1]
            + (1.0 - xi) * eta * w[iy + 1][:, ix]
        )
        surface = SurfaceVelocityField(
            x=x, y=y, velocity=velocity, weight_x=wx, weight_y=wy,
            label=f"fem[{self.boundary}](mode {mode_index})",
        )
        return surface if rms is None else surface.normalized(rms, label=surface.label)


def fem_modal_analysis(
    config: PlateAcousticConfig,
    *,
    boundary: str = BOUNDARY_SIMPLE,
    elements: int = 16,
    modes: int = 5,
    with_reference: bool = True,
) -> FemModalAnalysis:
    """Run the Q4 Mindlin eigenproblem and attach the analytical reference.

    ``elements`` is the number of Q4 elements per side of the square mesh (the
    plate aspect ratio is carried by the element sides ``length/elements`` x
    ``width/elements``).  The spurious-mechanism tolerance is disabled (``0.0``)
    because on a thin softly supported plate the lowest elastic eigenvalue can
    fall inside the default ``1e-9`` relative band of the mechanism detector.
    """
    if boundary not in (BOUNDARY_SIMPLE, BOUNDARY_CLAMPED):
        raise ValueError(
            f"boundary must be {BOUNDARY_SIMPLE!r} or {BOUNDARY_CLAMPED!r}"
        )
    elements = _count("elements", elements, minimum=2)
    modes = _count("modes", modes)
    result: PlateVibrationResult = solve_plate_vibration(
        elements, elements,
        length=config.length, width=config.width,
        young=config.young, poisson=config.poisson,
        thickness=config.thickness, density=config.density_plate,
        boundary=boundary, modes=modes, rigid_body_relative_tolerance=0.0,
    )
    reference = None
    if with_reference:
        if boundary == BOUNDARY_SIMPLE:
            _, reference = simply_supported_mode_frequencies(config, modes)
        elif (
            abs(config.length - config.width) <= 1e-12 * config.length
            and modes <= len(CLAMPED_SQUARE_COEFFICIENTS)
        ):
            reference = torch.tensor(
                [clamped_square_frequency(config, i) for i in range(1, modes + 1)],
                dtype=torch.float64,
            )
    return FemModalAnalysis(
        config=config,
        boundary=boundary,
        element_counts=(elements, elements),
        frequencies_hz=result.frequencies_hz.detach().clone(),
        mode_shapes=result.modes.detach().clone(),
        nodes=result.nodes,
        elements=result.elements,
        reference_frequencies_hz=reference,
    )


# ---------------------------------------------------------------------------
# Radiation efficiency of one mode
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlateRadiationResult:
    """Radiation efficiency of one plate mode from the acoustic cross-check."""

    frequency: float
    mode: tuple[int, int]
    mean_square_velocity: float
    sigma_rayleigh: float
    sigma_far_field: float
    sigma_spectral: Optional[float] = None
    ka: float = 0.0
    label: str = field(default="")

    @property
    def sigma(self) -> float:
        """Best available estimate: the Rayleigh surface integral."""
        return self.sigma_rayleigh

    @property
    def far_field_gap(self) -> float:
        return _relative_gap(self.sigma_far_field, self.sigma_rayleigh)

    @property
    def spectral_gap(self) -> Optional[float]:
        if self.sigma_spectral is None:
            return None
        return _relative_gap(self.sigma_spectral, self.sigma_rayleigh)


def lattice_label(analysis: FemModalAnalysis, mode_index: int) -> tuple[int, int]:
    """``(m, n)`` implied by the frequency rank (analytical lattice ordering)."""
    mode_index = _count("mode_index", mode_index)
    if analysis.reference_frequencies_hz is None:
        return (mode_index, 1)
    lattice, _ = simply_supported_mode_frequencies(analysis.config, analysis.mode_count)
    return lattice[mode_index - 1]


def fem_radiation_check(
    analysis: FemModalAnalysis,
    mode_index: int,
    *,
    frequency: Optional[float] = None,
    points: int = 40,
    n_theta: int = 48,
    n_phi: int = 128,
    chunk: int = 64,
) -> PlateRadiationResult:
    """Radiation efficiency of one FEM mode through the Rayleigh and far-field routes.

    The velocity field is the bilinearly interpolated FEM eigenvector scaled to
    unit mean square velocity, so ``sigma`` does not depend on the arbitrary
    eigenvector normalisation.  The closed-form angular-spectrum route only
    exists for the sine mode and is therefore ``None`` here.
    """
    config = analysis.config
    mode_index = _count("mode_index", mode_index)
    if mode_index > analysis.mode_count:
        raise ValueError(
            f"mode_index exceeds the {analysis.mode_count} computed modes"
        )
    frequency = (
        float(analysis.frequencies_hz[mode_index - 1])
        if frequency is None else _positive("frequency", frequency)
    )
    surface = analysis.surface_velocity_field(mode_index, points=points)
    msq = surface.mean_square_velocity
    power_a = rayleigh_power(
        surface, frequency=frequency, rho=config.rho_fluid, c=config.sound_speed,
        chunk=chunk,
    )
    power_b = far_field_power(
        surface, frequency=frequency, rho=config.rho_fluid, c=config.sound_speed,
        n_theta=n_theta, n_phi=n_phi,
    )
    return PlateRadiationResult(
        frequency=frequency,
        mode=lattice_label(analysis, mode_index),
        mean_square_velocity=msq,
        sigma_rayleigh=_efficiency(
            power_a, rho=config.rho_fluid, c=config.sound_speed,
            area=config.area, msq=msq,
        ),
        sigma_far_field=_efficiency(
            power_b, rho=config.rho_fluid, c=config.sound_speed,
            area=config.area, msq=msq,
        ),
        ka=config.acoustic_wavenumber(frequency) * config.length,
        label=surface.label,
    )


def sine_mode_radiation_check(
    config: PlateAcousticConfig,
    m: int,
    n: int,
    *,
    frequency: Optional[float] = None,
    points: int = 48,
    n_theta: int = 48,
    n_phi: int = 256,
    chunk: int = 64,
    with_spectral: bool = True,
) -> PlateRadiationResult:
    """Radiation efficiency of a sine mode through all three routes."""
    m = _count("m", m)
    n = _count("n", n)
    frequency = (
        config.mode_frequency(m, n) if frequency is None
        else _positive("frequency", frequency)
    )
    surface = sine_mode_field(config, m, n, points=points)
    msq = surface.mean_square_velocity
    power_a = rayleigh_power(
        surface, frequency=frequency, rho=config.rho_fluid, c=config.sound_speed,
        chunk=chunk,
    )
    power_b = far_field_power(
        surface, frequency=frequency, rho=config.rho_fluid, c=config.sound_speed,
        n_theta=n_theta, n_phi=n_phi,
    )
    spectral = (
        spectral_power_sine_mode(
            config, m, n, frequency=frequency,
            n_theta=max(n_theta, 32), n_phi=max(n_phi, 128),
        )
        if with_spectral else None
    )
    return PlateRadiationResult(
        frequency=frequency,
        mode=(m, n),
        mean_square_velocity=msq,
        sigma_rayleigh=_efficiency(
            power_a, rho=config.rho_fluid, c=config.sound_speed,
            area=config.area, msq=msq,
        ),
        sigma_far_field=_efficiency(
            power_b, rho=config.rho_fluid, c=config.sound_speed,
            area=config.area, msq=msq,
        ),
        sigma_spectral=None if spectral is None else _efficiency(
            spectral, rho=config.rho_fluid, c=config.sound_speed,
            area=config.area, msq=msq,
        ),
        ka=config.acoustic_wavenumber(frequency) * config.length,
        label=surface.label,
    )


# ---------------------------------------------------------------------------
# Frequency-band modal average and coincidence asymptotics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandRadiationResult:
    """Equal-modal-energy average of ``sigma`` over the modes of one band."""

    frequency: float
    ratio: float                 # f / f_c
    modal_count: int
    sigma: float
    sigma_min: float
    sigma_max: float
    modes: tuple[tuple[int, int], ...] = ()

    @property
    def sigma_over_ratio_squared(self) -> float:
        return self.sigma / self.ratio ** 2


def mode_pairs_in_band(
    config: PlateAcousticConfig,
    frequency: float,
    *,
    band_ratio: float = BAND_RATIO_THIRD_OCTAVE,
) -> tuple[tuple[int, int], ...]:
    """Simply-supported modes whose resonance lies inside the band around ``frequency``."""
    frequency = _positive("frequency", frequency)
    band_ratio = _positive("band_ratio", band_ratio)
    if band_ratio < 1.0:
        raise ValueError("band_ratio is the upper/lower edge ratio and must be >= 1")
    low = frequency / band_ratio
    high = frequency * band_ratio
    limit_m = int(
        math.ceil(config.length / math.pi * config.bending_wavenumber(high))
    ) + 1
    limit_n = int(
        math.ceil(config.width / math.pi * config.bending_wavenumber(high))
    ) + 1
    pairs = []
    for m in range(1, limit_m + 1):
        for n in range(1, limit_n + 1):
            if low <= config.mode_frequency(m, n) <= high:
                pairs.append((m, n))
    return tuple(pairs)


def band_radiation_efficiency(
    config: PlateAcousticConfig,
    frequency: float,
    *,
    band_ratio: float = BAND_RATIO_THIRD_OCTAVE,
    n_theta: int = 48,
    n_phi: int = 128,
    low_frequency_exact: bool = False,
    points: int = 40,
) -> BandRadiationResult:
    """Band average of the modal radiation efficiencies at ``frequency``.

    Every mode resonating inside the band is driven at the *same* frequency
    ``frequency`` (equal modal energy, the standard Maidanik / reverberant-field
    assumption) and its efficiency is evaluated with the closed-form angular
    spectrum, which needs no surface quadrature and therefore stays cheap for
    high-order modes.  ``low_frequency_exact`` switches the deep-subsonic bands
    to the 4-D Rayleigh integral.
    """
    frequency = _positive("frequency", frequency)
    pairs = mode_pairs_in_band(config, frequency, band_ratio=band_ratio)
    if not pairs:
        raise ValueError(
            f"no simply-supported mode resonates inside the band at {frequency} Hz"
        )
    sigmas = []
    for m, n in pairs:
        if low_frequency_exact:
            check = sine_mode_radiation_check(
                config, m, n, frequency=frequency,
                points=points, n_theta=n_theta, n_phi=n_phi, with_spectral=False,
            )
            sigmas.append(check.sigma_rayleigh)
        else:
            power = spectral_power_sine_mode(
                config, m, n, frequency=frequency, n_theta=n_theta, n_phi=n_phi
            )
            sigmas.append(
                _efficiency(
                    power, rho=config.rho_fluid, c=config.sound_speed,
                    area=config.area, msq=0.25,
                )
            )
    values = torch.tensor(sigmas, dtype=torch.float64)
    return BandRadiationResult(
        frequency=frequency,
        ratio=frequency / config.critical_frequency,
        modal_count=len(pairs),
        sigma=float(values.mean()),
        sigma_min=float(values.min()),
        sigma_max=float(values.max()),
        modes=pairs,
    )


def frequency_sweep(
    config: PlateAcousticConfig,
    frequencies: Sequence[float],
    *,
    band_ratio: float = BAND_RATIO_THIRD_OCTAVE,
    **kwargs,
) -> list[BandRadiationResult]:
    """Band-average radiation efficiency across a list of frequencies."""
    return [
        band_radiation_efficiency(config, value, band_ratio=band_ratio, **kwargs)
        for value in frequencies
    ]


def fit_power_law_exponent(
    ratios: Sequence[float],
    sigmas: Sequence[float],
    *,
    low: float = 0.0,
    high: float = math.inf,
) -> tuple[float, float]:
    """Least-squares ``log sigma = log A + p log (f/f_c)`` on ``low <= ratio <= high``.

    Returns ``(p, A)``.  Samples outside the window and non-positive samples are
    ignored; at least two usable samples are required.
    """
    xs, ys = [], []
    for ratio, sigma in zip(ratios, sigmas):
        if not (low <= ratio <= high) or sigma <= 0.0 or ratio <= 0.0:
            continue
        xs.append(math.log(ratio))
        ys.append(math.log(sigma))
    if len(xs) < 2:
        raise ValueError("need at least two positive samples inside the fit window")
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    slope = float(((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum())
    intercept = float(y_mean - slope * x_mean)
    return slope, math.exp(intercept)


def plateau_level(
    ratios: Sequence[float], sigmas: Sequence[float], *, above: float = 4.0
) -> float:
    """Median ``sigma`` over the samples with ``f/f_c >= above``."""
    values = [sigma for ratio, sigma in zip(ratios, sigmas) if ratio >= above]
    if not values:
        raise ValueError(f"no sample with f/f_c >= {above}")
    return float(torch.tensor(values, dtype=torch.float64).median())


def coincidence_knee_ratio(
    ratios: Sequence[float],
    sigmas: Sequence[float],
    *,
    low: float = 0.0,
    high: float = 0.6,
    plateau_above: float = 4.0,
) -> float:
    """Knee of the ``sigma(f/f_c)`` curve: low-frequency power law meets the plateau.

    The classical panel curve rises like ``(f/f_c)^p`` well below coincidence and
    flattens to ``sigma -> 1`` above it.  The intersection of the two asymptotes
    is the machine-detectable stand-in for the knee/peak position, and it is the
    quantity the coincidence-frequency check compares against
    ``c^2/(1.8 c_L h)``.
    """
    exponent, prefactor = fit_power_law_exponent(ratios, sigmas, low=low, high=high)
    if exponent <= 0.0:
        raise ValueError(
            "low-frequency fit has a non-positive exponent; check the fit window"
        )
    plateau = plateau_level(ratios, sigmas, above=plateau_above)
    return (plateau / prefactor) ** (1.0 / exponent)



def coincidence_peak_ratio(
    ratios: Sequence[float],
    sigmas: Sequence[float],
) -> tuple[float, float]:
    """Locate the maximum of the ``sigma(f/f_c)`` curve (parabolic refinement).

    Returns ``(ratio_peak, sigma_peak)``.  The discrete band average smears the
    coincidence singularity of the modal radiation efficiency, so the peak is
    refined by fitting a parabola to the three samples around the maximum in
    ``log f`` - ``log sigma`` coordinates.
    """
    if len(ratios) != len(sigmas):
        raise ValueError("ratios and sigmas must have the same length")
    if len(ratios) < 3:
        raise ValueError("need at least three samples to refine a peak")
    values = torch.tensor(list(sigmas), dtype=torch.float64)
    index = int(torch.argmax(values).item())
    if index == 0 or index == len(ratios) - 1:
        return float(ratios[index]), float(sigmas[index])
    x = torch.log(torch.tensor(list(ratios)[index - 1:index + 2], dtype=torch.float64))
    y = values[index - 1:index + 2]
    vandermonde = torch.stack((x * x, x, torch.ones_like(x)), dim=1)
    a, b, c = torch.linalg.solve(vandermonde, y).tolist()
    if a >= 0.0:
        return float(ratios[index]), float(sigmas[index])
    x_peak = -b / (2.0 * a)
    return math.exp(x_peak), float(a * x_peak ** 2 + b * x_peak + c)


def asymptotic_report(
    config: PlateAcousticConfig,
    results: Sequence[BandRadiationResult],
    *,
    low_window: tuple[float, float] = (0.1, 0.5),
    plateau_above: float = 4.0,
) -> dict:
    """Summarise the ``sigma(f/f_c)`` asymptotics and the coincidence knee.

    Returns the fitted low-frequency exponent (classical value 2), the fitted
    prefactor, the high-frequency plateau (classical value 1), the knee
    frequency and the relative errors against the analytical ``f_c`` - the
    numbers the acceptance criteria quote.
    """
    ratios = [item.ratio for item in results]
    sigmas = [item.sigma for item in results]
    exponent, prefactor = fit_power_law_exponent(
        ratios, sigmas, low=low_window[0], high=low_window[1]
    )
    plateau = plateau_level(ratios, sigmas, above=plateau_above)
    knee = coincidence_knee_ratio(
        ratios, sigmas, low=low_window[0], high=low_window[1],
        plateau_above=plateau_above,
    )
    knee_frequency = knee * config.critical_frequency
    return {
        "critical_frequency": config.critical_frequency,
        "coincidence_frequency_exact": config.coincidence_frequency_exact,
        "low_frequency_exponent": exponent,
        "low_frequency_exponent_error": abs(exponent - 2.0) / 2.0,
        "low_frequency_prefactor": prefactor,
        "high_frequency_plateau": plateau,
        "high_frequency_plateau_error": abs(plateau - 1.0),
        "knee_ratio": knee,
        "knee_frequency": knee_frequency,
        "knee_frequency_error": abs(knee_frequency - config.critical_frequency)
        / config.critical_frequency,
        "samples": len(results),
        "air_rho": config.rho_fluid,
        "air_c": config.sound_speed,
    }