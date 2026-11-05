"""Independent spectral-damage references for the rainflow + Miner fatigue path.

Reference
---------
* **Rainflow counting** -- ASTM E1049-85 (reapproved 2017), the four-point
  stack algorithm.  ``tensorfem.marine_fatigue_advanced.rainflow_cycles`` is
  the production implementation under test.
* **Narrow-band damage** -- Rayleigh closed form of the narrow-band
  approximation (Bendat 1964; Rice 1944 peak statistics; DNV-RP-C203 App. B
  narrow-band formula).  For a stationary narrow-band Gaussian stress process
  with RMS ``sigma`` the stress ranges are Rayleigh distributed with
  ``p(S) = S/(4 sigma^2) exp(-S^2/(8 sigma^2))`` and, with the two-parameter
  S-N curve ``N = C S^-m`` (``C = N_ref S_ref^m``), the expected damage per
  cycle is

      E[S^m] / C = (2 sqrt(2) sigma)^m Gamma(1 + m/2) / C .

  The narrow-band estimate is a *lower* bound on the rainflow damage of a
  finite-bandwidth process (Wirsching wide-band correction >= 1), so a
  measured ratio above one is expected and grows with the spectral width.
* **Paris law** -- constant-geometry integration of ``da/dN = C (Y ds sqrt(pi
  a))^m`` in closed form (see :func:`paris_power_law_closed_form`).

Why this module exists
----------------------
``marine_fatigue_advanced.run_advanced_fatigue_qualification`` verifies the
counting and damage primitives against a *single* triangular cycle and the
``m = 2`` (exponential) Paris branch.  Neither exercise touches the spectral
content of the algorithm: a single-range oracle cannot detect range-
distribution errors, half-cycle bookkeeping errors or sampling losses, and the
power-law Paris branch (``m != 2``) is left uncovered.  The references below
close that gap with literature/standalone mathematics instead of a second FE or
a second in-house counter.

Scope and known limitations (kept, not removed)
-----------------------------------------------
* The process generator is a finite sum of equal-amplitude cosines with
  uniform random phases: a Gaussian limit with a flat band spectrum of known
  moments.  It is *not* a class-rule wave load spectrum; it exists only to
  compare distributions.
* Rainflow counting of a *short* record depends on where the record starts
  because cycles spanning the record boundary can only be counted as residual
  half cycles.  Long records remove this (quantified by
  ``scripts/audit_marine_fatigue_spectrum.py``); it is a property of the ASTM
  convention, not of TensorFEM.
* Sampling a narrow-band process at too few points per cycle biases every
  extremum inward and therefore biases damage low by roughly
  ``(1 - pi^2/(6 N))^m``; see :func:`sampled_peak_damage_factor`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .marine_fatigue_advanced import (
    CrackGrowthBlock,
    ParisLaw,
    RainflowCycle,
    rainflow_cycles,
    spectrum_miner_damage,
)
from .marine_fatigue_qualification import SNLogLogCurve

__all__ = [
    "narrow_band_process",
    "rayleigh_range_pdf",
    "rayleigh_range_moment",
    "narrow_band_damage_per_cycle",
    "narrow_band_damage",
    "sampled_peak_damage_factor",
    "paris_power_law_closed_form",
    "narrow_band_damage_case",
    "rainflow_count_invariant",
    "full_cycle_multiset",
]


def narrow_band_process(
    *,
    central_frequency_hz: float = 1.0,
    relative_bandwidth: float = 0.05,
    components: int = 101,
    rms_stress_pa: float = 40.0e6,
    cycles: float = 20000.0,
    samples_per_cycle: float = 40.0,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Sample a band-limited Gaussian stress history of known spectral moments.

    ``relative_bandwidth`` is the width of the flat spectral band divided by
    its centre frequency.  The discrete moments ``m0..m4`` of the realised
    component set are returned so that the spectral width
    ``eps = sqrt(1 - m1^2/(m0 m2))`` and the mean upcrossing rate
    ``sqrt(m2/m0)`` are exact, not assumed.
    """
    if not (0.0 < relative_bandwidth < 1.0):
        raise ValueError("relative_bandwidth must lie in (0, 1)")
    if components < 8 or components % 2 == 0:
        raise ValueError("components must be an odd integer >= 9")
    if min(central_frequency_hz, rms_stress_pa, cycles, samples_per_cycle) <= 0.0:
        raise ValueError("frequency, RMS, cycles and samples per cycle must be > 0")
    rng = np.random.default_rng(seed)
    low = central_frequency_hz * (1.0 - 0.5 * relative_bandwidth)
    freqs = np.linspace(low, low + relative_bandwidth * central_frequency_hz, components)
    amplitude = math.sqrt(2.0 * rms_stress_pa**2 / components)
    phases = rng.uniform(0.0, 2.0 * math.pi, components)
    sample_rate = samples_per_cycle * float(freqs[-1])
    duration = cycles / central_frequency_hz
    count = int(round(duration * sample_rate))
    time = np.arange(count, dtype=np.float64) / sample_rate
    history = np.zeros(count, dtype=np.float64)
    for frequency, phase in zip(freqs, phases):
        history += amplitude * np.cos(2.0 * math.pi * frequency * time + phase)
    powers = np.full(components, 0.5 * amplitude**2)
    m0 = float(np.sum(powers))
    m1 = float(np.sum(powers * freqs))
    m2 = float(np.sum(powers * freqs**2))
    m4 = float(np.sum(powers * freqs**4))
    meta = {
        "central_frequency_hz": central_frequency_hz,
        "relative_bandwidth": relative_bandwidth,
        "components": float(components),
        "sample_rate_hz": sample_rate,
        "samples_per_cycle": samples_per_cycle,
        "duration_s": duration,
        "samples": float(count),
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "m4": m4,
        "rms_stress_pa": math.sqrt(m0),
        "spectral_width": math.sqrt(max(0.0, 1.0 - m1 * m1 / (m0 * m2))),
        "upcrossing_rate_hz": math.sqrt(m2 / m0),
        "peak_rate_hz": math.sqrt(m4 / m2),
        "seed": float(seed),
    }
    return history, meta


def rayleigh_range_pdf(stress_range_pa, rms_stress_pa: float):
    """Rayleigh density ``S/(4 sigma^2) exp(-S^2/(8 sigma^2))`` of narrow-band ranges.

    Stress ranges are twice the Rayleigh peaks of a narrow-band process with
    RMS ``sigma``; the density integrates to one and reproduces
    ``E[S^m] = (2 sqrt(2) sigma)^m Gamma(1 + m/2)``.
    """
    if rms_stress_pa <= 0.0:
        raise ValueError("rms_stress_pa must be > 0")
    ranges = np.asarray(stress_range_pa, dtype=np.float64)
    # S = 2 a, a Rayleigh with scale sigma = RMS, so p_S(S) = p_A(S/2)/2
    return ranges / (4.0 * rms_stress_pa**2) * np.exp(-(ranges**2) / (8.0 * rms_stress_pa**2))


def rayleigh_range_moment(rms_stress_pa: float, slope_m: float) -> float:
    """``E[S^m]`` for Rayleigh distributed narrow-band stress ranges."""
    if rms_stress_pa <= 0.0 or slope_m <= 0.0:
        raise ValueError("RMS stress and S-N slope must be > 0")
    return (2.0 * math.sqrt(2.0) * rms_stress_pa) ** slope_m * math.gamma(1.0 + 0.5 * slope_m)


def narrow_band_damage_per_cycle(rms_stress_pa: float, curve: SNLogLogCurve) -> float:
    """Narrow-band Rayleigh damage accumulated by one cycle."""
    if not isinstance(curve, SNLogLogCurve):
        raise TypeError("curve must be SNLogLogCurve")
    scale = curve.reference_cycles * curve.reference_stress_pa**curve.slope_m
    return rayleigh_range_moment(rms_stress_pa, curve.slope_m) / scale


def narrow_band_damage(rms_stress_pa: float, curve: SNLogLogCurve, cycles: float) -> float:
    """Narrow-band Rayleigh damage for ``cycles`` counted cycles."""
    if cycles <= 0.0:
        raise ValueError("cycles must be > 0")
    return narrow_band_damage_per_cycle(rms_stress_pa, curve) * cycles


def sampled_peak_damage_factor(samples_per_cycle: float, slope_m: float) -> float:
    """Leading-order damage deficit when a narrow-band peak is under-sampled.

    A discretely sampled extremum of a sinusoid is inward by ``1 - cos(delta)``
    with ``delta`` uniform on ``[0, pi/N]``, so the expected range is short by
    ``1 - pi^2/(6 N^2)`` and the damage by its ``m``-th power.  This is a
    *leading-order* estimate only; the measured curve is the evidence.
    """
    if samples_per_cycle <= 0.0 or slope_m <= 0.0:
        raise ValueError("samples per cycle and S-N slope must be > 0")
    range_factor = 1.0 - math.pi**2 / (6.0 * samples_per_cycle**2)
    return max(0.0, range_factor) ** slope_m


def paris_power_law_closed_form(
    initial_crack_m: float,
    blocks: Sequence[CrackGrowthBlock],
    law: ParisLaw,
) -> float:
    """Closed form of ``da/dN = C (Y ds sqrt(pi a))^m`` over constant blocks.

    Separating variables, ``da / a^(m/2) = C (Y ds sqrt(pi))^m dN``, integrates
    to ``a^(1-m/2) = a0^(1-m/2) + (1-m/2) k N`` with ``k = C (Y ds sqrt(pi))^m``;
    at ``m = 2`` the left side is ``ln a`` and the growth is exponential.  No
    finite crack is returned when the exponent ``m > 2`` drives the bracket to
    zero, which is the runaway growth TensorFEM reports as unbounded.
    """
    if not isinstance(law, ParisLaw):
        raise TypeError("law must be ParisLaw")
    if initial_crack_m <= 0.0:
        raise ValueError("initial_crack_m must be > 0")
    crack = float(initial_crack_m)
    power = 0.5 * law.exponent_m
    for block in blocks:
        if not isinstance(block, CrackGrowthBlock):
            raise TypeError("blocks must be CrackGrowthBlock values")
        rate = law.coefficient_c * (
            law.geometry_factor * block.stress_range_pa * math.sqrt(math.pi)
        ) ** law.exponent_m
        if abs(power - 1.0) < 1e-12:
            return_value = crack * math.exp(rate * block.cycles)
        else:
            transformed = crack ** (1.0 - power) + (1.0 - power) * rate * block.cycles
            if transformed <= 0.0:
                return math.inf
            return_value = transformed ** (1.0 / (1.0 - power))
        crack = return_value
    return crack


def rainflow_count_invariant(history: Sequence[float]) -> dict[str, float]:
    """Exact bookkeeping identity of the four-point algorithm.

    For a record that starts and ends at the same value the counted cycles and
    half cycles satisfy ``sum(count) = (n_rev - 1) / 2`` with ``n_rev`` the
    number of distilled reversals, i.e. every reversal interval is accounted
    for exactly once.  A violation means ranges are lost or double counted.
    """
    from .marine_fatigue_advanced import _reversals

    reversals = _reversals(history)
    cycles = rainflow_cycles(history)
    return {
        "reversals": float(len(reversals)),
        "counted": float(sum(row.count for row in cycles)),
        "expected": float(len(reversals) - 1) / 2.0,
        "defect": abs(sum(row.count for row in cycles) - (len(reversals) - 1) / 2.0),
        "closed": float(abs(reversals[0] - reversals[-1]) == 0.0),
    }


def full_cycle_multiset(history: Sequence[float]) -> tuple[tuple[float, float], ...]:
    """Sorted ``(range, mean)`` multiset of full cycles (``count == 1``)."""
    return tuple(
        sorted((row.range_pa, row.mean_pa) for row in rainflow_cycles(history) if row.count == 1.0)
    )


def narrow_band_damage_case(
    *,
    curve: SNLogLogCurve,
    relative_bandwidth: float = 0.05,
    cycles: float = 20000.0,
    samples_per_cycle: float = 40.0,
    rms_stress_pa: float = 40.0e6,
    seed: int = 0,
) -> dict[str, float]:
    """Compare production rainflow+Miner damage with the Rayleigh oracle.

    Two normalisations are returned because they answer different questions:

    * ``damage_ratio`` divides by the oracle taken over the cycles the
      production counter actually returned, isolating the *range distribution*
      (this is the quantity the Rayleigh law predicts cycle by cycle);
    * ``damage_rate_ratio`` divides by the oracle over the *spectral* cycle
      count ``nu_up T``, i.e. the physical damage rate.  The narrow-band law
      assumes those two rates coincide; for a finite spectral width the counter
      returns more (smaller) cycles than upcrossings, so ``damage_ratio`` can
      sit below one while the damage rate is at or above the narrow-band value.
    """
    history, meta = narrow_band_process(
        relative_bandwidth=relative_bandwidth,
        cycles=cycles,
        samples_per_cycle=samples_per_cycle,
        rms_stress_pa=rms_stress_pa,
        seed=seed,
    )
    counted: tuple[RainflowCycle, ...] = rainflow_cycles(history.tolist())
    cycles_counted = math.fsum(row.count for row in counted)
    damage = spectrum_miner_damage(counted, curve).damage
    oracle = narrow_band_damage(meta["rms_stress_pa"], curve, cycles_counted)
    spectral_cycles = meta["upcrossing_rate_hz"] * meta["duration_s"]
    rate_oracle = narrow_band_damage(meta["rms_stress_pa"], curve, spectral_cycles)
    return {
        "relative_bandwidth": meta["relative_bandwidth"],
        "spectral_width": meta["spectral_width"],
        "samples_per_cycle": meta["samples_per_cycle"],
        "samples": meta["samples"],
        "cycles_counted": cycles_counted,
        "count_rate_hz": cycles_counted / meta["duration_s"],
        "upcrossing_rate_hz": meta["upcrossing_rate_hz"],
        "count_rate_error": abs(
            cycles_counted / meta["duration_s"] / meta["upcrossing_rate_hz"] - 1.0
        ),
        "damage": damage,
        "damage_oracle": oracle,
        "damage_ratio": damage / oracle,
        "relative_error": abs(damage / oracle - 1.0),
        "spectral_cycles": spectral_cycles,
        "damage_rate_oracle": rate_oracle,
        "damage_rate_ratio": damage / rate_oracle,
        "sampling_factor_estimate": sampled_peak_damage_factor(
            meta["samples_per_cycle"], curve.slope_m
        ),
    }