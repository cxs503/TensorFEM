"""Auditable imperfect-plated-strip plasticity qualification model.

This is a displacement-controlled, one-mode *strip reduction*, not a shell
finite element.  Fibres across the plate width use an exact uniaxial
bilinear return map.  Initial sine imperfection and a self-equilibrated
residual-stress field couple membrane shortening to the local buckle mode.
The deliberately small model is intended for verification of state
evolution and gates, not for ship scantling or collapse prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _count(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class StripState:
    step: int
    shortening: float
    amplitude: float
    membrane_force: float
    maximum_yield_residual: float
    equilibrium_residual: float
    external_work: float
    stored_energy: float
    plastic_dissipation: float
    energy_residual: float
    yielded_fraction: float


def elastic_strip_oracle(*, shortening: float, young: float = 100.0,
                         imperfection: float = 0.02,
                         bending_stiffness: float = 2.0) -> tuple[float, float]:
    """Closed-form-continuum cubic oracle for the entirely elastic strip."""
    shortening = _positive("shortening", shortening)
    e = _positive("young", young); w0 = _positive("imperfection", imperfection)
    kb = _positive("bending_stiffness", bending_stiffness)
    # mean(0.5*pi^2*cos^2(pi*x)) = pi^2/4
    g = math.pi**2 / 4.0
    def residual(q):
        force = e * (shortening + g * (q*q - w0*w0))
        return kb * (q - w0) - force * q
    lo, last = w0, residual(w0)
    bracket = None
    for q in np.linspace(w0, w0 + 3.0, 10001):
        value = residual(float(q))
        if value * last <= 0.0 and q > w0 + 1e-12:
            bracket = (lo, float(q)); break
        lo, last = float(q), value
    if bracket is None:
        raise RuntimeError("elastic oracle has no connected positive root")
    lo, hi = bracket
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if residual(lo) * residual(mid) <= 0.0: hi = mid
        else: lo = mid
    q = 0.5 * (lo + hi)
    return q, e * (shortening + g * (q*q - w0*w0))


def _return_map(total_strain, plastic_old, alpha_old, residual, young, yield_stress, hardening):
    trial = residual + young * (total_strain - plastic_old)
    radius = yield_stress + hardening * alpha_old
    f = np.abs(trial) - radius
    active = f > 0.0
    increment = np.where(active, f / (young + hardening), 0.0)
    direction = np.sign(trial)
    plastic = plastic_old + increment * direction
    alpha = alpha_old + increment
    stress = residual + young * (total_strain - plastic)
    return stress, plastic, alpha, active


def imperfect_plastic_strip_path(
    *, maximum_shortening: float, steps: int, fibres: int,
    young: float = 100.0, yield_stress: float = 1.0, hardening: float = 2.0,
    imperfection: float = 0.02, residual_stress: float = 0.25,
    bending_stiffness: float = 2.0,
) -> list[StripState]:
    """Advance the reduced strip and return every committed audit state.

    Quantities are consistently nondimensional. Compression, shortening and
    membrane force are positive.  Midpoint fibres exactly self-equilibrate
    the prescribed ``residual_stress*cos(2*pi*x)`` field.
    """
    end = _positive("maximum_shortening", maximum_shortening)
    ns = _count("steps", steps, 4)
    nf = _count("fibres", fibres, 8)
    if nf % 2:
        raise ValueError("fibres must be even for residual-stress equilibrium")
    e = _positive("young", young); sy = _positive("yield_stress", yield_stress)
    h = _positive("hardening", hardening); w0 = _positive("imperfection", imperfection)
    sr = _positive("residual_stress", residual_stress); kb = _positive("bending_stiffness", bending_stiffness)
    if sr >= sy:
        raise ValueError("residual_stress must be below yield_stress")
    x = (np.arange(nf, dtype=float) + 0.5) / nf
    weight = 1.0 / nf
    residual = sr * np.cos(2.0 * math.pi * x)
    if abs(float(np.sum(residual) * weight)) > 1e-13:
        raise AssertionError("residual stress is not self-equilibrated")
    shape = 0.5 * math.pi**2 * np.cos(math.pi * x) ** 2
    plastic = np.zeros(nf); alpha = np.zeros(nf)
    amplitude = w0; old_strain = np.zeros(nf); old_stress = residual.copy()
    stored0 = float(np.sum(residual**2 / (2.0 * e)) * weight)
    work = 0.0; dissipation = 0.0
    states: list[StripState] = []
    for step in range(ns + 1):
        shortening = end * step / ns
        if step:
            # Frozen committed variables make this a proper incremental solve.
            p0, a0 = plastic.copy(), alpha.copy()
            lo, hi = w0, max(amplitude * 1.2, w0 + 0.05)
            def evaluate(q):
                strain = shortening + shape * (q*q - w0*w0)
                values = _return_map(strain, p0, a0, residual, e, sy, h)
                force = float(np.sum(values[0]) * weight)
                eq = kb * (q - w0) - force * q
                return eq, strain, values, force
            # Select the first stable positive root connected to the prior state.
            last_q, last_f = w0, evaluate(w0)[0]
            bracket = None
            for q in np.linspace(w0, hi + 3.0, 2001):
                f = evaluate(float(q))[0]
                if f * last_f <= 0.0 and q > w0 + 1e-12:
                    bracket = (last_q, float(q)); break
                last_q, last_f = float(q), f
            if bracket is None:
                raise RuntimeError("could not bracket strip equilibrium")
            lo, hi = bracket
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if evaluate(lo)[0] * evaluate(mid)[0] <= 0.0: hi = mid
                else: lo = mid
            amplitude = 0.5 * (lo + hi)
            eq, strain, (stress, plastic, alpha, active), force = evaluate(amplitude)
            work += float(np.sum(0.5 * (old_stress + stress) * (strain - old_strain)) * weight)
            dissipation += float(np.sum(sy * (alpha - a0)) * weight)
            old_strain, old_stress = strain, stress
        else:
            stress = residual.copy(); force = 0.0; active = np.zeros(nf, dtype=bool); eq = 0.0
        elastic = stress / e
        stored = float(np.sum(0.5 * e * elastic**2 + 0.5 * h * alpha**2) * weight) \
                 + 0.5 * kb * (amplitude - w0) ** 2
        yield_residual = float(np.max(np.maximum(np.abs(stress) - (sy + h * alpha), 0.0)))
        energy_residual = abs(work - (stored - stored0) - dissipation)
        states.append(StripState(step, shortening, amplitude, force, yield_residual,
                                 abs(eq), work, stored, dissipation, energy_residual,
                                 float(np.mean(active))))
    return states


def run_imperfect_strip_qualification() -> dict[str, object]:
    """Run discretisation convergence and physical-invariant gates."""
    resolutions = ((80, 32), (160, 64), (320, 128))
    paths = [imperfect_plastic_strip_path(maximum_shortening=0.025, steps=s, fibres=f)
             for s, f in resolutions]
    oracle = imperfect_plastic_strip_path(maximum_shortening=0.025, steps=1280, fibres=512)
    elastic = imperfect_plastic_strip_path(maximum_shortening=0.002, steps=80, fibres=128,
                                            yield_stress=10.0)
    elastic_oracle = elastic_strip_oracle(shortening=0.002)
    elastic_error = max(abs(elastic[-1].amplitude / elastic_oracle[0] - 1.0),
                        abs(elastic[-1].membrane_force / elastic_oracle[1] - 1.0))
    reference = np.array([oracle[-1].amplitude, oracle[-1].membrane_force,
                          oracle[-1].external_work])
    errors = []
    for path in paths:
        value = np.array([path[-1].amplitude, path[-1].membrane_force,
                          path[-1].external_work])
        errors.append(float(np.max(np.abs(value / reference - 1.0))))
    fine = paths[-1]
    max_equilibrium = max(p.equilibrium_residual for p in fine)
    max_yield = max(p.maximum_yield_residual for p in fine)
    energy_fraction = fine[-1].energy_residual / fine[-1].external_work
    passed = (errors[2] < errors[1] < errors[0] and errors[2] < 0.03 and elastic_error < 1e-10 and
              max_equilibrium < 1e-10 and max_yield < 1e-12 and energy_fraction < 0.03)
    if not passed:
        raise AssertionError("imperfect plastic strip qualification failed")
    return {"schema": "tensorfem.imperfect-plastic-strip/1.0", "tolerance": 0.03,
            "scope": "one-mode midpoint-fibre plated-strip reduction; not a shell FE; no TensorLBM",
            "resolutions": resolutions, "response_errors": errors,
            "oracle": {"steps": 1280, "fibres": 512, "final_response": reference.tolist()},
            "analytic_elastic_oracle_error": elastic_error,
            "checks": {"maximum_equilibrium_residual": max_equilibrium,
                       "maximum_yield_residual": max_yield,
                       "energy_residual_fraction": energy_fraction,
                       "residual_stress_self_equilibrated": True},
            "final": fine[-1], "passed": True}
