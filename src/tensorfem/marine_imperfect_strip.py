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
import torch


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
    for q in torch.linspace(w0, w0 + 3.0, 10001, dtype=torch.float64).tolist():
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
    f = torch.abs(trial) - radius
    active = f > 0.0
    increment = torch.where(active, f / (young + hardening), torch.zeros_like(f))
    direction = torch.sign(trial)
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
    x = (torch.arange(nf, dtype=torch.float64) + 0.5) / nf
    weight = 1.0 / nf
    residual = sr * torch.cos(2.0 * math.pi * x)
    if abs(float(torch.sum(residual) * weight)) > 1e-13:
        raise AssertionError("residual stress is not self-equilibrated")
    shape = 0.5 * math.pi**2 * torch.cos(math.pi * x) ** 2
    plastic = torch.zeros(nf,dtype=torch.float64); alpha = torch.zeros(nf,dtype=torch.float64)
    amplitude = w0; old_strain = torch.zeros(nf,dtype=torch.float64); old_stress = residual.clone()
    stored0 = float(torch.sum(residual**2 / (2.0 * e)) * weight)
    work = 0.0; dissipation = 0.0
    states: list[StripState] = []
    for step in range(ns + 1):
        shortening = end * step / ns
        if step:
            # Frozen committed variables make this a proper incremental solve.
            p0, a0 = plastic.clone(), alpha.clone()
            lo, hi = w0, max(amplitude * 1.2, w0 + 0.05)
            def evaluate(q):
                strain = shortening + shape * (q*q - w0*w0)
                values = _return_map(strain, p0, a0, residual, e, sy, h)
                force = float(torch.sum(values[0]) * weight)
                eq = kb * (q - w0) - force * q
                return eq, strain, values, force
            # Select the first stable positive root connected to the prior state.
            last_q, last_f = w0, evaluate(w0)[0]
            bracket = None
            for q in torch.linspace(w0, hi + 3.0, 2001,dtype=torch.float64).tolist():
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
            work += float(torch.sum(0.5 * (old_stress + stress) * (strain - old_strain)) * weight)
            dissipation += float(torch.sum(sy * (alpha - a0)) * weight)
            old_strain, old_stress = strain, stress
        else:
            stress = residual.clone(); force = 0.0; active = torch.zeros(nf,dtype=torch.bool); eq = 0.0
        elastic = stress / e
        stored = float(torch.sum(0.5 * e * elastic**2 + 0.5 * h * alpha**2) * weight) \
                 + 0.5 * kb * (amplitude - w0) ** 2
        yield_residual = float(torch.max(torch.clamp(torch.abs(stress) - (sy + h * alpha),min=0.0)))
        energy_residual = abs(work - (stored - stored0) - dissipation)
        states.append(StripState(step, shortening, amplitude, force, yield_residual,
                                 abs(eq), work, stored, dissipation, energy_residual,
                                 float(torch.mean(active.to(torch.float64)))))
    return states


def imperfect_plastic_strip_field_path(
    initial_fields, *, maximum_shortening: float, steps: int,
    young: float = 100.0, yield_stress: float = 1.0, hardening: float = 2.0,
    bending_stiffness: float = 2.0,
) -> list[StripState]:
    """Solve the strip using every imported nodal fabrication-field value.

    The nodal transverse imperfection defines a piecewise-linear profile. Its
    discrete gradient drives the geometric membrane strain, while nodal axial
    residual stress enters the return map without modal fitting or averaging.
    This is still the reduced strip model, not a Shell4 field projection.
    """
    from .initial_fields import imperfect_strip_inputs
    end=_positive("maximum_shortening",maximum_shortening);ns=_count("steps",steps,4)
    e=_positive("young",young);sy=_positive("yield_stress",yield_stress)
    h=_positive("hardening",hardening);kb=_positive("bending_stiffness",bending_stiffness)
    imported_w,imported_s=imperfect_strip_inputs(initial_fields)
    profile=imported_w.detach().cpu().to(torch.float64)
    residual=imported_s.detach().cpu().to(torch.float64)
    nf=len(profile)
    if nf<8: raise ValueError("field-driven strip requires at least eight nodes")
    if float(torch.max(torch.abs(profile)))<=0: raise ValueError("imperfection profile must be nonzero")
    if float(torch.max(torch.abs(residual)))>=sy: raise ValueError("residual stress must be below yield_stress")
    if abs(float(torch.mean(residual)))>1e-10*max(float(torch.mean(torch.abs(residual))),1.):
        raise ValueError("axial residual stress must be self-equilibrated")
    dx=1.0/nf
    # Periodic centered difference prevents endpoint weighting from creating a
    # false resultant for fabrication profiles sampled around a strip cell.
    gradient=(torch.roll(profile,-1)-torch.roll(profile,1))/(2.0*dx)
    shape=0.5*gradient**2
    modal_norm=float(torch.mean(profile**2))
    plastic=torch.zeros(nf,dtype=torch.float64);alpha=torch.zeros(nf,dtype=torch.float64);scale=1.0
    old_strain=torch.zeros(nf,dtype=torch.float64);old_stress=residual.clone();stored0=float(torch.mean(residual**2/(2*e)))
    work=0.;dissipation=0.;states=[]
    for step in range(ns+1):
        shortening=end*step/ns
        if step:
            p0,a0=plastic.clone(),alpha.clone()
            def evaluate(q):
                strain=shortening+shape*(q*q-1.)
                values=_return_map(strain,p0,a0,residual,e,sy,h)
                force=float(torch.mean(values[0]))
                # Derivative of mean membrane energy plus modal bending energy.
                eq=kb*modal_norm*(q-1.)-float(torch.mean(values[0]*gradient**2))*q
                return eq,strain,values,force
            grid=torch.linspace(max(.02,scale*.25),max(8.,scale*4.),6000,dtype=torch.float64).tolist()
            last_q=float(grid[0]);last_f=evaluate(last_q)[0];bracket=None
            for q in grid[1:]:
                f=evaluate(float(q))[0]
                if f*last_f<=0 and (last_q<=scale<=q or bracket is None):
                    bracket=(last_q,float(q))
                    if last_q<=scale<=q: break
                last_q,last_f=float(q),f
            if bracket is None: raise RuntimeError("could not bracket field-driven strip equilibrium")
            lo,hi=bracket
            for _ in range(60):
                mid=.5*(lo+hi)
                if evaluate(lo)[0]*evaluate(mid)[0]<=0:hi=mid
                else:lo=mid
            scale=.5*(lo+hi);eq,strain,(stress,plastic,alpha,active),force=evaluate(scale)
            work+=float(torch.mean(.5*(old_stress+stress)*(strain-old_strain)))
            dissipation+=float(torch.mean(sy*(alpha-a0)));old_strain,old_stress=strain,stress
        else:
            stress=residual.clone();force=float(torch.mean(stress));active=torch.zeros(nf,dtype=torch.bool);eq=0.
        stored=float(torch.mean(.5*e*(stress/e)**2+.5*h*alpha**2))+.5*kb*modal_norm*(scale-1.)**2
        states.append(StripState(step,shortening,float(torch.max(torch.abs(profile))*scale),force,
            float(torch.max(torch.clamp(torch.abs(stress)-(sy+h*alpha),min=0.))),abs(eq),work,stored,dissipation,
            abs(work-(stored-stored0)-dissipation),float(torch.mean(active.to(torch.float64)))))
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
    reference = torch.tensor([oracle[-1].amplitude, oracle[-1].membrane_force,
                              oracle[-1].external_work],dtype=torch.float64)
    errors = []
    for path in paths:
        value = torch.tensor([path[-1].amplitude, path[-1].membrane_force,
                              path[-1].external_work],dtype=torch.float64)
        errors.append(float(torch.max(torch.abs(value / reference - 1.0))))
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
