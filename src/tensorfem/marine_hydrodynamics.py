"""Auditable marine load generators and a TensorLBM load-exchange boundary.

This module implements linear Airy-wave kinematics and the fixed, slender-member
Morison equation.  It is not a CFD solver and does not model diffraction, breaking
waves, free-surface nonlinearity, shielding, or fluid-structure interaction.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import torch


MARINE_CASE_SCHEMA = "tensorfem.marine-load-case.v1"
LOAD_HISTORY_SCHEMA = "tensor-solver.load-history.v1"
_CASE_KEYS = {"schema", "name", "units", "water", "wave", "current", "member", "evaluation"}
_UNITS = {"length": "m", "time": "s", "mass": "kg", "force": "N", "pressure": "Pa"}


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


@dataclass(frozen=True)
class AiryWave:
    height: float
    period: float
    depth: float
    gravity: float = 9.80665

    def __post_init__(self) -> None:
        for name in ("height", "period", "depth", "gravity"):
            _number(getattr(self, name), name, positive=True)
        if self.height >= 2.0 * self.depth:
            raise ValueError("linear-wave input requires height < 2*depth")

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.period

    @property
    def wave_number(self) -> float:
        """Positive root of omega^2 = g k tanh(k h)."""
        target = self.omega**2
        k = max(target / self.gravity, self.omega / math.sqrt(self.gravity * self.depth))
        for _ in range(50):
            kh = k * self.depth
            th = math.tanh(kh)
            residual = self.gravity * k * th - target
            derivative = self.gravity * (th + kh * (1.0 - th * th))
            update = residual / derivative
            k -= update
            if k <= 0.0:
                k = target / self.gravity
            if abs(update) <= 1e-14 * max(1.0, k):
                return k
        raise RuntimeError("Airy dispersion solve did not converge")


@dataclass(frozen=True)
class MorisonMember:
    diameter: float
    drag_coefficient: float
    inertia_coefficient: float
    water_density: float = 1025.0

    def __post_init__(self) -> None:
        for name in ("diameter", "water_density"):
            _number(getattr(self, name), name, positive=True)
        for name in ("drag_coefficient", "inertia_coefficient"):
            if _number(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


def airy_kinematics(wave: AiryWave, z: torch.Tensor | float, x: float, time: float) -> dict[str, torch.Tensor]:
    """Return eta, horizontal/vertical velocity and acceleration in SI units.

    ``z=0`` is still-water level and ``z=-depth`` is the seabed. Inputs outside
    that closed vertical interval fail rather than extrapolate linear theory.
    """
    x = _number(x, "x")
    time = _number(time, "time")
    zz = torch.as_tensor(z, dtype=torch.float64)
    if not bool(torch.isfinite(zz).all()) or bool(((zz < -wave.depth) | (zz > 0.0)).any()):
        raise ValueError("z must be finite and within [-depth, 0]")
    k, omega, amplitude = wave.wave_number, wave.omega, wave.height / 2.0
    phase = k * x - omega * time
    # Exponential ratios avoid overflow in the nominally deep-water limit.
    y = zz + wave.depth
    denominator = -math.expm1(-2.0 * k * wave.depth)
    descending = torch.exp(k * (y - wave.depth))
    reflected = torch.exp(-k * (y + wave.depth))
    ch = (descending + reflected) / denominator
    sh = (descending - reflected) / denominator
    eta = torch.full_like(zz, amplitude * math.cos(phase))
    u = amplitude * omega * ch * math.cos(phase)
    w = amplitude * omega * sh * math.sin(phase)
    ax = amplitude * omega**2 * ch * math.sin(phase)
    az = -amplitude * omega**2 * sh * math.cos(phase)
    return {"eta": eta, "u": u, "w": w, "ax": ax, "az": az}


def morison_inline_load(
    wave: AiryWave,
    member: MorisonMember,
    z: torch.Tensor | float,
    x: float,
    time: float,
    *,
    current_velocity: float = 0.0,
) -> torch.Tensor:
    """Inline force per submerged length [N/m] on a fixed slender cylinder."""
    current = _number(current_velocity, "current_velocity")
    kin = airy_kinematics(wave, z, x, time)
    velocity = kin["u"] + current
    drag = 0.5 * member.water_density * member.drag_coefficient * member.diameter * velocity * velocity.abs()
    area = math.pi * member.diameter**2 / 4.0
    inertia = member.water_density * member.inertia_coefficient * area * kin["ax"]
    return drag + inertia


def _gauss_legendre_integrate(function: Callable[[torch.Tensor], torch.Tensor], a: float, b: float, order: int) -> float:
    if not isinstance(order, int) or isinstance(order, bool) or order < 2 or order > 256:
        raise ValueError("quadrature_order must be an integer in [2, 256]")
    roots: list[float] = []
    weights: list[float] = []
    for i in range(1, (order + 1) // 2 + 1):
        root = math.cos(math.pi * (i - 0.25) / (order + 0.5))
        for _ in range(30):
            p0, p1 = 1.0, root
            for degree in range(2, order + 1):
                p0, p1 = p1, ((2 * degree - 1) * root * p1 - (degree - 1) * p0) / degree
            derivative = order * (root * p1 - p0) / (root * root - 1.0)
            delta = p1 / derivative
            root -= delta
            if abs(delta) < 2e-15:
                break
        weight = 2.0 / ((1.0 - root * root) * derivative * derivative)
        if abs(root) < 1e-14:
            roots.append(0.0)
            weights.append(weight)
        else:
            roots.extend((-root, root))
            weights.extend((weight, weight))
    pairs = sorted(zip(roots, weights))
    nodes = torch.tensor([p[0] for p in pairs], dtype=torch.float64)
    ww = torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    mapped = 0.5 * ((b - a) * nodes + a + b)
    values = function(mapped)
    if values.shape != mapped.shape or not bool(torch.isfinite(values).all()):
        raise ValueError("integrand must return one finite value per quadrature point")
    return float(0.5 * (b - a) * torch.dot(ww, values))


def morison_base_actions(
    wave: AiryWave,
    member: MorisonMember,
    x: float,
    time: float,
    *,
    current_velocity: float = 0.0,
    quadrature_order: int = 32,
) -> dict[str, float]:
    """Integrate fixed-member base shear [N] and overturning moment [N m]."""
    q = lambda z: morison_inline_load(wave, member, z, x, time, current_velocity=current_velocity)
    shear = _gauss_legendre_integrate(q, -wave.depth, 0.0, quadrature_order)
    moment = _gauss_legendre_integrate(lambda z: q(z) * (z + wave.depth), -wave.depth, 0.0, quadrature_order)
    return {"base_shear": shear, "overturning_moment": moment}


def morison_quarter_phase_oracle(
    wave: AiryWave, member: MorisonMember, *, current_velocity: float = 0.0
) -> dict[str, float]:
    """Closed-form actions at phase pi/2, useful as an independent regression oracle."""
    current = _number(current_velocity, "current_velocity")
    k, h, amplitude, omega = wave.wave_number, wave.depth, wave.height / 2.0, wave.omega
    drag = 0.5 * member.water_density * member.drag_coefficient * member.diameter * current * abs(current)
    inertia_scale = member.water_density * member.inertia_coefficient * math.pi * member.diameter**2 / 4.0 * amplitude * omega**2
    inertia_shear = inertia_scale / k
    inertia_moment = inertia_scale * (h / k - math.tanh(k * h / 2.0) / (k * k))
    return {
        "base_shear": drag * h + inertia_shear,
        "overturning_moment": drag * h * h / 2.0 + inertia_moment,
    }


def load_marine_case(path: str | Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != _CASE_KEYS or data.get("schema") != MARINE_CASE_SCHEMA:
        raise ValueError("invalid marine load case schema or fields")
    if data.get("units") != _UNITS:
        raise ValueError("marine load case requires explicit SI units")
    expected = {
        "water": {"density", "gravity"},
        "wave": {"height", "period", "depth"},
        "current": {"velocity"},
        "member": {"diameter", "drag_coefficient", "inertia_coefficient"},
        "evaluation": {"x", "time", "quadrature_order"},
    }
    for key, fields in expected.items():
        if not isinstance(data[key], dict) or set(data[key]) != fields:
            raise ValueError(f"invalid {key} fields")
    return data


def run_marine_case(path: str | Path) -> dict[str, object]:
    data = load_marine_case(path)
    water, w, current, m, evaluation = (data[k] for k in ("water", "wave", "current", "member", "evaluation"))
    assert isinstance(water, dict) and isinstance(w, dict) and isinstance(current, dict)
    assert isinstance(m, dict) and isinstance(evaluation, dict)
    wave = AiryWave(w["height"], w["period"], w["depth"], water["gravity"])
    member = MorisonMember(m["diameter"], m["drag_coefficient"], m["inertia_coefficient"], water["density"])
    actions = morison_base_actions(wave, member, evaluation["x"], evaluation["time"],
                                   current_velocity=current["velocity"], quadrature_order=evaluation["quadrature_order"])
    return {"schema": "tensorfem.marine-load-result.v1", "name": data["name"], "wave_number": wave.wave_number, **actions}


@dataclass(frozen=True)
class ForceHistory:
    times: torch.Tensor
    node_ids: torch.Tensor
    forces: torch.Tensor
    source: str


def load_force_history(path: str | Path) -> ForceHistory:
    """Load the narrow TensorLBM -> TensorFEM nodal-force exchange contract."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema", "source", "target", "units", "times", "node_ids", "forces"}
    if not isinstance(data, dict) or set(data) != required or data.get("schema") != LOAD_HISTORY_SCHEMA:
        raise ValueError("invalid load-history schema or fields")
    if data.get("source") != "TensorLBM" or data.get("target") != "TensorFEM" or data.get("units") != {"time": "s", "force": "N"}:
        raise ValueError("load history requires TensorLBM -> TensorFEM and SI units")
    times = torch.as_tensor(data["times"], dtype=torch.float64)
    raw_ids = data["node_ids"]
    if not isinstance(raw_ids, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in raw_ids):
        raise ValueError("node_ids must be unique non-negative integers")
    ids = torch.as_tensor(raw_ids, dtype=torch.long)
    forces = torch.as_tensor(data["forces"], dtype=torch.float64)
    if times.ndim != 1 or len(times) == 0 or not bool(torch.isfinite(times).all()) or bool((times[1:] <= times[:-1]).any()):
        raise ValueError("times must be a non-empty strictly increasing finite vector")
    if ids.ndim != 1 or len(ids) == 0 or len(set(ids.tolist())) != len(ids) or bool((ids < 0).any()):
        raise ValueError("node_ids must be unique non-negative integers")
    if forces.shape != (len(times), len(ids), 3) or not bool(torch.isfinite(forces).all()):
        raise ValueError("forces must have finite shape [time, node, 3]")
    return ForceHistory(times, ids, forces, data["source"])


def force_history_resultants(history: ForceHistory, node_coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return force and origin moment histories for an already mapped FEM node set."""
    coordinates = torch.as_tensor(node_coordinates, dtype=torch.float64)
    if coordinates.shape != (len(history.node_ids), 3) or not bool(torch.isfinite(coordinates).all()):
        raise ValueError("node_coordinates must have finite shape [node, 3] in node_ids order")
    resultant = history.forces.sum(dim=1)
    moments = torch.cross(coordinates.unsqueeze(0).expand_as(history.forces), history.forces, dim=2).sum(dim=1)
    return {"force": resultant, "moment_about_origin": moments}
