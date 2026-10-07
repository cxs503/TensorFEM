"""Box-barge hydrostatics and intact initial-stability calculations.

This module deliberately covers upright hydrostatics and small heel angles only.
It is not a large-angle stability, flooding, seakeeping, or rules-compliance tool.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "tensorfem.marine-hydrostatics.v1"
MAX_SMALL_HEEL_DEG = 10.0


@dataclass(frozen=True)
class HydrostaticsResult:
    name: str
    mode: str
    length: float
    beam: float
    depth: float
    draft: float
    fluid_density: float
    gravity: float
    displacement_volume: float
    displacement_mass: float
    center_of_buoyancy: tuple[float, float, float]
    waterplane_inertia_roll: float
    kb: float
    bm: float
    km: float
    kg: float
    gm: float
    heel_angle_deg: float
    righting_arm: float
    restoring_moment: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], required: set[str], label: str) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        raise ValueError(f"invalid {label} keys: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be finite{' and positive' if positive else ''}")
    return result


def _mass_item(value: Any, label: str) -> tuple[float, float]:
    item = _object(value, label)
    _keys(item, {"name", "mass", "kg"}, {"name", "mass", "kg"}, label)
    if not isinstance(item["name"], str) or not item["name"].strip():
        raise ValueError(f"{label}.name must be a non-empty string")
    return _number(item["mass"], f"{label}.mass", positive=True), _number(item["kg"], f"{label}.kg")


def load_hydrostatics_case(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid hydrostatics JSON") from exc
    analyze_box_barge(value)
    return value


def analyze_box_barge(case: Mapping[str, Any]) -> HydrostaticsResult:
    """Evaluate a rectangular, wall-sided barge at upright and small heel."""
    case = _object(case, "case")
    _keys(case, {"schema", "name", "units", "geometry", "fluid", "condition"},
          {"schema", "name", "units", "geometry", "fluid", "condition"}, "case")
    if case["schema"] != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    if case["units"] != "SI":
        raise ValueError("only SI units are supported")
    if not isinstance(case["name"], str) or not case["name"].strip():
        raise ValueError("name must be a non-empty string")

    geometry = _object(case["geometry"], "geometry")
    _keys(geometry, {"length", "beam", "depth"}, {"length", "beam", "depth"}, "geometry")
    length = _number(geometry["length"], "geometry.length", positive=True)
    beam = _number(geometry["beam"], "geometry.beam", positive=True)
    depth = _number(geometry["depth"], "geometry.depth", positive=True)

    fluid = _object(case["fluid"], "fluid")
    _keys(fluid, {"density", "gravity"}, {"density", "gravity"}, "fluid")
    density = _number(fluid["density"], "fluid.density", positive=True)
    gravity = _number(fluid["gravity"], "fluid.gravity", positive=True)

    condition = _object(case["condition"], "condition")
    common = {"mode", "heel_angle_deg"}
    mode = condition.get("mode")
    if mode == "specified_draft":
        _keys(condition, common | {"draft", "kg"}, common | {"draft", "kg"}, "condition")
        draft = _number(condition["draft"], "condition.draft", positive=True)
        kg = _number(condition["kg"], "condition.kg")
    elif mode == "load_case":
        _keys(condition, common | {"lightship", "loads"}, common | {"lightship", "loads"}, "condition")
        light_mass, light_kg = _mass_item(condition["lightship"], "condition.lightship")
        loads = condition["loads"]
        if not isinstance(loads, list):
            raise ValueError("condition.loads must be an array")
        masses = [(light_mass, light_kg)]
        for index, load in enumerate(loads):
            masses.append(_mass_item(load, f"condition.loads[{index}]"))
        total_mass = sum(mass for mass, _ in masses)
        draft = total_mass / (density * length * beam)
        kg = sum(mass * height for mass, height in masses) / total_mass
    else:
        raise ValueError("condition.mode must be specified_draft or load_case")

    heel_deg = _number(condition["heel_angle_deg"], "condition.heel_angle_deg")
    if abs(heel_deg) > MAX_SMALL_HEEL_DEG:
        raise ValueError(f"heel angle exceeds small-angle limit of {MAX_SMALL_HEEL_DEG:g} degrees")
    if not 0.0 < draft < depth:
        raise ValueError("equilibrium draft must be positive and below molded depth")

    volume = length * beam * draft
    mass = density * volume
    kb = draft / 2.0
    inertia = length * beam**3 / 12.0
    bm = inertia / volume
    km = kb + bm
    gm = km - kg
    heel = math.radians(heel_deg)
    righting_arm = gm * math.sin(heel)
    restoring = mass * gravity * righting_arm
    return HydrostaticsResult(
        name=case["name"], mode=str(mode), length=length, beam=beam, depth=depth,
        draft=draft, fluid_density=density, gravity=gravity,
        displacement_volume=volume, displacement_mass=mass,
        center_of_buoyancy=(length / 2.0, 0.0, kb),
        waterplane_inertia_roll=inertia, kb=kb, bm=bm, km=km, kg=kg, gm=gm,
        heel_angle_deg=heel_deg, righting_arm=righting_arm, restoring_moment=restoring,
    )


def write_hydrostatics_report(result: HydrostaticsResult, path: str | Path) -> Path:
    """Write deterministic JSON output suitable for regression comparison."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return output
