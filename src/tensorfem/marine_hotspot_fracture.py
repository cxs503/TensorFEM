"""Result-field path extraction and elementary Mode-I fracture gates.

This is structural post-processing for shell/solid results.  It does not
compute a crack-tip FE field, crack growth, remeshing, or certification.
All quantities use SI units.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .result_db import ResultDB
from .result_db_v2 import ResultDBv2

QUALIFICATION_TOLERANCE = 0.03


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _vector(value, name, *, dimension=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1 or (dimension is not None and len(value) != dimension):
        raise ValueError(f"{name} has invalid shape")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite floating tensor")
    return value


@dataclass(frozen=True)
class PathField:
    distance_m: torch.Tensor
    coordinates_m: torch.Tensor
    values: torch.Tensor
    location: str
    component: int


@dataclass(frozen=True)
class StressLinearization:
    membrane_pa: float
    bending_surface_pa: float
    linearized_inner_pa: float
    linearized_outer_pa: float


def linearize_through_thickness(coordinate_m: torch.Tensor, stress_pa: torch.Tensor):
    """Return membrane and bending stress from a through-thickness SCL.

    Coordinates must span the complete thickness and are measured along the
    stress-classification line; trapezoidal integration permits nonuniform
    sampling. ``bending_surface_pa`` is the signed outer-surface amplitude.
    """
    z = _vector(coordinate_m, "coordinate_m")
    s = _vector(stress_pa, "stress_pa", dimension=len(z)).to(z)
    if len(z) < 2:
        raise ValueError("at least two through-thickness samples are required")
    if bool(torch.any(z[1:] <= z[:-1])):
        raise ValueError("coordinate_m must be strictly increasing")
    thickness = z[-1]-z[0]
    middle = 0.5*(z[-1]+z[0])
    local = z-middle
    membrane = torch.trapezoid(s,z)/thickness
    bending = 6.0*torch.trapezoid(s*local,z)/(thickness*thickness)
    return StressLinearization(float(membrane), float(bending),
                               float(membrane-bending), float(membrane+bending))


def _connectivity_positions(node_ids, connectivity):
    lookup = {int(v): i for i, v in enumerate(node_ids.tolist())}
    try:
        return torch.tensor([[lookup[int(v)] for v in row] for row in connectivity.tolist()],
                            dtype=torch.long, device=connectivity.device)
    except KeyError as exc:
        raise ValueError("connectivity references an unknown node") from exc


def _source_field(db, field, location, step, frame):
    if isinstance(db, ResultDB):
        db.validate()
        if location == "node":
            coordinates = db.node_fields.get("coordinates")
            values = db.node_fields.get(field)
        elif location == "integration_point":
            coordinates = db.node_fields.get("coordinates")
            values = db.element_fields.get(field)
        else:
            raise ValueError("location must be 'node' or 'integration_point'")
    elif isinstance(db, ResultDBv2):
        db.validate()
        if not isinstance(step, str) or isinstance(frame, bool) or not isinstance(frame, int):
            raise TypeError("v2 extraction requires string step and integer frame")
        selected_step = next((s for s in db.steps if s.name == step), None)
        selected = None if selected_step is None else next((f for f in selected_step.frames if f.index == frame), None)
        if selected is None:
            raise KeyError((step, frame))
        coordinates = selected.fields.get("node", {}).get("coordinates")
        values = selected.fields.get(location, {}).get(field)
    else:
        raise TypeError("db must be ResultDB or ResultDBv2")
    if coordinates is None:
        raise KeyError("coordinates")
    if values is None:
        raise KeyError(field)
    if coordinates.ndim != 2 or coordinates.shape[0] != len(db.node_ids) or coordinates.shape[1] not in (2, 3):
        raise ValueError("coordinates must have shape (nodes, 2|3)")
    if not coordinates.is_floating_point() or not bool(torch.isfinite(coordinates).all()):
        raise ValueError("coordinates must be finite floating values")
    if location == "integration_point":
        # ResultDB v1 stores element fields; v2 may retain multiple IPs per element.
        if len(values) != len(db.element_ids):
            raise ValueError("integration-point field must be element-major")
        positions = _connectivity_positions(db.node_ids, db.connectivity)
        coordinates = coordinates[positions].mean(dim=1)
        if values.ndim >= 3:
            values = values.mean(dim=1)
    return coordinates, values


def sample_result_path(db, field: str, start_m: torch.Tensor, end_m: torch.Tensor,
                       *, samples: int, location="node", component=0,
                       step=None, frame=None) -> PathField:
    """Sample a nodal or element/IP result along a straight surface path.

    Source points are projected onto the path and linearly interpolated in
    path distance.  This is intended for a selected row of shell nodes or
    solid element centroids, not an arbitrary 3-D cloud.
    """
    if not isinstance(field, str) or not field:
        raise TypeError("field must be a non-empty string")
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError("samples must be an integer")
    if samples < 2:
        raise ValueError("samples must be >= 2")
    if isinstance(component, bool) or not isinstance(component, int):
        raise TypeError("component must be an integer")
    coordinates, values = _source_field(db, field, location, step, frame)
    start = _vector(start_m, "start_m", dimension=coordinates.shape[1]).to(coordinates)
    end = _vector(end_m, "end_m", dimension=coordinates.shape[1]).to(coordinates)
    direction = end - start
    length = torch.linalg.vector_norm(direction)
    if not bool(length > 0):
        raise ValueError("path endpoints must differ")
    if values.ndim == 1:
        if component != 0:
            raise IndexError("scalar field only has component 0")
        scalar = values
    else:
        if component < 0 or component >= values.shape[-1]:
            raise IndexError("component is out of range")
        scalar = values[..., component]
    scalar = scalar.reshape(-1)
    if len(scalar) != len(coordinates) or not scalar.is_floating_point() or not bool(torch.isfinite(scalar).all()):
        raise ValueError("field values must be finite floating scalars at source points")
    unit = direction / length
    along = (coordinates - start) @ unit
    off = torch.linalg.vector_norm((coordinates - start) - along[:, None] * unit, dim=1)
    tolerance = max(float(length) * 1e-8, 1e-12)
    keep = (along >= -tolerance) & (along <= length + tolerance) & (off <= tolerance)
    if int(keep.sum()) < 2:
        raise ValueError("fewer than two source points lie on the requested path")
    x, order = torch.sort(torch.clamp(along[keep], 0, length))
    y = scalar[keep][order]
    if bool(torch.any(x[1:] <= x[:-1])):
        raise ValueError("source projections on path must be unique")
    target = torch.linspace(0.0, float(length), samples, dtype=coordinates.dtype, device=coordinates.device)
    if float(x[0]) > tolerance or float(x[-1]) < float(length) - tolerance:
        raise ValueError("source field does not bracket the complete path")
    right = torch.searchsorted(x, target).clamp(1, len(x) - 1)
    left = right - 1
    weight = (target - x[left]) / (x[right] - x[left])
    sampled = y[left] + weight * (y[right] - y[left])
    points = start + target[:, None] * unit
    return PathField(target, points, sampled, location, component)


def hot_spot_extrapolate(path: PathField, thickness_m: float, *, method="linear"):
    """Extrapolate structural stress to a weld toe at path distance zero."""
    thickness = _real(thickness_m, "thickness_m", positive=True)
    factors = {"linear": (0.4, 1.0), "quadratic": (0.4, 0.9, 1.4)}
    if method not in factors:
        raise ValueError("method must be 'linear' or 'quadratic'")
    if not isinstance(path, PathField):
        raise TypeError("path must be PathField")
    x, y = path.distance_m, path.values
    sites = x.new_tensor(factors[method]) * thickness
    if float(sites[-1]) > float(x[-1]):
        raise ValueError("path does not bracket required hot-spot sites")
    right = torch.searchsorted(x, sites).clamp(1, len(x)-1)
    left = right-1
    sampled = y[left] + (sites-x[left])/(x[right]-x[left])*(y[right]-y[left])
    result = sampled.new_zeros(())
    for i in range(len(sites)):
        weight = sampled.new_ones(())
        for j in range(len(sites)):
            if i != j:
                weight = weight * (-sites[j])/(sites[i]-sites[j])
        result = result + weight*sampled[i]
    return float(result)


@dataclass(frozen=True)
class FractureGate:
    stress_intensity_pa_sqrt_m: float
    j_integral_j_m2: float
    critical: bool
    utilization: float


def mode_i_fracture_gate(stress_pa, crack_m, geometry_factor, young_pa,
                         toughness_pa_sqrt_m, *, poisson=0.0, plane_strain=False):
    """LEFM Mode-I K, equivalent J=K^2/E', and toughness gate."""
    stress = _real(stress_pa, "stress_pa", nonnegative=True)
    crack = _real(crack_m, "crack_m", positive=True)
    factor = _real(geometry_factor, "geometry_factor", positive=True)
    young = _real(young_pa, "young_pa", positive=True)
    toughness = _real(toughness_pa_sqrt_m, "toughness_pa_sqrt_m", positive=True)
    nu = _real(poisson, "poisson")
    if not -1.0 < nu < 0.5:
        raise ValueError("poisson must lie in (-1, 0.5)")
    if not isinstance(plane_strain, bool):
        raise TypeError("plane_strain must be bool")
    k = factor * stress * math.sqrt(math.pi * crack)
    effective_young = young / (1.0-nu*nu) if plane_strain else young
    j = k*k/effective_young
    utilization = k/toughness
    return FractureGate(k, j, utilization >= 1.0, utilization)


def finite_width_edge_factor(crack_m, width_m):
    """Tada polynomial for a single-edge crack in finite-width tension."""
    a = _real(crack_m, "crack_m", positive=True)
    w = _real(width_m, "width_m", positive=True)
    ratio = a/w
    if ratio >= 0.6:
        raise ValueError("qualification polynomial requires crack/width < 0.6")
    return 1.12 - 0.231*ratio + 10.55*ratio**2 - 21.72*ratio**3 + 30.39*ratio**4


def run_hotspot_fracture_qualification() -> dict[str, object]:
    """Run affine-field, SCL and closed-form LEFM qualification oracles."""
    dtype = torch.float64
    x = torch.linspace(0.0, 0.02, 5, dtype=dtype)
    xyz = torch.stack((x, torch.zeros_like(x)), dim=1)
    ids = torch.arange(11, 16)
    db = ResultDB(ids, torch.arange(4), torch.tensor([[11, 12], [12, 13], [13, 14], [14, 15]]),
                  {"coordinates": xyz, "S": 150e6 - 2e9*x}, {})
    path = sample_result_path(db, "S", xyz[0], xyz[-1], samples=21)
    hotspot = hot_spot_extrapolate(path, 0.01, method="linear")
    z = torch.linspace(-0.01, 0.01, 17, dtype=dtype)
    linearized = linearize_through_thickness(z, 80e6 + 30e6*z/0.01)
    factor = finite_width_edge_factor(0.01, 0.2)
    fracture = mode_i_fracture_gate(120e6, 0.01, factor, 210e9, 20e6,
                                    poisson=0.3, plane_strain=True)
    k_ref = factor*120e6*math.sqrt(math.pi*0.01)
    j_ref = k_ref*k_ref*(1.0-0.3**2)/210e9
    values = (
        ("hot_spot", hotspot, 150e6),
        ("scl_membrane", linearized.membrane_pa, 80e6),
        ("mode_i_k", fracture.stress_intensity_pa_sqrt_m, k_ref),
        ("mode_i_j", fracture.j_integral_j_m2, j_ref),
    )
    evidence = [{"id": name, "actual": actual, "oracle": oracle,
                 "relative_error": abs(actual/oracle-1.0), "tolerance": QUALIFICATION_TOLERANCE,
                 "passed": abs(actual/oracle-1.0) < QUALIFICATION_TOLERANCE}
                for name, actual, oracle in values]
    return {"scope": "TensorFEM ResultDB field post-processing and LEFM gates; no crack-tip FE",
            "evidence": evidence, "passed": all(row["passed"] for row in evidence)}
