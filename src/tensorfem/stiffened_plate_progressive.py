"""Welded stiffened plate progressive-collapse validation surface.

Campaign IN421-W2-N01.  A welded, longitudinally T-stiffened plate is
compressed axially; the module provides the finite-element validation chain:

1. **Elastic buckling** (criterion a): FEM eigenvalue of the layered Shell4
   with the updated geometric stiffness, against the Paik2007 Eq.2.1 Navier
   closed form for the plate bay.
2. **Post-buckling path** (criterion c): stateful Riks arc-length with
   geometric stiffness and bilinear J2 plasticity, seeded with a 0.1t first
   buckling-mode imperfection; the descending branch and energy absorption are
   compared with the Paik2007 surrogate load-shortening curve.
3. **Smith method** (criterion d): the section is discretised into plate
   fields and T-stiffener columns; the M-kappa curve is integrated with
   neutral-axis equilibrium and compared with the FEM full-model ultimate
   moment from an end-rotation-controlled run.
4. **CSR check** (criterion e): section modulus Z_act vs Z_req.
5. **Paik2007** (criterion b): FEM ultimate stress vs the six-group average.
6. **Mesh convergence** (criterion f): GCI of the ultimate load on three
   meshes.

The element is the layered Shell4 of
:mod:`tensorfem.layered_shell4_plasticity` with the geometric-stiffness
extension added for this campaign (see that module's docstring).  The
bilinear material uses the tangent modulus ``E_t`` through the J2 radial
return (``hardening`` parameter).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import torch

from .layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    assemble_layered_shell4,
    layered_shell4_buckling_eigenvalue,
)
from .plate_stiffener_ultimate import (
    _PAIK2007_GROUPS,
    classical_plate_buckling_stress,
    paik_thayamballi_plate_ultimate,
)


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeldedPlateSpec:
    """Welded stiffened plate: geometry, material, mesh.

    All lengths in metres, stresses in pascals.  The plate mid-surface is
    ``z = 0``; the stiffener web hangs below (``z in [-hw, 0]``) and the
    flange mid-surface sits at ``z = -hw``.  Stiffeners are centred at
    ``y = spacing, 2*spacing, ...`` (``n_stiffeners`` internal stiffeners,
    ``n_stiffeners + 1`` plate bays of width ``spacing``).
    """

    length: float = 3.0
    width: float = 1.0
    plate_t: float = 0.01
    web_h: float = 0.15
    web_t: float = 0.01
    flange_w: float = 0.10
    flange_t: float = 0.01
    spacing: float = 0.5
    young: float = 206e9
    poisson: float = 0.3
    sigma_y: float = 315e6
    E_t: float = 2e9
    nx: int = 20
    ny: int = 10
    layers: int = 10
    imperfection: float = 0.1  # 0.1 * plate_t

    def validate(self) -> "WeldedPlateSpec":
        for name in ("length", "width", "plate_t", "web_h", "web_t",
                     "flange_w", "flange_t", "spacing", "young", "sigma_y"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a finite positive number")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must lie in (-1, 0.5)")
        if self.E_t < 0 or self.E_t > self.young:
            raise ValueError("E_t must lie in [0, E]")
        if self.imperfection < 0:
            raise ValueError("imperfection must be non-negative")
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx, ny must be >= 2")
        if self.layers < 1:
            raise ValueError("layers must be >= 1")
        if abs(self.width - (self.n_stiffeners + 1) * self.spacing) > 1e-9:
            raise ValueError("width must equal (n_stiffeners + 1) * spacing")
        return self

    @property
    def n_stiffeners(self) -> int:
        return int(round(self.width / self.spacing)) - 1

    @property
    def plate_area(self) -> float:
        return self.length * self.width * self.plate_t

    @property
    def stiffener_area(self) -> float:
        return (self.web_h * self.web_t + self.flange_w * self.flange_t)

    @property
    def total_area(self) -> float:
        return self.plate_area + self.n_stiffeners * self.stiffener_area

    @property
    def beta(self) -> float:
        """Plate slenderness (Paik-Thayamballi convention)."""
        return (self.spacing / self.plate_t) * math.sqrt(self.sigma_y / self.young)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeldedPlateCase:
    """Assembled welded-plate case: models, BCs, loads, probe."""

    nodes: torch.Tensor
    plate_model: LayeredShell4Model
    web_model: LayeredShell4Model
    flange_model: LayeredShell4Model
    layers: int
    fixed_dofs: torch.Tensor
    compression_load: torch.Tensor
    loaded_edge_dofs: torch.Tensor
    probe_dof: int
    reference_area: float
    spec: WeldedPlateSpec
    stiffener_y: tuple[float, ...] = field(default=(), compare=False)

    @property
    def n_dofs(self) -> int:
        return 6 * len(self.nodes)

    @property
    def models(self) -> tuple[LayeredShell4Model, ...]:
        return (self.plate_model, self.web_model, self.flange_model)

    def free_dofs(self) -> torch.Tensor:
        mask = torch.ones(self.n_dofs, dtype=torch.bool, device=self.nodes.device)
        mask[self.fixed_dofs] = False
        return torch.arange(self.n_dofs, device=self.nodes.device)[mask]

    def virgin_state(self) -> LayeredShell4State:
        """Virgin per-model histories packed as one state (``points[i]`` = model ``i``)."""
        return LayeredShell4State(tuple(
            LayeredShell4State.virgin(model).points for model in self.models))


def build_welded_plate(spec: WeldedPlateSpec, *, dtype: torch.dtype = torch.float64,
                       imperfection_mode: torch.Tensor | None = None) -> WeldedPlateCase:
    """Build the full welded plate (n_stiffeners internal stiffeners).

    ``imperfection_mode`` is an optional ``(n_plate_nodes,)`` transverse
    displacement field (the first buckling mode, peak-normalised); when given,
    the plate nodes are offset by ``spec.imperfection * plate_t * mode``.
    """
    spec.validate()
    a, b = spec.length, spec.width
    tp, hw = spec.plate_t, spec.web_h
    bf = spec.flange_w / 2.0
    nx, ny = spec.nx, spec.ny
    dx, dy = a / nx, b / ny

    def plate_id(i: int, j: int) -> int:
        return j * (nx + 1) + i

    nodes: list[list[float]] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([i * dx, j * dy, 0.0])
    n_plate = len(nodes)

    n_stiff = spec.n_stiffeners
    stiffener_y = tuple(spec.spacing * (k + 1) for k in range(n_stiff))
    web_nodes: list[tuple[list[int], list[int]]] = []
    flange_nodes: list[tuple[list[int], list[int], list[int]]] = []
    for ys in stiffener_y:
        j = int(round(ys / dy))
        base = [plate_id(i, j) for i in range(nx + 1)]
        web = list(range(len(nodes), len(nodes) + nx + 1))
        nodes += [[i * dx, ys, -hw] for i in range(nx + 1)]
        fl_in = list(range(len(nodes), len(nodes) + nx + 1))
        nodes += [[i * dx, ys - bf, -hw] for i in range(nx + 1)]
        fl_out = list(range(len(nodes), len(nodes) + nx + 1))
        nodes += [[i * dx, ys + bf, -hw] for i in range(nx + 1)]
        web_nodes.append((base, web))
        flange_nodes.append((web, fl_in, fl_out))

    plate_elements = [[plate_id(i, j), plate_id(i + 1, j),
                      plate_id(i + 1, j + 1), plate_id(i, j + 1)]
                     for j in range(ny) for i in range(nx)]
    web_elements: list[list[int]] = []
    flange_elements: list[list[int]] = []
    for (base, web), (w, fi, fo) in zip(web_nodes, flange_nodes):
        web_elements += [[base[i], base[i + 1], web[i + 1], web[i]] for i in range(nx)]
        flange_elements += [[w[i], w[i + 1], fo[i + 1], fo[i]] for i in range(nx)]
        flange_elements += [[w[i], w[i + 1], fi[i + 1], fi[i]] for i in range(nx)]

    coordinates = torch.tensor(nodes, dtype=dtype)
    if imperfection_mode is not None:
        mode = torch.as_tensor(imperfection_mode, dtype=dtype)
        if mode.shape[0] != n_plate:
            raise ValueError("imperfection_mode must span the plate nodes")
        coordinates[:n_plate, 2] += spec.imperfection * spec.plate_t * mode

    def make_model(element_list: list[list[int]], thickness: float) -> LayeredShell4Model:
        model = LayeredShell4Model(coordinates, torch.tensor(element_list, dtype=torch.long),
                                   spec.young, spec.poisson, thickness,
                                   spec.sigma_y, spec.E_t, layers=spec.layers)
        model.validate()
        return model

    plate_model = make_model(plate_elements, tp)
    web_model = make_model(web_elements, spec.web_t)
    flange_model = make_model(flange_elements, spec.flange_t)

    fixed: set[int] = set()
    for i in range(nx + 1):
        fixed.add(6 * i + 2)                      # w = 0 on y = 0
        fixed.add(6 * (ny * (nx + 1) + i) + 2)    # w = 0 on y = b
    for j in range(ny + 1):
        fixed.add(6 * (j * (nx + 1)) + 0)         # u_x = 0 on x = 0
        fixed.add(6 * (j * (nx + 1)) + 2)         # w = 0 on x = 0
        fixed.add(6 * (j * (nx + 1) + nx) + 2)    # w = 0 on x = a
        fixed.add(6 * (j * (nx + 1)) + 1)         # u_y = 0 on y = 0
    for (base, web), (w, fi, fo) in zip(web_nodes, flange_nodes):
        for nd in (web[0], fi[0], fo[0]):
            fixed.add(6 * nd)
            fixed.add(6 * nd + 2)
        for nd in (web[nx], fi[nx], fo[nx]):
            fixed.add(6 * nd + 2)

    # Reference compressive load: tributary areas times one pascal at x = a.
    area: dict[int, float] = {}
    for j in range(ny + 1):
        width = dy / 2.0 if j in (0, ny) else dy
        area[plate_id(nx, j)] = width * tp
    for (base, web), (w, fi, fo), ys in zip(web_nodes, flange_nodes, stiffener_y):
        j = int(round(ys / dy))
        area[plate_id(nx, j)] += 0.5 * hw * spec.web_t
        area[web[nx]] = 0.5 * hw * spec.web_t + 0.25 * spec.flange_w * spec.flange_t
        area[fi[nx]] = 0.25 * spec.flange_w * spec.flange_t
        area[fo[nx]] = 0.25 * spec.flange_w * spec.flange_t
    reference_area = sum(area.values())
    compression = torch.zeros(6 * len(nodes), dtype=dtype)
    for node, tributary in area.items():
        compression[6 * node] = -tributary

    loaded = torch.tensor(sorted(set(
        [6 * plate_id(nx, j) for j in range(ny + 1)]
        + [6 * web[nx] for _, web in web_nodes]
        + [6 * fi[nx] for _, fi, _ in flange_nodes]
        + [6 * fo[nx] for _, _, fo in flange_nodes])), dtype=torch.long)
    probe = 6 * plate_id(nx // 2, ny // 2) + 2
    return WeldedPlateCase(coordinates, plate_model, web_model, flange_model,
                           spec.layers, torch.tensor(sorted(fixed), dtype=torch.long),
                           compression, loaded, probe, reference_area, spec,
                           stiffener_y=stiffener_y)


# ---------------------------------------------------------------------------
# 1. Elastic buckling (criterion a)
# ---------------------------------------------------------------------------

def welded_plate_buckling(spec: WeldedPlateSpec, *, nx: int | None = None,
                          ny: int | None = None, modes: int = 1,
                          dtype: torch.dtype = torch.float64):
    """Elastic buckling of the plate bay (Paik2007 Eq.2.1 verification).

    The plate bay (``length x spacing``) is buckled with the qualified MITC4
    operator of :mod:`tensorfem.shell4_buckling` and compared with the Navier
    closed form.  The stiffener is not included here: criterion (a) is a
    verification of the buckling operator against the plate theory, and the
    stiffener's rotational restraint would inflate the load beyond the
    classical value (see docs/verified/fem-panel-buckling-anchor.md).

    Returns a dict with the FEM and Navier critical stresses, the error, and
    the first transverse mode shape on the bay nodes.
    """
    spec.validate()
    from .shell4_buckling import simply_supported_plate_buckling
    nx = spec.nx if nx is None else int(nx)
    ny = max(2, int(round(spec.ny * spec.spacing / spec.width))) if ny is None else int(ny)
    result = simply_supported_plate_buckling(
        a=spec.length, b=spec.spacing, nx=nx, ny=ny,
        young=spec.young, poisson=spec.poisson, thickness=spec.plate_t,
        modes=modes)
    navier = classical_plate_buckling_stress(
        a=spec.length, b=spec.spacing, t=spec.plate_t,
        E=spec.young, nu=spec.poisson)
    sigma_cr_fem = float(result.load_factors[0]) / spec.plate_t
    sigma_cr_navier = float(navier["sigma_cr"])
    return {
        "sigma_cr_fem": sigma_cr_fem,
        "sigma_cr_navier": sigma_cr_navier,
        "error": sigma_cr_fem / sigma_cr_navier - 1.0,
        "buckling_coefficient": float(navier["buckling_coefficient"]),
        "critical_mode": int(navier["critical_mode"]),
        "loads": [float(x) for x in result.load_factors],
        "modes": result.modes.numpy(),
    }


def bay_imperfection_mode(spec: WeldedPlateSpec, *, nx: int = 18,
                          ny: int = 6) -> torch.Tensor:
    """First bay buckling mode interpolated onto the plate nodes.

    ``welded_plate_buckling`` returns the mode on its own bay mesh ``(nx, ny)``
    as ``(3 * n_nodes, modes)``; :func:`build_welded_plate` needs one value per
    plate node of the ``(spec.nx, spec.ny)`` grid.  The mode is bilinearly
    interpolated inside every bay and its sign alternates between adjacent bays
    (the classical antisymmetric bay pattern), so the field is continuous at the
    stiffener lines.  The result is peak-normalised to ``+-1``.
    """
    spec.validate()
    nx, ny = int(nx), int(ny)
    buckling = welded_plate_buckling(spec, nx=nx, ny=ny, modes=1)
    bay_mode = buckling["modes"][2::3, 0].reshape(ny + 1, nx + 1)
    bays = spec.n_stiffeners + 1
    values: list[list[float]] = []
    for j in range(spec.ny + 1):
        y = j * spec.width / spec.ny
        bay = min(int(y / spec.spacing), bays - 1)
        eta = (y - bay * spec.spacing) / spec.spacing * ny
        j0 = min(int(eta), ny - 1)
        fy = eta - j0
        row: list[float] = []
        for i in range(spec.nx + 1):
            xi = i / spec.nx * nx
            i0 = min(int(xi), nx - 1)
            fx = xi - i0
            value = ((1 - fx) * (1 - fy) * bay_mode[j0, i0]
                     + fx * (1 - fy) * bay_mode[j0, i0 + 1]
                     + (1 - fx) * fy * bay_mode[j0 + 1, i0]
                     + fx * fy * bay_mode[j0 + 1, i0 + 1])
            row.append(float(value) * (-1.0) ** bay)
        values.append(row)
    mode = torch.tensor(values, dtype=torch.float64)
    peak = float(mode.abs().max())
    if not math.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("first buckling mode is identically zero")
    return mode.reshape(-1) / peak


# ---------------------------------------------------------------------------
# 2. Stateful Riks post-buckling path (criterion c)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathPoint:
    step: int
    load_factor: float
    displacement: torch.Tensor
    iterations: int
    residual: float
    end_shortening: float
    transverse_deflection: float
    maximum_plastic_strain: float


@dataclass(frozen=True)
class ArcTrace:
    points: tuple[PathPoint, ...]
    converged: bool
    final_step: float
    yield_stress: float

    @property
    def ultimate_load_factor(self) -> float:
        return max((p.load_factor for p in self.points), default=0.0)

    @property
    def ultimate_index(self) -> int:
        return max(range(len(self.points)),
                   key=lambda i: self.points[i].load_factor, default=0)


def _maximum_plastic_strain(state: LayeredShell4State) -> float:
    from .plasticity import J2State
    best = 0.0
    for model_points in state.points:
        for element in model_points:
            for q in element:
                for layer in q:
                    best = max(best, float(torch.linalg.vector_norm(layer.plastic_strain)))
    return best


def trace_welded_plate_path(case: WeldedPlateCase, *, steps: int = 60,
                            step_size: float | None = None,
                            load_scale: float | None = None,
                            tolerance: float = 1e-6, max_iterations: int = 12,
                            minimum_step: float = 1e-6) -> ArcTrace:
    """Stateful Riks (Crisfield) path with geometric stiffness and J2 plasticity.

    The load factor equals the applied average compressive stress in pascals
    (the reference load is one pascal over the gross section).  The tangent at
    every Newton iteration is ``K_E + K_M - K_G`` with ``K_G`` built from the
    current in-plane resultants, so the path can pass a limit point.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    ds = 0.02 * case.spec.sigma_y if step_size is None else float(step_size)
    if ds <= 0:
        raise ValueError("step_size must be positive")
    weight = ds / (0.02 * case.spec.sigma_y) if load_scale is None else float(load_scale)
    if weight <= 0:
        raise ValueError("load_scale must be positive")

    dtype = case.nodes.dtype
    free = case.free_dofs()
    reference = case.compression_load
    scale = max(float(torch.linalg.vector_norm(reference[free])), 1.0)
    state = case.virgin_state()
    displacement = torch.zeros(case.n_dofs, dtype=dtype)
    load = 0.0
    previous: torch.Tensor | None = None
    points: list[PathPoint] = []
    converged = True
    step = 0
    def assemble_all(u: torch.Tensor, st: LayeredShell4State):
        internal = torch.zeros(case.n_dofs, dtype=dtype)
        stiffness = torch.zeros((case.n_dofs, case.n_dofs), dtype=dtype, device=case.nodes.device)
        trial_states = []
        for index, model in enumerate(case.models):
            f, k, _s, t = assemble_layered_shell4(
                model, u, LayeredShell4State(st.points[index]), geometric=True)
            internal += f
            stiffness += k
            trial_states.append(t.points)
        return internal, 0.5 * (stiffness + stiffness.T), LayeredShell4State(tuple(trial_states))

    while step < steps:
        step += 1
        _internal, stiffness, _trial = assemble_all(displacement, state)
        try:
            predictor = torch.linalg.solve(stiffness[free][:, free], reference[free])
        except torch.linalg.LinAlgError:
            predictor = torch.linalg.pinv(stiffness[free][:, free]) @ reference[free]
        direction = torch.cat((predictor, predictor.new_tensor([1.0])))
        direction = direction / torch.linalg.vector_norm(direction)
        if previous is not None and float(torch.dot(direction, previous)) < 0.0:
            direction = -direction
        previous = direction
        previous_metric = torch.cat((direction[:-1],
                                     direction.new_tensor([weight * float(direction[-1])])))
        previous_metric = previous_metric / torch.linalg.vector_norm(previous_metric)
        committed_u, committed_load = displacement.clone(), load
        trial_u = displacement.clone()
        trial_u[free] += ds * direction[:-1]
        trial_load = load + ds * float(direction[-1]) / weight
        residual_norm = float("inf")
        ok = False
        iteration = 0
        trial = state
        for iteration in range(1, max_iterations + 1):
            internal, stiffness, trial = assemble_all(trial_u, state)
            residual = internal - trial_load * reference
            residual_norm = float(torch.linalg.vector_norm(residual[free])) / scale
            du = trial_u - committed_u
            dl = trial_load - committed_load
            constraint = float(torch.dot(du[free], du[free])) + (weight * dl) ** 2 - ds ** 2
            if residual_norm <= tolerance and abs(constraint) <= tolerance * ds ** 2:
                ok = True
                break
            try:
                step_residual = torch.linalg.solve(stiffness[free][:, free], -residual[free])
                step_load = torch.linalg.solve(stiffness[free][:, free], reference[free])
            except torch.linalg.LinAlgError:
                break
            base = du[free] + step_residual
            quadratic = float(torch.dot(step_load, step_load)) + weight ** 2
            linear = 2.0 * (float(torch.dot(base, step_load)) + weight ** 2 * dl)
            constant = float(torch.dot(base, base)) + (weight * dl) ** 2 - ds ** 2
            discriminant = linear ** 2 - 4.0 * quadratic * constant
            if discriminant < 0.0 or quadratic <= 0.0:
                delta_load = -constant / linear if abs(linear) > 0.0 else 0.0
            else:
                root = math.sqrt(discriminant)
                candidates = [(-linear + root) / (2.0 * quadratic),
                              (-linear - root) / (2.0 * quadratic)]
                scores = []
                for candidate in candidates:
                    vector = torch.cat((
                        base + candidate * step_load,
                        base.new_tensor([weight * (dl + candidate)])))
                    scores.append(float(torch.dot(vector, previous_metric)))
                delta_load = candidates[0] if scores[0] >= scores[1] else candidates[1]
            trial_u = trial_u.clone()
            trial_u[free] += step_residual + delta_load * step_load
            trial_load += delta_load
        if not ok:
            if ds > minimum_step:
                ds *= 0.5
                weight *= 0.5
                step -= 1
                continue
            converged = False
            break
        state = trial
        displacement = trial_u
        load = trial_load
        points.append(PathPoint(
            step=step, load_factor=load, displacement=displacement.clone(),
            iterations=iteration, residual=residual_norm,
            end_shortening=-float(displacement[case.loaded_edge_dofs].mean()),
            transverse_deflection=float(displacement[case.probe_dof]),
            maximum_plastic_strain=_maximum_plastic_strain(state)))
    return ArcTrace(tuple(points), converged and len(points) == steps, ds,
                    case.spec.sigma_y)


def path_summary(case: WeldedPlateCase, trace: ArcTrace) -> dict[str, object]:
    """Compress an arc-length trace into the acceptance quantities."""
    sy = trace.yield_stress
    peak = trace.ultimate_load_factor
    peak_i = trace.ultimate_index
    descending = [p for p in trace.points[peak_i + 1:]]
    # energy absorption: area under the load-shortening curve from the peak to
    # the last point (positive when the branch descends)
    energy = 0.0
    for i in range(peak_i, len(trace.points) - 1):
        p0, p1 = trace.points[i], trace.points[i + 1]
        energy += 0.5 * (p0.load_factor + p1.load_factor) * (p1.end_shortening - p0.end_shortening)
    return {
        "points_completed": len(trace.points),
        "converged": trace.converged,
        "ultimate_stress": peak,
        "ultimate_over_yield": peak / sy,
        "peak_step": peak_i + 1,
        "n_descending_points": len(descending),
        "has_limit_point": peak_i < len(trace.points) - 1 and len(descending) > 0,
        "energy_absorbed": energy,
        "final_stress_over_yield": trace.points[-1].load_factor / sy if trace.points else 0.0,
        "max_plastic_strain": max((p.maximum_plastic_strain for p in trace.points), default=0.0),
        "points": [
            {
                "step": p.step,
                "average_stress": p.load_factor,
                "stress_over_yield": p.load_factor / sy,
                "end_shortening": p.end_shortening,
                "transverse_deflection": p.transverse_deflection,
                "iterations": p.iterations,
                "residual": p.residual,
                "max_plastic_strain": p.maximum_plastic_strain,
            }
            for p in trace.points
        ],
    }
