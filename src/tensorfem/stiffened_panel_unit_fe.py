"""Elasto-plastic Shell4 unit cell with an arc-length (Riks) collapse path.

The model is the standard ``shell`` idealisation of a longitudinally stiffened
panel: the plate field is meshed with the layered Shell4 of
:mod:`tensorfem.layered_shell4_plasticity`, the stiffener is a web strip plus a
flange strip rigidly connected to the plate at shared nodes, and two symmetry
planes (the stiffener centreline and the mid-bay line) make the cell
``span x spacing/2`` wide.  Plate, web and flange carry their own thicknesses
through three element groups that share one node array.

Why a dedicated arc-length driver
---------------------------------
:func:`tensorfem.arc_length.solve_arc_length` is a qualified Crisfield
continuation but assumes a *stateless* residual: the internal force is a
function of the current displacement only.  Elasto-plastic shell response is
path dependent, so the committed material state has to advance at every
converged point.  :func:`trace_panel_unit_path` implements the same spherical
arc-length corrector with an explicit state commit per step; its elastic limit
is verified against ``solve_arc_length`` in
``tests/test_stiffened_panel_unit_fe.py``.

Conventions
-----------
* Plate mid-surface at ``z = 0``; the web spans ``z in [-hw, 0]`` and the
  flange mid-surface sits at ``z = -hw`` (the same convention as
  :class:`tensorfem.stiffened_panel_ultimate.StiffenedPanel`).
* ``x = 0`` is the frame line (``u_x = 0``), ``x = span`` is the loaded end;
  ``w = 0`` on both transverse edges.
* ``y = 0`` is the stiffener centreplane, ``y = spacing/2`` the mid-bay
  symmetry plane; both carry ``u_y = theta_x = theta_z = 0``.
* The reference compressive load is normalised to a unit *average stress* over
  the half cell, so the arc-length load factor equals the applied average
  stress in pascals.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    element_response,
)
from .stiffened_panel_ultimate import StiffenedPanel


@dataclass(frozen=True)
class PanelUnitCase:
    """A stiffened-panel unit cell and its analysis contract."""

    nodes: torch.Tensor
    plate_model: LayeredShell4Model
    web_model: LayeredShell4Model
    flange_model: LayeredShell4Model
    layers: int
    fixed_dofs: torch.Tensor
    compression_load: torch.Tensor
    pressure_load: torch.Tensor
    loaded_edge_dofs: torch.Tensor
    probe_dof: int
    reference_area: float
    panel: StiffenedPanel
    pressure_ratio: float = 0.0

    @property
    def n_dofs(self) -> int:
        return 6 * len(self.nodes)

    @property
    def models(self) -> tuple[LayeredShell4Model, ...]:
        return (self.plate_model, self.web_model, self.flange_model)

    @property
    def element_counts(self) -> tuple[int, int, int]:
        return tuple(len(m.elements) for m in self.models)

    @property
    def reference_load(self) -> torch.Tensor:
        return self.compression_load + self.pressure_ratio * self.pressure_load

    def free_dofs(self) -> torch.Tensor:
        mask = torch.ones(self.n_dofs, dtype=torch.bool, device=self.nodes.device)
        mask[self.fixed_dofs] = False
        return torch.arange(self.n_dofs, device=self.nodes.device)[mask]

    def virgin_state(self) -> LayeredShell4State:
        return LayeredShell4State(tuple(
            tuple(tuple(tuple(_virgin_point(self.nodes) for _ in range(self.layers))
                        for _ in range(4)) for _ in model.elements)
            for model in self.models))


def _virgin_point(nodes: torch.Tensor):
    from .plasticity import J2State
    zero = nodes.new_zeros((3, 3))
    return J2State(zero.clone(), nodes.new_zeros(()))


def _element_dofs(connections: torch.Tensor) -> torch.Tensor:
    return torch.stack(tuple(6 * connections + k for k in range(6)), 1).reshape(-1).long()


def assemble_panel_unit(case: PanelUnitCase, displacement: torch.Tensor,
                        state: LayeredShell4State):
    """Assemble internal force, tangent and trial state over the three groups."""
    if displacement.shape != (case.n_dofs,):
        raise ValueError("wrong displacement vector length")
    internal = torch.zeros_like(displacement)
    tangent = torch.zeros((case.n_dofs, case.n_dofs), dtype=displacement.dtype,
                          device=displacement.device)
    stresses = []
    trial: list[tuple] = []
    for model, committed in zip(case.models, state.points):
        group_stress = []
        group_trial = []
        for index, connections in enumerate(model.elements):
            ids = _element_dofs(connections)
            force, stiffness, stress, points = element_response(
                model, index, displacement[ids], committed[index])
            internal.index_add_(0, ids, force)
            tangent.index_put_((ids[:, None], ids[None, :]), stiffness, accumulate=True)
            group_stress.append(stress)
            group_trial.append(points)
        stresses.append(torch.stack(group_stress))
        trial.append(tuple(group_trial))
    return internal, tangent, tuple(stresses), LayeredShell4State(tuple(trial))


def build_panel_unit(
    panel: StiffenedPanel, *, nx: int = 4, ny: int = 2, layers: int = 3,
    column_imperfection_ratio: float = 1.0 / 1000.0,
    plate_imperfection_ratio: float = 0.1,
    pressure_ratio: float = 0.0, dtype: torch.dtype = torch.float64,
) -> PanelUnitCase:
    """Build the half-bay stiffened-panel unit cell.

    ``column_imperfection_ratio`` scales an overall sine imperfection by
    ``span``; ``plate_imperfection_ratio`` scales a local plate mode by the
    plate thickness.  Both are needed for a limit-point path: a perfect panel
    bifurcates instead of collapsing.
    """
    panel.validate()
    if nx < 2 or ny < 1:
        raise ValueError("nx must be >= 2 and ny >= 1")
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if column_imperfection_ratio < 0.0 or plate_imperfection_ratio < 0.0:
        raise ValueError("imperfection ratios must be non-negative")

    a, half = panel.span, panel.spacing / 2.0
    tp, hw = panel.plate_thickness, panel.web_height
    bf_half = panel.flange_width / 2.0
    dx, dy = a / nx, half / ny

    def plate_id(i: int, j: int) -> int:
        return j * (nx + 1) + i

    nodes: list[list[float]] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([i * dx, j * dy, 0.0])
    web_bottom = [len(nodes) + i for i in range(nx + 1)]
    nodes += [[i * dx, 0.0, -hw] for i in range(nx + 1)]
    flange_outer = [len(nodes) + i for i in range(nx + 1)]
    nodes += [[i * dx, bf_half, -hw] for i in range(nx + 1)]

    plate_elements = [[plate_id(i, j), plate_id(i + 1, j),
                       plate_id(i + 1, j + 1), plate_id(i, j + 1)]
                      for j in range(ny) for i in range(nx)]
    web_elements = [[plate_id(i, 0), plate_id(i + 1, 0), web_bottom[i + 1], web_bottom[i]]
                    for i in range(nx)]
    flange_elements = [[web_bottom[i], web_bottom[i + 1], flange_outer[i + 1], flange_outer[i]]
                       for i in range(nx)]

    coordinates = torch.tensor(nodes, dtype=dtype)
    x, y = coordinates[:, 0], coordinates[:, 1]
    coordinates[:, 2] += panel.span * column_imperfection_ratio * torch.sin(math.pi * x / a)
    coordinates[:, 2] += panel.plate_thickness * plate_imperfection_ratio \
        * torch.sin(math.pi * x / a) * torch.sin(math.pi * y / half)

    def make_model(element_list: list[list[int]], thickness: float) -> LayeredShell4Model:
        model = LayeredShell4Model(coordinates, torch.tensor(element_list, dtype=torch.long),
                                   panel.young, panel.poisson, thickness,
                                   panel.yield_stress, hardening=0.0, layers=layers)
        model.validate()
        return model

    plate_model = make_model(plate_elements, panel.plate_thickness)
    web_model = make_model(web_elements, panel.web_thickness)
    flange_model = make_model(flange_elements, panel.flange_thickness)

    fixed: set[int] = set()

    def symmetry(node: int) -> None:
        fixed.update((6 * node + 1, 6 * node + 3, 6 * node + 5))

    # The stiffener centreplane (j = 0) is a true symmetry plane: no transverse
    # displacement and no in-plane rotation.  The mid-bay line (j = ny) carries
    # only the rotational symmetry: prescribing u_y = 0 there as well would
    # suppress the overall transverse contraction of the cell and lock the
    # plating into biaxial compression, which inflates the apparent axial
    # strength by more than 20 % (see docs/stiffened-panel-ultimate-strength.md).
    for i in range(nx + 1):
        symmetry(plate_id(i, 0))
    for i in range(nx + 1):
        node = plate_id(i, ny)
        fixed.update((6 * node + 3, 6 * node + 5))
    for node in web_bottom:
        symmetry(node)
    for j in range(ny + 1):
        fixed.add(6 * plate_id(0, j))          # frame-line axial symmetry
        fixed.add(6 * plate_id(0, j) + 2)      # frame support
        fixed.add(6 * plate_id(nx, j) + 2)     # loaded end support
    for node in (web_bottom[0], flange_outer[0]):
        fixed.add(6 * node)
        fixed.add(6 * node + 2)
    for node in (web_bottom[nx], flange_outer[nx]):
        fixed.add(6 * node + 2)

    # Reference compressive load: unit average stress over the half cell.
    area: dict[int, float] = {}
    for j in range(ny + 1):
        width = dy / 2.0 if j in (0, ny) else dy
        area[plate_id(nx, j)] = width * tp
    # The web is a mid-surface shell on the symmetry plane, so the half model
    # carries the full web thickness; its area is split between the plate node
    # and the web-bottom node.  The flange contributes half its width.
    area[plate_id(nx, 0)] += 0.5 * hw * panel.web_thickness
    area[web_bottom[nx]] = (0.5 * hw * panel.web_thickness
                            + 0.25 * panel.flange_width * panel.flange_thickness)
    area[flange_outer[nx]] = 0.25 * panel.flange_width * panel.flange_thickness
    reference_area = sum(area.values())
    compression = torch.zeros(6 * len(nodes), dtype=dtype)
    # The nodal forces are the tributary areas times *one pascal*, so the
    # arc-length load factor is the applied average stress in pascals.
    for node, tributary in area.items():
        compression[6 * node] = -tributary

    # Reference pressure: 1 Pa on the plate field through consistent nodal loads.
    pressure = torch.zeros(6 * len(nodes), dtype=dtype)
    for element in plate_elements:
        corners = coordinates[torch.tensor(element)]
        normal = torch.linalg.cross(corners[1] - corners[0], corners[3] - corners[0])
        normal = normal / torch.linalg.vector_norm(normal)
        cell = 0.5 * float(torch.linalg.vector_norm(
            torch.linalg.cross(corners[1] - corners[0], corners[2] - corners[0]))) \
            + 0.5 * float(torch.linalg.vector_norm(
                torch.linalg.cross(corners[2] - corners[0], corners[3] - corners[0])))
        for node in element:
            pressure[6 * node:6 * node + 3] += normal * cell / 4.0

    loaded = torch.tensor([6 * plate_id(nx, j) for j in range(ny + 1)]
                          + [6 * web_bottom[nx], 6 * flange_outer[nx]], dtype=torch.long)
    probe = 6 * plate_id(nx // 2, ny) + 2
    return PanelUnitCase(coordinates, plate_model, web_model, flange_model, layers,
                         torch.tensor(sorted(fixed), dtype=torch.long), compression,
                         pressure, loaded, probe, reference_area, panel, pressure_ratio)


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
    step_size: float
    state: LayeredShell4State
    yield_stress: float

    @property
    def ultimate_load_factor(self) -> float:
        return max(p.load_factor for p in self.points)

    @property
    def ultimate_index(self) -> int:
        return max(range(len(self.points)), key=lambda i: self.points[i].load_factor)

    @property
    def has_peak(self) -> bool:
        index = self.ultimate_index
        return 0 < index < len(self.points) - 1

    @property
    def post_peak_drop(self) -> float:
        if not self.has_peak:
            return 0.0
        peak = self.points[self.ultimate_index].load_factor
        return (peak - self.points[-1].load_factor) / peak


def _maximum_plastic_strain(state: LayeredShell4State) -> float:
    value = 0.0
    for group in state.points:
        for element in group:
            for point in element:
                for layer in point:
                    value = max(value, float(layer.alpha))
    return value


def trace_panel_unit_path(
    case: PanelUnitCase, *, steps: int = 20, step_size: float | None = None,
    load_scale: float | None = None, tolerance: float = 1e-3,
    max_iterations: int = 12, minimum_step: float = 1e-7,
) -> ArcTrace:
    """Trace the collapse path with a stateful spherical arc-length corrector.

    ``step_size`` is the combined displacement/load arc length, defaulting to
    2 % of the plate thickness; ``load_scale`` weights the load axis and
    defaults to one arc step = 2 % of the full-cell squash stress.
    """
    ds = 0.02 * case.plate_model.thickness if step_size is None else float(step_size)
    if ds <= 0.0:
        raise ValueError("step_size must be positive")
    weight = ds / (0.02 * case.plate_model.yield_stress) if load_scale is None else float(load_scale)
    if weight <= 0.0:
        raise ValueError("load_scale must be positive")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    dtype = case.nodes.dtype
    free = case.free_dofs()
    reference = case.reference_load
    scale = max(float(torch.linalg.vector_norm(reference[free])), 1.0)
    state = case.virgin_state()
    displacement = torch.zeros(case.n_dofs, dtype=dtype)
    load = 0.0
    previous: torch.Tensor | None = None
    points: list[PathPoint] = []
    converged = True
    step = 0
    while step < steps:
        step += 1
        _internal, tangent, _stress, _trial = assemble_panel_unit(case, displacement, state)
        stiffness = tangent[free][:, free]
        try:
            predictor = torch.linalg.solve(stiffness, reference[free])
        except torch.linalg.LinAlgError:
            predictor = torch.linalg.pinv(stiffness) @ reference[free]
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
            internal, tangent, _stress, trial = assemble_panel_unit(case, trial_u, state)
            residual = internal - trial_load * reference
            residual_norm = float(torch.linalg.vector_norm(residual[free])) / scale
            du = trial_u - committed_u
            dl = trial_load - committed_load
            constraint = float(torch.dot(du[free], du[free])) + (weight * dl) ** 2 - ds**2
            if residual_norm <= tolerance and abs(constraint) <= tolerance * ds**2:
                ok = True
                break
            # Crisfield split: two linear solves per iteration, then the exact
            # root of the spherical arc constraint.  Solving the bordered system
            # instead makes the constraint converge only linearly (measured: a
            # factor ~4 per iteration), which stalls on curved paths.
            stiffness = tangent[free][:, free]
            try:
                step_residual = torch.linalg.solve(stiffness, -residual[free])
                step_load = torch.linalg.solve(stiffness, reference[free])
            except torch.linalg.LinAlgError:
                break
            base = du[free] + step_residual
            quadratic = float(torch.dot(step_load, step_load)) + weight**2
            linear = 2.0 * (float(torch.dot(base, step_load)) + weight**2 * dl)
            constant = float(torch.dot(base, base)) + (weight * dl) ** 2 - ds**2
            discriminant = linear**2 - 4.0 * quadratic * constant
            if discriminant < 0.0 or quadratic <= 0.0:
                # No admissible root: fall back to the linearised increment.
                delta_load = -constant / linear if abs(linear) > 0.0 else 0.0
            else:
                root = math.sqrt(discriminant)
                candidates = [(-linear + root) / (2.0 * quadratic),
                              (-linear - root) / (2.0 * quadratic)]
                # Keep the branch that continues the previous increment: the
                # root with the largest |delta lambda| would flip the path onto
                # the mirror branch (it does at the very first step).
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
            # Halve the arc step and retry the same step index; the path through
            # a limit point needs smaller increments than the elastic branch.
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
    return ArcTrace(tuple(points), converged and len(points) == steps, ds, state,
                    case.plate_model.yield_stress)


def trace_summary(case: PanelUnitCase, trace: ArcTrace) -> dict[str, object]:
    """Compress an arc-length trace into the acceptance quantities."""
    sy = trace.yield_stress
    peak = trace.ultimate_load_factor
    index = trace.ultimate_index
    return {
        "points": [
            {"step": p.step, "average_stress": p.load_factor,
             "stress_over_yield": p.load_factor / sy,
             "end_shortening": p.end_shortening,
             "transverse_deflection": p.transverse_deflection,
             "iterations": p.iterations, "residual": p.residual,
             "maximum_plastic_strain": p.maximum_plastic_strain}
            for p in trace.points],
        "ultimate_stress": peak,
        "ultimate_over_yield": peak / sy,
        "ultimate_end_shortening": trace.points[index].end_shortening,
        "ultimate_transverse_deflection": trace.points[index].transverse_deflection,
        "ultimate_step": trace.points[index].step,
        "has_peak": trace.has_peak,
        "post_peak_drop": trace.post_peak_drop,
        "converged": trace.converged,
        "points_completed": len(trace.points),
        "reference_area": case.reference_area,
        "pressure_ratio": case.pressure_ratio,
        "closed_form_panel_ultimate": case.panel.panel_ultimate_stress,
        "closed_form_over_yield": case.panel.panel_ultimate_stress / sy,
    }