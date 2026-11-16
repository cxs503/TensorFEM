"""Displacement-controlled elastoplastic path of the patch-loaded plate panel.

Anchor: ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET`` (板格低温钢弹塑性 + 永久挠度).

Why this module exists
----------------------
Two published gate documents of the anchor close with the same open item:

* ``docs/CASE-FEM-PLATE-PATCH-PLASTIC-PATH-ADMISSIBILITY.md`` §5 —
  "Closing the item properly needs **displacement-controlled (arc-length)
  loading** ... that is beyond a load-controlled ladder";
* ``docs/CASE-GAP-PP-3-3GRID-MATRIX-VERDICT.md`` §6 — the measured limit is
  "systematically below the ideal rigid-plastic upper bound" and the reported
  limit load is a *convention* on a curve that never turns over.

Both statements describe the same defect of the load-controlled path
(:func:`tensorfem.plate_patch_plastic_limit.run_patch_plastic_path`): the
plastic plateau is reached by *prescribing* the pressure, so

1. the measured "limit" is the last *committed rung of the prescribed ladder*
   (:class:`PatchPath.measured_limit_load` is ``None`` whenever load control
   never lost equilibrium), and it therefore moves with the ramp density --
   the archived ladders moved by 39 % between two meshes precisely because
   their ramps had 5 and 21 rungs;
2. the ramp can only *stop* at the limit point, so the post-limit branch does
   not exist and no turning point is ever measured;
3. a collapsed iterate that has left the section is indistinguishable, in the
   path history, from a genuine equilibrium state (the fail-open defect the
   admissibility gate had to be added for).

This module removes the prescribed pressure from the driver: the *centre
deflection* is the control parameter and the patch pressure is an **unknown**
of the augmented Newton system

    R_f(u, λ) = λ F_ref - F_int(u) = 0        (equilibrium on the free dofs)
    g(u)      = w_centre - w̄_target  = 0      (control equation)

with unknowns ``(u_free, λ)``.  Its Jacobian is

    [ K_ff   -F_ref,free ]
    [ e_c^T      0       ]

(unsymmetric, one extra row/column).  This is displacement control, *not*
arc-length continuation: no path-following constraint and no branch switching,
so a snap-back would still be missed -- but a pressure plateau or a genuine
turning point, which is what a *limit load* means here, is measured directly
and is independent of how the control deflections are spaced.

What the path reports
---------------------
``loading[i]`` gives ``(w_centre, λ = patch pressure)`` at every controlled
state, so the quantity of interest is read off the **curve** instead of the
end of a ramp:

* :attr:`DisplacementPath.peak_factor` — maximum pressure on the path
  (a measured limit load when the curve turns over);
* :attr:`DisplacementPath.limit_is_turning_point` — whether the pressure
  actually decreased after the peak (a genuine limit point) or merely
  flattened (a plateau, which is a convention and is labelled as such);
* :attr:`DisplacementPath.plateau_factor` — pressure at the largest controlled
  deflection (the plateau value for a hardening material, where no turning
  point exists);
* :attr:`DisplacementPath.tangent_ratio` — final ``dλ/dw`` normalised by the
  elastic slope: ``-> 0`` means the section has saturated;
* :attr:`DisplacementPath.permanent_deflection` — residual centre deflection
  after the (elastic, load-controlled) unloading leg, ``None`` when the unloading
  leg did not converge.

Honesty statement
-----------------
* The element is still the layered small-strain Mindlin Q4 shell, so the same
  validity limit applies: panel deflections up to a few thicknesses
  (membrane stiffening beyond that is *not* the physics of this model unless
  ``geometric=True``).
* ``geometric=True`` (updated geometric stiffness, i.e. membrane tension from
  finite deflection) is *not* switched on by default: with it the response
  stiffens with deflection and the plateau is pushed up by a physical effect
  that the small-deflection reference solutions do not contain.
* The reported limit is never tuned to a reference: the references
  (``yield_line_patch_upper_bound``, ``strip_lower_bound``,
  ``ilyushin_sine_upper_bound``) are consumed only for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import torch

from .layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    assemble_layered_shell4,
    solve_layered_shell4_increment,
)
from .plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    max_section_moment_ratio,
    patch_plate_mesh,
    section_moment_capacity,
    surface_extrapolated_mises,
    _centre_node,
)

__all__ = [
    "DisplacementPathPoint",
    "DisplacementPath",
    "run_patch_plastic_displacement_path",
]


def _max_alpha(state: LayeredShell4State) -> float:
    """Peak equivalent plastic strain of a committed layered state."""
    peak = 0.0
    for element in state.points:
        for gauss_point in element:
            for layer in gauss_point:
                peak = max(peak, float(layer.alpha))
    return peak


@dataclass
class DisplacementPathPoint:
    """One controlled state of the displacement-driven path."""

    centre_deflection: float          # control parameter w_centre [m]
    load_factor: float                # patch pressure λ [Pa] (an *unknown*)
    max_alpha: float                  # peak equivalent plastic strain
    max_surface_mises: float          # peak surface von Mises stress [Pa]
    max_moment_ratio: float           # |M|/m_p peak (1 = fully plastic section)
    newton_iterations: int            # augmented Newton iterations used


@dataclass
class DisplacementPath:
    model: LayeredShell4Model
    info: dict
    loading: list[DisplacementPathPoint] = field(default_factory=list)
    unloading: list[DisplacementPathPoint] = field(default_factory=list)
    elastic_slope: float | None = None      # m per Pa (initial, secant)
    capacity_ratio: float = 1.0             # m_p,layers/m_p of the model
    steps_bisected: int = 0
    unload_converged_steps: int = 0
    unload_failure: str | None = None
    #: control deflection at the resolved zero-pressure state [m]
    permanent_deflection_value: float | None = None
    state: LayeredShell4State | None = None
    displacement: torch.Tensor | None = None
    notes: list[str] = field(default_factory=list)

    # -- measured quantities -------------------------------------------------
    def _peak_point(self) -> DisplacementPathPoint | None:
        if not self.loading:
            return None
        return max(self.loading, key=lambda q: q.load_factor)

    @property
    def peak_factor(self) -> float | None:
        """Maximum patch pressure reached on the path (Pa)."""
        peak = self._peak_point()
        return None if peak is None else peak.load_factor

    @property
    def peak_deflection(self) -> float | None:
        peak = self._peak_point()
        return None if peak is None else peak.centre_deflection

    @property
    def plateau_factor(self) -> float | None:
        """Patch pressure at the largest controlled deflection (Pa)."""
        return self.loading[-1].load_factor if self.loading else None

    @property
    def limit_is_turning_point(self) -> bool:
        """True when the pressure *decreased* after its maximum.

        A hardening material saturates to a plateau, so a decrease is the
        signature of a genuine limit point rather than a convention.  Two
        consecutive drops below 0.98 of the peak are required so that a
        Newton-noise wiggle is not reported as a mechanism.
        """
        peak = self._peak_point()
        if peak is None or len(self.loading) < 3:
            return False
        index = self.loading.index(peak)
        after = self.loading[index + 1:]
        if len(after) < 2:
            return False
        return all(q.load_factor < 0.98 * peak.load_factor for q in after[-2:])

    @property
    def tangent_ratio(self) -> float | None:
        """Final ``dλ/dw`` divided by the elastic slope ``1/k0``."""
        if self.elastic_slope is None or len(self.loading) < 2:
            return None
        if self.elastic_slope <= 0.0:
            return None
        last, prev = self.loading[-1], self.loading[-2]
        dw = last.centre_deflection - prev.centre_deflection
        if dw <= 0.0:
            return None
        return ((last.load_factor - prev.load_factor) / dw) * self.elastic_slope

    @property
    def first_yield_factor(self) -> float | None:
        """Patch pressure at which the surface von Mises stress reaches σ_y."""
        previous: DisplacementPathPoint | None = None
        for point in self.loading:
            if point.max_surface_mises >= self.model.yield_stress:
                if previous is None:
                    return point.load_factor
                v0 = previous.max_surface_mises
                if v0 >= self.model.yield_stress:
                    return previous.load_factor
                frac = ((self.model.yield_stress - v0)
                        / (point.max_surface_mises - v0))
                return previous.load_factor + frac * (point.load_factor
                                                      - previous.load_factor)
            previous = point
        return None

    @property
    def permanent_deflection(self) -> float | None:
        """Residual centre deflection (λ = 0 state), ``None`` when withheld."""
        if not self.permanent_set_valid:
            return None
        return self.permanent_deflection_value

    @property
    def permanent_set_valid(self) -> bool:
        return (self.unload_failure is None
                and self.permanent_deflection_value is not None)

    def as_dict(self) -> dict:
        return {
            "elastic_slope_m_per_Pa": self.elastic_slope,
            "capacity_ratio": self.capacity_ratio,
            "peak_factor": self.peak_factor,
            "peak_deflection": self.peak_deflection,
            "plateau_factor": self.plateau_factor,
            "limit_is_turning_point": self.limit_is_turning_point,
            "tangent_ratio": self.tangent_ratio,
            "first_yield_factor": self.first_yield_factor,
            "permanent_deflection": self.permanent_deflection,
            "permanent_set_valid": self.permanent_set_valid,
            "steps_bisected": self.steps_bisected,
            "unload_converged_steps": self.unload_converged_steps,
            "unload_failure": self.unload_failure,
            "path": [[q.centre_deflection, q.load_factor, q.max_alpha,
                      q.max_moment_ratio] for q in self.loading],
            "unload_path": [[q.centre_deflection, q.load_factor]
                            for q in self.unloading],
            "notes": list(self.notes),
        }


def _solve_displacement_increment(model: LayeredShell4Model,
                                 reference_load: torch.Tensor,
                                 fixed_dofs: torch.Tensor,
                                 committed: LayeredShell4State,
                                 displacement: torch.Tensor,
                                 centre_dof: int,
                                 target_deflection: float,
                                 load_guess: float, *,
                                 tolerance: float = 1e-10,
                                 max_iterations: int = 25,
                                 geometric: bool = False):
    """One displacement-controlled increment: solve for ``(u_free, λ)`` together.

    Returns ``(u, λ, stress, state, iterations)``; raises ``RuntimeError`` when
    the augmented Newton iteration does not converge, so the caller can bisect
    the *control* increment (the state is never committed on failure).
    """
    device = model.nodes.device
    fixed = fixed_dofs.to(device=device, dtype=torch.long)
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten().to(torch.long)
    positions = {int(dof): i for i, dof in enumerate(free.tolist())}
    if centre_dof not in positions:
        raise ValueError("the control dof (centre deflection) must be free, not fixed")
    control = positions[centre_dof]
    n_free = int(free.numel())

    u = displacement.clone()
    lam = float(load_guess)
    reference = reference_load.to(model.nodes)

    def scaled_norm(vector: torch.Tensor) -> float:
        scale = max(1.0, float(torch.linalg.vector_norm(reference[free])) * abs(lam))
        return float(torch.linalg.vector_norm(vector)) / scale

    for iteration in range(1, max_iterations + 1):
        internal, tangent, stress, trial = assemble_layered_shell4(
            model, u, committed, geometric=geometric)
        residual = lam * reference - internal
        residual_free = residual[free]
        gap = target_deflection - float(u[centre_dof])
        if scaled_norm(residual_free) <= tolerance and abs(gap) <= tolerance * max(
                abs(target_deflection), 1e-9):
            return u, lam, stress, trial, iteration

        size = n_free + 1
        matrix = torch.zeros((size, size), dtype=torch.float64, device=device)
        matrix[:n_free, :n_free] = tangent[free][:, free].to(torch.float64)
        matrix[:n_free, n_free] = -reference[free].to(torch.float64)
        matrix[n_free, control] = 1.0
        rhs = torch.zeros(size, dtype=torch.float64, device=device)
        rhs[:n_free] = residual_free.to(torch.float64)
        rhs[n_free] = gap
        try:
            solution = torch.linalg.solve(matrix, rhs)
        except torch.linalg.LinAlgError as error:      # singular augmented tangent
            raise RuntimeError("displacement-controlled tangent is singular") from error
        if not torch.isfinite(solution).all():
            raise RuntimeError("non-finite displacement-controlled increment")
        u = u.clone()
        u[free] += solution[:n_free].to(u.dtype)
        # the control equation is enforced exactly: no drift on the control dof
        u[centre_dof] = target_deflection
        lam += float(solution[n_free])
    raise RuntimeError(
        f"displacement-controlled Newton failed after {max_iterations} iterations")


def run_patch_plastic_displacement_path(
        a: float, b: float, t: float, *, young: float, poisson: float,
        yield_stress: float, hardening: float,
        patch: tuple[float, float] = (0.5, 0.5),
        nx: int = 8, ny: int = 4, layers: int = 5,
        deflection_ratios: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
                                             6.0, 8.0, 12.0, 16.0, 24.0),
        unload_steps: int = 14,
        exact_thickness: bool = True,
        shear_correction: float = 5.0 / 6.0,
        geometric: bool = False,
        inplane_restrained: bool = True,
        tolerance: float = 1e-10,
        max_iterations: int = 25,
        control_bisection_limit: int = 8,
        plateau_stop_ratio: float = 0.9,
        max_deflection_factor: float | None = None) -> DisplacementPath:
    """Integrate the patch-loading path with the centre deflection as control.

    Parameters
    ----------
    a, b, t : float
        Plate length, width and thickness [m].
    young, poisson, yield_stress, hardening : float
        Material parameters [Pa] (isotropic hardening).
    patch : tuple of float
        Centred pressure footprint ``(u, v)`` [m]; both edges must fall on
        element edges (``patch_plate_mesh`` enforces it).
    nx, ny, layers : int
        Mesh and through-thickness layer count.
    deflection_ratios : sequence of float
        Control deflections, as multiples of the *elastic* centre deflection at
        the analytic first-yield pressure.  They must be increasing.
    unload_steps : int
        Bisections used to resolve the zero-pressure state (the permanent set).
    plateau_stop_ratio : float
        Stop early once the pressure has stayed below this fraction of its peak
        for two consecutive controlled states (the post-limit branch has been
        measured) -- the run is not extended needlessly.
    max_deflection_factor : float, optional
        Reject a controlled state whose deflection exceeds this multiple of the
        elastic prediction at the same pressure (small-rotation guard).
    """
    if not deflection_ratios:
        raise ValueError("deflection_ratios must not be empty")
    ratios = [float(r) for r in deflection_ratios]
    if any(r <= 0.0 for r in ratios):
        raise ValueError("deflection_ratios must be positive")
    if any(nxt <= r for r, nxt in zip(ratios, ratios[1:])):
        raise ValueError("deflection_ratios must be strictly increasing")
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if nx % 2 or ny % 2:
        raise ValueError("even element counts keep the plate centre a node")

    model, info = patch_plate_mesh(
        a, b, t, nx, ny, young=young, poisson=poisson,
        yield_stress=yield_stress, hardening=hardening, layers=layers,
        patch=patch, exact_thickness=exact_thickness,
        shear_correction=shear_correction, inplane_restrained=inplane_restrained)
    centre_node = _centre_node(info)
    centre_dof = 6 * centre_node + 2
    reference = info["reference_load"]
    fixed = info["fixed_dofs"]

    capacity = section_moment_capacity(model.thickness, model.yield_stress,
                                       model.layers)["ratio"]

    # Elastic slope from a unit-pressure linear solve: needed only to scale the
    # control deflections, never used as a result.
    state = LayeredShell4State.virgin(model)
    probe = solve_layered_shell4_increment(
        model, reference, 1.0, fixed, state,
        torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device),
        tolerance=1e-12, max_iterations=2)
    elastic_slope = float(probe.displacement[centre_dof])   # m per Pa
    if not math.isfinite(elastic_slope) or elastic_slope <= 0.0:
        raise RuntimeError("elastic probe gave a non-positive centre deflection")
    first_yield = analytic_first_yield_pressure(a, b, t, young, poisson,
                                                yield_stress, patch)
    reference_deflection = elastic_slope * first_yield["pressure"]

    path = DisplacementPath(model=model, info=info,
                            elastic_slope=elastic_slope,
                            capacity_ratio=capacity)
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    lam = 0.0
    state = LayeredShell4State.virgin(model)

    def commit(point: DisplacementPathPoint) -> None:
        path.loading.append(point)

    targets = [reference_deflection * r for r in ratios]
    for target in targets:
        current = path.loading[-1].centre_deflection if path.loading else 0.0
        if target <= current:
            continue
        reached = _advance(model, reference, fixed, centre_node, centre_dof,
                           state, u, current, target, lam, path,
                           tolerance=tolerance, max_iterations=max_iterations,
                           geometric=geometric,
                           bisection_limit=control_bisection_limit,
                           capacity=capacity,
                           max_deflection_factor=max_deflection_factor,
                           commit=commit)
        if reached is None:
            path.notes.append(
                "control increment rejected at every bisection level between "
                f"w = {current:.6g} and {target:.6g} m; path stops there")
            break
        state, u, lam = reached
        if len(path.loading) >= 3 and path.peak_factor is not None:
            tail = path.loading[-2:]
            if all(q.load_factor < plateau_stop_ratio * path.peak_factor
                   for q in tail):
                path.notes.append(
                    "post-limit branch reached: pressure stayed below "
                    f"{plateau_stop_ratio:.2f} x peak for two controlled states; "
                    "control ramp stopped early")
                break

    path.state, path.displacement = state, u

    # --- unloading: find the state at zero pressure with the *same* control --
    # Load-controlled unloading off a flattened path is exactly the operation
    # that fails in the archived ladders (the last increment to p = 0 does not
    # converge, so the residual is withheld).  Under displacement control the
    # residual deflection is a root of λ(w) = 0, which is bracketed
    # [0, w_end] (λ < 0 below the residual, λ > 0 above it) and bisected.
    if path.loading and unload_steps > 0:
        _unload_to_zero_pressure(model, reference, fixed, centre_dof, path,
                                 u, lam, unload_steps=unload_steps,
                                 tolerance=max(tolerance, 1e-10),
                                 max_iterations=max_iterations,
                                 geometric=geometric)
    return path


def _unload_to_zero_pressure(model, reference, fixed, centre_dof, path, u, lam,
                             *, unload_steps, tolerance, max_iterations,
                             geometric):
    """Bisect the control deflection for the state of zero patch pressure.

    Elastic unloading makes λ(w) monotone decreasing from λ_end at the peak
    deflection to a negative value at w = 0, so the permanent deflection is a
    bracketed root and needs no extrapolation.  The bracket is only reported as
    converged when the state it lands on is admissible (finite, inside the
    section); otherwise ``unload_failure`` says why and the set is withheld.
    """
    w_end = path.loading[-1].centre_deflection
    lam_end = path.loading[-1].load_factor
    state = path.state
    lo, hi = 0.0, w_end
    w_res = None
    for _ in range(unload_steps):
        mid = 0.5 * (lo + hi)
        try:
            u_new, lam_new, stress, trial, iterations = _solve_displacement_increment(
                model, reference, fixed, state, u, centre_dof, mid, lam,
                tolerance=tolerance, max_iterations=max_iterations,
                geometric=geometric)
        except RuntimeError as error:
            path.unload_failure = f"unloading Newton failed at w = {mid:.6g} m: {error}"
            path.notes.append("permanent set WITHHELD: unloading did not converge")
            return
        ratio = max_section_moment_ratio(stress, model.thickness,
                                         model.yield_stress, model.layers)
        if not (math.isfinite(lam_new) and math.isfinite(ratio)):
            path.unload_failure = f"non-finite unloaded state at w = {mid:.6g} m"
            path.notes.append("permanent set WITHHELD: unloading did not converge")
            return
        state, u, lam = trial, u_new, lam_new
        path.unloading.append(DisplacementPathPoint(
            centre_deflection=mid, load_factor=lam_new,
            max_alpha=_max_alpha(trial),
            max_surface_mises=surface_extrapolated_mises(stress, model.thickness),
            max_moment_ratio=ratio, newton_iterations=iterations))
        path.unload_converged_steps += 1
        if abs(lam_new) <= 2e-3 * abs(lam_end):
            w_res = mid
            break
        if lam_new > 0.0:
            hi = mid
        else:
            lo = mid
    if w_res is None:
        path.unload_failure = (
            f"zero-pressure bracket not resolved in {unload_steps} bisections "
            f"(last λ = {lam / 1e6:.6g} MPa)")
        path.notes.append("permanent set WITHHELD: zero-pressure state not resolved")
        return
    path.permanent_deflection_value = w_res
    path.state, path.displacement = state, u


def _advance(model, reference, fixed, centre_node, centre_dof, state, u,
             current, target, lam, path, *, tolerance, max_iterations,
             geometric, bisection_limit, capacity, max_deflection_factor,
             commit):
    """Drive the control dof from ``current`` to ``target``, bisecting rejections.

    Returns ``(state, u, lam)`` on success and ``None`` when the increment
    cannot be resolved.  Committed states are appended through ``commit``.
    """

    def reject(w: float, ratio: float, lam: float) -> str | None:
        """Reason to refuse a converged controlled state, or ``None``."""
        if not (math.isfinite(w) and math.isfinite(ratio) and math.isfinite(lam)):
            return "non-finite state"
        if ratio > capacity * (1.0 + 1e-3):
            return (f"section capacity exceeded: |M|/m_p = {ratio:.6f} > "
                    f"capacity {capacity:.6f}")
        if max_deflection_factor is not None and path.elastic_slope:
            elastic = lam * path.elastic_slope
            if elastic > 0.0 and w > max_deflection_factor * elastic:
                return (f"small-rotation guard: w = {w:.6g} m exceeds "
                        f"{max_deflection_factor:.3g} x the elastic prediction "
                        f"{elastic:.6g} m at p = {lam:.6g} Pa")
        return None

    stack = [(current, float(target))]
    while stack:
        start, end = stack.pop()
        try:
            u_new, lam_new, stress, trial, iterations = _solve_displacement_increment(
                model, reference, fixed, state, u, centre_dof, end, lam,
                tolerance=tolerance, max_iterations=max_iterations,
                geometric=geometric)
        except RuntimeError:
            if abs(end - start) < 1e-6 * max(abs(end), 1e-12) or bisection_limit <= 0:
                return None
            middle = 0.5 * (start + end)
            stack.append((middle, end))
            stack.append((start, middle))
            path.steps_bisected += 1
            bisection_limit -= 1
            continue

        w_new = float(u_new[centre_dof])
        ratio = max_section_moment_ratio(stress, model.thickness,
                                         model.yield_stress, model.layers)
        rejection = reject(w_new, ratio, lam_new)
        if rejection is not None:
            if abs(end - start) < 1e-6 * max(abs(end), 1e-12) or bisection_limit <= 0:
                return None
            middle = 0.5 * (start + end)
            stack.append((middle, end))
            stack.append((start, middle))
            path.steps_bisected += 1
            bisection_limit -= 1
            continue
        if not math.isfinite(lam_new) or lam_new <= 0.0:
            return None

        state, u, lam = trial, u_new, lam_new
        commit(DisplacementPathPoint(
            centre_deflection=w_new,
            load_factor=lam,
            max_alpha=_max_alpha(trial),
            max_surface_mises=surface_extrapolated_mises(stress, model.thickness),
            max_moment_ratio=ratio,
            newton_iterations=iterations))
    return state, u, lam