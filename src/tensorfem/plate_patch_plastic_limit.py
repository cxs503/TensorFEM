"""Plate panel elastoplastic capacity under patch ("ice-floe") loading.

Anchor: ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET`` — the literature gap stating that the
plate capability of this library was purely **elastic** (Kirchhoff/Mindlin), so
ice-structure interaction under a large *patch* load could only be answered with
an elastic stress upper bound: no plastic reserve, no permanent set, no
allowable-patch-pressure criterion.

This module supplies the three pieces demanded by the anchor acceptance:

1. **Elastoplastic plate with isotropic hardening and a low-temperature flow
   stress** (:func:`low_temperature_flow_stress`) feeding the layered Shell4 J2
   plate element (:mod:`tensorfem.layered_shell4_plasticity`).
2. **Permanent deflection post-processing** — unload path from a peak patch
   pressure back to zero and read the residual centre deflection
   (:func:`run_patch_plastic_path` + :attr:`PatchPath.permanent_deflection`).
3. **Consistency regression against the elastic solution on the same mesh** —
   below the analytic first-yield pressure the numerical deflection must
   reproduce the Navier (double-series) thin-plate solution
   (:func:`elastic_patch_plate`); across the yield point the load-deflection
   path must stay continuous.

Reference solutions used for the verification (all closed form, none fitted to
the FE result):

* :func:`elastic_patch_plate` — Navier double series for a simply supported
  rectangular plate under a rectangular central patch pressure.
* :func:`analytic_first_yield_pressure` — first-yield pressure from the Mises
  surface stress ``sigma = 6 M / t^2`` of that elastic field.
* :func:`yield_line_patch_upper_bound` — rigid-plastic yield-line (upper-bound
  theorem) collapse pressure ``p_u = W_int / (w0 * Int_patch w_hat dA)`` with
  ``m_p = sigma_y t^2/4``; reduces to the classical ``24 m_p / a^2`` for a fully
  loaded square plate.
* :func:`strip_lower_bound` — statically admissible two-way strip field, giving a
  lower bound so the numerical collapse load can be bracketed.

Honesty statement
-----------------
* The plate element is a flat-facet small-strain Mindlin shell (Q4, 2x2 Gauss,
  layered through thickness); rotations are small, so results are valid for
  panel deflections up to the order of the thickness (``w/t`` of a few) —
  membrane stiffening beyond that is *not* captured, hence the collapse pressure
  quoted here is the pure bending mechanism value.
* The low-temperature flow stress is a *documented engineering fit* to published
  ferritic ship-steel tensile data, not a material certificate.  Below the
  ductile-brittle transition temperature the plastic mechanism is not admissible
  and the function says so instead of extrapolating.
* The yield-line solution is an upper bound; agreement is therefore expected
  from *below* and the FE result is never tuned to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import torch

from .layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    solve_layered_shell4_increment,
)

__all__ = [
    "STEEL_GRADES",
    "FlowStress",
    "low_temperature_flow_stress",
    "flow_stress_table",
    "elastic_patch_plate",
    "analytic_first_yield_pressure",
    "plastic_moment",
    "yield_line_patch_upper_bound",
    "strip_lower_bound",
    "patch_plate_mesh",
    "patch_pressure_forces",
    "PatchPath",
    "PatchPathPoint",
    "run_patch_plastic_path",
    "plate_capacity_summary",
]


# ---------------------------------------------------------------------------
# 1. Low-temperature flow stress
# ---------------------------------------------------------------------------

#: ``fy20`` = nominal minimum yield strength [MPa] at 20 C, ``design_c`` = grade
#: design temperature, ``slope`` = measured flow-stress sensitivity
#: |d sigma_y / dT| [MPa/K] of normalized ferritic hull steel in the ductile
#: regime, ``dbtt_c`` = ductile-brittle transition temperature [C].
#: Typical published tensile data; kept in one table so a mill certificate can
#: overwrite it.
STEEL_GRADES: dict[str, dict] = {
    "AH32": dict(fy20=315.0, design_c=0.0, slope=0.42, dbtt_c=-55.0),
    "AH36": dict(fy20=355.0, design_c=0.0, slope=0.46, dbtt_c=-55.0),
    "DH32": dict(fy20=315.0, design_c=-20.0, slope=0.44, dbtt_c=-65.0),
    "DH36": dict(fy20=355.0, design_c=-20.0, slope=0.48, dbtt_c=-65.0),
    "EH36": dict(fy20=355.0, design_c=-40.0, slope=0.50, dbtt_c=-80.0),
    "FH36": dict(fy20=355.0, design_c=-60.0, slope=0.52, dbtt_c=-100.0),
    "NV-D36": dict(fy20=355.0, design_c=-20.0, slope=0.48, dbtt_c=-65.0),
    "NV-E36": dict(fy20=355.0, design_c=-40.0, slope=0.50, dbtt_c=-80.0),
    "NV-F36": dict(fy20=355.0, design_c=-60.0, slope=0.52, dbtt_c=-100.0),
}

REFERENCE_TEMPERATURE_C = 20.0


@dataclass(frozen=True)
class FlowStress:
    """Flow stress at a service temperature together with its validity flag."""

    grade: str
    temperature_c: float
    yield_stress: float            # Pa
    yield_stress_reference: float  # Pa at 20 C (nominal grade value)
    sensitivity_pa_per_k: float    # d sigma_y / dT (negative: colder = stronger)
    ductile: bool                  # False below DBTT: no plastic mechanism
    note: str

    @property
    def amplification(self) -> float:
        return self.yield_stress / self.yield_stress_reference

    def as_dict(self) -> dict:
        return {
            "grade": self.grade,
            "temperature_c": self.temperature_c,
            "yield_stress_MPa": self.yield_stress / 1e6,
            "yield_stress_reference_MPa": self.yield_stress_reference / 1e6,
            "sensitivity_MPa_per_K": self.sensitivity_pa_per_k / 1e6,
            "amplification": self.amplification,
            "ductile": self.ductile,
            "note": self.note,
        }


def low_temperature_flow_stress(grade: str, temperature_c: float,
                                *, slope_mpa_per_k: float | None = None) -> FlowStress:
    """Flow stress of a hull steel at service temperature.

    ``sigma_y(T) = sigma_y(20 C) + slope * (20 C - T)`` above the
    ductile-brittle transition temperature.  Below the DBTT the function does
    *not* extrapolate: it returns ``ductile=False`` and keeps the nominal
    strength, because the controlling failure mode there is cleavage rather than
    plastic collapse.
    """
    if grade not in STEEL_GRADES:
        raise KeyError(f"unknown steel grade {grade!r}; known: {sorted(STEEL_GRADES)}")
    record = STEEL_GRADES[grade]
    slope = record["slope"] if slope_mpa_per_k is None else float(slope_mpa_per_k)
    if slope < 0:
        raise ValueError("slope must be non-negative (flow stress rises as T falls)")
    fy20 = record["fy20"] * 1e6
    if temperature_c > 400.0:
        raise ValueError("fit valid below 400 C only (no creep/thermal softening model)")
    if temperature_c < record["dbtt_c"]:
        return FlowStress(grade, float(temperature_c), fy20, fy20, -slope * 1e6, False,
                          f"T={temperature_c:.0f} C below DBTT={record['dbtt_c']:.0f} C: "
                          "cleavage governs, plastic collapse not admissible")
    delta = slope * 1e6 * (REFERENCE_TEMPERATURE_C - float(temperature_c))
    return FlowStress(grade, float(temperature_c), fy20 + delta, fy20, -slope * 1e6, True,
                      f"linear flow-stress fit, slope {slope:.2f} MPa/K, "
                      f"design temperature {record['design_c']:.0f} C")


def flow_stress_table(grade: str, temperatures: Sequence[float]) -> list[dict]:
    return [low_temperature_flow_stress(grade, T).as_dict() for T in temperatures]


# ---------------------------------------------------------------------------
# 2. Elastic Navier solution for a rectangular central patch pressure
# ---------------------------------------------------------------------------

def _patch_fourier_amplitudes(a: float, b: float, patch: tuple[float, float],
                              pressure: float, m: int, n: int) -> float:
    """Sine-series coefficient of a centred rectangular patch of pressure."""
    u, v = patch
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    gx = 0.5 * (math.cos(m * math.pi * x1 / a) - math.cos(m * math.pi * x2 / a))
    gy = 0.5 * (math.cos(n * math.pi * y1 / b) - math.cos(n * math.pi * y2 / b))
    return 16.0 * pressure / (math.pi ** 2 * m * n) * gx * gy


def elastic_patch_plate(a: float, b: float, t: float, young: float, poisson: float,
                        patch: tuple[float, float], pressure: float,
                        *, terms: int = 41,
                        points: Iterable[tuple[float, float]] | None = None) -> dict:
    """Navier double-series solution: simply supported plate under a patch load.

    Returns lists of deflection and bending moments ``M_x``/``M_y`` (moment per
    unit width) at the requested points (default: plate centre), plus the Mises
    surface stress ``sigma_eq = 6 sqrt(Mx^2 - Mx My + My^2)/t^2``.
    """
    if a <= 0 or b <= 0 or t <= 0 or young <= 0:
        raise ValueError("geometry and modulus must be positive")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("invalid Poisson ratio")
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    points = [(0.5 * a, 0.5 * b)] if points is None else list(points)
    D = young * t ** 3 / (12.0 * (1.0 - poisson ** 2))
    w_out, mx_out, my_out, sig_out = [], [], [], []
    for (x, y) in points:
        w = mx = my = 0.0
        for m in range(1, terms + 1, 2):
            alpha = m * math.pi / a
            sx = math.sin(alpha * x)
            for n in range(1, terms + 1, 2):
                beta = n * math.pi / b
                q_mn = _patch_fourier_amplitudes(a, b, patch, pressure, m, n)
                if q_mn == 0.0:
                    continue
                s = sx * math.sin(beta * y)
                lam = alpha ** 2 + beta ** 2
                w += q_mn / (D * lam ** 2) * s
                mx += q_mn * (alpha ** 2 + poisson * beta ** 2) / lam ** 2 * s
                my += q_mn * (beta ** 2 + poisson * alpha ** 2) / lam ** 2 * s
        sig_x = 6.0 * mx / t ** 2
        sig_y = 6.0 * my / t ** 2
        sig = math.sqrt(max(sig_x ** 2 - sig_x * sig_y + sig_y ** 2, 0.0))
        w_out.append(w)
        mx_out.append(mx)
        my_out.append(my)
        sig_out.append(sig)
    return {"deflection": w_out, "Mx": mx_out, "My": my_out,
            "sigma_eq": sig_out, "D": D, "terms": terms}


def analytic_first_yield_pressure(a: float, b: float, t: float, young: float,
                                  poisson: float, yield_stress: float,
                                  patch: tuple[float, float], *,
                                  grid: int = 21, terms: int = 41) -> dict:
    """Pressure at which the elastic Mises surface stress first reaches yield.

    The maximum of the Navier field is searched on a ``grid x grid`` lattice of
    the plate (the patch-loaded plate peaks at the centre, but the lattice keeps
    the statement honest for elongated plates).
    """
    if grid < 5:
        raise ValueError("grid must be >= 5")
    points = [(a * (i + 0.5) / grid, b * (j + 0.5) / grid)
              for i in range(grid) for j in range(grid)]
    unit = elastic_patch_plate(a, b, t, young, poisson, patch, 1.0,
                               terms=terms, points=points)
    idx = max(range(len(points)), key=lambda k: unit["sigma_eq"][k])
    sig_unit = unit["sigma_eq"][idx]
    if sig_unit <= 0.0:
        raise ValueError("degenerate elastic field: zero surface stress")
    p_yield = yield_stress / sig_unit
    centre = elastic_patch_plate(a, b, t, young, poisson, patch, p_yield,
                                 terms=terms, points=[(0.5 * a, 0.5 * b)])
    return {"pressure": p_yield,
            "location": points[idx],
            "sigma_eq_per_pa": sig_unit,
            "mx_per_pa": unit["Mx"][idx],
            "centre_deflection": centre["deflection"][0],
            "centre_Mx": centre["Mx"][0]}


# ---------------------------------------------------------------------------
# 3. Rigid-plastic bounds (upper: yield line; lower: two-way strips)
# ---------------------------------------------------------------------------

def plastic_moment(thickness: float, yield_stress: float) -> float:
    """Full-plastic moment per unit width of a plate strip, ``m_p = sigma_y t^2/4``."""
    if thickness <= 0 or yield_stress <= 0:
        raise ValueError("thickness and yield stress must be positive")
    return yield_stress * thickness ** 2 / 4.0


def _patch_mechanism_volume_ratio(a: float, b: float, patch: tuple[float, float]) -> float:
    """``Int_patch w_hat dA`` for the yield-line surface ``w_hat = 1-2max(|x|/a,|y|/b)``.

    Closed form (centred coordinates, four-fold symmetry, split at the diagonal
    ``x'/a = y'/b``); no quadrature error.  A fully loaded plate gives ``ab/3``.
    """
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    hx, hy = 0.5 * u, 0.5 * v
    x_c = a * hy / b                      # diagonal crosses y' = hy here
    x1 = min(hx, x_c)
    quad = hy * x1 - hy ** 2 * x1 / b - (b / (3 * a ** 2)) * x1 ** 3
    if hx > x_c:
        x2 = hx
        quad += hy * ((x2 - x_c) - (x2 ** 2 - x_c ** 2) / a)
    return 4.0 * quad


def _patch_mechanism_volume_ratio_quadrature(a: float, b: float,
                                            patch: tuple[float, float],
                                            *, n: int = 240) -> float:
    """Mid-point quadrature cross-check of :func:`_patch_mechanism_volume_ratio`."""
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(4)
    u, v = patch
    xa, xb = 0.5 * (a - u), 0.5 * (a + u)
    ya, yb = 0.5 * (b - v), 0.5 * (b + v)
    # split the patch at both mid-lines and the diagonals: refine by midpoint sampling
    total = 0.0
    for i in range(n):
        x = xa + (i + 0.5) * (xb - xa) / n
        for j in range(n):
            y = ya + (j + 0.5) * (yb - ya) / n
            w = 1.0 - 2.0 * max(abs(x - 0.5 * a) / a, abs(y - 0.5 * b) / b)
            total += max(w, 0.0)
    return total * (xb - xa) * (yb - ya) / n ** 2


def yield_line_patch_upper_bound(a: float, b: float, t: float, yield_stress: float,
                                 patch: tuple[float, float]) -> dict:
    """Yield-line (upper bound) collapse pressure for a patch-loaded plate.

    Mechanism: ``w = w0 (1 - 2 max(|x|/a, |y|/b))``, diagonals as yield lines;
    internal work ``4 m_p w0 (a/b + b/a)``.
    """
    m_p = plastic_moment(t, yield_stress)
    volume = _patch_mechanism_volume_ratio(a, b, patch)
    if volume <= 0.0:
        raise ValueError("patch lies outside the collapse mechanism")
    internal = 4.0 * m_p * (a / b + b / a)
    return {"pressure": internal / volume, "m_p": m_p,
            "mechanism_volume": volume, "internal_work_per_w0": internal}


def strip_lower_bound(a: float, b: float, t: float, yield_stress: float,
                      patch: tuple[float, float]) -> dict:
    """Statically admissible two-way strip field (lower bound).

    ``p_x = 8 m_p/(u(2a-u))``, ``p_y = 8 m_p/(v(2b-v))``, ``p_lb = p_x + p_y``.
    """
    m_p = plastic_moment(t, yield_stress)
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    p_x = 8.0 * m_p / (u * (2.0 * a - u))
    p_y = 8.0 * m_p / (v * (2.0 * b - v))
    return {"pressure": p_x + p_y, "p_x": p_x, "p_y": p_y, "m_p": m_p}


# ---------------------------------------------------------------------------
# 4. FE model, load path, first yield, permanent set
# ---------------------------------------------------------------------------

def patch_pressure_forces(nodes: torch.Tensor, a: float, b: float, nx: int, ny: int,
                          patch: tuple[float, float], pressure: float) -> torch.Tensor:
    """Lumped nodal loads (w-direction) of a centred rectangular patch pressure."""
    u, v = patch
    dx, dy = a / nx, b / ny
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    force = torch.zeros(6 * len(nodes), dtype=nodes.dtype, device=nodes.device)
    tol = 1e-9 * max(a, b)
    for n in range(len(nodes)):
        x = float(nodes[n, 0])
        y = float(nodes[n, 1])
        if not (x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol):
            continue
        wx = 1.0 if (abs(x - x1) > tol and abs(x - x2) > tol) else 0.5
        wy = 1.0 if (abs(y - y1) > tol and abs(y - y2) > tol) else 0.5
        force[6 * n + 2] = pressure * dx * wx * dy * wy
    return force


def patch_plate_mesh(a: float, b: float, t: float, nx: int, ny: int, *,
                     young: float, poisson: float, yield_stress: float,
                     hardening: float, layers: int = 5,
                     patch: tuple[float, float] | None = None
                     ) -> tuple[LayeredShell4Model, dict]:
    """Simply supported plate mesh with a unit patch-pressure reference load.

    The patch boundary is required to fall on element edges, so that the patch
    size is a mesh-independent control variable (``ValueError`` otherwise).
    """
    patch = (0.5 * a, 0.5 * b) if patch is None else patch
    u, v = patch
    if not (0 < u <= a and 0 < v <= b):
        raise ValueError("patch must lie inside the plate")
    for size, step in ((u, a / nx), (v, b / ny)):
        ratio = size / step
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("patch edge must coincide with an element edge")
    dtype = torch.float64
    x = torch.linspace(0.0, a, nx + 1, dtype=dtype)
    y = torch.linspace(0.0, b, ny + 1, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    nodes = torch.stack([xx.ravel(), yy.ravel(), torch.zeros_like(xx.ravel())], dim=1)
    elements = []
    for i in range(nx):
        for j in range(ny):
            elements.append([i * (ny + 1) + j, (i + 1) * (ny + 1) + j,
                             (i + 1) * (ny + 1) + (j + 1), i * (ny + 1) + (j + 1)])
    elements = torch.tensor(elements, dtype=torch.long)
    model = LayeredShell4Model(nodes=nodes, elements=elements, young=young,
                               poisson=poisson, thickness=t,
                               yield_stress=yield_stress, hardening=hardening,
                               layers=layers)
    fixed = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            node = i * (ny + 1) + j
            if i == 0 or i == nx or j == 0 or j == ny:
                fixed.append(6 * node + 2)              # w = 0 (simple support)
            if i == 0 and j == 0:
                fixed.extend((6 * node, 6 * node + 1))
            elif i == 0:
                fixed.append(6 * node)                  # u = 0 on x = 0
            elif j == 0:
                fixed.append(6 * node + 1)              # v = 0 on y = 0
            fixed.append(6 * node + 5)                  # drilling
    fixed_dofs = torch.tensor(sorted(set(fixed)), dtype=torch.long)
    info = {"nx": nx, "ny": ny, "a": a, "b": b, "t": t, "patch": patch,
            "fixed_dofs": fixed_dofs,
            "reference_load": patch_pressure_forces(nodes, a, b, nx, ny, patch, 1.0)}
    return model, info


def surface_extrapolated_mises(stress: torch.Tensor, thickness: float) -> float:
    """Peak surface von Mises stress of a layered plate section.

    ``stress`` is the layered Gauss-point stress history ``[element, 4, layer, 3]``
    of :func:`tensorfem.layered_shell4_plasticity.assemble_layered_shell4`.  The
    layer values live at ``z = -t/2 + (k+1/2) t/n``; yielding is a *surface*
    event, so the outermost two layers are linearly extrapolated to ``z = +-t/2``
    (otherwise the layered model reports first yield at ``z = 0.8 t/2`` for five
    layers and delays the transition by a factor ``1/(1-1/n)``).
    """
    layers = int(stress.shape[2])
    if layers < 2:
        raise ValueError("need at least two layers to extrapolate to the surface")
    dz = thickness / layers
    z = [(-0.5 * thickness + (k + 0.5) * dz) for k in range(layers)]
    peak = 0.0
    for e in range(stress.shape[0]):
        for q in range(stress.shape[1]):
            for side in (1, -1):
                i1 = layers - 1 if side > 0 else 0
                i2 = layers - 2 if side > 0 else 1
                factor = (side * 0.5 * thickness - z[i1]) / (z[i1] - z[i2])
                a = stress[e, q, i1]
                b = stress[e, q, i2]
                sxx = float(a[0] + factor * (a[0] - b[0]))
                syy = float(a[1] + factor * (a[1] - b[1]))
                sxy = float(a[2] + factor * (a[2] - b[2]))
                eq = math.sqrt(max(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
                peak = max(peak, eq)
    return peak


@dataclass
class PatchPathPoint:
    load_factor: float
    centre_deflection: float
    max_alpha: float
    max_surface_mises: float = 0.0


@dataclass
class PatchPath:
    model: LayeredShell4Model
    info: dict
    loading: list[PatchPathPoint] = field(default_factory=list)
    unloading: list[PatchPathPoint] = field(default_factory=list)
    collapse_factor: float | None = None
    state: LayeredShell4State | None = None
    displacement: torch.Tensor | None = None

    def _interpolate_crossing(self, attribute: str, threshold: float) -> float | None:
        previous: PatchPathPoint | None = None
        for step in self.loading:
            value = getattr(step, attribute)
            if value > threshold:
                if previous is None:
                    return step.load_factor
                v0 = getattr(previous, attribute)
                if v0 > threshold:
                    return previous.load_factor
                frac = (threshold - v0) / (value - v0)
                return previous.load_factor + frac * (step.load_factor - previous.load_factor)
            previous = step
        return None

    @property
    def first_yield_factor(self) -> float | None:
        """Load factor at first yield, from the *surface* von Mises stress when it
        is available (physically the right place), else from the plastic strain
        of the outermost layer."""
        crossing = self._interpolate_crossing("max_surface_mises", self.model.yield_stress)
        if crossing is not None:
            return crossing
        previous: PatchPathPoint | None = None
        for step in self.loading:
            if step.max_alpha > 1e-12:
                if previous is None or previous.max_alpha > 1e-12:
                    return step.load_factor
                frac = step.max_alpha / (step.max_alpha - previous.max_alpha)
                return previous.load_factor + frac * (step.load_factor - previous.load_factor)
            previous = step
        return None

    @property
    def permanent_deflection(self) -> float | None:
        """Residual centre deflection after the unloading leg (``None`` if the
        path was not unloaded)."""
        if not self.unloading:
            return None
        return self.unloading[-1].centre_deflection

    @property
    def peak_factor(self) -> float | None:
        return self.loading[-1].load_factor if self.loading else None

    def stiffness_ratio(self) -> float | None:
        """Secant stiffness of the last step relative to the first (elastic) step;
        -> 0 at plastic collapse."""
        if len(self.loading) < 2:
            return None
        first, last = self.loading[0], self.loading[-1]
        prev = self.loading[-2]
        if first.centre_deflection == 0.0:
            return None
        k0 = first.load_factor / first.centre_deflection
        dw = last.centre_deflection - prev.centre_deflection
        dp = last.load_factor - prev.load_factor
        if dw <= 0.0:
            return 0.0
        return (dp / dw) / k0


def _centre_node(info: dict) -> int:
    nx, ny = info["nx"], info["ny"]
    return (nx // 2) * (ny + 1) + ny // 2


def run_patch_plastic_path(a: float, b: float, t: float, *, young: float,
                           poisson: float, yield_stress: float, hardening: float,
                           patch: tuple[float, float], load_factors: Sequence[float],
                           nx: int, ny: int, layers: int = 5,
                           tolerance: float = 1e-8, max_iterations: int = 25,
                           unload_steps: int = 0,
                           unload_to: float = 0.0) -> PatchPath:
    """Integrate the elastoplastic patch-loading path (adaptive load control).

    On a Newton failure the increment is bisected down to 1e-3 of the original
    step; the last converged factor is then the numerical collapse load (the
    equilibrium path has no point beyond it under load control).  Optionally the
    path is then unloaded to ``unload_to`` times the reference pressure and the
    residual centre deflection is read off (:attr:`PatchPath.permanent_deflection`).
    """
    if nx % 2 or ny % 2:
        raise ValueError("even element counts keep the plate centre a node")
    model, info = patch_plate_mesh(a, b, t, nx, ny, young=young, poisson=poisson,
                                   yield_stress=yield_stress, hardening=hardening,
                                   layers=layers, patch=patch)
    path = PatchPath(model=model, info=info)
    state = LayeredShell4State.virgin(model)
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype)
    centre = _centre_node(info)

    def advance(target: float, current: float, history: list[PatchPathPoint]) -> float | None:
        nonlocal state, u
        stack = [(current, float(target))]
        while stack:
            start, end = stack.pop()
            try:
                step = solve_layered_shell4_increment(
                    model, info["reference_load"], end, info["fixed_dofs"], state, u,
                    tolerance=tolerance, max_iterations=max_iterations)
            except RuntimeError:
                if (end - start) < 1e-3 * max(abs(end), 1e-12):
                    return None
                mid = 0.5 * (start + end)
                stack.append((mid, end))
                stack.append((start, mid))
                continue
            state, u = step.state, step.displacement
            history.append(PatchPathPoint(
                end, float(u[6 * centre + 2]), float(torch.max(step.alpha)),
                surface_extrapolated_mises(step.stress, model.thickness)))
            current = end
        return current

    previous = 0.0
    for target in load_factors:
        target = float(target)
        if target <= previous:
            continue
        reached = advance(target, previous, path.loading)
        if reached is None:
            path.collapse_factor = previous
            break
        previous = reached

    if unload_steps > 0 and path.loading:
        peak = path.loading[-1].load_factor
        for k in range(1, unload_steps + 1):
            target = peak + (unload_to - peak) * k / unload_steps
            if abs(target - previous) <= 1e-12:
                continue
            path.unloading.append(PatchPathPoint(
                target, float(path.loading[-1].centre_deflection if not path.unloading
                              else 0.0), 0.0))
            try:
                step = solve_layered_shell4_increment(
                    model, info["reference_load"], target, info["fixed_dofs"], state, u,
                    tolerance=tolerance, max_iterations=max_iterations)
            except RuntimeError:
                path.unloading[-1].centre_deflection = float(u[6 * centre + 2])
                break
            state, u = step.state, step.displacement
            path.unloading[-1].centre_deflection = float(u[6 * centre + 2])
            previous = target

    path.state = state
    path.displacement = u
    return path


def plate_capacity_summary(a: float, b: float, t: float, *, young: float,
                           poisson: float, grade: str, temperature_c: float,
                           patch: tuple[float, float], hardening: float,
                           load_factors: Sequence[float], nx: int, ny: int,
                           layers: int = 5, unload_steps: int = 0) -> dict:
    """Patch-pressure capacity of a plate panel at a service temperature.

    Returns the analytic elastic/yield-line bounds and the finite element path
    (first yield, collapse, permanent set), with the flow stress taken from
    :func:`low_temperature_flow_stress`.  Below the DBTT the plastic result is
    withheld and only the elastic first-yield pressure is reported.
    """
    flow = low_temperature_flow_stress(grade, temperature_c)
    elastic = analytic_first_yield_pressure(a, b, t, young, poisson,
                                            flow.yield_stress, patch)
    upper = yield_line_patch_upper_bound(a, b, t, flow.yield_stress, patch)
    lower = strip_lower_bound(a, b, t, flow.yield_stress, patch)
    summary = {
        "geometry": {"a": a, "b": b, "t": t, "patch": patch,
                     "aspect": a / b, "t_over_b": t / b},
        "material": {"young": young, "poisson": poisson, "hardening": hardening},
        "flow_stress": flow.as_dict(),
        "elastic_first_yield": elastic,
        "yield_line_upper": upper,
        "strip_lower": lower,
        "fe": None,
        "fe_permanent_deflection": None,
        "evidence": "analytic bounds + FE path",
    }
    if not flow.ductile:
        summary["evidence"] = "elastic only: below DBTT, plastic collapse withheld"
        return summary
    path = run_patch_plastic_path(a, b, t, young=young, poisson=poisson,
                                  yield_stress=flow.yield_stress,
                                  hardening=hardening, patch=patch,
                                  load_factors=load_factors, nx=nx, ny=ny,
                                  layers=layers, unload_steps=unload_steps)
    # Same-mesh elastic bias: the layered Shell4 element is not an exact thin-plate
    # element, so the transition point is reported both raw and normalised by the
    # element's own elastic surface-stress response (the "consistency regression
    # against the elastic solution on the same mesh" required by the anchor).
    probe = run_patch_plastic_path(a, b, t, young=young, poisson=poisson,
                                   yield_stress=flow.yield_stress, hardening=hardening,
                                   patch=patch,
                                   load_factors=[0.25 * elastic["pressure"]],
                                   nx=nx, ny=ny, layers=layers)
    fe_sigma_per_pa = probe.loading[0].max_surface_mises / probe.loading[0].load_factor
    bias = fe_sigma_per_pa / elastic["sigma_eq_per_pa"] - 1.0
    fe = {
        "elastic_bias": bias,
        "fe_surface_mises_per_pa": fe_sigma_per_pa,
        "analytic_surface_mises_per_pa": elastic["sigma_eq_per_pa"],
        "first_yield_factor": path.first_yield_factor,
        "collapse_factor": path.collapse_factor or path.loading[-1].load_factor,
        "peak_centre_deflection": path.loading[-1].centre_deflection,
        "stiffness_ratio": path.stiffness_ratio(),
        "mesh": {"nx": nx, "ny": ny, "layers": layers},
        "path": [(s.load_factor, s.centre_deflection, s.max_alpha) for s in path.loading],
    }
    if path.first_yield_factor is not None:
        fe["first_yield_error"] = (fe["first_yield_factor"] - elastic["pressure"]) / elastic["pressure"]
        # analytic first-yield pressure expressed in the *same-mesh* elastic field
        corrected = flow.yield_stress / fe_sigma_per_pa
        fe["first_yield_error_bias_normalised"] = \
            (fe["first_yield_factor"] - corrected) / corrected
        fe["analytic_first_yield_same_mesh"] = corrected
    fe["collapse_error_vs_upper"] = (fe["collapse_factor"] - upper["pressure"]) / upper["pressure"]
    summary["fe"] = fe
    summary["fe_permanent_deflection"] = path.permanent_deflection
    return summary