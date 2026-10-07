"""Traceable preliminary ship-structure analysis kernels.

The routines in this module are engineering screening models.  They are not a
replacement for class-rule checks or a classification-society approval.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .frame2d import FrameModel, FrameResult, solve_frame_static


@dataclass(frozen=True)
class HullGirderResult:
    frame: FrameResult
    x: torch.Tensor
    distributed_load: float
    computed_midship_deflection: float
    reference_midship_deflection: float
    computed_midship_moment: float
    reference_midship_moment: float

    @property
    def maximum_relative_error(self) -> float:
        errors = (
            abs(self.computed_midship_deflection / self.reference_midship_deflection - 1.0),
            abs(self.computed_midship_moment / self.reference_midship_moment - 1.0),
        )
        return max(errors)


def solve_hull_girder_uniform_load(
    *, length: float, young: float, area: float, inertia: float,
    still_water_load: float, wave_load: float, elements: int = 20,
) -> HullGirderResult:
    """Solve simply-supported hull-girder bending under equivalent line loads.

    ``still_water_load`` and ``wave_load`` use positive downward magnitudes.
    Their sum is applied to an Euler--Bernoulli beam.  The returned analytical
    references are ``5 q L^4/(384 EI)`` and ``q L^2/8``.
    """
    values = (length, young, area, inertia)
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("length, material and section properties must be positive and finite")
    if elements < 2 or elements % 2:
        raise ValueError("elements must be a positive even integer of at least two")
    if any(not math.isfinite(v) for v in (still_water_load, wave_load)):
        raise ValueError("loads must be finite")
    q = still_water_load + wave_load
    if q <= 0:
        raise ValueError("combined downward load must be positive")

    x = torch.linspace(0.0, length, elements + 1, dtype=torch.float64)
    nodes = torch.stack((x, torch.zeros_like(x)), dim=1)
    connectivity = torch.stack((torch.arange(elements), torch.arange(1, elements + 1)), dim=1)
    forces = torch.zeros(3 * (elements + 1), dtype=torch.float64)
    # Pin at x=0 and vertical roller at x=L; axial roller DOF remains free.
    fixed = torch.tensor([0, 1, 3 * elements + 1], dtype=torch.long)
    model = FrameModel(
        nodes, connectivity, torch.tensor(young, dtype=torch.float64),
        torch.tensor(area, dtype=torch.float64), torch.tensor(inertia, dtype=torch.float64),
        forces, fixed, torch.tensor(-q, dtype=torch.float64),
    )
    result = solve_frame_static(model)
    centre = elements // 2
    computed_deflection = -float(result.displacement[3 * centre + 1])
    # At the centre node, either adjacent member end moment has magnitude qL^2/8.
    computed_moment = abs(float(result.element_end_forces_local[centre - 1, 5]))
    reference_deflection = 5.0 * q * length**4 / (384.0 * young * inertia)
    reference_moment = q * length**2 / 8.0
    return HullGirderResult(
        result, x, q, computed_deflection, reference_deflection,
        computed_moment, reference_moment,
    )


@dataclass(frozen=True)
class OrthotropicPanelResult:
    bending_rigidity_x: float
    bending_rigidity_y: float
    twisting_rigidity: float
    computed_center_deflection: float
    reference_center_deflection: float
    relative_error: float


def stiffened_panel_sine_benchmark(
    *, length: float, width: float, plate_thickness: float,
    plate_young: float, plate_poisson: float, stiffener_spacing: float,
    stiffener_area: float, stiffener_young: float, stiffener_eccentricity: float,
    pressure_amplitude: float, quadrature_order: int = 16,
) -> OrthotropicPanelResult:
    """Equivalent-orthotropic simply supported stiffened-panel benchmark.

    Longitudinal stiffeners are smeared into ``D_x`` through their spacing.
    A one-term Ritz solve is evaluated by Gauss quadrature and checked against
    the closed-form Navier solution for ``q sin(pi*x/a) sin(pi*y/b)``.
    """
    positives = (length, width, plate_thickness, plate_young, stiffener_spacing,
                 stiffener_area, stiffener_young, pressure_amplitude)
    if any(not math.isfinite(v) or v <= 0 for v in positives):
        raise ValueError("dimensions, stiffnesses and pressure must be positive and finite")
    if not (-1.0 < plate_poisson < 0.5):
        raise ValueError("plate_poisson must be between -1 and 0.5")
    if not math.isfinite(stiffener_eccentricity) or quadrature_order < 2:
        raise ValueError("invalid eccentricity or quadrature order")

    base = plate_young * plate_thickness**3 / (12.0 * (1.0 - plate_poisson**2))
    dx = base + stiffener_young * stiffener_area * stiffener_eccentricity**2 / stiffener_spacing
    dy = base
    dxy = base  # D12 + 2D66 for the isotropic plate contribution.
    ax, ay = math.pi / length, math.pi / width
    denominator = dx * ax**4 + 2.0 * dxy * ax**2 * ay**2 + dy * ay**4
    reference = pressure_amplitude / denominator

    # Independent Rayleigh--Ritz amplitude: integrate load and curvature energy.
    # Tensor-only composite midpoint quadrature avoids an optional dependency.
    points = (torch.arange(quadrature_order, dtype=torch.float64) + 0.5) / quadrature_order
    xs = points * length
    ys = points * width
    wx = torch.full_like(points, length / quadrature_order)
    wy = torch.full_like(points, width / quadrature_order)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    ww = wx[:, None] * wy[None, :]
    phi = torch.sin(ax * xx) * torch.sin(ay * yy)
    # q(x,y)=q0*phi and the generalized load is integral(q*phi).
    load_generalized = float(torch.sum(ww * pressure_amplitude * phi**2))
    curvature_energy_factor = float(torch.sum(ww * (
        dx * (ax**2 * phi)**2 + 2.0 * dxy * ax**2 * ay**2 * phi**2
        + dy * (ay**2 * phi)**2
    )))
    computed = load_generalized / curvature_energy_factor
    error = abs(computed / reference - 1.0)
    return OrthotropicPanelResult(dx, dy, dxy, computed, reference, error)
